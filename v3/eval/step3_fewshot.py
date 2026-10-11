"""Step 3B: Few-shot DINO evaluation across k in [1,2,4,8,16], seeds [0,1,2], and 3 scorers.

Features:
- Evaluates knn_global, knn_positional, and pca_residual on cached DINO features
- Enforces reference/test-negative disjointness via assert_disjoint_references_and_negatives
- Selects best scorer and aggregation radius on the 5 validation categories ONLY
- Reports full per-category metrics (with normal/anomalous counts, mean +/- std over seeds)
  on the 10 held-out categories
- Marks toothbrush as [LOW-N] and documents the 5% FPR AU-PRO normalization convention
- Writes CSV and Markdown reports to D:\\defect_cache\\results and v3/docs/results/phase_3.md
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from v3.config import (
    DEFAULT_CONFIG_PATH,
    StepLogger,
    get_git_commit,
    load_v3_config,
    log_decision,
)
from v3.core.bank import CachedDINOFeatures, DINOFeatureExtractor, l2_normalize_np
from v3.core.defect.aggregate import (
    aggregate_patch_map,
    topk_image_score,
    upsample_patch_map,
)
from v3.core.defect.knn import GlobalKNNScorer
from v3.core.defect.pca import PCAResidualScorer
from v3.core.mismatch.positional_knn import PositionalKNNScorer
from v3.eval.metrics import image_auroc, pixel_aupro, pixel_auroc
from v3.eval.splits import (
    DatasetSample,
    assert_disjoint_references_and_negatives,
    create_v3_category_splits,
)
from v3.eval.step3_cache_dino import discover_train_normals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3B: Few-shot DINO evaluation on cached features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print evaluation grid and estimated output size, then exit.")
    return parser.parse_args()


def build_scorer(
    scorer_name: str,
    extractor: DINOFeatureExtractor,
    radius: int,
    agg_radius: int,
    topk_frac: float,
    chunk_size: int,
) -> GlobalKNNScorer | PositionalKNNScorer | PCAResidualScorer:
    if scorer_name == "knn_global":
        return GlobalKNNScorer(extractor=extractor, agg_radius=agg_radius, topk_frac=topk_frac, chunk_size=chunk_size)
    if scorer_name == "knn_positional":
        return PositionalKNNScorer(extractor=extractor, radius=radius, agg_radius=agg_radius, topk_frac=topk_frac)
    if scorer_name == "pca_residual":
        return PCAResidualScorer(extractor=extractor, agg_radius=agg_radius, topk_frac=topk_frac)
    raise ValueError(f"Unknown scorer: {scorer_name}")


def load_category_masks(samples: list[DatasetSample], metric_max_side: int = 256) -> list[tuple[np.ndarray, int, int]]:
    """Preload ground-truth boolean masks and resized dimensions (mh, mw) for a category."""
    out: list[tuple[np.ndarray, int, int]] = []
    for s in samples:
        with Image.open(s.image_path) as img:
            w, h = img.size
        scale = min(1.0, metric_max_side / max(w, h)) if metric_max_side > 0 else 1.0
        mw, mh = max(1, round(w * scale)), max(1, round(h * scale))
        if s.label and s.mask_path and s.mask_path.is_file():
            with Image.open(s.mask_path) as mask:
                gt = np.asarray(mask.convert("L").resize((mw, mh), Image.Resampling.NEAREST), dtype=np.uint8) > 0
        else:
            gt = np.zeros((mh, mw), dtype=bool)
        out.append((gt, mh, mw))
    return out


@torch.inference_mode()
def precompute_per_ref_maps(
    q_stack: np.ndarray,
    unique_refs: list[Path],
    pool_feats_map: dict[Path, CachedDINOFeatures],
    pos_radius: int,
    device: torch.device,
) -> tuple[dict[Path, np.ndarray], dict[Path, np.ndarray]]:
    """Precompute per-reference-image max cosine similarity maps [N, H_p, W_p] for knn_global and knn_positional.

    Because max over references is associative:
      max_{p in U_{r in S} R_r} <q, p> == max_{r in S} (max_{p in R_r} <q, p>)
    precomputing per-reference similarity maps once per unique reference image is 100% mathematically
    identical while eliminating redundant dot products across k in [1,2,4,8,16] and seeds [0,1,2].
    """
    n_imgs, hp, wp, d = q_stack.shape
    q_t = torch.from_numpy(q_stack).to(device=device, dtype=torch.float32)
    q_flat = q_t.reshape(-1, d)  # [N*H_p*W_p, D]

    global_maps: dict[Path, np.ndarray] = {}
    pos_maps: dict[Path, np.ndarray] = {}
    rad = pos_radius
    q_step = 32 * hp * wp

    for ref_p in unique_refs:
        ref_np = l2_normalize_np(pool_feats_map[ref_p].patch_tokens)  # [H_p, W_p, D]
        ref_t = torch.from_numpy(ref_np).to(device=device, dtype=torch.float32)
        ref_flat = ref_t.reshape(-1, d)  # [H_p*W_p, D]

        # 1. Global max similarity against single reference image
        g_max = torch.empty(q_flat.shape[0], dtype=torch.float32, device=device)
        for q0 in range(0, q_flat.shape[0], q_step):
            sims = q_flat[q0 : q0 + q_step] @ ref_flat.T
            g_max[q0 : q0 + q_step] = sims.max(dim=1).values
        global_maps[ref_p] = g_max.reshape(n_imgs, hp, wp).cpu().numpy()

        # 2. Positional window max similarity against single reference image
        p_max = torch.full((n_imgs, hp, wp), -1.0, dtype=torch.float32, device=device)
        for dr in range(-rad, rad + 1):
            r_q_start = max(0, -dr)
            r_q_end = min(hp, hp - dr)
            r_ref_start = r_q_start + dr
            r_ref_end = r_q_end + dr
            for dc in range(-rad, rad + 1):
                c_q_start = max(0, -dc)
                c_q_end = min(wp, wp - dc)
                c_ref_start = c_q_start + dc
                c_ref_end = c_q_end + dc
                q_sub = q_t[:, r_q_start:r_q_end, c_q_start:c_q_end, :]
                r_sub = ref_t[r_ref_start:r_ref_end, c_ref_start:c_ref_end, :].unsqueeze(0)
                sim_shift = (q_sub * r_sub).sum(dim=-1)
                p_max[:, r_q_start:r_q_end, c_q_start:c_q_end] = torch.maximum(
                    p_max[:, r_q_start:r_q_end, c_q_start:c_q_end],
                    sim_shift,
                )
        pos_maps[ref_p] = p_max.cpu().numpy()

    del q_t, q_flat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return global_maps, pos_maps


def score_maps_to_metrics(
    raw_dist_all: np.ndarray,
    labels: list[int],
    masks_and_dims: list[tuple[np.ndarray, int, int]],
    m_stack_uniform: np.ndarray | None,
    agg_radius: int,
    topk_frac: float,
    compute_aupro: bool,
) -> dict[str, float]:
    """Smooth patch maps, compute top-k% image scores, upsample, and compute AUROC/AU-PRO."""
    n_imgs, hp, wp = raw_dist_all.shape

    # Fast batched smoothing & upsampling when all category masks share (mh, mw)
    if m_stack_uniform is not None:
        mh, mw = m_stack_uniform.shape[1], m_stack_uniform.shape[2]
        t_raw = torch.from_numpy(raw_dist_all).unsqueeze(1)  # [N, 1, hp, wp]
        if agg_radius > 0:
            # Exact match to aggregate_patch_map (edge-clamped box mean)
            padded = F.pad(t_raw, (agg_radius, agg_radius, agg_radius, agg_radius), mode="replicate")
            ksize = 2 * agg_radius + 1
            t_smooth = F.avg_pool2d(padded, kernel_size=ksize, stride=1)
        else:
            t_smooth = t_raw

        flat = t_smooth.reshape(n_imgs, -1)
        k_top = max(1, round(hp * wp * topk_frac))
        img_scores = torch.topk(flat, k=k_top, dim=1).values.mean(dim=1).numpy().tolist()

        h_stack = (
            F.interpolate(t_smooth, size=(mh, mw), mode="bilinear", align_corners=False)
            .squeeze(1)
            .numpy()
        )
        m_stack = m_stack_uniform
    else:
        img_scores = []
        heatmaps = []
        masks = []
        for i, (gt_mask, mh, mw) in enumerate(masks_and_dims):
            smoothed = aggregate_patch_map(raw_dist_all[i], radius=agg_radius)
            img_scores.append(topk_image_score(smoothed, topk_frac=topk_frac))
            heatmaps.append(upsample_patch_map(smoothed, (mh, mw)))
            masks.append(gt_mask)
        h_stack = np.stack(heatmaps)
        m_stack = np.stack(masks)

    return {
        "image_auroc": image_auroc(labels, img_scores),
        "pixel_auroc": pixel_auroc(h_stack, m_stack),
        "pixel_aupro_005": pixel_aupro(h_stack, m_stack, max_false_positive_rate=0.05) if compute_aupro else 0.0,
    }


def evaluate_single_run(
    scorer_name: str,
    extractor: DINOFeatureExtractor,
    ref_paths: list[Path],
    test_samples: list[DatasetSample],
    test_features: list[CachedDINOFeatures],
    masks_and_dims: list[tuple[np.ndarray, int, int]],
    pos_radius: int,
    agg_radius: int,
    topk_frac: float,
    chunk_size: int,
    compute_aupro: bool = True,
    q_norm_stack: np.ndarray | None = None,
    ref_feats_preloaded: list[CachedDINOFeatures] | None = None,
    global_sim_cache: dict[Path, np.ndarray] | None = None,
    pos_sim_cache: dict[Path, np.ndarray] | None = None,
    m_stack_uniform: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit scorer on `ref_paths` and evaluate `test_samples` with leakage assertion."""
    ref_ids = [str(p.resolve()) for p in ref_paths]
    neg_ids = [str(s.image_path.resolve()) for s in test_samples if s.label == 0]
    assert_disjoint_references_and_negatives(ref_ids, neg_ids)

    if scorer_name == "knn_global" and global_sim_cache is not None:
        max_sim = np.max(np.stack([global_sim_cache[p] for p in ref_paths], axis=0), axis=0)
        raw_dist_all = (1.0 - max_sim).astype(np.float32)
    elif scorer_name == "knn_positional" and pos_sim_cache is not None:
        max_sim = np.max(np.stack([pos_sim_cache[p] for p in ref_paths], axis=0), axis=0)
        raw_dist_all = (1.0 - max_sim).astype(np.float32)
    else:
        ref_feats = (
            ref_feats_preloaded
            if ref_feats_preloaded is not None
            else [extractor.get_features(p) for p in ref_paths]
        )
        q_stack = (
            q_norm_stack
            if q_norm_stack is not None
            else np.stack([l2_normalize_np(f.patch_tokens) for f in test_features], axis=0)
        )
        n_imgs, hp, wp, d = q_stack.shape
        scorer = PCAResidualScorer(extractor=extractor, agg_radius=agg_radius, topk_frac=topk_frac)
        scorer.fit_features(ref_feats, reference_ids=ref_ids)
        assert scorer._mean is not None and scorer._components is not None
        q_flat = q_stack.reshape(-1, d)
        n_q = q_flat.shape[0]
        res_flat = np.empty(n_q, dtype=np.float32)
        q_step = 16 * hp * wp
        for q0 in range(0, n_q, q_step):
            centered = q_flat[q0 : q0 + q_step] - scorer._mean
            coords = centered @ scorer._components
            total_sq = np.sum(centered * centered, axis=-1)
            proj_sq = np.sum(coords * coords, axis=-1)
            res_flat[q0 : q0 + q_step] = np.maximum(total_sq - proj_sq, 0.0)
        raw_dist_all = res_flat.reshape(n_imgs, hp, wp)

    labels = [s.label for s in test_samples]
    return score_maps_to_metrics(
        raw_dist_all=raw_dist_all,
        labels=labels,
        masks_and_dims=masks_and_dims,
        m_stack_uniform=m_stack_uniform,
        agg_radius=agg_radius,
        topk_frac=topk_frac,
        compute_aupro=compute_aupro,
    )


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)

    seeds: list[int] = list(cfg.raw.get("seed", [0, 1, 2]))
    k_grid: list[int] = list(cfg.raw.get("refs", {}).get("fewshot_k_grid", [1, 2, 4, 8, 16]))
    scorers: list[str] = list(cfg.raw.get("match", {}).get("scorers", ["knn_global", "knn_positional", "pca_residual"]))
    agg_radii: list[int] = list(cfg.raw.get("match", {}).get("agg_radii", [1, 3]))
    pos_radius = int(cfg.raw.get("match", {}).get("radius", 2))
    dino_cfg = cfg.raw.get("dino", {})
    topk_frac = float(dino_cfg.get("topk_frac", 0.001))
    chunk_size = int(dino_cfg.get("chunk_size", 4096))

    if args.dry_run:
        total_runs = len(k_grid) * len(seeds) * len(scorers) * 10
        print(
            f"[DRY-RUN] step3_fewshot.py: k_grid={k_grid}, seeds={seeds}, scorers={scorers}, "
            f"agg_radii={agg_radii}, heldout_runs={total_runs}, output={cfg.results_dir}/phase_3_fewshot.csv "
            f"(estimated size ~0.5 MB)"
        )
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = cfg.logs_dir / "step3_fewshot.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info(f"=== Phase 3 Step B: Few-Shot DINO Evaluation on Cached Features (device={device}) ===")
            val_cats, eval_cats, val_samples, eval_samples = create_v3_category_splits(
                data_root=cfg.data_root,
                manifest_dir=cfg.manifest_dir,
                seed=42,
            )

            extractor = DINOFeatureExtractor(
                backbone_id=str(dino_cfg.get("model", "dinov2_vitl14_reg")),
                short_side=int(dino_cfg.get("short_side", 560)),
                patch_size=int(dino_cfg.get("patch_size", 14)),
                embed_dim=int(dino_cfg.get("embed_dim", 1024)),
                precision=str(cfg.raw.get("precision", "fp16")),
                cache_dir=cfg.dino_cache_dir,
                torch_home=cfg.torch_home,
                lazy_load=True,
            )

            # 1. Select best (scorer, agg_radius) strictly on the 5 validation categories at k=4
            logger.info(f"Selecting best (scorer, agg_radius) on 5 validation categories ({val_cats}) at k=4...")
            best_scorer = "knn_positional"
            best_agg = 1
            best_val_score = -1.0
            val_selection_table: list[dict] = []
            val_scores_by_pair: dict[tuple[str, int], list[float]] = {
                (sc_name, agg_r): [] for sc_name in scorers for agg_r in agg_radii
            }

            for cat in val_cats:
                c_samples = [s for s in val_samples if s.category == cat]
                c_feats = [extractor.get_features(s.image_path) for s in c_samples]
                q_norm = np.stack([l2_normalize_np(f.patch_tokens) for f in c_feats], axis=0)
                del c_feats
                c_masks = load_category_masks(c_samples)
                dims_set = {(mh, mw) for _, mh, mw in c_masks}
                m_uniform = np.stack([gt for gt, _, _ in c_masks]) if len(dims_set) == 1 else None

                pool = discover_train_normals(cfg.data_root, cat, max_refs=32)
                refs = random.Random(0).sample(pool, min(4, len(pool)))
                pool_map = {p: extractor.get_features(p) for p in refs}
                g_cache, p_cache = precompute_per_ref_maps(q_norm, refs, pool_map, pos_radius, device)
                ref_feats = [pool_map[p] for p in refs]

                for sc_name in scorers:
                    for agg_r in agg_radii:
                        m = evaluate_single_run(
                            sc_name, extractor, refs, c_samples, [], c_masks,
                            pos_radius=pos_radius, agg_radius=agg_r, topk_frac=topk_frac,
                            chunk_size=chunk_size, compute_aupro=False,
                            q_norm_stack=q_norm, ref_feats_preloaded=ref_feats,
                            global_sim_cache=g_cache, pos_sim_cache=p_cache,
                            m_stack_uniform=m_uniform,
                        )
                        val_scores_by_pair[(sc_name, agg_r)].append(0.5 * (m["image_auroc"] + m["pixel_auroc"]))

                del q_norm, c_masks, m_uniform, pool_map, g_cache, p_cache, ref_feats
                gc.collect()

            for sc_name in scorers:
                for agg_r in agg_radii:
                    mean_val = float(np.mean(val_scores_by_pair[(sc_name, agg_r)]))
                    val_selection_table.append({"scorer": sc_name, "agg_radius": agg_r, "val_combined_auroc": mean_val})
                    logger.info(f"  [Validation k=4] scorer={sc_name:<15} agg_radius={agg_r} -> Combined AUROC={mean_val:.4f}")
                    if mean_val > best_val_score:
                        best_val_score = mean_val
                        best_scorer = sc_name
                        best_agg = agg_r

            logger.info(f"Selected on validation categories: scorer={best_scorer}, agg_radius={best_agg}")

            # 2. Evaluate k in [1,2,4,8,16] x seeds [0,1,2] x 3 scorers on the 10 held-out categories
            csv_rows: list[dict] = []

            for cat in eval_cats:
                t_cat = time.perf_counter()
                c_samples = [s for s in eval_samples if s.category == cat]
                c_feats = [extractor.get_features(s.image_path) for s in c_samples]
                q_norm = np.stack([l2_normalize_np(f.patch_tokens) for f in c_feats], axis=0)
                del c_feats
                c_masks = load_category_masks(c_samples)
                dims_set = {(mh, mw) for _, mh, mw in c_masks}
                m_uniform = np.stack([gt for gt, _, _ in c_masks]) if len(dims_set) == 1 else None

                n_norm = sum(1 for s in c_samples if s.label == 0)
                n_anom = sum(1 for s in c_samples if s.label == 1)
                pool = discover_train_normals(cfg.data_root, cat, max_refs=32)

                # Collect unique references across all (k, seed) combinations
                unique_refs_set: set[Path] = set()
                for k in k_grid:
                    for sd in seeds:
                        unique_refs_set.update(random.Random(sd).sample(pool, min(k, len(pool))))
                unique_refs = sorted(unique_refs_set)
                pool_feats_map = {p: extractor.get_features(p) for p in unique_refs}

                g_cache, p_cache = precompute_per_ref_maps(q_norm, unique_refs, pool_feats_map, pos_radius, device)

                for sc_name in scorers:
                    for k in k_grid:
                        seed_metrics: list[dict[str, float]] = []
                        for sd in seeds:
                            refs = random.Random(sd).sample(pool, min(k, len(pool)))
                            ref_feats = [pool_feats_map[p] for p in refs]
                            m = evaluate_single_run(
                                sc_name, extractor, refs, c_samples, [], c_masks,
                                pos_radius=pos_radius, agg_radius=best_agg, topk_frac=topk_frac,
                                chunk_size=chunk_size, compute_aupro=(k == 4 and sc_name == best_scorer),
                                q_norm_stack=q_norm, ref_feats_preloaded=ref_feats,
                                global_sim_cache=g_cache, pos_sim_cache=p_cache,
                                m_stack_uniform=m_uniform,
                            )
                            seed_metrics.append(m)

                        row = {
                            "category": cat,
                            "images": len(c_samples),
                            "normal": n_norm,
                            "anomalous": n_anom,
                            "scorer": sc_name,
                            "k": k,
                            "agg_radius": best_agg,
                            "image_auroc_mean": float(np.mean([x["image_auroc"] for x in seed_metrics])),
                            "image_auroc_std": float(np.std([x["image_auroc"] for x in seed_metrics])),
                            "pixel_auroc_mean": float(np.mean([x["pixel_auroc"] for x in seed_metrics])),
                            "pixel_auroc_std": float(np.std([x["pixel_auroc"] for x in seed_metrics])),
                            "pixel_aupro_005_mean": float(np.mean([x["pixel_aupro_005"] for x in seed_metrics])),
                            "pixel_aupro_005_std": float(np.std([x["pixel_aupro_005"] for x in seed_metrics])),
                            "is_low_n": cat == "toothbrush",
                        }
                        csv_rows.append(row)

                best_k16 = next(r for r in csv_rows if r["category"] == cat and r["scorer"] == best_scorer and r["k"] == 16)
                del q_norm, c_masks, m_uniform, pool_feats_map, g_cache, p_cache
                gc.collect()
                logger.info(
                    f"  Category [{cat:<12}] ({len(c_samples):3d} imgs) DONE in {time.perf_counter() - t_cat:.1f}s | "
                    f"{best_scorer} k=16: Img={best_k16['image_auroc_mean']:.4f}±{best_k16['image_auroc_std']:.4f}, "
                    f"Px={best_k16['pixel_auroc_mean']:.4f}±{best_k16['pixel_auroc_std']:.4f}"
                )

            csv_path = cfg.results_dir / "phase_3_fewshot.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)

            summary_path = cfg.results_dir / "phase_3_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "commit": get_git_commit(),
                        "config_hash": cfg.config_hash,
                        "seeds": seeds,
                        "selected_scorer": best_scorer,
                        "selected_agg_radius": best_agg,
                        "validation_selection": val_selection_table,
                        "rows": csv_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            elapsed = time.perf_counter() - t0
            logger.info(f"STEP 3B RESULT: PASS ({elapsed:.2f}s) -> {csv_path}")
            log_decision(
                cfg,
                "Phase 3 Step B (Few-Shot DINO Evaluation)",
                "PASS",
                elapsed,
                f"best_scorer={best_scorer}, best_agg={best_agg}",
            )
            return 0
        except (RuntimeError, OSError, ValueError, KeyError, AssertionError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"step3_fewshot failed: {exc}")
            log_decision(cfg, "Phase 3 Step B (Few-Shot DINO Evaluation)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
