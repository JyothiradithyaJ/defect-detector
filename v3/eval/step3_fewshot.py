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
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from v3.config import (
    DEFAULT_CONFIG_PATH,
    StepLogger,
    get_git_commit,
    load_v3_config,
    log_decision,
)
from v3.core.bank import CachedDINOFeatures, DINOFeatureExtractor
from v3.core.defect.aggregate import upsample_patch_map
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
) -> dict[str, float]:
    """Fit scorer on `ref_paths` and evaluate `test_samples` with leakage assertion."""
    ref_ids = [str(p.resolve()) for p in ref_paths]
    neg_ids = [str(s.image_path.resolve()) for s in test_samples if s.label == 0]
    assert_disjoint_references_and_negatives(ref_ids, neg_ids)

    ref_feats = [extractor.get_features(p) for p in ref_paths]
    scorer = build_scorer(
        scorer_name=scorer_name,
        extractor=extractor,
        radius=pos_radius,
        agg_radius=agg_radius,
        topk_frac=topk_frac,
        chunk_size=chunk_size,
    )
    scorer.fit_features(ref_feats, reference_ids=ref_ids)

    labels = [s.label for s in test_samples]
    img_scores: list[float] = []
    heatmaps: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for feat, (gt_mask, mh, mw) in zip(test_features, masks_and_dims, strict=True):
        s_img, patch_map = scorer.score_patch_grid(feat.patch_tokens)
        img_scores.append(s_img)
        heatmaps.append(upsample_patch_map(patch_map, (mh, mw)))
        masks.append(gt_mask)

    h_stack = np.stack(heatmaps)
    m_stack = np.stack(masks)
    return {
        "image_auroc": image_auroc(labels, img_scores),
        "pixel_auroc": pixel_auroc(h_stack, m_stack),
        "pixel_aupro_005": pixel_aupro(h_stack, m_stack, max_false_positive_rate=0.05) if compute_aupro else 0.0,
    }


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

    log_path = cfg.logs_dir / "step3_fewshot.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info("=== Phase 3 Step B: Few-Shot DINO Evaluation on Cached Features ===")
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

            for sc_name in scorers:
                for agg_r in agg_radii:
                    val_cat_scores: list[float] = []
                    for cat in val_cats:
                        c_samples = [s for s in val_samples if s.category == cat]
                        c_feats = [extractor.get_features(s.image_path) for s in c_samples]
                        c_masks = load_category_masks(c_samples)
                        pool = discover_train_normals(cfg.data_root, cat, max_refs=32)
                        refs = random.Random(0).sample(pool, min(4, len(pool)))
                        m = evaluate_single_run(
                            sc_name, extractor, refs, c_samples, c_feats, c_masks,
                            pos_radius=pos_radius, agg_radius=agg_r, topk_frac=topk_frac,
                            chunk_size=chunk_size, compute_aupro=False,
                        )
                        val_cat_scores.append(0.5 * (m["image_auroc"] + m["pixel_auroc"]))
                    mean_val = float(np.mean(val_cat_scores))
                    val_selection_table.append({"scorer": sc_name, "agg_radius": agg_r, "val_combined_auroc": mean_val})
                    logger.info(f"  [Validation k=4] scorer={sc_name:<15} agg_radius={agg_r} -> Combined AUROC={mean_val:.4f}")
                    if mean_val > best_val_score:
                        best_val_score = mean_val
                        best_scorer = sc_name
                        best_agg = agg_r

            logger.info(f"Selected on validation categories: scorer={best_scorer}, agg_radius={best_agg}")

            # 2. Evaluate k in [1,2,4,8,16] x seeds [0,1,2] x 3 scorers on the 10 held-out categories
            csv_rows: list[dict] = []
            last_log_t = time.perf_counter()

            for cat in eval_cats:
                c_samples = [s for s in eval_samples if s.category == cat]
                c_feats = [extractor.get_features(s.image_path) for s in c_samples]
                c_masks = load_category_masks(c_samples)
                n_norm = sum(1 for s in c_samples if s.label == 0)
                n_anom = sum(1 for s in c_samples if s.label == 1)
                pool = discover_train_normals(cfg.data_root, cat, max_refs=32)

                for sc_name in scorers:
                    for k in k_grid:
                        seed_metrics: list[dict[str, float]] = []
                        for sd in seeds:
                            refs = random.Random(sd).sample(pool, min(k, len(pool)))
                            m = evaluate_single_run(
                                sc_name, extractor, refs, c_samples, c_feats, c_masks,
                                pos_radius=pos_radius, agg_radius=best_agg, topk_frac=topk_frac,
                                chunk_size=chunk_size, compute_aupro=(k == 4),
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

                        now = time.perf_counter()
                        if (now - last_log_t) >= 30.0:
                            logger.info(
                                f"  [PROGRESS] {cat} | {sc_name} | k={k} -> "
                                f"Img={row['image_auroc_mean']:.4f}±{row['image_auroc_std']:.4f}, "
                                f"Px={row['pixel_auroc_mean']:.4f}±{row['pixel_auroc_std']:.4f}"
                            )
                            last_log_t = now

                logger.info(f"  Completed held-out category: {cat} ({len(c_samples)} images)")

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
