"""Step 5A: Compare CLIP-only, DINO-only, Stage A only, and Fused (A+B) on held-out categories.

Selects fusion weights (w_d, w_c) from config `fusion.weight_grid` strictly on the 5 validation
categories, fits logistic calibration on validation categories ("calibrated, not strictly zero-shot"),
and evaluates the 10 held-out MVTec categories on cached features.
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
from v3.core.bank import DINOFeatureExtractor
from v3.core.calibrate import LogisticCalibrator
from v3.core.contracts import ROI, Recipe
from v3.core.defect.aggregate import upsample_patch_map
from v3.core.features import CLIPFeatureExtractor
from v3.core.fusion import BranchFuser
from v3.core.roi import StageAROIEngine
from v3.eval.metrics import image_auroc, pixel_aupro, pixel_auroc, recall_at_frr
from v3.eval.phase2_engine import Phase2Scorer
from v3.eval.splits import (
    DatasetSample,
    assert_disjoint_references_and_negatives,
    create_v3_category_splits,
)
from v3.eval.step3_cache_dino import discover_train_normals
from v3.eval.step3_fewshot import build_scorer, load_category_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5A: Compare CLIP-only, DINO-only, Stage A only, and Fused A+B.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned comparison and estimated output size, then exit.")
    return parser.parse_args()


def compute_category_branch_outputs(
    cat: str,
    samples: list[DatasetSample],
    ref_paths: list[Path],
    dino_ext: DINOFeatureExtractor,
    clip_scorer: Phase2Scorer,
    scorer_name: str,
    pos_radius: int,
    agg_radius: int,
    topk_frac: float,
    chunk_size: int,
    min_mad: float,
) -> dict:
    """Compute per-sample scores and maps for CLIP, DINO, and Stage A on a category."""
    ref_ids = [str(p.resolve()) for p in ref_paths]
    neg_ids = [str(s.image_path.resolve()) for s in samples if s.label == 0]
    assert_disjoint_references_and_negatives(ref_ids, neg_ids)

    ref_feats = [dino_ext.get_features(p) for p in ref_paths]
    dino_scorer = build_scorer(
        scorer_name=scorer_name,
        extractor=dino_ext,
        radius=pos_radius,
        agg_radius=agg_radius,
        topk_frac=topk_frac,
        chunk_size=chunk_size,
    )
    dino_scorer.fit_features(ref_feats, reference_ids=ref_ids)

    # Fit Stage A global presence ROI on reference normal images
    ref_imgs = [Image.open(p).convert("RGB") for p in ref_paths]
    w0, h0 = ref_imgs[0].size
    cat_recipe = Recipe(
        name=cat,
        template_path=str(ref_paths[0]),
        resolution=(w0, h0),
        view="fixed",
        symmetry="none",
        stable_roi=[(0, 0), (w0, 0), (w0, h0), (0, h0)],
        rois=[ROI(id="global_presence", kind="presence", polygon=[(0, 0), (w0, 0), (w0, h0), (0, h0)], severity=1, hard=False, params={})],
    )
    stage_a_engine = StageAROIEngine(min_mad=min_mad)
    stage_a_engine.fit(ref_imgs, cat_recipe)
    for im in ref_imgs:
        im.close()

    # Reference normal scores for robust normalization
    ref_dino_scores = [dino_scorer.score_patch_grid(rf.patch_tokens)[0] for rf in ref_feats]
    ref_clip_scores = [
        clip_scorer.score_sample(DatasetSample(p, None, 0, cat, "ref"), need_heatmap=False)[0]
        for p in ref_paths
    ]

    masks_and_dims = load_category_masks(samples, metric_max_side=256)
    labels = [s.label for s in samples]
    clip_scores, dino_scores, stage_a_scores = [], [], []
    clip_maps, dino_maps, gt_masks = [], [], []

    for s, (gt_m, mh, mw) in zip(samples, masks_and_dims, strict=True):
        c_logit, _, c_map, _ = clip_scorer.score_sample(s, metric_max_side=256, need_heatmap=True)
        d_feat = dino_ext.get_features(s.image_path)
        d_score, d_patch_map = dino_scorer.score_patch_grid(d_feat.patch_tokens)
        d_map = upsample_patch_map(d_patch_map, (mh, mw))

        with Image.open(s.image_path) as q_img:
            _, z_dict = stage_a_engine.inspect(q_img.convert("RGB").resize((w0, h0)), cat_recipe)
        a_score = float(z_dict.get("global_presence", 0.0))

        clip_scores.append(c_logit)
        dino_scores.append(d_score)
        stage_a_scores.append(a_score)
        clip_maps.append(c_map)
        dino_maps.append(d_map)
        gt_masks.append(gt_m)

    return {
        "labels": labels,
        "ref_dino_scores": ref_dino_scores,
        "ref_clip_scores": ref_clip_scores,
        "clip_scores": clip_scores,
        "dino_scores": dino_scores,
        "stage_a_scores": stage_a_scores,
        "clip_maps": np.stack(clip_maps),
        "dino_maps": np.stack(dino_maps),
        "gt_masks": np.stack(gt_masks),
    }


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    weight_grid = cfg.raw.get("fusion", {}).get("weight_grid", [[1.0, 0.0], [0.85, 0.15], [0.70, 0.30], [0.50, 0.50], [0.0, 1.0]])

    if args.dry_run:
        print(
            f"[DRY-RUN] step5_compare.py: weight_grid={weight_grid}, modes=[clip_only, dino_only, stage_a_only, fused_ab] "
            f"-> output={cfg.results_dir}/phase_5_compare.csv (estimated size ~0.1 MB)"
        )
        return 0

    log_path = cfg.logs_dir / "step5_compare.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info("=== Phase 5 Step A: Branch Comparison & Validation Weight Selection ===")
            val_cats, eval_cats, val_samples, eval_samples = create_v3_category_splits(
                data_root=cfg.data_root,
                manifest_dir=cfg.manifest_dir,
                seed=42,
            )

            dino_cfg = cfg.raw.get("dino", {})
            dino_ext = DINOFeatureExtractor(
                backbone_id=str(dino_cfg.get("model", "dinov2_vitl14_reg")),
                short_side=int(dino_cfg.get("short_side", 560)),
                patch_size=int(dino_cfg.get("patch_size", 14)),
                embed_dim=int(dino_cfg.get("embed_dim", 1024)),
                precision=str(cfg.raw.get("precision", "fp16")),
                cache_dir=cfg.dino_cache_dir,
                torch_home=cfg.torch_home,
                lazy_load=True,
            )
            clip_ext = CLIPFeatureExtractor(backbone_id="clip_l14_336", device="cpu", precision="fp16", cache_dir=cfg.cache_dir)
            clip_scorer = Phase2Scorer(extractor=clip_ext, vocabulary_mode="domain_generic", polarity_mode="intermediate_inverted")

            scorer_name = str(cfg.raw.get("match", {}).get("mode", "knn_positional"))
            pos_radius = int(cfg.raw.get("match", {}).get("radius", 2))
            agg_radius = int(cfg.raw.get("match", {}).get("agg_radii", [1, 3])[0])
            topk_frac = float(dino_cfg.get("topk_frac", 0.001))
            chunk_size = int(dino_cfg.get("chunk_size", 4096))
            min_mad = float(cfg.raw.get("decide", {}).get("min_mad", 1e-3))
            k_refs = int(cfg.raw.get("refs", {}).get("k", 4))

            # 1. Validation category evaluation for fusion weight selection
            val_cache: dict[str, dict] = {}
            for cat in val_cats:
                c_samples = [s for s in val_samples if s.category == cat]
                pool = discover_train_normals(cfg.data_root, cat, max_refs=32)
                refs = random.Random(0).sample(pool, min(k_refs, len(pool)))
                val_cache[cat] = compute_category_branch_outputs(
                    cat, c_samples, refs, dino_ext, clip_scorer, scorer_name, pos_radius, agg_radius, topk_frac, chunk_size, min_mad
                )

            best_wd, best_wc = 0.85, 0.15
            best_val_auroc = -1.0
            for wd, wc in weight_grid:
                cat_aurocs = []
                for cat in val_cats:
                    vc = val_cache[cat]
                    fuser = BranchFuser(w_d=wd, w_c=wc, min_mad=min_mad)
                    fuser.fit(vc["ref_dino_scores"], vc["ref_clip_scores"])
                    fused_s = [fuser.fuse_score(ds, cs) for ds, cs in zip(vc["dino_scores"], vc["clip_scores"], strict=True)]
                    cat_aurocs.append(image_auroc(vc["labels"], fused_s))
                m_auroc = float(np.mean(cat_aurocs))
                logger.info(f"  [Validation Fusion Grid] w_d={wd:.2f}, w_c={wc:.2f} -> Val Macro Image AUROC={m_auroc:.4f}")
                if m_auroc > best_val_auroc:
                    best_val_auroc = m_auroc
                    best_wd, best_wc = float(wd), float(wc)

            logger.info(f"Selected fusion weights on validation set: w_d={best_wd:.2f}, w_c={best_wc:.2f}")

            # Fit logistic calibrator on validation fused scores ("calibrated, not strictly zero-shot")
            all_val_fused: list[float] = []
            all_val_labels: list[int] = []
            for cat in val_cats:
                vc = val_cache[cat]
                fuser = BranchFuser(w_d=best_wd, w_c=best_wc, min_mad=min_mad)
                fuser.fit(vc["ref_dino_scores"], vc["ref_clip_scores"])
                all_val_fused.extend(fuser.fuse_score(ds, cs) for ds, cs in zip(vc["dino_scores"], vc["clip_scores"], strict=True))
                all_val_labels.extend(vc["labels"])

            calibrator = LogisticCalibrator()
            temp, bias = calibrator.fit(all_val_fused, all_val_labels)
            val_ece = calibrator.evaluate_ece(all_val_fused, all_val_labels)
            logger.info(f"Validation Calibration (not strictly zero-shot): T={temp:.5f}, bias={bias:.5f}, Val ECE={val_ece:.4f}")

            # 2. Held-out evaluation across all 4 modes
            rows: list[dict] = []
            for cat in eval_cats:
                c_samples = [s for s in eval_samples if s.category == cat]
                pool = discover_train_normals(cfg.data_root, cat, max_refs=32)
                refs = random.Random(0).sample(pool, min(k_refs, len(pool)))
                ec = compute_category_branch_outputs(
                    cat, c_samples, refs, dino_ext, clip_scorer, scorer_name, pos_radius, agg_radius, topk_frac, chunk_size, min_mad
                )

                fuser = BranchFuser(w_d=best_wd, w_c=best_wc, min_mad=min_mad)
                fuser.fit(ec["ref_dino_scores"], ec["ref_clip_scores"])
                fused_scores = [
                    max(fuser.fuse_score(ds, cs), 0.25 * a_s)
                    for ds, cs, a_s in zip(ec["dino_scores"], ec["clip_scores"], ec["stage_a_scores"], strict=True)
                ]
                fused_maps = best_wd * ec["dino_maps"] + best_wc * ec["clip_maps"]

                y = ec["labels"]
                n_norm = sum(1 for lbl in y if lbl == 0)
                n_anom = sum(1 for lbl in y if lbl == 1)

                row = {
                    "category": cat,
                    "images": len(y),
                    "normal": n_norm,
                    "anomalous": n_anom,
                    "clip_only_img_auroc": image_auroc(y, ec["clip_scores"]),
                    "clip_only_px_auroc": pixel_auroc(ec["clip_maps"], ec["gt_masks"]),
                    "dino_only_img_auroc": image_auroc(y, ec["dino_scores"]),
                    "dino_only_px_auroc": pixel_auroc(ec["dino_maps"], ec["gt_masks"]),
                    "stage_a_only_img_auroc": image_auroc(y, ec["stage_a_scores"]),
                    "fused_ab_img_auroc": image_auroc(y, fused_scores),
                    "fused_ab_px_auroc": pixel_auroc(fused_maps, ec["gt_masks"]),
                    "fused_ab_aupro_005": pixel_aupro(fused_maps, ec["gt_masks"], max_false_positive_rate=0.05),
                    "fused_ab_recall_5frr": recall_at_frr(y, fused_scores, false_reject_rate=0.05),
                    "fused_ab_ece": calibrator.evaluate_ece(fused_scores, y),
                    "is_low_n": cat == "toothbrush",
                }
                rows.append(row)
                logger.info(
                    f"  {cat:<12} ({len(y)}): CLIP={row['clip_only_img_auroc']:.4f} | "
                    f"DINO={row['dino_only_img_auroc']:.4f} | StageA={row['stage_a_only_img_auroc']:.4f} | "
                    f"Fused={row['fused_ab_img_auroc']:.4f} (Px={row['fused_ab_px_auroc']:.4f}, PRO@0.05={row['fused_ab_aupro_005']:.4f})"
                )

            csv_path = cfg.results_dir / "phase_5_compare.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            summary = {
                "status": "PASS",
                "commit": get_git_commit(),
                "config_hash": cfg.config_hash,
                "calibration_status": "calibrated, not strictly zero-shot",
                "selected_w_d": best_wd,
                "selected_w_c": best_wc,
                "temperature": temp,
                "bias": bias,
                "macro_clip_only_img_auroc": float(np.mean([r["clip_only_img_auroc"] for r in rows])),
                "macro_dino_only_img_auroc": float(np.mean([r["dino_only_img_auroc"] for r in rows])),
                "macro_stage_a_only_img_auroc": float(np.mean([r["stage_a_only_img_auroc"] for r in rows])),
                "macro_fused_ab_img_auroc": float(np.mean([r["fused_ab_img_auroc"] for r in rows])),
                "macro_fused_ab_px_auroc": float(np.mean([r["fused_ab_px_auroc"] for r in rows])),
                "macro_fused_ab_aupro_005": float(np.mean([r["fused_ab_aupro_005"] for r in rows])),
                "categories": rows,
            }
            (cfg.results_dir / "phase_5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            elapsed = time.perf_counter() - t0
            logger.info(f"STEP 5A RESULT: PASS ({elapsed:.2f}s) -> {csv_path}")
            log_decision(cfg, "Phase 5 Step A (Branch Comparison)", "PASS", elapsed, f"Fused_AUROC={summary['macro_fused_ab_img_auroc']:.4f}")
            return 0
        except (RuntimeError, OSError, ValueError, KeyError, AssertionError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"step5_compare failed: {exc}")
            log_decision(cfg, "Phase 5 Step A (Branch Comparison)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
