"""Step 4: Stage A standalone evaluation on validation normals and synthetic/LOCO logical anomalies.

Reports:
- Precision and recall per ROI kind (presence, count, order, position) on validation perturbations
- Empirical false-reject rate on validation normals vs configured decide.false_reject_rate (5%)
- Asserts that all recipe YAML file hashes remain unchanged after evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from v3.config import (
    DEFAULT_CONFIG_PATH,
    StepLogger,
    get_git_commit,
    load_v3_config,
    log_decision,
)
from v3.core.align import Stage1Aligner
from v3.core.contracts import ROI, Recipe
from v3.core.qc import QualityChecker
from v3.core.roi import StageAROIEngine, compute_recipe_file_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4: Evaluate Stage A alone (QC, Align, ROI detectors).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned Stage A evaluation and exit.")
    return parser.parse_args()


def build_validation_assembly_dataset(seed: int = 42) -> tuple[Recipe, list[Image.Image], list[Image.Image], list[tuple[Image.Image, str]]]:
    """Generate a deterministic canonical assembly validation suite covering all 4 ROI kinds."""
    rng = np.random.default_rng(seed)
    w, h = 256, 256
    recipe = Recipe(
        name="val_assembly_board",
        template_path="",
        resolution=(w, h),
        view="fixed",
        symmetry="none",
        stable_roi=[(16, 16), (240, 16), (240, 240), (16, 240)],
        rois=[
            ROI(id="roi_presence", kind="presence", polygon=[(24, 24), (104, 24), (104, 104), (24, 104)], severity=0, hard=True, params={}, z_threshold=3.5),
            ROI(id="roi_count", kind="count", polygon=[(136, 24), (232, 24), (232, 104), (136, 104)], severity=1, hard=True, params={"expected_count": 3, "intensity_threshold": 160.0, "min_blob_pixels": 6}, z_threshold=3.5),
            ROI(id="roi_order", kind="order", polygon=[(24, 136), (116, 136), (116, 224), (24, 224)], severity=0, hard=True, params={"num_slots": 2, "order_axis": "horizontal"}, z_threshold=3.5),
            ROI(id="roi_position", kind="position", polygon=[(140, 136), (228, 136), (228, 224), (140, 224)], severity=1, hard=False, params={}, z_threshold=3.5),
        ],
    )

    def make_normal() -> Image.Image:
        noise = rng.integers(-3, 4, size=(h, w, 3), dtype=np.int16)
        base = np.clip(64 + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(base, mode="RGB")
        draw = ImageDraw.Draw(img)
        # 1. Presence block (bright green connector)
        draw.rectangle([(36, 36), (92, 92)], fill=(40, 200, 80))
        # 2. Count block (3 bright pins)
        for px in (152, 180, 208):
            draw.rectangle([(px - 5, 52), (px + 5, 72)], fill=(235, 235, 235))
        # 3. Order block (left red wire, right blue wire)
        draw.rectangle([(30, 148), (66, 212)], fill=(220, 50, 50))
        draw.rectangle([(74, 148), (110, 212)], fill=(50, 90, 220))
        # 4. Position block (centered bright component at (184, 180))
        draw.rectangle([(174, 170), (194, 190)], fill=(240, 200, 60))
        return img

    ref_normals = [make_normal() for _ in range(12)]
    val_normals = [make_normal() for _ in range(40)]
    anomalies: list[tuple[Image.Image, str]] = []

    for _ in range(10):
        # Missing presence component
        im_p = make_normal()
        ImageDraw.Draw(im_p).rectangle([(36, 36), (92, 92)], fill=(64, 64, 64))
        anomalies.append((im_p, "presence"))

        # Wrong pin count (only 1 pin instead of 3)
        im_c = make_normal()
        ImageDraw.Draw(im_c).rectangle([(142, 44), (222, 82)], fill=(64, 64, 64))
        ImageDraw.Draw(im_c).rectangle([(152 - 5, 52), (152 + 5, 72)], fill=(235, 235, 235))
        anomalies.append((im_c, "count"))

        # Swapped wire order (left blue, right red)
        im_o = make_normal()
        ImageDraw.Draw(im_o).rectangle([(30, 148), (66, 212)], fill=(50, 90, 220))
        ImageDraw.Draw(im_o).rectangle([(74, 148), (110, 212)], fill=(220, 50, 50))
        anomalies.append((im_o, "order"))

        # Shifted component position
        im_pos = make_normal()
        ImageDraw.Draw(im_pos).rectangle([(174, 170), (194, 190)], fill=(64, 64, 64))
        ImageDraw.Draw(im_pos).rectangle([(144, 140), (164, 160)], fill=(240, 200, 60))
        anomalies.append((im_pos, "position"))

    return recipe, ref_normals, val_normals, anomalies


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)

    if args.dry_run:
        print(
            f"[DRY-RUN] step4_stage_a_eval.py: evaluates Stage A (QC + Align + 4 ROI kinds) -> "
            f"output={cfg.results_dir}/phase_4_stage_a.json (estimated size ~8 KB)"
        )
        return 0

    log_path = cfg.logs_dir / "step4_stage_a_eval.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info("=== Phase 4: Stage A Standalone Validation Evaluation ===")
            recipe_files = sorted(cfg.recipes_dir.glob("*.yaml"))
            hashes_before = {p.name: compute_recipe_file_hash(p) for p in recipe_files}

            recipe, ref_normals, val_normals, val_anomalies = build_validation_assembly_dataset(seed=42)
            qc = QualityChecker(
                blur_floor_percentile=float(cfg.raw.get("qc", {}).get("blur_floor_percentile", 5.0)),
                min_mad=float(cfg.raw.get("decide", {}).get("min_mad", 1e-3)),
            )
            aligner = Stage1Aligner(
                max_align_residual=float(cfg.raw.get("stage_a", {}).get("max_align_residual", 0.35)),
                variant_min_cosine=float(cfg.raw.get("stage_a", {}).get("variant_min_cosine", 0.80)),
            )
            engine = StageAROIEngine(
                min_mad=float(cfg.raw.get("decide", {}).get("min_mad", 1e-3)),
                default_z_threshold=float(cfg.raw.get("stage_a", {}).get("default_z_threshold", 3.5)),
                early_exit=bool(cfg.raw.get("early_exit", False)),
            )

            qc.fit(ref_normals)
            engine.fit(ref_normals, recipe)
            template = ref_normals[0]

            # Evaluate normal false-reject rate
            false_rejects = 0
            for img in val_normals:
                qc_res = qc.check(img)
                aln = aligner.align(img, template, recipe)
                mms, _ = engine.inspect(aln.aligned_image, recipe, skip_positional=aln.diverged)
                if not qc_res.passed or aln.diverged or len(mms) > 0:
                    false_rejects += 1
            normal_frr = false_rejects / max(1, len(val_normals))
            target_frr = float(cfg.raw.get("decide", {}).get("false_reject_rate", 0.05))
            logger.info(f"Normal False-Reject Rate: {normal_frr:.4f} (Target <= {target_frr:.4f})")

            # Evaluate per-kind precision and recall
            kinds = ["presence", "count", "order", "position"]
            tp = {k: 0 for k in kinds}
            fp = {k: 0 for k in kinds}
            fn = {k: 0 for k in kinds}

            for img, gt_kind in val_anomalies:
                aln = aligner.align(img, template, recipe)
                mms, _ = engine.inspect(aln.aligned_image, recipe, skip_positional=aln.diverged)
                pred_kinds = {m.kind for m in mms}
                for k in kinds:
                    if k == gt_kind:
                        if k in pred_kinds:
                            tp[k] += 1
                        else:
                            fn[k] += 1
                    elif k in pred_kinds:
                        fp[k] += 1

            per_kind_metrics: dict[str, dict[str, float]] = {}
            for k in kinds:
                prec = tp[k] / max(1, tp[k] + fp[k])
                rec = tp[k] / max(1, tp[k] + fn[k])
                per_kind_metrics[k] = {"precision": round(prec, 4), "recall": round(rec, 4), "tp": tp[k], "fp": fp[k], "fn": fn[k]}
                logger.info(f"  ROI Kind [{k:<10}]: Precision={prec:.4f}, Recall={rec:.4f} (TP={tp[k]}, FP={fp[k]}, FN={fn[k]})")

            hashes_after = {p.name: compute_recipe_file_hash(p) for p in recipe_files}
            if hashes_before != hashes_after:
                raise AssertionError("Recipe YAML file hash changed during evaluation!")
            logger.info("Recipe YAML immutability check: PASS (all SHA-256 hashes unchanged)")

            out_payload = {
                "status": "PASS",
                "commit": get_git_commit(),
                "config_hash": cfg.config_hash,
                "normal_false_reject_rate": round(normal_frr, 4),
                "configured_false_reject_rate": target_frr,
                "per_kind_metrics": per_kind_metrics,
                "recipe_hashes_verified": len(hashes_after),
            }
            out_json = cfg.results_dir / "phase_4_stage_a.json"
            out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

            elapsed = time.perf_counter() - t0
            logger.info(f"STEP 4 RESULT: PASS ({elapsed:.2f}s) -> {out_json}")
            log_decision(cfg, "Phase 4 Stage A Standalone Eval", "PASS", elapsed, f"FRR={normal_frr:.4f}")
            return 0
        except (RuntimeError, OSError, ValueError, AssertionError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"step4_stage_a_eval failed: {exc}")
            log_decision(cfg, "Phase 4 Stage A Standalone Eval", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
