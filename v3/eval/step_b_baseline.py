"""Step B: Baseline 6-decimal reproduction check for clip_b32_224 (fp32).

Compares clip_b32_224 (fp32) across the 10 held-out MVTec categories (full test sets)
against the official Phase 1 baseline numbers to 6 decimal places.
Prints PASS/FAIL and per-category differences.
Writes log to D:/defect_cache/logs/step_b.log and appends status to v3/docs/decisions.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from v3.config import DEFAULT_CONFIG_PATH, StepLogger, load_v3_config, log_decision
from v3.eval.harness import V1BaselineScorer, evaluate_category_samples
from v3.eval.splits import create_v3_category_splits

# Official Phase 1 reference numbers (10 held-out categories, full test sets, fp32)
PHASE1_EXPECTED = {
    "macro": {
        "macro_image_auroc": 0.790817,
        "macro_pixel_auroc": 0.680949,
        "macro_pixel_aupro_005": 0.081959,
        "macro_pixel_aupro_030": 0.304970,
        "macro_recall_at_5_frr": 0.562958,
        "macro_ece": 0.256343,
    },
    "categories": {
        "bottle": {"image_auroc": 0.805556, "pixel_auroc": 0.625113},
        "cable": {"image_auroc": 0.713163, "pixel_auroc": 0.382461},
        "capsule": {"image_auroc": 0.752692, "pixel_auroc": 0.779333},
        "grid": {"image_auroc": 0.727651, "pixel_auroc": 0.665643},
        "leather": {"image_auroc": 0.999321, "pixel_auroc": 0.903283},
        "pill": {"image_auroc": 0.733223, "pixel_auroc": 0.692978},
        "screw": {"image_auroc": 0.659472, "pixel_auroc": 0.870615},
        "tile": {"image_auroc": 0.968254, "pixel_auroc": 0.742942},
        "toothbrush": {"image_auroc": 0.761111, "pixel_auroc": 0.371158},
        "zipper": {"image_auroc": 0.787723, "pixel_auroc": 0.775965},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step B: Baseline 6-decimal reproduction check.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-run full fp32 evaluation from images instead of using existing v3-category_results.json if present.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Execution device if recomputing.")
    return parser.parse_args()


def run_fresh_baseline(cfg_data_root: Path, cfg_manifest_dir: Path, device: str, logger: StepLogger) -> dict:
    """Run V1BaselineScorer (clip_b32_224, fp32) across the 10 held-out categories."""
    logger.info(f"Running fresh fp32 baseline evaluation on device={device} from {cfg_data_root}...")
    _, eval_cats, val_samples, eval_samples = create_v3_category_splits(
        data_root=cfg_data_root,
        manifest_dir=cfg_manifest_dir,
        seed=42,
    )
    scorer = V1BaselineScorer(device=device)
    temp, bias = scorer.fit_calibration_on_samples(val_samples)
    logger.info(f"Fitted calibration on 5 validation categories: T={temp:.6f}, bias={bias:.6f}")

    cat_results = []
    for cat in eval_cats:
        c_samples = [s for s in eval_samples if s.category == cat]
        res = evaluate_category_samples(cat, c_samples, scorer, seed=0)
        cat_results.append(asdict(res))
        logger.info(
            f"  {cat:<12} | N={res.images:<3} | Img AUROC={res.image_auroc:.6f} | Px AUROC={res.pixel_auroc:.6f}"
        )

    n = len(cat_results)
    return {
        "macro_image_auroc": sum(r["image_auroc"] for r in cat_results) / n,
        "macro_pixel_auroc": sum(r["pixel_auroc"] for r in cat_results) / n,
        "macro_pixel_aupro_005": sum(r["pixel_aupro_005"] for r in cat_results) / n,
        "macro_pixel_aupro_030": sum(r["pixel_aupro_030"] for r in cat_results) / n,
        "macro_recall_at_5_frr": sum(r["recall_at_5_frr"] for r in cat_results) / n,
        "macro_ece": sum(r["expected_calibration_error"] for r in cat_results) / n,
        "categories": cat_results,
    }


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    log_path = cfg.logs_dir / "step_b.log"

    with StepLogger(log_path) as logger:
        try:
            logger.info("=== STEP B: Baseline Reproduction Check (clip_b32_224, fp32, 6 decimals) ===")
            if not args.recompute and cfg.phase1_baseline_json.is_file():
                logger.info(f"Loading completed baseline run from: {cfg.phase1_baseline_json}")
                actual_data = json.loads(cfg.phase1_baseline_json.read_text(encoding="utf-8"))
            else:
                actual_data = run_fresh_baseline(cfg.data_root, cfg.manifest_dir, args.device, logger)

            actual_cats = {row["category"]: row for row in actual_data["categories"]}
            max_diff = 0.0
            all_passed = True

            logger.info("\nPer-Category Comparison (6 Decimals):")
            logger.info(f"{'Category':<12} | {'Img AUROC (Act/Exp/Diff)':<32} | {'Px AUROC (Act/Exp/Diff)':<32}")
            logger.info("-" * 82)

            cat_diffs = {}
            for cat, exp_vals in PHASE1_EXPECTED["categories"].items():
                if cat not in actual_cats:
                    raise RuntimeError(f"Missing held-out category '{cat}' in baseline results.")
                act_row = actual_cats[cat]
                img_diff = abs(round(act_row["image_auroc"], 6) - exp_vals["image_auroc"])
                px_diff = abs(round(act_row["pixel_auroc"], 6) - exp_vals["pixel_auroc"])
                max_diff = max(max_diff, img_diff, px_diff)
                if img_diff > 1e-6 or px_diff > 1e-6:
                    all_passed = False
                cat_diffs[cat] = {"image_auroc_diff": round(img_diff, 6), "pixel_auroc_diff": round(px_diff, 6)}
                logger.info(
                    f"{cat:<12} | {act_row['image_auroc']:.6f} / {exp_vals['image_auroc']:.6f} ({img_diff:+.6f}) | "
                    f"{act_row['pixel_auroc']:.6f} / {exp_vals['pixel_auroc']:.6f} ({px_diff:+.6f})"
                )

            logger.info("\nMacro Metric Comparison (6 Decimals):")
            macro_diffs = {}
            for key, exp_val in PHASE1_EXPECTED["macro"].items():
                act_val = float(actual_data[key])
                diff = abs(round(act_val, 6) - exp_val)
                max_diff = max(max_diff, diff)
                if diff > 1e-6:
                    all_passed = False
                macro_diffs[key] = round(diff, 6)
                logger.info(f"  {key:<24}: Actual={act_val:.6f} | Expected={exp_val:.6f} | Diff={diff:+.6f}")

            status = "PASS" if all_passed else "FAIL"
            elapsed = time.perf_counter() - t0
            logger.info(f"\nSTEP B RESULT: {status} (max_diff={max_diff:.6f}, elapsed={elapsed:.2f}s)")

            out_json = cfg.results_dir / "step_b_baseline_check.json"
            out_json.write_text(
                json.dumps(
                    {
                        "status": status,
                        "max_abs_diff": round(max_diff, 6),
                        "macro_diffs": macro_diffs,
                        "category_diffs": cat_diffs,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            log_decision(
                cfg,
                "Step B (Baseline 6-Decimal Check)",
                status,
                elapsed,
                f"max_diff={max_diff:.6f}",
            )
            if not all_passed:
                raise RuntimeError(f"Baseline check FAILED with max_diff={max_diff:.6f} > 1e-6")
            return 0
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"Step B failed: {exc}")
            log_decision(cfg, "Step B (Baseline 6-Decimal Check)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())

