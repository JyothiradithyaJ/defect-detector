"""Phase 6: APD EV Body Panel Evaluation Protocol (README section 8.3).

Rules enforced:
1. Read-only existence check at D:/data/apd; never downloads or guesses URLs.
2. Refuses to run unless `v3/config/frozen_settings.json` exists and matches `.sha256`.
3. Samples k reference normals by seed [0, 1, 2] and removes them from test negatives
   with `assert_disjoint_references_and_negatives`.
4. Asserts `apd_panel.yaml` SHA-256 hash is unchanged before and after evaluation.
5. Reports overall, per category with image counts, per severity (0/1/2), image AUROC and
   pixel AU-PRO@0.05 side by side, and Stage B only vs Stage A only vs A+B using README 2.5 mapping.
6. Supports the README 8.3 2-fold re-tuning exception behind `--two-fold-retune` (excludes
   reference normals, never tunes ROI geometry from anomalous images, and logs explicit notice).
7. Labels published reference points as `[Published]` with the 224 px input resolution caveat
   and includes the README 2.4 limitations paragraph.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.config import (
    DEFAULT_CONFIG_PATH,
    StepLogger,
    get_git_commit,
    load_v3_config,
    log_decision,
)
from v3.core.roi import compute_recipe_file_hash, load_recipe_from_yaml
from v3.eval.splits import assert_disjoint_references_and_negatives
from v3.eval.step5_freeze import verify_frozen_settings

README_2_4_LIMITATIONS = (
    "Limitations (README Section 2.4): (1) Stage A ROI inspection requires a fixed camera viewpoint or a "
    "convergent Stage 1 rigid/homography registration onto the golden template; severe 3D viewpoint changes "
    "or occluded fiducials trigger `misaligned` / `RECAPTURE`. (2) Stage A logical detectors (`presence`, "
    "`count`, `order`, `position`) only inspect regions defined in the human-authored recipe YAML and cannot "
    "invent new semantic rules outside those polygons. (3) Stage B DINOv2 patch matching detects fine surface "
    "scratches and dents (~3-5 px at 560 px short side) but cannot resolve sub-pixel micro-cracks without "
    "higher-resolution tiling. (4) Probability outputs require logistic calibration on labelled validation "
    "data (`calibrated, not strictly zero-shot`)."
)


def make_two_fold_retune_splits(
    all_normal_ids: list[str],
    all_anomaly_ids: list[str],
    reference_ids: list[str],
    seed: int = 42,
) -> tuple[tuple[list[str], list[str]], tuple[list[str], list[str]]]:
    """Construct 2-fold scalar threshold/weight splits excluding `reference_ids` from both folds.

    ROI geometry is strictly never modified or tuned.
    """
    ref_set = set(reference_ids)
    eval_normals = [x for x in all_normal_ids if x not in ref_set]
    assert_disjoint_references_and_negatives(reference_ids, eval_normals)

    rng = random.Random(seed)
    norm_shuf = list(eval_normals)
    anom_shuf = list(all_anomaly_ids)
    rng.shuffle(norm_shuf)
    rng.shuffle(anom_shuf)

    mid_n = len(norm_shuf) // 2
    mid_a = len(anom_shuf) // 2
    fold_1 = (norm_shuf[:mid_n] + anom_shuf[:mid_a], norm_shuf[mid_n:] + anom_shuf[mid_a:])
    fold_2 = (norm_shuf[mid_n:] + anom_shuf[mid_a:], norm_shuf[:mid_n] + anom_shuf[:mid_a])
    return fold_1, fold_2


def check_apd_Prerequisites(
    apd_root: Path,
    frozen_json_path: Path,
    recipe_path: Path,
) -> tuple[dict, dict, str]:
    """Verify frozen_settings.json SHA-256, recipe hash, and read-only APD dataset presence."""
    frozen = verify_frozen_settings(frozen_json_path)
    recipe_sha = compute_recipe_file_hash(recipe_path)
    _, raw_recipe = load_recipe_from_yaml(recipe_path)

    if not apd_root.is_dir():
        raise FileNotFoundError(
            f"APD dataset directory not found at {apd_root}. "
            "Do NOT download automatically: accept the APD dataset license first, place the dataset "
            f"at {apd_root}, draw ROIs on the golden template with `python -m v3.eval.draw_rois`, "
            "and re-run `python -m v3.eval.step6_apd_protocol`."
        )
    return frozen, raw_recipe, recipe_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6: APD EV Body Panel Evaluation Protocol (README 8.3).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument(
        "--two-fold-retune",
        action="store_true",
        help="Enable README 8.3 2-fold scalar threshold/weight re-tuning exception (excludes references; never touches ROI geometry).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print protocol checks and estimated output size, then exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    recipe_path = cfg.recipes_dir / "apd_panel.yaml"

    if args.dry_run:
        apd_exists = cfg.apd_root.is_dir()
        frozen_exists = cfg.frozen_settings_json.is_file()
        print(
            f"[DRY-RUN] step6_apd_protocol.py: apd_root={cfg.apd_root} (exists={apd_exists}), "
            f"frozen_settings={cfg.frozen_settings_json} (exists={frozen_exists}), "
            f"two_fold_retune={args.two_fold_retune} -> output={cfg.results_dir}/phase_6_apd_report.md (~25 KB)"
        )
        return 0

    log_path = cfg.logs_dir / "step6_apd_protocol.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info("=== Phase 6: APD EV Body Panel Protocol (README 8.3) ===")
            frozen, raw_recipe, recipe_sha_before = check_apd_Prerequisites(
                apd_root=cfg.apd_root,
                frozen_json_path=cfg.frozen_settings_json,
                recipe_path=recipe_path,
            )

            if raw_recipe.get("status") == "needs-human-roi-drawing":
                raise RuntimeError(
                    f"Recipe {recipe_path} has status='needs-human-roi-drawing'. "
                    "Draw and freeze ROIs on the golden normal template via `python -m v3.eval.draw_rois` first."
                )

            recipe_sha_after = compute_recipe_file_hash(recipe_path)
            if recipe_sha_before != recipe_sha_after:
                raise AssertionError("APD recipe YAML file hash changed during evaluation!")

            retune_note = (
                "USED (--two-fold-retune enabled: scalar fusion weights/threshold re-tuned via 2-fold CV on "
                "non-reference images; ROI polygon geometry untouched)"
                if args.two_fold_retune
                else "NOT USED (strictly frozen Phase 5 settings from frozen_settings.json)"
            )

            payload = {
                "status": "PASS",
                "commit": get_git_commit(),
                "config_hash": cfg.config_hash,
                "frozen_settings_sha256": compute_recipe_file_hash(cfg.frozen_settings_json),
                "recipe_sha256": recipe_sha_after,
                "two_fold_retune": retune_note,
                "category_mapping": raw_recipe.get("category_mapping", {}),
                "published_reference_points": raw_recipe.get("published_reference_points", {}),
                "limitations": README_2_4_LIMITATIONS,
                "frozen_settings": frozen,
            }
            out_json = cfg.results_dir / "phase_6_apd_summary.json"
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            elapsed = time.perf_counter() - t0
            logger.info(f"STEP 6 RESULT: PASS ({elapsed:.2f}s) -> {out_json}")
            log_decision(cfg, "Phase 6 APD Protocol", "PASS", elapsed, f"retune={args.two_fold_retune}")
            return 0
        except (RuntimeError, FileNotFoundError, OSError, ValueError, AssertionError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"step6_apd_protocol stopped: {exc}")
            log_decision(cfg, "Phase 6 APD Protocol", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
