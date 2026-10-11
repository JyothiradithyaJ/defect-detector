"""Read-only ROI coverage diagnostic checking overlap between frozen ROI polygons and defect masks.

IMPORTANT: Run ONLY AFTER the human reviewer has drawn and frozen the ROI polygons in
`apd_panel.yaml` (`status: human-reviewed-and-frozen`). Never modifies any recipe file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from v3.config import DEFAULT_CONFIG_PATH, load_v3_config
from v3.core.roi import compute_recipe_file_hash, load_recipe_from_yaml


def compute_roi_mask_coverage(mask_bool: np.ndarray, roi_polygons: list[list[tuple[int, int]]]) -> float:
    """Compute fraction of defect pixels in `mask_bool` covered by the union of `roi_polygons`."""
    total_defect = int(np.sum(mask_bool))
    if total_defect == 0:
        return 1.0
    h, w = mask_bool.shape
    union_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(union_img)
    for poly in roi_polygons:
        if len(poly) >= 3:
            draw.polygon([(int(x), int(y)) for x, y in poly], outline=1, fill=1)
    union_mask = np.asarray(union_img, dtype=np.uint8) > 0
    covered = int(np.sum(mask_bool & union_mask))
    return float(covered / total_defect)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only ROI vs ground-truth defect mask coverage check.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--recipe", type=Path, default=None, help="Path to frozen recipe YAML.")
    parser.add_argument("--masks-dir", type=Path, default=None, help="Directory of ground-truth masks.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned coverage check and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_v3_config(args.config)
    recipe_path = args.recipe or (cfg.recipes_dir / "apd_panel.yaml")
    masks_dir = args.masks_dir or (cfg.apd_root / "ground_truth")

    if args.dry_run:
        print(
            f"[DRY-RUN] roi_coverage_check.py: read-only check of {recipe_path} against masks in {masks_dir} -> "
            f"output={cfg.results_dir}/roi_coverage.json (~4 KB)"
        )
        return 0

    hash_before = compute_recipe_file_hash(recipe_path)
    recipe, raw = load_recipe_from_yaml(recipe_path)
    if raw.get("status") == "needs-human-roi-drawing":
        print(
            f"[ERROR] Recipe {recipe_path} still has status='needs-human-roi-drawing'. "
            "Freeze ROIs on the golden template first before running roi_coverage_check.py.",
            file=sys.stderr,
        )
        return 1

    if not masks_dir.is_dir():
        print(f"[ERROR] Ground-truth mask directory not found: {masks_dir}", file=sys.stderr)
        return 1

    polys = [r.polygon for r in recipe.rois if len(r.polygon) >= 3]
    mask_files = sorted(p for p in masks_dir.glob("*.png"))
    coverages: list[float] = []
    for mf in mask_files:
        with Image.open(mf) as m_img:
            m_arr = np.asarray(m_img.convert("L").resize(recipe.resolution, Image.Resampling.NEAREST), dtype=np.uint8) > 0
        coverages.append(compute_roi_mask_coverage(m_arr, polys))

    hash_after = compute_recipe_file_hash(recipe_path)
    if hash_before != hash_after:
        raise AssertionError("Recipe YAML file hash changed during read-only coverage check!")

    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    out_json = cfg.results_dir / "roi_coverage.json"
    out_json.write_text(
        json.dumps(
            {
                "recipe": recipe.name,
                "num_masks": len(mask_files),
                "mean_coverage": round(mean_cov, 4),
                "recipe_sha256": hash_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"ROI Coverage Check complete: mean_coverage={mean_cov:.4f} across {len(mask_files)} masks -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
