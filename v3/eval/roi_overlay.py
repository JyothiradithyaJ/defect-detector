"""Render recipe ROI polygons onto normal reference images as PNG overlays for human review.

Automatically stamps commit hash, seed, and config hash onto every generated PNG (Hard Rule 8).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

from v3.config import DEFAULT_CONFIG_PATH, get_git_commit, load_v3_config
from v3.core.roi import load_recipe_from_yaml

KIND_COLORS: dict[str, tuple[int, int, int]] = {
    "presence": (46, 204, 113),
    "count": (52, 152, 219),
    "order": (241, 196, 15),
    "position": (155, 89, 182),
    "surface": (231, 76, 60),
}


def render_recipe_overlay(
    image: Image.Image,
    recipe_path: Path,
    commit_hash: str,
    config_hash: str,
    seed: int = 0,
) -> Image.Image:
    """Draw stable_roi and all ROI polygons with labels and reproducibility stamp."""
    recipe, raw = load_recipe_from_yaml(recipe_path)
    canvas = image.convert("RGB").resize(recipe.resolution, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(canvas)

    if len(recipe.stable_roi) >= 3:
        pts = [*recipe.stable_roi, recipe.stable_roi[0]]
        draw.line(pts, fill=(200, 200, 200), width=2)
        draw.text((recipe.stable_roi[0][0] + 4, recipe.stable_roi[0][1] + 4), "stable_roi", fill=(220, 220, 220))

    for roi in recipe.rois:
        if len(roi.polygon) >= 3:
            color = KIND_COLORS.get(roi.kind, (255, 255, 255))
            pts = [*roi.polygon, roi.polygon[0]]
            draw.line(pts, fill=color, width=3)
            label = f"{roi.id} [{roi.kind}, sev={roi.severity}, hard={roi.hard}]"
            draw.text((roi.polygon[0][0] + 6, roi.polygon[0][1] + 6), label, fill=color)

    status = raw.get("status", "unknown")
    stamp = f"recipe={recipe.name} | status={status} | commit={commit_hash} | seed={seed} | cfg={config_hash}"
    draw.rectangle([(0, 0), (canvas.width, 24)], fill=(20, 20, 20))
    draw.text((8, 5), stamp, fill=(255, 255, 255))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render recipe ROI overlays to PNG for human review.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--recipe", type=Path, default=None, help="Specific recipe YAML (defaults to all LOCO recipes).")
    parser.add_argument("--dry-run", action="store_true", help="Print planned overlay outputs and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_v3_config(args.config)
    recipe_files = [args.recipe] if args.recipe else sorted(cfg.recipes_dir.glob("*.yaml"))

    out_dir = cfg.results_dir / "roi_overlays"
    if args.dry_run:
        print(
            f"[DRY-RUN] roi_overlay.py: recipes={[p.name for p in recipe_files]} -> "
            f"output_dir={out_dir} (estimated size ~{len(recipe_files) * 0.25:.2f} MB)"
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    commit = get_git_commit()

    for r_path in recipe_files:
        recipe, _ = load_recipe_from_yaml(r_path)
        tpl_path = Path(recipe.template_path)
        if tpl_path.is_file():
            with Image.open(tpl_path) as img:
                base_img = img.convert("RGB")
        else:
            base_img = Image.new("RGB", recipe.resolution, color=(48, 52, 60))

        overlay = render_recipe_overlay(base_img, r_path, commit_hash=commit, config_hash=cfg.config_hash, seed=0)
        out_png = out_dir / f"{recipe.name}_rois.png"
        overlay.save(out_png)
        print(f"Saved ROI overlay: {out_png}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
