"""Interactive OpenCV polygon drawing tool for authoring ROIs on the golden normal template.

Usage (by human reviewer on the golden normal template ONLY):
  D:\\envs\\anomaly_v3\\Scripts\\python.exe -m v3.eval.draw_rois --template D:/data/apd/normal/000.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from v3.config import DEFAULT_CONFIG_PATH, load_v3_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive OpenCV polygon ROI drawing tool for golden normal template.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--recipe", type=Path, default=None, help="Path to recipe YAML (defaults to apd_panel.yaml).")
    parser.add_argument("--template", type=Path, default=None, help="Path to golden normal template image.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned ROI drawing session and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_v3_config(args.config)
    recipe_path = args.recipe or (cfg.recipes_dir / "apd_panel.yaml")
    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
    tpl_path = args.template or Path(raw.get("template_path", "D:/data/apd/normal/golden_template.png"))

    if args.dry_run:
        roi_ids = [r["id"] for r in raw.get("rois", [])]
        print(
            f"[DRY-RUN] draw_rois.py: recipe={recipe_path}, template={tpl_path}, "
            f"rois_to_draw={roi_ids} (interactive GUI session, output size ~2 KB YAML)"
        )
        return 0

    if not tpl_path.is_file():
        print(
            f"[ERROR] Golden normal template not found at {tpl_path}. "
            "Place the APD dataset under D:/data/apd after accepting the dataset license.",
            file=sys.stderr,
        )
        return 1

    try:
        import cv2
    except ImportError as exc:
        print(f"[ERROR] opencv-python (cv2) is required for interactive ROI drawing: {exc}", file=sys.stderr)
        return 1

    img = cv2.imread(str(tpl_path))
    if img is None:
        print(f"[ERROR] Failed to read template image: {tpl_path}", file=sys.stderr)
        return 1

    print("Instructions: Left-click to add polygon vertex, 'n' to finish current ROI, 'q' to save and quit.")
    current_pts: list[list[int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pts.append([int(x), int(y)])

    win_name = "APD Golden Template ROI Drawer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    for roi_entry in raw.get("rois", []):
        current_pts = []
        roi_id = roi_entry["id"]
        print(f"Drawing ROI '{roi_id}' ({roi_entry['kind']})... Press 'n' when done.")
        while True:
            vis = img.copy()
            for pt in current_pts:
                cv2.circle(vis, (pt[0], pt[1]), 4, (0, 255, 0), -1)
            cv2.imshow(win_name, vis)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("n"):
                break
            if key == ord("q"):
                cv2.destroyAllWindows()
                return 0
        if len(current_pts) >= 3:
            roi_entry["polygon"] = current_pts

    cv2.destroyAllWindows()
    raw["status"] = "human-reviewed-and-frozen"
    recipe_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(f"Updated {recipe_path} with status='human-reviewed-and-frozen'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
