"""Single-image end-to-end inspection demo (S0 QC -> S1 Align -> S2 Stage A ROIs -> S3 Stage B -> S4 Verdict -> S5 Report).

Enforces:
- Refuses to run unless `v3/config/frozen_settings.json` exists and matches `v3/config/frozen_settings.sha256`.
- Runs S0 through S5 on ONE query image using `k` reference normal images (skip-if-exists feature caching).
- A bad photo failing S0 Quality Check returns `RECAPTURE`, never `REJECT`.
- Writes `result.json` and `overlay.png` (with ROI polygon boxes and blended anomaly heatmap) to `--out-dir`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from v3.config import DEFAULT_CONFIG_PATH, get_git_commit, load_v3_config
from v3.core.bank import DINOFeatureExtractor
from v3.core.contracts import Recipe, Result
from v3.core.pipeline import TwoStagePipeline
from v3.core.report import render_inspection_overlay, validate_result_contract
from v3.core.roi import load_recipe_from_yaml
from v3.eval.roi_overlay import KIND_COLORS
from v3.eval.step5_freeze import verify_frozen_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S0-S5 v3 inspection on a single image with frozen settings.")
    parser.add_argument("--image", type=Path, required=True, help="Path to single query image.")
    parser.add_argument("--recipe", type=Path, required=True, help="Path to recipe YAML.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for result.json and overlay.png.")
    parser.add_argument("--k", type=int, default=4, help="Number of normal reference images (default: 4).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reference normal selection.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Verify frozen settings and arguments without running.")
    return parser.parse_args(argv)


def _discover_reference_normals(recipe: Recipe, image_path: Path, data_root: Path, k: int, seed: int) -> list[Path]:
    """Locate up to `k` normal reference images disjoint from `image_path`."""
    candidates: list[Path] = []
    tpl = Path(recipe.template_path) if recipe.template_path else None
    if tpl and tpl.is_file():
        candidates.extend(sorted(tpl.parent.glob("*.png")) + sorted(tpl.parent.glob("*.jpg")))

    if not candidates:
        cat_train_good = data_root / recipe.name / "train" / "good"
        if cat_train_good.is_dir():
            candidates.extend(sorted(cat_train_good.glob("*.png")) + sorted(cat_train_good.glob("*.jpg")))

    if not candidates:
        # Fallback to parent/sibling train/good if image is inside an MVTec-style directory
        for parent in image_path.parents:
            train_good = parent / "train" / "good"
            if train_good.is_dir():
                candidates.extend(sorted(train_good.glob("*.png")) + sorted(train_good.glob("*.jpg")))
                break

    q_resolved = image_path.resolve()
    unique_refs = [p for p in candidates if p.is_file() and p.resolve() != q_resolved]
    if not unique_refs:
        return []
    rng = random.Random(seed)
    return rng.sample(unique_refs, min(k, len(unique_refs)))


def _format_mismatch_entry(m: Any, recipe: Recipe, query_img: Image.Image) -> dict[str, Any]:
    from v3.core.mismatch.count import estimate_blob_count_and_area

    raw = asdict(m)
    ev = dict(raw.get("evidence", {}) or {})
    roi_map = {r.id: r for r in recipe.rois}
    roi = roi_map.get(raw["roi_id"])

    if raw["kind"] == "count" and roi is not None:
        blob_count, blob_area = estimate_blob_count_and_area(query_img, roi)
        expected: Any = int(roi.params.get("expected_count", 0))
        found: Any = int(blob_count)
        ev["expected_count"] = expected
        ev["found_count"] = found
        ev["found_area_ratio"] = round(float(blob_area), 5)
    elif raw["kind"] == "order" and roi is not None:
        expected = roi.params.get("expected_sequence", f"z <= {ev.get('z_threshold', 3.5)}")
        found = f"z = {ev.get('z_score', 0.0)} (raw={ev.get('raw_score', 0.0)})"
    else:
        expected = f"z <= {ev.get('z_threshold', 3.5)}"
        found = f"z = {ev.get('z_score', 0.0)} (raw={ev.get('raw_score', 0.0)})"
    return {
        "roi_id": raw["roi_id"],
        "kind": raw["kind"],
        "severity": int(raw["severity"]),
        "hard": bool(raw["hard"]),
        "confidence": float(raw["confidence"]),
        "expected": expected,
        "found": found,
        "evidence": ev,
    }


def _render_overlay_with_rois(
    image: Image.Image,
    recipe: Recipe,
    result: Result,
    commit_hash: str,
    seed: int,
    config_hash: str,
) -> Image.Image:
    canvas = render_inspection_overlay(
        image=image.convert("RGB").resize(recipe.resolution, Image.Resampling.BILINEAR),
        result=result,
        commit_hash=commit_hash,
        seed=seed,
        config_hash=config_hash,
    )
    draw = ImageDraw.Draw(canvas)
    mismatched_ids = {m.roi_id for m in result.mismatches}

    if len(recipe.stable_roi) >= 3:
        pts = [*recipe.stable_roi, recipe.stable_roi[0]]
        draw.line(pts, fill=(200, 200, 200), width=2)

    for roi in recipe.rois:
        if len(roi.polygon) >= 3:
            is_bad = roi.id in mismatched_ids
            color = (255, 40, 40) if is_bad else KIND_COLORS.get(roi.kind, (46, 204, 113))
            pts = [*roi.polygon, roi.polygon[0]]
            draw.line(pts, fill=color, width=3 if is_bad else 2)
            tag = f"{roi.id}:{roi.kind}[{'FAIL' if is_bad else 'OK'}]"
            draw.text((roi.polygon[0][0] + 4, max(26, roi.polygon[0][1] + 4)), tag, fill=color)

    return canvas


def run_single_image_demo(
    image_path: Path,
    recipe_path: Path,
    out_dir: Path,
    k: int = 4,
    seed: int = 0,
    config_path: Path = DEFAULT_CONFIG_PATH,
    frozen_json_path: Path | None = None,
    dino_extractor: DINOFeatureExtractor | None = None,
    reference_images: list[Image.Image] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run S0-S5 on one image and write `result.json` and `overlay.png` into `out_dir`."""
    t0 = time.perf_counter()
    cfg = load_v3_config(config_path)
    f_path = frozen_json_path or cfg.frozen_settings_json
    frozen = verify_frozen_settings(f_path)
    frozen_sha = hashlib.sha256(f_path.read_bytes()).hexdigest()
    recipe, _ = load_recipe_from_yaml(recipe_path)

    if dry_run:
        msg = (
            f"[DRY-RUN] demo_single_image.py: image={image_path}, recipe={recipe_path} ({recipe.name}), "
            f"k={k}, frozen_sha256={frozen_sha[:12]} -> {out_dir}/result.json, {out_dir}/overlay.png"
        )
        print(msg, flush=True)
        return {"status": "DRY_RUN", "frozen_settings_sha256": frozen_sha}

    if not image_path.is_file():
        raise FileNotFoundError(f"Query image not found: {image_path}")

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dino_cfg = cfg.raw.get("dino", {})
    ext = dino_extractor or DINOFeatureExtractor(
        backbone_id=str(frozen.get("dino_backbone", dino_cfg.get("model", "dinov2_vitl14_reg"))),
        short_side=int(frozen.get("dino_short_side", dino_cfg.get("short_side", 560))),
        patch_size=int(dino_cfg.get("patch_size", 14)),
        embed_dim=int(dino_cfg.get("embed_dim", 1024)),
        precision=str(cfg.raw.get("precision", "fp16")),
        cache_dir=cfg.dino_cache_dir,
        torch_home=cfg.torch_home,
        lazy_load=True,
    )

    if reference_images is not None:
        normals = [im.convert("RGB").resize(recipe.resolution, Image.Resampling.BILINEAR) for im in reference_images[:k]]
    else:
        ref_paths = _discover_reference_normals(recipe, image_path, cfg.data_root, k=k, seed=seed)
        if not ref_paths:
            raise RuntimeError(f"No normal reference images found for recipe '{recipe.name}'.")
        normals = []
        for rp in ref_paths:
            with Image.open(rp) as rim:
                normals.append(rim.convert("RGB").resize(recipe.resolution, Image.Resampling.BILINEAR))

    fw = frozen.get("fusion_weights", {})
    tau_rule = frozen.get("tau_rule", {})
    cal_cfg = frozen.get("calibration", {})

    pipe = TwoStagePipeline(
        dino_extractor=ext,
        scorer_mode=str(frozen.get("scorer", "knn_global")),
        pos_radius=int(frozen.get("positional_radius", 2)),
        agg_radius=int(frozen.get("agg_radius", 1)),
        topk_frac=float(frozen.get("dino_topk_frac", 0.001)),
        w_d=float(fw.get("w_d", 0.70)),
        w_c=float(fw.get("w_c", 0.30)),
        false_reject_rate=float(tau_rule.get("false_reject_rate", 0.05)),
        min_mad=float(tau_rule.get("min_mad", 1e-3)),
    )
    pipe.calibrator.temperature = float(cal_cfg.get("temperature", 1.0))
    pipe.calibrator.bias = float(cal_cfg.get("bias", 0.0))
    pipe.calibrator.is_calibrated = True

    pipe.fit(normals, recipe)

    with Image.open(image_path) as q_raw:
        query_img = q_raw.convert("RGB").resize(recipe.resolution, Image.Resampling.BILINEAR)

    result = pipe.score(query_img)
    # Hard safety invariant: a bad photo failing S0 QC must return RECAPTURE, never REJECT
    if not result.meta.get("qc", {}).get("passed", True) or "qc_reasons" in result.meta:
        result = Result(
            verdict="RECAPTURE",
            mismatches=[],
            image_score=0.0,
            anomaly_map=result.anomaly_map,
            meta=result.meta,
        )
    validate_result_contract(result)

    elapsed = round(time.perf_counter() - t0, 4)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0.0
    commit_hash = get_git_commit()

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": result.verdict,
        "mismatches": [_format_mismatch_entry(m, recipe, query_img) for m in result.mismatches],
        "image_score": float(result.image_score),
        "timing_seconds": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "commit_hash": commit_hash,
        "seed": int(seed),
        "config_hash": cfg.config_hash,
        "frozen_settings_sha256": frozen_sha,
        "recipe": recipe.name,
        "image_path": str(image_path),
        "k": len(normals),
        "meta": result.meta,
    }

    result_json_path = out_dir / "result.json"
    result_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    overlay_img = _render_overlay_with_rois(
        image=query_img,
        recipe=recipe,
        result=result,
        commit_hash=commit_hash,
        seed=seed,
        config_hash=cfg.config_hash,
    )
    overlay_png_path = out_dir / "overlay.png"
    overlay_img.save(overlay_png_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run_single_image_demo(
            image_path=args.image,
            recipe_path=args.recipe,
            out_dir=args.out_dir,
            k=args.k,
            seed=args.seed,
            config_path=args.config,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
