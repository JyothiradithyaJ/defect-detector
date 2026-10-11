"""Stage 5: Structured JSON + overlay report writer and Result contract validator.

Stamps commit hash, seed, and config hash onto every JSON and PNG output (Hard Rule 8).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from v3.core.contracts import Result

ALLOWED_VERDICTS = {"PASS", "REWORK", "REJECT", "RECAPTURE"}


def validate_result_contract(result: Result) -> None:
    """Validate that `result` strictly conforms to the v3.core.contracts.Result specification."""
    if not isinstance(result, Result):
        raise TypeError(f"Expected Result instance, got {type(result)}")
    if result.verdict not in ALLOWED_VERDICTS:
        raise ValueError(f"Invalid verdict '{result.verdict}'. Allowed: {sorted(ALLOWED_VERDICTS)}")
    if not isinstance(result.mismatches, list):
        raise TypeError("result.mismatches must be a list")
    if not isinstance(result.image_score, (float, int, np.floating)):
        raise TypeError(f"result.image_score must be float, got {type(result.image_score)}")
    if not isinstance(result.anomaly_map, np.ndarray) or result.anomaly_map.ndim != 2:
        raise ValueError("result.anomaly_map must be a 2D numpy.ndarray")
    if not isinstance(result.meta, dict):
        raise TypeError("result.meta must be a dict")


def render_inspection_overlay(
    image: Image.Image,
    result: Result,
    commit_hash: str,
    seed: int,
    config_hash: str,
) -> Image.Image:
    """Render anomaly heatmap blend, mismatch annotations, verdict banner, and reproducibility stamp."""
    validate_result_contract(result)
    base = image.convert("RGB")
    w, h = base.size
    amap = result.anomaly_map

    # Normalize anomaly map to [0, 255] red-channel alpha overlay
    amin, amax = float(np.min(amap)), float(np.max(amap))
    norm_map = (amap - amin) / max(amax - amin, 1e-6) if amax > amin else np.zeros_like(amap, dtype=np.float32)
    map_pil = Image.fromarray(np.clip(norm_map * 255.0, 0, 255).astype(np.uint8), mode="L").resize(
        (w, h),
        Image.Resampling.BILINEAR,
    )
    heat_arr = np.asarray(map_pil, dtype=np.float32) / 255.0
    base_arr = np.asarray(base, dtype=np.float32)
    red_layer = np.zeros_like(base_arr)
    red_layer[..., 0] = 255.0
    alpha = (0.35 * heat_arr)[..., None]
    blended = np.clip((1.0 - alpha) * base_arr + alpha * red_layer, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(blended, mode="RGB")

    draw = ImageDraw.Draw(canvas)
    banner_str = (
        f"VERDICT: {result.verdict} | score={result.image_score:.4f} | "
        f"mismatches={len(result.mismatches)} | commit={commit_hash} | seed={seed} | cfg={config_hash}"
    )
    draw.rectangle([(0, 0), (w, 24)], fill=(15, 15, 15))
    draw.text((6, 5), banner_str, fill=(255, 255, 255))
    return canvas


def write_inspection_report(
    result: Result,
    image: Image.Image,
    out_dir: Path,
    sample_id: str,
    commit_hash: str,
    seed: int,
    config_hash: str,
) -> tuple[Path, Path]:
    """Validate `result`, stamp metadata, and write `<sample_id>.json` and `<sample_id>_overlay.png`."""
    validate_result_contract(result)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.meta.update(
        {
            "commit_hash": commit_hash,
            "seed": seed,
            "config_hash": config_hash,
            "calibration_note": "calibrated on validation split (not strictly zero-shot)",
        }
    )

    json_payload = {
        "sample_id": sample_id,
        "verdict": result.verdict,
        "image_score": float(result.image_score),
        "mismatches": [asdict(m) for m in result.mismatches],
        "anomaly_map_shape": list(result.anomaly_map.shape),
        "anomaly_map_max": float(np.max(result.anomaly_map)) if result.anomaly_map.size > 0 else 0.0,
        "meta": result.meta,
    }
    json_path = out_dir / f"{sample_id}.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    overlay_img = render_inspection_overlay(
        image=image,
        result=result,
        commit_hash=commit_hash,
        seed=seed,
        config_hash=config_hash,
    )
    png_path = out_dir / f"{sample_id}_overlay.png"
    overlay_img.save(png_path)
    return json_path, png_path
