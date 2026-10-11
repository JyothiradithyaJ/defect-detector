"""Synthetic unit tests for v3/eval/demo_single_image.py output schema and S0 RECAPTURE guard (no weights, no GPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from v3.core.bank import DINOFeatureExtractor, default_synthetic_feature_fn
from v3.eval.demo_single_image import run_single_image_demo


def _write_test_frozen_settings(tmp_path: Path) -> Path:
    frozen_path = tmp_path / "frozen_settings.json"
    payload = {
        "freeze_version": "v3-phase-5-gate",
        "scorer": "knn_global",
        "positional_radius": 2,
        "agg_radius": 1,
        "dino_backbone": "dinov2_vitl14_reg",
        "dino_short_side": 112,
        "dino_topk_frac": 0.05,
        "fusion_weights": {"w_d": 0.70, "w_c": 0.30},
        "calibration": {"status": "calibrated, not strictly zero-shot", "temperature": 1.1, "bias": -1.0},
        "tau_rule": {"false_reject_rate": 0.05, "min_mad": 1e-3},
    }
    raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    sha = hashlib.sha256(raw_bytes).hexdigest()
    frozen_path.write_bytes(raw_bytes)
    frozen_path.with_suffix(".sha256").write_text(f"{sha}  frozen_settings.json\n", encoding="utf-8")
    return frozen_path


def _write_test_recipe(tmp_path: Path) -> Path:
    recipe_path = tmp_path / "test_recipe.yaml"
    recipe_dict = {
        "status": "reviewed",
        "name": "synthetic_panel",
        "template_path": "",
        "resolution": [128, 128],
        "view": "fixed",
        "symmetry": "none",
        "stable_roi": [[8, 8], [120, 8], [120, 120], [8, 120]],
        "rois": [
            {
                "id": "wire_slot",
                "kind": "count",
                "polygon": [[16, 16], [96, 16], [96, 64], [16, 64]],
                "severity": 1,
                "hard": True,
                "params": {"expected_count": 2, "intensity_threshold": 170.0, "min_blob_pixels": 4},
                "z_threshold": 3.5,
            }
        ],
    }
    recipe_path.write_text(yaml.safe_dump(recipe_dict), encoding="utf-8")
    return recipe_path


def _make_normal_panel(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(70, 90, size=(128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([(28, 28), (38, 48)], fill=(230, 230, 230))
    draw.rectangle([(60, 28), (70, 48)], fill=(230, 230, 230))
    return img


def test_demo_single_image_schema_and_mismatch(tmp_path: Path) -> None:
    """Verify result.json schema, mismatch evidence fields, and overlay.png generation."""
    frozen_path = _write_test_frozen_settings(tmp_path)
    recipe_path = _write_test_recipe(tmp_path)
    normals = [_make_normal_panel(i) for i in range(4)]

    # Query with missing wire (only 1 wire instead of expected 2 -> hard count mismatch -> REJECT)
    query = _make_normal_panel(99)
    ImageDraw.Draw(query).rectangle([(55, 24), (75, 52)], fill=(80, 80, 80))
    q_path = tmp_path / "query_missing_wire.png"
    query.save(q_path)

    out_dir = tmp_path / "demo_out"
    ext = DINOFeatureExtractor(short_side=112, patch_size=14, embed_dim=64, feature_fn=default_synthetic_feature_fn)

    payload = run_single_image_demo(
        image_path=q_path,
        recipe_path=recipe_path,
        out_dir=out_dir,
        k=4,
        seed=0,
        frozen_json_path=frozen_path,
        dino_extractor=ext,
        reference_images=normals,
    )

    result_json = out_dir / "result.json"
    overlay_png = out_dir / "overlay.png"
    assert result_json.is_file()
    assert overlay_png.is_file()

    loaded = json.loads(result_json.read_text(encoding="utf-8"))
    required_keys = {
        "verdict",
        "mismatches",
        "image_score",
        "timing_seconds",
        "peak_vram_mb",
        "commit_hash",
        "seed",
        "config_hash",
        "frozen_settings_sha256",
    }
    assert required_keys.issubset(loaded.keys())
    assert loaded["verdict"] == "REJECT"
    assert len(loaded["mismatches"]) >= 1
    m0 = loaded["mismatches"][0]
    assert m0["roi_id"] == "wire_slot"
    assert m0["expected"] == 2
    assert m0["found"] == 1
    assert payload["verdict"] == "REJECT"


def test_demo_single_image_bad_photo_returns_recapture(tmp_path: Path) -> None:
    """Verify a black/underexposed photo fails S0 QC and returns RECAPTURE, never REJECT."""
    frozen_path = _write_test_frozen_settings(tmp_path)
    recipe_path = _write_test_recipe(tmp_path)
    normals = [_make_normal_panel(i) for i in range(4)]

    bad_photo = Image.new("RGB", (128, 128), color=(0, 0, 0))
    bad_path = tmp_path / "bad_photo.png"
    bad_photo.save(bad_path)

    out_dir = tmp_path / "demo_bad_out"
    ext = DINOFeatureExtractor(short_side=112, patch_size=14, embed_dim=64, feature_fn=default_synthetic_feature_fn)

    payload = run_single_image_demo(
        image_path=bad_path,
        recipe_path=recipe_path,
        out_dir=out_dir,
        k=4,
        seed=0,
        frozen_json_path=frozen_path,
        dino_extractor=ext,
        reference_images=normals,
    )
    assert payload["verdict"] == "RECAPTURE"
    assert payload["verdict"] != "REJECT"


def test_demo_refuses_tampered_frozen_settings(tmp_path: Path) -> None:
    """Verify demo_single_image refuses to run when frozen_settings.json does not match .sha256."""
    frozen_path = _write_test_frozen_settings(tmp_path)
    frozen_path.write_text('{"tampered": true}', encoding="utf-8")
    recipe_path = _write_test_recipe(tmp_path)
    q_path = tmp_path / "q.png"
    _make_normal_panel(0).save(q_path)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        run_single_image_demo(
            image_path=q_path,
            recipe_path=recipe_path,
            out_dir=tmp_path / "out",
            frozen_json_path=frozen_path,
        )

