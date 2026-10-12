"""Synthetic unit tests for Phase 6 APD EV track protocol, freeze verification, and 2-fold split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from v3.core.roi import compute_recipe_file_hash
from v3.eval.roi_coverage_check import compute_roi_mask_coverage
from v3.eval.step5_freeze import verify_frozen_settings
from v3.eval.step6_apd_protocol import (
    README_2_4_LIMITATIONS,
    make_two_fold_retune_splits,
)


def test_apd_panel_recipe_status_and_mapping() -> None:
    """Verify apd_panel.yaml has status='needs-human-roi-drawing', README 2.5 mapping, and [Published] label."""
    recipe_path = Path(__file__).resolve().parents[1] / "config" / "recipes" / "apd_panel.yaml"
    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

    assert raw["status"] == "needs-human-roi-drawing"
    assert "category_mapping" in raw
    assert "published_reference_points" in raw
    assert "[Published]" in raw["published_reference_points"]["note"]
    assert "224 px" in raw["published_reference_points"]["note"]
    assert "Limitations" in README_2_4_LIMITATIONS


def test_frozen_settings_verification_guard(tmp_path: Path) -> None:
    """Verify verify_frozen_settings passes on valid SHA-256 and raises on missing or tampered file."""
    frozen_file = tmp_path / "frozen_settings.json"
    with pytest.raises(RuntimeError, match="Frozen settings missing"):
        verify_frozen_settings(frozen_file)

    payload = {"w_d": 0.85, "w_c": 0.15}
    raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    sha = hashlib.sha256(raw_bytes).hexdigest()

    frozen_file.write_bytes(raw_bytes)
    sha_file = frozen_file.with_suffix(".sha256")
    sha_file.write_text(f"{sha}  frozen_settings.json\n", encoding="utf-8")

    loaded = verify_frozen_settings(frozen_file)
    assert loaded == payload

    # Tamper with frozen_settings.json
    frozen_file.write_text('{"w_d": 0.50, "w_c": 0.50}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_frozen_settings(frozen_file)


def test_two_fold_retune_excludes_references_and_preserves_recipe_hash() -> None:
    """Verify 2-fold retune split excludes reference normals and never mutates apd_panel.yaml."""
    recipe_path = Path(__file__).resolve().parents[1] / "config" / "recipes" / "apd_panel.yaml"
    h_before = compute_recipe_file_hash(recipe_path)

    all_normals = [f"norm_{i:03d}" for i in range(20)]
    all_anomalies = [f"anom_{i:03d}" for i in range(20)]
    refs = all_normals[:4]

    (f1_train, f1_eval), (f2_train, f2_eval) = make_two_fold_retune_splits(all_normals, all_anomalies, refs, seed=42)
    for r in refs:
        assert r not in f1_train
        assert r not in f1_eval
        assert r not in f2_train
        assert r not in f2_eval

    h_after = compute_recipe_file_hash(recipe_path)
    assert h_before == h_after


def test_roi_mask_coverage_calculation() -> None:
    """Verify compute_roi_mask_coverage on synthetic defect mask and polygon."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 8:16] = True
    poly_full = [[(4, 4), (20, 4), (20, 20), (4, 20)]]
    poly_miss = [[(22, 22), (30, 22), (30, 30), (22, 30)]]

    assert np.isclose(compute_roi_mask_coverage(mask, poly_full), 1.0)
    assert np.isclose(compute_roi_mask_coverage(mask, poly_miss), 0.0)
