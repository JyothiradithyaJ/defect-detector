"""Phase 0 verification tests for setup, configuration, and contracts."""

from pathlib import Path

import numpy as np
import yaml

from v3.core.contracts import ROI, Mismatch, Recipe, Result


def test_contracts_instantiation() -> None:
    """Verify that core data contracts instantiate properly."""
    roi = ROI(
        id="test_roi",
        kind="presence",
        polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        severity=1,
        hard=False,
        params={"hsv": "red"},
    )
    recipe = Recipe(
        name="test_recipe",
        template_path="dummy.png",
        resolution=(1920, 1080),
        view="fixed",
        symmetry="none",
        stable_roi=[(0, 0), (100, 100)],
        rois=[roi],
    )
    mismatch = Mismatch(
        roi_id=roi.id,
        kind="presence",
        severity=1,
        hard=False,
        confidence=0.95,
        evidence={"observed": False},
    )
    result = Result(
        verdict="REWORK",
        mismatches=[mismatch],
        image_score=0.82,
        anomaly_map=np.zeros((100, 100), dtype=np.float32),
        meta={"seed": 0},
    )

    assert recipe.name == "test_recipe"
    assert len(recipe.rois) == 1
    assert result.verdict == "REWORK"
    assert result.anomaly_map.shape == (100, 100)


def test_default_config_loads() -> None:
    """Verify default.yaml exists, parses, and contains required top-level keys."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    assert config_path.is_file(), f"Missing config file: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "seed" in cfg
    assert "clip" in cfg
    assert "dino" in cfg
    assert "match" in cfg
    assert "refs" in cfg
    assert "fusion" in cfg
    assert "decide" in cfg


def test_apd_panel_recipe_loads() -> None:
    """Verify apd_panel.yaml exists and parses with valid ROIs."""
    recipe_path = Path(__file__).resolve().parents[1] / "config" / "recipes" / "apd_panel.yaml"
    assert recipe_path.is_file(), f"Missing recipe file: {recipe_path}"
    with open(recipe_path, "r", encoding="utf-8") as f:
        recipe = yaml.safe_load(f)

    assert recipe["name"] == "apd_panel"
    assert recipe["resolution"] == [1920, 1080]
    assert len(recipe["rois"]) > 0
