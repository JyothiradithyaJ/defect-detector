"""Tests for data leakage prevention, reference/negative disjointness, and recipe YAML hash immutability."""

from __future__ import annotations

from pathlib import Path

import pytest

from v3.core.roi import StageAROIEngine, compute_recipe_file_hash, load_recipe_from_yaml
from v3.eval.splits import assert_disjoint_references_and_negatives
from v3.eval.step4_stage_a_eval import build_validation_assembly_dataset


def test_disjoint_sets_pass_leakage_assertion() -> None:
    """Verify that disjoint reference normals and test negatives pass with no errors."""
    refs = ["ref_001", "ref_002", "ref_003"]
    test_negs = ["test_neg_001", "test_neg_002", "test_neg_003"]
    assert_disjoint_references_and_negatives(refs, test_negs)


def test_overlapping_sets_fail_leakage_assertion() -> None:
    """Verify that overlapping reference normals and test negatives trigger an AssertionError."""
    refs = ["ref_001", "ref_002", "leakage_image"]
    test_negs = ["test_neg_001", "leakage_image", "test_neg_002"]
    with pytest.raises(AssertionError, match="Data leakage detected!"):
        assert_disjoint_references_and_negatives(refs, test_negs)


def test_recipe_file_hash_unchanged_after_anomalous_scoring() -> None:
    """Verify that loading a recipe and scoring an anomalous image never mutates the recipe YAML on disk."""
    recipe_path = Path(__file__).resolve().parents[1] / "config" / "recipes" / "breakfast_box.yaml"
    hash_before = compute_recipe_file_hash(recipe_path)

    recipe, _ = load_recipe_from_yaml(recipe_path)
    _, ref_normals, _, val_anomalies = build_validation_assembly_dataset(seed=42)
    engine = StageAROIEngine(min_mad=1e-3, default_z_threshold=3.5)
    engine.fit(ref_normals, recipe)
    engine.inspect(val_anomalies[0][0], recipe)

    hash_after = compute_recipe_file_hash(recipe_path)
    assert hash_before == hash_after, "Recipe YAML file hash was modified!"
