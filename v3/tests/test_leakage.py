"""Tests for data leakage prevention and disjointness assertions."""

import pytest

from v3.eval.splits import assert_disjoint_references_and_negatives


def test_disjoint_sets_pass_leakage_assertion() -> None:
    """Verify that disjoint reference normals and test negatives pass with no errors."""
    refs = ["ref_001", "ref_002", "ref_003"]
    test_negs = ["test_neg_001", "test_neg_002", "test_neg_003"]
    # Should complete without error
    assert_disjoint_references_and_negatives(refs, test_negs)


def test_overlapping_sets_fail_leakage_assertion() -> None:
    """Verify that overlapping reference normals and test negatives trigger an AssertionError."""
    refs = ["ref_001", "ref_002", "leakage_image"]
    test_negs = ["test_neg_001", "leakage_image", "test_neg_002"]
    with pytest.raises(AssertionError, match="Data leakage detected!"):
        assert_disjoint_references_and_negatives(refs, test_negs)
