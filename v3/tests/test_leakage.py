"""Tests for reference and evaluation split leakage guard."""

import pytest


@pytest.mark.skip(reason="Phase 1 harness implementation pending")
def test_reference_normals_excluded_from_test_negatives() -> None:
    """Verify reference normal IDs and test negative IDs are completely disjoint."""
