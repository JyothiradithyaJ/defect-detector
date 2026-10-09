"""Synthetic tests for Stage A mismatch detection."""

import pytest


@pytest.mark.skip(reason="Phase 4 implementation pending")
def test_synthetic_swap_detection() -> None:
    """Verify swap between ROIs flags anomaly above normal false-reject rate."""
