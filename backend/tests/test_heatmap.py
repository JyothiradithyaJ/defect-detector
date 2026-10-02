"""Unit tests for heatmap rendering."""

from pathlib import Path
import sys

import pytest
import torch
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.heatmap import heatmap_to_overlay  # noqa: E402


def test_heatmap_overlay_keeps_original_image_size() -> None:
    """A 7x7 heatmap becomes an overlay with the original dimensions."""
    image = Image.new("RGB", (640, 480), "white")
    heatmap = torch.zeros(1, 7, 7)
    heatmap[0, 3, 4] = 1.0

    overlay = heatmap_to_overlay(image, heatmap)

    assert overlay.mode == "RGB"
    assert overlay.size == (640, 480)


def test_heatmap_accepts_unbatched_grid() -> None:
    """The utility accepts one unbatched 7x7 grid."""
    image = Image.new("RGB", (100, 100), "white")
    heatmap = torch.zeros(7, 7)

    overlay = heatmap_to_overlay(image, heatmap)

    assert overlay.size == image.size


def test_heatmap_rejects_invalid_opacity() -> None:
    """Opacity must stay within the valid blend range."""
    image = Image.new("RGB", (100, 100), "white")
    heatmap = torch.zeros(7, 7)

    with pytest.raises(ValueError, match="opacity must be between"):
        heatmap_to_overlay(image, heatmap, opacity=1.1)


def test_heatmap_rejects_invalid_shape() -> None:
    """Only a 7x7 grid, with or without a batch dimension, is valid."""
    image = Image.new("RGB", (100, 100), "white")
    invalid_heatmap = torch.zeros(8, 8)

    with pytest.raises(ValueError, match="Heatmap must have shape"):
        heatmap_to_overlay(image, invalid_heatmap)