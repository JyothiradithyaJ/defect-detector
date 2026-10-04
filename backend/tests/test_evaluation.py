"""Unit tests for anomaly-detection evaluation metrics."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.evaluation import (  # noqa: E402
    expected_calibration_error,
    image_auroc,
    pixel_aupro,
    tensor_heatmap_to_numpy,
)


def test_image_auroc_is_one_for_perfect_ranking() -> None:
    """Perfectly ranked normal and defective images have AUROC 1."""
    labels = [0, 0, 1, 1]
    defect_scores = [-0.4, -0.1, 0.2, 0.8]

    assert image_auroc(labels, defect_scores) == 1.0


def test_image_auroc_requires_both_classes() -> None:
    """AUROC cannot be calculated from only good or only defective images."""
    with pytest.raises(ValueError, match="requires both"):
        image_auroc(labels=[0, 0], scores=[0.1, 0.2])


def test_ece_is_zero_for_perfectly_calibrated_predictions() -> None:
    """Predictions matching observed label frequency have ECE zero."""
    probabilities = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]

    assert expected_calibration_error(probabilities, labels) == 0.0


def test_ece_rejects_invalid_probability() -> None:
    """Probabilities outside [0, 1] are invalid."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        expected_calibration_error(
            probabilities=[-0.1, 0.9],
            labels=[0, 1],
        )


def test_pixel_aupro_is_high_for_a_perfect_heatmap() -> None:
    """A heatmap that exactly highlights defects has high AU-PRO."""
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 0:2, 2:4] = True

    score_maps = masks.astype(np.float64)

    score = pixel_aupro(
        score_maps=score_maps,
        masks=masks,
        max_false_positive_rate=0.30,
    )

    assert score > 0.95


def test_pixel_aupro_handles_regions_from_multiple_images() -> None:
    """Defect regions must be matched with their own image heatmaps."""
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 3, 3] = True

    score_maps = np.zeros((2, 4, 4), dtype=np.float64)
    score_maps[0, 0, 0] = 1.0
    score_maps[1, 3, 3] = 1.0

    score = pixel_aupro(
        score_maps=score_maps,
        masks=masks,
    )

    assert score > 0.95


def test_tensor_heatmap_to_numpy_accepts_batched_heatmap() -> None:
    """A [1, 7, 7] Torch heatmap becomes a [7, 7] NumPy array."""
    heatmap = torch.zeros(1, 7, 7)

    result = tensor_heatmap_to_numpy(heatmap)

    assert isinstance(result, np.ndarray)
    assert result.shape == (7, 7)


def test_tensor_heatmap_to_numpy_rejects_invalid_shape() -> None:
    """A batch containing multiple heatmaps is not accepted."""
    with pytest.raises(ValueError, match="heatmap must have shape"):
        tensor_heatmap_to_numpy(torch.zeros(2, 7, 7))


@pytest.mark.parametrize(
    ("scores", "mask", "expected"),
    [
        (
            np.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
            np.array([[[1, 0], [0, 0]]], dtype=bool),
            1.0,
        ),
        (
            np.zeros((1, 2, 2), dtype=np.float32),
            np.array([[[1, 0], [0, 0]]], dtype=bool),
            0.0,
        ),
        (
            np.array([[[1.0, 0.5], [0.0, 0.0]]], dtype=np.float32),
            np.array([[[1, 1], [0, 0]]], dtype=bool),
            1.0,
        ),
    ],
)
def test_pixel_aupro_protocol_cases(scores, mask, expected):
    """Perfect, zero, and fully covered partial-region maps are exact."""
    assert pixel_aupro(scores, mask) == pytest.approx(expected)
