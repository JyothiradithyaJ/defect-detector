"""Evaluation metrics for image anomaly detection and localisation."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def image_auroc(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> float:
    """Calculate image-level AUROC from binary labels and defect scores."""
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)

    if labels_array.shape != scores_array.shape:
        raise ValueError("labels and scores must have identical shapes.")

    if set(np.unique(labels_array)) != {0, 1}:
        raise ValueError("AUROC requires both good (0) and defective (1) labels.")

    return float(roc_auc_score(labels_array, scores_array))


def expected_calibration_error(
    probabilities: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    bins: int = 15,
) -> float:
    """Calculate binary Expected Calibration Error (ECE)."""
    if bins < 1:
        raise ValueError("bins must be at least 1.")

    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)

    if probabilities_array.shape != labels_array.shape:
        raise ValueError("probabilities and labels must have identical shapes.")

    if np.any(probabilities_array < 0.0) or np.any(probabilities_array > 1.0):
        raise ValueError("probabilities must be between 0 and 1.")

    ece = 0.0
    total = len(probabilities_array)

    if total == 0:
        raise ValueError("probabilities must not be empty.")

    boundaries = np.linspace(0.0, 1.0, bins + 1)

    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            in_bin = (
                (probabilities_array >= lower)
                & (probabilities_array <= upper)
            )
        else:
            in_bin = (
                (probabilities_array >= lower)
                & (probabilities_array < upper)
            )

        count = int(in_bin.sum())

        if count == 0:
            continue

        mean_probability = probabilities_array[in_bin].mean()
        observed_frequency = labels_array[in_bin].mean()

        ece += (count / total) * abs(
            mean_probability - observed_frequency
        )

    return float(ece)


def _connected_regions(mask: np.ndarray) -> list[np.ndarray]:
    """Find 8-connected positive regions in one binary ground-truth mask."""
    if mask.ndim != 2:
        raise ValueError("Each ground-truth mask must be two-dimensional.")

    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    regions: list[np.ndarray] = []

    neighbours = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for row in range(height):
        for column in range(width):
            if not mask[row, column] or visited[row, column]:
                continue

            queue: deque[tuple[int, int]] = deque([(row, column)])
            visited[row, column] = True
            coordinates: list[tuple[int, int]] = []

            while queue:
                current_row, current_column = queue.popleft()
                coordinates.append((current_row, current_column))

                for row_offset, column_offset in neighbours:
                    next_row = current_row + row_offset
                    next_column = current_column + column_offset

                    is_inside_image = (
                        0 <= next_row < height
                        and 0 <= next_column < width
                    )

                    if (
                        is_inside_image
                        and mask[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))

            rows, columns = zip(*coordinates)
            regions.append(
                np.ravel_multi_index(
                    (np.asarray(rows), np.asarray(columns)),
                    mask.shape,
                )
            )

    return regions


def pixel_aupro(
    score_maps: list[np.ndarray] | np.ndarray,
    masks: list[np.ndarray] | np.ndarray,
    max_false_positive_rate: float = 0.30,
    thresholds: int = 200,
) -> float:
    """
    Calculate approximate pixel-level AU-PRO.

    AU-PRO measures the mean fraction of each ground-truth defect region
    covered by the predicted heatmap, integrated up to a fixed false-positive
    rate. A 200-threshold approximation is used for practical CPU evaluation.
    """
    if not 0.0 < max_false_positive_rate <= 1.0:
        raise ValueError("max_false_positive_rate must be in (0, 1].")

    if thresholds < 2:
        raise ValueError("thresholds must be at least 2.")

    # Heatmaps originate from PyTorch as float32. Retain that precision: float64
    # would duplicate large MVTec maps (for example 67 x 900 x 900 bottle maps)
    # and can exhaust RAM without improving this threshold-based metric.
    scores = np.asarray(score_maps, dtype=np.float32)
    ground_truth = np.asarray(masks, dtype=bool)

    if scores.shape != ground_truth.shape:
        raise ValueError("score_maps and masks must have identical shapes.")

    if scores.ndim != 3:
        raise ValueError(
            "score_maps and masks must have shape [images, height, width]."
        )

    # Each connected-region index is local to one image. Keep its image index
    # so it is never accidentally applied to a different heatmap.
    all_regions: list[tuple[int, np.ndarray]] = []

    for image_index, mask in enumerate(ground_truth):
        all_regions.extend(
            (image_index, region)
            for region in _connected_regions(mask)
        )

    if not all_regions:
        raise ValueError("AU-PRO requires at least one defective ground-truth region.")

    flattened_scores = scores.reshape(-1)
    flattened_masks = ground_truth.reshape(-1)

    negative_pixel_count = int((~flattened_masks).sum())

    if negative_pixel_count == 0:
        raise ValueError("AU-PRO requires at least one normal pixel.")

    maximum = float(flattened_scores.max())
    minimum = float(flattened_scores.min())

    # First threshold produces no predictions: FPR=0 and PRO=0.
    threshold_values = np.concatenate(
        (
            np.array([np.nextafter(maximum, np.inf)]),
            np.linspace(maximum, minimum, thresholds),
        )
    )

    false_positive_rates = []
    pro_values = []

    for threshold in threshold_values:
        predictions = scores >= threshold

        false_positives = int(
            (predictions & ~ground_truth).sum()
        )
        false_positive_rate = false_positives / negative_pixel_count

        region_overlaps = [
            predictions[image_index].ravel()[region].mean()
            for image_index, region in all_regions
        ]

        false_positive_rates.append(false_positive_rate)
        pro_values.append(float(np.mean(region_overlaps)))

    false_positive_rates_array = np.asarray(false_positive_rates)
    pro_values_array = np.asarray(pro_values)

    valid = false_positive_rates_array <= max_false_positive_rate

    fpr = false_positive_rates_array[valid]
    pro = pro_values_array[valid]

    if len(fpr) == 0:
        return 0.0

    # Add an interpolated endpoint exactly at the requested FPR limit. This
    # matters for a perfect binary heatmap, whose curve can jump straight from
    # FPR=0 to FPR=1 without naturally sampling a point at (for example) 0.30.
    if fpr[-1] < max_false_positive_rate:
        pro_at_limit = np.interp(
            max_false_positive_rate,
            false_positive_rates_array,
            pro_values_array,
        )
        fpr = np.append(fpr, max_false_positive_rate)
        pro = np.append(pro, pro_at_limit)

    if len(fpr) < 2:
        return 0.0

    # Normalize the x-axis so AU-PRO is always in the 0–1 range.
    return float(
        np.trapezoid(
            pro,
            fpr / max_false_positive_rate,
        )
    )


def tensor_heatmap_to_numpy(heatmap: torch.Tensor) -> np.ndarray:
    """Convert one 2D heatmap, optionally with a singleton batch, to NumPy."""
    if heatmap.ndim == 2:
        return heatmap.detach().cpu().numpy()

    if heatmap.ndim == 3 and heatmap.shape[0] == 1:
        return heatmap.squeeze(0).detach().cpu().numpy()

    raise ValueError(
        "heatmap must have shape [height, width] or [1, height, width]."
    )
