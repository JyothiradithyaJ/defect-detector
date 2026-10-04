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
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
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


def _official_aupro_curve(
    score_maps: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the MVTec reference PRO curve from sorted score changes.

    This follows the released MVTec evaluation logic: every connected
    ground-truth region contributes equally, while normal pixels determine the
    false-positive rate. Thresholds are represented exactly by unique score
    values rather than a fixed threshold grid.
    """
    all_regions: list[tuple[int, np.ndarray]] = []
    normal_pixel_count = 0

    for image_index, mask in enumerate(masks):
        regions = _connected_regions(mask)
        all_regions.extend((image_index, region) for region in regions)
        normal_pixel_count += int((~mask.astype(bool)).sum())

    region_count = len(all_regions)
    if region_count == 0:
        raise ValueError("AU-PRO requires at least one defective ground-truth region.")
    if normal_pixel_count == 0:
        raise ValueError("AU-PRO requires at least one normal pixel.")

    # Each pixel contributes its FPR increment if it is normal. Each anomaly
    # region pixel contributes 1/region_size to the PRO increment.
    scores_flat = score_maps.reshape(-1).astype(np.float64, copy=False)
    fp_changes = (~masks.astype(bool)).reshape(-1).astype(np.uint8)
    pro_changes = np.zeros(scores_flat.shape, dtype=np.float64)

    for image_index, region in all_regions:
        offset = image_index * masks.shape[1] * masks.shape[2]
        pro_changes[offset + region] = 1.0 / len(region)

    order = np.argsort(scores_flat)[::-1]
    sorted_scores = scores_flat[order]
    sorted_fp = fp_changes[order].astype(np.uint64, copy=False)
    sorted_pro = pro_changes[order]

    fprs = np.cumsum(sorted_fp, dtype=np.uint64).astype(np.float64) / normal_pixel_count
    pros = np.cumsum(sorted_pro, dtype=np.float64) / region_count

    keep = np.r_[np.diff(sorted_scores) != 0, True]
    fprs = np.clip(fprs[keep], 0.0, 1.0)
    pros = np.clip(pros[keep], 0.0, 1.0)

    return (
        np.concatenate(([0.0], fprs, [1.0])),
        np.concatenate(([0.0], pros, [1.0])),
    )


def pixel_aupro(
    score_maps: list[np.ndarray] | np.ndarray,
    masks: list[np.ndarray] | np.ndarray,
    max_false_positive_rate: float = 0.30,
) -> float:
    """Calculate MVTec-compatible AU-PRO up to the requested FPR limit.

    The released MVTec protocol uses connected ground-truth regions with equal
    weighting and integrates the PRO curve only through FPR=0.30 by default.
    """
    if not 0.0 < max_false_positive_rate <= 1.0:
        raise ValueError("max_false_positive_rate must be in (0, 1].")

    scores = np.asarray(score_maps, dtype=np.float32)
    ground_truth = np.asarray(masks, dtype=bool)

    if scores.shape != ground_truth.shape:
        raise ValueError("score_maps and masks must have identical shapes.")
    if scores.ndim != 3:
        raise ValueError(
            "score_maps and masks must have shape [images, height, width]."
        )

    fpr, pro = _official_aupro_curve(scores, ground_truth)

    if max_false_positive_rate not in fpr:
        index = np.searchsorted(fpr, max_false_positive_rate)
        if index == 0 or index == len(fpr):
            raise RuntimeError("FPR integration limit is outside the computed curve.")
        left_fpr, right_fpr = fpr[index - 1], fpr[index]
        left_pro, right_pro = pro[index - 1], pro[index]
        interpolated = left_pro + (
            (right_pro - left_pro)
            * (max_false_positive_rate - left_fpr)
            / (right_fpr - left_fpr)
        )
        fpr = np.insert(fpr, index, max_false_positive_rate)
        pro = np.insert(pro, index, interpolated)

    valid = fpr <= max_false_positive_rate
    return float(
        np.trapezoid(
            pro[valid],
            fpr[valid] / max_false_positive_rate,
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
