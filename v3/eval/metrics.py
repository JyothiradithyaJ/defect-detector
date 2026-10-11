"""Standard evaluation metrics for anomaly detection and localization.

Implements metrics per README_v3.md section 8.4:
- Image AUROC
- Pixel AUROC
- AU-PRO at configurable FPR limit (default 0.05 per README, 0.30 for legacy v1)
- Recall at fixed false-reject rate (default 5%)
- Expected Calibration Error (ECE)
"""

from __future__ import annotations

from collections import deque
from itertools import pairwise

import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> float:
    """Calculate image-level AUROC from binary labels and anomaly scores."""
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)

    if labels_array.shape != scores_array.shape:
        raise ValueError("labels and scores must have identical shapes.")

    unique_labels = set(np.unique(labels_array))
    if unique_labels != {0, 1}:
        raise ValueError("AUROC requires both normal (0) and anomalous (1) labels.")

    return float(roc_auc_score(labels_array, scores_array))


def pixel_auroc(
    score_maps: list[np.ndarray] | np.ndarray,
    masks: list[np.ndarray] | np.ndarray,
) -> float:
    """Calculate pixel-level AUROC from anomaly maps and binary ground truth masks."""
    scores = np.asarray(score_maps, dtype=np.float32)
    ground_truth = np.asarray(masks, dtype=bool)

    if scores.shape != ground_truth.shape:
        raise ValueError("score_maps and masks must have identical shapes.")
    if scores.ndim != 3:
        raise ValueError("score_maps and masks must have shape [N, H, W].")

    if scores.size >= 100_000:
        import torch

        if torch.cuda.is_available():
            gt_flat_np = ground_truth.reshape(-1)
            n_pos = int(gt_flat_np.sum())
            if 0 < n_pos < gt_flat_np.size:
                device = torch.device("cuda")
                s_flat = torch.from_numpy(np.ascontiguousarray(scores.reshape(-1))).to(device)
                gt_flat = torch.from_numpy(np.ascontiguousarray(gt_flat_np)).to(device)
                order = torch.argsort(s_flat, descending=True)
                s_sorted = s_flat[order]
                y_sorted = gt_flat[order].to(torch.float64)
                tps = torch.cumsum(y_sorted, dim=0)
                fps = torch.arange(1, len(y_sorted) + 1, device=device, dtype=torch.float64) - tps
                keep = torch.cat([torch.diff(s_sorted) != 0, torch.ones(1, dtype=torch.bool, device=device)])
                tps_k = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), tps[keep]])
                fps_k = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), fps[keep]])
                return float((torch.trapezoid(tps_k, fps_k) / (tps_k[-1] * fps_k[-1])).item())

    return image_auroc(ground_truth.reshape(-1), scores.reshape(-1))


def recall_at_frr(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    false_reject_rate: float = 0.05,
) -> float:
    """Calculate anomalous image recall at a fixed false-reject rate on normals.

    FRR is the false-positive rate on normal images (fraction of normals with score >= tau).
    tau is selected as the (1 - FRR) percentile of normal image scores.
    Recall is the fraction of defective images scored >= tau.
    """
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)

    normals = scores_array[labels_array == 0]
    anomalies = scores_array[labels_array == 1]

    if len(normals) == 0 or len(anomalies) == 0:
        raise ValueError("Both normal and anomalous samples required to compute recall at FRR.")

    tau = float(np.percentile(normals, 100.0 * (1.0 - false_reject_rate)))
    return float(np.mean(anomalies >= tau))


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
        raise ValueError("probabilities must be between 0.0 and 1.0.")

    total = len(probabilities_array)
    if total == 0:
        raise ValueError("probabilities must not be empty.")

    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)

    for lower, upper in pairwise(boundaries):
        if upper == 1.0:
            in_bin = (probabilities_array >= lower) & (probabilities_array <= upper)
        else:
            in_bin = (probabilities_array >= lower) & (probabilities_array < upper)

        count = int(in_bin.sum())
        if count == 0:
            continue

        mean_prob = probabilities_array[in_bin].mean()
        obs_freq = labels_array[in_bin].mean()
        ece += (count / total) * abs(mean_prob - obs_freq)

    return float(ece)


def _connected_regions(mask: np.ndarray) -> list[np.ndarray]:
    """Find 8-connected positive regions in a 2D binary ground-truth mask."""
    if mask.ndim != 2:
        raise ValueError("Each ground-truth mask must be two-dimensional.")

    mask_bool = mask.astype(bool)
    visited = np.zeros_like(mask_bool, dtype=bool)
    height, width = mask_bool.shape
    regions: list[np.ndarray] = []
    neighbours = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for row in range(height):
        for column in range(width):
            if not mask_bool[row, column] or visited[row, column]:
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
                        and mask_bool[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
            rows, columns = zip(*coordinates)
            regions.append(
                np.ravel_multi_index(
                    (np.asarray(rows), np.asarray(columns)),
                    mask_bool.shape,
                )
            )
    return regions


def _aupro_curve(
    score_maps: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the PRO curve from sorted score changes."""
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

    if score_maps.size >= 100_000:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            scores_flat_t = torch.from_numpy(np.ascontiguousarray(score_maps.reshape(-1), dtype=np.float32)).to(device)
            fp_changes_t = torch.from_numpy(np.ascontiguousarray((~masks.astype(bool)).reshape(-1), dtype=np.uint8)).to(device)
            pro_changes_np = np.zeros(score_maps.size, dtype=np.float32)
            for image_index, region in all_regions:
                offset = image_index * masks.shape[1] * masks.shape[2]
                pro_changes_np[offset + region] = 1.0 / len(region)
            pro_changes_t = torch.from_numpy(pro_changes_np).to(device)

            order = torch.argsort(scores_flat_t, descending=True)
            sorted_scores = scores_flat_t[order]
            fprs_t = torch.cumsum(fp_changes_t[order].to(torch.float64), dim=0) / normal_pixel_count
            pros_t = torch.cumsum(pro_changes_t[order].to(torch.float64), dim=0) / region_count
            keep = torch.cat([torch.diff(sorted_scores) != 0, torch.ones(1, dtype=torch.bool, device=device)])
            fprs = np.clip(fprs_t[keep].cpu().numpy(), 0.0, 1.0)
            pros = np.clip(pros_t[keep].cpu().numpy(), 0.0, 1.0)
            return (
                np.concatenate(([0.0], fprs, [1.0])),
                np.concatenate(([0.0], pros, [1.0])),
            )

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
    max_false_positive_rate: float = 0.05,
) -> float:
    """Calculate AU-PRO up to max_false_positive_rate (default 0.05 per README/BUILD_PROMPT)."""
    if not 0.0 < max_false_positive_rate <= 1.0:
        raise ValueError("max_false_positive_rate must be in (0, 1].")

    scores = np.asarray(score_maps, dtype=np.float32)
    ground_truth = np.asarray(masks, dtype=bool)

    if scores.shape != ground_truth.shape:
        raise ValueError("score_maps and masks must have identical shapes.")
    if scores.ndim != 3:
        raise ValueError("score_maps and masks must have shape [N, H, W].")

    fpr, pro = _aupro_curve(scores, ground_truth)

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
