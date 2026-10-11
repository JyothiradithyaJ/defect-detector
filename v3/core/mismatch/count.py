"""Count mismatch detector inside an ROI polygon."""

from __future__ import annotations

import numpy as np
from PIL import Image

from v3.core.contracts import ROI
from v3.core.mismatch.presence import extract_roi_crop_and_mask


def estimate_blob_count_and_area(image: Image.Image, roi: ROI) -> tuple[int, float]:
    """Estimate foreground connected component count and foreground area fraction inside `roi`."""
    crop, mask = extract_roi_crop_and_mask(image, roi)
    if crop.size == 0 or not np.any(mask):
        return 0, 0.0

    gray = np.mean(crop, axis=-1)
    threshold = float(roi.params.get("intensity_threshold", np.median(gray[mask]) + 15.0))
    polarity = str(roi.params.get("polarity", "bright"))
    fg = ((gray > threshold) if polarity == "bright" else (gray < threshold)) & mask
    area_frac = float(np.mean(fg[mask])) if np.any(mask) else 0.0

    # Simple 4-connected component counting on foreground mask
    visited = np.zeros_like(fg, dtype=bool)
    min_blob_px = int(roi.params.get("min_blob_pixels", 4))
    h, w = fg.shape
    blob_count = 0

    for r in range(h):
        for c in range(w):
            if fg[r, c] and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                size = 0
                while stack:
                    cr, cc = stack.pop()
                    size += 1
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if size >= min_blob_px:
                    blob_count += 1

    return blob_count, area_frac


def score_count(query_image: Image.Image, ref_counts_and_areas: list[tuple[int, float]], roi: ROI) -> float:
    """Score deviation in component count and foreground area fraction relative to references."""
    q_count, q_area = estimate_blob_count_and_area(query_image, roi)
    expected_count = roi.params.get("expected_count")
    if expected_count is not None:
        count_diff = abs(float(q_count) - float(expected_count))
    elif ref_counts_and_areas:
        med_count = float(np.median([c for c, _ in ref_counts_and_areas]))
        count_diff = abs(float(q_count) - med_count)
    else:
        count_diff = 0.0

    if ref_counts_and_areas:
        med_area = float(np.median([a for _, a in ref_counts_and_areas]))
        area_diff = abs(q_area - med_area)
    else:
        area_diff = 0.0

    return float(count_diff + 2.0 * area_diff)
