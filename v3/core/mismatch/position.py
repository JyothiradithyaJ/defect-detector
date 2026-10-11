"""Position / centroid offset mismatch detector inside an ROI polygon."""

from __future__ import annotations

import numpy as np
from PIL import Image

from v3.core.contracts import ROI
from v3.core.mismatch.presence import extract_roi_crop_and_mask


def compute_roi_centroid(image: Image.Image, roi: ROI) -> tuple[float, float]:
    """Compute normalized intensity-weighted centroid (cx, cy) in [0, 1] x [0, 1] inside `roi`."""
    crop, mask = extract_roi_crop_and_mask(image, roi)
    h, w = crop.shape[:2]
    if crop.size == 0 or not np.any(mask) or h == 0 or w == 0:
        return 0.5, 0.5

    gray = np.mean(crop, axis=-1)
    bg = float(np.median(gray[mask]))
    weights = np.abs(gray - bg) * mask.astype(np.float32)
    total_w = float(np.sum(weights))
    if total_w <= 1e-6:
        return 0.5, 0.5

    ys, xs = np.indices((h, w), dtype=np.float32)
    cx = float(np.sum(xs * weights) / (total_w * max(1.0, float(w))))
    cy = float(np.sum(ys * weights) / (total_w * max(1.0, float(h))))
    return cx, cy


def score_position(query_image: Image.Image, ref_centroids: list[tuple[float, float]], roi: ROI) -> float:
    """Compute Euclidean centroid shift relative to reference normal centroids."""
    q_cx, q_cy = compute_roi_centroid(query_image, roi)
    if not ref_centroids:
        return 0.0
    med_cx = float(np.median([c[0] for c in ref_centroids]))
    med_cy = float(np.median([c[1] for c in ref_centroids]))
    return float(np.hypot(q_cx - med_cx, q_cy - med_cy))
