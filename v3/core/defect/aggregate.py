"""Neighbourhood spatial aggregation and top-k image scoring for patch anomaly maps."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def aggregate_patch_map(patch_map: np.ndarray, radius: int = 1) -> np.ndarray:
    """Smooth a 2D patch anomaly map using a (2*radius + 1) x (2*radius + 1) uniform window.

    Args:
        patch_map: 2D array of shape [H_p, W_p].
        radius: Spatial aggregation radius (e.g. 0, 1, 3 from config match.agg_radii).

    Returns:
        Smoothed 2D float32 array of shape [H_p, W_p].
    """
    arr = np.asarray(patch_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D patch_map, got shape {arr.shape}")
    r = int(radius)
    if r <= 0:
        return arr.copy()

    h, w = arr.shape
    padded = np.pad(arr, ((r, r), (r, r)), mode="edge")
    k = 2 * r + 1
    acc = np.zeros((h, w), dtype=np.float32)
    for dr in range(k):
        for dc in range(k):
            acc += padded[dr : dr + h, dc : dc + w]
    return (acc / float(k * k)).astype(np.float32)


def topk_image_score(patch_map: np.ndarray, topk_frac: float = 0.001) -> float:
    """Compute image-level anomaly score as the mean of the top-k fraction of patch scores."""
    flat = np.asarray(patch_map, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.0
    k = max(1, math.ceil(flat.size * float(topk_frac)))
    k = min(k, flat.size)
    top_vals = np.partition(flat, -k)[-k:]
    return float(np.mean(top_vals))


def upsample_patch_map(patch_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly upsample a 2D patch map [H_p, W_p] to target image height and width (H, W)."""
    h, w = int(target_hw[0]), int(target_hw[1])
    arr = np.asarray(patch_map, dtype=np.float32)
    pil_map = Image.fromarray(arr, mode="F")
    resized = pil_map.resize((w, h), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)
