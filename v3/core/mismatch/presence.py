"""Presence mismatch detector inside an ROI polygon."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from v3.core.bank import l2_normalize_np
from v3.core.contracts import ROI


def extract_roi_crop_and_mask(image: Image.Image, roi: ROI) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb_crop [H_c, W_c, 3], bool_mask [H_c, W_c]) for `roi.polygon`."""
    w, h = image.size
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if not roi.polygon:
        return rgb, np.ones((h, w), dtype=bool)

    mask_img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_img).polygon([(int(x), int(y)) for x, y in roi.polygon], outline=1, fill=1)
    mask = np.asarray(mask_img, dtype=np.uint8) > 0

    xs = [int(p[0]) for p in roi.polygon]
    ys = [int(p[1]) for p in roi.polygon]
    x0 = max(0, min(w - 1, min(xs)))
    y0 = max(0, min(h - 1, min(ys)))
    x1 = max(x0 + 1, min(w, max(xs) + 1))
    y1 = max(y0 + 1, min(h, max(ys) + 1))
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def compute_roi_descriptor(image: Image.Image, roi: ROI) -> np.ndarray:
    """Compute a normalized appearance + edge descriptor vector for an ROI crop."""
    crop, mask = extract_roi_crop_and_mask(image, roi)
    if crop.size == 0 or not np.any(mask):
        return np.zeros(16, dtype=np.float32)

    pixels = crop[mask] / 255.0  # [N, 3]
    mean_rgb = np.mean(pixels, axis=0)
    std_rgb = np.std(pixels, axis=0)
    q25_rgb = np.percentile(pixels, 25, axis=0)
    q75_rgb = np.percentile(pixels, 75, axis=0)

    gray = np.mean(crop / 255.0, axis=-1)
    gx = np.gradient(gray, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
    gy = np.gradient(gray, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
    grad_mag = np.hypot(gx, gy)[mask]
    edge_stats = np.array(
        [
            float(np.mean(grad_mag)),
            float(np.std(grad_mag)),
            float(np.percentile(grad_mag, 90)),
            float(np.mean(grad_mag > 0.08)),
        ],
        dtype=np.float32,
    )
    vec = np.concatenate([mean_rgb, std_rgb, q25_rgb, q75_rgb, edge_stats]).astype(np.float32)
    return l2_normalize_np(vec)


def score_presence(query_image: Image.Image, ref_descriptors: list[np.ndarray], roi: ROI) -> float:
    """Return raw presence dissimilarity score (1 - max cosine similarity + L1 intensity shift)."""
    q_desc = compute_roi_descriptor(query_image, roi)
    if not ref_descriptors:
        return 0.0
    sims = [float(np.dot(q_desc, r)) for r in ref_descriptors]
    return float(max(0.0, 1.0 - max(sims)))
