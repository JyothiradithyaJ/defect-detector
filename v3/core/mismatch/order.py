"""Order / sequence / color-or-slot swap detector inside an ROI or across paired sub-regions."""

from __future__ import annotations

import numpy as np
from PIL import Image

from v3.core.bank import l2_normalize_np
from v3.core.contracts import ROI
from v3.core.mismatch.presence import extract_roi_crop_and_mask


def compute_ordered_slot_signature(image: Image.Image, roi: ROI) -> np.ndarray:
    """Split an ROI into ordered sub-slots (default 2 horizontal or vertical halves) and concatenate descriptors."""
    crop, mask = extract_roi_crop_and_mask(image, roi)
    if crop.size == 0 or not np.any(mask):
        return np.zeros(12, dtype=np.float32)

    num_slots = int(roi.params.get("num_slots", 2))
    axis = str(roi.params.get("order_axis", "horizontal"))
    h, w, _ = crop.shape
    slot_vecs: list[np.ndarray] = []

    for idx in range(num_slots):
        if axis == "vertical":
            r0 = int(round(idx * h / num_slots))
            r1 = int(round((idx + 1) * h / num_slots))
            sub = crop[max(0, r0) : max(r0 + 1, r1), :, :]
        else:
            c0 = int(round(idx * w / num_slots))
            c1 = int(round((idx + 1) * w / num_slots))
            sub = crop[:, max(0, c0) : max(c0 + 1, c1), :]

        mean_rgb = np.mean(sub / 255.0, axis=(0, 1))
        std_rgb = np.std(sub / 255.0, axis=(0, 1))
        slot_vecs.append(np.concatenate([mean_rgb, std_rgb]).astype(np.float32))

    full_sig = np.concatenate(slot_vecs).astype(np.float32)
    return l2_normalize_np(full_sig)


def score_order(query_image: Image.Image, ref_signatures: list[np.ndarray], roi: ROI) -> float:
    """Compute ordered-slot dissimilarity score (1 - max cosine similarity)."""
    q_sig = compute_ordered_slot_signature(query_image, roi)
    if not ref_signatures:
        return 0.0
    sims = [float(np.dot(q_sig, r)) for r in ref_signatures]
    return float(max(0.0, 1.0 - max(sims)))
