

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def heatmap_to_overlay(
    image: Image.Image,
    heatmap: torch.Tensor,
    opacity: float = 0.5,
) -> Image.Image:
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0.0 and 1.0.")

    if heatmap.shape not in {(7, 7), (1, 7, 7)}:
        raise ValueError("Heatmap must have shape [7, 7] or [1, 7, 7].")

    heatmap = heatmap.squeeze(0).detach().cpu().float()
    original = image.convert("RGB")

    heatmap_image = Image.fromarray(
        np.uint8(
            (heatmap - heatmap.min())
            / (heatmap.max() - heatmap.min() + 1e-8)
            * 255
        )
    ).resize(original.size, Image.Resampling.BILINEAR)

    heatmap_values = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    # Blue -> white -> red colour scale.
    red = np.clip(2.0 * heatmap_values - 0.5, 0.0, 1.0)
    green = np.clip(1.5 - 2.0 * np.abs(heatmap_values - 0.5), 0.0, 1.0)
    blue = np.clip(1.5 - 2.0 * heatmap_values, 0.0, 1.0)

    colour_map = np.stack((red, green, blue), axis=-1)
    colour_overlay = Image.fromarray(
        np.uint8(colour_map * 255),
        mode="RGB",
    )

    return Image.blend(original, colour_overlay, opacity)