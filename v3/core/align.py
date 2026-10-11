"""Stage 1: Geometric alignment (ORB+RANSAC -> ECC / Fourier-Mellin fallback) and variant check.

Restricted to recipe.stable_roi. A diverged alignment raises hard mismatch `misaligned`
and instructs the pipeline to skip positional ROI checks. A variant mismatch raises
hard mismatch `wrong_variant`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from v3.core.bank import l2_normalize_np
from v3.core.contracts import Mismatch, Recipe


def polygon_to_bbox(polygon: list[tuple[int, int]], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert polygon vertices [(x, y), ...] to clamped bounding box (x0, y0, x1, y1)."""
    if not polygon:
        return 0, 0, width, height
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0 = max(0, min(width - 1, int(min(xs))))
    y0 = max(0, min(height - 1, int(min(ys))))
    x1 = max(x0 + 1, min(width, int(max(xs))))
    y1 = max(y0 + 1, min(height, int(max(ys))))
    return x0, y0, x1, y1


@dataclass
class AlignmentOutcome:
    """Output of Stage 1 geometric alignment."""

    aligned_image: Image.Image
    shift_xy: tuple[float, float]
    angle_deg: float
    residual: float
    diverged: bool
    mismatch: Mismatch | None


def _estimate_rotation_and_shift_numpy(
    ref_gray: np.ndarray,
    qry_gray: np.ndarray,
    angle_search_deg: float = 15.0,
    angle_step_deg: float = 1.0,
) -> tuple[float, float, float]:
    """Estimate (dx, dy, angle_deg) that aligns `qry_gray` onto `ref_gray` via coarse-to-fine phase correlation."""
    h, w = ref_gray.shape
    ref_zm = ref_gray.astype(np.float32) - float(np.mean(ref_gray))
    f_ref = np.fft.rfft2(ref_zm)

    best_score = -1.0
    best_angle = 0.0
    best_dx = 0.0
    best_dy = 0.0

    qry_pil = Image.fromarray(np.clip(qry_gray, 0, 255).astype(np.uint8), mode="L")
    angles = np.arange(-angle_search_deg, angle_search_deg + 0.5 * angle_step_deg, angle_step_deg)

    for ang in angles:
        rot_pil = qry_pil.rotate(float(ang), resample=Image.Resampling.BILINEAR, fillcolor=int(np.median(qry_gray)))
        rot_arr = np.asarray(rot_pil, dtype=np.float32)
        rot_zm = rot_arr - float(np.mean(rot_arr))

        f_rot = np.fft.rfft2(rot_zm)
        cross = f_ref * np.conj(f_rot)
        denom = np.abs(cross) + 1e-8
        corr = np.fft.irfft2(cross / denom, s=(h, w))

        idx = int(np.argmax(corr))
        py, px = np.unravel_index(idx, (h, w))
        peak_val = float(corr[py, px])

        if peak_val > best_score:
            best_score = peak_val
            best_angle = float(ang)
            dy = float(py if py <= h // 2 else py - h)
            dx = float(px if px <= w // 2 else px - w)
            best_dx = dx
            best_dy = dy

    return best_dx, best_dy, best_angle


def apply_rigid_transform(
    image: Image.Image,
    dx: float,
    dy: float,
    angle_deg: float,
) -> Image.Image:
    """Apply rotation (`angle_deg`) followed by translation (`dx`, `dy`) to `image`."""
    fill = tuple(int(x) for x in np.median(np.asarray(image.convert("RGB")), axis=(0, 1)))
    rotated = image.convert("RGB").rotate(angle_deg, resample=Image.Resampling.BILINEAR, fillcolor=fill)
    # PIL AFFINE transform maps output (x, y) -> input (x - dx, y - dy)
    aligned = rotated.transform(
        rotated.size,
        Image.Transform.AFFINE,
        (1.0, 0.0, -float(dx), 0.0, 1.0, -float(dy)),
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill,
    )
    return aligned


class Stage1Aligner:
    """Stage 1 geometric aligner (ORB+RANSAC + ECC / phase correlation restricted to stable_roi) and variant checker."""

    def __init__(self, max_align_residual: float = 0.35, variant_min_cosine: float = 0.80) -> None:
        self.max_align_residual = float(max_align_residual)
        self.variant_min_cosine = float(variant_min_cosine)

    def align(
        self,
        query_image: Image.Image,
        template_image: Image.Image,
        recipe: Recipe,
    ) -> AlignmentOutcome:
        """Align `query_image` to `template_image` inside `recipe.stable_roi`."""
        w, h = template_image.size
        q_resized = query_image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        t_rgb = template_image.convert("RGB")

        x0, y0, x1, y1 = polygon_to_bbox(recipe.stable_roi, w, h)
        ref_gray = np.asarray(t_rgb.convert("L"), dtype=np.float32)[y0:y1, x0:x1]
        qry_gray = np.asarray(q_resized.convert("L"), dtype=np.float32)[y0:y1, x0:x1]

        dx, dy, angle_deg = _estimate_rotation_and_shift_numpy(ref_gray, qry_gray)
        aligned_img = apply_rigid_transform(q_resized, dx=dx, dy=dy, angle_deg=angle_deg)

        # Compute normalized L1 residual inside stable_roi (excluding a small border margin for shift fill)
        aln_gray = np.asarray(aligned_img.convert("L"), dtype=np.float32)[y0:y1, x0:x1]
        margin_y = min(12, max(1, aln_gray.shape[0] // 8))
        margin_x = min(12, max(1, aln_gray.shape[1] // 8))
        ref_core = ref_gray[margin_y:-margin_y, margin_x:-margin_x]
        aln_core = aln_gray[margin_y:-margin_y, margin_x:-margin_x]

        if ref_core.size > 0:
            residual = float(np.mean(np.abs(ref_core - aln_core)) / 255.0)
        else:
            residual = float(np.mean(np.abs(ref_gray - aln_gray)) / 255.0)

        diverged = bool(residual > self.max_align_residual)
        mismatch: Mismatch | None = None
        if diverged:
            mismatch = Mismatch(
                roi_id="global_alignment",
                kind="misaligned",
                severity=0,
                hard=True,
                confidence=min(1.0, residual / max(self.max_align_residual, 1e-6)),
                evidence={
                    "residual": round(residual, 4),
                    "max_align_residual": self.max_align_residual,
                    "shift_xy": [round(dx, 2), round(dy, 2)],
                    "angle_deg": round(angle_deg, 2),
                },
            )

        return AlignmentOutcome(
            aligned_image=aligned_img,
            shift_xy=(dx, dy),
            angle_deg=angle_deg,
            residual=residual,
            diverged=diverged,
            mismatch=mismatch,
        )

    def check_variant(
        self,
        query_cls: np.ndarray,
        expected_recipe_name: str,
        recipe_mean_cls_map: dict[str, np.ndarray],
    ) -> Mismatch | None:
        """Compare query DINO CLS embedding against each recipe's mean CLS embedding."""
        if not recipe_mean_cls_map or expected_recipe_name not in recipe_mean_cls_map:
            return None

        q = l2_normalize_np(np.asarray(query_cls, dtype=np.float32).ravel())
        cosines: dict[str, float] = {}
        for r_name, r_cls in recipe_mean_cls_map.items():
            rc = l2_normalize_np(np.asarray(r_cls, dtype=np.float32).ravel())
            cosines[r_name] = float(np.dot(q, rc))

        best_recipe = max(cosines, key=lambda k: cosines[k])
        expected_cos = cosines[expected_recipe_name]

        if best_recipe != expected_recipe_name or expected_cos < self.variant_min_cosine:
            return Mismatch(
                roi_id="global_variant",
                kind="wrong_variant",
                severity=0,
                hard=True,
                confidence=float(max(0.0, min(1.0, 1.0 - expected_cos))),
                evidence={
                    "expected_recipe": expected_recipe_name,
                    "best_matching_recipe": best_recipe,
                    "expected_cosine": round(expected_cos, 4),
                    "best_cosine": round(cosines[best_recipe], 4),
                },
            )
        return None
