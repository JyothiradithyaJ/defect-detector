"""Stage 0: Quality check (blur, exposure, contrast, framing).

Thresholds are fitted from the median and MAD of reference normals (with blur_floor_percentile
from config). A QC failure always returns verdict='RECAPTURE', never 'REJECT'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


def compute_robust_median_mad(values: list[float] | np.ndarray, min_mad: float = 1e-3) -> tuple[float, float]:
    """Compute (median, scaled_MAD) with floor `min_mad`."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, float(min_mad)
    med = float(np.median(arr))
    mad = float(1.4826 * np.median(np.abs(arr - med)))
    return med, max(mad, float(min_mad))


def laplacian_variance(gray: np.ndarray) -> float:
    """Compute discrete 3x3 Laplacian variance on a 2D grayscale image [0..255]."""
    g = np.asarray(gray, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    center = g[1:-1, 1:-1]
    lap = (
        g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.var(lap))


def extract_qc_metrics(image: Image.Image) -> dict[str, float]:
    """Extract blur (Laplacian variance), exposure (mean), contrast (std), and framing coverage."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    blur = laplacian_variance(gray)
    exposure = float(np.mean(gray))
    contrast = float(np.std(gray))
    # Framing coverage: fraction of non-border pixels that are not saturated/clipped (2 < pixel < 253)
    valid_mask = (gray > 2.0) & (gray < 253.0)
    framing = float(np.mean(valid_mask)) if valid_mask.size > 0 else 1.0
    return {
        "blur": blur,
        "exposure": exposure,
        "contrast": contrast,
        "framing": framing,
    }


@dataclass
class QCResult:
    """Result of Stage 0 Quality Check."""

    passed: bool
    verdict: str  # "PASS" or "RECAPTURE" (never "REJECT")
    reasons: list[str]
    metrics: dict[str, float]


class QualityChecker:
    """Stage 0 quality gate calibrated from reference normal images."""

    def __init__(
        self,
        blur_floor_percentile: float = 5.0,
        exposure_mad_multiplier: float = 4.0,
        contrast_mad_multiplier: float = 4.0,
        min_framing_coverage: float = 0.85,
        min_mad: float = 1e-3,
    ) -> None:
        self.blur_floor_percentile = float(blur_floor_percentile)
        self.exposure_mad_multiplier = float(exposure_mad_multiplier)
        self.contrast_mad_multiplier = float(contrast_mad_multiplier)
        self.min_framing_coverage = float(min_framing_coverage)
        self.min_mad = float(min_mad)

        self.blur_threshold: float = 1.0
        self.exposure_median: float = 128.0
        self.exposure_mad: float = 30.0
        self.contrast_median: float = 30.0
        self.contrast_mad: float = 10.0
        self._is_fitted: bool = False

    def fit(self, normals: list[Image.Image]) -> None:
        """Fit QC thresholds from the median, MAD, and blur floor percentile of reference normals."""
        if not normals:
            raise ValueError("Cannot fit QualityChecker on empty normals list.")

        metrics_list = [extract_qc_metrics(img) for img in normals]
        blurs = np.array([m["blur"] for m in metrics_list], dtype=np.float64)
        exposures = np.array([m["exposure"] for m in metrics_list], dtype=np.float64)
        contrasts = np.array([m["contrast"] for m in metrics_list], dtype=np.float64)

        blur_med, blur_mad = compute_robust_median_mad(blurs, min_mad=self.min_mad)
        pct_floor = float(np.percentile(blurs, self.blur_floor_percentile))
        # Allow a generous margin below the normal floor so minor variations pass while blurred captures fail
        self.blur_threshold = max(0.0, min(pct_floor * 0.5, blur_med - 4.0 * blur_mad))

        self.exposure_median, self.exposure_mad = compute_robust_median_mad(exposures, min_mad=max(self.min_mad, 2.0))
        self.contrast_median, self.contrast_mad = compute_robust_median_mad(contrasts, min_mad=max(self.min_mad, 1.0))
        self._is_fitted = True

    def check(self, image: Image.Image) -> QCResult:
        """Evaluate an image and return QCResult. Failures always return verdict='RECAPTURE'."""
        m = extract_qc_metrics(image)
        reasons: list[str] = []

        if m["blur"] < self.blur_threshold:
            reasons.append(f"blur_below_threshold ({m['blur']:.3f} < {self.blur_threshold:.3f})")

        exp_z = abs(m["exposure"] - self.exposure_median) / max(self.exposure_mad, self.min_mad)
        if exp_z > self.exposure_mad_multiplier and abs(m["exposure"] - self.exposure_median) > 45.0:
            reasons.append(f"exposure_out_of_range (z={exp_z:.2f})")

        con_z = (self.contrast_median - m["contrast"]) / max(self.contrast_mad, self.min_mad)
        if con_z > self.contrast_mad_multiplier and m["contrast"] < 2.0:
            reasons.append(f"low_contrast (z={con_z:.2f})")

        if m["framing"] < self.min_framing_coverage and (m["exposure"] < 5.0 or m["exposure"] > 250.0):
            reasons.append(f"framing_clipping ({m['framing']:.2f} < {self.min_framing_coverage:.2f})")

        passed = len(reasons) == 0
        verdict = "PASS" if passed else "RECAPTURE"
        return QCResult(passed=passed, verdict=verdict, reasons=reasons, metrics=m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur_threshold": self.blur_threshold,
            "exposure_median": self.exposure_median,
            "exposure_mad": self.exposure_mad,
            "contrast_median": self.contrast_median,
            "contrast_mad": self.contrast_mad,
        }
