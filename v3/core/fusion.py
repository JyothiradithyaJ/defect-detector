"""Stage B3: Robust z-score normalisation and weighted fusion of DINO and CLIP branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v3.core.qc import compute_robust_median_mad


@dataclass
class RobustNormalizer:
    """Robust median/MAD normalizer fitted on normal reference or calibration scores."""

    min_mad: float = 1e-3
    median: float = 0.0
    mad: float = 1.0
    fitted: bool = False

    def fit(self, scores: list[float] | np.ndarray) -> None:
        self.median, self.mad = compute_robust_median_mad(scores, min_mad=self.min_mad)
        self.fitted = True

    def transform_scalar(self, score: float) -> float:
        if not self.fitted:
            return float(score)
        return float(( float(score) - self.median ) / max(self.mad, self.min_mad))

    def transform_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        arr = np.asarray(anomaly_map, dtype=np.float32)
        if not self.fitted:
            return arr
        return ((arr - float(self.median)) / max(float(self.mad), self.min_mad)).astype(np.float32)


class BranchFuser:
    """Weighted fusion of DINOv2 (w_d) and CLIP (w_c) normalized scores and anomaly maps."""

    def __init__(self, w_d: float = 1.0, w_c: float = 0.0, min_mad: float = 1e-3) -> None:
        self.w_d = float(w_d)
        self.w_c = float(w_c)
        self.min_mad = float(min_mad)
        self.dino_img_norm = RobustNormalizer(min_mad=min_mad)
        self.clip_img_norm = RobustNormalizer(min_mad=min_mad)
        self.dino_map_norm = RobustNormalizer(min_mad=min_mad)
        self.clip_map_norm = RobustNormalizer(min_mad=min_mad)

    def fit(
        self,
        dino_normal_scores: list[float],
        clip_normal_scores: list[float] | None = None,
        dino_normal_maps: list[np.ndarray] | None = None,
        clip_normal_maps: list[np.ndarray] | None = None,
    ) -> None:
        """Fit robust normalizers on reference/validation normal scores and maps."""
        if dino_normal_scores:
            self.dino_img_norm.fit(dino_normal_scores)
        if clip_normal_scores:
            self.clip_img_norm.fit(clip_normal_scores)
        if dino_normal_maps:
            flat_d = np.concatenate([m.ravel() for m in dino_normal_maps])
            self.dino_map_norm.fit(flat_d)
        if clip_normal_maps:
            flat_c = np.concatenate([m.ravel() for m in clip_normal_maps])
            self.clip_map_norm.fit(flat_c)

    def fuse_score(self, dino_score: float, clip_score: float = 0.0) -> float:
        """Compute fused image anomaly score: w_d * z_dino + w_c * z_clip."""
        zd = self.dino_img_norm.transform_scalar(dino_score)
        if self.w_c == 0.0:
            return float(self.w_d * zd)
        zc = self.clip_img_norm.transform_scalar(clip_score)
        return float(self.w_d * zd + self.w_c * zc)

    def fuse_map(self, dino_map: np.ndarray, clip_map: np.ndarray | None = None) -> np.ndarray:
        """Compute fused pixel anomaly map: w_d * z_map_dino + w_c * z_map_clip."""
        md = self.dino_map_norm.transform_map(dino_map)
        if self.w_c == 0.0 or clip_map is None:
            return (self.w_d * md).astype(np.float32)
        mc = self.clip_map_norm.transform_map(clip_map)
        return (self.w_d * md + self.w_c * mc).astype(np.float32)
