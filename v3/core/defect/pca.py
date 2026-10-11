"""Subspace PCA residual scoring on normal patch features (pca_residual)."""

from __future__ import annotations

import numpy as np
from PIL.Image import Image

from v3.core.bank import (
    CachedDINOFeatures,
    DINOFeatureExtractor,
    ReferenceBank,
    default_synthetic_feature_fn,
    l2_normalize_np,
)
from v3.core.contracts import Recipe, Result
from v3.core.defect.aggregate import (
    aggregate_patch_map,
    topk_image_score,
    upsample_patch_map,
)


class PCAResidualScorer:
    """PCA subspace residual scorer on normal reference patch features."""

    def __init__(
        self,
        extractor: DINOFeatureExtractor | None = None,
        variance_retained: float = 0.95,
        max_components: int = 128,
        agg_radius: int = 1,
        topk_frac: float = 0.001,
    ) -> None:
        self.extractor = extractor or DINOFeatureExtractor(feature_fn=default_synthetic_feature_fn)
        self.variance_retained = float(variance_retained)
        self.max_components = int(max_components)
        self.agg_radius = int(agg_radius)
        self.topk_frac = float(topk_frac)
        self.bank = ReferenceBank()
        self._mean: np.ndarray | None = None        # [1, D]
        self._components: np.ndarray | None = None  # [D, K]

    def _fit_subspace(self) -> None:
        assert self.bank.patch_bank is not None
        d = self.bank.patch_bank.shape[-1]
        flat = self.bank.patch_bank.reshape(-1, d).astype(np.float32)

        # Subsample if very large to keep SVD fast and deterministic
        if flat.shape[0] > 8192:
            rng = np.random.default_rng(42)
            idx = rng.choice(flat.shape[0], size=8192, replace=False)
            flat = flat[idx]

        mean = np.mean(flat, axis=0, keepdims=True)
        centered = flat - mean
        cov = centered.T @ centered  # [D, D]
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        order = np.argsort(eigvals)[::-1]
        var = np.maximum(eigvals[order], 0.0)
        vecs = eigvecs[:, order]
        total_var = float(np.sum(var))
        if total_var <= 1e-12:
            k = 1
        else:
            cum_var = np.cumsum(var) / total_var
            k = int(np.searchsorted(cum_var, self.variance_retained) + 1)
            k = max(1, min(k, self.max_components, vecs.shape[1]))

        self._mean = mean.astype(np.float32)
        self._components = vecs[:, :k].astype(np.float32)  # [D, K]

    def fit_features(
        self,
        features_list: list[CachedDINOFeatures],
        reference_ids: list[str] | None = None,
        symmetry: str = "none",
    ) -> None:
        """Fit PCA subspace from cached DINO features."""
        self.bank.fit_from_features(features_list, reference_ids=reference_ids, symmetry=symmetry)
        self._fit_subspace()

    def fit(self, normals: list[Image], recipe: Recipe) -> None:
        """Fit PCA subspace on normal reference images per Scorer protocol."""
        self.bank.fit_from_images(normals, self.extractor, recipe=recipe)
        self._fit_subspace()

    def score_patch_grid(self, patch_tokens: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute (image_score, patch_anomaly_map [H_p, W_p]) from PCA residual norm."""
        if self._mean is None or self._components is None:
            raise RuntimeError("PCAResidualScorer must be fit before calling score.")

        q = l2_normalize_np(patch_tokens)
        hp, wp, d = q.shape
        centered = q.reshape(-1, d) - self._mean  # [Q, D]
        coords = centered @ self._components      # [Q, K]
        total_sq = np.sum(centered * centered, axis=-1)
        proj_sq = np.sum(coords * coords, axis=-1)
        res_energy = np.maximum(total_sq - proj_sq, 0.0).reshape(hp, wp).astype(np.float32)

        smoothed = aggregate_patch_map(res_energy, radius=self.agg_radius)
        img_score = topk_image_score(smoothed, topk_frac=self.topk_frac)
        return img_score, smoothed

    def score(self, image: Image) -> Result:
        """Score a PIL image and return a Result adhering to v3.core.contracts.Scorer."""
        feats = self.extractor.extract_pil(image)
        img_score, patch_map = self.score_patch_grid(feats.patch_tokens)
        w, h = image.size
        full_map = upsample_patch_map(patch_map, (h, w))
        return Result(
            verdict="PASS",
            mismatches=[],
            image_score=img_score,
            anomaly_map=full_map,
            meta={"scorer": "pca_residual", "agg_radius": self.agg_radius},
        )
