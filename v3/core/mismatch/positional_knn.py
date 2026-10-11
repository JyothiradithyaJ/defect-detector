"""Positional k-NN patch matching algorithm (knn_positional)."""

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


class PositionalKNNScorer:
    """Positional k-NN scorer matching query patch (r, c) against reference patches in [r±radius, c±radius]."""

    def __init__(
        self,
        extractor: DINOFeatureExtractor | None = None,
        radius: int = 2,
        agg_radius: int = 1,
        topk_frac: float = 0.001,
    ) -> None:
        self.extractor = extractor or DINOFeatureExtractor(feature_fn=default_synthetic_feature_fn)
        self.radius = int(radius)
        self.agg_radius = int(agg_radius)
        self.topk_frac = float(topk_frac)
        self.bank = ReferenceBank()

    def fit_features(
        self,
        features_list: list[CachedDINOFeatures],
        reference_ids: list[str] | None = None,
        symmetry: str = "none",
    ) -> None:
        """Fit positional memory bank directly from cached DINO features."""
        self.bank.fit_from_features(features_list, reference_ids=reference_ids, symmetry=symmetry)

    def fit(self, normals: list[Image], recipe: Recipe) -> None:
        """Fit positional memory bank on normal reference images per Scorer protocol."""
        self.bank.fit_from_images(normals, self.extractor, recipe=recipe)

    def score_patch_grid(self, patch_tokens: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute (image_score, patch_anomaly_map [H_p, W_p]) using local spatial window matching."""
        if self.bank.patch_bank is None:
            raise RuntimeError("PositionalKNNScorer must be fit before calling score.")

        q = l2_normalize_np(patch_tokens)  # [H_p, W_p, D]
        ref = self.bank.patch_bank         # [N_ref, H_p, W_p, D]
        hp, wp, _ = q.shape
        rad = self.radius

        # Vectorized shifted-window cosine similarity across (-rad..+rad, -rad..+rad)
        max_sim = np.full((hp, wp), -1.0, dtype=np.float32)
        for dr in range(-rad, rad + 1):
            r_q_start = max(0, -dr)
            r_q_end = min(hp, hp - dr)
            r_ref_start = r_q_start + dr
            r_ref_end = r_q_end + dr

            for dc in range(-rad, rad + 1):
                c_q_start = max(0, -dc)
                c_q_end = min(wp, wp - dc)
                c_ref_start = c_q_start + dc
                c_ref_end = c_q_end + dc

                q_sub = q[r_q_start:r_q_end, c_q_start:c_q_end, :]            # [h, w, D]
                ref_sub = ref[:, r_ref_start:r_ref_end, c_ref_start:c_ref_end, :]  # [N_ref, h, w, D]
                # Cosine similarity per spatial position across N_ref
                sims = np.einsum("hwd,nhwd->nhw", q_sub, ref_sub)
                best_n = np.max(sims, axis=0)
                max_sim[r_q_start:r_q_end, c_q_start:c_q_end] = np.maximum(
                    max_sim[r_q_start:r_q_end, c_q_start:c_q_end],
                    best_n,
                )

        raw_dist = (1.0 - max_sim).astype(np.float32)
        smoothed = aggregate_patch_map(raw_dist, radius=self.agg_radius)
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
            meta={"scorer": "knn_positional", "radius": self.radius, "agg_radius": self.agg_radius},
        )
