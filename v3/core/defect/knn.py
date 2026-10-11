"""Global nearest-neighbor DINO patch scoring (knn_global)."""

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


class GlobalKNNScorer:
    """Global k-NN patch scorer comparing every query patch against all reference patches."""

    def __init__(
        self,
        extractor: DINOFeatureExtractor | None = None,
        agg_radius: int = 1,
        topk_frac: float = 0.001,
        chunk_size: int = 4096,
    ) -> None:
        self.extractor = extractor or DINOFeatureExtractor(feature_fn=default_synthetic_feature_fn)
        self.agg_radius = int(agg_radius)
        self.topk_frac = float(topk_frac)
        self.chunk_size = int(chunk_size)
        self.bank = ReferenceBank()
        self._flat_bank: np.ndarray | None = None  # [M, D]
        self._flat_bank_gpu: object | None = None

    def _sync_gpu_bank(self) -> None:
        self._flat_bank_gpu = None
        if self._flat_bank is not None and self._flat_bank.shape[0] >= 1000:
            import torch

            if torch.cuda.is_available():
                self._flat_bank_gpu = torch.from_numpy(np.ascontiguousarray(self._flat_bank)).to(
                    device="cuda", dtype=torch.float16
                )

    def fit_features(
        self,
        features_list: list[CachedDINOFeatures],
        reference_ids: list[str] | None = None,
        symmetry: str = "none",
    ) -> None:
        """Fit memory bank directly from cached DINO features."""
        self.bank.fit_from_features(features_list, reference_ids=reference_ids, symmetry=symmetry)
        assert self.bank.patch_bank is not None
        d = self.bank.patch_bank.shape[-1]
        self._flat_bank = self.bank.patch_bank.reshape(-1, d)
        self._sync_gpu_bank()

    def fit(self, normals: list[Image], recipe: Recipe) -> None:
        """Fit memory bank on normal reference images per Scorer protocol."""
        self.bank.fit_from_images(normals, self.extractor, recipe=recipe)
        assert self.bank.patch_bank is not None
        d = self.bank.patch_bank.shape[-1]
        self._flat_bank = self.bank.patch_bank.reshape(-1, d)
        self._sync_gpu_bank()

    def score_patch_grid(self, patch_tokens: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute (image_score, patch_anomaly_map [H_p, W_p]) from query patch tokens [H_p, W_p, D]."""
        if self._flat_bank is None:
            raise RuntimeError("GlobalKNNScorer must be fit before calling score.")

        q = l2_normalize_np(patch_tokens)
        hp, wp, d = q.shape
        q_flat = q.reshape(-1, d)  # [Q, D]

        if self._flat_bank_gpu is not None:
            import torch

            with torch.inference_mode():
                bank_t: torch.Tensor = self._flat_bank_gpu  # type: ignore[assignment]
                q_t = torch.from_numpy(np.ascontiguousarray(q_flat)).to(device=bank_t.device, dtype=bank_t.dtype)
                sims = q_t @ bank_t.T
                max_sim = sims.max(dim=1).values.float().cpu().numpy()
        else:
            m = self._flat_bank.shape[0]
            max_sim = np.full(q_flat.shape[0], -1.0, dtype=np.float32)
            for start in range(0, m, self.chunk_size):
                ref_chunk = self._flat_bank[start : start + self.chunk_size]  # [C, D]
                sims = q_flat @ ref_chunk.T  # [Q, C]
                chunk_max = np.max(sims, axis=1)
                max_sim = np.maximum(max_sim, chunk_max)

        raw_dist = (1.0 - max_sim).reshape(hp, wp).astype(np.float32)
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
            meta={"scorer": "knn_global", "agg_radius": self.agg_radius},
        )
