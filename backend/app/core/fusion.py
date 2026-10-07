"""Pure zero-shot CLIP language + local anomaly score fusion.

The Phase 1 detector is intentionally reference-free. Global and local CLIP
evidence are combined with deliberately tuned weights:
    GLOBAL_WEIGHT=0.95
    LOCAL_WEIGHT=0.05
Held-out calibration scoring showed that the global prompt contrast carries
the useful image-level ranking signal; the local term remains as a small,
robust complement and continues to provide the localization map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from app.core.clip_encoder import (
    EMBEDDING_DIM,
    PATCH_COUNT,
    PATCH_GRID_SIZE,
    ImageEmbeddings,
)
from app.core.prompts import PromptEmbeddings

GLOBAL_WEIGHT = 0.95
LOCAL_WEIGHT = 0.05
LOCAL_TOP_FRACTION = 0.10


def average_flip_heatmaps(
    original_heatmap: torch.Tensor,
    flipped_heatmap: torch.Tensor,
) -> torch.Tensor:
    """Average original and horizontally flipped heatmaps in one coordinate frame."""
    if original_heatmap.shape != flipped_heatmap.shape:
        raise ValueError("Flip heatmaps must have identical shapes.")
    if original_heatmap.ndim != 3:
        raise ValueError("Flip heatmaps must have shape [batch, height, width].")
    return 0.5 * (original_heatmap + torch.flip(flipped_heatmap, dims=(-1,)))


@dataclass(frozen=True)
class FusionResult:
    normal_global: torch.Tensor
    anomalous_global: torch.Tensor
    normal_local: torch.Tensor
    anomalous_local: torch.Tensor
    normal_patch_grid: torch.Tensor
    anomalous_patch_grid: torch.Tensor
    heatmap: torch.Tensor
    normal_fused: torch.Tensor
    anomalous_fused: torch.Tensor
    defect_logit: torch.Tensor


def _validate_embeddings(
    image_embeddings: ImageEmbeddings,
    prompt_embeddings: PromptEmbeddings,
) -> None:
    global_embedding = image_embeddings.global_embedding
    patches = image_embeddings.patch_embeddings
    if global_embedding.ndim != 2 or global_embedding.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Global embedding must have shape [batch, {EMBEDDING_DIM}]."
        )
    if patches.ndim != 3 or patches.shape[1:] != (PATCH_COUNT, EMBEDDING_DIM):
        raise ValueError(
            f"Patch embeddings must have shape "
            f"[batch, {PATCH_COUNT}, {EMBEDDING_DIM}]."
        )
    if patches.shape[0] != global_embedding.shape[0]:
        raise ValueError("Global and patch embeddings must have the same batch size.")
    if (
        prompt_embeddings.normal.shape != (EMBEDDING_DIM,)
        or prompt_embeddings.anomalous.shape != (EMBEDDING_DIM,)
    ):
        raise ValueError("Prompt embeddings must each have shape [512].")


def _global_similarity(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return functional.cosine_similarity(x, p.unsqueeze(0), dim=-1)


def _patch_similarity(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return functional.cosine_similarity(x, p.view(1, 1, -1), dim=-1)


def _robust_topk(values: torch.Tensor) -> torch.Tensor:
    count = max(1, int(values.shape[1] * LOCAL_TOP_FRACTION))
    return values.topk(count, dim=1).values.mean(dim=1)


def fuse_scores(
    image_embeddings: ImageEmbeddings,
    prompt_embeddings: PromptEmbeddings,
    global_weight: float = GLOBAL_WEIGHT,
    local_weight: float = LOCAL_WEIGHT,
) -> FusionResult:
    """Fuse pure zero-shot CLIP evidence into one anomaly logit.

    GLOBAL_WEIGHT=0.95 and LOCAL_WEIGHT=0.05 are selected on the held-out
    calibration split. The global prompt contrast drives image-level ranking;
    the local term remains for a robust complementary signal and localization.
    No category-specific reference memory is used in Phase 1.
    """
    _validate_embeddings(image_embeddings, prompt_embeddings)
    if min(global_weight, local_weight) < 0:
        raise ValueError("Fusion weights must be non-negative.")

    normal_global = _global_similarity(
        image_embeddings.global_embedding,
        prompt_embeddings.normal,
    )
    anomalous_global = _global_similarity(
        image_embeddings.global_embedding,
        prompt_embeddings.anomalous,
    )

    layers = image_embeddings.patch_embeddings_by_layer or (
        image_embeddings.patch_embeddings,
    )
    normal_maps, anomalous_maps = [], []
    for patches in layers:
        normal_maps.append(_patch_similarity(patches, prompt_embeddings.normal))
        anomalous_maps.append(
            _patch_similarity(patches, prompt_embeddings.anomalous)
        )

    normal_patch_scores = torch.stack(normal_maps).mean(dim=0)
    anomalous_patch_scores = torch.stack(anomalous_maps).mean(dim=0)
    normal_local = _robust_topk(normal_patch_scores)
    anomalous_local = _robust_topk(anomalous_patch_scores)

    language_score = (
        global_weight * (anomalous_global - normal_global)
        + local_weight * (anomalous_local - normal_local)
    )
    defect_logit = language_score

    heatmap = anomalous_patch_scores - normal_patch_scores
    normal_fused = normal_global + normal_local
    anomalous_fused = normal_fused + defect_logit

    return FusionResult(
        normal_global=normal_global,
        anomalous_global=anomalous_global,
        normal_local=normal_local,
        anomalous_local=anomalous_local,
        normal_patch_grid=normal_patch_scores.reshape(
            -1, PATCH_GRID_SIZE, PATCH_GRID_SIZE
        ),
        anomalous_patch_grid=anomalous_patch_scores.reshape(
            -1, PATCH_GRID_SIZE, PATCH_GRID_SIZE
        ),
        heatmap=heatmap.reshape(-1, PATCH_GRID_SIZE, PATCH_GRID_SIZE),
        normal_fused=normal_fused,
        anomalous_fused=anomalous_fused,
        defect_logit=defect_logit,
    )
