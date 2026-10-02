

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


ALPHA = 0.5


@dataclass(frozen=True)
class FusionResult:
  

    normal_global: torch.Tensor          # [batch]
    anomalous_global: torch.Tensor       # [batch]
    normal_local: torch.Tensor           # [batch]
    anomalous_local: torch.Tensor        # [batch]

    normal_patch_grid: torch.Tensor      # [batch, 7, 7]
    anomalous_patch_grid: torch.Tensor   # [batch, 7, 7]

    normal_fused: torch.Tensor           # [batch]
    anomalous_fused: torch.Tensor        # [batch]

    defect_logit: torch.Tensor           # [batch]
    heatmap: torch.Tensor                # [batch, 7, 7]


def _validate_embeddings(
    image_embeddings: ImageEmbeddings,
    prompt_embeddings: PromptEmbeddings,
) -> None:
    """Validate tensor shapes before similarity scoring."""
    global_embedding = image_embeddings.global_embedding
    patch_embeddings = image_embeddings.patch_embeddings

    if global_embedding.ndim != 2:
        raise ValueError("Global embedding must have shape [batch, 512].")

    if global_embedding.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Global embedding must have {EMBEDDING_DIM} dimensions."
        )

    if patch_embeddings.ndim != 3:
        raise ValueError(
            "Patch embeddings must have shape [batch, 49, 512]."
        )

    if patch_embeddings.shape[0] != global_embedding.shape[0]:
        raise ValueError("Global and patch embeddings must have the same batch size.")

    if patch_embeddings.shape[1:] != (PATCH_COUNT, EMBEDDING_DIM):
        raise ValueError(
            f"Patch embeddings must have shape [batch, {PATCH_COUNT}, "
            f"{EMBEDDING_DIM}]."
        )

    if prompt_embeddings.normal.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Normal prompt embedding must have shape [{EMBEDDING_DIM}]."
        )

    if prompt_embeddings.anomalous.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Anomalous prompt embedding must have shape [{EMBEDDING_DIM}]."
        )


def _global_similarity(
    global_embedding: torch.Tensor,
    prompt_embedding: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity between each full image and one text prompt."""
    return functional.cosine_similarity(
        global_embedding,
        prompt_embedding.unsqueeze(0),
        dim=-1,
    )


def _patch_similarities(
    patch_embeddings: torch.Tensor,
    prompt_embedding: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity between every patch and one text prompt."""
    return functional.cosine_similarity(
        patch_embeddings,
        prompt_embedding.view(1, 1, -1),
        dim=-1,
    )


def fuse_scores(
    image_embeddings: ImageEmbeddings,
    prompt_embeddings: PromptEmbeddings,
) -> FusionResult:

    _validate_embeddings(image_embeddings, prompt_embeddings)

    normal_global = _global_similarity(
        image_embeddings.global_embedding,
        prompt_embeddings.normal,
    )
    anomalous_global = _global_similarity(
        image_embeddings.global_embedding,
        prompt_embeddings.anomalous,
    )

    normal_patch_scores = _patch_similarities(
        image_embeddings.patch_embeddings,
        prompt_embeddings.normal,
    )
    anomalous_patch_scores = _patch_similarities(
        image_embeddings.patch_embeddings,
        prompt_embeddings.anomalous,
    )

    normal_local = normal_patch_scores.max(dim=1).values
    anomalous_local = anomalous_patch_scores.max(dim=1).values

    normal_fused = ALPHA * normal_global + (1 - ALPHA) * normal_local
    anomalous_fused = (
        ALPHA * anomalous_global + (1 - ALPHA) * anomalous_local
    )

    defect_logit = anomalous_fused - normal_fused

    normal_patch_grid = normal_patch_scores.reshape(
        -1,
        PATCH_GRID_SIZE,
        PATCH_GRID_SIZE,
    )
    anomalous_patch_grid = anomalous_patch_scores.reshape(
        -1,
        PATCH_GRID_SIZE,
        PATCH_GRID_SIZE,
    )


    heatmap = anomalous_patch_grid - normal_patch_grid

    return FusionResult(
        normal_global=normal_global,
        anomalous_global=anomalous_global,
        normal_local=normal_local,
        anomalous_local=anomalous_local,
        normal_patch_grid=normal_patch_grid,
        anomalous_patch_grid=anomalous_patch_grid,
        normal_fused=normal_fused,
        anomalous_fused=anomalous_fused,
        defect_logit=defect_logit,
        heatmap=heatmap,
    )