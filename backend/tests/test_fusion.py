"""Unit tests for pure zero-shot CLIP language fusion."""

from pathlib import Path
import sys

import pytest
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.clip_encoder import (  # noqa: E402
    EMBEDDING_DIM,
    PATCH_COUNT,
    PATCH_GRID_SIZE,
    ImageEmbeddings,
)
from app.core.fusion import GLOBAL_WEIGHT, LOCAL_WEIGHT, fuse_scores  # noqa: E402
from app.core.prompts import PromptEmbeddings  # noqa: E402


def _make_embeddings():
    normal = torch.zeros(EMBEDDING_DIM)
    normal[0] = 1.0
    anomalous = torch.zeros(EMBEDDING_DIM)
    anomalous[1] = 1.0
    global_embedding = normal.unsqueeze(0)
    patches = normal.repeat(PATCH_COUNT, 1)
    patches[10] = anomalous
    return (
        ImageEmbeddings(
            global_embedding=global_embedding,
            patch_embeddings=patches.unsqueeze(0),
        ),
        PromptEmbeddings(normal=normal, anomalous=anomalous),
    )


def test_fusion_returns_expected_output_shapes():
    image_embeddings, prompts = _make_embeddings()
    result = fuse_scores(image_embeddings, prompts)
    assert result.normal_global.shape == (1,)
    assert result.anomalous_global.shape == (1,)
    assert result.normal_local.shape == (1,)
    assert result.anomalous_local.shape == (1,)
    assert result.normal_patch_grid.shape == (
        1,
        PATCH_GRID_SIZE,
        PATCH_GRID_SIZE,
    )
    assert result.anomalous_patch_grid.shape == (
        1,
        PATCH_GRID_SIZE,
        PATCH_GRID_SIZE,
    )
    assert result.defect_logit.shape == (1,)
    assert result.heatmap.shape == (
        1,
        PATCH_GRID_SIZE,
        PATCH_GRID_SIZE,
    )


def test_fusion_uses_explicit_language_weights():
    image_embeddings, prompts = _make_embeddings()
    result = fuse_scores(image_embeddings, prompts)
    expected = GLOBAL_WEIGHT * (0.0 - 1.0) + LOCAL_WEIGHT * (1.0 - 1.0)
    assert torch.allclose(result.defect_logit, torch.tensor([expected]))
    assert torch.allclose(result.normal_fused, torch.tensor([2.0]))
    assert torch.allclose(
        result.anomalous_fused,
        torch.tensor([2.0 + expected]),
    )


def test_fusion_rejects_invalid_patch_shape():
    image_embeddings, prompts = _make_embeddings()
    invalid = ImageEmbeddings(
        global_embedding=image_embeddings.global_embedding,
        patch_embeddings=torch.zeros(
            1,
            PATCH_COUNT - 1,
            EMBEDDING_DIM,
        ),
    )
    with pytest.raises(ValueError, match="Patch embeddings must have shape"):
        fuse_scores(invalid, prompts)
