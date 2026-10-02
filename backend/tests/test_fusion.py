"""Unit tests for global-local similarity fusion."""

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
from app.core.fusion import ALPHA, fuse_scores  # noqa: E402
from app.core.prompts import PromptEmbeddings  # noqa: E402


def _make_embeddings() -> tuple[ImageEmbeddings, PromptEmbeddings]:
    """Create simple normalized embeddings with predictable similarities."""
    normal_vector = torch.zeros(EMBEDDING_DIM)
    normal_vector[0] = 1.0

    anomalous_vector = torch.zeros(EMBEDDING_DIM)
    anomalous_vector[1] = 1.0

    global_embedding = normal_vector.unsqueeze(0)

    patch_embeddings = normal_vector.repeat(PATCH_COUNT, 1)
    patch_embeddings[10] = anomalous_vector

    return (
        ImageEmbeddings(
            global_embedding=global_embedding,
            patch_embeddings=patch_embeddings.unsqueeze(0),
        ),
        PromptEmbeddings(
            normal=normal_vector,
            anomalous=anomalous_vector,
        ),
    )


def test_fusion_returns_expected_output_shapes() -> None:
    """Fusion returns one score and one 7x7 heatmap per image."""
    image_embeddings, prompt_embeddings = _make_embeddings()

    result = fuse_scores(image_embeddings, prompt_embeddings)

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
    assert result.heatmap.shape == (1, PATCH_GRID_SIZE, PATCH_GRID_SIZE)


def test_fusion_uses_global_and_local_similarity() -> None:
    """The fused score follows alpha * global + (1 - alpha) * local."""
    image_embeddings, prompt_embeddings = _make_embeddings()

    result = fuse_scores(image_embeddings, prompt_embeddings)

    assert torch.allclose(result.normal_global, torch.tensor([1.0]))
    assert torch.allclose(result.anomalous_global, torch.tensor([0.0]))

    # One patch exactly matches each prompt.
    assert torch.allclose(result.normal_local, torch.tensor([1.0]))
    assert torch.allclose(result.anomalous_local, torch.tensor([1.0]))

    expected_normal = ALPHA * 1.0 + (1 - ALPHA) * 1.0
    expected_anomalous = ALPHA * 0.0 + (1 - ALPHA) * 1.0

    assert torch.allclose(
        result.normal_fused,
        torch.tensor([expected_normal]),
    )
    assert torch.allclose(
        result.anomalous_fused,
        torch.tensor([expected_anomalous]),
    )
    assert torch.allclose(
        result.defect_logit,
        torch.tensor([expected_anomalous - expected_normal]),
    )


def test_heatmap_marks_anomalous_patch() -> None:
    """The anomalous patch has a positive anomalous-minus-normal heatmap value."""
    image_embeddings, prompt_embeddings = _make_embeddings()

    result = fuse_scores(image_embeddings, prompt_embeddings)

    anomalous_row = 10 // PATCH_GRID_SIZE
    anomalous_column = 10 % PATCH_GRID_SIZE

    assert result.heatmap[0, anomalous_row, anomalous_column] == 1.0
    assert result.heatmap[0, 0, 0] == -1.0


def test_fusion_rejects_invalid_patch_shape() -> None:
    """Fusion rejects patch tensors that are not [batch, 49, 512]."""
    image_embeddings, prompt_embeddings = _make_embeddings()

    invalid_image_embeddings = ImageEmbeddings(
        global_embedding=image_embeddings.global_embedding,
        patch_embeddings=torch.zeros(1, PATCH_COUNT - 1, EMBEDDING_DIM),
    )

    with pytest.raises(ValueError, match="Patch embeddings must have shape"):
        fuse_scores(invalid_image_embeddings, prompt_embeddings)