"""Unit tests for the frozen CLIP encoder."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from PIL import Image

# Allow tests to import backend/app when pytest runs from the project root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.clip_encoder import (  # noqa: E402
    EMBEDDING_DIM,
    PATCH_COUNT,
    CLIPEncoder,
)


class FakeVisual(torch.nn.Module):
    """Small stand-in for CLIP's visual projection layer."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Parameter(torch.eye(EMBEDDING_DIM))


class FakeCLIPModel(torch.nn.Module):
    """Small CLIP replacement so tests never download a model."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = FakeVisual()
        self.test_parameter = torch.nn.Parameter(torch.ones(1))

    def forward_intermediates(self, image: torch.Tensor, **_: object) -> dict:
        batch_size = image.shape[0]

        return {
            "image_features": torch.ones(batch_size, EMBEDDING_DIM),
            "image_intermediates": [
                torch.ones(batch_size, PATCH_COUNT, EMBEDDING_DIM)
            ],
        }

    def encode_text(self, tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        embeddings = torch.ones(tokens.shape[0], EMBEDDING_DIM)

        if normalize:
            return torch.nn.functional.normalize(embeddings, dim=-1)

        return embeddings


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch) -> CLIPEncoder:
    """Create an encoder using the fake model instead of downloading CLIP."""

    fake_model = FakeCLIPModel()

    def fake_create_model_and_transforms(*_: object, **__: object) -> tuple:
        preprocess = lambda image: torch.zeros(3, 224, 224)
        return fake_model, None, preprocess

    def fake_tokenizer(prompts: list[str]) -> torch.Tensor:
        return torch.zeros(len(prompts), 77, dtype=torch.long)

    monkeypatch.setattr(
        "app.core.clip_encoder.open_clip.create_model_and_transforms",
        fake_create_model_and_transforms,
    )
    monkeypatch.setattr(
        "app.core.clip_encoder.open_clip.get_tokenizer",
        lambda _: fake_tokenizer,
    )

    return CLIPEncoder()


def test_encoder_freezes_all_parameters(encoder: CLIPEncoder) -> None:
    """The backbone must remain frozen for Phase 1."""
    assert encoder.model.training is False
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())


def test_prepare_image_returns_one_image_batch(encoder: CLIPEncoder) -> None:
    """A PIL image becomes a batched CLIP input tensor."""
    image_batch = encoder.prepare_image(Image.new("RGB", (100, 100)))

    assert image_batch.shape == (1, 3, 224, 224)


def test_encode_image_returns_expected_shapes(encoder: CLIPEncoder) -> None:
    """Global and local embeddings follow the scoring contract."""
    image_batch = torch.zeros(1, 3, 224, 224)

    embeddings = encoder.encode_image(image_batch)

    assert embeddings.global_embedding.shape == (1, EMBEDDING_DIM)
    assert embeddings.patch_embeddings.shape == (1, PATCH_COUNT, EMBEDDING_DIM)


def test_encode_image_rejects_non_batched_input(encoder: CLIPEncoder) -> None:
    """The encoder requires a four-dimensional image batch."""
    with pytest.raises(ValueError, match="Expected image_batch"):
        encoder.encode_image(torch.zeros(3, 224, 224))


def test_encode_text_returns_one_embedding_per_prompt(encoder: CLIPEncoder) -> None:
    """Each prompt receives one 512-dimensional text embedding."""
    embeddings = encoder.encode_text(
        ["a flawless bottle", "a damaged bottle"]
    )

    assert embeddings.shape == (2, EMBEDDING_DIM)