"""Unit tests for the WinCLIP-style prompt bank."""

from pathlib import Path
import sys

import pytest
import torch

# Allow tests to import backend/app when pytest runs from the project root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.clip_encoder import EMBEDDING_DIM  # noqa: E402
from app.core.prompts import PromptBank  # noqa: E402


class FakeEncoder:
    """Fake text encoder that avoids downloading or running CLIP."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        self.calls.append(prompts)

        embeddings = []

        for prompt in prompts:
            vector = torch.zeros(EMBEDDING_DIM)

            if "flawless" in prompt or "undamaged" in prompt:
                vector[0] = 1.0
            else:
                vector[1] = 1.0

            embeddings.append(vector)

        return torch.stack(embeddings)


class FailingEncoder:
    """Used to prove that disk-cached embeddings avoid re-encoding."""

    device = torch.device("cpu")

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        raise AssertionError("Prompt embeddings should have loaded from cache.")


def test_prompt_bank_creates_normal_and_anomalous_embeddings(
    tmp_path: Path,
) -> None:
    encoder = FakeEncoder()
    prompt_bank = PromptBank(encoder, cache_dir=tmp_path)

    embeddings = prompt_bank.get("bottle")

    assert embeddings.normal.shape == (EMBEDDING_DIM,)
    assert embeddings.anomalous.shape == (EMBEDDING_DIM,)
    assert torch.allclose(embeddings.normal.norm(), torch.tensor(1.0))
    assert torch.allclose(embeddings.anomalous.norm(), torch.tensor(1.0))

    assert embeddings.normal[0] == 1.0
    assert embeddings.anomalous[1] == 1.0

    assert len(encoder.calls) == 2
    assert len(encoder.calls[0]) == 2
    assert len(encoder.calls[1]) == 2


def test_prompt_bank_uses_human_readable_category_names(
    tmp_path: Path,
) -> None:
    encoder = FakeEncoder()
    prompt_bank = PromptBank(encoder, cache_dir=tmp_path)

    prompt_bank.get("metal_nut")

    all_prompts = [prompt for call in encoder.calls for prompt in call]

    assert any("metal nut" in prompt for prompt in all_prompts)
    assert not any("metal_nut" in prompt for prompt in all_prompts)


def test_prompt_bank_loads_embeddings_from_disk_cache(
    tmp_path: Path,
) -> None:
    first_encoder = FakeEncoder()
    first_prompt_bank = PromptBank(first_encoder, cache_dir=tmp_path)

    first_embeddings = first_prompt_bank.get("bottle")

    cached_tensor_files = list(tmp_path.glob("*.pt"))
    cached_metadata_files = list(tmp_path.glob("*.json"))

    assert len(cached_tensor_files) == 1
    assert len(cached_metadata_files) == 1

    second_prompt_bank = PromptBank(
        FailingEncoder(),
        cache_dir=tmp_path,
    )
    cached_embeddings = second_prompt_bank.get("bottle")

    assert torch.equal(cached_embeddings.normal, first_embeddings.normal)
    assert torch.equal(
        cached_embeddings.anomalous,
        first_embeddings.anomalous,
    )


def test_prompt_bank_rejects_unknown_category(tmp_path: Path) -> None:
    prompt_bank = PromptBank(FakeEncoder(), cache_dir=tmp_path)

    with pytest.raises(ValueError, match="Unknown MVTec category"):
        prompt_bank.get("unknown_product")