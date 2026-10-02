"""Unit tests for Monte Carlo Dropout uncertainty estimation."""

from pathlib import Path
import math
import sys

import pytest
import torch
from torch import nn

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.mc_dropout import (  # noqa: E402
    MC_DROPOUT_PASSES,
    DropoutScoringHead,
    estimate_uncertainty,
)


def test_dropout_head_returns_one_logit_per_image() -> None:
    """The lightweight head produces one defect logit for each embedding."""
    head = DropoutScoringHead(embedding_dim=512, dropout_probability=0.2)
    embeddings = torch.zeros(3, 512)

    logits = head(embeddings)

    assert logits.shape == (3,)


def test_mc_dropout_returns_expected_shapes_and_ranges() -> None:
    """Repeated stochastic passes produce valid probabilities and uncertainty."""
    torch.manual_seed(42)

    head = DropoutScoringHead(embedding_dim=512, dropout_probability=0.5)
    embeddings = torch.ones(2, 512)

    result = estimate_uncertainty(
        scoring_head=head,
        image_embeddings=embeddings,
        passes=MC_DROPOUT_PASSES,
    )

    assert result.probabilities.shape == (MC_DROPOUT_PASSES, 2)
    assert result.mean_probability.shape == (2,)
    assert result.variance.shape == (2,)
    assert result.predictive_entropy.shape == (2,)

    assert torch.all(result.probabilities >= 0.0)
    assert torch.all(result.probabilities <= 1.0)
    assert torch.all(result.variance >= 0.0)

    # Bernoulli predictive entropy is between 0 and ln(2).
    assert torch.all(result.predictive_entropy >= 0.0)
    assert torch.all(result.predictive_entropy <= math.log(2.0))


def test_mc_dropout_keeps_only_dropout_layers_in_train_mode() -> None:
    """LayerNorm and Linear stay in eval mode; Dropout stays stochastic."""
    head = DropoutScoringHead(embedding_dim=512, dropout_probability=0.5)
    embeddings = torch.ones(1, 512)

    estimate_uncertainty(
        scoring_head=head,
        image_embeddings=embeddings,
        passes=2,
    )

    dropout_layers = [
        module
        for module in head.modules()
        if isinstance(module, nn.Dropout)
    ]
    non_dropout_layers = [
        module
        for module in head.modules()
        if isinstance(module, (nn.LayerNorm, nn.Linear))
    ]

    assert all(layer.training for layer in dropout_layers)
    assert all(not layer.training for layer in non_dropout_layers)


def test_mc_dropout_produces_stochastic_predictions() -> None:
    """Active dropout should create variation across repeated passes."""
    torch.manual_seed(42)

    head = DropoutScoringHead(embedding_dim=512, dropout_probability=0.5)
    embeddings = torch.ones(1, 512)

    result = estimate_uncertainty(
        scoring_head=head,
        image_embeddings=embeddings,
        passes=10,
    )

    assert result.variance.item() > 0.0


def test_mc_dropout_requires_at_least_two_passes() -> None:
    """One pass cannot measure variance or predictive uncertainty."""
    head = DropoutScoringHead()

    with pytest.raises(ValueError, match="passes must be at least 2"):
        estimate_uncertainty(
            scoring_head=head,
            image_embeddings=torch.ones(1, 512),
            passes=1,
        )


@pytest.mark.parametrize("dropout_probability", [0.0, 1.0, -0.1])
def test_invalid_dropout_probability_is_rejected(
    dropout_probability: float,
) -> None:
    """Dropout probability must be strictly between zero and one."""
    with pytest.raises(ValueError, match="dropout_probability must be"):
        DropoutScoringHead(dropout_probability=dropout_probability)
        