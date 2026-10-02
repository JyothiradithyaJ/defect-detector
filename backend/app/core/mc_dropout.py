

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


MC_DROPOUT_PASSES = 10


@dataclass(frozen=True)
class UncertaintyResult:
   

    mean_probability: torch.Tensor
    variance: torch.Tensor
    predictive_entropy: torch.Tensor
    probabilities: torch.Tensor  # [passes, batch]


class DropoutScoringHead(nn.Module):
    

    def __init__(
        self,
        embedding_dim: int = 512,
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()

        if not 0.0 < dropout_probability < 1.0:
            raise ValueError("dropout_probability must be between 0 and 1.")

        self.layers = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(p=dropout_probability),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        
        return self.layers(embeddings).squeeze(-1)


def _set_only_dropout_to_train(module: nn.Module) -> None:
    
    module.eval()

    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.train()


@torch.inference_mode()
def estimate_uncertainty(
    scoring_head: nn.Module,
    image_embeddings: torch.Tensor,
    passes: int = MC_DROPOUT_PASSES,
) -> UncertaintyResult:
   
    if passes < 2:
        raise ValueError("passes must be at least 2.")

    _set_only_dropout_to_train(scoring_head)

    probability_passes = []

    for _ in range(passes):
        logits = scoring_head(image_embeddings)
        probabilities = torch.sigmoid(logits)
        probability_passes.append(probabilities)

    all_probabilities = torch.stack(probability_passes, dim=0)
    mean_probability = all_probabilities.mean(dim=0)
    variance = all_probabilities.var(dim=0, unbiased=False)

    # Entropy of the mean Bernoulli probability, not mean pass entropy.
    epsilon = torch.finfo(mean_probability.dtype).eps
    safe_probability = mean_probability.clamp(epsilon, 1.0 - epsilon)

    predictive_entropy = -(
        safe_probability * safe_probability.log()
        + (1.0 - safe_probability) * (1.0 - safe_probability).log()
    )

    return UncertaintyResult(
        mean_probability=mean_probability,
        variance=variance,
        predictive_entropy=predictive_entropy,
        probabilities=all_probabilities,
    )