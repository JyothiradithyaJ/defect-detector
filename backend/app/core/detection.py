"""Typed assembly of calibrated detection and MC Dropout uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from app.core.calibration import calibrate_probability
from app.core.fusion import FusionResult
from app.core.mc_dropout import DropoutScoringHead, estimate_uncertainty


@dataclass(frozen=True)
class DetectionResult:
    """Complete model output for one or more images."""

    defect_logit: torch.Tensor
    defect_probability: torch.Tensor
    mc_mean_probability: torch.Tensor
    mc_variance: torch.Tensor
    mc_predictive_entropy: torch.Tensor
    heatmap: torch.Tensor
    normal_global: torch.Tensor
    anomalous_global: torch.Tensor
    normal_local: torch.Tensor
    anomalous_local: torch.Tensor


def assemble_detection_result(
    fusion: FusionResult,
    temperature: float,
    scoring_head: DropoutScoringHead,
    image_embeddings: torch.Tensor,
    passes: int = 10,
) -> DetectionResult:
    """Assemble calibrated language scoring and existing MC Dropout metadata."""
    defect_probability = calibrate_probability(
        fusion.defect_logit,
        temperature,
    )
    uncertainty = estimate_uncertainty(
        scoring_head=scoring_head,
        image_embeddings=image_embeddings,
        passes=passes,
    )
    return DetectionResult(
        defect_logit=fusion.defect_logit,
        defect_probability=defect_probability,
        mc_mean_probability=uncertainty.mean_probability,
        mc_variance=uncertainty.variance,
        mc_predictive_entropy=uncertainty.predictive_entropy,
        heatmap=fusion.heatmap,
        normal_global=fusion.normal_global,
        anomalous_global=fusion.anomalous_global,
        normal_local=fusion.normal_local,
        anomalous_local=fusion.anomalous_local,
    )
