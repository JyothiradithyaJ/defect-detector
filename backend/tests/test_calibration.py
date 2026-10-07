"""Unit tests for temperature-scaling calibration."""

from pathlib import Path
import sys

import pytest
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.calibration import (  # noqa: E402
    CalibrationConfig,
    calibrate_probability,
    load_calibration,
    save_calibration,
)


def test_temperature_one_applies_standard_sigmoid() -> None:
    """T=1 does not rescale the raw defect logit."""
    logits = torch.tensor([-1.0, 0.0, 1.0])

    probabilities = calibrate_probability(logits, temperature=1.0)

    assert torch.allclose(probabilities, torch.sigmoid(logits))


def test_higher_temperature_softens_probabilities() -> None:
    """A larger T moves probabilities closer to 0.5."""
    logit = torch.tensor([2.0])

    unscaled_probability = calibrate_probability(logit, temperature=1.0)
    softened_probability = calibrate_probability(logit, temperature=2.0)

    assert softened_probability < unscaled_probability
    assert softened_probability > 0.5


def test_bias_shifts_the_calibrated_base_rate() -> None:
    probabilities = calibrate_probability(
        torch.tensor([0.0]), temperature=1.0, bias=1.0
    )

    assert probabilities.item() > 0.5


def test_calibration_file_round_trip(tmp_path: Path) -> None:
    """A saved calibration configuration can be loaded unchanged."""
    calibration_path = tmp_path / "calibration.json"

    saved_config = CalibrationConfig(
        temperature=0.84,
        bias=-0.23,
        model_name="ViT-B-32-quickgelu",
        pretrained_checkpoint="openai",
        calibration_manifest_hash="test-manifest-hash",
        fitted_at="2026-09-30T12:00:00+00:00",
        calibration_nll=0.52,
    )

    save_calibration(saved_config, calibration_path)
    loaded_config = load_calibration(calibration_path)

    assert loaded_config == saved_config


def test_missing_calibration_file_uses_temperature_one(
    tmp_path: Path,
) -> None:
    """The application can run before offline temperature fitting."""
    config = load_calibration(tmp_path / "does_not_exist.json")

    assert config.temperature == 1.0


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
def test_invalid_temperature_is_rejected(temperature: float) -> None:
    """Temperature must be finite and strictly positive."""
    with pytest.raises(ValueError, match="Temperature must be"):
        calibrate_probability(torch.tensor([0.1]), temperature)
