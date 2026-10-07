"""Unit tests for offline temperature fitting."""

from pathlib import Path
import json
import sys

import pytest
import torch
import torch.nn.functional as functional

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from scripts.fit_temperature import (  # noqa: E402
    fit_temperature,
    load_records,
)


def test_temperature_fitting_reduces_calibration_nll() -> None:
    """
    Temperature scaling should reduce NLL for overconfident imperfect logits.

    Two labels disagree with their logits, so a larger temperature should
    soften probabilities and improve calibration loss.
    """
    defect_logits = torch.tensor([-4.0, -1.0, 1.0, 4.0])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    uncalibrated_nll = functional.binary_cross_entropy_with_logits(
        defect_logits,
        labels,
    ).item()

    temperature, bias, calibrated_nll = fit_temperature(defect_logits, labels)

    assert temperature > 0.0
    assert torch.isfinite(torch.tensor(bias))
    assert calibrated_nll < uncalibrated_nll


def test_temperature_fitting_returns_finite_values() -> None:
    """The learned temperature and resulting loss must be finite."""
    defect_logits = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    temperature, bias, nll = fit_temperature(defect_logits, labels)

    assert torch.isfinite(torch.tensor(temperature))
    assert torch.isfinite(torch.tensor(bias))
    assert torch.isfinite(torch.tensor(nll))


def test_intercept_prevents_temperature_from_collapsing_for_shifted_logits() -> None:
    """A non-zero class prevalence needs an intercept, not a capped temperature."""
    defect_logits = torch.tensor([-0.014, -0.012, -0.010, -0.008, -0.006])
    labels = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])

    temperature, bias, nll = fit_temperature(defect_logits, labels)
    probabilities = torch.sigmoid(defect_logits / temperature + bias)

    assert temperature > 0.0
    assert bias > 0.0
    assert nll < 0.7
    assert probabilities.mean() > 0.5


def test_calibration_manifest_requires_both_labels(tmp_path: Path) -> None:
    """A calibration set containing only one class is invalid."""
    manifest_path = tmp_path / "only_good.json"

    manifest_path.write_text(
        json.dumps(
            [
                {
                    "image_id": "bottle/test/good/000.png",
                    "label": 0,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Calibration manifest must contain both labels 0 and 1\.",
    ):
        load_records(manifest_path)


def test_calibration_manifest_accepts_good_and_defective_labels(
    tmp_path: Path,
) -> None:
    """A calibration set with binary labels loads successfully."""
    manifest_path = tmp_path / "valid.json"

    records = [
        {
            "image_id": "bottle/test/good/000.png",
            "label": 0,
        },
        {
            "image_id": "bottle/test/broken_large/000.png",
            "label": 1,
        },
    ]

    manifest_path.write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    assert load_records(manifest_path) == records
