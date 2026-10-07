"""Calibration configuration and safe probability conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path

import torch

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_PATH = BACKEND_DIR / "config" / "calibration.json"


@dataclass(frozen=True)
class CalibrationConfig:
    temperature: float
    bias: float = 0.0
    model_name: str = ""
    pretrained_checkpoint: str = ""
    calibration_manifest_hash: str = ""
    fitted_at: str = ""
    prompt_cache_key: str = ""
    calibration_nll: float | None = None

    def validate(self) -> None:
        if not isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and > 0.")
        if not isfinite(self.bias):
            raise ValueError("Calibration bias must be finite.")


def load_calibration(path: Path = DEFAULT_CALIBRATION_PATH) -> CalibrationConfig:
    if not path.exists():
        return CalibrationConfig(temperature=1.0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = CalibrationConfig(**payload)
    config.validate()
    return config


def save_calibration(
    config: CalibrationConfig,
    path: Path = DEFAULT_CALIBRATION_PATH,
) -> None:
    config.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def calibrate_probability(
    defect_logit: torch.Tensor,
    temperature: float,
    bias: float = 0.0,
) -> torch.Tensor:
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and > 0.")
    if not isfinite(bias):
        raise ValueError("Calibration bias must be finite.")
    return torch.sigmoid(defect_logit / temperature + bias)
