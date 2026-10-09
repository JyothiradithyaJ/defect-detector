"""Core data contracts and interfaces for v3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from PIL.Image import Image


@dataclass(frozen=True)
class ROI:
    """Region of interest specification in golden template coordinates."""

    id: str
    kind: Literal["presence", "count", "order", "position", "surface"]
    polygon: list[tuple[int, int]]
    severity: Literal[0, 1, 2]
    hard: bool
    params: dict[str, Any]
    z_threshold: float | None = None


@dataclass(frozen=True)
class Recipe:
    """Recipe describing golden template, expected resolution, symmetry, and ROIs."""

    name: str
    template_path: str
    resolution: tuple[int, int]
    view: Literal["fixed", "free"]
    symmetry: Literal["none", "flip", "rot"]
    stable_roi: list[tuple[int, int]]
    rois: list[ROI]


@dataclass
class Mismatch:
    """Stage A mismatch finding for a specific ROI."""

    roi_id: str
    kind: str
    severity: int
    hard: bool
    confidence: float
    evidence: dict[str, Any]


@dataclass
class Result:
    """Inspection result combining Stage A mismatches and Stage B defect scoring."""

    verdict: Literal["PASS", "REWORK", "REJECT", "RECAPTURE"]
    mismatches: list[Mismatch]
    image_score: float
    anomaly_map: np.ndarray
    meta: dict[str, Any]


class Scorer(Protocol):
    """Protocol that evaluation harness depends on."""

    def fit(self, normals: list[Image], recipe: Recipe) -> None: ...

    def score(self, image: Image) -> Result: ...

