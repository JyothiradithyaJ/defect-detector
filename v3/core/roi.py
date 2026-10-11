"""Stage A2: Recipe loader, leave-one-out robust z-score calibration, and ROI inspection engine."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from v3.core.contracts import ROI, Mismatch, Recipe
from v3.core.mismatch.count import estimate_blob_count_and_area, score_count
from v3.core.mismatch.order import compute_ordered_slot_signature, score_order
from v3.core.mismatch.position import compute_roi_centroid, score_position
from v3.core.mismatch.presence import compute_roi_descriptor, score_presence
from v3.core.qc import compute_robust_median_mad


def compute_recipe_file_hash(recipe_path: Path) -> str:
    """Compute SHA-256 hex digest of a recipe YAML file on disk."""
    return hashlib.sha256(recipe_path.read_bytes()).hexdigest()


def load_recipe_from_yaml(recipe_path: Path) -> tuple[Recipe, dict[str, Any]]:
    """Load a Recipe dataclass and raw metadata dictionary from a YAML file."""
    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
    rois: list[ROI] = []
    for r in raw.get("rois", []):
        poly = [(int(pt[0]), int(pt[1])) for pt in r.get("polygon", [])]
        rois.append(
            ROI(
                id=str(r["id"]),
                kind=r["kind"],
                polygon=poly,
                severity=int(r.get("severity", 1)),  # type: ignore[arg-type]
                hard=bool(r.get("hard", False)),
                params=dict(r.get("params", {})),
                z_threshold=float(r["z_threshold"]) if r.get("z_threshold") is not None else None,
            )
        )

    res_raw = raw.get("resolution", [1024, 1024])
    stable_poly = [(int(pt[0]), int(pt[1])) for pt in raw.get("stable_roi", [])]
    recipe = Recipe(
        name=str(raw.get("name", recipe_path.stem)),
        template_path=str(raw.get("template_path", "")),
        resolution=(int(res_raw[0]), int(res_raw[1])),
        view=raw.get("view", "fixed"),
        symmetry=raw.get("symmetry", "none"),
        stable_roi=stable_poly,
        rois=rois,
    )
    return recipe, raw


@dataclass
class ROICalibrationStats:
    """Leave-one-out normal calibration statistics for a single ROI."""

    roi_id: str
    kind: str
    median: float
    mad: float
    z_threshold: float
    ref_features: list[Any]


class StageAROIEngine:
    """Stage A2 ROI engine calibrating robust z-scores from leave-one-out normals."""

    def __init__(
        self,
        min_mad: float = 1e-3,
        default_z_threshold: float = 3.5,
        early_exit: bool = False,
    ) -> None:
        self.min_mad = float(min_mad)
        self.default_z_threshold = float(default_z_threshold)
        self.early_exit = bool(early_exit)
        self.roi_stats: dict[str, ROICalibrationStats] = {}

    def _extract_roi_feature(self, image: Image.Image, roi: ROI) -> Any:
        if roi.kind in ("presence", "surface"):
            return compute_roi_descriptor(image, roi)
        if roi.kind == "count":
            return estimate_blob_count_and_area(image, roi)
        if roi.kind == "order":
            return compute_ordered_slot_signature(image, roi)
        if roi.kind == "position":
            return compute_roi_centroid(image, roi)
        raise ValueError(f"Unsupported ROI kind: {roi.kind}")

    def _score_raw(self, image: Image.Image, roi: ROI, ref_features: list[Any]) -> float:
        if roi.kind in ("presence", "surface"):
            return score_presence(image, ref_features, roi)
        if roi.kind == "count":
            return score_count(image, ref_features, roi)
        if roi.kind == "order":
            return score_order(image, ref_features, roi)
        if roi.kind == "position":
            return score_position(image, ref_features, roi)
        raise ValueError(f"Unsupported ROI kind: {roi.kind}")

    def fit(self, normals: list[Image.Image], recipe: Recipe) -> None:
        """Fit per-ROI median and MAD from leave-one-out normal scores with `min_mad` floor."""
        if not normals:
            raise ValueError("Cannot fit StageAROIEngine on empty normals list.")

        self.roi_stats.clear()
        for roi in recipe.rois:
            feats = [self._extract_roi_feature(img, roi) for img in normals]
            loo_scores: list[float] = []
            for idx, img in enumerate(normals):
                other_feats = [f for j, f in enumerate(feats) if j != idx] or feats
                s = self._score_raw(img, roi, other_feats)
                loo_scores.append(s)

            med, mad = compute_robust_median_mad(loo_scores, min_mad=self.min_mad)
            z_thr = float(roi.z_threshold) if roi.z_threshold is not None else self.default_z_threshold
            self.roi_stats[roi.id] = ROICalibrationStats(
                roi_id=roi.id,
                kind=roi.kind,
                median=med,
                mad=mad,
                z_threshold=z_thr,
                ref_features=feats,
            )

    def inspect(
        self,
        aligned_image: Image.Image,
        recipe: Recipe,
        skip_positional: bool = False,
    ) -> tuple[list[Mismatch], dict[str, float]]:
        """Evaluate all ROIs on `aligned_image`. Returns (mismatches, per_roi_z_scores)."""
        mismatches: list[Mismatch] = []
        z_scores: dict[str, float] = {}

        for roi in recipe.rois:
            if roi.kind == "surface":
                continue
            if skip_positional and roi.kind in ("position", "order"):
                continue
            if roi.id not in self.roi_stats:
                continue

            st = self.roi_stats[roi.id]
            raw_val = self._score_raw(aligned_image, roi, st.ref_features)
            z = (raw_val - st.median) / max(st.mad, self.min_mad)
            z_scores[roi.id] = float(z)

            if z > st.z_threshold:
                conf = float(min(1.0, z / max(st.z_threshold * 2.0, 1e-6)))
                mm = Mismatch(
                    roi_id=roi.id,
                    kind=roi.kind,
                    severity=roi.severity,
                    hard=roi.hard,
                    confidence=conf,
                    evidence={
                        "raw_score": round(raw_val, 5),
                        "z_score": round(float(z), 3),
                        "z_threshold": st.z_threshold,
                        "loo_median": round(st.median, 5),
                        "loo_mad": round(st.mad, 5),
                    },
                )
                mismatches.append(mm)
                if self.early_exit and roi.hard:
                    break

        return mismatches, z_scores
