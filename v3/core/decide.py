"""Stage 4: Decision rules (PASS, REWORK, REJECT, RECAPTURE) per README section S4.

Rules:
1. RECAPTURE if QC fails (never REJECT on a QC failure).
2. REJECT if any hard mismatch (m.hard is True), OR any severity-0 mismatch (m.severity == 0),
   OR a surface defect is detected inside a non-reversible surface ROI (roi.kind == 'surface' and roi.hard).
3. REWORK (with rework_level = worst/minimum numeric severity among mismatches, or default 2 if triggered
   by Stage B S_img > tau alone) if any mismatch is present OR S_img > tau.
4. Else PASS.

Threshold tau is fitted at the fixed false-reject rate (default 5%) on normal samples that are
disjoint from both reference bank normals and test negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from v3.core.contracts import Mismatch, Recipe
from v3.eval.splits import assert_disjoint_references_and_negatives


def fit_threshold_at_frr(
    val_normal_scores: list[float] | np.ndarray,
    false_reject_rate: float = 0.05,
    reference_ids: list[str] | None = None,
    val_normal_ids: list[str] | None = None,
    test_negative_ids: list[str] | None = None,
) -> float:
    """Fit threshold tau at fixed `false_reject_rate` on validation normals disjoint from references and test."""
    if reference_ids is not None and val_normal_ids is not None:
        assert_disjoint_references_and_negatives(reference_ids, val_normal_ids)
    if val_normal_ids is not None and test_negative_ids is not None:
        assert_disjoint_references_and_negatives(val_normal_ids, test_negative_ids)

    arr = np.asarray(val_normal_scores, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    q = float(np.clip(1.0 - float(false_reject_rate), 0.0, 1.0))
    return float(np.quantile(arr, q))


def check_non_reversible_surface_rois(
    anomaly_map: np.ndarray,
    recipe: Recipe | None,
    pixel_threshold: float,
) -> list[Mismatch]:
    """Check whether `anomaly_map` exceeds `pixel_threshold` inside any non-reversible (`hard=True`) surface ROI."""
    if recipe is None or anomaly_map.size == 0:
        return []

    h, w = anomaly_map.shape[:2]
    surface_mismatches: list[Mismatch] = []

    for roi in recipe.rois:
        if roi.kind != "surface":
            continue
        mask_img = Image.new("L", (w, h), 0)
        if len(roi.polygon) >= 3:
            ImageDraw.Draw(mask_img).polygon([(int(x), int(y)) for x, y in roi.polygon], outline=1, fill=1)
            mask = np.asarray(mask_img, dtype=np.uint8) > 0
        else:
            mask = np.ones((h, w), dtype=bool)

        if not np.any(mask):
            continue
        roi_peak = float(np.max(anomaly_map[mask]))
        if roi_peak > pixel_threshold:
            surface_mismatches.append(
                Mismatch(
                    roi_id=roi.id,
                    kind="surface",
                    severity=roi.severity,
                    hard=roi.hard,
                    confidence=float(min(1.0, roi_peak / max(pixel_threshold * 1.5, 1e-6))),
                    evidence={"roi_peak": round(roi_peak, 5), "pixel_threshold": round(pixel_threshold, 5)},
                )
            )
    return surface_mismatches


@dataclass(frozen=True)
class DecisionOutcome:
    """Structured Stage 4 decision output."""

    verdict: str             # "PASS", "REWORK", "REJECT", or "RECAPTURE"
    rework_level: int | None
    reason: str


def decide_verdict(
    qc_passed: bool,
    mismatches: list[Mismatch],
    image_score: float,
    tau: float,
    anomaly_map: np.ndarray | None = None,
    recipe: Recipe | None = None,
    surface_pixel_threshold: float | None = None,
) -> DecisionOutcome:
    """Apply Stage 4 decision hierarchy per README section S4."""
    # Rule 1: QC failure -> RECAPTURE (never REJECT)
    if not qc_passed:
        return DecisionOutcome(verdict="RECAPTURE", rework_level=None, reason="qc_failed")

    all_mismatches = list(mismatches)
    if anomaly_map is not None and recipe is not None and surface_pixel_threshold is not None:
        surf_mms = check_non_reversible_surface_rois(anomaly_map, recipe, surface_pixel_threshold)
        all_mismatches.extend(surf_mms)

    # Rule 2: Any hard mismatch, severity-0 mismatch, or defect in non-reversible surface ROI -> REJECT
    for m in all_mismatches:
        if m.hard or m.severity == 0:
            return DecisionOutcome(
                verdict="REJECT",
                rework_level=0,
                reason=f"hard_or_sev0_mismatch:{m.roi_id}({m.kind})",
            )

    # Rule 3: Any soft mismatch (severity >= 1, hard == False) or S_img > tau -> REWORK
    if all_mismatches:
        worst_severity = int(min(m.severity for m in all_mismatches))
        return DecisionOutcome(
            verdict="REWORK",
            rework_level=worst_severity,
            reason=f"soft_mismatch_sev{worst_severity}",
        )

    if float(image_score) > float(tau):
        # Look up surface ROI severity from recipe if defined, else default to severity 2
        surf_severities = [r.severity for r in (recipe.rois if recipe else []) if r.kind == "surface"]
        level = int(min(surf_severities)) if surf_severities else 2
        return DecisionOutcome(
            verdict="REWORK",
            rework_level=level,
            reason=f"image_score_exceeded_tau ({image_score:.4f} > {tau:.4f})",
        )

    # Rule 4: Otherwise -> PASS
    return DecisionOutcome(verdict="PASS", rework_level=None, reason="within_normal_tolerances")
