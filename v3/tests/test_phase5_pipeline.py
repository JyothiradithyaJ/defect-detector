"""Unit tests for Phase 5: determinism, S4 decision truth table, and Result contract validation."""

from __future__ import annotations

import numpy as np
import pytest

from v3.core.contracts import ROI, Mismatch, Recipe, Result
from v3.core.decide import decide_verdict
from v3.core.pipeline import TwoStagePipeline
from v3.core.report import validate_result_contract
from v3.eval.step4_stage_a_eval import build_validation_assembly_dataset


def test_decide_verdict_truth_table() -> None:
    """Verify every branch of README S4 decision logic (RECAPTURE, REJECT, REWORK, PASS)."""
    # 1. QC failed -> always RECAPTURE (even if hard mismatch is present)
    hard_mm = Mismatch(roi_id="r1", kind="presence", severity=0, hard=True, confidence=1.0, evidence={})
    out_qc = decide_verdict(qc_passed=False, mismatches=[hard_mm], image_score=99.0, tau=1.0)
    assert out_qc.verdict == "RECAPTURE"

    # 2. Hard mismatch (hard=True) -> REJECT
    out_hard = decide_verdict(qc_passed=True, mismatches=[hard_mm], image_score=0.1, tau=1.0)
    assert out_hard.verdict == "REJECT"

    # 3. Severity-0 mismatch (even if hard=False) -> REJECT
    sev0_mm = Mismatch(roi_id="r2", kind="order", severity=0, hard=False, confidence=0.9, evidence={})
    out_sev0 = decide_verdict(qc_passed=True, mismatches=[sev0_mm], image_score=0.1, tau=1.0)
    assert out_sev0.verdict == "REJECT"

    # 4. Defect inside non-reversible (hard=True) surface ROI -> REJECT
    recipe = Recipe(
        name="surf_test",
        template_path="",
        resolution=(32, 32),
        view="fixed",
        symmetry="none",
        stable_roi=[(0, 0), (32, 0), (32, 32), (0, 32)],
        rois=[ROI(id="crit_seal", kind="surface", polygon=[(4, 4), (20, 4), (20, 20), (4, 20)], severity=1, hard=True, params={})],
    )
    amap = np.zeros((32, 32), dtype=np.float32)
    amap[10, 10] = 8.0
    out_surf = decide_verdict(
        qc_passed=True,
        mismatches=[],
        image_score=0.5,
        tau=2.0,
        anomaly_map=amap,
        recipe=recipe,
        surface_pixel_threshold=5.0,
    )
    assert out_surf.verdict == "REJECT"

    # 5. Soft mismatches (severity 1 and 2, hard=False) -> REWORK with worst (minimum numeric) severity
    soft1 = Mismatch(roi_id="s1", kind="position", severity=2, hard=False, confidence=0.8, evidence={})
    soft2 = Mismatch(roi_id="s2", kind="count", severity=1, hard=False, confidence=0.8, evidence={})
    out_rework_mm = decide_verdict(qc_passed=True, mismatches=[soft1, soft2], image_score=0.2, tau=1.0)
    assert out_rework_mm.verdict == "REWORK"
    assert out_rework_mm.rework_level == 1

    # 6. No Stage A mismatch, but S_img > tau -> REWORK
    out_rework_s = decide_verdict(qc_passed=True, mismatches=[], image_score=2.5, tau=1.0)
    assert out_rework_s.verdict == "REWORK"
    assert out_rework_s.rework_level == 2

    # 7. QC passed, no mismatches, S_img <= tau -> PASS
    out_pass = decide_verdict(qc_passed=True, mismatches=[], image_score=0.4, tau=1.0)
    assert out_pass.verdict == "PASS"
    assert out_pass.rework_level is None


def test_result_contract_validation() -> None:
    """Verify validate_result_contract accepts valid Result and rejects malformed instances."""
    valid = Result(
        verdict="PASS",
        mismatches=[],
        image_score=0.123,
        anomaly_map=np.zeros((16, 16), dtype=np.float32),
        meta={"seed": 0},
    )
    validate_result_contract(valid)

    invalid_verdict = Result(
        verdict="UNKNOWN",  # type: ignore[arg-type]
        mismatches=[],
        image_score=0.0,
        anomaly_map=np.zeros((16, 16), dtype=np.float32),
        meta={},
    )
    with pytest.raises(ValueError, match="Invalid verdict"):
        validate_result_contract(invalid_verdict)


def test_end_to_end_pipeline_determinism() -> None:
    """Verify TwoStagePipeline produces identical scores and maps across repeated runs."""
    recipe, ref_normals, val_normals, _ = build_validation_assembly_dataset(seed=42)
    p1 = TwoStagePipeline()
    p1.fit(ref_normals[:4], recipe)
    r1 = p1.score(val_normals[0])

    p2 = TwoStagePipeline()
    p2.fit(ref_normals[:4], recipe)
    r2 = p2.score(val_normals[0])

    assert r1.verdict == r2.verdict
    assert r1.image_score == r2.image_score
    assert np.array_equal(r1.anomaly_map, r2.anomaly_map)

