"""Synthetic tests for Stage A mismatch detection (swap, presence, QC RECAPTURE)."""

from __future__ import annotations

from PIL import ImageFilter

from v3.core.qc import QualityChecker
from v3.core.roi import StageAROIEngine
from v3.eval.step4_stage_a_eval import build_validation_assembly_dataset


def test_synthetic_swap_and_presence_detection() -> None:
    """Verify swap (order) and missing component (presence) flag mismatches while normals pass."""
    recipe, ref_normals, val_normals, val_anomalies = build_validation_assembly_dataset(seed=42)
    engine = StageAROIEngine(min_mad=1e-3, default_z_threshold=3.5, early_exit=False)
    engine.fit(ref_normals, recipe)

    # Normal sample should produce zero mismatches
    normal_mms, _ = engine.inspect(val_normals[0], recipe)
    assert len(normal_mms) == 0

    # Find first presence and first order anomaly
    presence_img = next(img for img, k in val_anomalies if k == "presence")
    order_img = next(img for img, k in val_anomalies if k == "order")

    pres_mms, _ = engine.inspect(presence_img, recipe)
    assert any(m.kind == "presence" for m in pres_mms)

    order_mms, _ = engine.inspect(order_img, recipe)
    assert any(m.kind == "order" for m in order_mms)


def test_qc_failure_returns_recapture_never_reject() -> None:
    """Verify that a blurred or dark capture returns verdict='RECAPTURE' and never 'REJECT'."""
    _, ref_normals, _, _ = build_validation_assembly_dataset(seed=42)
    qc = QualityChecker(blur_floor_percentile=5.0, min_mad=1e-3)
    qc.fit(ref_normals)

    ok_res = qc.check(ref_normals[0])
    assert ok_res.passed is True
    assert ok_res.verdict == "PASS"

    blurred = ref_normals[0].filter(ImageFilter.GaussianBlur(radius=8))
    fail_res = qc.check(blurred)
    assert fail_res.passed is False
    assert fail_res.verdict == "RECAPTURE"
    assert fail_res.verdict != "REJECT"
