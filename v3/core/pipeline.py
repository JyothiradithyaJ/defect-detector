"""End-to-end Two-Stage Inspection Pipeline wiring S0 (QC) through S5 (Report)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from v3.core.align import Stage1Aligner
from v3.core.bank import DINOFeatureExtractor, default_synthetic_feature_fn
from v3.core.calibrate import LogisticCalibrator
from v3.core.contracts import Mismatch, Recipe, Result
from v3.core.decide import decide_verdict, fit_threshold_at_frr
from v3.core.defect.knn import GlobalKNNScorer
from v3.core.defect.pca import PCAResidualScorer
from v3.core.fusion import BranchFuser
from v3.core.mismatch.positional_knn import PositionalKNNScorer
from v3.core.qc import QualityChecker
from v3.core.report import validate_result_contract
from v3.core.roi import StageAROIEngine


class TwoStagePipeline:
    """Unified Stage A + Stage B inspection pipeline implementing v3.core.contracts.Scorer."""

    def __init__(
        self,
        dino_extractor: DINOFeatureExtractor | None = None,
        scorer_mode: str = "knn_positional",
        pos_radius: int = 2,
        agg_radius: int = 1,
        topk_frac: float = 0.001,
        w_d: float = 1.0,
        w_c: float = 0.0,
        false_reject_rate: float = 0.05,
        min_mad: float = 1e-3,
        early_exit: bool = False,
        enable_stage_a: bool = True,
        enable_stage_b: bool = True,
    ) -> None:
        self.dino_extractor = dino_extractor or DINOFeatureExtractor(feature_fn=default_synthetic_feature_fn)
        self.scorer_mode = scorer_mode
        self.pos_radius = int(pos_radius)
        self.agg_radius = int(agg_radius)
        self.topk_frac = float(topk_frac)
        self.w_d = float(w_d)
        self.w_c = float(w_c)
        self.false_reject_rate = float(false_reject_rate)
        self.min_mad = float(min_mad)
        self.early_exit = bool(early_exit)
        self.enable_stage_a = bool(enable_stage_a)
        self.enable_stage_b = bool(enable_stage_b)

        self.qc = QualityChecker(min_mad=min_mad)
        self.aligner = Stage1Aligner()
        self.roi_engine = StageAROIEngine(min_mad=min_mad, early_exit=early_exit)
        self.fuser = BranchFuser(w_d=w_d, w_c=w_c, min_mad=min_mad)
        self.calibrator = LogisticCalibrator()

        self.stage_b_scorer: GlobalKNNScorer | PositionalKNNScorer | PCAResidualScorer = self._make_stage_b_scorer()
        self.recipe: Recipe | None = None
        self.template_image: Image.Image | None = None
        self.tau: float = 3.0
        self.surface_pixel_threshold: float = 4.0

    def _make_stage_b_scorer(self) -> GlobalKNNScorer | PositionalKNNScorer | PCAResidualScorer:
        if self.scorer_mode == "knn_global":
            return GlobalKNNScorer(extractor=self.dino_extractor, agg_radius=self.agg_radius, topk_frac=self.topk_frac)
        if self.scorer_mode == "pca_residual":
            return PCAResidualScorer(extractor=self.dino_extractor, agg_radius=self.agg_radius, topk_frac=self.topk_frac)
        return PositionalKNNScorer(
            extractor=self.dino_extractor,
            radius=self.pos_radius,
            agg_radius=self.agg_radius,
            topk_frac=self.topk_frac,
        )

    def fit(self, normals: list[Image.Image], recipe: Recipe) -> None:
        """Fit Stage 0 QC, Stage 1 template, Stage A ROIs, Stage B bank, and threshold tau on normals."""
        if not normals:
            raise ValueError("Cannot fit TwoStagePipeline with empty normals list.")

        self.recipe = recipe
        self.template_image = normals[0].convert("RGB")
        self.qc.fit(normals)

        if self.enable_stage_a:
            self.roi_engine.fit(normals, recipe)

        if self.enable_stage_b:
            self.stage_b_scorer.fit(normals, recipe)
            raw_dino_scores: list[float] = []
            raw_dino_maps: list[np.ndarray] = []
            for img in normals:
                res = self.stage_b_scorer.score(img)
                raw_dino_scores.append(res.image_score)
                raw_dino_maps.append(res.anomaly_map)

            self.fuser.fit(dino_normal_scores=raw_dino_scores, dino_normal_maps=raw_dino_maps)
            fused_norm_scores = [self.fuser.fuse_score(s, 0.0) for s in raw_dino_scores]
            self.tau = fit_threshold_at_frr(fused_norm_scores, false_reject_rate=self.false_reject_rate)
            fused_norm_maps = [self.fuser.fuse_map(m) for m in raw_dino_maps]
            all_px = np.concatenate([m.ravel() for m in fused_norm_maps])
            self.surface_pixel_threshold = float(np.quantile(all_px, 0.995))

    def score(self, image: Image.Image, clip_score: float = 0.0, clip_map: np.ndarray | None = None) -> Result:
        """Execute S0 -> S1 -> S2 -> S3 -> S4 -> S5 on `image` and return a validated Result."""
        if self.recipe is None or self.template_image is None:
            raise RuntimeError("TwoStagePipeline must be fit before calling score().")

        w, h = image.size
        # S0: Quality Check
        qc_res = self.qc.check(image)
        if not qc_res.passed:
            res = Result(
                verdict="RECAPTURE",
                mismatches=[],
                image_score=0.0,
                anomaly_map=np.zeros((h, w), dtype=np.float32),
                meta={"qc": qc_res.metrics, "qc_reasons": qc_res.reasons, "calibrated": self.calibrator.is_calibrated},
            )
            validate_result_contract(res)
            return res

        # S1: Alignment & Variant Check
        mismatches: list[Mismatch] = []
        aln = self.aligner.align(image, self.template_image, self.recipe)
        if aln.mismatch is not None:
            mismatches.append(aln.mismatch)

        # S2: Stage A ROI Mismatch Inspection
        roi_z_scores: dict[str, float] = {}
        if self.enable_stage_a:
            stage_a_mms, roi_z_scores = self.roi_engine.inspect(
                aln.aligned_image,
                self.recipe,
                skip_positional=aln.diverged,
            )
            mismatches.extend(stage_a_mms)

        # Early exit if configured and a hard mismatch was raised in Stage A
        if self.early_exit and any(m.hard for m in mismatches):
            dec = decide_verdict(qc_passed=True, mismatches=mismatches, image_score=0.0, tau=self.tau)
            res = Result(
                verdict=dec.verdict,  # type: ignore[arg-type]
                mismatches=mismatches,
                image_score=float(max((z for z in roi_z_scores.values()), default=10.0)),
                anomaly_map=np.zeros((h, w), dtype=np.float32),
                meta={"early_exit": True, "rework_level": dec.rework_level, "decision_reason": dec.reason},
            )
            validate_result_contract(res)
            return res

        # S3: Stage B DINO + CLIP Fusion
        if self.enable_stage_b:
            b_res = self.stage_b_scorer.score(aln.aligned_image)
            fused_score = self.fuser.fuse_score(b_res.image_score, clip_score)
            fused_map = self.fuser.fuse_map(b_res.anomaly_map, clip_map)
        else:
            fused_score = float(max(roi_z_scores.values(), default=0.0))
            fused_map = np.zeros((h, w), dtype=np.float32)

        # S4: Decision
        dec = decide_verdict(
            qc_passed=True,
            mismatches=mismatches,
            image_score=fused_score,
            tau=self.tau,
            anomaly_map=fused_map,
            recipe=self.recipe,
            surface_pixel_threshold=self.surface_pixel_threshold,
        )

        meta = {
            "rework_level": dec.rework_level,
            "decision_reason": dec.reason,
            "tau": self.tau,
            "roi_z_scores": roi_z_scores,
            "alignment_residual": round(aln.residual, 4),
            "calibrated": self.calibrator.is_calibrated,
            "calibration_note": "calibrated, not strictly zero-shot" if self.calibrator.is_calibrated else "raw_anomaly_score",
        }
        if self.calibrator.is_calibrated:
            meta["calibrated_probability"] = self.calibrator.predict_proba(fused_score)

        res = Result(
            verdict=dec.verdict,  # type: ignore[arg-type]
            mismatches=mismatches,
            image_score=float(fused_score),
            anomaly_map=fused_map,
            meta=meta,
        )
        validate_result_contract(res)
        return res
