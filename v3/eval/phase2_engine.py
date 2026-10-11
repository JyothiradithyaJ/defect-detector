"""Phase 2 evaluation engine and scorer for multi-backbone CLIP inspection.

Supports:
- Backbones: clip_b32_224 and clip_l14_336
- Embedding caching to disk (data/cache/visual_embeddings/)
- Domain vocabulary prompt bank (v3/core/vocabulary.py) vs v1 prompt bank
- Polarity modes:
  * 'final_tokens_native': final layer patch tokens (ln_post + proj), defect - normal
  * 'intermediate_native': intermediate layer patch tokens, defect - normal
  * 'intermediate_inverted': legacy v1 intermediate tokens, -(defect - normal)
- Fusion weight sweeps: global_weight in [1.0, 0.95, 0.90, 0.80, 0.0]
- Temperature calibration via L-BFGS on validation categories
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.nn import functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.calibration import calibrate_probability
from app.core.prompts import PromptBank as V1PromptBank

from v3.core.features import CachedVisualFeatures, CLIPFeatureExtractor
from v3.core.vocabulary import DomainPromptBank
from v3.eval.metrics import (
    expected_calibration_error,
    image_auroc,
    pixel_aupro,
    pixel_auroc,
    recall_at_frr,
)
from v3.eval.splits import DatasetSample

PolarityMode = Literal["final_tokens_native", "intermediate_native", "intermediate_inverted"]
VocabularyMode = Literal["domain_generic", "v1_legacy"]


def robust_topk_mean(patch_scores: np.ndarray, top_fraction: float = 0.10) -> float:
    """Mean of top-k% highest patch scores."""
    flat = patch_scores.flatten()
    k = max(1, int(len(flat) * top_fraction))
    top_indices = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[top_indices]))


def resize_heatmap_np(heatmap: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Bilinear resize of numpy heatmap [H, W] to target dimensions [target_height, target_width]."""
    tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    resized = functional.interpolate(
        tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).numpy()


@dataclass
class Phase2CategoryResult:
    category: str
    images: int
    normal_images: int
    anomalous_images: int
    image_auroc: float
    pixel_auroc: float
    pixel_aupro_005: float
    pixel_aupro_030: float
    recall_at_5_frr: float
    expected_calibration_error: float
    seconds_per_image: float
    peak_vram_mb: float
    is_low_n: bool = False


class Phase2Scorer:
    """Zero-shot CLIP scorer with configurable backbone, vocabulary, polarity, and fusion weights."""

    def __init__(
        self,
        extractor: CLIPFeatureExtractor,
        vocabulary_mode: VocabularyMode = "domain_generic",
        polarity_mode: PolarityMode = "final_tokens_native",
        global_weight: float = 0.95,
        local_weight: float = 0.05,
        temperature: float = 0.011036,
        bias: float = 0.418684,
    ) -> None:
        self.extractor = extractor
        self.vocabulary_mode = vocabulary_mode
        self.polarity_mode = polarity_mode
        self.global_weight = float(global_weight)
        self.local_weight = float(local_weight)
        self.temperature = float(temperature)
        self.bias = float(bias)

        if vocabulary_mode == "domain_generic":
            self.domain_prompts = DomainPromptBank(extractor)
            self.v1_prompts = None
        else:
            self.domain_prompts = None
            # Wrap extractor for v1 PromptBank compatibility
            self.v1_prompts = V1PromptBank(extractor.model)  # type: ignore

    def get_text_embeddings(self, category: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (normal_emb [D], anomalous_emb [D])."""
        if self.domain_prompts is not None:
            return self.domain_prompts.get_embeddings(category)
        
        # Legacy v1 prompts
        prompts = self.v1_prompts.get(category)
        return (
            prompts.normal.detach().cpu().numpy().astype(np.float32),
            prompts.anomalous.detach().cpu().numpy().astype(np.float32),
        )

    def score_sample(
        self,
        sample: DatasetSample,
        metric_max_side: int = 256,
        need_heatmap: bool = True,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Compute image logit, calibrated probability, anomaly heatmap, and ground truth mask."""
        norm_emb, def_emb = self.get_text_embeddings(sample.category)
        vis: CachedVisualFeatures = self.extractor.get_features(sample.image_path)

        # 1. Global similarities
        g_orig = vis.global_embedding
        cos_norm_g = float(np.dot(g_orig, norm_emb))
        cos_def_g = float(np.dot(g_orig, def_emb))
        s_global = cos_def_g - cos_norm_g

        # 2. Patch representations based on polarity mode
        if self.polarity_mode == "final_tokens_native":
            p_orig = vis.final_patch_tokens            # [H, W, D]
            p_flip = vis.final_patch_tokens_flipped    # [H, W, D]
            invert_sign = False
        elif self.polarity_mode == "intermediate_native":
            p_orig = vis.patch_embeddings              # [H, W, D]
            p_flip = vis.patch_embeddings_flipped      # [H, W, D]
            invert_sign = False
        elif self.polarity_mode == "intermediate_inverted":
            p_orig = vis.patch_embeddings              # [H, W, D]
            p_flip = vis.patch_embeddings_flipped      # [H, W, D]
            invert_sign = True
        else:
            raise ValueError(f"Unknown polarity_mode: {self.polarity_mode}")

        # Patch cosine differences
        # p_orig: [H, W, D], def_emb: [D], norm_emb: [D]
        map_orig = np.dot(p_orig, def_emb) - np.dot(p_orig, norm_emb)  # [H, W]
        map_flip = np.dot(p_flip, def_emb) - np.dot(p_flip, norm_emb)  # [H, W]

        # Horizontal flip alignment
        map_flip_aligned = np.fliplr(map_flip)
        avg_map = 0.5 * (map_orig + map_flip_aligned)

        if invert_sign:
            avg_map = -avg_map

        # Local anomaly score via robust top-k% mean
        s_local = robust_topk_mean(avg_map, top_fraction=0.10)

        # Fused logit
        logit = self.global_weight * s_global + self.local_weight * s_local

        # Calibrated probability
        t_tensor = torch.tensor(logit, dtype=torch.float32)
        prob = float(calibrate_probability(t_tensor, self.temperature, self.bias).item())

        if not need_heatmap:
            return logit, prob, np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=bool)

        # Ground truth mask & heatmap resizing
        with Image.open(sample.image_path) as img:
            w, h = img.size
        scale = min(1.0, metric_max_side / max(w, h)) if metric_max_side > 0 else 1.0
        mw, mh = max(1, round(w * scale)), max(1, round(h * scale))

        if sample.label and sample.mask_path and sample.mask_path.is_file():
            with Image.open(sample.mask_path) as mask:
                gt_mask = np.asarray(
                    mask.convert("L").resize((mw, mh), Image.Resampling.NEAREST),
                    dtype=np.uint8,
                ) > 0
        else:
            gt_mask = np.zeros((mh, mw), dtype=bool)

        heatmap = resize_heatmap_np(avg_map, mh, mw)

        return logit, prob, heatmap, gt_mask

    def fit_calibration_on_samples(
        self,
        samples: list[DatasetSample],
    ) -> tuple[float, float]:
        """Fit temperature and intercept on calibration/validation samples via L-BFGS."""
        scores: list[float] = []
        labels: list[int] = []

        for sample in samples:
            logit, _, _, _ = self.score_sample(sample, need_heatmap=False)
            scores.append(logit)
            labels.append(sample.label)

        logits = torch.tensor(scores, dtype=torch.float64)
        targets = torch.tensor(labels, dtype=torch.float64)

        log_temp = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        optimizer = torch.optim.LBFGS([log_temp, bias], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            t = log_temp.exp()
            loss = functional.binary_cross_entropy_with_logits(logits / t + bias, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temp.exp().item())
        self.bias = float(bias.item())
        return self.temperature, self.bias


def evaluate_category(
    category: str,
    samples: list[DatasetSample],
    scorer: Phase2Scorer,
    metric_max_side: int = 256,
) -> Phase2CategoryResult:
    """Evaluate samples of a category and return structured metrics."""
    logits: list[float] = []
    probs: list[float] = []
    heatmaps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = [s.label for s in samples]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    for s in samples:
        logit, prob, hmap, mask = scorer.score_sample(s, metric_max_side=metric_max_side)
        logits.append(logit)
        probs.append(prob)
        heatmaps.append(hmap)
        masks.append(mask)

    elapsed = time.perf_counter() - start_time
    sec_per_img = elapsed / len(samples) if samples else 0.0

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

    h_stack = np.stack(heatmaps)
    m_stack = np.stack(masks)

    img_auroc = image_auroc(labels, logits)
    px_auroc = pixel_auroc(h_stack, m_stack)
    px_aupro_005 = pixel_aupro(h_stack, m_stack, max_false_positive_rate=0.05)
    px_aupro_030 = pixel_aupro(h_stack, m_stack, max_false_positive_rate=0.30)
    rec_frr = recall_at_frr(labels, logits, false_reject_rate=0.05)
    ece = expected_calibration_error(probs, labels)

    is_low_n = (category == "toothbrush") or (len(samples) < 50)

    return Phase2CategoryResult(
        category=category,
        images=len(samples),
        normal_images=sum(1 for y in labels if y == 0),
        anomalous_images=sum(1 for y in labels if y == 1),
        image_auroc=img_auroc,
        pixel_auroc=px_auroc,
        pixel_aupro_005=px_aupro_005,
        pixel_aupro_030=px_aupro_030,
        recall_at_5_frr=rec_frr,
        expected_calibration_error=ece,
        seconds_per_image=sec_per_img,
        peak_vram_mb=peak_vram,
        is_low_n=is_low_n,
    )

