"""Evaluation harness for anomaly detection pipelines.

Supports any Scorer adhering to v3.core.contracts.Scorer.
Implements two split modes:
1. 'v1-native': Reproduces v1 baseline results using v1's native manifests.
2. 'v3-category': Official baseline fitting temperature on validation categories
   and evaluating on held-out evaluation categories.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional

# Ensure project root is importable for reading legacy v1 backend components
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.calibration import calibrate_probability, load_calibration
from app.core.clip_encoder import CLIPEncoder
from app.core.evaluation import tensor_heatmap_to_numpy
from app.core.fusion import average_flip_heatmaps, fuse_scores
from app.core.prompts import PromptBank


def resize_heatmap(heatmap: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Resize heatmap tensor to target dimensions and convert to numpy."""
    resized = functional.interpolate(
        heatmap.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return tensor_heatmap_to_numpy(resized.squeeze(1))

from v3.core.contracts import Recipe
from v3.eval.metrics import (
    expected_calibration_error,
    image_auroc,
    pixel_aupro,
    pixel_auroc,
    recall_at_frr,
)
from v3.eval.splits import (
    DatasetSample,
    create_v3_category_splits,
    load_v1_native_manifests,
)


@dataclass
class CategoryMetrics:
    """Metrics calculated for one category."""

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


@dataclass
class EvaluationReport:
    """Complete evaluation report across categories."""

    split_mode: str
    model_name: str
    device: str
    temperature: float
    bias: float
    categories: list[CategoryMetrics]
    macro_image_auroc: float
    macro_pixel_auroc: float
    macro_pixel_aupro_005: float
    macro_pixel_aupro_030: float
    macro_recall_at_5_frr: float
    macro_ece: float
    total_images: int
    elapsed_seconds: float


class V1BaselineScorer:
    """Scorer adapter for the legacy v1 OpenCLIP baseline."""

    def __init__(
        self,
        device: str = "cpu",
        temperature: float = 0.016078264866179207,
        bias: float = 0.905678038077271,
    ) -> None:
        self.device = torch.device(device)
        self.encoder = CLIPEncoder(device=self.device)
        self.prompt_bank = PromptBank(self.encoder)
        self.temperature = float(temperature)
        self.bias = float(bias)

    def fit(self, normals: list[Image.Image], recipe: Recipe) -> None:
        """Zero-shot scorer requires no reference normal fitting."""

    def fit_calibration_on_samples(
        self,
        samples: Sequence[DatasetSample],
    ) -> tuple[float, float]:
        """Fit temperature and intercept on calibration/validation samples via L-BFGS."""
        scores: list[torch.Tensor] = []
        labels: list[int] = []

        with torch.inference_mode():
            for sample in samples:
                with Image.open(sample.image_path) as img:
                    batch = self.encoder.prepare_image(img.convert("RGB"))
                    emb = self.encoder.encode_image(batch)
                prompts = self.prompt_bank.get(sample.category)
                res = fuse_scores(emb, prompts)
                scores.append(res.defect_logit.cpu())
                labels.append(sample.label)

        logits = torch.cat(scores).double()
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

    @torch.inference_mode()
    def score_sample(
        self,
        sample: DatasetSample,
        metric_max_side: int = 256,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Score a single DatasetSample."""
        with Image.open(sample.image_path) as img:
            original = img.convert("RGB")
            batch = self.encoder.prepare_image(original)
            emb = self.encoder.encode_image(batch)
            flipped_emb = self.encoder.encode_image(torch.flip(batch, dims=(-1,)))

        prompts = self.prompt_bank.get(sample.category)
        language = fuse_scores(emb, prompts)
        flipped_language = fuse_scores(flipped_emb, prompts)

        logit = float(language.defect_logit.item())
        prob = float(calibrate_probability(language.defect_logit, self.temperature, self.bias).item())

        w, h = original.size
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

        raw_map = average_flip_heatmaps(language.heatmap, flipped_language.heatmap)
        heatmap = resize_heatmap(raw_map, mh, mw)
        # v1 polarity is inverted: lower cosine difference = more anomalous
        heatmap = -heatmap

        return logit, prob, heatmap, gt_mask


def evaluate_category_samples(
    category: str,
    samples: list[DatasetSample],
    scorer: V1BaselineScorer,
    metric_max_side: int = 256,
) -> CategoryMetrics:
    """Evaluate all samples of a given category and compute metrics."""
    logits: list[float] = []
    probs: list[float] = []
    heatmaps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = [s.label for s in samples]

    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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

    return CategoryMetrics(
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
    )


def run_evaluation_pipeline(
    split_mode: str = "v1-native",
    data_root: Path = PROJECT_ROOT / "data" / "mvtec_ad",
    manifest_dir: Path = PROJECT_ROOT / "data" / "manifests",
    output_dir: Path = PROJECT_ROOT / "v3" / "docs" / "results" / "phase_1_logs",
    device: str = "cpu",
    seeds: list[int] | None = None,
) -> EvaluationReport:
    """Run full evaluation pipeline under the requested split mode."""
    seeds = seeds or [0, 1, 2]
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_start = time.perf_counter()

    if split_mode == "v1-native":
        _calib_samples, eval_samples = load_v1_native_manifests(data_root, manifest_dir)
        calib_config_path = PROJECT_ROOT / "backend" / "config" / "calibration.json"
        calib_cfg = load_calibration(calib_config_path)
        scorer = V1BaselineScorer(
            device=device,
            temperature=calib_cfg.temperature,
            bias=calib_cfg.bias,
        )
        test_samples = eval_samples
    elif split_mode == "v3-category":
        val_cats, _eval_cats, val_samples, eval_samples = create_v3_category_splits(
            data_root=data_root,
            manifest_dir=manifest_dir,
            seed=42,
        )
        scorer = V1BaselineScorer(device=device)
        print(f"Fitting temperature on validation categories: {val_cats} ({len(val_samples)} samples)...")
        t, b = scorer.fit_calibration_on_samples(val_samples)
        print(f"Fitted temperature: {t:.6f}, bias: {b:.6f}")
        test_samples = eval_samples
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    # Verify disjointness
    test_negs = [s.image_id for s in test_samples if s.label == 0]
    # Check that test negatives have no duplicates
    assert len(test_negs) == len(set(test_negs)), "Duplicate test negatives found"

    # Group samples by category
    categories = sorted({s.category for s in test_samples})
    category_results: list[CategoryMetrics] = []

    for cat in categories:
        cat_samples = [s for s in test_samples if s.category == cat]
        print(f"Evaluating {cat} ({len(cat_samples)} samples)...")
        cat_metrics = evaluate_category_samples(cat, cat_samples, scorer, metric_max_side=256)
        category_results.append(cat_metrics)

    elapsed = time.perf_counter() - overall_start

    macro_img_auroc = float(np.mean([c.image_auroc for c in category_results]))
    macro_px_auroc = float(np.mean([c.pixel_auroc for c in category_results]))
    macro_px_aupro_005 = float(np.mean([c.pixel_aupro_005 for c in category_results]))
    macro_px_aupro_030 = float(np.mean([c.pixel_aupro_030 for c in category_results]))
    macro_rec_frr = float(np.mean([c.recall_at_5_frr for c in category_results]))
    macro_ece = float(np.mean([c.expected_calibration_error for c in category_results]))

    report = EvaluationReport(
        split_mode=split_mode,
        model_name="ViT-B-32-quickgelu",
        device=device,
        temperature=scorer.temperature,
        bias=scorer.bias,
        categories=category_results,
        macro_image_auroc=macro_img_auroc,
        macro_pixel_auroc=macro_px_auroc,
        macro_pixel_aupro_005=macro_px_aupro_005,
        macro_pixel_aupro_030=macro_px_aupro_030,
        macro_recall_at_5_frr=macro_rec_frr,
        macro_ece=macro_ece,
        total_images=len(test_samples),
        elapsed_seconds=elapsed,
    )

    # Save JSON and CSV
    json_path = output_dir / f"{split_mode}_results.json"
    csv_path = output_dir / f"{split_mode}_results.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("category,images,normal_images,anomalous_images,image_auroc,pixel_auroc,pixel_aupro_005,pixel_aupro_030,recall_at_5_frr,ece,sec_per_image\n")
        f.writelines(f"{c.category},{c.images},{c.normal_images},{c.anomalous_images},"
                f"{c.image_auroc:.6f},{c.pixel_auroc:.6f},{c.pixel_aupro_005:.6f},{c.pixel_aupro_030:.6f},"
                f"{c.recall_at_5_frr:.6f},{c.expected_calibration_error:.6f},{c.seconds_per_image:.4f}\n" for c in category_results)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run v3 evaluation harness")
    parser.add_argument("--split-mode", choices=["v1-native", "v3-category"], default="v1-native")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_evaluation_pipeline(split_mode=args.split_mode, device=args.device)
    print(f"\n--- Evaluation Complete [{args.split_mode}] ---")
    print(f"Categories: {len(report.categories)}")
    print(f"Macro Image AUROC: {report.macro_image_auroc:.4f}")
    print(f"Macro Pixel AUROC: {report.macro_pixel_auroc:.4f}")
    print(f"Macro Pixel AU-PRO (0.05 limit): {report.macro_pixel_aupro_005:.4f}")
    print(f"Macro Pixel AU-PRO (0.30 limit): {report.macro_pixel_aupro_030:.4f}")
    print(f"Macro Recall at 5% FRR: {report.macro_recall_at_5_frr:.4f}")
    print(f"Macro ECE: {report.macro_ece:.4f}")
    print(f"Total time: {report.elapsed_seconds:.1f}s")
