"""Run Phase 2 experiments: polarity ablation, local-term sweep, and held-out evaluation.

Outputs:
- v3/docs/results/phase_2_polarity.md (Polarity mechanism investigation with per-category numbers)
- v3/docs/results/phase_2_local_term.md (Local term weight selection on 5 validation categories)
- v3/docs/results/phase_2_results.json and phase_2.md (Final held-out evaluation table per backbone)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from v3.core.features import CLIPFeatureExtractor
from v3.eval.phase2_engine import Phase2Scorer, evaluate_category
from v3.eval.splits import create_v3_category_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "v3" / "docs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_polarity_ablation(extractor_b32: CLIPFeatureExtractor, val_samples: list) -> dict:
    """Compare intermediate-inverted, intermediate-native, and final-layer tokens on validation categories."""
    print("\n================== Running Polarity Ablation on Validation Categories ==================")
    val_cats = sorted({s.category for s in val_samples})
    results = {}

    modes = ["intermediate_inverted", "intermediate_native", "final_tokens_native"]

    for mode in modes:
        print(f"\nEvaluating polarity mode: {mode}...")
        scorer = Phase2Scorer(
            extractor=extractor_b32,
            vocabulary_mode="domain_generic",
            polarity_mode=mode,  # type: ignore
        )
        cat_metrics = {}
        for cat in val_cats:
            samples = [s for s in val_samples if s.category == cat]
            res = evaluate_category(cat, samples, scorer)
            cat_metrics[cat] = {
                "image_auroc": res.image_auroc,
                "pixel_auroc": res.pixel_auroc,
                "pixel_aupro_005": res.pixel_aupro_005,
                "pixel_aupro_030": res.pixel_aupro_030,
            }
            print(f"  {cat}: Pixel AUROC = {res.pixel_auroc:.4f}, Image AUROC = {res.image_auroc:.4f}")
        macro_px = float(np.mean([m["pixel_auroc"] for m in cat_metrics.values()]))
        macro_img = float(np.mean([m["image_auroc"] for m in cat_metrics.values()]))
        print(f"  --> Macro Pixel AUROC: {macro_px:.4f}, Macro Image AUROC: {macro_img:.4f}")
        results[mode] = {
            "categories": cat_metrics,
            "macro_pixel_auroc": macro_px,
            "macro_image_auroc": macro_img,
        }

    return results


def run_local_term_sweep(extractor_b32: CLIPFeatureExtractor, val_samples: list) -> dict:
    """Evaluate global-only, local-only, and several fusion weights on validation categories."""
    print("\n================== Running Local Term Sweep on Validation Categories ==================")
    val_cats = sorted({s.category for s in val_samples})
    weights = [
        ("global_only", 1.0, 0.0),
        ("fused_095", 0.95, 0.05),
        ("fused_090", 0.90, 0.10),
        ("fused_080", 0.80, 0.20),
        ("fused_050", 0.50, 0.50),
        ("local_only", 0.0, 1.0),
    ]

    sweep_results = {}

    for name, gw, lw in weights:
        print(f"\nEvaluating weight config {name}: global={gw:.2f}, local={lw:.2f}...")
        scorer = Phase2Scorer(
            extractor=extractor_b32,
            vocabulary_mode="domain_generic",
            polarity_mode="final_tokens_native",
            global_weight=gw,
            local_weight=lw,
        )
        cat_metrics = {}
        for cat in val_cats:
            samples = [s for s in val_samples if s.category == cat]
            res = evaluate_category(cat, samples, scorer)
            cat_metrics[cat] = {
                "image_auroc": res.image_auroc,
                "pixel_auroc": res.pixel_auroc,
            }
        macro_img = float(np.mean([m["image_auroc"] for m in cat_metrics.values()]))
        macro_px = float(np.mean([m["pixel_auroc"] for m in cat_metrics.values()]))
        print(f"  --> Macro Image AUROC: {macro_img:.4f}, Macro Pixel AUROC: {macro_px:.4f}")
        sweep_results[name] = {
            "global_weight": gw,
            "local_weight": lw,
            "macro_image_auroc": macro_img,
            "macro_pixel_auroc": macro_px,
            "categories": cat_metrics,
        }

    return sweep_results


def run_heldout_evaluation(
    extractor: CLIPFeatureExtractor,
    backbone_id: str,
    val_samples: list,
    eval_samples: list,
    best_gw: float = 0.95,
    best_lw: float = 0.05,
) -> dict:
    """Fit calibration on validation categories and evaluate on full held-out test sets."""
    print(f"\n================== Running Held-Out Evaluation for {backbone_id} ==================")
    eval_cats = sorted({s.category for s in eval_samples})

    scorer = Phase2Scorer(
        extractor=extractor,
        vocabulary_mode="domain_generic",
        polarity_mode="final_tokens_native",
        global_weight=best_gw,
        local_weight=best_lw,
    )

    print(f"Fitting temperature and bias on 5 validation categories ({len(val_samples)} samples)...")
    temp, bias = scorer.fit_calibration_on_samples(val_samples)
    print(f"Fitted temperature: {temp:.6f}, bias: {bias:.6f}")

    results = []
    start_total = time.perf_counter()

    for cat in eval_cats:
        samples = [s for s in eval_samples if s.category == cat]
        print(f"Evaluating {cat} ({len(samples)} samples)...")
        res = evaluate_category(cat, samples, scorer)
        results.append(res)
        print(
            f"  {cat}: Img AUROC={res.image_auroc:.4f}, Px AUROC={res.pixel_auroc:.4f}, "
            f"AU-PRO(0.30)={res.pixel_aupro_030:.4f}, Rec@5%FRR={res.recall_at_5_frr:.4f}, "
            f"ECE={res.expected_calibration_error:.4f}, Sec/img={res.seconds_per_image:.4f}"
            + (" [LOW-N]" if res.is_low_n else "")
        )

    elapsed = time.perf_counter() - start_total

    macro_img = float(np.mean([r.image_auroc for r in results]))
    macro_px = float(np.mean([r.pixel_auroc for r in results]))
    macro_pro05 = float(np.mean([r.pixel_aupro_005 for r in results]))
    macro_pro30 = float(np.mean([r.pixel_aupro_030 for r in results]))
    macro_rec = float(np.mean([r.recall_at_5_frr for r in results]))
    macro_ece = float(np.mean([r.expected_calibration_error for r in results]))
    avg_sec = float(np.mean([r.seconds_per_image for r in results]))
    peak_vram = float(np.max([r.peak_vram_mb for r in results]))

    print(f"\n--- Held-Out Results Summary for {backbone_id} ---")
    print(f"Macro Image AUROC: {macro_img:.4f}")
    print(f"Macro Pixel AUROC: {macro_px:.4f}")
    print(f"Macro Pixel AU-PRO (0.05 limit): {macro_pro05:.4f}")
    print(f"Macro Pixel AU-PRO (0.30 limit): {macro_pro30:.4f}")
    print(f"Macro Recall at 5% FRR: {macro_rec:.4f}")
    print(f"Macro ECE: {macro_ece:.4f}")
    print(f"Avg Sec/Image: {avg_sec:.4f}")
    print(f"Peak VRAM: {peak_vram:.1f} MB")
    print(f"Total time: {elapsed:.1f}s")

    return {
        "backbone_id": backbone_id,
        "temperature": temp,
        "bias": bias,
        "best_global_weight": best_gw,
        "best_local_weight": best_lw,
        "macro_image_auroc": macro_img,
        "macro_pixel_auroc": macro_px,
        "macro_pixel_aupro_005": macro_pro05,
        "macro_pixel_aupro_030": macro_pro30,
        "macro_recall_at_5_frr": macro_rec,
        "macro_ece": macro_ece,
        "avg_seconds_per_image": avg_sec,
        "peak_vram_mb": peak_vram,
        "elapsed_seconds": elapsed,
        "categories": [
            {
                "category": r.category,
                "images": r.images,
                "normal_images": r.normal_images,
                "anomalous_images": r.anomalous_images,
                "image_auroc": r.image_auroc,
                "pixel_auroc": r.pixel_auroc,
                "pixel_aupro_005": r.pixel_aupro_005,
                "pixel_aupro_030": r.pixel_aupro_030,
                "recall_at_5_frr": r.recall_at_5_frr,
                "expected_calibration_error": r.expected_calibration_error,
                "seconds_per_image": r.seconds_per_image,
                "peak_vram_mb": r.peak_vram_mb,
                "is_low_n": r.is_low_n,
            }
            for r in results
        ],
    }


def main() -> None:
    # 1. Load splits
    val_cats, eval_cats, val_samples, eval_samples = create_v3_category_splits(
        data_root=DATA_ROOT,
        manifest_dir=MANIFEST_DIR,
        seed=42,
    )
    print(f"Validation categories ({len(val_cats)}): {val_cats}")
    print(f"Held-out evaluation categories ({len(eval_cats)}): {eval_cats}")

    # 2. Initialize backbones
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing clip_b32_224 on {device} (fp32)...")
    extractor_b32 = CLIPFeatureExtractor(
        backbone_id="clip_b32_224",
        device=device,
        precision="fp32",
    )

    # 3. Polarity ablation on validation categories
    polarity_checkpoint = RESULTS_DIR / "phase_2_polarity.json"
    if polarity_checkpoint.exists():
        print(f"\nLoading cached polarity results from {polarity_checkpoint}...")
        with open(polarity_checkpoint) as f:
            polarity_results = json.load(f)
    else:
        polarity_results = run_polarity_ablation(extractor_b32, val_samples)
        with open(polarity_checkpoint, "w") as f:
            json.dump(polarity_results, f, indent=2)

    # 4. Local term sweep on validation categories
    sweep_checkpoint = RESULTS_DIR / "phase_2_local_sweep.json"
    if sweep_checkpoint.exists():
        print(f"\nLoading cached local sweep results from {sweep_checkpoint}...")
        with open(sweep_checkpoint) as f:
            sweep_results = json.load(f)
    else:
        sweep_results = run_local_term_sweep(extractor_b32, val_samples)
        with open(sweep_checkpoint, "w") as f:
            json.dump(sweep_results, f, indent=2)

    # Pick best weights on validation set
    best_config_name = max(sweep_results, key=lambda k: sweep_results[k]["macro_image_auroc"])
    best_gw = sweep_results[best_config_name]["global_weight"]
    best_lw = sweep_results[best_config_name]["local_weight"]
    print(f"\nBest config on validation set: {best_config_name} (gw={best_gw}, lw={best_lw})")

    # Check if b32 heldout already computed
    b32_checkpoint = RESULTS_DIR / "phase_2_b32_heldout.json"
    if b32_checkpoint.exists():
        print(f"\nLoading cached clip_b32_224 held-out results from {b32_checkpoint}...")
        with open(b32_checkpoint) as f:
            b32_heldout = json.load(f)
    else:
        b32_heldout = run_heldout_evaluation(
            extractor=extractor_b32,
            backbone_id="clip_b32_224",
            val_samples=val_samples,
            eval_samples=eval_samples,
            best_gw=best_gw,
            best_lw=best_lw,
        )
        with open(b32_checkpoint, "w") as f:
            json.dump(b32_heldout, f, indent=2)

    # Free b32 extractor and clear CUDA memory before loading l14
    del extractor_b32
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Held-out evaluation: Backbone 2: clip_l14_336 (fp16 on CUDA if available)
    print(f"\nInitializing clip_l14_336 on {device} (fp16)...")
    extractor_l14 = CLIPFeatureExtractor(
        backbone_id="clip_l14_336",
        device=device,
        precision="fp16" if device == "cuda" else "fp32",
    )

    l14_heldout = run_heldout_evaluation(
        extractor=extractor_l14,
        backbone_id="clip_l14_336",
        val_samples=val_samples,
        eval_samples=eval_samples,
        best_gw=best_gw,
        best_lw=best_lw,
    )

    # Save complete Phase 2 report JSON
    final_output = {
        "polarity_ablation": polarity_results,
        "local_term_sweep": sweep_results,
        "best_fusion_config": {
            "name": best_config_name,
            "global_weight": best_gw,
            "local_weight": best_lw,
        },
        "heldout_evaluations": {
            "clip_b32_224": b32_heldout,
            "clip_l14_336": l14_heldout,
        },
    }

    with open(RESULTS_DIR / "phase_2_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("\nPhase 2 experiments successfully completed and saved to v3/docs/results/phase_2_results.json")


if __name__ == "__main__":
    main()
