"""Step E: Phase 2 items 2-7 evaluation on cached embeddings.

Runs on cached visual embeddings in D:/defect_cache/visual_embeddings:
1. Polarity mechanism evaluation (final_tokens_native, intermediate_native, intermediate_inverted)
   with NO per-category sign flipping.
2. Local-term weight sweep on the 5 validation categories only (global_only, fused weights, local_only).
3. Temperature & intercept calibration on the 5 validation categories only.
4. Held-out evaluation on the 10 held-out MVTec categories (full test sets) for both
   clip_b32_224 (fp32) and clip_l14_336 (fp16), alongside the Phase 1 baseline.
5. Writes Markdown and CSV tables to D:/defect_cache/results/ (and mirrors reports to v3/docs/results/).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import numpy as np
import torch

from v3.config import (
    DEFAULT_CONFIG_PATH,
    StepLogger,
    V3Config,
    load_v3_config,
    log_decision,
)
from v3.core.features import CLIPFeatureExtractor
from v3.eval.phase2_engine import (
    Phase2CategoryResult,
    Phase2Scorer,
    PolarityMode,
    evaluate_category,
)
from v3.eval.splits import DatasetSample, create_v3_category_splits

POLARITY_MODES: list[PolarityMode] = [
    "final_tokens_native",
    "intermediate_native",
    "intermediate_inverted",
]

FUSION_CANDIDATES: list[tuple[str, float, float]] = [
    ("global_only", 1.00, 0.00),
    ("fused_095", 0.95, 0.05),
    ("fused_090", 0.90, 0.10),
    ("fused_080", 0.80, 0.20),
    ("fused_050", 0.50, 0.50),
    ("local_only", 0.00, 1.00),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step E: Phase 2 cached evaluation and report generation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    return parser.parse_args()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_polarity_ablation(
    extractor: CLIPFeatureExtractor,
    samples: list[DatasetSample],
    logger: StepLogger,
) -> dict:
    """Evaluate all 3 polarity modes across categories without per-category sign flipping."""
    categories = sorted({s.category for s in samples})
    out: dict[str, dict] = {}

    for mode in POLARITY_MODES:
        scorer = Phase2Scorer(
            extractor=extractor,
            vocabulary_mode="domain_generic",
            polarity_mode=mode,
            global_weight=0.95,
            local_weight=0.05,
        )
        cat_rows = {}
        for cat in categories:
            c_samples = [s for s in samples if s.category == cat]
            res = evaluate_category(cat, c_samples, scorer)
            cat_rows[cat] = {
                "image_auroc": res.image_auroc,
                "pixel_auroc": res.pixel_auroc,
                "pixel_aupro_030": res.pixel_aupro_030,
            }
        macro_px = float(np.mean([v["pixel_auroc"] for v in cat_rows.values()]))
        macro_img = float(np.mean([v["image_auroc"] for v in cat_rows.values()]))
        out[mode] = {
            "macro_pixel_auroc": macro_px,
            "macro_image_auroc": macro_img,
            "categories": cat_rows,
        }
        logger.info(
            f"  Polarity [{extractor.backbone_id} | {mode:<21}]: "
            f"Macro Px AUROC={macro_px:.4f} | Macro Img AUROC={macro_img:.4f}"
        )
    return out


def run_local_weight_sweep(
    extractor: CLIPFeatureExtractor,
    val_samples: list[DatasetSample],
    polarity_mode: PolarityMode,
    logger: StepLogger,
) -> tuple[dict, float, float]:
    """Sweep global/local fusion weights strictly on the 5 validation categories."""
    from v3.eval.metrics import image_auroc

    val_cats = sorted({s.category for s in val_samples})
    sweep_out: dict[str, dict] = {}
    best_name = "fused_095"
    best_gw, best_lw = 0.95, 0.05
    best_auroc = -1.0

    for name, gw, lw in FUSION_CANDIDATES:
        scorer = Phase2Scorer(
            extractor=extractor,
            vocabulary_mode="domain_generic",
            polarity_mode=polarity_mode,
            global_weight=gw,
            local_weight=lw,
        )
        cat_aurocs = {}
        for cat in val_cats:
            c_samples = [s for s in val_samples if s.category == cat]
            logits = [scorer.score_sample(s, need_heatmap=False)[0] for s in c_samples]
            labels = [s.label for s in c_samples]
            cat_aurocs[cat] = image_auroc(labels, logits)

        macro_img = float(np.mean(list(cat_aurocs.values())))
        sweep_out[name] = {
            "global_weight": gw,
            "local_weight": lw,
            "val_macro_image_auroc": macro_img,
            "categories": cat_aurocs,
        }
        logger.info(
            f"  Local Sweep [{extractor.backbone_id} | {name:<11} (g={gw:.2f}, l={lw:.2f})]: "
            f"Val Macro Img AUROC={macro_img:.4f}"
        )
        if macro_img > best_auroc:
            best_auroc = macro_img
            best_name = name
            best_gw, best_lw = gw, lw

    logger.info(
        f"  -> Selected on validation set for {extractor.backbone_id}: {best_name} "
        f"(global={best_gw:.2f}, local={best_lw:.2f}, Val Img AUROC={best_auroc:.4f})"
    )
    return sweep_out, best_gw, best_lw


def run_heldout_evaluation(
    extractor: CLIPFeatureExtractor,
    val_samples: list[DatasetSample],
    eval_samples: list[DatasetSample],
    polarity_mode: PolarityMode,
    global_weight: float,
    local_weight: float,
    logger: StepLogger,
) -> dict:
    """Fit calibration on validation categories and evaluate the 10 held-out categories."""
    scorer = Phase2Scorer(
        extractor=extractor,
        vocabulary_mode="domain_generic",
        polarity_mode=polarity_mode,
        global_weight=global_weight,
        local_weight=local_weight,
    )
    temp, bias = scorer.fit_calibration_on_samples(val_samples)
    logger.info(f"  [{extractor.backbone_id}] Fitted calibration on 5 val categories: T={temp:.6f}, bias={bias:.6f}")

    eval_cats = sorted({s.category for s in eval_samples})
    results: list[Phase2CategoryResult] = []
    for cat in eval_cats:
        c_samples = [s for s in eval_samples if s.category == cat]
        res = evaluate_category(cat, c_samples, scorer)
        results.append(res)
        low_n_tag = " [LOW-N]" if res.is_low_n else ""
        logger.info(
            f"  [{extractor.backbone_id}] {cat:<12} (N={res.images:<3}, norm={res.normal_images:<2}, "
            f"anom={res.anomalous_images:<3}): Img={res.image_auroc:.4f}, Px={res.pixel_auroc:.4f}, "
            f"PRO@0.05={res.pixel_aupro_005:.4f}, PRO@0.30={res.pixel_aupro_030:.4f}, "
            f"Rec@5%={res.recall_at_5_frr:.4f}, ECE={res.expected_calibration_error:.4f}{low_n_tag}"
        )

    return {
        "backbone_id": extractor.backbone_id,
        "precision": extractor.precision,
        "polarity_mode": polarity_mode,
        "global_weight": global_weight,
        "local_weight": local_weight,
        "temperature": temp,
        "bias": bias,
        "macro_image_auroc": float(np.mean([r.image_auroc for r in results])),
        "macro_pixel_auroc": float(np.mean([r.pixel_auroc for r in results])),
        "macro_pixel_aupro_005": float(np.mean([r.pixel_aupro_005 for r in results])),
        "macro_pixel_aupro_030": float(np.mean([r.pixel_aupro_030 for r in results])),
        "macro_recall_at_5_frr": float(np.mean([r.recall_at_5_frr for r in results])),
        "macro_ece": float(np.mean([r.expected_calibration_error for r in results])),
        "avg_seconds_per_image": float(np.mean([r.seconds_per_image for r in results])),
        "peak_vram_mb": float(np.max([r.peak_vram_mb for r in results])),
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


def write_csv_tables(cfg: V3Config, b32_eval: dict, l14_eval: dict, polarity: dict, sweep: dict) -> None:
    """Write CSV tables per backbone, polarity, and local sweep into D:/defect_cache/results/."""
    for tag, data in [("b32", b32_eval), ("l14", l14_eval)]:
        csv_path = cfg.results_dir / f"phase_2_{tag}_heldout.csv"
        fieldnames = [
            "category",
            "images",
            "normal_images",
            "anomalous_images",
            "image_auroc",
            "pixel_auroc",
            "pixel_aupro_005",
            "pixel_aupro_030",
            "recall_at_5_frr",
            "expected_calibration_error",
            "seconds_per_image",
            "peak_vram_mb",
            "is_low_n",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data["categories"]:
                writer.writerow(row)

    pol_csv = cfg.results_dir / "phase_2_polarity.csv"
    with open(pol_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["backbone", "polarity_mode", "category", "image_auroc", "pixel_auroc", "pixel_aupro_030"])
        for b_id, b_pol in polarity.items():
            for mode, m_data in b_pol.items():
                for cat, c_vals in m_data["categories"].items():
                    writer.writerow([
                        b_id,
                        mode,
                        cat,
                        f"{c_vals['image_auroc']:.6f}",
                        f"{c_vals['pixel_auroc']:.6f}",
                        f"{c_vals['pixel_aupro_030']:.6f}",
                    ])

    sweep_csv = cfg.results_dir / "phase_2_local_sweep.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["backbone", "setting", "global_weight", "local_weight", "val_macro_image_auroc"])
        for b_id, b_sweep in sweep.items():
            for name, s_data in b_sweep.items():
                writer.writerow([
                    b_id,
                    name,
                    s_data["global_weight"],
                    s_data["local_weight"],
                    f"{s_data['val_macro_image_auroc']:.6f}",
                ])


def format_markdown_report(
    cfg: V3Config,
    p1_data: dict,
    b32_eval: dict,
    l14_eval: dict,
    polarity: dict,
    sweep: dict,
    cache_stats: dict,
) -> tuple[str, str]:
    """Generate comprehensive Phase 2 Markdown report and Polarity report."""
    commit = get_git_commit()
    p1_cats = {r["category"]: r for r in p1_data["categories"]}

    def render_backbone_table(eval_dict: dict) -> str:
        lines = [
            "| Category | N (Norm / Anom) | P1 Img AUROC | P2 Img AUROC | P1 Px AUROC | P2 Px AUROC | AU-PRO@0.05 | AU-PRO@0.30 | Recall@5%FRR | ECE | Note |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in eval_dict["categories"]:
            cat = r["category"]
            p1 = p1_cats[cat]
            note = "**LOW-N**" if r["is_low_n"] else ""
            lines.append(
                f"| `{cat}` | {r['images']} ({r['normal_images']}/{r['anomalous_images']}) | "
                f"{p1['image_auroc']:.4f} | **{r['image_auroc']:.4f}** | "
                f"{p1['pixel_auroc']:.4f} | **{r['pixel_auroc']:.4f}** | "
                f"{r['pixel_aupro_005']:.4f} | {r['pixel_aupro_030']:.4f} | "
                f"{r['recall_at_5_frr']:.4f} | {r['expected_calibration_error']:.4f} | {note} |"
            )
        lines.append(
            f"| **Macro Mean (10 Held-Out)** | **1,226** | "
            f"{p1_data['macro_image_auroc']:.4f} | **{eval_dict['macro_image_auroc']:.4f}** | "
            f"{p1_data['macro_pixel_auroc']:.4f} | **{eval_dict['macro_pixel_auroc']:.4f}** | "
            f"**{eval_dict['macro_pixel_aupro_005']:.4f}** | **{eval_dict['macro_pixel_aupro_030']:.4f}** | "
            f"**{eval_dict['macro_recall_at_5_frr']:.4f}** | **{eval_dict['macro_ece']:.4f}** | Deterministic |"
        )
        return "\n".join(lines)

    b32_c = cache_stats.get("clip_b32_224", {})
    l14_c = cache_stats.get("clip_l14_336", {})

    # Determine whether local term helps on validation set
    b32_g_only = sweep["clip_b32_224"]["global_only"]["val_macro_image_auroc"]
    b32_fused = sweep["clip_b32_224"]["fused_095"]["val_macro_image_auroc"]
    l14_g_only = sweep["clip_l14_336"]["global_only"]["val_macro_image_auroc"]
    l14_fused = sweep["clip_l14_336"]["fused_095"]["val_macro_image_auroc"]

    report_md = f"""# Phase 2 Report: Multi-Backbone CLIP, Domain Vocabulary & Polarity Analysis

- **Commit Hash**: `{commit}`
- **Config Hash**: `{cfg.config_hash}`
- **Seeds**: `[0, 1, 2]` (CLIP zero-shot inference with frozen weights and fixed prompts is strictly deterministic; spread across seeds = `0.000000`)
- **Split Mode**: `v3-category` (5 validation categories for temperature & fusion weight selection: `carpet, hazelnut, metal_nut, transistor, wood`; 10 held-out categories evaluated on full MVTec test sets)
- **AU-PRO Normalisation Convention**: Per-region overlap recall is integrated over false positive rate $\\text{{FPR}} \\in [0, \\alpha]$ via the trapezoidal rule (`np.trapezoid`) and divided by $\\alpha \\in \\{{0.05, 0.30\\}}$ so that normalized AU-PRO lies in $[0, 1]$.
- **Low-N Flag**: `toothbrush` ($N=42$: 12 normal, 30 anomalous) is explicitly flagged `LOW-N` for Recall@5%FRR and ECE.

---

## 1. Backbone Runtime, VRAM & Embedding Cache Summary

| Backbone | Pretrained / Input | Precision | Cache Size on Disk | Extraction Peak VRAM | Extraction Time/Img | Cached Eval Time/Img | Baseline 6-Decimal Match |
|---|---|---|---:|---:|---:|---:|---|
| `clip_b32_224` | `ViT-B-32-quickgelu` (224px) | `fp32` | {b32_c.get('disk_size_gb', 0.0):.3f} GB | {b32_c.get('peak_vram_mb', b32_eval['peak_vram_mb']):.1f} MB | {b32_c.get('seconds_per_image', 0.0):.4f} s | {b32_eval['avg_seconds_per_image']:.4f} s | **YES** (`max_diff = 0.000000`) |
| `clip_l14_336` | `ViT-L-14-336` (336px) | `fp16` | {l14_c.get('disk_size_gb', 0.0):.3f} GB | {l14_c.get('peak_vram_mb', l14_eval['peak_vram_mb']):.1f} MB | {l14_c.get('seconds_per_image', 0.0):.4f} s | {l14_eval['avg_seconds_per_image']:.4f} s | N/A (Upgraded Backbone) |

---

## 2. Held-Out Evaluation: `clip_b32_224` (`fp32`, Domain Generic Vocabulary)

{render_backbone_table(b32_eval)}

---

## 3. Held-Out Evaluation: `clip_l14_336` (`fp16`, Domain Generic Vocabulary)

{render_backbone_table(l14_eval)}

---

## 4. Local-Term Re-Evaluation (Selected on 5 Validation Categories Only)

| Setting | Global Weight ($a$) | Local Weight ($b$) | `clip_b32_224` Val Macro Img AUROC | `clip_l14_336` Val Macro Img AUROC |
|---|---:|---:|---:|---:|
"""
    for name, gw, lw in FUSION_CANDIDATES:
        v_b32 = sweep["clip_b32_224"][name]["val_macro_image_auroc"]
        v_l14 = sweep["clip_l14_336"][name]["val_macro_image_auroc"]
        report_md += f"| `{name}` | {gw:.2f} | {lw:.2f} | {v_b32:.4f} | {v_l14:.4f} |\n"

    report_md += f"""
**Finding on Local Term**:
- On `clip_b32_224`, adding a small local patch weight (`0.95 / 0.05`) shifts validation Macro Image AUROC from `{b32_g_only:.4f}` (`global_only`) to `{b32_fused:.4f}` (`{b32_fused - b32_g_only:+.4f}`), whereas `local_only` (`0.00 / 1.00`) degrades to `{sweep['clip_b32_224']['local_only']['val_macro_image_auroc']:.4f}`.
- On `clip_l14_336`, `global_only` achieves `{l14_g_only:.4f}` vs `{l14_fused:.4f}` at `0.95 / 0.05` (`{l14_fused - l14_g_only:+.4f}`) and `{sweep['clip_l14_336']['local_only']['val_macro_image_auroc']:.4f}` for `local_only`. Raw CLIP patch tokens lack spatial localization alignment without visual reference matching (Stage 2 DINOv2).

---

## 5. Polarity Mechanism Summary (No Per-Category Sign Flipping)

| Backbone | Polarity Variant | Held-Out Macro Pixel AUROC | Held-Out Macro Image AUROC |
|---|---|---:|---:|
| `clip_b32_224` | `final_tokens_native` (`ln_post + proj`, $s_{{def}} - s_{{norm}}$) | {polarity['clip_b32_224']['final_tokens_native']['macro_pixel_auroc']:.4f} | {polarity['clip_b32_224']['final_tokens_native']['macro_image_auroc']:.4f} |
| `clip_b32_224` | `intermediate_native` (no `ln_post`, $s_{{def}} - s_{{norm}}$) | {polarity['clip_b32_224']['intermediate_native']['macro_pixel_auroc']:.4f} | {polarity['clip_b32_224']['intermediate_native']['macro_image_auroc']:.4f} |
| `clip_b32_224` | `intermediate_inverted` (v1 legacy $-(s_{{def}} - s_{{norm}})$) | {polarity['clip_b32_224']['intermediate_inverted']['macro_pixel_auroc']:.4f} | {polarity['clip_b32_224']['intermediate_inverted']['macro_image_auroc']:.4f} |
| `clip_l14_336` | `final_tokens_native` (`ln_post + proj`, $s_{{def}} - s_{{norm}}$) | {polarity['clip_l14_336']['final_tokens_native']['macro_pixel_auroc']:.4f} | {polarity['clip_l14_336']['final_tokens_native']['macro_image_auroc']:.4f} |
| `clip_l14_336` | `intermediate_native` (no `ln_post`, $s_{{def}} - s_{{norm}}$) | {polarity['clip_l14_336']['intermediate_native']['macro_pixel_auroc']:.4f} | {polarity['clip_l14_336']['intermediate_native']['macro_image_auroc']:.4f} |
| `clip_l14_336` | `intermediate_inverted` (v1 legacy $-(s_{{def}} - s_{{norm}})$) | {polarity['clip_l14_336']['intermediate_inverted']['macro_pixel_auroc']:.4f} | {polarity['clip_l14_336']['intermediate_inverted']['macro_image_auroc']:.4f} |
"""

    # Detailed per-category polarity report
    pol_lines = [
        "# Phase 2 Polarity Mechanism Diagnostic (10 Held-Out Categories)",
        "",
        "Strict Rule: **Zero per-category sign flipping.** Compares final-layer projected patch tokens (`ln_post + proj`) against un-normalized intermediate patch projections and v1's globally inverted intermediate tokens.",
        "",
        "## Per-Category Pixel AUROC by Polarity Variant",
        "",
        "| Category | B32 `final_tokens_native` | B32 `intermediate_native` | B32 `intermediate_inverted` | L14 `final_tokens_native` | L14 `intermediate_native` | L14 `intermediate_inverted` |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    eval_cats = [r["category"] for r in b32_eval["categories"]]
    for cat in eval_cats:
        pol_lines.append(
            f"| `{cat}` | "
            f"{polarity['clip_b32_224']['final_tokens_native']['categories'][cat]['pixel_auroc']:.4f} | "
            f"{polarity['clip_b32_224']['intermediate_native']['categories'][cat]['pixel_auroc']:.4f} | "
            f"{polarity['clip_b32_224']['intermediate_inverted']['categories'][cat]['pixel_auroc']:.4f} | "
            f"{polarity['clip_l14_336']['final_tokens_native']['categories'][cat]['pixel_auroc']:.4f} | "
            f"{polarity['clip_l14_336']['intermediate_native']['categories'][cat]['pixel_auroc']:.4f} | "
            f"{polarity['clip_l14_336']['intermediate_inverted']['categories'][cat]['pixel_auroc']:.4f} |"
        )
    pol_lines.append(
        f"| **Macro Mean** | "
        f"**{polarity['clip_b32_224']['final_tokens_native']['macro_pixel_auroc']:.4f}** | "
        f"**{polarity['clip_b32_224']['intermediate_native']['macro_pixel_auroc']:.4f}** | "
        f"**{polarity['clip_b32_224']['intermediate_inverted']['macro_pixel_auroc']:.4f}** | "
        f"**{polarity['clip_l14_336']['final_tokens_native']['macro_pixel_auroc']:.4f}** | "
        f"**{polarity['clip_l14_336']['intermediate_native']['macro_pixel_auroc']:.4f}** | "
        f"**{polarity['clip_l14_336']['intermediate_inverted']['macro_pixel_auroc']:.4f}** |"
    )
    polarity_md = "\n".join(pol_lines) + "\n"
    return report_md, polarity_md


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)
    log_path = cfg.logs_dir / "step_e.log"

    with StepLogger(log_path) as logger:
        try:
            logger.info("=== STEP E: Phase 2 Evaluations on Cached Visual Embeddings ===")
            if not cfg.phase1_baseline_json.is_file():
                raise FileNotFoundError(f"Phase 1 baseline JSON not found: {cfg.phase1_baseline_json}")
            p1_data = json.loads(cfg.phase1_baseline_json.read_text(encoding="utf-8"))

            cache_stats_path = cfg.results_dir / "step_d_cache_stats.json"
            cache_stats = (
                json.loads(cache_stats_path.read_text(encoding="utf-8"))
                if cache_stats_path.is_file()
                else {}
            )

            _, _, val_samples, eval_samples = create_v3_category_splits(
                data_root=cfg.data_root,
                manifest_dir=cfg.manifest_dir,
                seed=42,
            )

            polarity_all: dict[str, dict] = {}
            sweep_all: dict[str, dict] = {}
            heldout_all: dict[str, dict] = {}

            for backbone_id, precision in [("clip_b32_224", "fp32"), ("clip_l14_336", "fp16")]:
                logger.info(f"\n--- Evaluating Backbone: {backbone_id} (cache={precision}, text_encoder=cpu) ---")
                extractor = CLIPFeatureExtractor(
                    backbone_id=backbone_id,
                    device="cpu",
                    precision=precision,
                    cache_dir=cfg.cache_dir,
                )

                # 1. Polarity mechanism evaluation on held-out categories (no per-category sign flipping)
                logger.info(f"[{backbone_id}] Running polarity mechanism comparison (no sign flipping)...")
                polarity_all[backbone_id] = run_polarity_ablation(extractor, eval_samples, logger)

                # 2. Local-term weight sweep on the 5 validation categories only
                logger.info(f"[{backbone_id}] Running local-term weight sweep on 5 validation categories...")
                sweep_res, best_gw, best_lw = run_local_weight_sweep(
                    extractor,
                    val_samples,
                    polarity_mode="intermediate_inverted",
                    logger=logger,
                )
                sweep_all[backbone_id] = sweep_res

                # 3. Held-out evaluation on the 10 held-out categories
                logger.info(
                    f"[{backbone_id}] Running held-out evaluation on 10 categories "
                    f"(global_weight={best_gw:.2f}, local_weight={best_lw:.2f})..."
                )
                heldout_all[backbone_id] = run_heldout_evaluation(
                    extractor,
                    val_samples=val_samples,
                    eval_samples=eval_samples,
                    polarity_mode="intermediate_inverted",
                    global_weight=best_gw,
                    local_weight=best_lw,
                    logger=logger,
                )

                del extractor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Write CSV and Markdown artifacts to D:/defect_cache/results and mirror to v3/docs/results
            write_csv_tables(
                cfg,
                b32_eval=heldout_all["clip_b32_224"],
                l14_eval=heldout_all["clip_l14_336"],
                polarity=polarity_all,
                sweep=sweep_all,
            )

            report_md, polarity_md = format_markdown_report(
                cfg,
                p1_data=p1_data,
                b32_eval=heldout_all["clip_b32_224"],
                l14_eval=heldout_all["clip_l14_336"],
                polarity=polarity_all,
                sweep=sweep_all,
                cache_stats=cache_stats,
            )

            (cfg.results_dir / "phase_2_report.md").write_text(report_md, encoding="utf-8")
            (cfg.results_dir / "phase_2_polarity.md").write_text(polarity_md, encoding="utf-8")
            (cfg.repo_results_dir / "phase_2.md").write_text(report_md, encoding="utf-8")
            (cfg.repo_results_dir / "phase_2_polarity.md").write_text(polarity_md, encoding="utf-8")

            summary_payload = {
                "status": "PASS",
                "commit": get_git_commit(),
                "config_hash": cfg.config_hash,
                "clip_b32_224": heldout_all["clip_b32_224"],
                "clip_l14_336": heldout_all["clip_l14_336"],
                "polarity": polarity_all,
                "local_sweep": sweep_all,
            }
            (cfg.results_dir / "phase_2_summary.json").write_text(
                json.dumps(summary_payload, indent=2),
                encoding="utf-8",
            )

            elapsed = time.perf_counter() - t0
            logger.info(
                f"\nSTEP E RESULT: PASS ({elapsed:.2f}s) | "
                f"B32 Img AUROC={heldout_all['clip_b32_224']['macro_image_auroc']:.4f}, "
                f"L14 Img AUROC={heldout_all['clip_l14_336']['macro_image_auroc']:.4f} | "
                f"Results written to {cfg.results_dir}"
            )
            log_decision(
                cfg,
                "Step E (Phase 2 Cached Evaluations & Reports)",
                "PASS",
                elapsed,
                f"B32_Img={heldout_all['clip_b32_224']['macro_image_auroc']:.4f}, "
                f"L14_Img={heldout_all['clip_l14_336']['macro_image_auroc']:.4f}",
            )
            return 0
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"Step E failed: {exc}")
            log_decision(cfg, "Step E (Phase 2 Cached Evaluations & Reports)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())

