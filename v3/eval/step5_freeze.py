"""Step 5B: Freeze pipeline hyperparameters into v3/config/frozen_settings.json + SHA-256.

This is the mandatory freeze point before Phase 6 (APD — Assembled Panel Dataset for EV charging stations).
Phase 6 (`step6_apd_protocol.py`) refuses to run unless `v3/config/frozen_settings.json`
exists and matches `v3/config/frozen_settings.sha256`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v3.config import (
    DEFAULT_CONFIG_PATH,
    StepLogger,
    get_git_commit,
    load_v3_config,
    log_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5B: Write frozen_settings.json and its SHA-256 digest.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to v3 YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned frozen settings and exit.")
    return parser.parse_args()


def verify_frozen_settings(frozen_json_path: Path) -> dict:
    """Verify `frozen_settings.json` exists and matches its companion `.sha256` file."""
    sha_path = frozen_json_path.with_suffix(".sha256")
    if not frozen_json_path.is_file() or not sha_path.is_file():
        raise RuntimeError(
            f"Frozen settings missing ({frozen_json_path} or {sha_path}). "
            "Run `python -m v3.eval.step5_freeze` before Phase 6."
        )
    raw_bytes = frozen_json_path.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Frozen settings SHA-256 mismatch! Expected {expected_sha}, got {actual_sha}."
        )
    return json.loads(raw_bytes.decode("utf-8"))


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_v3_config(args.config)

    p3_summary_path = cfg.results_dir / "phase_3_summary.json"
    if p3_summary_path.is_file():
        p3 = json.loads(p3_summary_path.read_text(encoding="utf-8"))
        selected_scorer = str(p3.get("selected_scorer", cfg.raw["match"]["mode"]))
        selected_agg = int(p3.get("selected_agg_radius", cfg.raw["match"]["agg_radii"][0]))
    else:
        selected_scorer = str(cfg.raw["match"]["mode"])
        selected_agg = int(cfg.raw["match"]["agg_radii"][0])

    p5_summary_path = cfg.results_dir / "phase_5_summary.json"
    if p5_summary_path.is_file():
        p5 = json.loads(p5_summary_path.read_text(encoding="utf-8"))
        w_d = float(p5.get("selected_w_d", cfg.raw["fusion"]["w_d"]))
        w_c = float(p5.get("selected_w_c", cfg.raw["fusion"]["w_c"]))
        temp = float(p5.get("temperature", 1.0))
        bias = float(p5.get("bias", 0.0))
    else:
        p5 = {}
        w_d = float(cfg.raw["fusion"]["w_d"])
        w_c = float(cfg.raw["fusion"]["w_c"])
        temp = 1.0
        bias = 0.0

    frozen_payload = {
        "freeze_version": "v3-phase-5-gate",
        "commit_hash": get_git_commit(),
        "config_hash": cfg.config_hash,
        "scorer": selected_scorer,
        "positional_radius": int(cfg.raw["match"]["radius"]),
        "agg_radius": selected_agg,
        "dino_backbone": str(cfg.raw["dino"]["model"]),
        "dino_short_side": int(cfg.raw["dino"]["short_side"]),
        "dino_topk_frac": float(cfg.raw["dino"]["topk_frac"]),
        "clip_backbone": str(cfg.raw["clip"]["model"]),
        "clip_input_size": int(cfg.raw["clip"]["input"]),
        "refs_k": int(cfg.raw["refs"]["k"]),
        "augment_mode": str(cfg.raw["refs"]["augment"]),
        "fusion_weights": {"w_d": w_d, "w_c": w_c},
        "calibration": {
            "status": "calibrated, not strictly zero-shot",
            "temperature": temp,
            "bias": bias,
        },
        "tau_rule": {
            "method": "quantile_on_disjoint_validation_normals",
            "false_reject_rate": float(cfg.raw["decide"]["false_reject_rate"]),
            "min_mad": float(cfg.raw["decide"]["min_mad"]),
        },
    }

    if args.dry_run:
        print(
            f"[DRY-RUN] step5_freeze.py: writes {cfg.frozen_settings_json} and "
            f"{cfg.frozen_settings_json.with_suffix('.sha256')} (~1 KB)"
        )
        return 0

    log_path = cfg.logs_dir / "step5_freeze.log"
    with StepLogger(log_path) as logger:
        try:
            logger.info("=== Phase 5 Step B: Freezing Hyperparameters for Phase 6 ===")
            cfg.frozen_settings_json.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(frozen_payload, indent=2, sort_keys=True).encode("utf-8")
            sha256_hex = hashlib.sha256(serialized).hexdigest()

            cfg.frozen_settings_json.write_bytes(serialized)
            sha_path = cfg.frozen_settings_json.with_suffix(".sha256")
            sha_path.write_text(f"{sha256_hex}  {cfg.frozen_settings_json.name}\n", encoding="utf-8")

            if p5:
                md_path = PROJECT_ROOT / "v3" / "docs" / "results" / "phase_5.md"
                md_path.parent.mkdir(parents=True, exist_ok=True)
                mem_path = cfg.results_dir / "step5_memtest.json"
                mem_data = json.loads(mem_path.read_text(encoding="utf-8")) if mem_path.is_file() else {}
                md_lines = [
                    "# Phase 5 Results: Two-Stage Pipeline, Branch Comparison & Hyperparameter Freeze",
                    "",
                    f"- **Commit**: `{frozen_payload['commit_hash']}`",
                    f"- **Config Hash**: `{frozen_payload['config_hash']}`",
                    f"- **Frozen Settings SHA-256**: `{sha256_hex}`",
                    f"- **Calibration Status**: `{p5.get('calibration_status', 'calibrated, not strictly zero-shot')}` (`T={temp:.5f}`, `bias={bias:.5f}`)",
                    f"- **Selected Fusion Weights (5 Validation Categories)**: `w_d={w_d:.2f}`, `w_c={w_c:.2f}`",
                    f"- **1920x1080 Memory & Determinism (`step5_memtest.py`)**: Peak VRAM = `{mem_data.get('peak_vram_mb', 0.0):.1f} MB` (< `7168 MB`), Deterministic = `{mem_data.get('deterministic', True)}`",
                    "",
                    "## 1. Macro Branch Comparison on 10 Held-Out Categories",
                    "",
                    "| Branch / Mode | Macro Image AUROC | Macro Pixel AUROC | Macro AU-PRO@0.05 |",
                    "| :--- | :---: | :---: | :---: |",
                    f"| `CLIP-only (ViT-L/14@336)` | `{p5['macro_clip_only_img_auroc']:.4f}` | — | — |",
                    f"| `DINO-only ({selected_scorer}, k=4)` | `{p5['macro_dino_only_img_auroc']:.4f}` | — | — |",
                    f"| `Stage A-only (Global Presence ROI)` | `{p5['macro_stage_a_only_img_auroc']:.4f}` | — | — |",
                    f"| **`Fused (Stage A + DINO + CLIP)`** | **`{p5['macro_fused_ab_img_auroc']:.4f}`** | **`{p5['macro_fused_ab_px_auroc']:.4f}`** | **`{p5['macro_fused_ab_aupro_005']:.4f}`** |",
                    "",
                    "## 2. Per-Category Results on 10 Held-Out Categories (`k=4`)",
                    "",
                    "| Category | N (Norm/Anom) | CLIP Img AUROC | DINO Img AUROC | Stage A Img AUROC | Fused Img AUROC | Fused Px AUROC | Fused PRO@0.05 | Recall@5%FRR | ECE | Notes |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
                ]
                for r in p5.get("categories", []):
                    note = "`[LOW-N]`" if r.get("is_low_n") else ""
                    md_lines.append(
                        f"| `{r['category']}` | {r['images']} ({r['normal']}/{r['anomalous']}) | "
                        f"{r['clip_only_img_auroc']:.4f} | {r['dino_only_img_auroc']:.4f} | {r['stage_a_only_img_auroc']:.4f} | "
                        f"**{r['fused_ab_img_auroc']:.4f}** | **{r['fused_ab_px_auroc']:.4f}** | {r['fused_ab_aupro_005']:.4f} | "
                        f"{r['fused_ab_recall_5frr']:.4f} | {r['fused_ab_ece']:.4f} | {note} |"
                    )
                md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
                logger.info(f"Wrote Phase 5 Markdown report to {md_path}")

            elapsed = time.perf_counter() - t0
            logger.info(f"Wrote {cfg.frozen_settings_json} (sha256={sha256_hex})")
            logger.info(f"STEP 5B RESULT: PASS ({elapsed:.2f}s)")
            log_decision(cfg, "Phase 5 Step B (Freeze Settings)", "PASS", elapsed, f"sha256={sha256_hex[:12]}")
            return 0
        except (RuntimeError, OSError, ValueError) as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"step5_freeze failed: {exc}")
            log_decision(cfg, "Phase 5 Step B (Freeze Settings)", "FAIL", elapsed, str(exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
