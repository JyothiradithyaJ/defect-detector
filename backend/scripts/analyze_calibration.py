"""Inspect raw calibration logits independently of the project pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


def parse_arguments() -> argparse.Namespace:
    """Read locations of the diagnostic arrays."""
    parser = argparse.ArgumentParser(
        description="Diagnose raw logits used for temperature calibration."
    )
    parser.add_argument("--logits", type=Path, default=Path("calib_logits.npy"))
    parser.add_argument("--labels", type=Path, default=Path("calib_labels.npy"))
    return parser.parse_args()


def load_arrays(logits_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate matching one-dimensional logits and binary labels."""
    logits = np.asarray(np.load(logits_path), dtype=np.float64).reshape(-1)
    labels = np.asarray(np.load(labels_path), dtype=np.int64).reshape(-1)

    if logits.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: logits {logits.shape} vs labels {labels.shape}."
        )
    if len(labels) == 0:
        raise ValueError("Diagnostic arrays must not be empty.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Labels must contain both 0 (good) and 1 (defective).")

    return logits, labels


def main() -> None:
    """Print raw-score diagnostics and a likely failure mode."""
    args = parse_arguments()
    logits, labels = load_arrays(args.logits, args.labels)

    good_logits = logits[labels == 0]
    defective_logits = logits[labels == 1]
    auroc = float(roc_auc_score(labels, logits)) if roc_auc_score else None

    print("=" * 60)
    print("CALIBRATION LOGIT DIAGNOSTIC")
    print("=" * 60)
    print(f"Total samples:         {len(labels)}")
    print(f"  Good (label=0):      {len(good_logits)}")
    print(f"  Defective (label=1): {len(defective_logits)}")
    print("-" * 60)
    print(f"Logit mean | label=0 (good):      {good_logits.mean():.6f}")
    print(f"Logit mean | label=1 (defective): {defective_logits.mean():.6f}")
    print(f"Logit std (overall):               {logits.std():.6f}")
    print(f"Logit min / max:                   {logits.min():.6f} / {logits.max():.6f}")
    print("-" * 60)

    if auroc is None:
        print("scikit-learn not installed; skipping raw AUROC.")
    else:
        print(f"Raw (uncalibrated) AUROC:          {auroc:.4f}")

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    if auroc is not None and auroc < 0.55:
        print("-> AUROC near 0.5: raw logits carry little useful signal.")
        print("   Investigate fusion, prompts, or CLIP image embeddings.")
    elif defective_logits.mean() < good_logits.mean():
        print("-> SIGN INVERSION: defective images score below good images.")
        print("   Check the anomalous-minus-normal subtraction order.")
    elif logits.std() < 1e-3:
        print("-> Logits barely vary across images.")
        print("   Check prompt caching and image-feature generation.")
    elif auroc is not None:
        print("-> Raw AUROC has signal; inspect temperature-fitting behavior.")
    else:
        print("-> Inspect the summary statistics above.")


if __name__ == "__main__":
    main()
