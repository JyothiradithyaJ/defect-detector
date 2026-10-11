"""Score-to-probability logistic calibration on labelled validation data.

NOTE: Calibration requires labelled validation samples (normal + anomalous) and is therefore
calibrated, not strictly zero-shot. Raw model outputs are anomaly scores; outputs are called
calibrated probabilities only after fitting LogisticCalibrator on validation categories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v3.eval.metrics import expected_calibration_error


def sigmoid_np(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    res = 1.0 / (1.0 + np.exp(-arr))
    if np.isscalar(x):
        return float(res)
    return res.astype(np.float32)


@dataclass
class LogisticCalibrator:
    """Temperature (T) and intercept (bias) logistic calibrator: P(defect) = sigmoid(score / T + bias)."""

    temperature: float = 1.0
    bias: float = 0.0
    is_calibrated: bool = False

    def fit(self, scores: list[float] | np.ndarray, labels: list[int] | np.ndarray, max_iter: int = 250) -> tuple[float, float]:
        """Fit (temperature, bias) on labelled validation data via Newton / gradient descent on BCE."""
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if s.size == 0 or len(np.unique(y)) < 2:
            self.is_calibrated = False
            return self.temperature, self.bias

        log_t = 0.0
        b = 0.0
        lr = 0.05

        for _ in range(max_iter):
            t = np.exp(np.clip(log_t, -6.0, 6.0))
            logits = s / t + b
            probs = np.asarray(sigmoid_np(logits), dtype=np.float64)
            diff = probs - y

            grad_log_t = float(np.mean(diff * (-s / t))) + 1e-4 * log_t
            grad_b = float(np.mean(diff)) + 1e-4 * b

            log_t -= lr * np.clip(grad_log_t, -5.0, 5.0)
            b -= lr * np.clip(grad_b, -5.0, 5.0)

        self.temperature = float(np.exp(np.clip(log_t, -6.0, 6.0)))
        self.bias = float(b)
        self.is_calibrated = True
        return self.temperature, self.bias

    def predict_proba(self, score: float) -> float:
        """Convert raw anomaly score into calibrated probability (only after calibration)."""
        logit = float(score) / max(self.temperature, 1e-6) + self.bias
        return float(sigmoid_np(logit))

    def evaluate_ece(self, scores: list[float], labels: list[int], num_bins: int = 10) -> float:
        probs = [self.predict_proba(x) for x in scores]
        return expected_calibration_error(probs, labels, bins=num_bins)
