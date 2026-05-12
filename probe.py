"""
probe.py — Hallucination probe classifier (student-implemented).

``HallucinationProbe`` subclasses ``torch.nn.Module`` for compatibility with
the competition harness, while the discriminator itself is implemented as a
small deterministic scikit-learn pipeline.

Public API: ``fit``, ``fit_hyperparameters``, ``predict``, ``predict_proba``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RNG_SEED = 42


def _safe_pca_dim(n_samples: int, n_features: int) -> int:
    """Choose a conservative PCA size from the training fold only."""
    upper = min(n_features, max(1, n_samples - 2))
    target = min(upper, max(8, n_samples // 5), 96)
    return int(max(1, target))


def _build_estimator(n_samples: int, n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=_safe_pca_dim(n_samples, n_features),
                    svd_solver="randomized",
                    random_state=RNG_SEED,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    C=0.08,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RNG_SEED,
                ),
            ),
        ]
    )


class HallucinationProbe(nn.Module):
    """Regularized sklearn classifier behind a lightweight ``nn.Module`` shell."""

    def __init__(self) -> None:
        super().__init__()
        self._threshold: float = 0.5
        self._model: Pipeline | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        raise RuntimeError(
            "This probe routes inference through sklearn; use predict / predict_proba."
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self._model = _build_estimator(X.shape[0], X.shape[1])
        self._model.fit(X, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        if self._model is None:
            raise RuntimeError("Call fit before fit_hyperparameters.")

        probs = self.predict_proba(np.asarray(X_val, dtype=np.float32))[:, 1]
        y_val = np.asarray(y_val, dtype=np.int64)

        candidates = np.linspace(0.25, 0.75, 101)

        best_threshold = 0.5
        best_key = (-1.0, -1.0, -abs(best_threshold - 0.5))
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            acc = accuracy_score(y_val, y_pred_t)
            f1 = f1_score(y_val, y_pred_t, zero_division=0)
            key = (acc, f1, -abs(float(t) - 0.5))
            if key > best_key:
                best_key = key
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Probe is not fitted.")
        X = np.asarray(X, dtype=np.float32)
        proba = self._model.predict_proba(X)
        return proba.astype(np.float64)
