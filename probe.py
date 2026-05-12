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
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


RNG_SEED = 42
GEOMETRIC_FEATURE_DIM = 66


def _safe_pca_dim(n_samples: int, n_features: int) -> int:
    semantic_dim = max(1, n_features - GEOMETRIC_FEATURE_DIM)
    return int(max(1, min(48, semantic_dim, n_samples - 1)))


def _build_estimator(n_samples: int, n_features: int) -> Pipeline:
    pca_dim = _safe_pca_dim(n_samples, n_features)

    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        (
                            "geom",
                            RobustScaler(),
                            slice(0, min(GEOMETRIC_FEATURE_DIM, n_features)),
                        ),
                        (
                            "semantic",
                            Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    (
                                        "pca",
                                        PCA(
                                            n_components=pca_dim,
                                            svd_solver="randomized",
                                            random_state=RNG_SEED,
                                        ),
                                    ),
                                ]
                            ),
                            slice(min(GEOMETRIC_FEATURE_DIM, n_features), n_features),
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    C=0.03,
                    max_iter=3000,
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
        self._calibrator: LogisticRegression | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        raise RuntimeError(
            "This probe routes inference through sklearn; use predict / predict_proba."
        )

    def _decision_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Probe is not fitted.")
        if hasattr(self._model, "decision_function"):
            scores = self._model.decision_function(X)
        else:
            probs = self._model.predict_proba(X)[:, 1]
            eps = np.finfo(np.float64).eps
            scores = np.log(np.clip(probs, eps, 1.0 - eps) / np.clip(1.0 - probs, eps, 1.0))
        return np.asarray(scores, dtype=np.float64).reshape(-1, 1)

    @staticmethod
    def _select_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        candidates = np.linspace(0.10, 0.90, 161)
        best_threshold = 0.5
        best_key = (-1.0, -1.0, -1.0, -abs(best_threshold - 0.5))
        target_rate = float(np.mean(y_true))

        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            positive_rate = float(y_pred_t.mean())
            if positive_rate < 0.05 or positive_rate > 0.95:
                continue
            acc = accuracy_score(y_true, y_pred_t)
            f1 = f1_score(y_true, y_pred_t, zero_division=0)
            balance_score = -abs(positive_rate - target_rate)
            key = (acc, f1, balance_score, -abs(float(t) - 0.5))
            if key > best_key:
                best_key = key
                best_threshold = float(t)

        return best_threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        indices = np.arange(X.shape[0])
        _, class_counts = np.unique(y, return_counts=True)
        can_calibrate = X.shape[0] >= 40 and class_counts.size == 2 and class_counts.min() >= 6

        if can_calibrate:
            idx_fit, idx_cal = train_test_split(
                indices,
                test_size=0.18,
                stratify=y,
                random_state=RNG_SEED,
            )
        else:
            idx_fit = indices
            idx_cal = np.array([], dtype=np.int64)

        self._model = _build_estimator(idx_fit.size, X.shape[1])
        self._model.fit(X[idx_fit], y[idx_fit])

        self._calibrator = None
        if idx_cal.size > 0 and np.unique(y[idx_cal]).size == 2:
            self._calibrator = LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=1000,
                random_state=RNG_SEED,
            )
            self._calibrator.fit(self._decision_scores(X[idx_cal]), y[idx_cal])
            calib_probs = self.predict_proba(X[idx_cal])[:, 1]
            self._threshold = self._select_threshold(calib_probs, y[idx_cal])
        else:
            self._threshold = 0.5

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        if self._model is None:
            raise RuntimeError("Call fit before fit_hyperparameters.")

        probs = self.predict_proba(np.asarray(X_val, dtype=np.float32))[:, 1]
        y_val = np.asarray(y_val, dtype=np.int64)

        self._threshold = self._select_threshold(probs, y_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Probe is not fitted.")
        X = np.asarray(X, dtype=np.float32)
        if self._calibrator is not None:
            proba = self._calibrator.predict_proba(self._decision_scores(X))
        else:
            proba = self._model.predict_proba(X)
        return proba.astype(np.float64)
