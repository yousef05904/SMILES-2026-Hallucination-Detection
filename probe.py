"""
probe.py — Hallucination probe classifier (student-implemented).

``HallucinationProbe`` subclasses ``torch.nn.Module`` for compatibility with
the competition harness, while the discriminator itself is implemented with
deterministic ``scikit-learn`` ensembles (tab-friendly for ~10k–dim features).

Public API: ``fit``, ``fit_hyperparameters``, ``predict``, ``predict_proba``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RNG_SEED = 42


def _safe_pca_dim(n_samples: int, n_features: int) -> int:
    upper = min(n_features, max(1, n_samples - 2))
    target = min(upper, max(12, n_samples // 4))
    return int(max(8, min(target, 160)))


def _build_estimator(n_samples: int, n_features: int) -> VotingClassifier:
    pca_lr = _safe_pca_dim(n_samples, n_features)
    pca_hgb = max(24, min(96, _safe_pca_dim(n_samples, n_features)))

    lr_path = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=pca_lr,
                    svd_solver="randomized",
                    random_state=RNG_SEED,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    C=1.5,
                    max_iter=2400,
                    class_weight="balanced",
                    random_state=RNG_SEED,
                ),
            ),
        ]
    )

    et = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=220,
                    max_depth=None,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    bootstrap=False,
                    class_weight="balanced_subsample",
                    random_state=RNG_SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )

    hgbt = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=pca_hgb,
                    svd_solver="randomized",
                    random_state=RNG_SEED,
                ),
            ),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    max_iter=260,
                    learning_rate=0.07,
                    l2_regularization=1e-4,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=20,
                    random_state=RNG_SEED,
                ),
            ),
        ]
    )

    # ExtraTrees excels on heterogeneous nonlinearities; PCA+LR anchors linear
    # structure; calibrated boosting via early stopping catches residual signal.
    return VotingClassifier(
        estimators=[
            ("lr_pca", lr_path),
            ("extra_trees", et),
            ("hgbt_pc", hgbt),
        ],
        voting="soft",
        weights=[1.45, 1.05, 1.0],
        n_jobs=1,
    )


class HallucinationProbe(nn.Module):
    """Sklearn-soft-voting ensemble behind a lightweight ``nn.Module`` shell."""

    def __init__(self) -> None:
        super().__init__()
        self._threshold: float = 0.5
        self._model: VotingClassifier | None = None

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

        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 101)])
        )

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
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
