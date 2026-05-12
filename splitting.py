"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` returns stratified K-fold outer splits; each fold holds out a
disjoint test block and carves a stratified validation slice from the outer
training pool for threshold tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _safe_n_splits(
    y: np.ndarray,
    requested: int,
) -> int:
    _, counts = np.unique(y, return_counts=True)
    if counts.size < 2:
        return 2
    return int(max(2, min(requested, counts.min())))


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.18,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Deterministic stratified folds with held-out validation.

    The outer test fold is never reused for fitting or threshold tuning. The
    validation slice is carved only from the corresponding outer training pool.
    """
    del df

    y = np.asarray(y)
    outer_splits = _safe_n_splits(y, n_splits)
    skf = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    placeholder = np.zeros(len(y), dtype=np.int64)
    for fold_id, (idx_train_full, idx_test) in enumerate(skf.split(placeholder, y)):
        y_outer = y[idx_train_full]

        idx_rel_train, idx_rel_val = train_test_split(
            np.arange(idx_train_full.size, dtype=np.int64),
            test_size=val_size,
            stratify=y_outer,
            random_state=random_state + fold_id,
        )

        idx_train = idx_train_full[idx_rel_train]
        idx_val = idx_train_full[idx_rel_val]
        splits.append((idx_train, idx_val, idx_test))

    return splits
