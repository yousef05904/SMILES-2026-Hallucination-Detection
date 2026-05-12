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


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.18,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Stratified K-fold Evaluation with inner stratified validation.

    Args mirror the starter template loosely; ``df`` remains unused — reserved
    for potential group stratification extensions without widening the harness.
    """
    del df

    y = np.asarray(y)
    rng = random_state

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=rng,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    placeholder = np.zeros(len(y), dtype=np.int64)

    for fold_id, (idx_train_full, idx_test) in enumerate(skf.split(placeholder, y)):
        y_outer = y[idx_train_full]

        idx_rel_train, idx_rel_val = train_test_split(
            np.arange(idx_train_full.size, dtype=np.int64),
            test_size=val_size,
            stratify=y_outer,
            random_state=rng + fold_id,
        )

        idx_train = idx_train_full[idx_rel_train]
        idx_val = idx_train_full[idx_rel_val]
        splits.append((idx_train, idx_val, idx_test))

    return splits
