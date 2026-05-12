"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` returns stratified K-fold outer splits; each fold holds out a
disjoint test block and carves a stratified validation slice from the outer
training pool for threshold tuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _make_groups(y: np.ndarray, df: pd.DataFrame | None) -> np.ndarray:
    """Group repeated prompts so related examples never span train/val/test."""
    if df is None or len(df) != len(y):
        return np.arange(len(y), dtype=np.int64)

    if "prompt" in df.columns:
        keys = df["prompt"].fillna("").astype(str)
    elif "response" in df.columns:
        keys = df["response"].fillna("").astype(str)
    else:
        return np.arange(len(y), dtype=np.int64)

    return pd.factorize(keys, sort=False)[0].astype(np.int64)


def _safe_n_splits(
    y: np.ndarray,
    requested: int,
    groups: np.ndarray | None = None,
) -> int:
    _, counts = np.unique(y, return_counts=True)
    group_limit = np.unique(groups).size if groups is not None else len(y)
    if counts.size < 2:
        return int(max(2, min(requested, group_limit)))
    return int(max(2, min(requested, counts.min(), group_limit)))


def _splitter(y: np.ndarray, groups: np.ndarray, n_splits: int, random_state: int):
    if np.unique(groups).size < len(groups):
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        ).split(np.zeros(len(y), dtype=np.int64), y, groups)

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    ).split(np.zeros(len(y), dtype=np.int64), y)


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.18,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Deterministic stratified folds with leak-safe validation.

    When ``df`` contains prompt text, repeated prompts are kept in the same group
    across the outer test split and inner validation split.
    """
    y = np.asarray(y)
    groups = _make_groups(y, df)
    outer_splits = _safe_n_splits(y, n_splits, groups)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_id, (idx_train_full, idx_test) in enumerate(
        _splitter(y, groups, outer_splits, random_state)
    ):
        y_outer = y[idx_train_full]
        groups_outer = groups[idx_train_full]
        inner_requested = max(2, round(1.0 / val_size))
        inner_splits = _safe_n_splits(y_outer, inner_requested, groups_outer)

        inner_iter = _splitter(
            y_outer,
            groups_outer,
            inner_splits,
            random_state + 10_000 + fold_id,
        )
        inner_fold = fold_id % inner_splits
        for inner_id, (idx_rel_train, idx_rel_val) in enumerate(inner_iter):
            if inner_id == inner_fold:
                break

        idx_train = idx_train_full[idx_rel_train]
        idx_val = idx_train_full[idx_rel_val]
        splits.append((idx_train, idx_val, idx_test))

    return splits
