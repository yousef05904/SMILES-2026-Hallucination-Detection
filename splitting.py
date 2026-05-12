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
) -> int:
    _, counts = np.unique(y, return_counts=True)
    if counts.size < 2:
        return 2
    return int(max(2, min(requested, counts.min())))


def _splitter(y: np.ndarray, groups: np.ndarray, n_splits: int, random_state: int):
    if _should_group_split(y, groups, n_splits):
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


def _should_group_split(y: np.ndarray, groups: np.ndarray, n_splits: int) -> bool:
    """Use group folds only when repeated groups are numerous and well balanced."""
    unique_groups, group_inverse, group_counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )
    if unique_groups.size == len(groups):
        return False

    if unique_groups.size < n_splits:
        return False

    # A very large prompt group can destabilize fold balance more than it helps.
    if group_counts.max() > max(2, len(y) // n_splits):
        return False

    labels = np.unique(y)
    for label in labels:
        label_group_count = np.unique(group_inverse[y == label]).size
        if label_group_count < n_splits:
            return False

    return True


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.18,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Deterministic stratified folds with leak-safe validation.

    When prompt grouping is sufficiently balanced, repeated prompts are kept in
    the same group. Otherwise the function falls back to normal stratified folds.
    """
    y = np.asarray(y)
    groups = _make_groups(y, df)
    outer_splits = _safe_n_splits(y, n_splits)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_id, (idx_train_full, idx_test) in enumerate(
        _splitter(y, groups, outer_splits, random_state)
    ):
        y_outer = y[idx_train_full]
        groups_outer = groups[idx_train_full]
        inner_requested = max(2, round(1.0 / val_size))
        inner_splits = _safe_n_splits(y_outer, inner_requested)

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
