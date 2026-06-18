"""
Time-series cross-validation splitter with PIT (point-in-time) safety.

Borrowed inspiration: Qlib's rolling-window training framework
(`qlib.contrib.data.rolling`), with the addition of an explicit "purge gap"
to prevent look-ahead leakage between train and validation sets.

Two splitter flavors are provided:

1. ``TimeSeriesCV`` – simple sliding window of fixed train/valid/test
   lengths, with optional purge gap and embargo.

2. ``PurgedKFold`` – K folds where each fold's training set is purged
   for ``purge_gap`` days after the validation boundary, mimicking the
   "embargo" technique from Lopez de Prado's "Advances in Financial
   Machine Learning".

A ``leakage_check`` helper detects common leakage patterns such as
future-dated features or duplicated timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class CVSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    fold_id: int
    train_period: Tuple[pd.Timestamp, pd.Timestamp]
    valid_period: Tuple[pd.Timestamp, pd.Timestamp]
    test_period: Tuple[pd.Timestamp, pd.Timestamp]


class TimeSeriesCV:
    """Sliding window CV for time-series panels.

    Parameters
    ----------
    train_size : int
        Number of rows (or periods) in each training window.
    valid_size : int
        Number of rows in each validation window.
    test_size  : int
        Number of rows in each test window.
    step       : int
        How many rows to advance between successive splits.
    purge_gap  : int
        Number of rows to drop between train and valid/test sets to
        prevent look-ahead leakage.
    embargo    : int
        Additional rows to drop at the start of each test set.
    """

    def __init__(self, train_size: int, valid_size: int, test_size: int,
                 step: int = 1, purge_gap: int = 0, embargo: int = 0):
        if train_size <= 0 or valid_size <= 0 or test_size <= 0:
            raise ValueError("train/valid/test sizes must be positive")
        if step <= 0:
            raise ValueError("step must be positive")
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.step = step
        self.purge_gap = purge_gap
        self.embargo = embargo

    def split(self, dates: Sequence[pd.Timestamp]) -> Iterator[CVSplit]:
        dates = pd.Series(sorted(pd.to_datetime(list(dates)))).drop_duplicates().reset_index(drop=True)
        n = len(dates)
        window = self.train_size + self.purge_gap + self.valid_size + self.test_size
        if n < window:
            return iter(())
        fold_id = 0
        start = 0
        while start + window <= n:
            train_end = start + self.train_size
            valid_start = train_end + self.purge_gap
            valid_end = valid_start + self.valid_size
            test_end = valid_end + self.test_size

            train_dates = dates.iloc[start:train_end]
            valid_dates = dates.iloc[valid_start:valid_end]
            test_dates = dates.iloc[valid_end:test_end]
            if self.embargo:
                test_dates = test_dates.iloc[self.embargo:]

            yield CVSplit(
                train=pd.DataFrame({"date": train_dates.values}),
                valid=pd.DataFrame({"date": valid_dates.values}),
                test=pd.DataFrame({"date": test_dates.values}),
                fold_id=fold_id,
                train_period=(train_dates.iloc[0], train_dates.iloc[-1]),
                valid_period=(valid_dates.iloc[0], valid_dates.iloc[-1]),
                test_period=(test_dates.iloc[0] if not test_dates.empty else valid_dates.iloc[-1],
                             test_dates.iloc[-1] if not test_dates.empty else valid_dates.iloc[-1]),
            )
            fold_id += 1
            start += self.step

    def n_splits(self, n_dates: int) -> int:
        window = self.train_size + self.purge_gap + self.valid_size + self.test_size
        if n_dates < window:
            return 0
        return ((n_dates - window) // self.step) + 1


@dataclass
class PurgedFold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold_id: int


class PurgedKFold:
    """K-Fold with a purge gap between train and test (Lopez de Prado)."""

    def __init__(self, n_splits: int = 5, purge_gap: int = 0, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo = embargo

    def split(self, n_samples: int) -> Iterator[PurgedFold]:
        if n_samples <= 0:
            return
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        indices = np.arange(n_samples)
        for k in range(self.n_splits):
            test_start = sum(fold_sizes[:k])
            test_end = test_start + fold_sizes[k]

            train_mask = np.ones(n_samples, dtype=bool)
            test_mask = np.zeros(n_samples, dtype=bool)

            # mark test set
            test_mask[test_start:test_end] = True
            # remove test from training
            train_mask[test_start:test_end] = False
            # purge gap: also remove `purge_gap` rows adjacent to test
            if self.purge_gap > 0:
                purge_lo = max(0, test_start - self.purge_gap)
                purge_hi = min(n_samples, test_end + self.purge_gap)
                train_mask[purge_lo:purge_hi] = False
            # embargo at the beginning of test
            if self.embargo > 0 and test_start < self.embargo:
                keep_from = min(self.embargo, n_samples)
                train_mask[:keep_from] = False

            train_idx = indices[train_mask]
            test_idx = indices[test_mask]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield PurgedFold(train_idx=train_idx, test_idx=test_idx, fold_id=k)


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

class LeakageReport:
    def __init__(self):
        self.issues: List[str] = []

    def add(self, msg: str) -> None:
        self.issues.append(msg)

    @property
    def is_clean(self) -> bool:
        return not self.issues

    def __repr__(self) -> str:
        if self.is_clean:
            return "LeakageReport(clean)"
        return "LeakageReport(\n  " + "\n  ".join(self.issues) + "\n)"


def leakage_check(
    data: pd.DataFrame,
    date_col: str = "date",
    code_col: str = "code",
    feature_cols: Optional[Sequence[str]] = None,
) -> LeakageReport:
    """Run a battery of simple point-in-time checks.

    Checks:
    - duplicate (code, date) rows
    - future-dated rows (date > today)
    - features that are entirely constant over time
    - features that are perfectly correlated with the forward return
    """
    report = LeakageReport()
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if df.duplicated(subset=[code_col, date_col]).any():
        report.add("Duplicate (code, date) rows present")

    max_date = df[date_col].max()
    if max_date > pd.Timestamp.today().normalize():
        report.add(f"Future-dated rows present (max date {max_date.date()})")

    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c not in {date_col, code_col, "industry"}]

    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=True) <= 1:
            report.add(f"Feature {col!r} is constant; could mask data issues")

    # forward-return leakage check: if a 'ret_fwd_1' (or similar) column is
    # *identical* to any feature, that's a strong leakage signal.
    for fwd_col in ("ret_forward_1d", "ret_forward_5d", "label", "y"):
        if fwd_col in df.columns:
            for col in feature_cols:
                if col == fwd_col:
                    continue
                # per-code correlation to avoid cross-section confounders
                corrs = df.groupby(code_col)[[col, fwd_col]].corr().iloc[0::2, -1]
                if (corrs.abs() > 0.999).any():
                    report.add(
                        f"Feature {col!r} is perfectly correlated with {fwd_col!r} "
                        f"for at least one code (look-ahead leakage)."
                    )
    return report
