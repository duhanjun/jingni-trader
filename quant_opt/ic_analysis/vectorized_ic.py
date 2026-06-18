"""
Vectorized IC (Information Coefficient) analysis.

Borrowed inspiration: Qlib's cross-sectional rank-IC computation, which is
much faster than the per-date Python loop used in jingni-trader's
`FactorEngine._calc_ic`.

The new implementation:
- Vectorizes the per-date Spearman / Pearson computation
  using `scipy.stats.rankdata` and `groupby`
- Returns the same information (mean, std, IR, positive ratio, t-stat) as
  the original loop-based version
- Supports both Panel (MultiIndex) and long (code, date columns) inputs
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _spearman_per_date(factor: pd.Series, target: pd.Series) -> pd.Series:
    """Vectorized per-date Spearman correlation using scipy.stats.rankdata.

    `factor` and `target` must share the same MultiIndex (code, date) and
    be aligned element-wise.
    """
    # rank per date
    f_rank = factor.groupby(level="date").rank(method="average")
    t_rank = target.groupby(level="date").rank(method="average")

    # pearson on ranks == spearman
    return _pearson_per_date(f_rank, t_rank)


def _pearson_per_date(factor: pd.Series, target: pd.Series) -> pd.Series:
    """Per-date Pearson correlation; vectorized using groupby + cov."""
    f_mean = factor.groupby(level="date").transform("mean")
    t_mean = target.groupby(level="date").transform("mean")
    f_dev = factor - f_mean
    t_dev = target - t_mean
    cov = (f_dev * t_dev).groupby(level="date").transform("mean")
    f_std = factor.groupby(level="date").transform("std").replace(0, np.nan)
    t_std = target.groupby(level="date").transform("std").replace(0, np.nan)
    return cov / (f_std * t_std)


def _series_per_date(ic_per_row: pd.Series) -> pd.Series:
    """Reduce per-row IC contributions to one value per date.

    The trick: for normalized (ranked) inputs, per-date correlation is a
    single scalar. We can compute it directly. Here we just take the mean
    per date after applying the per-date correlation formula in the
    helper functions above, which already returns one value per date
    via the `groupby(level="date").transform("mean")` reduction in the
    numerator; to collapse to one value per date we group again.
    """
    # Each per-date value appears `n_assets_that_date` times in ic_per_row.
    # A simple groupby.mean gives the same constant per date; we therefore
    # return a deduplicated series indexed by date.
    return ic_per_row.groupby(level="date").mean()


def compute_ic_series(
    factor: pd.Series,
    target: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """Return a per-date IC series.

    Parameters
    ----------
    factor, target : Series indexed by (code, date)
    method          : 'spearman' or 'pearson'
    """
    factor = factor.dropna()
    target = target.reindex(factor.index)
    # align
    common = factor.index.intersection(target.index)
    factor = factor.loc[common]
    target = target.loc[common]

    if method == "spearman":
        per_date = _series_per_date(_spearman_per_date(factor, target))
    elif method == "pearson":
        per_date = _series_per_date(_pearson_per_date(factor, target))
    else:
        raise ValueError(f"Unknown method: {method}")
    return per_date.dropna()


def summarize_ic(ic_series: pd.Series) -> Dict[str, float]:
    """Summarize an IC series into the standard set of statistics."""
    if ic_series.empty:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_positive_ratio": 0.0,
            "ic_t_stat": 0.0,
            "n_periods": 0,
        }
    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std(ddof=1))
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    pos_ratio = float((ic_series > 0).mean())
    n = len(ic_series)
    t_stat = float(ic_mean / (ic_std / np.sqrt(n))) if ic_std > 0 else 0.0
    return {
        "ic_mean": round(ic_mean, 6),
        "ic_std": round(ic_std, 6),
        "ic_ir": round(ic_ir, 4),
        "ic_positive_ratio": round(pos_ratio, 4),
        "ic_t_stat": round(t_stat, 4),
        "n_periods": n,
    }


def batch_ic(
    factor_df: pd.DataFrame,
    target: pd.Series,
    factor_cols: Optional[List[str]] = None,
    method: str = "spearman",
) -> Dict[str, Dict[str, float]]:
    """Compute IC summary for many factors at once."""
    if factor_cols is None:
        factor_cols = [c for c in factor_df.columns
                       if c not in {"code", "date", "industry"}]

    if not isinstance(factor_df.index, pd.MultiIndex):
        factor_df = factor_df.set_index(["code", "date"])

    results: Dict[str, Dict[str, float]] = {}
    for col in factor_cols:
        s = factor_df[col]
        ic_series = compute_ic_series(s, target, method=method)
        results[col] = summarize_ic(ic_series)
    return results


def rank_ic_decay(
    factor: pd.Series,
    target: pd.Series,
    max_lag: int = 20,
    method: str = "spearman",
) -> pd.Series:
    """Compute IC for various forward-return horizons to study factor decay.

    Returns a Series indexed by horizon (1..max_lag).
    """
    out = []
    for lag in range(1, max_lag + 1):
        shifted = target.groupby(level="code").shift(-lag)
        ic_series = compute_ic_series(factor, shifted, method=method)
        out.append(summarize_ic(ic_series)["ic_mean"])
    return pd.Series(out, index=range(1, max_lag + 1), name="rank_ic")
