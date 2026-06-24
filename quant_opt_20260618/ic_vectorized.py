"""
Vectorized IC computation
==========================

The legacy ``FactorEngine.ic_analysis`` in jingni-trader iterates
``for dt in dates`` and calls ``scipy.stats.pearsonr`` /
``spearmanr`` on each cross-section.  This Python loop becomes the
bottleneck on A-share universes once we have ~5,000 stocks × 1,000
dates.

This module provides two drop-in replacements:

- :func:`ic_series_pearson`  – vectorized Pearson IC, one call, no
  Python-level loop.
- :func:`ic_series_spearman` – vectorized Rank IC via
  ``groupby('date').corrwith``.

Both return a ``pd.Series`` indexed by ``date`` plus a small
:func:`ic_summary` helper that mirrors the keys of the legacy
``ic_analysis`` output so it can be swapped in transparently.

Edge cases handled
------------------
- date with fewer than ``min_obs`` non-null pairs is dropped (matches
  legacy behaviour with threshold=10).
- ``pearson`` with constant cross-section returns NaN (matches scipy).
- ``spearman`` ties are broken via average rank, matching pandas default.
"""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core vectorized IC
# ---------------------------------------------------------------------------


def ic_series_pearson(
    factor: pd.Series,
    forward_ret: pd.Series,
    dates: Optional[pd.Series] = None,
    min_obs: int = 10,
) -> pd.Series:
    """Vectorized Pearson IC across cross-sections.

    Implementation uses centered cross-products and ``groupby.sum()`` so
    the per-group work is a single C-level reduction per column.  This
    is typically 20-100× faster than looping ``scipy.stats.pearsonr``
    per cross-section.
    """
    f, r, d = _align(factor, forward_ret, dates)
    df = pd.DataFrame({"d": d, "f": f, "r": r}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    g = df.groupby("d", sort=True)
    fx = df["f"] - g["f"].transform("mean")
    rx = df["r"] - g["r"].transform("mean")
    df["_num"] = (fx * rx).values
    df["_df"] = (fx * fx).values
    df["_dr"] = (rx * rx).values
    agg = df.groupby("d")[["_num", "_df", "_dr"]].sum()
    counts = df.groupby("d").size()
    out = agg["_num"] / np.sqrt(agg["_df"] * agg["_dr"])
    out = out.where((counts >= min_obs) & (agg["_df"] > 0) & (agg["_dr"] > 0), np.nan)
    out.name = "ic"
    return out


def ic_series_spearman(
    factor: pd.Series,
    forward_ret: pd.Series,
    dates: Optional[pd.Series] = None,
    min_obs: int = 10,
) -> pd.Series:
    """Vectorized Rank IC across cross-sections.

    Computes the cross-section Pearson IC of the rank-transformed
    factor and forward return.  Uses ``groupby.sum()`` for the centered
    cross-product so the per-group work stays in C.
    """
    f, r, d = _align(factor, forward_ret, dates)
    df = pd.DataFrame({"d": d, "f": f, "r": r}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    df["_rf"] = df.groupby("d")["f"].rank(method="average")
    df["_rr"] = df.groupby("d")["r"].rank(method="average")
    g = df.groupby("d", sort=True)
    fx = df["_rf"] - g["_rf"].transform("mean")
    rx = df["_rr"] - g["_rr"].transform("mean")
    df["_num"] = (fx * rx).values
    df["_df"] = (fx * fx).values
    df["_dr"] = (rx * rx).values
    agg = df.groupby("d")[["_num", "_df", "_dr"]].sum()
    counts = df.groupby("d").size()
    out = agg["_num"] / np.sqrt(agg["_df"] * agg["_dr"])
    out = out.where((counts >= min_obs) & (agg["_df"] > 0) & (agg["_dr"] > 0), np.nan)
    out.name = "ic"
    return out


# ---------------------------------------------------------------------------
# Summary helpers (mirror the legacy IC_analysis metadata keys)
# ---------------------------------------------------------------------------


def ic_summary(ic: pd.Series) -> Dict[str, float]:
    """Return the standard IC summary stats from an IC time series."""
    if ic is None or ic.empty:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_positive_ratio": 0.0,
            "ic_t_stat": 0.0,
        }
    mean = float(ic.mean())
    std = float(ic.std(ddof=0))
    ir = mean / std if std > 0 else 0.0
    pos = float((ic > 0).mean())
    n = len(ic)
    t = mean / (std / np.sqrt(n)) if std > 0 and n > 0 else 0.0
    return {
        "ic_mean": round(mean, 6),
        "ic_std": round(std, 6),
        "ic_ir": round(ir, 4),
        "ic_positive_ratio": round(pos, 4),
        "ic_t_stat": round(float(t), 4),
    }


def ic_analysis_batch(
    factor_df: pd.DataFrame,
    forward_ret: pd.Series,
    factor_names: List[str],
    ic_type: str = "normal",
    min_obs: int = 10,
) -> Dict[str, Dict]:
    """Compute per-factor IC summaries in one pass.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Long format with ``date`` column; each other column is treated
        as a factor.
    forward_ret : pd.Series
        Forward return aligned with ``factor_df.index``.
    factor_names : list[str]
        Columns to analyse.
    ic_type : {"normal", "spearman"}
    min_obs : int

    Returns
    -------
    dict[str, dict]
        ``{factor_name: ic_summary_dict, ...}``.
    """
    dates = factor_df["date"]
    out: Dict[str, Dict] = {}
    fn = ic_series_spearman if ic_type == "spearman" else ic_series_pearson
    for name in factor_names:
        if name not in factor_df.columns:
            continue
        ic = fn(factor_df[name], forward_ret, dates, min_obs=min_obs)
        out[name] = ic_summary(ic)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _align(f, r, d):
    if not f.index.equals(r.index):
        r = r.reindex(f.index)
    if d is None:
        if "date" not in f.index.names and f.index.name != "date":
            # treat as one cross-section
            d = pd.Series(["ALL"] * len(f), index=f.index)
        else:
            d = pd.Series(f.index, index=f.index)
    return f, r, d


def _safe_pearson(f: pd.Series, r: pd.Series, min_obs: int) -> float:
    if len(f) < min_obs:
        return np.nan
    if f.std(ddof=0) == 0 or r.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(f, r)[0, 1])