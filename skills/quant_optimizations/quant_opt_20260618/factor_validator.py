"""
Factor validator
================

A small, framework-free validation utility for a single factor.  The
goal is to answer the question "is this factor worth plugging into a
backtest?" before spending minutes on a full backtest.  Inspired by:

- Jesse's *Rule Significance Testing* (bootstrap resampling of rule
  returns to distinguish luck from signal).
- 传统 IC/ICIR 指标.

Outputs three decisions:

- ``ACCEPT``    : ICIR > 0.5 and bootstrap p-value < 0.05
- ``REVIEW``    : ICIR > 0.3 or bootstrap p-value < 0.1
- ``REJECT``    : otherwise

The bootstrap is intentionally cheap (1,000 resamples of dates, not
stocks) and uses 5-day block bootstrap to account for serial
correlation.
"""
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .ic_vectorized import ic_summary, ic_series_pearson


@dataclass
class FactorVerdict:
    decision: str
    ic_mean: float
    ic_ir: float
    ic_positive_ratio: float
    bootstrap_p: float
    n_obs: int

    def to_dict(self) -> dict:
        return asdict(self)


def validate_factor(
    factor: pd.Series,
    forward_ret: pd.Series,
    dates: pd.Series,
    *,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> FactorVerdict:
    """Compute a final verdict for a factor expression.

    Parameters
    ----------
    factor, forward_ret : pd.Series
        Aligned long-format series.
    dates : pd.Series
        Cross-section date key.
    n_bootstrap : int
        Number of bootstrap resamples for the significance test.
    block_size : int
        Block size in *days* for the time-series block bootstrap.
    """
    ic = ic_series_pearson(factor, forward_ret, dates, min_obs=10)
    summary = ic_summary(ic)

    p = _bootstrap_pvalue(ic, n_bootstrap=n_bootstrap,
                          block_size=block_size, seed=seed)

    decision = _decide(summary["ic_ir"], p)

    return FactorVerdict(
        decision=decision,
        ic_mean=summary["ic_mean"],
        ic_ir=summary["ic_ir"],
        ic_positive_ratio=summary["ic_positive_ratio"],
        bootstrap_p=p,
        n_obs=int(len(ic)),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _decide(ic_ir: float, p: float) -> str:
    if p < 0.05 and abs(ic_ir) > 0.5:
        return "ACCEPT"
    if p < 0.10 or abs(ic_ir) > 0.3:
        return "REVIEW"
    return "REJECT"


def _bootstrap_pvalue(ic: pd.Series, n_bootstrap: int, block_size: int, seed: int) -> float:
    """Two-sided p-value for the daily-IC mean.

    Block bootstrap is used to preserve the *serial* structure of the
    IC series.  We center the data at zero (so the surrogate's mean is
    drawn from the null distribution) and then compare the observed
    t-statistic to the surrogate t-statistics.

    Returns the *two-sided* p-value.
    """
    ic = ic.dropna()
    n = len(ic)
    if n < 20:
        return 1.0

    values = ic.values
    obs_mean = values.mean()
    obs_std = values.std(ddof=0)
    if obs_std == 0:
        return 1.0
    t_obs = obs_mean / (obs_std / np.sqrt(n))

    # Center at zero so the null distribution has mean 0
    centered = values - obs_mean
    starts = n - block_size + 1

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_bootstrap):
        surrogate = _circular_block_sample(centered, block_size, n, rng, starts)
        s_std = surrogate.std(ddof=0)
        if s_std == 0:
            continue
        t = surrogate.mean() / (s_std / np.sqrt(n))
        if abs(t) >= abs(t_obs):
            count += 1

    return count / n_bootstrap


def _circular_block_sample(values: np.ndarray, block: int, n: int,
                           rng: np.random.Generator, starts: int) -> np.ndarray:
    """Sample ``n`` observations from ``values`` using circular block bootstrap."""
    out = np.empty(n, dtype=values.dtype)
    i = 0
    circular = np.concatenate([values, values[:block - 1]])
    while i < n:
        s = rng.integers(0, starts)
        chunk = circular[s:s + block]
        take = min(block, n - i)
        out[i:i + take] = chunk[:take]
        i += take
    return out


if __name__ == "__main__":  # manual smoke test
    rng = np.random.default_rng(0)
    n_dates = 250
    n_stocks = 50
    dates = pd.Series(np.tile(pd.date_range("2023-01-01", periods=n_dates), n_stocks))
    factor = pd.Series(rng.standard_normal(n_dates * n_stocks))
    forward_ret = pd.Series(rng.standard_normal(n_dates * n_stocks)) + factor * 0.3
    print(validate_factor(factor, forward_ret, dates))