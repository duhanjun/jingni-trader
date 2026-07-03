"""Shared synthetic-data fixtures for the optimization tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_OPT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_OPT))


def make_synth_panel(
    n_codes: int = 20,
    n_days: int = 250,
    start_date: str = "2023-01-01",
    seed: int = 42,
    drift: float = 0.0003,
    vol: float = 0.02,
) -> pd.DataFrame:
    """
    生成合规的 MultiIndex(code, date) 行情面板.
    """
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SH" for i in range(1, n_codes + 1)]
    dates = pd.bdate_range(start=start_date, periods=n_days)
    panel = []
    for c in codes:
        # 引入异质性: 不同股票有不同 alpha
        a = rng.normal(0, 0.0001)
        rets = rng.normal(loc=drift + a, scale=vol, size=n_days)
        price = 100 * (1 + pd.Series(rets)).cumprod()
        df = pd.DataFrame({
            "code": c,
            "date": dates,
            "open": price * (1 + rng.normal(0, 0.001, n_days)),
            "high": price * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "low":  price * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "close": price,
            "volume": rng.integers(1_000_000, 10_000_000, n_days).astype(float),
            "amount": price * rng.integers(1_000_000, 10_000_000, n_days).astype(float),
        })
        panel.append(df)
    out = pd.concat(panel, ignore_index=True)
    return out


def make_synth_factor(
    panel: pd.DataFrame,
    signal_strength: float = 0.4,
    seed: int = 7,
) -> pd.DataFrame:
    """
    生成与 close 真实未来收益相关的因子值, 用于 tearsheet/walk-forward 测试.

    真实信号 = (next day return) * signal_strength + noise
    """
    rng = np.random.default_rng(seed)
    df = panel.sort_values(["code", "date"]).copy()
    df["next_ret"] = df.groupby("code")["close"].pct_change().shift(-1)
    # 真实信号与未来收益有相关性
    df["alpha_factor"] = (
        signal_strength * df["next_ret"].fillna(0)
        + (1 - signal_strength) * rng.normal(0, 0.02, len(df))
    )
    return df[["code", "date", "alpha_factor"]]


def make_synth_equity(
    n_days: int = 252,
    start_date: str = "2023-01-01",
    seed: int = 123,
    alpha: float = 0.0002,
    beta: float = 0.9,
    bench_vol: float = 0.012,
    strat_idio_vol: float = 0.005,
) -> pd.DataFrame:
    """
    生成策略 vs 基准的净值序列, 用于 benchmark comparison 测试.

    strategy_return_t = alpha + beta * bench_return_t + idio_noise
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    bench_ret = rng.normal(0.0003, bench_vol, n_days)
    idio = rng.normal(0, strat_idio_vol, n_days)
    strat_ret = alpha + beta * bench_ret + idio

    bench_eq = 1_000_000 * (1 + pd.Series(bench_ret)).cumprod()
    strat_eq = 1_000_000 * (1 + pd.Series(strat_ret)).cumprod()
    return pd.DataFrame({
        "date": dates,
        "strategy_equity": strat_eq.values,
        "benchmark_equity": bench_eq.values,
    })
