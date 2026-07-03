"""
共享合成数据生成器

为各优化模块的验证测试提供可复现的合成行情数据与信号。
不依赖外部数据源，确保测试在任意环境下可独立运行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(
    n_codes: int = 5,
    n_days: int = 250,
    start_date: str = "2023-01-03",
    seed: int = 42,
    include_limit_flags: bool = True,
) -> pd.DataFrame:
    """
    生成多股票合成 OHLCV 日线数据。

    返回列: code, date, open, high, low, close, volume, amount,
            pre_close, change_pct, is_st, is_limit_up, is_limit_down, listed_days
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_codes + 1)]

    rows = []
    for code in codes:
        # 几何布朗运动 + 轻微漂移，模拟真实股价
        rets = rng.normal(loc=0.0004, scale=0.018, size=n_days)
        prices = 10.0 * np.exp(np.cumsum(rets))
        for i, dt in enumerate(dates):
            close = float(prices[i])
            open_ = close * (1 + rng.normal(0, 0.003))
            high = max(open_, close) * (1 + abs(rng.normal(0, 0.004)))
            low = min(open_, close) * (1 - abs(rng.normal(0, 0.004)))
            volume = float(rng.integers(1_000_000, 10_000_000))
            amount = volume * close
            pre_close = float(prices[i - 1]) if i > 0 else open_
            change_pct = (close / pre_close - 1) if pre_close > 0 else 0.0
            rows.append({
                "code": code,
                "date": dt,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
                "pre_close": pre_close,
                "change_pct": change_pct,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
                "listed_days": 1000 + i,
            })

    df = pd.DataFrame(rows)
    if include_limit_flags:
        # A股主板涨跌停 ±10%
        df["is_limit_up"] = df["change_pct"] >= 0.095
        df["is_limit_down"] = df["change_pct"] <= -0.095
    return df


def make_signal_from_factor(
    data: pd.DataFrame,
    factor_name: str = "momentum_20",
    top_n: int = 2,
    rebalance_freq: str = "M",
) -> pd.DataFrame:
    """
    基于简单动量因子生成 Top-N 等权调仓信号。

    返回列: code, date, signal (1=买入/持有, -1=卖出, 0=无操作)
    信号在每个调仓日生成，非调仓日 signal=0。
    """
    df = data.sort_values(["code", "date"]).copy()
    df["momentum_20"] = df.groupby("code")["close"].transform(
        lambda s: s.pct_change(20)
    )

    # 计算调仓日（每月/每周最后一个交易日）
    dates = pd.Series(sorted(data["date"].unique()))
    if rebalance_freq == "M":
        rebalance_dates = (
            pd.Series(dates.values, index=pd.DatetimeIndex(dates.values))
            .groupby(pd.Grouper(freq="ME")).last().tolist()
        )
    elif rebalance_freq == "W":
        rebalance_dates = (
            pd.Series(dates.values, index=pd.DatetimeIndex(dates.values))
            .groupby(pd.Grouper(freq="W-FRI")).last().tolist()
        )
    else:
        rebalance_dates = dates.tolist()
    rebalance_dates = [d for d in rebalance_dates if d in set(dates)]

    signal_rows = []
    for dt in rebalance_dates:
        day_df = df[df["date"] == dt].dropna(subset=["momentum_20"])
        if day_df.empty:
            continue
        top_codes = day_df.nlargest(top_n, "momentum_20")["code"].tolist()
        all_codes = day_df["code"].tolist()
        for code in all_codes:
            signal_rows.append({
                "date": dt,
                "code": code,
                "signal": 1 if code in top_codes else -1,
            })

    return pd.DataFrame(signal_rows)


def make_target_weight_signal(
    data: pd.DataFrame,
    factor_name: str = "momentum_20",
    top_n: int = 2,
    rebalance_freq: str = "M",
) -> pd.DataFrame:
    """
    生成目标权重信号（向量化回测专用）。

    返回列: code, date, target_weight
    调仓日 Top-N 等权 1/N，非 Top-N 为 0。
    """
    df = data.sort_values(["code", "date"]).copy()
    df["momentum_20"] = df.groupby("code")["close"].transform(
        lambda s: s.pct_change(20)
    )

    dates = pd.Series(sorted(data["date"].unique()))
    if rebalance_freq == "M":
        rebalance_dates = (
            pd.Series(dates.values, index=pd.DatetimeIndex(dates.values))
            .groupby(pd.Grouper(freq="ME")).last().tolist()
        )
    else:
        rebalance_dates = dates.tolist()
    rebalance_dates = [d for d in rebalance_dates if d in set(dates)]

    rows = []
    for dt in rebalance_dates:
        day_df = df[df["date"] == dt].dropna(subset=["momentum_20"])
        if day_df.empty:
            continue
        top_codes = day_df.nlargest(top_n, "momentum_20")["code"].tolist()
        weight = 1.0 / len(top_codes) if top_codes else 0.0
        for code in day_df["code"].unique():
            rows.append({
                "date": dt,
                "code": code,
                "target_weight": weight if code in top_codes else 0.0,
            })

    return pd.DataFrame(rows)
