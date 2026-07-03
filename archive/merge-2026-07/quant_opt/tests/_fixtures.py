"""
测试用工具：构造 A 股风格的模拟数据
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def make_synthetic_a_share_data(
    n_stocks: int = 50,
    n_days: int = 252,
    start_date: str = "2023-01-01",
    seed: int = 20240101,
) -> pd.DataFrame:
    """生成与 A 股日线统计特征近似的模拟数据。

    返回 long-form DataFrame，包含 columns:
        date, code, open, high, low, close, volume, pre_close,
        change_pct, is_st, is_limit_up, is_limit_down
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    rows = []
    for code in codes:
        start_price = rng.uniform(8.0, 50.0)
        drift = rng.uniform(-0.0003, 0.001)
        vol = rng.uniform(0.012, 0.025)
        # AR(1) 收益
        rets = np.zeros(n_days)
        rets[0] = rng.normal(0, vol)
        for i in range(1, n_days):
            rets[i] = drift + 0.1 * rets[i - 1] + rng.normal(0, vol)
        prices = start_price * np.exp(np.cumsum(rets))
        opens = prices * (1 + rng.normal(0, 0.003, n_days))
        highs = np.maximum(opens, prices) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        lows = np.minimum(opens, prices) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        vols = rng.lognormal(10, 0.5, n_days).astype(int)
        pre_close = np.concatenate([[prices[0]], prices[:-1]])
        change_pct = (prices - pre_close) / pre_close * 100

        rows.append(pd.DataFrame({
            "date": dates,
            "code": code,
            "open": opens.round(4),
            "high": highs.round(4),
            "low": lows.round(4),
            "close": prices.round(4),
            "volume": vols,
            "pre_close": pre_close.round(4),
            "change_pct": change_pct.round(4),
            "is_st": False,
            "is_limit_up": change_pct >= 9.9,
            "is_limit_down": change_pct <= -9.9,
        }))

    df = pd.concat(rows, ignore_index=True)
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def make_signals(
    data: pd.DataFrame,
    n_dates: int = 252,
    top_quantile: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """生成随机买卖信号用于回测（top quantile = 1，其余 = 0）"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=data["date"].min(), periods=n_dates)
    codes = data["code"].unique()
    rows = []
    for d in dates:
        # 当日选 top 20% 作为买入
        selected = rng.choice(codes, size=int(len(codes) * top_quantile), replace=False)
        for c in codes:
            rows.append({"date": d, "code": c, "signal": 1 if c in selected else 0})
    return pd.DataFrame(rows)


__all__ = ["make_synthetic_a_share_data", "make_signals"]
