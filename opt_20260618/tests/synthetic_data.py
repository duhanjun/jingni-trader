"""
合成测试数据生成器

为本次验证准备一个可重复使用的合成数据集：
- 500 只股票，1 年（约 250 个交易日）的日线数据
- 模拟 A 股的统计特征（涨跌停、换手率、波动率）
- 附带 alpha_score 因子

借鉴 data-engine._generate_synthetic_data 但加入更多特征以便回测验证。
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_stocks: int = 100,
    n_days: int = 250,
    start_date: str = "2024-01-01",
    seed: int = 42,
    with_factor: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成合成 A 股数据

    参数:
        n_stocks: 股票数量
        n_days: 交易日数量
        start_date: 起始日期
        seed: 随机种子
        with_factor: 是否生成对应的 alpha_score 因子

    返回:
        (data, factors)
        - data: 标准行情 (date, code, open, high, low, close, volume, ...)
        - factors: 因子数据 (date, code, alpha_score, ret_5d, ret_20d, ...)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    n = len(dates)

    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks // 2)] + \
            [f"{1 + i:06d}.SZ" for i in range(n_stocks // 2)]
    codes = codes[:n_stocks]

    rows = []
    factor_rows = []
    for code in codes:
        # 每只股票独立的随机过程
        drift = rng.uniform(-0.0003, 0.0008)
        vol = rng.uniform(0.012, 0.025)
        ret = rng.normal(drift, vol, n)
        for i in range(1, n):
            ret[i] += 0.10 * ret[i - 1]  # 弱自相关

        # 价格序列
        start_price = rng.uniform(8, 50)
        close = start_price * np.exp(np.cumsum(ret))
        # OHLC
        open_ = close * (1 + rng.normal(0, 0.003, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
        # 成交量
        volume = rng.integers(1_000_000, 50_000_000, n)
        amount = close * volume

        pre_close = np.concatenate([[close[0]], close[:-1]])
        change_pct = (close - pre_close) / pre_close * 100
        is_limit_up = change_pct >= 9.9
        is_limit_down = change_pct <= -9.9
        turnover_rate = rng.uniform(0.005, 0.05, n)

        df = pd.DataFrame({
            "date": dates,
            "code": code,
            "open": open_.round(4),
            "high": high.round(4),
            "low": low.round(4),
            "close": close.round(4),
            "volume": volume,
            "amount": amount.round(2),
            "pre_close": pre_close.round(4),
            "change_pct": change_pct.round(4),
            "is_st": False,
            "is_limit_up": is_limit_up,
            "is_limit_down": is_limit_down,
            "turnover_rate": turnover_rate.round(4),
        })
        rows.append(df)

        if with_factor:
            # 生成 alpha_score：与未来 5 日收益有部分正相关
            future_ret_5d = pd.Series(close).pct_change(5).shift(-5)
            noise = rng.normal(0, 0.02, n)
            # 真实 alpha：未来 5 日收益 + 噪声（强度约 0.3）
            alpha = 0.3 * future_ret_5d.fillna(0) + noise
            # 滞后 1 日对齐
            alpha_lagged = pd.Series(alpha).shift(1).fillna(0)

            ret_5d = pd.Series(close).pct_change(5).fillna(0)
            ret_20d = pd.Series(close).pct_change(20).fillna(0)
            ret_60d = pd.Series(close).pct_change(60).fillna(0)

            fac = pd.DataFrame({
                "date": dates,
                "code": code,
                "alpha_score": alpha_lagged.round(6),
                "ret_5d": ret_5d.round(6),
                "ret_20d": ret_20d.round(6),
                "ret_60d": ret_60d.round(6),
                "volatility_20d": pd.Series(ret).rolling(20, min_periods=5).std().fillna(0).round(6),
            })
            factor_rows.append(fac)

    data = pd.concat(rows, ignore_index=True)
    factors = pd.concat(factor_rows, ignore_index=True) if with_factor else pd.DataFrame()
    return data, factors


def generate_signals(
    factors: pd.DataFrame,
    top_quantile: float = 0.8,
    bottom_quantile: float = 0.2,
) -> pd.DataFrame:
    """
    从 alpha_score 生成 topk 离散信号
    """
    if factors.empty or "alpha_score" not in factors.columns:
        return pd.DataFrame(columns=["date", "code", "signal"])
    df = factors.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "code"])
    df["rank"] = df.groupby("date")["alpha_score"].rank(pct=True)
    df["signal"] = 0
    df.loc[df["rank"] > top_quantile, "signal"] = 1
    df.loc[df["rank"] < bottom_quantile, "signal"] = -1
    return df[["date", "code", "signal"]]
