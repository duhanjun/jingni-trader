"""
测试数据生成工具

为验证模块提供可复现的、模拟 A 股多只股票多日行情的合成数据，
便于：
  1. 因子计算正确性测试
  2. PIT 合并边界测试
  3. Walk-Forward 框架 smoke test
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ashare_data(
    n_stocks: int = 30,
    n_days: int = 500,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成模拟 A 股日线数据。

    Returns
    -------
    pd.DataFrame
        包含 code, date, open, high, low, close, volume, amount, turnover_rate,
        change_pct, is_limit_up, is_limit_down, industry 等字段
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    rows = []
    industries = ["科技", "医药", "消费", "金融", "制造", "能源"]
    for code in codes:
        industry = industries[hash(code) % len(industries)]
        # 几何布朗运动 + 漂移
        log_ret = rng.normal(loc=0.0005, scale=0.02, size=n_days)
        close = 10.0 * np.exp(np.cumsum(log_ret))
        # 加入一些规律性，便于检验因子
        momentum = np.sin(np.arange(n_days) / 20.0) * 0.01
        close += momentum * close
        open_ = close * (1 + rng.normal(0, 0.005, size=n_days))
        high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.008, size=n_days)))
        low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.008, size=n_days)))
        volume = rng.integers(1_000_000, 50_000_000, size=n_days)
        amount = volume * close
        change_pct = np.concatenate([[0.0], np.diff(close) / close[:-1]])
        turnover_rate = rng.uniform(0.001, 0.05, size=n_days)
        # 涨跌停标记
        is_limit_up = change_pct >= 0.095
        is_limit_down = change_pct <= -0.095
        for i, dt in enumerate(dates):
            rows.append({
                "code": code,
                "date": dt,
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "volume": int(volume[i]),
                "amount": float(amount[i]),
                "turnover_rate": float(turnover_rate[i]),
                "change_pct": float(change_pct[i]),
                "is_limit_up": bool(is_limit_up[i]),
                "is_limit_down": bool(is_limit_down[i]),
                "industry": industry,
            })
    return pd.DataFrame(rows)


def make_financial_data(n_stocks: int = 30, n_periods: int = 8, seed: int = 7) -> pd.DataFrame:
    """
    模拟季报财务数据，并故意制造"announce_date > period_end"的发布延迟，
    用于 PIT 测试。
    """
    rng = np.random.default_rng(seed)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    period_ends = pd.date_range("2022-12-31", periods=n_periods, freq="QE")
    rows = []
    for code in codes:
        for pe in period_ends:
            # 公告日：报告期 + 30~90 天随机
            announce = pe + pd.Timedelta(days=int(rng.integers(30, 90)))
            rows.append({
                "code": code,
                "period_end": pe,
                "announce_date": announce,
                "pe_ttm": float(rng.normal(20, 8)),
                "roe": float(rng.normal(0.1, 0.05)),
                "revenue_growth": float(rng.normal(0.15, 0.1)),
            })
    return pd.DataFrame(rows)


__all__ = ["make_synthetic_ashare_data", "make_financial_data"]
