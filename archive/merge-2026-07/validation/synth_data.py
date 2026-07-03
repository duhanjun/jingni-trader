"""
合成数据生成器 (供验证脚本和测试用例使用)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def make_synthetic_panel(
    n_stocks: int = 50,
    n_days: int = 504,
    start_date: str = "2020-01-01",
    seed: int = 42,
    include_industry: bool = True,
) -> pd.DataFrame:
    """
    生成多只股票的 OHLCV 面板数据 + 行业标签
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

    rows = []
    industries = ["Tech", "Finance", "Consumer", "Health", "Energy", "Industrial"]
    for code in codes:
        industry = industries[hash(code) % len(industries)] if include_industry else None
        mu = rng.normal(0.0005, 0.001)
        sigma = rng.uniform(0.01, 0.04)
        # GBM
        log_ret = rng.normal(mu, sigma, n_days)
        close = 10 * np.exp(np.cumsum(log_ret))
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        volume = rng.integers(1_000_000, 50_000_000, n_days)
        for i, d in enumerate(dates):
            row = {
                "date": d,
                "code": code,
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "volume": float(volume[i]),
            }
            if include_industry:
                row["industry"] = industry
            rows.append(row)
    return pd.DataFrame(rows)


def make_synthetic_returns(
    n_stocks: int = 50,
    n_days: int = 504,
    seed: int = 42,
    signal_strength: float = 0.02,
) -> pd.DataFrame:
    """
    生成带"信号-收益"关系的截面日数据, 用于因子 IC 验证
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    rows = []
    for code in codes:
        # 每只股票有稳定 latent factor
        latent = rng.normal(0, 1)
        for d in dates:
            noise = rng.normal(0, 1)
            factor = 0.7 * latent + 0.3 * noise
            # forward return 与 factor 线性相关
            forward_return = signal_strength * factor + rng.normal(0, 0.01)
            rows.append({
                "date": d,
                "code": code,
                "factor": float(factor),
                "forward_return": float(forward_return),
            })
    return pd.DataFrame(rows)


def make_synthetic_equity(
    n_days: int = 504,
    start_date: str = "2020-01-01",
    seed: int = 7,
    annual_return: float = 0.10,
    annual_vol: float = 0.18,
) -> pd.Series:
    """合成单条净值曲线"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    daily_mu = annual_return / TRADING_DAYS
    daily_sigma = annual_vol / np.sqrt(TRADING_DAYS)
    rets = rng.normal(daily_mu, daily_sigma, n_days)
    equity = 100 * np.cumprod(1 + rets)
    return pd.Series(equity, index=dates, name="equity")
