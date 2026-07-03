"""
测试数据生成器：构造与 jingni-trader 数据契约一致的合成数据

用于在无真实数据源的情况下验证新模块的正确性与性能。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_data(
    n_codes: int = 100,
    n_days: int = 500,
    start_date: str = "2022-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成合成 A 股日线数据，列与 jingni-trader 数据契约一致：
    code, date, open, high, low, close, volume, amount, turnover_rate,
    is_st, is_limit_up, is_limit_down
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_codes + 1)]

    rows = []
    for code in codes:
        # 每只股票一个随机漂移的随机游走
        rets = rng.normal(0.0005, 0.02, n_days)
        price = 10.0 * np.exp(np.cumsum(rets))
        for i, dt in enumerate(dates):
            close = float(price[i])
            open_ = close * (1 + rng.normal(0, 0.005))
            high = max(close, open_) * (1 + abs(rng.normal(0, 0.005)))
            low = min(close, open_) * (1 - abs(rng.normal(0, 0.005)))
            volume = float(rng.integers(1_000_000, 10_000_000))
            amount = volume * close
            turnover = float(rng.uniform(0.5, 5.0))
            rows.append({
                "code": code,
                "date": dt,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
                "turnover_rate": turnover,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            })
    return pd.DataFrame(rows)


def make_signals(data: pd.DataFrame, strategy: str = "reversal_5d") -> pd.DataFrame:
    """
    基于数据生成简单信号：5日反转
    过去5日跌幅最大的 -> 买入信号
    """
    df = data.sort_values(["code", "date"]).copy()
    df["ret_5d"] = df.groupby("code")["close"].pct_change(5)
    # 每日选 ret_5d 最低的 10% 作为买入，最高的 10% 作为卖出
    df["pct_rank"] = df.groupby("date")["ret_5d"].rank(pct=True)
    df["signal"] = 0
    df.loc[df["pct_rank"] < 0.1, "signal"] = 1
    df.loc[df["pct_rank"] > 0.9, "signal"] = -1
    return df[["code", "date", "signal"]].reset_index(drop=True)
