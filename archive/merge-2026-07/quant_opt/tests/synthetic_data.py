"""
测试工具: 生成合成 A 股数据与信号

用于在不依赖真实数据源的情况下验证优化代码的正确性与性能。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_synthetic_data(
    n_stocks: int = 50,
    n_days: int = 250,
    start_date: str = "2023-01-03",
    seed: int = 42,
    with_limits: bool = True,
    with_turnover: bool = True,
) -> pd.DataFrame:
    """
    生成合成 A 股日线数据。

    返回列: code, date, open, high, low, close, volume, amount,
            turnover_rate, is_limit_up, is_limit_down
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    rows = []
    for code in codes:
        # 几何布朗运动 + 均值回归
        rets = rng.normal(0.0005, 0.02, n_days)
        rets += -0.1 * (np.arange(n_days) % 20 == 0) * 0.05  # 偶尔大跌
        close = 10 * np.exp(np.cumsum(rets))
        open_ = close * (1 + rng.normal(0, 0.005, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, n_days)))
        volume = rng.lognormal(15, 0.5, n_days).astype(int)
        amount = close * volume
        turnover = rng.uniform(0.005, 0.05, n_days)

        # 涨跌停 (简化: 收益 > 9.5% 视为涨停)
        ret_pct = np.concatenate([[0], np.diff(close) / close[:-1]])
        is_limit_up = with_limits & (ret_pct > 0.095)
        is_limit_down = with_limits & (ret_pct < -0.095)

        for i in range(n_days):
            row = {
                "code": code,
                "date": dates[i],
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "volume": int(volume[i]),
                "amount": float(amount[i]),
            }
            if with_turnover:
                row["turnover_rate"] = float(turnover[i])
            if with_limits:
                row["is_limit_up"] = bool(is_limit_up[i])
                row["is_limit_down"] = bool(is_limit_down[i])
            rows.append(row)

    return pd.DataFrame(rows)


def make_signals(
    data: pd.DataFrame,
    strategy: str = "momentum",
    rebalance_freq: int = 5,
    top_pct: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成交易信号。

    strategy:
        - "momentum": 每 rebalance_freq 天，选过去 20 日涨幅最高的 top_pct 买入
        - "reversal": 反转
        - "random": 随机
    """
    rng = np.random.default_rng(seed)
    df = data.sort_values(["code", "date"]).copy()
    df["ret_20d"] = df.groupby("code")["close"].pct_change(20)

    dates = sorted(df["date"].unique())
    sig_rows = []
    for i, dt in enumerate(dates):
        if i % rebalance_freq != 0 or i < 20:
            continue
        cross = df[df["date"] == dt].dropna(subset=["ret_20d"])
        if cross.empty:
            continue
        if strategy == "momentum":
            score = cross["ret_20d"]
        elif strategy == "reversal":
            score = -cross["ret_20d"]
        else:
            score = pd.Series(rng.normal(0, 1, len(cross)), index=cross.index)
        threshold = score.quantile(1 - top_pct)
        buy_codes = cross.loc[score >= threshold, "code"].tolist()
        for code in buy_codes:
            sig_rows.append({"date": dt, "code": code, "signal": 1})
        # 之前持仓但不在新买入名单的，卖出
        prev_buy = df[df["date"] == dates[i - rebalance_freq]]["code"].tolist() if i >= rebalance_freq else []
        sell_codes = [c for c in prev_buy if c not in buy_codes]
        for code in sell_codes:
            sig_rows.append({"date": dt, "code": code, "signal": -1})

    return pd.DataFrame(sig_rows)
