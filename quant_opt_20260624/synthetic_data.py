"""
合成 A 股 OHLCV 数据生成器

目的：
    为优化验证提供可复现的测试数据，避免依赖 tushare token / 网络，
    保证正确性测试与性能测试在任何环境下都能稳定运行。

数据 schema 与 jingni-trader 的 NativeAdapter 保持一致：
    code, date, open, high, low, close, volume, amount, turnover_rate,
    change_pct, is_st, is_limit_up, is_limit_down

借鉴：Qlib 的 benchmarks 数据集思路（合成数据用于可复现基准测试）
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional


def generate_synthetic_ohlcv(
    n_codes: int = 50,
    n_days: int = 250,
    start_date: str = "2023-01-03",
    seed: int = 42,
    base_price: float = 20.0,
    include_turnover: bool = True,
) -> pd.DataFrame:
    """
    生成合成 A 股日线数据。

    参数:
        n_codes: 股票数量
        n_days: 交易日数量
        start_date: 起始日期
        seed: 随机种子（保证可复现）
        base_price: 初始价格基准
        include_turnover: 是否生成换手率字段

    返回:
        DataFrame，列同 NativeAdapter 期望的输入
    """
    rng = np.random.default_rng(seed)

    # 构造交易日（跳过周末）
    dates = pd.bdate_range(start=start_date, periods=n_days)
    date_str = dates.strftime("%Y-%m-%d").tolist()

    # 股票代码：000001.SZ ~ 000xxx.SZ
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    rows = []
    for code in codes:
        # 几何布朗运动模拟价格
        mu = rng.uniform(-0.0005, 0.0015)  # 日均漂移
        sigma = rng.uniform(0.015, 0.035)  # 日波动率
        rets = rng.normal(mu, sigma, size=n_days)
        # 偶尔插入跳空
        jumps = rng.random(n_days) < 0.02
        rets[jumps] += rng.normal(0, 0.05, size=jumps.sum())

        close = base_price * np.exp(np.cumsum(rets))
        # open 在前一日 close 附近
        open_ = np.empty(n_days)
        open_[0] = base_price
        open_[1:] = close[:-1] * (1 + rng.normal(0, 0.005, size=n_days - 1))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, size=n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, size=n_days)))

        volume = rng.integers(5_000_000, 50_000_000, size=n_days).astype(np.int64)
        amount = close * volume * rng.uniform(0.99, 1.01, size=n_days)
        turnover_rate = (volume / rng.uniform(2e8, 1e9)) * 100  # 百分比

        change_pct = np.concatenate([[0.0], np.diff(close) / close[:-1]])

        # 涨跌停标记（A 股 10% 限制）
        is_limit_up = change_pct >= 0.095
        is_limit_down = change_pct <= -0.095
        # ST 标记：随机 5% 的股票在某段时期为 ST
        is_st = np.zeros(n_days, dtype=bool)
        if rng.random() < 0.05:
            st_start = rng.integers(0, n_days // 2)
            is_st[st_start:] = True

        df = pd.DataFrame({
            "code": code,
            "date": date_str,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume,
            "amount": amount.round(2),
            "change_pct": change_pct,
            "is_st": is_st,
            "is_limit_up": is_limit_up,
            "is_limit_down": is_limit_down,
        })
        if include_turnover:
            df["turnover_rate"] = turnover_rate.round(4)
        rows.append(df)

    data = pd.concat(rows, ignore_index=True)
    # 保证排序稳定
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    return data


def generate_signals(
    data: pd.DataFrame,
    strategy: str = "ma_cross",
    fast: int = 5,
    slow: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    基于行情数据生成交易信号，schema 与 NativeAdapter 一致：
        code, date, signal  (1 买入, -1 卖出, 0 持仓)

    参数:
        data: generate_synthetic_ohlcv 的输出
        strategy: 信号策略名（ma_cross / reversal）
    """
    rng = np.random.default_rng(seed)
    df = data.sort_values(["code", "date"]).copy()

    if strategy == "ma_cross":
        df["ma_fast"] = df.groupby("code")["close"].transform(
            lambda x: x.rolling(fast, min_periods=1).mean()
        )
        df["ma_slow"] = df.groupby("code")["close"].transform(
            lambda x: x.rolling(slow, min_periods=1).mean()
        )
        df["prev_fast"] = df.groupby("code")["ma_fast"].shift(1)
        df["prev_slow"] = df.groupby("code")["ma_slow"].shift(1)
        # 金叉买入、死叉卖出
        buy = (df["ma_fast"] > df["ma_slow"]) & (df["prev_fast"] <= df["prev_slow"])
        sell = (df["ma_fast"] < df["ma_slow"]) & (df["prev_fast"] >= df["prev_slow"])
        df["signal"] = 0
        df.loc[buy.fillna(False), "signal"] = 1
        df.loc[sell.fillna(False), "signal"] = -1
    elif strategy == "reversal":
        # 20 日反转：过去 20 日跌幅大的买入
        df["ret_20d"] = df.groupby("code")["close"].pct_change(slow)
        df["signal"] = 0
        # 每日选跌幅前 20% 买入，涨幅前 20% 卖出
        df["rank"] = df.groupby("date")["ret_20d"].rank(pct=True)
        df.loc[df["rank"] < 0.2, "signal"] = 1
        df.loc[df["rank"] > 0.8, "signal"] = -1
    else:
        raise ValueError(f"未知策略: {strategy}")

    return df[["code", "date", "signal"]].copy()


if __name__ == "__main__":
    # 自检：生成小样本并打印摘要
    d = generate_synthetic_ohlcv(n_codes=10, n_days=60)
    print("数据形状:", d.shape)
    print("列:", list(d.columns))
    print(d.head())
    sig = generate_signals(d)
    print("信号形状:", sig.shape, "买入:", (sig["signal"] == 1).sum(), "卖出:", (sig["signal"] == -1).sum())
