"""
测试数据生成器

生成合成的 A 股行情/因子/信号数据，用于验证测试
不依赖外部数据源，确保测试可独立运行
"""
import numpy as np
import pandas as pd
from typing import Tuple


def generate_market_data(
    n_stocks: int = 50,
    n_days: int = 250,
    start_date: str = "2023-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成合成 A 股日线数据

    返回含: code, date, open, high, low, close, volume, amount,
            turnover_rate, change_pct, is_limit_up, is_limit_down
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    records = []
    for code in codes:
        price = 10.0 + rng.normal(0, 5)
        base_vol = 1e6 + rng.normal(0, 2e5)
        for dt in dates:
            ret = rng.normal(0, 0.02)
            price = max(price * (1 + ret), 1.0)
            volume = max(base_vol * (1 + rng.normal(0, 0.3)), 1e4)
            amount = price * volume
            turnover = volume / (1e8 + rng.normal(0, 1e7))
            change_pct = ret * 100
            is_lu = 1 if ret > 0.095 else 0
            is_ld = 1 if ret < -0.095 else 0
            records.append({
                "code": code, "date": dt,
                "open": price * (1 + rng.normal(0, 0.005)),
                "high": price * (1 + abs(rng.normal(0, 0.01))),
                "low": price * (1 - abs(rng.normal(0, 0.01))),
                "close": price,
                "volume": volume, "amount": amount,
                "turnover_rate": max(turnover, 0.001),
                "change_pct": change_pct,
                "is_limit_up": is_lu, "is_limit_down": is_ld,
            })
    return pd.DataFrame(records)


def generate_factor_data(
    market_data: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """基于行情数据生成因子数据(模拟 factor-engine 输出)"""
    rng = np.random.default_rng(seed)
    df = market_data.sort_values(["code", "date"]).copy()

    result = df[["code", "date"]].copy()
    result["ret_5d"] = df.groupby("code")["close"].pct_change(5)
    result["ret_20d"] = df.groupby("code")["close"].pct_change(20)
    result["ret_60d"] = df.groupby("code")["close"].pct_change(60)
    result["reversal_5d"] = -result["ret_5d"]
    result["reversal_20d"] = -result["ret_20d"]
    result["lncap"] = np.log(df["amount"] / df["turnover_rate"].replace(0, np.nan) * 100)
    result["turnover_20d"] = df.groupby("code")["turnover_rate"].transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    result["volatility_20d"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std())
    result["volume_ratio"] = df["volume"] / df.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()).replace(0, np.nan)
    result["money_flow_20d"] = (df["change_pct"] * df["amount"]).groupby(df["code"]).transform(
        lambda x: x.rolling(20, min_periods=5).sum())
    # 注入一个真实有效因子(带预测能力)
    fwd = df.groupby("code")["close"].transform(lambda x: x.shift(-5) / x - 1)
    result["synthetic_alpha"] = fwd.shift(1) * 0.5 + rng.normal(0, 0.01, len(df))
    return result


def generate_forward_returns(market_data: pd.DataFrame) -> pd.DataFrame:
    """生成前瞻收益数据"""
    df = market_data.sort_values(["code", "date"]).copy()
    result = df[["code", "date"]].copy()
    for p in [1, 5, 10, 20, 60]:
        result[f"ret_forward_{p}d"] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-p) / x - 1)
    return result


def generate_signals(factor_data: pd.DataFrame, top_pct: float = 0.2) -> pd.DataFrame:
    """基于因子生成买卖信号(模拟 backtest-engine 信号生成)"""
    df = factor_data.copy()
    df["rank"] = df.groupby("date")["synthetic_alpha"].rank(pct=True)
    df["signal"] = 0
    df.loc[df["rank"] > (1 - top_pct), "signal"] = 1
    return df[["code", "date", "signal"]]
