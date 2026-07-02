"""
测试工具：合成 A 股数据生成器

生成符合 jingni-trader 数据格式的合成数据，用于验证优化代码的正确性与性能。
数据格式与 native_adapter 期望一致：code, date, open, high, low, close, volume,
amount, turnover_rate, is_st, is_limit_up, is_limit_down
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_codes: int = 50,
    n_days: int = 250,
    start_date: str = "2023-01-03",
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成合成 A 股日线数据

    参数:
        n_codes: 股票数量
        n_days: 交易日数
        start_date: 起始日期
        seed: 随机种子

    返回:
        DataFrame，含 code, date, open, high, low, close, volume, amount,
        turnover_rate, is_st, is_limit_up, is_limit_down
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    records = []
    for code in codes:
        # 每只股票独立的价格路径（几何布朗运动）
        ret = rng.normal(0.0005, 0.02, n_days)
        price = 10.0 * np.exp(np.cumsum(ret))
        # 加少量趋势与波动
        open_ = price * (1 + rng.normal(0, 0.005, n_days))
        high = np.maximum(open_, price) * (1 + np.abs(rng.normal(0, 0.008, n_days)))
        low = np.minimum(open_, price) * (1 - np.abs(rng.normal(0, 0.008, n_days)))
        volume = rng.lognormal(15, 1, n_days).astype(float)
        amount = volume * price
        turnover_rate = rng.uniform(0.005, 0.05, n_days)

        # 涨跌停（约 1% 概率）
        is_limit_up = (rng.random(n_days) < 0.01)
        is_limit_down = (rng.random(n_days) < 0.005)
        is_st = np.zeros(n_days, dtype=bool)

        for i in range(n_days):
            records.append({
                "code": code,
                "date": dates[i],
                "open": round(float(open_[i]), 2),
                "high": round(float(high[i]), 2),
                "low": round(float(low[i]), 2),
                "close": round(float(price[i]), 2),
                "volume": float(volume[i]),
                "amount": float(amount[i]),
                "turnover_rate": float(turnover_rate[i]),
                "is_st": bool(is_st[i]),
                "is_limit_up": bool(is_limit_up[i]),
                "is_limit_down": bool(is_limit_down[i]),
            })

    df = pd.DataFrame(records)
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def generate_signals(
    data: pd.DataFrame,
    strategy: str = "momentum",
    top_pct: float = 0.2,
    rebalance_days: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    基于行情数据生成交易信号

    参数:
        data: 行情数据
        strategy: "momentum"（动量）或 "reversal"（反转）
        top_pct: 选股比例
        rebalance_days: 调仓间隔（日）
        seed: 随机种子

    返回:
        DataFrame，含 code, date, signal (1/-1/0)
    """
    rng = np.random.default_rng(seed)
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 计算 20 日动量
    df = df.sort_values(["code", "date"])
    df["mom20"] = df.groupby("code")["close"].pct_change(20)

    # 每隔 rebalance_days 日调仓
    all_dates = sorted(df["date"].unique())
    rebalance_dates = all_dates[::rebalance_days]

    signals = []
    for dt in rebalance_dates:
        cross = df[df["date"] == dt].dropna(subset=["mom20"])
        if len(cross) < 10:
            continue
        if strategy == "momentum":
            threshold = cross["mom20"].quantile(1 - top_pct)
            buy_codes = cross[cross["mom20"] >= threshold]["code"].tolist()
        else:
            threshold = cross["mom20"].quantile(top_pct)
            buy_codes = cross[cross["mom20"] <= threshold]["code"].tolist()

        # 买入信号
        for code in buy_codes:
            signals.append({"date": dt, "code": code, "signal": 1})
        # 卖出信号：不在买入名单且之前可能持仓的
        sell_candidates = cross[~cross["code"].isin(buy_codes)]["code"].tolist()
        # 随机选一部分卖出（模拟换仓）
        n_sell = min(len(sell_candidates), len(buy_codes))
        if n_sell > 0:
            sell_codes = rng.choice(sell_candidates, size=n_sell, replace=False)
            for code in sell_codes:
                signals.append({"date": dt, "code": code, "signal": -1})

    sig_df = pd.DataFrame(signals)
    if sig_df.empty:
        return sig_df
    return sig_df.sort_values(["date", "code"]).reset_index(drop=True)


def generate_factor_data(
    data: pd.DataFrame,
    n_factors: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成合成因子数据与前瞻收益，用于 IC 分析测试

    参数:
        data: 行情数据
        n_factors: 因子数量
        seed: 随机种子

    返回:
        (factor_df, forward_returns) 元组
    """
    rng = np.random.default_rng(seed)
    df = data.copy().sort_values(["code", "date"])

    factor_df = df[["code", "date"]].copy()
    # 生成 n_factors 个因子，其中第 0 个与未来收益有真实相关性
    for i in range(n_factors):
        if i == 0:
            # 有效因子：与 5 日前瞻收益相关
            fwd5 = df.groupby("code")["close"].transform(
                lambda x: x.shift(-5) / x - 1
            )
            factor_df[f"factor_{i}"] = fwd5.shift(1) * 0.5 + rng.normal(0, 0.01, len(df))
        else:
            factor_df[f"factor_{i}"] = rng.normal(0, 1, len(df))

    # 前瞻收益
    fwd = df[["code", "date"]].copy()
    for period in [1, 5, 20]:
        fwd[f"ret_forward_{period}d"] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-period) / x - 1
        )

    return factor_df, fwd
