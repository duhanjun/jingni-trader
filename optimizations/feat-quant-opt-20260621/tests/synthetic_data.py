"""
测试数据生成器 - 模拟 A 股日线数据
用于在无网络/无数据源环境下验证优化代码的正确性与性能
"""
import numpy as np
import pandas as pd
from typing import List, Optional


def generate_synthetic_a_share_data(
    n_stocks: int = 50,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    include_benchmark: bool = True,
) -> pd.DataFrame:
    """
    生成模拟 A 股日线数据

    返回:
        DataFrame, 列: date, code, open, high, low, close, volume,
                      amount, turnover_rate, pre_close, change_pct,
                      is_st, is_limit_up, is_limit_down
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

    # 生成股票代码 (沪深主板 + 创业板)
    codes = []
    for i in range(n_stocks):
        if i < n_stocks // 2:
            code = f"{600000 + i:06d}.SH"
        else:
            code = f"{300000 + (i - n_stocks // 2):06d}.SZ"
        codes.append(code)

    rows = []
    for code in codes:
        # 起始价 5~50 元
        start_price = rng.uniform(5, 50)
        # 日漂移与波动
        drift = rng.uniform(-0.0003, 0.0008)
        vol = rng.uniform(0.012, 0.025)

        # 几何布朗运动 + 微弱动量
        rets = rng.normal(drift, vol, n_days)
        for i in range(1, n_days):
            rets[i] += 0.1 * rets[i - 1]

        prices = np.cumprod(1 + rets) * start_price
        prices = np.maximum(prices, 1.0)  # 不低于1元

        # OHLC
        open_prices = prices * (1 + rng.normal(0, 0.005, n_days))
        high = np.maximum(prices, open_prices) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(prices, open_prices) * (1 - np.abs(rng.normal(0, 0.005, n_days)))

        # 成交量 (对数正态)
        base_vol = rng.uniform(5e5, 5e6)
        volume = (base_vol * (1 + rng.normal(0, 0.3, n_days))).astype(int)
        volume = np.maximum(volume, 10000)

        # 成交额与换手率
        amount = prices * volume
        turnover_rate = volume / rng.uniform(5e8, 2e9)

        # 涨跌幅
        pre_close = np.concatenate([[start_price], prices[:-1]])
        change_pct = (prices - pre_close) / pre_close * 100

        df_one = pd.DataFrame({
            "date": dates,
            "code": code,
            "open": open_prices.round(4),
            "high": high.round(4),
            "low": low.round(4),
            "close": prices.round(4),
            "volume": volume,
            "amount": amount.round(2),
            "turnover_rate": turnover_rate.round(6),
            "pre_close": pre_close.round(4),
            "change_pct": change_pct.round(4),
            "is_st": False,
            "is_limit_up": change_pct >= 9.9,
            "is_limit_down": change_pct <= -9.9,
        })
        rows.append(df_one)

    df = pd.concat(rows, ignore_index=True)

    if include_benchmark:
        # 生成基准 (沪深300模拟)
        bench_ret = rng.normal(0.0002, 0.011, n_days)
        bench_prices = np.cumprod(1 + bench_ret) * 4000
        bench_df = pd.DataFrame({
            "date": dates,
            "code": "000300.SH",
            "open": bench_prices,
            "high": bench_prices * 1.005,
            "low": bench_prices * 0.995,
            "close": bench_prices,
            "volume": 0,
            "amount": 0.0,
            "turnover_rate": 0.0,
            "pre_close": np.concatenate([[4000], bench_prices[:-1]]),
            "change_pct": np.concatenate([[0], bench_ret[1:] * 100]),
            "is_st": False,
            "is_limit_up": False,
            "is_limit_down": False,
        })
        df = pd.concat([df, bench_df], ignore_index=True)

    return df.sort_values(["date", "code"]).reset_index(drop=True)


def generate_random_signals(
    data: pd.DataFrame,
    n_signals_per_day: int = 5,
    seed: int = 100,
    signal_mode: str = "signal",
) -> pd.DataFrame:
    """
    生成随机交易信号 (用于回测对比)

    参数:
        data: 行情数据
        n_signals_per_day: 每日信号数
        signal_mode: "signal" 或 "target_weight"
    """
    rng = np.random.default_rng(seed)
    # 排除基准
    codes = [c for c in data["code"].unique() if c != "000300.SH"]
    dates = sorted(data["date"].unique())

    records = []
    for dt in dates:
        # 随机选 n 只股票
        n = min(n_signals_per_day, len(codes))
        chosen = rng.choice(codes, size=n, replace=False)
        for code in chosen:
            if signal_mode == "signal":
                sig = rng.choice([1, -1])
                records.append({"date": dt, "code": code, "signal": sig})
            else:
                w = rng.uniform(0.05, 0.15)
                records.append({"date": dt, "code": code, "target_weight": w})

    return pd.DataFrame(records).sort_values(["date", "code"]).reset_index(drop=True)


def generate_momentum_signals(
    data: pd.DataFrame,
    lookback: int = 20,
    hold_days: int = 5,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    生成动量策略信号 (用于回测对比，更接近真实策略)

    每 hold_days 天调仓一次，选过去 lookback 天涨幅最高的 top_n 只股票
    """
    df = data[data["code"] != "000300.SH"].copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # 计算过去 lookback 天涨幅
    df["ret_lookback"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change(lookback)
    )

    dates = sorted(df["date"].unique())
    records = []
    last_rebalance = None

    for dt in dates:
        if last_rebalance is not None:
            days_since = (dt - last_rebalance).days
            if days_since < hold_days:
                continue
        last_rebalance = dt

        day_data = df[df["date"] == dt].dropna(subset=["ret_lookback"])
        if len(day_data) < top_n:
            continue

        # 选 top_n
        top = day_data.nlargest(top_n, "ret_lookback")
        for _, row in top.iterrows():
            records.append({"date": dt, "code": row["code"], "signal": 1})

    return pd.DataFrame(records).sort_values(["date", "code"]).reset_index(drop=True)
