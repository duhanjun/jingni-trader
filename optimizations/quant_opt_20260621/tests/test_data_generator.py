"""
测试数据生成器

生成合成的 A 股行情数据, 用于验证三个优化模块的正确性与性能.
不依赖外部数据源, 保证测试可重复.
"""
import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_codes: int = 20,
    n_days: int = 250,
    start_date: str = "2024-01-01",
    seed: int = 42,
    include_limit_flags: bool = True,
) -> pd.DataFrame:
    """
    生成合成 A 股日线数据

    返回 DataFrame, 列: code, date, open, high, low, close, volume, amount,
                       turnover_rate, is_limit_up, is_limit_down
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    rows = []
    for code in codes:
        # 每只股票一个随机漂移与波动
        drift = rng.normal(0.0003, 0.0005)
        vol = rng.uniform(0.015, 0.03)
        # 几何布朗运动
        log_ret = rng.normal(drift, vol, size=n_days)
        # 偶尔有大涨大跌
        shock_idx = rng.choice(n_days, size=max(1, n_days // 30), replace=False)
        log_ret[shock_idx] += rng.normal(0, 0.05, size=len(shock_idx))

        close = 10.0 * np.exp(np.cumsum(log_ret))
        open_ = close * (1 + rng.normal(0, 0.005, size=n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, size=n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, size=n_days)))
        volume = rng.integers(1_000_000, 50_000_000, size=n_days).astype(float)
        amount = volume * close
        turnover_rate = rng.uniform(0.5, 5.0, size=n_days)

        # 涨跌停标记 (简化: 日收益 > 9.5% 视为涨停)
        daily_ret = np.concatenate([[0], close[1:] / close[:-1] - 1])
        is_limit_up = daily_ret > 0.095
        is_limit_down = daily_ret < -0.095

        for i in range(n_days):
            row = {
                "code": code,
                "date": dates[i],
                "open": round(float(open_[i]), 4),
                "high": round(float(high[i]), 4),
                "low": round(float(low[i]), 4),
                "close": round(float(close[i]), 4),
                "volume": float(volume[i]),
                "amount": float(amount[i]),
                "turnover_rate": float(turnover_rate[i]),
            }
            if include_limit_flags:
                row["is_limit_up"] = bool(is_limit_up[i])
                row["is_limit_down"] = bool(is_limit_down[i])
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    return df


def generate_signal_data(
    data: pd.DataFrame,
    strategy: str = "ma_cross",
    fast_window: int = 5,
    slow_window: int = 20,
) -> pd.DataFrame:
    """
    基于行情数据生成交易信号 (用于回测验证)

    返回 DataFrame, 列: code, date, signal (1=买, -1=卖, 0=持有)
    """
    df = data.sort_values(["code", "date"]).copy()
    df["ma_fast"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(fast_window, min_periods=1).mean()
    )
    df["ma_slow"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(slow_window, min_periods=1).mean()
    )

    df["above"] = df["ma_fast"] > df["ma_slow"]
    df["prev_above"] = df.groupby("code")["above"].shift(1).fillna(False)

    # 金叉买入, 死叉卖出
    df["signal"] = 0
    df.loc[df["above"] & ~df["prev_above"], "signal"] = 1
    df.loc[~df["above"] & df["prev_above"], "signal"] = -1

    return df[["code", "date", "signal"]].reset_index(drop=True)
