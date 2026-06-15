"""
构造一份合成的 A 股日线 panel，用于离线验证
- 6 只股票
- 500 个交易日（约 2 年）
- 几何布朗运动 + 不同 drift，混入一些"动量/反转"结构便于 IC 分析
"""
import numpy as np
import pandas as pd


def make_synthetic_panel(
    n_stocks: int = 6,
    n_days: int = 500,
    start_date: str = "2022-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """生成可控的 A 股合成数据，每只股票带特定动量/反转倾向"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    rows = []
    for i, code in enumerate(codes):
        # 不同股票有不同"真实 alpha"，便于后续 IC 评估
        true_alpha = rng.uniform(-0.001, 0.001)
        reversal_strength = rng.uniform(0.0, 0.3)  # 反转强度

        start_price = rng.uniform(10, 50)
        drift = true_alpha
        vol = rng.uniform(0.015, 0.025)

        rets = rng.normal(drift, vol, n_days)
        # 注入反转信号：5 日收益为正则下日负
        for t in range(5, n_days):
            past5 = rets[t - 5:t].sum()
            rets[t] += -reversal_strength * past5

        prices = start_price * np.exp(np.cumsum(rets))

        df = pd.DataFrame({
            "date": dates,
            "code": code,
            "open": prices * (1 + rng.normal(0, 0.003, n_days)),
            "close": prices,
            "high": prices * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "low": prices * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "volume": rng.integers(1_000_000, 10_000_000, n_days),
        })
        df["high"] = df[["high", "close", "open"]].max(axis=1)
        df["low"] = df[["low", "close", "open"]].min(axis=1)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
