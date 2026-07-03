"""
共享测试夹具：合成数据生成器

为避免依赖外部 tushare API，所有测试使用合成的 OHLCV 数据。
数据特征模拟 A 股：100 股整数手、涨跌停标记、多股票多日期面板。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_panel(
    n_codes: int = 5,
    n_days: int = 30,
    start_price: float = 10.0,
    seed: int = 42,
    include_benchmark: bool = True,
    benchmark_code: str = "000300.SH",
) -> pd.DataFrame:
    """生成合成 OHLCV 面板数据。

    返回列：date, code, open, high, low, close, volume, amount,
            is_limit_up, is_limit_down, pre_close, change_pct
    若 include_benchmark=True，额外附加 benchmark_code 行。
    """
    rng = np.random.default_rng(seed)
    codes = [f"00000{i}.SZ" for i in range(1, n_codes + 1)]
    dates = pd.bdate_range("2024-01-02", periods=n_days)

    rows = []
    for code in codes:
        price = start_price
        for dt in dates:
            ret = rng.normal(0, 0.02)
            pre_close = price
            new_price = round(pre_close * (1 + ret), 2)
            new_price = max(new_price, 1.0)
            change_pct = (new_price - pre_close) / pre_close * 100
            is_limit_up = change_pct >= 9.9
            is_limit_down = change_pct <= -9.9
            # 若涨跌停则锁定价格
            if is_limit_up:
                new_price = round(pre_close * 1.1, 2)
            elif is_limit_down:
                new_price = round(pre_close * 0.9, 2)
            high = max(new_price, pre_close) * (1 + abs(rng.normal(0, 0.005)))
            low = min(new_price, pre_close) * (1 - abs(rng.normal(0, 0.005)))
            volume = int(rng.integers(100000, 500000))
            rows.append({
                "date": dt,
                "code": code,
                "open": round(pre_close, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": new_price,
                "volume": volume,
                "amount": volume * new_price,
                "pre_close": round(pre_close, 2),
                "change_pct": round(change_pct, 4),
                "is_limit_up": is_limit_up,
                "is_limit_down": is_limit_down,
                "turnover_rate": round(rng.uniform(0.5, 3.0), 4),
            })
            price = new_price

    if include_benchmark:
        bm_price = 3500.0
        for dt in dates:
            bm_ret = rng.normal(0, 0.01)
            bm_price = round(bm_price * (1 + bm_ret), 2)
            rows.append({
                "date": dt, "code": benchmark_code,
                "open": bm_price, "high": bm_price, "low": bm_price,
                "close": bm_price, "volume": 0, "amount": 0,
                "pre_close": bm_price, "change_pct": 0,
                "is_limit_up": False, "is_limit_down": False,
                "turnover_rate": 0,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    return df


def make_signals(data: pd.DataFrame, strategy: str = "rotate") -> pd.DataFrame:
    """生成买卖信号。

    strategy='rotate': 每日轮换买入前 2 只、卖出其余
    strategy='buy_day1_sell_day3': 第1天买入，第3天卖出（用于测试 T+1）
    """
    codes = [c for c in data["code"].unique() if c != "000300.SH"]
    dates = sorted(data["date"].unique())

    if strategy == "buy_day1_sell_day3":
        rows = []
        if len(dates) >= 3:
            for code in codes[:2]:
                rows.append({"date": dates[0], "code": code, "signal": 1})
                rows.append({"date": dates[2], "code": code, "signal": -1})
        return pd.DataFrame(rows)

    # rotate
    rows = []
    for i, dt in enumerate(dates):
        buy_idx = i % len(codes)
        for j, code in enumerate(codes):
            if j == buy_idx:
                rows.append({"date": dt, "code": code, "signal": 1})
            elif j == (buy_idx - 1) % len(codes):
                rows.append({"date": dt, "code": code, "signal": -1})
    return pd.DataFrame(rows)