"""
合成 A 股数据生成器
===================
用于量化优化验证的确定性测试数据。

借鉴思路：
- AKShare / Tushare 的真实 A 股日线字段结构（code/date/OHLCV/amount/turnover_rate/change_pct/涨跌停标记）
- 用随机游走生成“足够真实”的价格序列，便于回测/因子/IC 验证

提供：
- generate_test_data(n_stocks, n_days, seed) -> (data, signals)
  * data 列: code, date, open, high, low, close, volume, amount,
             turnover_rate, change_pct, is_limit_up, is_limit_down
  * signals 列: code, date, signal (取值 {-1, 0, 1})，由 MA5/MA20 金叉死叉生成
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def generate_test_data(n_stocks: int = 50, n_days: int = 300, seed: int = 42):
    """
    生成合成 A 股日线数据 + MA 交叉信号。

    参数:
        n_stocks: 股票数量
        n_days:   交易日天数
        seed:     随机种子（确定性可复现）

    返回:
        (data, signals) 两个 DataFrame
    """
    rng = np.random.default_rng(seed)

    # 交易日（工作日序列）
    dates = pd.date_range(start="2020-01-02", periods=n_days, freq="B")

    # 股票代码：沪市主板风格 600000.SH 起步
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    rows = []
    for code in codes:
        # 每只股票一个独立的起始价 (8~30 元)
        price = float(rng.uniform(8.0, 30.0))
        for dt in dates:
            # 日收益率：均值为小幅漂移，波动 2%
            ret = rng.normal(0.0005, 0.02)
            prev_close = price
            new_close = prev_close * (1.0 + ret)
            if new_close < 1.0:
                new_close = 1.0

            open_p = prev_close * (1.0 + rng.normal(0.0, 0.005))
            high = max(open_p, new_close) * (1.0 + abs(rng.normal(0.0, 0.004)))
            low = min(open_p, new_close) * (1.0 - abs(rng.normal(0.0, 0.004)))
            if low < 0.05:
                low = 0.05
            if high < new_close:
                high = new_close

            volume = int(rng.integers(100_000, 1_000_000))
            amount = volume * new_close
            turnover_rate = float(rng.uniform(0.5, 5.0))
            change_pct = (new_close - prev_close) / prev_close * 100.0

            # A 股主板涨跌停 ±10%
            is_limit_up = bool(change_pct >= 9.9)
            is_limit_down = bool(change_pct <= -9.9)

            rows.append({
                "code": code,
                "date": dt,
                "open": round(open_p, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(new_close, 4),
                "volume": volume,
                "amount": round(amount, 2),
                "turnover_rate": round(turnover_rate, 4),
                "change_pct": round(change_pct, 4),
                "is_limit_up": is_limit_up,
                "is_limit_down": is_limit_down,
            })
            price = new_close

    data = pd.DataFrame(rows)
    data = data.sort_values(["code", "date"]).reset_index(drop=True)

    # ---- 信号：MA5 / MA20 金叉死叉 ----
    df = data.copy()
    df["ma5"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["ma20"] = df.groupby("code")["close"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    prev_ma5 = df.groupby("code")["ma5"].shift(1)
    prev_ma20 = df.groupby("code")["ma20"].shift(1)

    golden = ((df["ma5"] > df["ma20"]) & (prev_ma5 <= prev_ma20)).fillna(False)
    death = ((df["ma5"] < df["ma20"]) & (prev_ma5 >= prev_ma20)).fillna(False)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig[golden] = 1
    sig[death] = -1

    # 信号文件只保留“有动作”的行（signal != 0），与真实信号文件口径一致；
    # 这样回测逐日循环里 day_signal 的 iterrows 开销可忽略，
    # 让“消除 O(n²) 布尔掩码”的优化效果在性能对比中更清晰地体现。
    mask = sig != 0
    signals = pd.DataFrame({
        "code": df["code"].values[mask],
        "date": df["date"].values[mask],
        "signal": sig.values[mask],
    })
    signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

    return data, signals


if __name__ == "__main__":
    d, s = generate_test_data(n_stocks=10, n_days=60, seed=42)
    print("data shape:", d.shape)
    print("signals shape:", s.shape)
    print("signal value counts:\n", s["signal"].value_counts())
    print(d.head())
