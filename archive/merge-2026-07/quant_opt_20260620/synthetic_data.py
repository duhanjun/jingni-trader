"""
合成数据生成器
为回测/因子优化验证提供可复现的测试数据

设计目标:
- 完全确定性（固定随机种子）
- 覆盖正常行情、涨跌停、ST、缺失值等场景
- 输出格式与 jingni-trader data-engine 产物一致 (code, date, open/high/low/close/volume/amount/turnover_rate/is_st/is_limit_up/is_limit_down)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def generate_panel(
    n_codes: int = 50,
    n_days: int = 250,
    start_price: float = 10.0,
    seed: int = 42,
    with_limits: bool = True,
    with_st: bool = True,
    vol_base: float = 1e6,
) -> pd.DataFrame:
    """
    生成多股票面板行情数据

    参数:
        n_codes: 股票数量
        n_days: 交易日数量
        start_price: 起始价格
        seed: 随机种子
        with_limits: 是否生成涨跌停标记
        with_st: 是否生成 ST 标记
        vol_base: 成交量基准

    返回:
        DataFrame, 列: code, date, open, high, low, close, volume, amount,
                       turnover_rate, is_st, is_limit_up, is_limit_down
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    rows = []
    for code in codes:
        # 几何布朗运动生成价格
        mu = rng.normal(0.0005, 0.0008)
        sigma = rng.uniform(0.015, 0.03)
        log_ret = rng.normal(mu, sigma, n_days)
        close = start_price * np.exp(np.cumsum(log_ret))
        # open 略微偏离前一日 close（模拟跳空）
        open_ = np.empty(n_days)
        open_[0] = start_price
        open_[1:] = close[:-1] * (1 + rng.normal(0, 0.005, n_days - 1))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        volume = vol_base * (1 + rng.uniform(-0.3, 0.5, n_days))
        amount = volume * (open_ + close) / 2
        turnover_rate = volume / (1e8 * rng.uniform(0.5, 2.0))

        df = pd.DataFrame({
            "code": code,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "turnover_rate": turnover_rate,
        })

        if with_limits:
            # A股涨跌停 ±10%
            prev_close = df["close"].shift(1).fillna(start_price)
            limit_up_price = prev_close * 1.1
            limit_down_price = prev_close * 0.9
            df["is_limit_up"] = (df["close"] >= limit_up_price * 0.999).astype(bool)
            df["is_limit_down"] = (df["close"] <= limit_down_price * 1.001).astype(bool)
            # 首日不标记
            df.loc[df.index[0], ["is_limit_up", "is_limit_down"]] = False

        if with_st:
            # 随机标记少量股票为 ST
            is_st_code = code in codes[: max(1, n_codes // 20)]
            df["is_st"] = is_st_code

        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)
    return panel


def generate_signals(
    data: pd.DataFrame,
    strategy: str = "reversal",
    top_pct: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    基于行情数据生成交易信号

    策略:
        - reversal: 20日反转（跌多了买入）
        - momentum: 20日动量（涨多了买入）
        - random: 随机信号（用于性能测试）

    信号语义与 jingni-trader 一致:
        signal=1 买入, signal=-1 卖出, signal=0 持有/无信号
    """
    rng = np.random.default_rng(seed)
    df = data.sort_values(["code", "date"]).copy()
    df["ret_20d"] = df.groupby("code")["close"].pct_change(20)

    if strategy == "reversal":
        df["score"] = -df["ret_20d"]
    elif strategy == "momentum":
        df["score"] = df["ret_20d"]
    elif strategy == "random":
        df["score"] = rng.standard_normal(len(df))
    else:
        raise ValueError(f"未知策略: {strategy}")

    df["rank_pct"] = df.groupby("date")["score"].rank(pct=True)
    df["signal"] = 0
    df.loc[df["rank_pct"] >= (1 - top_pct), "signal"] = 1
    # 卖出信号：排名靠后的持仓标的
    df.loc[df["rank_pct"] <= top_pct, "signal"] = -1

    return df[["code", "date", "signal"]].reset_index(drop=True)


def generate_factor_panel(
    data: pd.DataFrame,
    n_extra_factors: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    基于行情数据生成因子面板（含行业字段），用于中性化/IC 分析验证
    """
    rng = np.random.default_rng(seed)
    df = data.sort_values(["code", "date"]).copy()

    df["ret_5d"] = df.groupby("code")["close"].pct_change(5)
    df["ret_20d"] = df.groupby("code")["close"].pct_change(20)
    df["reversal_20d"] = -df["ret_20d"]
    df["volatility_20d"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )
    df["turnover_20d"] = df.groupby("code")["turnover_rate"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    df["lncap"] = np.log(df["amount"] / df["turnover_rate"].replace(0, np.nan) * 100)
    # 额外随机因子
    for i in range(n_extra_factors):
        df[f"factor_{i}"] = rng.standard_normal(len(df))

    # 行业字段（随机分配 5 个行业）
    industries = [f"IND_{i}" for i in range(5)]
    code_to_industry = {c: industries[i % len(industries)] for i, c in enumerate(df["code"].unique())}
    df["industry"] = df["code"].map(code_to_industry)

    factor_cols = ["reversal_20d", "volatility_20d", "turnover_20d"] + [
        f"factor_{i}" for i in range(n_extra_factors)
    ]
    return df[["code", "date", "industry", "lncap"] + factor_cols].reset_index(drop=True)


if __name__ == "__main__":
    # 自检：生成数据并打印摘要
    panel = generate_panel(n_codes=30, n_days=120)
    sig = generate_signals(panel, strategy="reversal")
    fac = generate_factor_panel(panel)
    print(f"行情面板: {panel.shape}, 日期范围: {panel['date'].min()} ~ {panel['date'].max()}")
    print(f"信号: {sig.shape}, 买入信号数: {(sig['signal']==1).sum()}, 卖出: {(sig['signal']==-1).sum()}")
    print(f"因子面板: {fac.shape}, 因子列: {[c for c in fac.columns if c not in ['code','date','industry','lncap']]}")
