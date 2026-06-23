"""
测试数据生成器：构造合成 A 股 OHLCV 数据用于验证测试。

设计原则：
    - 可复现（固定随机种子）
    - 覆盖多种场景：正常交易日、停牌、涨跌停
    - 规模可调，便于性能测试
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n_codes: int = 50,
    n_days: int = 250,
    start_date: str = "2024-01-02",
    seed: int = 42,
    include_limits: bool = True,
    include_turnover: bool = True,
    codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    生成合成 A 股日线数据。

    参数:
        n_codes: 股票数量
        n_days: 交易日数量
        start_date: 起始日期
        seed: 随机种子
        include_limits: 是否生成涨跌停标记
        include_turnover: 是否生成换手率

    返回:
        DataFrame，列: code, date, open, high, low, close, volume, amount,
                      [turnover_rate], [is_limit_up], [is_limit_down]
    """
    rng = np.random.default_rng(seed)

    if codes is None:
        codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    dates = pd.bdate_range(start=start_date, periods=n_days)

    records = []
    for code in codes:
        # 每只股票独立的价格路径（带轻微漂移和波动）
        drift = rng.normal(0.0002, 0.0005)
        vol = rng.uniform(0.015, 0.03)
        rets = rng.normal(drift, vol, n_days)
        # 偶尔的跳空
        jumps = rng.choice([0, 0, 0, 0.02, -0.02], size=n_days, p=[0.96, 0.01, 0.01, 0.01, 0.01])
        rets = rets + jumps

        price = 10.0 * np.exp(np.cumsum(rets))
        # 构造 OHLC
        open_ = price * (1 + rng.normal(0, 0.005, n_days))
        close = price
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        volume = rng.integers(1_000_000, 50_000_000, n_days).astype(float)
        amount = volume * (open_ + high + low + close) / 4

        for i in range(n_days):
            rec = {
                'code': code,
                'date': dates[i],
                'open': round(float(open_[i]), 2),
                'high': round(float(high[i]), 2),
                'low': round(float(low[i]), 2),
                'close': round(float(close[i]), 2),
                'volume': float(volume[i]),
                'amount': float(amount[i]),
            }
            if include_turnover:
                rec['turnover_rate'] = round(float(rng.uniform(0.005, 0.08)), 4)

            if include_limits:
                # A 股涨跌停 ±10%（简化）
                if i > 0:
                    chg = (close[i] / close[i - 1] - 1) if close[i - 1] > 0 else 0
                    rec['is_limit_up'] = chg >= 0.095
                    rec['is_limit_down'] = chg <= -0.095
                else:
                    rec['is_limit_up'] = False
                    rec['is_limit_down'] = False

            records.append(rec)

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['date', 'code']).reset_index(drop=True)


def generate_forward_returns(data: pd.DataFrame, periods: List[int] = (1, 5, 20)) -> pd.DataFrame:
    """基于 close 生成远期收益率，用于 IC 分析"""
    df = data[['code', 'date', 'close']].copy()
    df = df.sort_values(['code', 'date'])
    for p in periods:
        df[f'ret_forward_{p}d'] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-p) / x - 1
        )
    return df[['code', 'date'] + [f'ret_forward_{p}d' for p in periods]]


def generate_signals_from_factor(
    factor_df: pd.DataFrame,
    factor_col: str = 'alpha_score',
    top_quantile: float = 0.8,
) -> pd.DataFrame:
    """从因子列生成买卖信号（top quantile 为买入信号）"""
    df = factor_df[['code', 'date', factor_col]].copy()
    df['rank'] = df.groupby('date')[factor_col].rank(pct=True)
    df['signal'] = 0.0
    df.loc[df['rank'] >= top_quantile, 'signal'] = 1.0
    return df[['code', 'date', 'signal']]
