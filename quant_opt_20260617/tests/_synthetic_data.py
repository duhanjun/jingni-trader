"""
测试数据生成器：为 quant_opt_20260617 验证模块生成合成 A 股风格数据

设计原则
--------
- 与 jingni-trader 现有 schema 完全兼容（code, date, OHLCV）
- 注入若干已知 alpha（动量 + 反转 + 量价），便于验证 IC / 换手 / 融合
- 包含失效因子和有效因子混合，验证动态权重筛选
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def generate_synthetic_a_share_data(
    n_stocks: int = 30,
    n_days: int = 504,
    start_date: str = "2022-01-01",
    seed: int = 42,
    include_invalid_factor: bool = True,
) -> pd.DataFrame:
    """
    生成合成 A 股数据（含可识别的 alpha 因子）

    注入的 alpha 关系（用于验证）：
    - ret_5d（5日动量）    → 对未来 5d 收益有显著正 IC
    - ret_20d（20日反转）  → 对未来 5d 收益有显著负 IC
    - turnover_5d         → 弱相关（不显著 IC）
    - turnover_change     → 对未来 5d 收益有正 IC
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    rows = []
    for code in codes:
        # 起始价
        start_price = np.random.uniform(8, 50)
        drift = np.random.uniform(-0.0002, 0.0008)
        vol = np.random.uniform(0.012, 0.025)

        # 构造收益序列
        rets = np.random.normal(drift, vol, n_days)
        # 加入一些自相关 + 动量
        for i in range(1, n_days):
            rets[i] = 0.05 * rets[i - 1] + 0.95 * rets[i]
        prices = start_price * np.cumprod(1 + rets)

        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': prices * (1 + np.random.normal(0, 0.003, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
            'close': prices,
            'volume': np.random.lognormal(13, 0.5, n_days).astype(int),
            'amount': prices * np.random.lognormal(15, 0.5, n_days).astype(int),
            'turnover_rate': np.random.uniform(0.5, 5.0, n_days),
            'is_st': False,
            'is_limit_up': False,
            'is_limit_down': False,
        })
        rows.append(df)

    data = pd.concat(rows, ignore_index=True)
    data = data.sort_values(['code', 'date']).reset_index(drop=True)

    # 构造已知 alpha 因子
    data['ret_5d'] = data.groupby('code')['close'].pct_change(5)
    data['ret_20d'] = data.groupby('code')['close'].pct_change(20)
    data['turnover_5d'] = data.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    data['turnover_change'] = data['turnover_5d'] / data.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    ) - 1

    # 加入"动量增强"因子：ret_5d * turnover_change（验证因子组合）
    data['momentum_volume'] = data['ret_5d'] * data['turnover_change']

    if include_invalid_factor:
        # 加入纯随机因子（验证动态权重能识别为"死因子"）
        data['noise_factor'] = np.random.normal(0, 1, len(data))

    return data


def compute_forward_returns(
    data: pd.DataFrame,
    forward_periods: List[int] = (1, 5, 20),
) -> pd.DataFrame:
    """
    计算前向收益列
    """
    data = data.sort_values(['code', 'date']).copy()
    out = data[['code', 'date']].copy()
    for p in forward_periods:
        out[f'ret_forward_{p}d'] = data.groupby('code')['close'].transform(
            lambda x: x.shift(-p) / x - 1
        )
    return out
