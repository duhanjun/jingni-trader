"""
向量化 IC 分析（验证用）
========================
OPTIMIZATION 3 (part 1): 把 factor-engine `_calc_ic` 的 O(n²) 逐日布尔掩码循环
替换为 groupby('date').apply 的向量化实现。

借鉴来源：
- Qlib 的 IC 计算用 groupby 向量化
- 原始 factor-engine/engine.py `_calc_ic` 用 `for dt in dates: cross = data[data['date']==dt]`

提供：
- calc_ic_original:  复刻原始逐日循环（基准对比）
- calc_ic_vectorized: groupby('date').apply + scipy.stats.spearmanr

两者结果一致（IC Series 按日期索引，忽略 NaN）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def _spearman_ic(group: pd.DataFrame, factor_col: str, forward_col: str) -> float:
    """对单日横截面计算 Spearman 秩相关 IC"""
    if len(group) < 10:
        return np.nan
    ic, _ = stats.spearmanr(group[factor_col], group[forward_col], nan_policy="omit")
    return float(ic) if not np.isnan(ic) else np.nan


def calc_ic_original(factor_df: pd.DataFrame, factor_col: str, forward_col: str) -> pd.Series:
    """
    原始 O(n²) 实现：逐日布尔掩码 + spearmanr。
    与 factor-engine/engine.py `_calc_ic` 逻辑一致。
    返回按 date 索引的 IC Series。
    """
    if factor_col not in factor_df.columns or forward_col not in factor_df.columns:
        return pd.Series(dtype=float)

    ic_list = []
    dates = sorted(factor_df["date"].unique())
    for dt in dates:
        cross = factor_df[factor_df["date"] == dt].dropna(subset=[factor_col, forward_col])
        if len(cross) < 10:
            continue
        ic, _ = stats.spearmanr(cross[factor_col], cross[forward_col], nan_policy="omit")
        if not np.isnan(ic):
            ic_list.append({"date": dt, "ic": float(ic)})

    if not ic_list:
        return pd.Series(dtype=float)
    ic_df = pd.DataFrame(ic_list)
    ic_df["date"] = pd.to_datetime(ic_df["date"])
    return ic_df.set_index("date")["ic"]


def calc_ic_vectorized(factor_df: pd.DataFrame, factor_col: str, forward_col: str) -> pd.Series:
    """
    向量化实现：先全局 dropna，再 groupby('date').apply 计算每日 IC。
    消除逐日布尔掩码的 O(n²) 开销。

    与原始实现等价：
    - 全局 dropna(factor_col, forward_col) 后按 date 分组，等价于每日 dropna
    - len(group) < 10 返回 NaN，最后 dropna，等价于原始 continue
    - 索引转为 DatetimeIndex，与原始一致
    """
    if factor_col not in factor_df.columns or forward_col not in factor_df.columns:
        return pd.Series(dtype=float)

    df = factor_df.dropna(subset=[factor_col, forward_col]).copy()
    if df.empty:
        return pd.Series(dtype=float)

    # 仅保留需要的列，避免 groupby.apply 把分组列带入产生告警/歧义
    sub = df[["date", factor_col, forward_col]]
    ic_series = sub.groupby("date", sort=True).apply(
        lambda g: _spearman_ic(g, factor_col, forward_col)
    )
    ic_series = ic_series.dropna()
    # 与原始一致：索引转为 DatetimeIndex
    ic_series.index = pd.to_datetime(ic_series.index)
    ic_series.name = "ic"
    return ic_series


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_test_data

    data, _ = generate_test_data(n_stocks=30, n_days=120, seed=7)
    # 构造一个因子列与 forward 收益列
    df = data.sort_values(["code", "date"]).copy()
    df["factor"] = df.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["forward"] = df.groupby("code")["close"].shift(-5) / df["close"] - 1

    ic_o = calc_ic_original(df, "factor", "forward")
    ic_v = calc_ic_vectorized(df, "factor", "forward")
    print("original IC count:", len(ic_o), "mean:", ic_o.mean())
    print("vectorized IC count:", len(ic_v), "mean:", ic_v.mean())
    common = ic_o.index.intersection(ic_v.index)
    print("max abs diff:", (ic_o.loc[common] - ic_v.loc[common]).abs().max())
