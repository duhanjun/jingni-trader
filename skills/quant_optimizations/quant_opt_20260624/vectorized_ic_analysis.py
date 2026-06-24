"""
向量化 IC 分析

借鉴来源：
    - Quantopian Alphalens：因子 IC / 分层收益分析的标准实现
    - Qlib 的 ic 分析：基于 groupby 的批量 IC 计算

对比 jingni-trader 现状：
    skills/factor-engine/engine.py 的 _calc_ic 使用
    `for dt in dates: cross = data[data['date']==dt]; stats.spearmanr(...)`
    逐日 Python 循环 + 逐日 scipy 调用，在日期数多时是显著瓶颈。

本实现的核心改进：
    将 Spearman/Pearson IC 的逐日计算【完全向量化】为 groupby transform：
        corr = sum((x-x̄)(y-ȳ)) / sqrt(sum((x-x̄)²) · sum((y-ȳ)²))
    全程零 Python 逐日循环、零 apply，计算下沉到 C 层。

    对 Spearman IC：先对 (factor, forward_return) 在每个截面内 rank，
    再对 rank 序列做上述向量化相关系数计算（数值上等价于 spearmanr）。
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def _vectorized_corr_per_group(
    x: pd.Series, y: pd.Series, group: pd.Series
) -> pd.Series:
    """
    向量化计算每个 group 内 (x, y) 的 Pearson 相关系数。

    返回与 x 等长的 Series（每行填充其所属 group 的相关系数），
    调用方可用 .groupby(group).first() 取每组的唯一值。

    数学等价于 scipy.stats.pearsonr（无 NaN 时）。
    """
    # 按 group 计算均值（transform 保持原索引）
    x_mean = x.groupby(group).transform("mean")
    y_mean = y.groupby(group).transform("mean")

    xc = x - x_mean
    yc = y - y_mean

    xy = xc * yc
    xx = xc * xc
    yy = yc * yc

    sum_xy = xy.groupby(group).transform("sum")
    sum_xx = xx.groupby(group).transform("sum")
    sum_yy = yy.groupby(group).transform("sum")

    denom = np.sqrt(sum_xx * sum_yy)
    # 避免除零
    corr = sum_xy / denom.replace(0, np.nan)
    return corr


class VectorizedICAnalysis:
    """向量化 IC 分析引擎"""

    @staticmethod
    def calc_ic_series(
        data: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        ic_type: str = "spearman",
        min_stocks: int = 10,
    ) -> pd.Series:
        """
        计算单因子的 IC 时间序列（向量化）

        参数:
            data: 含 date, factor_col, forward_col 的 DataFrame
            factor_col: 因子列名
            forward_col: 远期收益列名
            ic_type: "spearman" 或 "pearson"
            min_stocks: 截面最少股票数，否则该日 IC 置 NaN

        返回:
            Series，index=date，value=IC
        """
        if factor_col not in data.columns or forward_col not in data.columns:
            return pd.Series(dtype=float)

        df = data[["date", factor_col, forward_col]].dropna().copy()
        if df.empty:
            return pd.Series(dtype=float)

        # 截面股票数过滤（向量化）
        counts = df.groupby("date")[factor_col].transform("size")
        df = df[counts >= min_stocks]
        if df.empty:
            return pd.Series(dtype=float)

        x = df[factor_col].astype(float)
        y = df[forward_col].astype(float)
        g = df["date"]

        if ic_type == "spearman":
            # 先截面 rank，再做 Pearson —— 数值上等价于 spearmanr
            x = x.groupby(g).rank(pct=True)
            y = y.groupby(g).rank(pct=True)

        corr_per_row = _vectorized_corr_per_group(x, y, g)
        # 每个 date 取一行作为该日 IC
        ic_series = corr_per_row.groupby(g).first()
        ic_series.name = "ic"
        return ic_series

    @staticmethod
    def calc_ic_stats(
        data: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        ic_type: str = "spearman",
        min_stocks: int = 10,
    ) -> Dict[str, float]:
        """计算单因子的 IC 统计量（mean/std/IR/t_stat/positive_ratio）"""
        ic = VectorizedICAnalysis.calc_ic_series(
            data, factor_col, forward_col, ic_type, min_stocks
        )
        if ic.empty:
            return {}
        ic_mean = float(ic.mean())
        ic_std = float(ic.std())
        n = len(ic)
        return {
            "factor": factor_col,
            "forward_period": forward_col,
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "ic_ir": round(ic_mean / ic_std, 4) if ic_std > 0 else 0.0,
            "ic_positive_ratio": round(float((ic > 0).mean()), 4),
            "ic_t_stat": round(ic_mean / (ic_std / np.sqrt(n)), 4) if ic_std > 0 and n > 0 else 0.0,
            "n_days": n,
        }

    @staticmethod
    def calc_ic_matrix(
        data: pd.DataFrame,
        factor_names: List[str],
        forward_cols: List[str],
        ic_type: str = "spearman",
        min_stocks: int = 10,
    ) -> Dict[str, Any]:
        """
        批量计算多因子 × 多远期收益的 IC（向量化）

        返回:
            { forward_col: [ {factor, ic_mean, ic_ir, ...}, ... ] }
            结构与 jingni-trader FactorEngine.ic_analysis 输出一致，便于对比。
        """
        results: Dict[str, Any] = {}
        for fwd in forward_cols:
            if fwd not in data.columns:
                continue
            per_fwd = []
            for f in factor_names:
                if f not in data.columns:
                    continue
                stat = VectorizedICAnalysis.calc_ic_stats(data, f, fwd, ic_type, min_stocks)
                if stat:
                    per_fwd.append(stat)
            results[fwd] = per_fwd
        return results

    @staticmethod
    def calc_quantile_returns(
        data: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        n_quantiles: int = 5,
        min_stocks: int = 10,
    ) -> pd.DataFrame:
        """
        向量化分层收益（借鉴 Alphalens）

        返回: DataFrame, index=date, columns=分位组, values=组内平均远期收益
        """
        if factor_col not in data.columns or forward_col not in data.columns:
            return pd.DataFrame()

        df = data[["date", factor_col, forward_col]].dropna().copy()
        counts = df.groupby("date")[factor_col].transform("size")
        df = df[counts >= min_stocks]
        if df.empty:
            return pd.DataFrame()

        # 截面分位（向量化）
        df["quantile"] = df.groupby("date")[factor_col].transform(
            lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop")
        )
        df = df.dropna(subset=["quantile"])
        # 组内平均远期收益
        return df.groupby(["date", "quantile"])[forward_col].mean().unstack("quantile")


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from synthetic_data import generate_synthetic_ohlcv
    import pandas as pd

    data = generate_synthetic_ohlcv(n_codes=50, n_days=120)
    # 构造因子与远期收益
    df = data.sort_values(["code", "date"]).copy()
    df["factor"] = -df.groupby("code")["close"].transform(lambda x: x.pct_change(5))  # 5日反转
    df["fwd_5d"] = df.groupby("code")["close"].transform(lambda x: x.shift(-5) / x - 1)

    ic = VectorizedICAnalysis.calc_ic_series(df, "factor", "fwd_5d", "spearman")
    print("IC 序列长度:", len(ic))
    print("IC 均值:", round(ic.mean(), 4), "IR:", round(ic.mean() / ic.std(), 4) if ic.std() > 0 else 0)
    print(VectorizedICAnalysis.calc_ic_stats(df, "factor", "fwd_5d", "spearman"))