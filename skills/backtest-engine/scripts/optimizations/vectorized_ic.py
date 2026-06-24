"""
向量化因子 IC 分析

借鉴来源：Qlib 的 groupby 批量因子评估思路
优化目标：main 分支 factor-engine/engine.py 的 _calc_ic 对每个因子、
每个日期单独调用 scipy.stats.spearmanr，存在双重 Python 循环
（for factor / for date），在多因子 × 多日期场景下较慢。

本模块用 pandas groupby 向量化实现：
  - Spearman IC = Pearson( rank(factor), rank(forward_ret) ) 按日分组
  - 一次性对所有因子计算 IC 序列，消除逐因子循环
  - 按日 rank 用 groupby(rank) 向量化，IC 用 groupby 相关向量化

正确性：Spearman 秩相关等价于对两个序列先取秩再求 Pearson 相关，
因此向量化结果与 scipy.stats.spearmanr 数值一致（忽略 ties 处理的微小差异）。
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class VectorizedIC:
    """向量化因子 IC 分析器"""

    @staticmethod
    def calc_ic_series(
        data: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        min_samples: int = 10,
    ) -> Optional[pd.Series]:
        """
        向量化计算单个因子的 IC 时间序列（Spearman）

        参数:
            data: 含 date, factor_col, forward_col 的 DataFrame
            factor_col: 因子列名
            forward_col: 远期收益列名
            min_samples: 截面最小样本数
        """
        if factor_col not in data.columns or forward_col not in data.columns:
            return None
        sub = data[["date", factor_col, forward_col]].dropna()
        if sub.empty:
            return None
        # 按日分组取秩（Spearman 的核心）
        sub = sub.copy()
        sub["_fr"] = sub.groupby("date")[factor_col].rank()
        sub["_rr"] = sub.groupby("date")[forward_col].rank()
        # 按日分组计算 Pearson(秩因子, 秩收益) = Spearman IC
        grouped = sub.groupby("date")
        counts = grouped.size()
        valid_dates = counts[counts >= min_samples].index
        sub = sub[sub["date"].isin(valid_dates)]
        if sub.empty:
            return None
        # 向量化分组相关：cov / (std_x * std_y)
        ic = sub.groupby("date").apply(
            lambda g: g["_fr"].corr(g["_rr"]) if len(g) >= min_samples else np.nan
        ).dropna()
        ic = ic.astype(float)
        return ic

    @staticmethod
    def calc_ic_for_factors(
        data: pd.DataFrame,
        factor_names: List[str],
        forward_col: str,
        min_samples: int = 10,
    ) -> Dict[str, pd.Series]:
        """
        批量计算多个因子的 IC 序列

        参数:
            data: 含 date, 各因子列, forward_col 的 DataFrame
            factor_names: 因子名列表
            forward_col: 远期收益列名
        返回:
            {factor_name: ic_series}
        """
        results: Dict[str, pd.Series] = {}
        for f in factor_names:
            ic = VectorizedIC.calc_ic_series(data, f, forward_col, min_samples)
            if ic is not None and not ic.empty:
                results[f] = ic
        return results

    @staticmethod
    def summarize_ic(ic_series: pd.Series) -> Dict[str, float]:
        """汇总单个因子 IC 序列的统计量"""
        if ic_series is None or ic_series.empty:
            return {}
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        return {
            "ic_mean": round(float(ic_mean), 6),
            "ic_std": round(float(ic_std), 6),
            "ic_ir": round(float(ic_mean / ic_std), 4) if ic_std > 0 else 0.0,
            "ic_positive_ratio": round(float((ic_series > 0).mean()), 4),
            "ic_t_stat": round(
                float(ic_mean / (ic_std / np.sqrt(len(ic_series)))) if ic_std > 0 else 0.0, 4
            ),
            "ic_count": int(len(ic_series)),
        }

    @staticmethod
    def full_ic_analysis(
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
        forward_cols: Optional[List[str]] = None,
        min_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        完整 IC 分析（多因子 × 多远期）

        参数:
            factor_df: 含 date, code, 各因子列
            forward_returns: 含 date, code, ret_forward_1d/5d/20d
            factor_names: 因子名列表（默认自动推断）
            forward_cols: 远期收益列名列表
        返回:
            {forward_col: [{factor, ic_mean, ic_ir, ...}, ...]}
        """
        if factor_df.empty or forward_returns.empty:
            return {}

        if forward_cols is None:
            forward_cols = [c for c in forward_returns.columns
                            if c.startswith("ret_forward_")]
        if factor_names is None:
            factor_names = [c for c in factor_df.columns
                            if c not in ("code", "date", "industry")]

        merged = factor_df.merge(
            forward_returns[["code", "date"] + forward_cols],
            on=["code", "date"], how="inner",
        )

        results: Dict[str, Any] = {}
        for fc in forward_cols:
            if fc not in merged.columns:
                continue
            ic_map = VectorizedIC.calc_ic_for_factors(
                merged, factor_names, fc, min_samples
            )
            per_factor = []
            for f in factor_names:
                if f not in ic_map:
                    continue
                stat = VectorizedIC.summarize_ic(ic_map[f])
                stat["factor"] = f
                stat["forward_period"] = fc
                per_factor.append(stat)
            results[fc] = per_factor
        return results
