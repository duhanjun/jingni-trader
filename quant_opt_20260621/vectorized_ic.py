"""
向量化 IC 分析（验证原型）

优化目标：
jingni-trader 现有 skills/factor-engine/engine.py 的 _calc_ic 方法使用
`for dt in dates: data[data['date'] == dt]` 逐日循环计算 Spearman/Pearson 相关，
复杂度 O(n_days * n_rows)，在因子筛选阶段（多因子 × 多周期）极慢。

本模块通过 groupby('date') 一次性向量化计算所有日期的 IC，
消除 per-date Python 循环。

借鉴：VectorBT 的「将多实例打包为多维数组一次处理」思想。
"""
from __future__ import annotations
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats


class VectorizedIC:
    """向量化 IC 分析"""

    @staticmethod
    def calc_ic_series(data: pd.DataFrame, factor_col: str,
                       forward_col: str, ic_type: str = "spearman") -> Optional[pd.Series]:
        """
        一次性计算因子对远期收益的 IC 时间序列

        参数:
            data: 含 code, date, factor_col, forward_col 的 DataFrame
            factor_col: 因子列名
            forward_col: 远期收益列名
            ic_type: "spearman" 或 "pearson"

        返回:
            以 date 为索引的 IC 序列
        """
        if factor_col not in data.columns or forward_col not in data.columns:
            return None

        sub = data[['date', factor_col, forward_col]].dropna()
        if sub.empty:
            return None

        if ic_type == "spearman":
            # groupby + rank 后用 Pearson 等价于 Spearman
            ranked = sub.copy()
            ranked[factor_col] = sub.groupby('date')[factor_col].rank()
            ranked[forward_col] = sub.groupby('date')[forward_col].rank()
            ic_series = ranked.groupby('date').apply(
                lambda g: VectorizedIC._safe_pearson(g[factor_col].values, g[forward_col].values)
            )
        else:
            ic_series = sub.groupby('date').apply(
                lambda g: VectorizedIC._safe_pearson(g[factor_col].values, g[forward_col].values)
            )

        ic_series = ic_series.dropna()
        if ic_series.empty:
            return None
        ic_series.index = pd.to_datetime(ic_series.index)
        return ic_series

    @staticmethod
    def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
        """单点 Pearson，要求至少 10 个样本"""
        if len(x) < 10:
            return np.nan
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            return np.nan
        xv, yv = x[mask], y[mask]
        if xv.std() == 0 or yv.std() == 0:
            return np.nan
        r, _ = stats.pearsonr(xv, yv)
        return float(r)

    @staticmethod
    def calc_ic_summary(data: pd.DataFrame, factor_col: str,
                        forward_col: str, ic_type: str = "spearman") -> dict:
        """计算 IC 摘要统计"""
        ic_series = VectorizedIC.calc_ic_series(data, factor_col, forward_col, ic_type)
        if ic_series is None or ic_series.empty:
            return {}
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        n = len(ic_series)
        return {
            "factor": factor_col,
            "forward_period": forward_col,
            "ic_mean": round(float(ic_mean), 6),
            "ic_std": round(float(ic_std), 6),
            "ic_ir": round(float(ic_ir), 4),
            "ic_positive_ratio": round(float((ic_series > 0).mean()), 4),
            "ic_t_stat": round(float(ic_mean / (ic_std / np.sqrt(n))) if ic_std > 0 else 0, 4),
            "n_periods": n,
        }

    @staticmethod
    def batch_ic(data: pd.DataFrame, factor_names: List[str],
                 forward_cols: List[str], ic_type: str = "spearman") -> dict:
        """批量计算多因子 × 多周期的 IC"""
        results = {}
        for fwd in forward_cols:
            if fwd not in data.columns:
                continue
            rows = []
            for f in factor_names:
                if f not in data.columns:
                    continue
                summary = VectorizedIC.calc_ic_summary(data, f, fwd, ic_type)
                if summary:
                    rows.append(summary)
            results[fwd] = rows
        return results
