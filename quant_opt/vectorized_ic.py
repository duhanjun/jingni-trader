"""
向量化 IC 分析 —— 借鉴 Qlib IC 分析与 vectorbt 向量化思想

jingni-trader 现有 factor-engine 的 _calc_ic 方法逐日循环：
    for dt in dates:
        cross = data[data['date'] == dt]
        ic, _ = stats.spearmanr(cross[factor], cross[forward])
        ic_list.append(...)
对每个因子、每个前瞻周期都重复 O(交易日) 次 Python 层 spearmanr 调用，极慢。

向量化思路：用 pandas groupby + corr 一次性计算所有日期的 IC 序列。
- Spearman IC = corr(rank(factor), rank(forward_return)) 按日分组
- Pearson IC = corr(factor, forward_return) 按日分组

借鉴来源：
- Qlib IC 分析: https://github.com/microsoft/qlib/blob/main/qlib/contrib/eval.py
- vectorbt 向量化: 用 groupby 替代显式循环
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("vectorized-ic")


class VectorizedICAnalyzer:
    """
    向量化 IC 分析器

    用法:
        analyzer = VectorizedICAnalyzer()
        ic_series = analyzer.calc_ic_series(factor_df, forward_returns, 'my_factor', 'ret_forward_5d')
        ic_summary = analyzer.calc_ic_summary(factor_df, forward_returns, ['f1', 'f2'])
    """

    def calc_ic_series(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        ic_type: str = "spearman",
    ) -> pd.Series:
        """
        向量化计算单个因子的 IC 时间序列

        参数:
            factor_df: 含 code, date, factor_col 的 DataFrame
            forward_returns: 含 code, date, forward_col 的 DataFrame
            factor_col: 因子列名
            forward_col: 前瞻收益列名
            ic_type: "spearman" 或 "pearson"

        返回:
            以 date 为索引的 IC 序列
        """
        merged = factor_df[["code", "date", factor_col]].merge(
            forward_returns[["code", "date", forward_col]],
            on=["code", "date"],
            how="inner",
        )
        merged = merged.dropna(subset=[factor_col, forward_col])

        if merged.empty:
            return pd.Series(dtype=float)

        if ic_type == "spearman":
            # Spearman = Pearson on ranks
            merged[factor_col] = merged.groupby("date")[factor_col].rank()
            merged[forward_col] = merged.groupby("date")[forward_col].rank()

        # 向量化：按 date 分组计算 corr，一次完成
        # 使用 groupby + corr，避免逐日循环
        ic_series = merged.groupby("date").apply(
            lambda g: g[factor_col].corr(g[forward_col])
            if len(g) >= 10 else np.nan
        )
        ic_series = ic_series.dropna()
        ic_series.name = f"ic_{factor_col}_{forward_col}"
        return ic_series

    def calc_ic_summary(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
        forward_cols: Optional[List[str]] = None,
        ic_type: str = "spearman",
    ) -> Dict[str, Any]:
        """
        向量化计算多因子 IC 汇总统计

        参数:
            factor_df: 含 code, date, 各因子列的 DataFrame
            forward_returns: 含 code, date, 各前瞻收益列的 DataFrame
            factor_names: 因子列名列表，默认自动推断
            forward_cols: 前瞻收益列名列表，默认 ['ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']
            ic_type: "spearman" 或 "pearson"

        返回:
            {forward_col: [{factor, ic_mean, ic_std, ic_ir, ic_positive_ratio, ic_t_stat}, ...]}
        """
        if factor_df.empty or forward_returns.empty:
            return {}

        if factor_names is None:
            factor_names = [
                c for c in factor_df.columns
                if c not in ("code", "date", "industry")
            ]

        if forward_cols is None:
            forward_cols = [
                c for c in forward_returns.columns
                if c.startswith("ret_forward_")
            ]

        # 一次性 merge，避免每个因子重复 merge
        all_cols = ["code", "date"] + [
            f for f in factor_names if f in factor_df.columns
        ] + [c for c in forward_cols if c in forward_returns.columns]
        merged = factor_df.merge(
            forward_returns[["code", "date"] + forward_cols],
            on=["code", "date"],
            how="inner",
        )

        results: Dict[str, List[Dict[str, Any]]] = {}

        for forward_col in forward_cols:
            if forward_col not in merged.columns:
                continue

            ic_list = []
            for factor in factor_names:
                if factor not in merged.columns:
                    continue

                ic_series = self._calc_ic_from_merged(
                    merged, factor, forward_col, ic_type
                )
                if ic_series.empty:
                    continue

                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                ic_ir = ic_mean / ic_std if ic_std > 0 else 0
                n = len(ic_series)
                ic_t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0

                ic_list.append({
                    "factor": factor,
                    "forward_period": forward_col,
                    "ic_mean": round(float(ic_mean), 6),
                    "ic_std": round(float(ic_std), 6),
                    "ic_ir": round(float(ic_ir), 4),
                    "ic_positive_ratio": round(float((ic_series > 0).mean()), 4),
                    "ic_t_stat": round(float(ic_t_stat), 4),
                })

            results[forward_col] = ic_list

        return results

    @staticmethod
    def _calc_ic_from_merged(
        merged: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        ic_type: str,
    ) -> pd.Series:
        """从已 merge 的 DataFrame 计算 IC 序列（向量化）"""
        sub = merged.dropna(subset=[factor_col, forward_col])
        if sub.empty:
            return pd.Series(dtype=float)

        if ic_type == "spearman":
            sub = sub.copy()
            sub[factor_col] = sub.groupby("date")[factor_col].rank()
            sub[forward_col] = sub.groupby("date")[forward_col].rank()

        # 按 date 分组计算 corr，过滤样本不足的日期
        ic_series = sub.groupby("date").apply(
            lambda g: g[factor_col].corr(g[forward_col])
            if len(g.dropna(subset=[factor_col, forward_col])) >= 10
            else np.nan
        )
        return ic_series.dropna()


def calc_ic_vectorized(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    ic_type: str = "spearman",
) -> Dict[str, Any]:
    """
    便捷函数：向量化 IC 分析

    与 factor-engine.ic_analysis 接口对齐，便于对比验证
    """
    analyzer = VectorizedICAnalyzer()
    return analyzer.calc_ic_summary(
        factor_df, forward_returns, factor_names, ic_type=ic_type
    )
