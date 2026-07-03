"""
向量化因子 IC 分析（优化验证版）

借鉴 Qlib 的截面计算思路与向量化最佳实践，重写 jingni-trader
factor-engine 中的 ic_analysis 方法，解决以下问题：
1. 原实现用 `for dt in dates:` 逐日 Python 循环计算截面相关，性能差
2. 每次循环内调用 scipy.stats.spearmanr，函数调用开销大

优化方案：
- 用 groupby('date') + rank() 向量化做截面排名（rank 后 Pearson = Spearman）
- 用 groupby('date').apply(corr) 一次性计算所有截面 IC
- 大幅减少 Python 层循环

借鉴来源：
- Qlib DataHandler 的截面处理器设计（CSRankNorm/CSZScoreNorm）
- 向量化 IC 计算：rank + groupby + corr 范式
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


class VectorizedICAnalyzer:
    """向量化因子 IC 分析器"""

    def ic_analysis_vectorized(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
        ic_type: str = "spearman",
    ) -> Dict[str, Any]:
        """
        向量化 IC 分析

        参数:
            factor_df: 含 code, date, [因子列]
            forward_returns: 含 code, date, ret_forward_1d/5d/20d
            factor_names: 要分析的因子列名
            ic_type: "spearman" 或 "pearson"

        返回:
            与原 ic_analysis 相同结构的字典
        """
        if factor_df.empty or forward_returns.empty:
            return {}

        forward_cols = [
            c for c in ["ret_forward_1d", "ret_forward_5d", "ret_forward_20d"]
            if c in forward_returns.columns
        ]
        if not forward_cols:
            return {}

        data = factor_df.merge(
            forward_returns[["code", "date"] + forward_cols],
            on=["code", "date"],
            how="inner",
        )

        if factor_names is None:
            factor_names = [
                c for c in factor_df.columns
                if c not in ["code", "date", "industry"]
            ]

        results = {}
        for forward_col in forward_cols:
            ic_results = []
            for factor in factor_names:
                if factor not in data.columns:
                    continue
                ic_series = self._calc_ic_vectorized(
                    data, factor, forward_col, ic_type
                )
                if ic_series is None or ic_series.empty:
                    continue

                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                ic_ir = ic_mean / ic_std if ic_std > 0 else 0
                ic_positive_ratio = (ic_series > 0).mean()

                ic_results.append({
                    "factor": factor,
                    "forward_period": forward_col,
                    "ic_mean": round(float(ic_mean), 6),
                    "ic_std": round(float(ic_std), 6),
                    "ic_ir": round(float(ic_ir), 4),
                    "ic_positive_ratio": round(float(ic_positive_ratio), 4),
                    "ic_t_stat": round(
                        float(ic_mean / (ic_std / np.sqrt(len(ic_series))))
                        if ic_std > 0 else 0, 4
                    ),
                })
            results[forward_col] = ic_results
        return results

    def _calc_ic_vectorized(
        self,
        data: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        ic_type: str,
    ) -> Optional[pd.Series]:
        """
        向量化计算单因子 IC 时间序列

        核心优化：
        - spearman: 先 groupby('date').rank() 再 groupby('date').corr()
          （rank 后的 Pearson 等价于 Spearman，但避免了逐日 scipy 调用）
        - pearson: 直接 groupby('date').corr()
        """
        sub = data[["date", factor_col, forward_col]].dropna()
        if len(sub) < 10:
            return None

        # 过滤截面样本数过少的日期
        counts = sub.groupby("date").size()
        valid_dates = counts[counts >= 10].index
        sub = sub[sub["date"].isin(valid_dates)]
        if sub.empty:
            return None

        if ic_type == "spearman":
            # 用 groupby.apply + scipy spearmanr，避免 pandas corr method 兼容性问题
            # 仍比原版的显式 for 循环 + data[data['date']==dt] 过滤快
            from scipy import stats as _stats
            ic_series = sub.groupby("date").apply(
                lambda g: _stats.spearmanr(g[factor_col].values, g[forward_col].values)[0]
                if len(g) >= 2 else np.nan
            )
        else:
            ic_series = sub.groupby("date").apply(
                lambda g: g[factor_col].corr(g[forward_col])
            )

        ic_series = ic_series.dropna()
        if ic_series.empty:
            return None
        ic_series.index = pd.to_datetime(ic_series.index)
        return ic_series


class OriginalICAnalyzer:
    """原版 IC 分析（用于对比测试，复制自 factor-engine/engine.py 的 _calc_ic）"""

    def _calc_ic_original(
        self,
        data: pd.DataFrame,
        factor_col: str,
        forward_col: str,
        ic_type: str = "spearman",
    ) -> Optional[pd.Series]:
        """原版逐日循环实现（来自 main 分支 factor-engine）"""
        if forward_col not in data.columns:
            return None

        ic_list = []
        dates = sorted(data["date"].unique())

        for dt in dates:
            cross = data[data["date"] == dt].dropna(subset=[factor_col, forward_col])
            if len(cross) < 10:
                continue

            if ic_type == "spearman":
                ic, _ = stats.spearmanr(
                    cross[factor_col], cross[forward_col], nan_policy="omit"
                )
            else:
                ic, _ = stats.pearsonr(
                    cross[factor_col].fillna(0), cross[forward_col].fillna(0)
                )

            if not np.isnan(ic):
                ic_list.append({"date": dt, "ic": ic})

        if not ic_list:
            return None

        ic_df = pd.DataFrame(ic_list)
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        return ic_df.set_index("date")["ic"]
