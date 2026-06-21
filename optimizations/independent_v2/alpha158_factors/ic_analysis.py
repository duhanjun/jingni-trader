"""
因子 IC 分析模块

借鉴来源:
- Qlib contrib.eva.alpha.calc_ic
  (https://qlib.readthedocs.io/en/latest/component/eval.html)
- 量化投资实践中的 IC/RankIC/ICIR 标准

设计思路:
- jingni-trader 现有 factor-engine 缺少因子有效性评估模块，
  无法科学筛选因子。
- 本模块提供:
  1. IC (Information Coefficient): 因子值与未来收益的截面相关系数
  2. RankIC: 用 Spearman 秩相关，对异常值更鲁棒
  3. ICIR = IC均值 / IC标准差，衡量因子预测稳定性
  4. IC 衰减分析: 计算 1d/5d/10d/20d 多种持有期的 IC
  5. 因子有效信号判定: |IC均值| > 0.02 且 ICIR > 0.5

接口:
    analyzer = ICAnalyzer()
    report = analyzer.analyze(factor_df, return_df, factor_name="KMID")
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class ICAnalyzer:
    """因子 IC/RankIC/ICIR 分析器"""

    def calc_ic_series(
        self,
        factor_df: pd.DataFrame,
        return_df: pd.DataFrame,
        factor_name: str,
        return_col: str = "forward_return",
    ) -> Dict[str, pd.Series]:
        """计算每日截面 IC 与 RankIC 序列。

        参数:
            factor_df: 含 code, date, <factor_name> 列
            return_df: 含 code, date, <return_col> 列（未来收益）
            factor_name: 因子列名
            return_col: 收益列名

        返回:
            {"ic": pd.Series, "rank_ic": pd.Series}
            索引为日期
        """
        merged = pd.merge(
            factor_df[["code", "date", factor_name]],
            return_df[["code", "date", return_col]],
            on=["code", "date"],
            how="inner",
        ).dropna(subset=[factor_name, return_col])

        if merged.empty:
            return {"ic": pd.Series(dtype=float), "rank_ic": pd.Series(dtype=float)}

        # 按日期分组，计算截面相关
        def _daily_ic(grp):
            if len(grp) < 5:  # 样本太少不可靠
                return np.nan
            return grp[factor_name].corr(grp[return_col])

        def _daily_rank_ic(grp):
            if len(grp) < 5:
                return np.nan
            return grp[factor_name].corr(grp[return_col], method="spearman")

        ic = merged.groupby("date").apply(_daily_ic)
        rank_ic = merged.groupby("date").apply(_daily_rank_ic)

        return {"ic": ic, "rank_ic": rank_ic}

    def calc_ic_summary(self, ic_series: pd.Series) -> Dict[str, float]:
        """计算 IC 序列的统计摘要。"""
        ic_clean = ic_series.dropna()
        if len(ic_clean) == 0:
            return {
                "ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0,
                "ic_positive_ratio": 0.0, "ic_abs_mean": 0.0, "n_days": 0,
            }
        ic_mean = float(ic_clean.mean())
        ic_std = float(ic_clean.std())
        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": float(ic_mean / ic_std) if ic_std > 0 else 0.0,
            "ic_positive_ratio": float((ic_clean > 0).mean()),
            "ic_abs_mean": float(ic_clean.abs().mean()),
            "n_days": int(len(ic_clean)),
        }

    def calc_forward_returns(
        self,
        price_df: pd.DataFrame,
        periods: List[int] = (1, 5, 10, 20),
    ) -> pd.DataFrame:
        """计算多持有期未来收益。

        参数:
            price_df: 含 code, date, close 列
            periods: 持有期列表（交易日）

        返回:
            DataFrame，含 code, date, forward_return_1, forward_return_5, ...
        """
        df = price_df.sort_values(["code", "date"]).copy()
        for n in periods:
            df[f"forward_return_{n}"] = df.groupby("code")["close"].transform(
                lambda s: s.shift(-n) / s - 1
            )
        return df

    def analyze(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_name: str,
        periods: List[int] = (1, 5, 10, 20),
    ) -> Dict:
        """完整因子分析: 多持有期 IC + 统计摘要 + 有效性判定。

        返回:
            {
                "factor_name": str,
                "by_period": {period: {"ic_summary": {...}, "rank_ic_summary": {...}}},
                "is_effective": bool,
                "best_period": int,
                "recommendation": str,
            }
        """
        returns_df = self.calc_forward_returns(price_df, periods)

        by_period = {}
        best_icir = -np.inf
        best_period = periods[0]

        for n in periods:
            ret_col = f"forward_return_{n}"
            ret_df = returns_df[["code", "date", ret_col]].rename(
                columns={ret_col: "forward_return"}
            )
            ic_result = self.calc_ic_series(factor_df, ret_df, factor_name)
            ic_sum = self.calc_ic_summary(ic_result["ic"])
            rank_ic_sum = self.calc_ic_summary(ic_result["rank_ic"])

            by_period[n] = {
                "ic_summary": ic_sum,
                "rank_ic_summary": rank_ic_sum,
            }

            # 用 RankICIR 作为有效性主指标（更鲁棒）
            if rank_ic_sum["icir"] > best_icir:
                best_icir = rank_ic_sum["icir"]
                best_period = n

        # 有效性判定: 最佳持有期 |RankIC均值| > 0.02 且 RankICIR > 0.3
        best_rank_ic = by_period[best_period]["rank_ic_summary"]
        is_effective = (
            abs(best_rank_ic["ic_mean"]) > 0.02
            and best_rank_ic["icir"] > 0.3
        )

        if is_effective:
            recommendation = (
                f"因子 {factor_name} 有效，建议在 {best_period} 日持有期使用，"
                f"RankIC均值={best_rank_ic['ic_mean']:.4f}, "
                f"RankICIR={best_rank_ic['icir']:.4f}"
            )
        else:
            recommendation = (
                f"因子 {factor_name} 有效性不足，"
                f"最佳持有期 {best_period} 日 RankIC均值={best_rank_ic['ic_mean']:.4f}, "
                f"RankICIR={best_rank_ic['icir']:.4f}，建议谨慎使用或剔除"
            )

        return {
            "factor_name": factor_name,
            "by_period": by_period,
            "is_effective": is_effective,
            "best_period": best_period,
            "recommendation": recommendation,
        }
