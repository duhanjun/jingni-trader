"""
因子 IC/IR 分析模块（验证版）
============================

借鉴来源：
- 微软 Qlib 的 `qlib.contrib.data.handler.Alpha158`：
  - 每周期对每个因子计算与下期收益的 Spearman Rank IC
  - 输出 IC mean / IC std / ICIR
  - 通过 ICIR > 阈值来筛选有效因子
- WORLDQUANT 101 Alpha 论文中的 IC 评估方法
- 国内量化行业普遍采用的「IC 衰退曲线 + 分层回测」评估范式

设计目标：
为 `BaseFactorCalculator` 增加标准化 IC/ICIR 分析能力。
对每个因子：
1. 计算与下期收益（默认 1 日）的 Pearson & Spearman Rank 相关系数序列。
2. 聚合统计：mean, std, IR = mean/std, t-stat, 胜率（IC>0 占比）。
3. 因子相关性矩阵：用于冗余因子剔除。

不直接修改 main 分支代码，仅作为验证参考实现。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FactorICReport:
    """单个因子的 IC 报告"""

    factor: str
    n_samples: int
    ic_mean: float
    ic_std: float
    icir: float
    ic_pos_ratio: float  # IC>0 占比（胜率）
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float
    rank_ic_pos_ratio: float
    ic_t_stat: float
    ic_p_value: float
    abs_ic_mean: float
    direction: int  # 1=正向因子，-1=反向因子


class FactorICAnalyzer:
    """因子 IC 分析器

    使用方法::

        analyzer = FactorICAnalyzer(forward=1)
        reports = analyzer.analyze(factor_df, price_df)
        matrix = analyzer.correlation_matrix(factor_df)
    """

    def __init__(self, forward: int = 1):
        """参数:
        forward: 预测期（计算未来 forward 日的收益）
        """
        self.forward = forward

    def _compute_forward_return(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """计算 forward 期收益：行 t = close[t+forward] / close[t] - 1"""
        price_df = price_df.sort_values(["code", "date"]).copy()
        # 统一转为字符串日期
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.strftime("%Y-%m-%d")
        price_df["fwd_ret"] = (
            price_df.groupby("code")["close"].shift(-self.forward) / price_df["close"] - 1
        )
        return price_df

    def analyze(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_cols: Optional[List[str]] = None,
    ) -> List[FactorICReport]:
        """对每个因子计算 IC 报告

        参数:
            factor_df: 必须包含 date/code 列 + 多个因子列
            price_df: 必须包含 date/code/close 列
            factor_cols: 指定要分析的因子列；None 表示自动检测所有数值列
        """
        if factor_cols is None:
            factor_cols = [
                c for c in factor_df.columns
                if c not in ("date", "code") and pd.api.types.is_numeric_dtype(factor_df[c])
            ]
        if not factor_cols:
            return []

        # 计算 forward return
        ret_df = self._compute_forward_return(price_df[["date", "code", "close"]])
        fdf = factor_df.copy()
        fdf["date"] = pd.to_datetime(fdf["date"]).dt.strftime("%Y-%m-%d")

        merged = fdf.merge(
            ret_df[["date", "code", "fwd_ret"]],
            on=["date", "code"],
            how="inner",
        )

        # 按日对截面做 Spearman & Pearson IC
        reports: List[FactorICReport] = []
        for col in factor_cols:
            sub = merged[[col, "fwd_ret"]].dropna()
            if len(sub) < 30:
                continue
            # 一次性按日期计算截面 IC
            ic_series = []
            rank_ic_series = []
            for _d, g in merged.groupby("date"):
                g = g[[col, "fwd_ret"]].dropna()
                if len(g) < 5:
                    continue
                try:
                    pearson_r, _ = stats.pearsonr(g[col], g["fwd_ret"])
                    spearman_r, _ = stats.spearmanr(g[col], g["fwd_ret"])
                except Exception:
                    continue
                if np.isfinite(pearson_r):
                    ic_series.append(pearson_r)
                if np.isfinite(spearman_r):
                    rank_ic_series.append(spearman_r)

            if not ic_series:
                continue
            ic_arr = np.asarray(ic_series, dtype=float)
            rk_arr = np.asarray(rank_ic_series, dtype=float)

            ic_mean = float(ic_arr.mean())
            ic_std = float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else 0.0
            rk_mean = float(rk_arr.mean())
            rk_std = float(rk_arr.std(ddof=1)) if len(rk_arr) > 1 else 0.0

            t_stat = float(ic_mean / (ic_std / np.sqrt(len(ic_arr)))) if ic_std > 0 else 0.0
            p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(len(ic_arr) - 1, 1))))

            reports.append(
                FactorICReport(
                    factor=col,
                    n_samples=int(len(ic_arr)),
                    ic_mean=ic_mean,
                    ic_std=ic_std,
                    icir=float(ic_mean / ic_std) if ic_std > 0 else 0.0,
                    ic_pos_ratio=float((ic_arr > 0).mean()),
                    rank_ic_mean=rk_mean,
                    rank_ic_std=rk_std,
                    rank_icir=float(rk_mean / rk_std) if rk_std > 0 else 0.0,
                    rank_ic_pos_ratio=float((rk_arr > 0).mean()),
                    ic_t_stat=t_stat,
                    ic_p_value=p_value,
                    abs_ic_mean=float(np.abs(ic_arr).mean()),
                    direction=1 if ic_mean >= 0 else -1,
                )
            )
        return reports

    def correlation_matrix(
        self, factor_df: pd.DataFrame, factor_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """计算因子截面 Spearman 相关性矩阵"""
        if factor_cols is None:
            factor_cols = [
                c for c in factor_df.columns
                if c not in ("date", "code") and pd.api.types.is_numeric_dtype(factor_df[c])
            ]
        if len(factor_cols) < 2:
            return pd.DataFrame()
        # 标准化：按日截面去极值 + Z-Score
        sub = factor_df[["date"] + factor_cols].copy()
        sub = sub.groupby("date").apply(self._cross_sectional_norm).reset_index(drop=True)
        return sub[factor_cols].corr(method="spearman")

    @staticmethod
    def _cross_sectional_norm(g: pd.DataFrame) -> pd.DataFrame:
        """截面标准化：去极值 + Z-Score"""
        for col in g.columns:
            if col == "date":
                continue
            x = g[col]
            if x.std() == 0 or pd.isna(x.std()):
                continue
            med = x.median()
            mad = (x - med).abs().median()
            if mad > 0:
                x_clip = (x - med) / (mad * 1.4826)  # 标准化到 N(0,1)
                x_clip = x_clip.clip(-3, 3)
                g[col] = x_clip
        return g

    def redundant_factor_filter(
        self,
        reports: List[FactorICReport],
        corr_matrix: pd.DataFrame,
        threshold: float = 0.7,
    ) -> List[str]:
        """冗余因子剔除：保留 |ICIR| 更高、与同组相关性 > threshold 的因子只留一个"""
        if not reports or corr_matrix.empty:
            return [r.factor for r in reports]

        # 按 |ICIR| 降序
        sorted_factors = sorted(reports, key=lambda r: abs(r.rank_icir), reverse=True)
        keep: List[str] = []
        for r in sorted_factors:
            if r.factor not in corr_matrix.columns:
                keep.append(r.factor)
                continue
            # 与已保留因子的最大相关性
            max_corr = 0.0
            for kept in keep:
                if kept not in corr_matrix.columns:
                    continue
                c = abs(corr_matrix.loc[r.factor, kept])
                if c > max_corr:
                    max_corr = c
            if max_corr < threshold:
                keep.append(r.factor)
        return keep


def reports_to_dataframe(reports: List[FactorICReport]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in reports])