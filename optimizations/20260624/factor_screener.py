"""
三重约束因子筛选器（借鉴 QuantaAlpha 因子池维护机制）

解决 jingni-trader 现有 factor-engine 的筛选缺陷：
  - 现有 correlation_analysis() 仅按因子间相关性去冗余，无预测力门槛、无容量约束
  - 去冗余策略简单（按名称长度取舍），非贪心 RankIC 降序

借鉴来源：
  - QuantaAlpha 论文 arXiv:2602.07085 Section 3.4 因子池维护
  - 三重门槛：Rank IC 显著 + 低冗余(相关<0.7) + 容量达标
  - 贪心 RankIC 降序入库策略
  - 东方证券研报优化方向：A 股需加行业/市值中性化 IC

设计要点：
  1. 三重约束串联过滤：Rank IC 门槛 → 容量门槛 → 低冗余贪心去重
  2. 贪心 RankIC 降序：高 IC 因子优先入库，后续因子与池中所有因子相关性 < 阈值才加入
  3. A 股增强：支持行业/市值中性化 IC（剥离系统性风格暴露，过滤伪 Alpha）
  4. 因子 lineage 追踪（借鉴 QuantaAlpha StrategyTrajectory.parent_ids）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("opt-factor-screener")


@dataclass
class FactorMetrics:
    """单因子评估指标"""
    name: str
    # Rank IC 指标
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    rank_ic_ir: float = 0.0           # IC 信息比率 = IC均值/IC标准差
    rank_ic_t_stat: float = 0.0       # IC t 统计量
    rank_ic_positive_ratio: float = 0.0
    # 中性化 IC（A 股增强：剥离行业/市值暴露后的纯 Alpha）
    neutral_ic_mean: float = 0.0
    neutral_ic_ir: float = 0.0
    # 容量指标
    turnover_rate: float = 0.0        # 因子多空组合换手率
    capacity_score: float = 0.0       # 容量评分（0-1，越高越可承载大资金）
    avg_liquidity: float = 0.0        # 平均流动性（成分股日均成交额）
    # 是否通过各重约束
    pass_ic: bool = False
    pass_capacity: bool = False
    pass_redundancy: bool = False
    # 入库状态
    selected: bool = False
    reject_reason: str = ""


@dataclass
class FactorLineage:
    """
    因子谱系追踪（借鉴 QuantaAlpha StrategyTrajectory.parent_ids）

    记录因子来源、进化谱系，可审计。
    """
    name: str
    expression: str = ""
    source: str = "manual"            # manual / expression_engine / llm_mining
    parent_factors: List[str] = field(default_factory=list)
    created_at: str = ""
    metrics: Optional[FactorMetrics] = None


class ThreeConstraintScreener:
    """
    三重约束因子筛选器

    筛选流程（借鉴 QuantaAlpha 因子池维护）：
      1. Rank IC 门槛：|IC均值| >= min_ic 且 IC_IR >= min_ic_ir 且 t_stat 显著
      2. 容量门槛：换手率 <= max_turnover 且 流动性 >= min_liquidity
      3. 低冗余贪心去重：按 RankIC 降序，与池中因子相关性 < max_correlation 才入库

    A 股增强（东方证券研报优化方向）：
      - 支持行业/市值中性化 IC，剥离系统性风格暴露
      - 中性化 IC 不达标的因子即使原始 IC 高也降级处理
    """

    def __init__(
        self,
        min_rank_ic: float = 0.02,         # 最小 |Rank IC| 均值
        min_ic_ir: float = 0.3,            # 最小 IC 信息比率
        min_ic_t_stat: float = 2.0,        # 最小 IC t 统计量（显著性）
        max_correlation: float = 0.7,      # 因子间最大相关性
        max_turnover: float = 0.5,         # 因子多空组合最大换手率
        min_liquidity: float = 1e7,        # 最小日均成交额（元）
        use_neutral_ic: bool = True,       # 是否使用中性化 IC 作为主排序依据
        neutral_ic_weight: float = 0.5,    # 中性化 IC 在排序中的权重
    ):
        self.min_rank_ic = min_rank_ic
        self.min_ic_ir = min_ic_ir
        self.min_ic_t_stat = min_ic_t_stat
        self.max_correlation = max_correlation
        self.max_turnover = max_turnover
        self.min_liquidity = min_liquidity
        self.use_neutral_ic = use_neutral_ic
        self.neutral_ic_weight = neutral_ic_weight

    # ── Rank IC 计算 ────────────────────────────────────────
    def calc_rank_ic_series(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_return_col: str,
    ) -> pd.Series:
        """
        计算 Rank IC 时间序列（Spearman 秩相关）

        借鉴 jingni-trader 现有 _calc_ic，但返回完整 Series 供后续统计。
        """
        ic_list = []
        for dt, cross in factor_df.groupby("date"):
            valid = cross.dropna(subset=[factor_col, forward_return_col])
            if len(valid) < 10:
                continue
            ic, _ = stats.spearmanr(valid[factor_col], valid[forward_return_col])
            if not np.isnan(ic):
                ic_list.append({"date": dt, "ic": ic})
        if not ic_list:
            return pd.Series(dtype=float)
        return pd.DataFrame(ic_list).set_index("date")["ic"]

    def calc_neutral_ic_series(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_return_col: str,
        industry_col: str = "industry",
        mcap_col: str = "lncap",
    ) -> pd.Series:
        """
        计算行业/市值中性化 IC（A 股增强）

        流程：
          1. 每期对因子值做行业+市值回归，取残差（中性化因子）
          2. 对前瞻收益做同样中性化
          3. 计算两个残差的 Spearman 秩相关

        这样剥离了系统性风格暴露，过滤伪 Alpha 信号。
        """
        from sklearn.linear_model import LinearRegression

        ic_list = []
        for dt, cross in factor_df.groupby("date"):
            valid = cross.dropna(subset=[factor_col, forward_return_col]).copy()
            if len(valid) < 30:
                continue

            # 构造风格因子矩阵
            X_cols = []
            if mcap_col in valid.columns:
                X_cols.append(mcap_col)
            if industry_col in valid.columns:
                dummies = pd.get_dummies(valid[industry_col], prefix="ind")
                for c in dummies.columns:
                    valid[c] = dummies[c].values
                    X_cols.append(c)

            if not X_cols:
                # 无风格因子，回退到普通 Rank IC
                ic, _ = stats.spearmanr(valid[factor_col], valid[forward_return_col])
            else:
                X = valid[X_cols].fillna(0).values
                # 中性化因子值
                try:
                    reg_f = LinearRegression().fit(X, valid[factor_col].fillna(0).values)
                    factor_resid = valid[factor_col].fillna(0).values - reg_f.predict(X)
                    # 中性化收益
                    reg_r = LinearRegression().fit(X, valid[forward_return_col].fillna(0).values)
                    return_resid = valid[forward_return_col].fillna(0).values - reg_r.predict(X)
                    ic, _ = stats.spearmanr(factor_resid, return_resid)
                except Exception:
                    ic, _ = stats.spearmanr(valid[factor_col], valid[forward_return_col])

            if not np.isnan(ic):
                ic_list.append({"date": dt, "ic": ic})

        if not ic_list:
            return pd.Series(dtype=float)
        return pd.DataFrame(ic_list).set_index("date")["ic"]

    # ── 容量评估 ────────────────────────────────────────────
    def calc_capacity(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        amount_col: str = "amount",
        top_pct: float = 0.2,
    ) -> Tuple[float, float]:
        """
        评估因子容量

        返回: (换手率, 平均流动性)
          - 换手率：多空组合每日调仓比例（越高容量越小）
          - 平均流动性：多头组合成分股日均成交额
        """
        daily_turnover = []
        daily_liquidity = []

        for dt, cross in factor_df.groupby("date"):
            valid = cross.dropna(subset=[factor_col])
            if len(valid) < 20:
                continue
            # 多头 = 因子值最高的 top_pct
            threshold = valid[factor_col].quantile(1 - top_pct)
            long_pool = valid[valid[factor_col] >= threshold].sort_values(factor_col, ascending=False)

            if amount_col in long_pool.columns:
                daily_liquidity.append(long_pool[amount_col].mean())

            # 简化换手率：用多头池成员变化率近似
            daily_turnover.append(1.0)  # 占位，实际需对比前后两期持仓

        # 真实换手率：计算多期多头池重叠度
        long_pools = []
        for dt, cross in factor_df.groupby("date"):
            valid = cross.dropna(subset=[factor_col])
            if len(valid) < 20:
                long_pools.append(set())
                continue
            threshold = valid[factor_col].quantile(1 - top_pct)
            long_pools.append(set(valid[valid[factor_col] >= threshold]["code"]))

        turnovers = []
        for i in range(1, len(long_pools)):
            prev, cur = long_pools[i - 1], long_pools[i]
            if not prev and not cur:
                continue
            union = prev | cur
            if union:
                turnovers.append(len(prev ^ cur) / len(union))

        avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
        avg_liquidity = float(np.mean(daily_liquidity)) if daily_liquidity else 0.0
        return avg_turnover, avg_liquidity

    # ── 单因子评估 ──────────────────────────────────────────
    def evaluate_factor(
        self,
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_return_col: str,
        industry_col: str = "industry",
        mcap_col: str = "lncap",
        amount_col: str = "amount",
    ) -> FactorMetrics:
        """完整评估单个因子"""
        metrics = FactorMetrics(name=factor_col)

        # 1) Rank IC
        ic_series = self.calc_rank_ic_series(factor_df, factor_col, forward_return_col)
        if ic_series.empty:
            metrics.reject_reason = "IC 序列为空"
            return metrics

        metrics.rank_ic_mean = float(ic_series.mean())
        metrics.rank_ic_std = float(ic_series.std())
        metrics.rank_ic_ir = float(
            metrics.rank_ic_mean / metrics.rank_ic_std if metrics.rank_ic_std > 0 else 0
        )
        n = len(ic_series)
        metrics.rank_ic_t_stat = float(
            metrics.rank_ic_mean / (metrics.rank_ic_std / np.sqrt(n))
            if metrics.rank_ic_std > 0 else 0
        )
        metrics.rank_ic_positive_ratio = float((ic_series > 0).mean())

        # 2) 中性化 IC（A 股增强）
        if self.use_neutral_ic and industry_col in factor_df.columns:
            neutral_series = self.calc_neutral_ic_series(
                factor_df, factor_col, forward_return_col, industry_col, mcap_col
            )
            if not neutral_series.empty:
                metrics.neutral_ic_mean = float(neutral_series.mean())
                neutral_std = float(neutral_series.std())
                metrics.neutral_ic_ir = float(
                    metrics.neutral_ic_mean / neutral_std if neutral_std > 0 else 0
                )

        # 3) 容量
        turnover, liquidity = self.calc_capacity(factor_df, factor_col, amount_col)
        metrics.turnover_rate = turnover
        metrics.avg_liquidity = liquidity
        # 容量评分：换手率越低、流动性越高，容量越大
        turnover_score = max(0, 1 - turnover / max(self.max_turnover, 1e-9))
        liquidity_score = min(1, liquidity / max(self.min_liquidity * 10, 1e-9))
        metrics.capacity_score = 0.5 * turnover_score + 0.5 * liquidity_score

        # 4) 约束判定
        metrics.pass_ic = (
            abs(metrics.rank_ic_mean) >= self.min_rank_ic
            and abs(metrics.rank_ic_ir) >= self.min_ic_ir
            and abs(metrics.rank_ic_t_stat) >= self.min_ic_t_stat
        )
        metrics.pass_capacity = (
            metrics.turnover_rate <= self.max_turnover
            and metrics.avg_liquidity >= self.min_liquidity
        )
        # pass_redundancy 在贪心去重阶段判定

        if not metrics.pass_ic:
            metrics.reject_reason = (
                f"IC 不达标: |IC|={abs(metrics.rank_ic_mean):.4f}<{self.min_rank_ic}, "
                f"|ICIR|={abs(metrics.rank_ic_ir):.4f}<{self.min_ic_ir}, "
                f"|t|={abs(metrics.rank_ic_t_stat):.4f}<{self.min_ic_t_stat}"
            )
        elif not metrics.pass_capacity:
            metrics.reject_reason = (
                f"容量不达标: 换手率={metrics.turnover_rate:.4f}>{self.max_turnover}, "
                f"流动性={metrics.avg_liquidity:.0f}<{self.min_liquidity}"
            )

        return metrics

    # ── 排序键（A 股增强：融合原始 IC 与中性化 IC）──────────
    def _sort_key(self, metrics: FactorMetrics) -> float:
        """贪心入库的排序键：IC 越高越优先"""
        if self.use_neutral_ic and self.neutral_ic_weight < 1.0:
            # 融合：原始 IC IR + 中性化 IC IR
            raw = abs(metrics.rank_ic_ir)
            neutral = abs(metrics.neutral_ic_ir)
            w = self.neutral_ic_weight
            return (1 - w) * raw + w * neutral
        return abs(metrics.rank_ic_ir)

    # ── 三重约束筛选主流程 ──────────────────────────────────
    def screen(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
        forward_return_col: str,
        industry_col: str = "industry",
        mcap_col: str = "lncap",
        amount_col: str = "amount",
    ) -> Tuple[List[FactorMetrics], List[FactorLineage]]:
        """
        三重约束筛选主流程

        流程（借鉴 QuantaAlpha 因子池维护）：
          1. 逐因子评估 Rank IC + 容量
          2. 通过前两重的因子按排序键降序排列
          3. 贪心去重：依次尝试入库，与池中所有因子相关性 < max_correlation 才加入

        返回:
            all_metrics: 全部因子的评估指标
            selected_lineages: 入库因子的谱系记录
        """
        logger.info(f"开始三重约束筛选，共 {len(factor_cols)} 个候选因子")

        # 第一、二重：IC + 容量
        all_metrics: List[FactorMetrics] = []
        for col in factor_cols:
            if col not in factor_df.columns:
                logger.warning(f"因子列 {col} 不存在，跳过")
                continue
            m = self.evaluate_factor(
                factor_df, col, forward_return_col,
                industry_col, mcap_col, amount_col,
            )
            all_metrics.append(m)

        passed = [m for m in all_metrics if m.pass_ic and m.pass_capacity]
        logger.info(
            f"第一、二重约束（IC + 容量）：{len(passed)}/{len(all_metrics)} 通过"
        )

        # 按排序键降序
        passed.sort(key=self._sort_key, reverse=True)

        # 第三重：低冗余贪心去重
        selected: List[FactorMetrics] = []
        for m in passed:
            col = m.name
            redundant = False
            for sel in selected:
                sel_col = sel.name
                # 计算两因子截面均值的相关性
                corr = self._factor_correlation(factor_df, col, sel_col)
                if abs(corr) > self.max_correlation:
                    m.reject_reason = (
                        f"冗余: 与已入库因子 {sel_col} 相关性 {corr:.4f} > {self.max_correlation}"
                    )
                    redundant = True
                    break
            if not redundant:
                m.pass_redundancy = True
                m.selected = True
                selected.append(m)

        logger.info(
            f"第三重约束（低冗余）：{len(selected)}/{len(passed)} 入库，"
            f"剔除 {len(passed) - len(selected)} 个冗余因子"
        )

        # 构建谱系
        lineages = [
            FactorLineage(
                name=m.name,
                source="expression_engine",
                created_at=pd.Timestamp.now().isoformat(),
                metrics=m,
            )
            for m in selected
        ]

        return all_metrics, lineages

    def _factor_correlation(
        self, factor_df: pd.DataFrame, col_a: str, col_b: str
    ) -> float:
        """
        计算两因子的相关性（池化相关性）

        用全部 (code, date) 样本的因子值直接计算相关系数，
        而非截面均值相关性（截面均值会洗掉个股层面的信号差异）。
        """
        if col_a not in factor_df.columns or col_b not in factor_df.columns:
            return 0.0
        valid = factor_df[[col_a, col_b]].dropna()
        if len(valid) < 30:
            return 0.0
        return float(valid[col_a].corr(valid[col_b]))


def compare_with_legacy(
    factor_df: pd.DataFrame,
    factor_cols: List[str],
    forward_return_col: str,
    max_correlation: float = 0.7,
) -> Dict[str, Any]:
    """
    与 jingni-trader 现有 correlation_analysis 对比

    现有方法：仅按因子间相关性去冗余，无 IC 门槛、无容量约束、
             去冗余策略按名称长度取舍（非贪心 RankIC）。

    返回对比报告。
    """
    # 现有方法复刻
    factor_means = factor_df.groupby("date")[factor_cols].mean()
    corr_matrix = factor_means.corr()

    legacy_removed = set()
    for i in range(len(factor_cols)):
        for j in range(i + 1, len(factor_cols)):
            fi, fj = factor_cols[i], factor_cols[j]
            if fi in legacy_removed or fj in legacy_removed:
                continue
            if abs(corr_matrix.loc[fi, fj]) > max_correlation:
                # 现有策略：按名称长度取舍
                if len(fj) < len(fi):
                    legacy_removed.add(fi)
                else:
                    legacy_removed.add(fj)
    legacy_selected = [f for f in factor_cols if f not in legacy_removed]

    # 三重约束方法
    screener = ThreeConstraintScreener(max_correlation=max_correlation)
    all_metrics, lineages = screener.screen(
        factor_df, factor_cols, forward_return_col
    )
    new_selected = [m.name for m in all_metrics if m.selected]

    return {
        "legacy_selected": legacy_selected,
        "legacy_removed": list(legacy_removed),
        "new_selected": new_selected,
        "new_rejected": [
            {"name": m.name, "reason": m.reject_reason}
            for m in all_metrics if not m.selected
        ],
        "legacy_count": len(legacy_selected),
        "new_count": len(new_selected),
        "improvement": {
            "added_ic_filter": len(factor_cols) - len([m for m in all_metrics if m.pass_ic]),
            "added_capacity_filter": len([m for m in all_metrics if m.pass_ic and not m.pass_capacity]),
        },
    }
