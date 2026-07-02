"""
因子分层回测分析 (Factor Layered Analysis)

借鉴来源:
- alphalens (Quantopian): 因子分层收益、IC 衰减、换手率分析
- Microsoft Qlib: Signal Analysis 模块，TopK / Dropout 策略评估
- JoinQuant/BigQuant: 因子检测标准流程

针对 jingni-trader factor-engine 的优化点:
原版 ic_analysis 只计算 IC 均值/IR/正比例，缺少:
1. 分层收益 (按因子值分5层，看各层收益单调性)
2. 多空组合收益 (Top层 - Bottom层)
3. IC 衰减分析 (1d/5d/10d/20d 的 IC 变化)
4. 因子单调性检验 (Spearman 秩相关 + 分层收益相关性)
5. 分层换手率 (评估因子的稳定性)
6. 因子覆盖率与有效样本数

这些指标是判断因子是否"可交易"的关键，alphalens 把它们作为标准因子评估流程。
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("factor-layered-analysis")


class FactorLayeredAnalysis:
    """因子分层回测分析"""

    def __init__(
        self,
        n_quantiles: int = 5,
        forward_periods: List[int] = (1, 5, 10, 20),
        benchmark_return: Optional[pd.Series] = None,
    ):
        """
        参数:
            n_quantiles: 分层数 (默认5层)
            forward_periods: 远期收益计算周期 (交易日)
            benchmark_return: 基准日收益率 (可选，用于计算超额收益分层)
        """
        self.n_quantiles = n_quantiles
        self.forward_periods = list(forward_periods)
        self.benchmark_return = benchmark_return

    def analyze(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_col: str = "alpha_score",
        code_col: str = "code",
        date_col: str = "date",
    ) -> Dict[str, Any]:
        """
        执行完整的因子分层分析

        参数:
            factor_df: 含 code, date, factor_col 的 DataFrame
            price_df: 含 code, date, close 的 DataFrame (用于计算远期收益)
            factor_col: 因子列名
            code_col, date_col: 股票代码与日期列名

        返回:
            {
                "ic_analysis": {...},          # IC 分析 (含衰减)
                "quantile_returns": DataFrame, # 各层各周期收益
                "long_short_returns": DataFrame, # 多空组合日收益序列
                "monotonicity": {...},         # 单调性检验
                "turnover": {...},             # 分层换手率
                "coverage": {...},             # 覆盖率统计
                "summary": {...},              # 综合评分
            }
        """
        # 1) 数据准备
        merged = self._prepare_data(factor_df, price_df, factor_col, code_col, date_col)
        if merged.empty:
            return self._empty_result()

        # 2) 远期收益计算
        for period in self.forward_periods:
            merged[f"forward_ret_{period}d"] = (
                merged.groupby(code_col)["close"].transform(
                    lambda x: x.shift(-period) / x - 1
                )
            )

        # 3) 分层 (每日按因子值分 n_quantiles 层)
        merged["quantile"] = (
            merged.groupby(date_col)[factor_col]
            .transform(self._assign_quantile)
        )

        # 4) IC 分析 (含衰减)
        ic_result = self._ic_analysis(merged, factor_col, code_col, date_col)

        # 5) 分层收益
        quantile_returns = self._quantile_returns(merged, date_col)

        # 6) 多空组合日收益
        long_short_returns = self._long_short_returns(merged, date_col)

        # 7) 单调性检验
        monotonicity = self._monotonicity_test(quantile_returns)

        # 8) 分层换手率
        turnover = self._turnover_analysis(merged, code_col, date_col)

        # 9) 覆盖率
        coverage = self._coverage_analysis(merged, factor_col, code_col, date_col)

        # 10) 综合评分
        summary = self._build_summary(ic_result, monotonicity, turnover, coverage)

        return {
            "ic_analysis": ic_result,
            "quantile_returns": quantile_returns,
            "long_short_returns": long_short_returns,
            "monotonicity": monotonicity,
            "turnover": turnover,
            "coverage": coverage,
            "summary": summary,
            "n_quantiles": self.n_quantiles,
            "forward_periods": self.forward_periods,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_col: str,
        code_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """合并因子与价格数据"""
        factor_df = factor_df.copy()
        price_df = price_df.copy()
        factor_df[date_col] = pd.to_datetime(factor_df[date_col])
        price_df[date_col] = pd.to_datetime(price_df[date_col])

        if factor_col not in factor_df.columns:
            raise ValueError(f"因子列 {factor_col} 不在 factor_df 中")

        merged = pd.merge(
            factor_df[[code_col, date_col, factor_col]],
            price_df[[code_col, date_col, "close"]],
            on=[code_col, date_col],
            how="inner",
        )
        merged = merged.dropna(subset=[factor_col, "close"])
        merged = merged.sort_values([date_col, code_col]).reset_index(drop=True)
        return merged

    def _assign_quantile(self, series: pd.Series) -> pd.Series:
        """横截面分层 (1~n_quantiles, NaN 保持 NaN)"""
        try:
            return pd.qcut(
                series, self.n_quantiles, labels=False, duplicates="drop"
            ) + 1
        except Exception:
            # 数据点过少或全部相同
            return pd.Series(1, index=series.index)

    def _ic_analysis(
        self,
        merged: pd.DataFrame,
        factor_col: str,
        code_col: str,
        date_col: str,
    ) -> Dict[str, Any]:
        """IC 分析 (含衰减)"""
        result = {}
        for period in self.forward_periods:
            ret_col = f"forward_ret_{period}d"
            if ret_col not in merged.columns:
                continue

            # 逐日计算 Spearman IC
            ic_series = []
            for dt, group in merged.groupby(date_col):
                valid = group.dropna(subset=[factor_col, ret_col])
                if len(valid) < 10:
                    continue
                ic, _ = stats.spearmanr(valid[factor_col], valid[ret_col])
                if not np.isnan(ic):
                    ic_series.append({"date": dt, "ic": ic})

            if not ic_series:
                result[f"{period}d"] = {}
                continue

            ic_df = pd.DataFrame(ic_series).set_index("date")["ic"]
            ic_mean = ic_df.mean()
            ic_std = ic_df.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_t = ic_mean / (ic_std / np.sqrt(len(ic_df))) if ic_std > 0 else 0
            positive_ratio = (ic_df > 0).mean()

            result[f"{period}d"] = {
                "ic_mean": round(float(ic_mean), 6),
                "ic_std": round(float(ic_std), 6),
                "ic_ir": round(float(ic_ir), 4),
                "ic_t_stat": round(float(ic_t), 4),
                "ic_positive_ratio": round(float(positive_ratio), 4),
                "n_obs": int(len(ic_df)),
            }

        # IC 衰减: 1d -> 5d -> 10d -> 20d 的 IC 均值变化
        ic_means = [result.get(f"{p}d", {}).get("ic_mean", 0) for p in self.forward_periods]
        decay = {
            "periods": self.forward_periods,
            "ic_means": ic_means,
            "decay_ratio": (
                round(ic_means[-1] / ic_means[0], 4)
                if ic_means and ic_means[0] != 0 else None
            ),
        }
        result["decay"] = decay
        return result

    def _quantile_returns(self, merged: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """计算各层各周期平均收益"""
        records = []
        for period in self.forward_periods:
            ret_col = f"forward_ret_{period}d"
            if ret_col not in merged.columns:
                continue
            for q in range(1, self.n_quantiles + 1):
                q_data = merged[merged["quantile"] == q].dropna(subset=[ret_col])
                if q_data.empty:
                    continue
                # 平均收益 (等权)
                avg_ret = q_data.groupby(date_col)[ret_col].mean()
                records.append({
                    "period": f"{period}d",
                    "quantile": q,
                    "mean_return": round(float(avg_ret.mean()), 6),
                    "std_return": round(float(avg_ret.std()), 6),
                    "sharpe_like": round(
                        float(avg_ret.mean() / avg_ret.std() * np.sqrt(252))
                        if avg_ret.std() > 0 else 0, 4
                    ),
                    "n_days": int(len(avg_ret)),
                })
        return pd.DataFrame(records)

    def _long_short_returns(
        self, merged: pd.DataFrame, date_col: str
    ) -> pd.DataFrame:
        """多空组合日收益 (Top层 - Bottom层)"""
        records = []
        for period in self.forward_periods:
            ret_col = f"forward_ret_{period}d"
            if ret_col not in merged.columns:
                continue

            top = merged[merged["quantile"] == self.n_quantiles].dropna(subset=[ret_col])
            bottom = merged[merged["quantile"] == 1].dropna(subset=[ret_col])

            if top.empty or bottom.empty:
                continue

            top_ret = top.groupby(date_col)[ret_col].mean()
            bottom_ret = bottom.groupby(date_col)[ret_col].mean()
            ls = top_ret - bottom_ret

            records.append({
                "period": f"{period}d",
                "long_short_mean": round(float(ls.mean()), 6),
                "long_short_std": round(float(ls.std()), 6),
                "long_short_sharpe": round(
                    float(ls.mean() / ls.std() * np.sqrt(252))
                    if ls.std() > 0 else 0, 4
                ),
                "win_rate": round(float((ls > 0).mean()), 4),
                "n_days": int(len(ls)),
            })
        return pd.DataFrame(records)

    def _monotonicity_test(self, quantile_returns: pd.DataFrame) -> Dict[str, Any]:
        """因子单调性检验"""
        if quantile_returns.empty:
            return {}

        result = {}
        for period in quantile_returns["period"].unique():
            sub = quantile_returns[quantile_returns["period"] == period].sort_values("quantile")
            if len(sub) < 2:
                continue
            # Spearman 秩相关: quantile vs mean_return
            rho, p_value = stats.spearmanr(sub["quantile"], sub["mean_return"])
            # 线性相关
            lin_corr, _ = stats.pearsonr(sub["quantile"], sub["mean_return"])
            # 单调性得分: |rho|, 越接近1越单调
            result[period] = {
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(p_value), 4),
                "linear_corr": round(float(lin_corr), 4),
                "is_monotonic": abs(rho) > 0.7 and p_value < 0.05,
                "direction": "positive" if rho > 0 else "negative",
            }
        return result

    def _turnover_analysis(
        self, merged: pd.DataFrame, code_col: str, date_col: str
    ) -> Dict[str, Any]:
        """分层换手率分析"""
        result = {}
        for q in range(1, self.n_quantiles + 1):
            q_data = merged[merged["quantile"] == q].copy()
            if q_data.empty:
                continue
            # 按日期排序，计算每只股票在层内的停留时间
            q_data = q_data.sort_values([code_col, date_col])
            # 简化: 计算每日层内成员变化率
            daily_members = q_data.groupby(date_col)[code_col].apply(set)
            turnovers = []
            dates = sorted(daily_members.index)
            for i in range(1, len(dates)):
                prev = daily_members.iloc[i - 1]
                curr = daily_members.iloc[i]
                if len(curr) == 0:
                    continue
                # 换手率 = 新增 + 退出 / 2 / 当前层大小
                new = len(curr - prev)
                out = len(prev - curr)
                turnover = (new + out) / 2 / max(len(curr), 1)
                turnovers.append(turnover)
            result[f"Q{q}"] = {
                "avg_turnover": round(float(np.mean(turnovers)), 4) if turnovers else 0.0,
                "max_turnover": round(float(np.max(turnovers)), 4) if turnovers else 0.0,
                "n_days": len(turnovers),
            }
        return result

    def _coverage_analysis(
        self,
        merged: pd.DataFrame,
        factor_col: str,
        code_col: str,
        date_col: str,
    ) -> Dict[str, Any]:
        """覆盖率统计"""
        total = len(merged)
        valid = merged[factor_col].notna().sum()
        daily_counts = merged.groupby(date_col)[code_col].nunique()
        return {
            "total_rows": int(total),
            "valid_factor_rows": int(valid),
            "coverage_ratio": round(float(valid / total), 4) if total > 0 else 0.0,
            "avg_daily_stocks": round(float(daily_counts.mean()), 2) if len(daily_counts) > 0 else 0,
            "min_daily_stocks": int(daily_counts.min()) if len(daily_counts) > 0 else 0,
            "max_daily_stocks": int(daily_counts.max()) if len(daily_counts) > 0 else 0,
        }

    def _build_summary(
        self,
        ic_result: Dict,
        monotonicity: Dict,
        turnover: Dict,
        coverage: Dict,
    ) -> Dict[str, Any]:
        """综合评分 (0-100)"""
        # IC IR 评分 (40分): |IC_IR| 越高越好, 0.5 算优秀
        ic_5d = ic_result.get("5d", {})
        ic_ir = abs(ic_5d.get("ic_ir", 0))
        ic_score = min(ic_ir / 0.5, 1.0) * 40

        # 单调性评分 (30分): |rho| 越接近1越好
        mono_5d = monotonicity.get("5d", {})
        rho = abs(mono_5d.get("spearman_rho", 0))
        mono_score = rho * 30

        # 换手率评分 (15分): 换手率越低越稳定, 0.3 算可接受
        avg_turnover = np.mean([v.get("avg_turnover", 0) for v in turnover.values()]) if turnover else 1.0
        turn_score = max(0, (0.5 - avg_turnover) / 0.5) * 15

        # 覆盖率评分 (15分)
        cov_score = coverage.get("coverage_ratio", 0) * 15

        total_score = round(ic_score + mono_score + turn_score + cov_score, 2)

        # 评级
        if total_score >= 75:
            grade = "A"
        elif total_score >= 60:
            grade = "B"
        elif total_score >= 45:
            grade = "C"
        elif total_score >= 30:
            grade = "D"
        else:
            grade = "F"

        return {
            "total_score": total_score,
            "grade": grade,
            "scores": {
                "ic_ir": round(ic_score, 2),
                "monotonicity": round(mono_score, 2),
                "turnover": round(turn_score, 2),
                "coverage": round(cov_score, 2),
            },
            "interpretation": {
                "A": "优秀因子，可考虑纳入实盘",
                "B": "良好因子，可纳入模拟盘观察",
                "C": "一般因子，需进一步优化",
                "D": "较弱因子，谨慎使用",
                "F": "无效因子，建议放弃",
            }.get(grade, ""),
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "ic_analysis": {},
            "quantile_returns": pd.DataFrame(),
            "long_short_returns": pd.DataFrame(),
            "monotonicity": {},
            "turnover": {},
            "coverage": {},
            "summary": {},
        }
