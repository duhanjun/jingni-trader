"""
增强版因子分析（借鉴 Qlib 的因子评估体系）

jingni-trader 现有 factor-engine 的 IC 分析存在两个问题:
  1. 性能: _calc_ic 逐日循环 data[data['date']==dt] 过滤，O(n²)
  2. 分析维度不足: 仅有 IC/ICIR，缺少因子换手率与衰减分析

Qlib 标准因子评估包含:
  - IC / Rank IC / ICIR（jingni-trader 已有）
  - 因子换手率（factor turnover = corr(factor_t, factor_{t-1})）
    —— 衡量因子值的稳定性，换手率越高交易成本越大
  - 因子衰减曲线（IC at multiple lags）
    —— 判断因子信号的失效速度，指导持仓周期

本模块用 groupby 向量化 IC 计算，并新增换手率与衰减分析。
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy import stats


class EnhancedFactorAnalysis:
    """增强版因子分析器"""

    def ic_analysis_vectorized(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_names: List[str],
        ic_type: str = "spearman",
    ) -> Dict[str, Any]:
        """
        向量化 IC 计算（替代逐日循环）

        核心优化: 用 groupby('date').apply 一次性计算所有日期的截面 IC，
        避免 O(n_dates * n_rows) 的重复过滤。
        """
        if factor_df.empty or forward_returns.empty:
            return {}

        # 仅合并存在的前瞻收益列，避免 KeyError
        fwd_cols = [c for c in ['ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']
                    if c in forward_returns.columns]
        if not fwd_cols:
            return {}

        merged = factor_df.merge(
            forward_returns[['code', 'date'] + fwd_cols],
            on=['code', 'date'], how='inner'
        )

        results = {}
        for forward_col in fwd_cols:
            if forward_col not in merged.columns:
                continue

            ic_results = []
            for factor in factor_names:
                if factor not in merged.columns:
                    continue

                ic_series = self._calc_ic_vectorized(merged, factor, forward_col, ic_type)
                if ic_series is None or ic_series.empty:
                    continue

                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                ic_ir = ic_mean / ic_std if ic_std > 0 else 0
                ic_positive_ratio = (ic_series > 0).mean()
                n = len(ic_series)
                ic_t = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0

                ic_results.append({
                    "factor": factor,
                    "forward_period": forward_col,
                    "ic_mean": round(float(ic_mean), 6),
                    "ic_std": round(float(ic_std), 6),
                    "ic_ir": round(float(ic_ir), 4),
                    "ic_positive_ratio": round(float(ic_positive_ratio), 4),
                    "ic_t_stat": round(float(ic_t), 4),
                })
            results[forward_col] = ic_results

        return results

    def _calc_ic_vectorized(
        self, data: pd.DataFrame, factor_col: str, forward_col: str, ic_type: str
    ) -> Optional[pd.Series]:
        """
        向量化截面 IC 计算

        优化点: groupby('date') 后对每组计算相关系数，
        相比逐日 data[data['date']==dt] 过滤减少重复扫描。
        """
        sub = data[['date', factor_col, forward_col]].dropna()
        if len(sub) < 10:
            return None

        def _cross_ic(group):
            if len(group) < 10:
                return np.nan
            if ic_type == "spearman":
                r, _ = stats.spearmanr(group[factor_col], group[forward_col])
            else:
                r, _ = stats.pearsonr(group[factor_col], group[forward_col])
            return r

        ic_series = sub.groupby('date').apply(_cross_ic).dropna()
        ic_series.name = 'ic'
        return ic_series

    def factor_turnover(
        self, factor_df: pd.DataFrame, factor_names: List[str], lags: List[int] = None
    ) -> Dict[str, Any]:
        """
        因子换手率分析（借鉴 Qlib）

        因子换手率 = corr(factor_t, factor_{t-lag})
        衡量因子值的时间稳定性:
          - 换手率低（自相关高）: 因子稳定，交易成本低，适合低频
          - 换手率高（自相关低）: 因子变化快，交易成本高

        Qlib 中通常报告 lag=1 的自相关，本模块支持多 lag。
        """
        if factor_df.empty or not factor_names:
            return {}

        if lags is None:
            lags = [1, 5, 20]

        result = {}
        for factor in factor_names:
            if factor not in factor_df.columns:
                continue

            # 按 code 分组，计算各 lag 的自相关
            pivot = factor_df.pivot_table(index='date', columns='code', values=factor)
            factor_turnover_stats = {}

            for lag in lags:
                shifted = pivot.shift(lag)
                # 截面自相关: 每日 corr(factor_t, factor_{t-lag}) 的均值
                common = pivot.notna() & shifted.notna()
                daily_corr = []
                for dt in pivot.index:
                    if common.loc[dt].sum() < 10:
                        continue
                    a = pivot.loc[dt][common.loc[dt]]
                    b = shifted.loc[dt][common.loc[dt]]
                    if len(a) < 10:
                        continue
                    c, _ = stats.spearmanr(a, b)
                    if not np.isnan(c):
                        daily_corr.append(c)

                if daily_corr:
                    factor_turnover_stats[f"autocorr_lag{lag}"] = round(float(np.mean(daily_corr)), 4)
                    # 换手率 = 1 - 自相关（自相关越高换手越低）
                    factor_turnover_stats[f"turnover_lag{lag}"] = round(float(1 - np.mean(daily_corr)), 4)

            result[factor] = factor_turnover_stats

        return result

    def factor_decay(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_names: List[str],
        horizons: List[int] = None,
    ) -> Dict[str, Any]:
        """
        因子衰减曲线（借鉴 Qlib 因子有效期分析）

        计算因子在不同持有期（1,3,5,10,20,40日）的 IC，
        绘制 IC 衰减曲线，判断因子信号的有效持续时间。

        应用价值:
          - IC 在 N 日后降至 0: 因子有效期约 N 日，持仓周期应 <= N
          - IC 快速衰减: 适合短线；IC 缓慢衰减: 适合中长线
        """
        if factor_df.empty or not factor_names:
            return {}

        if horizons is None:
            horizons = [1, 3, 5, 10, 20, 40]

        # 计算多期前瞻收益
        fwd = price_df[['code', 'date', 'close']].copy()
        fwd = fwd.sort_values(['code', 'date'])
        for h in horizons:
            fwd[f'fwd_ret_{h}'] = fwd.groupby('code')['close'].transform(
                lambda x: x.shift(-h) / x - 1
            )

        merged = factor_df.merge(fwd, on=['code', 'date'], how='inner')

        result = {}
        for factor in factor_names:
            if factor not in merged.columns:
                continue

            decay_curve = {}
            for h in horizons:
                col = f'fwd_ret_{h}'
                if col not in merged.columns:
                    continue
                ic_series = self._calc_ic_vectorized(merged, factor, col, "spearman")
                if ic_series is not None and not ic_series.empty:
                    decay_curve[f"ic_horizon_{h}d"] = round(float(ic_series.mean()), 6)
                else:
                    decay_curve[f"ic_horizon_{h}d"] = None

            # 找到 IC 首次降至阈值以下的期限（因子有效期）
            ic_values = [decay_curve.get(f"ic_horizon_{h}d") for h in horizons]
            valid = [(h, v) for h, v in zip(horizons, ic_values) if v is not None]
            half_life = None
            if valid:
                first_ic = abs(valid[0][1])
                if first_ic > 0:
                    for h, v in valid:
                        if abs(v) < first_ic * 0.5:
                            half_life = h
                            break
            decay_curve["estimated_half_life_days"] = half_life
            result[factor] = decay_curve

        return result
