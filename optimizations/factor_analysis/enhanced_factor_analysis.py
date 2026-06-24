"""
增强因子分析模块 — 借鉴 alphalens-reloaded 的设计思想
========================================================

借鉴来源: alphalens-reloaded (https://github.com/stefan-jansen/alphalens-reloaded)
原项目由 Quantopian 开发，stefan-jansen 维护，是因子分析的业界标准。

jingni-trader 现有 factor-engine 的 IC 分析只计算:
  - ic_mean, ic_std, ic_ir, ic_positive_ratio, ic_t_stat
  且使用 `for dt in dates:` 逐日循环计算 IC (慢)。

本模块补充 alphalens 的核心能力:
  1. IC 衰减分析 (IC Decay): 计算因子对 [1,5,10,20] 多个前视期的 IC，
     观察因子预测能力随持有期衰减的速度 — 这是判断因子适用周期
     的关键指标，现有引擎完全缺失。
  2. 因子换手率 (Factor Turnover): 因子排名的自相关性，衡量因子信号
     的稳定性。高换手率因子交易成本高，实盘可行性差。
  3. 分层收益 (Quantile Returns): 将股票按因子值分 N 层，计算每层
     的平均前视收益，直观展示因子的单调性。多空收益 (Q1-QN) 是
     因子纯 Alpha 的标准度量。
  4. Rank IC (Spearman) vs Normal IC (Pearson): 同时给出两种 IC，
     Rank IC 对极端值更鲁棒，是业界首选。

所有计算均向量化 (pandas groupby + transform)，无逐日 Python 循环。

本模块为验证代码，独立于 main 分支，不修改现有 factor-engine。
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class EnhancedFactorAnalyzer:
    """
    增强因子分析器 (alphalens 风格)

    输入约定 (与 jingni-trader factor-engine 输出对齐):
      - factor_df: 长表，含 code, date, <factor_name> 列
      - prices:    宽表收盘价 (date × code) 或长表含 code,date,close
    """

    def __init__(self, quantiles: int = 5, min_obs: int = 10):
        self.quantiles = quantiles
        self.min_obs = min_obs

    # ------------------------------------------------------------------
    # 1. IC 衰减分析 (核心增强点)
    # ------------------------------------------------------------------
    def ic_decay(
        self,
        factor_df: pd.DataFrame,
        prices: pd.DataFrame,
        factor_name: str,
        forward_periods: List[int] = [1, 5, 10, 20],
        method: str = "spearman",
    ) -> pd.DataFrame:
        """
        计算因子在多个前视期上的 IC 衰减

        参数:
            factor_df: 含 code, date, factor_name
            prices: 宽表 (date × code) 收盘价
            factor_name: 因子列名
            forward_periods: 前视期列表 (交易日)
            method: 'spearman' (Rank IC) 或 'pearson' (Normal IC)

        返回:
            DataFrame, index=forward_period, columns:
              ic_mean, ic_std, ic_ir, ic_t_stat, ic_positive_ratio, n_obs
        """
        # 宽表前视收益
        fwd_returns = {}
        for p in forward_periods:
            fwd_returns[p] = prices.shift(-p) / prices - 1.0

        # 长表因子 → 宽表
        factor_wide = factor_df.pivot_table(
            index="date", columns="code", values=factor_name
        ).reindex(index=prices.index, columns=prices.columns)

        results = []
        for p in forward_periods:
            fr = fwd_returns[p]
            ic_series = self._cross_sectional_ic_series(factor_wide, fr, method)
            if ic_series is None or len(ic_series) == 0:
                results.append({
                    "forward_period": p, "ic_mean": np.nan, "ic_std": np.nan,
                    "ic_ir": np.nan, "ic_t_stat": np.nan,
                    "ic_positive_ratio": np.nan, "n_obs": 0,
                })
                continue
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
            n = len(ic_series)
            ic_t = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0.0
            results.append({
                "forward_period": p,
                "ic_mean": float(ic_mean),
                "ic_std": float(ic_std),
                "ic_ir": float(ic_ir),
                "ic_t_stat": float(ic_t),
                "ic_positive_ratio": float((ic_series > 0).mean()),
                "n_obs": int(n),
            })
        return pd.DataFrame(results).set_index("forward_period")

    def _cross_sectional_ic_series(
        self,
        factor_wide: pd.DataFrame,
        fwd_return_wide: pd.DataFrame,
        method: str,
    ) -> Optional[pd.Series]:
        """
        向量化计算逐日 IC 序列 (纯矩阵运算, 无逐日 Python 循环)

        原理:
          - Spearman IC = Pearson IC of rank(factor) vs rank(fwd_ret)
          - Pearson IC_t = cov(f_t, r_t) / (std(f_t) * std(r_t))
          - 上述运算均可按行 (axis=1) 向量化, 无需 groupby.apply 循环
        """
        # 对齐
        common_idx = factor_wide.index.intersection(fwd_return_wide.index)
        common_cols = factor_wide.columns.intersection(fwd_return_wide.columns)
        f = factor_wide.loc[common_idx, common_cols].astype(float)
        r = fwd_return_wide.loc[common_idx, common_cols].astype(float)

        if f.empty:
            return None

        # Spearman: 按行 (每个日期横截面) 排名 — 向量化
        if method == "spearman":
            f = f.rank(axis=1)
            r = r.rank(axis=1)

        # 有效观测数 (逐行, f 与 r 均非 NaN)
        both_valid = f.notna() & r.notna()
        n_per_date = both_valid.sum(axis=1)

        # 逐行均值/标准差/协方差 (pandas axis=1 走 C 内核, 无 Python 循环)
        # 用有效观测计算 (skipna=True)
        f_mean = f.mean(axis=1, skipna=True)
        r_mean = r.mean(axis=1, skipna=True)
        f_centered = f.sub(f_mean, axis=0)
        r_centered = r.sub(r_mean, axis=0)

        # 协方差: sum((f-fm)*(r-rm)) / (n-1), 仅计入两者均有效的位置
        product = f_centered * r_centered
        product = product.where(both_valid)
        cov = product.sum(axis=1, skipna=True) / (n_per_date - 1).replace(0, np.nan)

        std_f = f.std(axis=1, skipna=True)
        std_r = r.std(axis=1, skipna=True)

        ic_series = cov / (std_f * std_r)
        ic_series = ic_series.replace([np.inf, -np.inf], np.nan)

        # 过滤: 有效观测数不足
        ic_series = ic_series.where(n_per_date >= self.min_obs).dropna()
        ic_series.name = "ic"
        return ic_series

    # ------------------------------------------------------------------
    # 2. 因子换手率 (自相关性)
    # ------------------------------------------------------------------
    def factor_turnover(
        self,
        factor_df: pd.DataFrame,
        factor_name: str,
        lag: int = 1,
    ) -> Dict[str, float]:
        """
        计算因子排名的自相关性作为换手率代理指标

        换手率 = 1 - corr(rank_t, rank_{t-lag})
        高换手率 → 因子排名变动频繁 → 实盘交易成本高

        参数:
            factor_df: 含 code, date, factor_name
            factor_name: 因子列名
            lag: 滞后期 (交易日)

        返回:
            dict: mean_turnover, std_turnover, autocorrelation
        """
        factor_wide = factor_df.pivot_table(
            index="date", columns="code", values=factor_name
        )
        # 横截面排名 (pct)
        ranks = factor_wide.rank(axis=1, pct=True)
        # 滞后自相关
        autocorr = ranks.corrwith(ranks.shift(lag), axis=1).dropna()
        turnover = (1.0 - autocorr).clip(lower=0.0)

        return {
            "mean_turnover": float(turnover.mean()) if len(turnover) > 0 else np.nan,
            "std_turnover": float(turnover.std()) if len(turnover) > 0 else np.nan,
            "autocorrelation": float(autocorr.mean()) if len(autocorr) > 0 else np.nan,
            "lag": lag,
            "n_obs": int(len(turnover)),
        }

    # ------------------------------------------------------------------
    # 3. 分层收益 (Quantile Returns)
    # ------------------------------------------------------------------
    def quantile_returns(
        self,
        factor_df: pd.DataFrame,
        prices: pd.DataFrame,
        factor_name: str,
        forward_period: int = 5,
    ) -> pd.DataFrame:
        """
        分层收益分析

        将每个日期的股票按因子值分为 self.quantiles 层，
        计算每层的平均前视收益。

        返回:
            DataFrame, index=quantile (1=最低, N=最高), columns:
              mean_return, std_return, n_obs
        """
        fwd_ret = (prices.shift(-forward_period) / prices - 1.0)
        factor_wide = factor_df.pivot_table(
            index="date", columns="code", values=factor_name
        ).reindex(index=prices.index, columns=prices.columns)

        # 分层 (qcut 按行)
        quantile_labels = self._cross_sectional_quantile(factor_wide)

        # 转长表
        q_long = quantile_labels.stack().rename("quantile")
        r_long = fwd_ret.stack().rename("fwd_ret")
        merged = pd.concat([q_long, r_long], axis=1).dropna()
        merged = merged.reset_index().rename(columns={"level_0": "date", "level_1": "code"})

        if len(merged) == 0:
            return pd.DataFrame()

        grouped = merged.groupby("quantile")["fwd_ret"]
        result = pd.DataFrame({
            "mean_return": grouped.mean(),
            "std_return": grouped.std(),
            "n_obs": grouped.count(),
        })
        result.index.name = "quantile"

        # 多空收益 (最高层 - 最低层)
        if len(result) >= 2:
            long_short = result.loc[result.index.max(), "mean_return"] - \
                         result.loc[result.index.min(), "mean_return"]
            result.attrs["long_short_return"] = float(long_short)

        return result

    def _cross_sectional_quantile(self, factor_wide: pd.DataFrame) -> pd.DataFrame:
        """逐行 qcut 分层 (向量化用 apply，比逐行 Python 循环快)"""
        def _qcut_row(row):
            valid = row.dropna()
            if len(valid) < self.quantiles:
                return pd.Series(np.nan, index=row.index)
            try:
                return pd.qcut(valid, self.quantiles, labels=False, duplicates="drop") + 1
            except ValueError:
                return pd.Series(np.nan, index=row.index)
        return factor_wide.apply(_qcut_row, axis=1)

    # ------------------------------------------------------------------
    # 4. 完整分析报告
    # ------------------------------------------------------------------
    def full_analysis(
        self,
        factor_df: pd.DataFrame,
        prices: pd.DataFrame,
        factor_name: str,
        forward_periods: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, Any]:
        """一次性产出完整因子分析报告"""
        return {
            "factor_name": factor_name,
            "ic_decay": self.ic_decay(factor_df, prices, factor_name, forward_periods),
            "ic_decay_rank": self.ic_decay(
                factor_df, prices, factor_name, forward_periods, method="spearman"
            ),
            "turnover": self.factor_turnover(factor_df, factor_name, lag=1),
            "quantile_returns_5d": self.quantile_returns(
                factor_df, prices, factor_name, forward_period=5
            ),
        }


if __name__ == "__main__":
    # 简易自测
    np.random.seed(0)
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    codes = [f"S{i:03d}.SZ" for i in range(50)]
    px = pd.DataFrame(
        10 * np.cumprod(1 + np.random.normal(0, 0.01, (len(dates), len(codes))), axis=0),
        index=dates, columns=codes,
    )
    # 构造一个有预测力的因子: 与未来收益正相关 + 噪声
    fwd5 = px.shift(-5) / px - 1
    factor_values = fwd5.shift(5) * 0.5 + np.random.normal(0, 0.02, px.shape)
    factor_long = factor_values.stack().rename("alpha_test").reset_index()
    factor_long.columns = ["date", "code", "alpha_test"]

    an = EnhancedFactorAnalyzer(quantiles=5)
    report = an.full_analysis(factor_long, px, "alpha_test")
    print("=== IC Decay (Pearson) ===")
    print(report["ic_decay"])
    print("\n=== IC Decay (Spearman/Rank) ===")
    print(report["ic_decay_rank"])
    print("\n=== Turnover ===")
    print(report["turnover"])
    print("\n=== Quantile Returns (5d) ===")
    print(report["quantile_returns_5d"])
    print("Long-Short return:", report["quantile_returns_5d"].attrs.get("long_short_return"))
