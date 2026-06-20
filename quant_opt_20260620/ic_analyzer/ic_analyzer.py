"""
增强版 IC 分析器

借鉴自微软 Qlib (https://github.com/microsoft/qlib) 的因子评估体系
（contrib.evaluate 中的 risk_analysis / IC decay / 分层回测）。

针对 jingni-trader 现有 ic_analysis 的改进点：
1. 增加 RankIC（Spearman 秩相关系数）作为标配
2. 引入分层回测（quantile returns）— Qlib/WorldQuant 101 Alphas 的标配
3. 计算 IC 衰减曲线（1d/5d/10d/20d forward）
4. 单调性（monotonicity）— 评估因子分层收益是否单调
5. 因子收益率时间序列与 t 检验
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """数值稳定的 Spearman 相关系数（去 nan）"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return 0.0
    x_v = x[mask]
    y_v = y[mask]
    if np.std(x_v) < 1e-12 or np.std(y_v) < 1e-12:
        return 0.0
    rho, _ = stats.spearmanr(x_v, y_v)
    return float(rho) if not np.isnan(rho) else 0.0


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return 0.0
    x_v = x[mask]
    y_v = y[mask]
    if np.std(x_v) < 1e-12 or np.std(y_v) < 1e-12:
        return 0.0
    r, _ = stats.pearsonr(x_v, y_v)
    return float(r) if not np.isnan(r) else 0.0


def calc_ic_series(factor: pd.Series, fwd_ret: pd.Series,
                   method: str = "spearman") -> float:
    """单日 IC（横截面相关）"""
    if method == "spearman":
        return _safe_spearman(factor.values, fwd_ret.values)
    return _safe_pearson(factor.values, fwd_ret.values)


def calc_ic_decay(factor_df: pd.DataFrame,
                  ret_df: pd.DataFrame,
                  forward_periods: List[int] = (1, 5, 10, 20),
                  factor_col: str = "factor",
                  ret_col_template: str = "fwd_ret_{}d",
                  method: str = "spearman") -> Dict[int, Dict[str, float]]:
    """
    因子 IC 衰减曲线
    对每个 forward period 算 IC 的 mean / std / IR / t-stat
    """
    decay: Dict[int, Dict[str, float]] = {}
    dates = sorted(set(factor_df["date"]).intersection(ret_df["date"]))
    for fp in forward_periods:
        ret_col = ret_col_template.format(fp)
        if ret_col not in ret_df.columns:
            continue
        merged = factor_df[["date", "code", factor_col]].merge(
            ret_df[["date", "code", ret_col]], on=["date", "code"], how="inner"
        )
        merged = merged.dropna(subset=[factor_col, ret_col])
        ic_series = []
        for d, g in merged.groupby("date"):
            if len(g) < 10:
                continue
            ic = calc_ic_series(g[factor_col], g[ret_col], method=method)
            ic_series.append(ic)
        if not ic_series:
            continue
        arr = np.array(ic_series)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        ir = mean / std if std > 1e-12 else 0.0
        t_stat = mean / (std / np.sqrt(arr.size)) if std > 1e-12 else 0.0
        decay[fp] = {
            "ic_mean": mean,
            "ic_std": std,
            "ic_ir": ir,
            "ic_t_stat": t_stat,
            "ic_pos_ratio": float(np.mean(arr > 0)),
            "n_periods": int(arr.size),
        }
    return decay


def calc_layered_returns(factor_df: pd.DataFrame,
                         ret_df: pd.DataFrame,
                         n_quantiles: int = 5,
                         factor_col: str = "factor",
                         ret_col: str = "fwd_ret_5d") -> pd.DataFrame:
    """
    分层回测 (Quantile Returns) — Qlib / 101 Alphas 标准评估方法
    每天按因子值分 n_quantiles 层，计算每层平均 forward 收益。
    """
    merged = factor_df[["date", "code", factor_col]].merge(
        ret_df[["date", "code", ret_col]], on=["date", "code"], how="inner"
    ).dropna(subset=[factor_col, ret_col])
    if merged.empty:
        return pd.DataFrame()

    # 使用 transform 替代 groupby.apply，避免 pandas 3.0 中 group_keys=False 丢失列的问题
    def _qassign(s: pd.Series) -> pd.Series:
        try:
            return pd.qcut(s, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series([0] * len(s), index=s.index, dtype=float)

    merged = merged.copy()
    merged["quantile"] = merged.groupby("date")[factor_col].transform(_qassign)
    merged = merged.dropna(subset=["quantile"])
    if merged.empty:
        return pd.DataFrame()
    merged["quantile"] = merged["quantile"].astype(int)
    layered = merged.groupby(["date", "quantile"])[ret_col].mean().unstack("quantile")
    return layered


def calc_long_short(layered: pd.DataFrame) -> Dict[str, float]:
    """
    多空组合指标：long = 最高分位组, short = 最低分位组
    """
    if layered.empty or layered.shape[1] < 2:
        return {}
    qs = sorted(layered.columns)
    long_q, short_q = qs[-1], qs[0]
    ls = (layered[long_q] - layered[short_q]).dropna()
    if ls.empty:
        return {}
    arr = ls.values
    return {
        "long_short_mean": float(arr.mean()),
        "long_short_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "long_short_sharpe": float(arr.mean() / arr.std(ddof=1) * np.sqrt(252)) if arr.size > 1 and arr.std(ddof=1) > 0 else 0.0,
        "long_short_cum": float(np.prod(1 + arr) - 1),
        "long_short_t_stat": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(arr.size))) if arr.size > 1 and arr.std(ddof=1) > 0 else 0.0,
    }


def calc_monotonicity(layered: pd.DataFrame) -> float:
    """
    单调性：每层平均收益与分位序号的 Spearman 秩相关
    越接近 1 表示分层收益单调性越好（典型好因子应 > 0.5）
    """
    if layered.empty:
        return 0.0
    mean_by_q = layered.mean(axis=0)
    qs = np.array(mean_by_q.index, dtype=float)
    rs = mean_by_q.values
    if np.std(rs) < 1e-12:
        return 0.0
    rho, _ = stats.spearmanr(qs, rs)
    return float(rho) if not np.isnan(rho) else 0.0


def full_factor_evaluation(factor_df: pd.DataFrame,
                           ret_df: pd.DataFrame,
                           factor_col: str = "factor",
                           forward_periods: List[int] = (1, 5, 10, 20),
                           n_quantiles: int = 5,
                           ic_method: str = "spearman") -> Dict:
    """
    完整因子评估：IC decay + 分层回测 + 多空组合 + 单调性
    """
    decay = calc_ic_decay(factor_df, ret_df, forward_periods, factor_col,
                          method=ic_method)
    layered = calc_layered_returns(factor_df, ret_df, n_quantiles, factor_col,
                                   ret_col=f"fwd_ret_{forward_periods[1] if len(forward_periods) > 1 else forward_periods[0]}d")
    ls = calc_long_short(layered)
    mono = calc_monotonicity(layered)
    return {
        "ic_decay": decay,
        "layered_returns": layered,
        "long_short": ls,
        "monotonicity": mono,
    }
