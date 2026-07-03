"""
综合量化指标库（借鉴来源: VectorBT Portfolio.stats / quantstats / pyfolio）

设计动机
========
当前 jingni-trader 的 `_calculate_metrics` 只输出 5 个指标:
    skills/backtest-engine/scripts/adapters/rqalpha_adapter.py:152-163
        annual_return / sharpe_ratio / max_drawdown / total_return / volatility
无法满足现代量化研究的归因 / 风险 / 稳定性分析需求。

借鉴 VectorBT 的 `Portfolio.stats()` 一键输出 60+ 指标:
    https://vectorbt.dev/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.stats

借鉴 quantstats 的 70+ 报告指标:
    https://github.com/ranaroussi/quantstats

本模块提供
==========
1. return_metrics: 收益类 (年化 / 累计 / 复合 / 月度收益等)
2. risk_metrics:   风险类 (波动率 / 回撤 / VaR / CVaR / Ulcer / 偏度 / 峰度等)
3. ratio_metrics:  比率类 (Sharpe / Sortino / Calmar / Sterling / Information / Omega)
4. factor_metrics: 因子分析 (IC / ICIR / Rank IC / 换手率 / 胜率)
5. calc_all_stats: 一键综合输出 (类似 VectorBT.stats)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _to_returns(equity: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(equity, pd.DataFrame):
        return equity.pct_change().iloc[1:]
    s = pd.Series(equity).astype(float)
    return s.pct_change().iloc[1:]


def _drawdown(equity: pd.Series) -> pd.DataFrame:
    """计算回撤序列"""
    s = pd.Series(equity).astype(float)
    running_max = s.cummax()
    dd = s / running_max - 1.0
    return pd.DataFrame({"equity": s, "drawdown": dd})


# ---------------------------------------------------------------------------
# 收益类
# ---------------------------------------------------------------------------
def return_metrics(equity: pd.Series, period: str = "daily") -> Dict[str, float]:
    s = pd.Series(equity).astype(float).dropna()
    if len(s) < 2:
        return {}
    rets = _to_returns(s)
    total_return = float(s.iloc[-1] / s.iloc[0] - 1)
    n_periods = len(rets)
    if period == "daily":
        ann = TRADING_DAYS
    elif period == "weekly":
        ann = 52
    elif period == "monthly":
        ann = 12
    else:
        ann = TRADING_DAYS
    cagr = float(s.iloc[-1] / s.iloc[0]) ** (ann / max(n_periods, 1)) - 1
    monthly = s.resample("ME").last().pct_change().dropna()
    return {
        "total_return": total_return,
        "annual_return": cagr,
        "monthly_return": float(monthly.mean()) if len(monthly) else 0.0,
        "daily_return_mean": float(rets.mean()),
        "daily_return_median": float(rets.median()),
        "best_day": float(rets.max()),
        "worst_day": float(rets.min()),
        "positive_days": int((rets > 0).sum()),
        "negative_days": int((rets < 0).sum()),
        "total_periods": n_periods,
    }


# ---------------------------------------------------------------------------
# 风险类
# ---------------------------------------------------------------------------
def risk_metrics(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    confidence: float = 0.95,
) -> Dict[str, float]:
    s = pd.Series(equity).astype(float).dropna()
    if len(s) < 2:
        return {}
    rets = _to_returns(s)
    ann_factor = TRADING_DAYS
    vol_d = float(rets.std(ddof=1))
    vol_ann = vol_d * math.sqrt(ann_factor)

    dd = _drawdown(s)["drawdown"]
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    dd_duration = _max_dd_duration(dd)
    ulcer = float(np.sqrt((dd ** 2).mean()))

    var_hist = float(np.percentile(rets, (1 - confidence) * 100))
    cvar_hist = float(rets[rets <= var_hist].mean()) if (rets <= var_hist).any() else var_hist

    # parametric VaR (normal assumption)
    var_param = float(stats.norm.ppf(1 - confidence, loc=rets.mean(), scale=vol_d))

    out = {
        "volatility_daily": vol_d,
        "volatility_annual": vol_ann,
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "max_drawdown_duration_days": dd_duration,
        "ulcer_index": ulcer,
        "var_historical": var_hist,
        "cvar_historical": cvar_hist,
        "var_parametric": var_param,
        "skewness": float(stats.skew(rets)) if len(rets) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(rets, fisher=True)) if len(rets) > 3 else 0.0,
        "tail_ratio": float(abs(rets.quantile(0.95)) / abs(rets.quantile(0.05)))
            if abs(rets.quantile(0.05)) > 1e-12 else 0.0,
    }

    if benchmark is not None:
        b = pd.Series(benchmark).astype(float).dropna().reindex(s.index).ffill()
        b_rets = b.pct_change().iloc[1:]
        common = rets.index.intersection(b_rets.index)
        ar, br = rets.loc[common], b_rets.loc[common]
        out["beta"] = float(np.cov(ar, br, ddof=1)[0, 1] / np.var(br, ddof=1)) \
            if np.var(br) > 1e-12 else 0.0
        out["alpha_annual"] = float(
            (ar.mean() - br.mean()) * ann_factor + risk_free
        ) if len(ar) > 0 else 0.0
        out["tracking_error_annual"] = float((ar - br).std() * math.sqrt(ann_factor)) \
            if len(ar) > 1 else 0.0
    return out


def _max_dd_duration(dd: pd.Series) -> int:
    """最长回撤持续天数 (从高点到恢复)"""
    underwater = dd < 0
    if not underwater.any():
        return 0
    runs = (underwater != underwater.shift()).cumsum()
    if not underwater.any():
        return 0
    durations = underwater.groupby(runs).sum()
    if len(durations) == 0:
        return 0
    return int(durations.max())


# ---------------------------------------------------------------------------
# 比率类
# ---------------------------------------------------------------------------
def ratio_metrics(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free: float = 0.03,
) -> Dict[str, float]:
    s = pd.Series(equity).astype(float).dropna()
    if len(s) < 2:
        return {}
    rets = _to_returns(s)
    ann_factor = TRADING_DAYS
    rf_d = (1 + risk_free) ** (1 / ann_factor) - 1
    excess = rets - rf_d
    vol_d = excess.std(ddof=1)
    if vol_d < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(excess.mean() / vol_d * math.sqrt(ann_factor))

    downside = rets[rets < rf_d]
    if len(downside) > 0 and downside.std() > 1e-12:
        sortino = float((rets.mean() - rf_d) / downside.std() * math.sqrt(ann_factor))
    else:
        sortino = 0.0

    dd = _drawdown(s)["drawdown"]
    max_dd = abs(float(dd.min()))
    cagr = float(s.iloc[-1] / s.iloc[0]) ** (ann_factor / max(len(rets), 1)) - 1
    calmar = cagr / max_dd if max_dd > 1e-12 else 0.0

    # Omega
    threshold = 0.0
    gains = (rets - threshold).clip(lower=0).sum()
    losses = (threshold - rets).clip(lower=0).sum()
    omega = float(gains / losses) if losses > 1e-12 else float("inf")

    out = {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "omega_ratio": omega,
        "sterling_ratio": cagr / max(abs(avg_dd := dd[dd < 0].mean()), 1e-12) if (dd < 0).any() else 0.0,
        "annual_return": cagr,
    }
    if benchmark is not None:
        b = pd.Series(benchmark).astype(float).dropna().reindex(s.index).ffill()
        b_rets = b.pct_change().iloc[1:]
        common = rets.index.intersection(b_rets.index)
        ar, br = rets.loc[common], b_rets.loc[common]
        if len(ar) > 1 and (ar - br).std() > 1e-12:
            out["information_ratio"] = float(
                (ar.mean() - br.mean()) / (ar - br).std() * math.sqrt(ann_factor)
            )
        else:
            out["information_ratio"] = 0.0
    return out


# ---------------------------------------------------------------------------
# 因子 / 信号类
# ---------------------------------------------------------------------------
def factor_metrics(
    factor_panel: pd.DataFrame,
    forward_returns_panel: pd.DataFrame,
    quantiles: int = 5,
) -> Dict[str, float]:
    """
    计算因子 IC / Rank IC / 多空收益 / 换手率 / 胜率

    参数
    -----
    factor_panel : 包含 date / code / factor 列
    forward_returns_panel : 包含 date / code / forward_return 列
    quantiles : 分层数
    """
    if "factor" not in factor_panel.columns:
        raise ValueError("factor_panel 必须包含 'factor' 列")
    if "forward_return" not in forward_returns_panel.columns:
        raise ValueError("forward_returns_panel 必须包含 'forward_return' 列")
    if factor_panel.empty or forward_returns_panel.empty:
        return {}
    df = factor_panel.merge(
        forward_returns_panel[["date", "code", "forward_return"]],
        on=["date", "code"],
        how="inner",
    ).dropna(subset=["factor", "forward_return"])
    if df.empty:
        return {}

    ics, rank_ics = [], []
    for date, g in df.groupby("date"):
        if len(g) < 5:
            continue
        ic = g["factor"].corr(g["forward_return"])
        ric = g["factor"].corr(g["forward_return"], method="spearman")
        if pd.notna(ic):
            ics.append(ic)
        if pd.notna(ric):
            rank_ics.append(ric)

    if not ics:
        return {}

    ic_arr = np.array(ics)
    ric_arr = np.array(rank_ics)
    out = {
        "ic_mean": float(ic_arr.mean()),
        "ic_std": float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else 0.0,
        "ic_ir": float(ic_arr.mean() / ic_arr.std(ddof=1))
            if len(ic_arr) > 1 and ic_arr.std(ddof=1) > 1e-12 else 0.0,
        "ic_positive_ratio": float((ic_arr > 0).mean()),
        "rank_ic_mean": float(ric_arr.mean()),
        "rank_ic_std": float(ric_arr.std(ddof=1)) if len(ric_arr) > 1 else 0.0,
        "rank_ic_ir": float(ric_arr.mean() / ric_arr.std(ddof=1))
            if len(ric_arr) > 1 and ric_arr.std(ddof=1) > 1e-12 else 0.0,
        "n_dates": int(len(ic_arr)),
    }

    # 多空收益: top quintile - bottom quintile
    df["quantile"] = df.groupby("date")["factor"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=quantiles, labels=False, duplicates="drop")
    )
    long_short = []
    for date, g in df.groupby("date"):
        if g["quantile"].nunique() < 2:
            continue
        max_q = g["quantile"].max()
        min_q = g["quantile"].min()
        long_short.append(g[g["quantile"] == max_q]["forward_return"].mean()
                          - g[g["quantile"] == min_q]["forward_return"].mean())
    if long_short:
        out["long_short_return_mean"] = float(np.mean(long_short))
        out["long_short_return_std"] = float(np.std(long_short, ddof=1)) if len(long_short) > 1 else 0.0
        out["long_short_win_rate"] = float(np.mean([r > 0 for r in long_short]))
    return out


# ---------------------------------------------------------------------------
# 一键综合输出
# ---------------------------------------------------------------------------
def calc_all_stats(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    period: str = "daily",
) -> Dict[str, float]:
    """
    一键输出 30+ 综合指标, 参考 VectorBT Portfolio.stats 接口
    """
    out = {}
    out.update(return_metrics(equity, period=period))
    out.update(risk_metrics(equity, benchmark=benchmark, risk_free=risk_free))
    out.update(ratio_metrics(equity, benchmark=benchmark, risk_free=risk_free))
    return out
