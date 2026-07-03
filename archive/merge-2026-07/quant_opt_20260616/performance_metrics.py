"""
performance_metrics.py
======================

借鉴 vectorbt (https://github.com/polakowo/vectorbt) 性能指标体系
为 jingni-trader 提供更完善的回测绩效分析。

设计目标
--------
1. 单一函数入口 ``compute_metrics(equity, returns, benchmark_returns, ...)`` 
   返回完整的绩效字典。
2. 实现 jingni-trader 现有引擎缺失的 17 项关键指标:
   - 收益类: total_return, annual_return, cumulative_return
   - 风险类: volatility, downside_vol, max_drawdown, ulcer_index, max_drawdown_duration
   - 风险调整收益: sharpe, sortino, calmar, omega, tail_ratio
   - 相对基准: alpha, beta, information_ratio, tracking_error, capture_ratio
   - 稳健性: deflated_sharpe_ratio, stability, win_rate, profit_factor
3. 零外部依赖，纯 numpy/pandas 实现，便于集成与测试。
4. 严格数学定义 + 与教科书公式对齐，方便回测报告审计。

参考公式
--------
- Sharpe:    (μ_p - r_f) / σ_p, 年化
- Sortino:   (μ_p - r_f) / σ_downside, 年化
- Calmar:    年化收益 / |max_drawdown|
- Omega:     ∫_threshold^∞ (1-F(x)) dx / ∫_{-∞}^threshold F(x) dx
- Stability: R² of cumulative log returns vs time
- Deflated Sharpe: 校正了多重检验偏差的 Sharpe (Bailey & López de Prado, 2014)
- Information Ratio: (μ_p - μ_b) / tracking_error
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _prep_returns(returns: pd.Series) -> np.ndarray:
    """清洗并返回 numpy 数组，去除 nan/inf"""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    arr = returns.replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)
    return arr


def _annualize_factor(returns: pd.Series) -> float:
    """年化因子：根据日收益序列推断样本频率"""
    if len(returns) < 2:
        return TRADING_DAYS_PER_YEAR
    # 用时间差推断频率（保留扩展性）
    if isinstance(returns.index, pd.DatetimeIndex):
        delta = (returns.index[-1] - returns.index[0]).days / max(1, len(returns) - 1)
        if delta <= 0:
            return TRADING_DAYS_PER_YEAR
        return float(365.25 / delta)
    return TRADING_DAYS_PER_YEAR


# ============================================================================
# 各单项指标
# ============================================================================

def total_return(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annual_return(returns: pd.Series) -> float:
    arr = _prep_returns(returns)
    if len(arr) == 0:
        return 0.0
    cum = float(np.prod(1 + arr) - 1)
    periods = len(arr)
    af = _annualize_factor(returns)
    return float((1 + cum) ** (af / periods) - 1)


def volatility(returns: pd.Series) -> float:
    arr = _prep_returns(returns)
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=1) * np.sqrt(_annualize_factor(returns)))


def downside_volatility(returns: pd.Series, threshold: float = 0.0) -> float:
    arr = _prep_returns(returns)
    if len(arr) < 2:
        return 0.0
    downside = arr[arr < threshold]
    if len(downside) == 0:
        return 0.0
    af = _annualize_factor(returns)
    return float(np.sqrt(np.mean(np.minimum(arr - threshold, 0.0) ** 2)) * np.sqrt(af))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """最长回撤期长度 (交易日数)"""
    if equity.empty:
        return 0
    peak = equity.cummax()
    in_dd = (equity < peak).astype(int)
    if in_dd.sum() == 0:
        return 0
    # 用累计求和识别连续回撤段
    groups = (in_dd != in_dd.shift()).cumsum()
    dd_lengths = in_dd.groupby(groups).sum()
    return int(dd_lengths.max())


def ulcer_index(equity: pd.Series) -> float:
    """Ulcer Index: 回撤深度的均方根"""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd_pct = (equity / peak - 1.0) * 100.0
    return float(np.sqrt(np.mean(dd_pct.clip(upper=0) ** 2)))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.03) -> float:
    arr = _prep_returns(returns)
    if len(arr) < 2:
        return 0.0
    af = _annualize_factor(returns)
    excess = arr - risk_free / af
    sd = np.std(arr, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(af))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.03) -> float:
    arr = _prep_returns(returns)
    if len(arr) < 2:
        return 0.0
    af = _annualize_factor(returns)
    excess = arr - risk_free / af
    downside = arr[arr < 0]
    if len(downside) == 0:
        return 0.0
    dsd = np.sqrt(np.mean(downside ** 2))
    if dsd == 0:
        return 0.0
    return float(np.mean(excess) / dsd * np.sqrt(af))


def calmar_ratio(returns: pd.Series, equity: pd.Series) -> float:
    ar = annual_return(returns)
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(ar / mdd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio"""
    arr = _prep_returns(returns) - threshold
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def tail_ratio(returns: pd.Series) -> float:
    """右尾95% / 左尾95% 的绝对值之比"""
    arr = _prep_returns(returns)
    if len(arr) == 0:
        return 0.0
    right = np.percentile(arr, 95)
    left = np.percentile(arr, 5)
    if left == 0:
        return 0.0
    return float(abs(right) / abs(left))


def stability_of_returns(returns: pd.Series) -> float:
    """对累计对数收益对时间做线性回归，取 R²"""
    arr = _prep_returns(returns)
    if len(arr) < 2:
        return 0.0
    cum_log = np.cumsum(np.log1p(arr))
    t = np.arange(len(cum_log))
    if np.std(t) == 0 or np.std(cum_log) == 0:
        return 0.0
    corr = np.corrcoef(t, cum_log)[0, 1]
    return float(corr ** 2)


def win_rate(returns: pd.Series) -> float:
    arr = _prep_returns(returns)
    if len(arr) == 0:
        return 0.0
    return float((arr > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    arr = _prep_returns(returns)
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


# ----------------------------------------------------------------------------
# 相对基准的指标
# ----------------------------------------------------------------------------

def _aligned(returns: pd.Series, bench: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """按 index 对齐"""
    df = pd.concat([returns.rename("p"), bench.rename("b")], axis=1).dropna()
    return df["p"].values, df["b"].values


def alpha_beta(returns: pd.Series, bench: pd.Series) -> Tuple[float, float]:
    """CAPM alpha & beta, 年化 alpha"""
    p, b = _aligned(returns, bench)
    if len(p) < 2 or np.std(b) == 0:
        return 0.0, 0.0
    cov = np.cov(p, b, ddof=1)
    beta = float(cov[0, 1] / cov[1, 1])
    af = _annualize_factor(returns)
    rf_per_period = 0.03 / af
    alpha_daily = float(np.mean(p - rf_per_period) - beta * (np.mean(b) - rf_per_period))
    return alpha_daily * af, beta


def information_ratio(returns: pd.Series, bench: pd.Series) -> float:
    """IR = mean(active_return) / tracking_error, 年化"""
    p, b = _aligned(returns, bench)
    if len(p) < 2:
        return 0.0
    active = p - b
    te = np.std(active, ddof=1)
    if te == 0:
        return 0.0
    af = _annualize_factor(returns)
    return float(np.mean(active) / te * np.sqrt(af))


def tracking_error(returns: pd.Series, bench: pd.Series) -> float:
    """跟踪误差 (年化)"""
    p, b = _aligned(returns, bench)
    if len(p) < 2:
        return 0.0
    af = _annualize_factor(returns)
    return float(np.std(p - b, ddof=1) * np.sqrt(af))


def capture_ratio(returns: pd.Series, bench: pd.Series) -> float:
    """上行捕获/下行捕获, 二者之积"""
    p, b = _aligned(returns, bench)
    if len(p) == 0 or np.sum(b > 0) == 0 or np.sum(b < 0) == 0:
        return 0.0
    up = (p[b > 0].sum() / max(1, (b > 0).sum())) / (b[b > 0].sum() / max(1, (b > 0).sum()))
    dn = (p[b < 0].sum() / max(1, (b < 0).sum())) / (b[b < 0].sum() / max(1, (b < 0).sum()))
    if dn == 0:
        return 0.0
    return float(up / abs(dn))


# ----------------------------------------------------------------------------
# 校正多重检验的 Sharpe
# ----------------------------------------------------------------------------

def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    returns_skewness: float = 0.0,
    returns_kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Próy, 2014)
    简化实现: 用 Cornish-Fisher 展开估计 SR 的 p 值并转为 1-p 作为分数

    参数:
        observed_sharpe: 样本外观测到的 Sharpe
        n_trials: 试验次数 (策略/参数组合数)
        returns_skewness: 收益偏度
        returns_kurtosis: 收益峰度
        benchmark_sharpe: 期望的基准 SR
    """
    if n_trials <= 0:
        return 0.0
    # E[max SR] 的近似
    euler_gamma = 0.5772156649
    if n_trials == 1:
        expected_max = 0.0
    else:
        z = (1.0 - 1.0 / n_trials) * (
            (1 - euler_gamma) * (1 + (euler_gamma - math.log(n_trials)) / max(1.0, n_trials - 1))
        )
        # 用反 Erf 近似 (Abramowitz & Stegun)
        def _norm_ppf(p):
            # 简化版近似
            if p <= 0 or p >= 1:
                return 0.0
            t = math.sqrt(-2 * math.log(min(p, 1 - p)))
            c0, c1, c2 = 2.515517, 0.802853, 0.010328
            d1, d2, d3 = 1.432788, 0.189269, 0.001308
            approx = t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
            return approx if p > 0.5 else -approx

        expected_max = _norm_ppf(0.95) + (euler_gamma * _norm_ppf(0.95) - 1) / max(1.0, n_trials)
    # Cornish-Fisher 修正
    if returns_kurtosis > 0:
        cf = (observed_sharpe - expected_max) / (
            1
            - returns_skewness * observed_sharpe
            + (returns_kurtosis - 1) / 4 * observed_sharpe ** 2
        )
    else:
        cf = observed_sharpe - expected_max
    p_value = 0.5 * (1 - math.erf(cf / math.sqrt(2)))
    return float(max(0.0, 1.0 - p_value))


# ============================================================================
# 统一接口
# ============================================================================

def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    n_trials: int = 1,
) -> Dict[str, float]:
    """
    计算完整绩效指标

    参数:
        equity: 资金曲线 (pd.Series, index=date)
        returns: 收益序列 (pd.Series, index=date)
        benchmark_returns: 基准收益序列 (可选)
        risk_free: 年化无风险利率
        n_trials: 试验次数, 用于 deflated_sharpe

    返回:
        字典, 包含 20+ 项指标
    """
    if equity.empty or returns.empty:
        return {}
    metrics: Dict[str, float] = {
        "total_return": total_return(equity),
        "annual_return": annual_return(returns),
        "volatility": volatility(returns),
        "downside_volatility": downside_volatility(returns),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_duration": max_drawdown_duration(equity),
        "ulcer_index": ulcer_index(equity),
        "sharpe_ratio": sharpe_ratio(returns, risk_free),
        "sortino_ratio": sortino_ratio(returns, risk_free),
        "calmar_ratio": calmar_ratio(returns, equity),
        "omega_ratio": omega_ratio(returns),
        "tail_ratio": tail_ratio(returns),
        "stability": stability_of_returns(returns),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
    }
    if benchmark_returns is not None and not benchmark_returns.empty:
        a, b = alpha_beta(returns, benchmark_returns)
        metrics["alpha"] = a
        metrics["beta"] = b
        metrics["information_ratio"] = information_ratio(returns, benchmark_returns)
        metrics["tracking_error"] = tracking_error(returns, benchmark_returns)
        metrics["capture_ratio"] = capture_ratio(returns, benchmark_returns)
    if len(returns.dropna()) > 5:
        skew = float(returns.dropna().skew())
        kurt = float(returns.dropna().kurtosis() + 3)  # 转为原始峰度
        metrics["deflated_sharpe"] = deflated_sharpe_ratio(
            metrics["sharpe_ratio"], n_trials, skew, kurt
        )
    # 清理 inf 以便 JSON 序列化
    for k, v in list(metrics.items()):
        if math.isinf(v) or math.isnan(v):
            metrics[k] = 0.0
    return metrics


__all__ = [
    "compute_metrics", "TRADING_DAYS_PER_YEAR",
    "total_return", "annual_return", "volatility", "downside_volatility",
    "max_drawdown", "max_drawdown_duration", "ulcer_index",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
    "tail_ratio", "stability_of_returns", "win_rate", "profit_factor",
    "alpha_beta", "information_ratio", "tracking_error", "capture_ratio",
    "deflated_sharpe_ratio",
]
