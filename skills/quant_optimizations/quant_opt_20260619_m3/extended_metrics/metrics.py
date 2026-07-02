"""
Extended Performance Metrics (借鉴 VectorBT 的 stats API)

jingni-trader 当前的 BaseBacktestMetrics 只覆盖 9 个指标,
VectorBT 则有 60+ 指标. 本模块实现 14 个常用扩展指标, 保持与原接口兼容.

设计: 全为 @staticmethod, 接受 pd.Series (净值或收益), 便于复用与测试.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _returns(equity: pd.Series) -> pd.Series:
    """等价的简单收益率序列, 已 dropna"""
    return equity.pct_change().dropna()


def _maybe_to_returns(x: pd.Series, threshold_hint: float = 0.0) -> pd.Series:
    """如果输入看起来像价格序列(全正且非均值附近), 转换为收益率;
    否则直接当作收益率返回.
    """
    if x.empty:
        return x
    # 启发: 若序列全正且数值大于 1 大概率是价格, 用 pct_change 转
    if (x > 0).all() and x.median() > 1.0:
        return _returns(x)
    return x.dropna()


# ---------------------------------------------------------------------------
# 收益/风险类
# ---------------------------------------------------------------------------
def omega_ratio(returns: pd.Series, threshold: float = 0.0, trading_days: int = 252) -> float:
    """
    Omega 比率 = (收益超过 threshold 的部分累计) / (亏损部分累计绝对值)
    借鉴 VectorBT.stats.omega_ratio
    """
    r = _maybe_to_returns(returns)
    excess = r - threshold
    pos = excess[excess > 0].sum()
    neg = -excess[excess < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def ulcer_index(equity: pd.Series) -> float:
    """
    Ulcer Index: 衡量回撤深度与持续时间
    UI = sqrt(mean(drawdown^2))
    """
    if len(equity) < 2:
        return 0.0
    cummax = equity.cummax()
    dd_pct = (equity - cummax) / cummax
    return float(np.sqrt(np.mean(dd_pct[dd_pct < 0] ** 2))) if (dd_pct < 0).any() else 0.0


def ulcer_performance_index(equity: pd.Series, risk_free: float = 0.0,
                            trading_days: int = 252) -> float:
    """UPI = (年化收益 - 无风险) / UlcerIndex"""
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    n_years = len(r) / trading_days
    if n_years <= 0:
        return 0.0
    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    ui = ulcer_index(equity)
    if ui == 0:
        return 0.0
    return float((ann_ret - risk_free) / ui)


def serenity_index(equity: pd.Series, risk_free: float = 0.0, trading_days: int = 252) -> float:
    """
    Serenity Index: 类似 UPI 但用绝对回撤, 对异常回撤更稳健
    """
    r = _returns(equity)
    if len(r) < 2:
        return 0.0
    n_years = len(r) / trading_days
    if n_years <= 0:
        return 0.0
    ann_ret = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    abs_dd = dd.abs().mean()
    if abs_dd == 0:
        return 0.0
    return float((ann_ret - risk_free) / abs_dd)


def deflated_sharpe_ratio(returns: pd.Series, risk_free: float = 0.0,
                          trading_days: int = 252,
                          n_trials: int = 100) -> float:
    """
    通胀夏普 (DSR): 考虑多重检验偏差, 借鉴 Lopez de Prado 的方法
    当 n_trials=1 时退化为普通夏普
    """
    r = _maybe_to_returns(returns)
    if len(r) < 2:
        return 0.0
    sr = (r.mean() - risk_free / trading_days) / r.std() * np.sqrt(trading_days) if r.std() > 0 else 0.0
    T = len(r)
    if T <= 0 or n_trials <= 1:
        return float(sr)
    skew = float(r.skew()) if hasattr(r, "skew") else 0.0
    kurt = float(r.kurt() + 3) if hasattr(r, "kurt") else 3.0
    if kurt <= 3:
        kurt = 3.0001
    denom = 1 - skew * sr + ((kurt - 1) / 4) * sr ** 2
    if denom <= 0:
        return 0.0
    return float(sr / np.sqrt(denom))


def tail_ratio(returns: pd.Series, percentile: float = 0.95) -> float:
    """
    Tail Ratio = 右尾分位 / |左尾分位|
    >1 表示右尾厚, 策略偏向收益; <1 表示左尾厚, 偏向亏损
    """
    r = _maybe_to_returns(returns)
    if len(r) < 2:
        return 0.0
    right = float(np.percentile(r, percentile * 100))
    left = float(np.percentile(r, (1 - percentile) * 100))
    if left == 0:
        return 0.0
    return float(abs(right / left))


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """Gain-to-Pain Ratio: 累计正收益 / |累计负收益|"""
    r = _maybe_to_returns(returns)
    pos_sum = r[r > 0].sum()
    neg_sum = -r[r < 0].sum()
    if neg_sum == 0:
        return float("inf") if pos_sum > 0 else 0.0
    return float(pos_sum / neg_sum)


def profit_factor(returns: pd.Series) -> float:
    """Profit Factor: 总盈利 / |总亏损|"""
    return gain_to_pain_ratio(returns)


# ---------------------------------------------------------------------------
# 稳定性/分布类
# ---------------------------------------------------------------------------
def stability_of_returns(returns: pd.Series, trading_days: int = 252) -> float:
    """
    R^2 of log-cumulative return over time
    借鉴 vectorbt 的 stability_of_returns, 衡量收益曲线的稳定性
    """
    r = _returns(returns) if returns.iloc[0] != 0 else returns
    if len(r) < 2:
        return 0.0
    cum_log = np.log1p(r).cumsum()
    x = np.arange(len(cum_log))
    if x.std() == 0 or cum_log.std() == 0:
        return 0.0
    corr = np.corrcoef(x, cum_log.values)[0, 1]
    return float(corr ** 2)


def r2_in_sample(returns: pd.Series, trading_days: int = 252) -> float:
    """与 stability_of_returns 等价, 命名兼容"""
    return stability_of_returns(returns, trading_days)


def max_drawdown_duration(equity: pd.Series) -> int:
    """最大回撤持续期 (交易日数)"""
    if len(equity) < 2:
        return 0
    cummax = equity.cummax()
    is_dd = equity < cummax
    groups = (~is_dd).cumsum()
    durations = is_dd.groupby(groups).sum()
    if len(durations) == 0:
        return 0
    return int(durations.max())


# ---------------------------------------------------------------------------
# 基准对比类
# ---------------------------------------------------------------------------
def beta(returns: pd.Series, bench_returns: pd.Series) -> float:
    """策略相对基准的 beta"""
    aligned = pd.concat([returns.rename("s"), bench_returns.rename("b")], axis=1).dropna()
    if len(aligned) < 2 or aligned["b"].var() == 0:
        return 0.0
    return float(aligned[["s", "b"]].cov().iloc[0, 1] / aligned["b"].var())


def alpha(returns: pd.Series, bench_returns: pd.Series, risk_free: float = 0.0,
          trading_days: int = 252) -> float:
    """CAPM alpha (年化)"""
    aligned = pd.concat([returns.rename("s"), bench_returns.rename("b")], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    b = beta(aligned["s"], aligned["b"])
    sr_ann = aligned["s"].mean() * trading_days
    br_ann = aligned["b"].mean() * trading_days
    return float(sr_ann - (risk_free + b * (br_ann - risk_free)))


def information_ratio(returns: pd.Series, bench_returns: pd.Series,
                      trading_days: int = 252) -> float:
    """IR = 年化超额收益 / 跟踪误差"""
    aligned = pd.concat([returns.rename("s"), bench_returns.rename("b")], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    excess = aligned["s"] - aligned["b"]
    tracking_err = excess.std() * np.sqrt(trading_days)
    if tracking_err == 0:
        return 0.0
    return float(excess.mean() * trading_days / tracking_err)


# ---------------------------------------------------------------------------
# 一键计算
# ---------------------------------------------------------------------------
def calc_extended_metrics(
    equity: pd.Series,
    bench_equity: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """
    一次性计算扩展指标 (与 base_backtest.BaseBacktestMetrics.calc_all_metrics 接口一致)

    输入:
        equity: 净值曲线
        bench_equity: 基准净值曲线(可选)
    """
    out: Dict[str, Any] = {}
    if len(equity) < 2:
        return out
    r = _returns(equity)
    if len(r) < 1:
        return out
    out["omega_ratio"] = omega_ratio(r, threshold=0.0, trading_days=trading_days)
    out["ulcer_index"] = ulcer_index(equity)
    out["upi"] = ulcer_performance_index(equity, risk_free=risk_free, trading_days=trading_days)
    out["serenity_index"] = serenity_index(equity, risk_free=risk_free, trading_days=trading_days)
    out["deflated_sharpe"] = deflated_sharpe_ratio(r, risk_free=risk_free, trading_days=trading_days, n_trials=1)
    out["tail_ratio"] = tail_ratio(r)
    out["gain_to_pain"] = gain_to_pain_ratio(r)
    out["profit_factor"] = profit_factor(r)
    out["stability_r2"] = stability_of_returns(r, trading_days=trading_days)
    out["max_dd_duration"] = max_drawdown_duration(equity)

    if bench_equity is not None and len(bench_equity) >= 2:
        br = _returns(bench_equity)
        aligned = pd.concat([r, br], axis=1).dropna()
        if len(aligned) >= 2:
            out["beta"] = beta(aligned.iloc[:, 0], aligned.iloc[:, 1])
            out["alpha_annual"] = alpha(aligned.iloc[:, 0], aligned.iloc[:, 1],
                                         risk_free=risk_free, trading_days=trading_days)
            out["information_ratio"] = information_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1],
                                                          trading_days=trading_days)
    return out


__all__ = [
    "omega_ratio", "ulcer_index", "ulcer_performance_index",
    "serenity_index", "deflated_sharpe_ratio", "tail_ratio",
    "gain_to_pain_ratio", "profit_factor",
    "stability_of_returns", "max_drawdown_duration",
    "beta", "alpha", "information_ratio",
    "calc_extended_metrics",
]