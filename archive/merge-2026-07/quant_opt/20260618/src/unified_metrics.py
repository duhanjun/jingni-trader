"""
统一量化指标库 (unified_metrics)
==================================

借鉴来源:
  - QuantStats (ranaroussi/quantstats): 30+ 风险调整指标、滚动指标、基准比较
  - Alphalens (quantopian/alphalens): 因子分析指标（IC、Rank IC、turnover、因子收益）
  - Qlib (microsoft/qlib): 绩效指标体系与基准归因

设计目标:
  1. 替代 jingni-trader/skills/backtest-engine/scripts/base/base_backtest.py 的 7 个指标
  2. 与 jingni-trader 的 equity_curve (date, equity) 数据结构兼容
  3. 纯 numpy/pandas 实现，不强依赖 quantstats/alphalens

注意事项:
  - 所有函数以 pd.Series 形式的日度收益/权益曲线作为输入
  - 年化假设 252 个交易日（可通过 periods 参数调整）
  - 收益序列需按日期升序
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ============================================================
# Section 1. 基础收益 / 风险指标
# ============================================================

def cagr(equity: pd.Series, periods: int = 252) -> float:
    """复合年化增长率 (Compound Annual Growth Rate)"""
    if equity is None or len(equity) < 2:
        return 0.0
    n_years = len(equity) / periods
    if n_years <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    if total <= 0:
        return -1.0
    return float(total ** (1.0 / n_years) - 1.0)


def total_return(equity: pd.Series) -> float:
    """累计收益率"""
    if equity is None or len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def volatility(returns: pd.Series, periods: int = 252) -> float:
    """年化波动率"""
    if returns is None or len(returns) < 2:
        return 0.0
    return float(returns.std() * math.sqrt(periods))


def downside_deviation(returns: pd.Series, mar: float = 0.0, periods: int = 252) -> float:
    """下行偏差 (Downside Deviation)，仅考虑低于 MAR 的波动"""
    if returns is None or len(returns) < 2:
        return 0.0
    excess = returns - mar
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    return float(math.sqrt((downside ** 2).mean()) * math.sqrt(periods))


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """最大回撤，返回 (mdd, peak_date, trough_date)"""
    if equity is None or len(equity) < 2:
        return 0.0, pd.Timestamp(0), pd.Timestamp(0)
    cummax = equity.cummax()
    drawdown = equity / cummax - 1.0
    mdd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = equity.loc[:trough_date].idxmax()
    return float(mdd), peak_date, trough_date


def max_drawdown_duration(equity: pd.Series) -> int:
    """最长回撤持续期 (交易日数)"""
    if equity is None or len(equity) < 2:
        return 0
    cummax = equity.cummax()
    underwater = equity < cummax
    groups = (~underwater).cumsum()
    durations = underwater.groupby(groups).sum()
    return int(durations.max()) if len(durations) else 0


# ============================================================
# Section 2. 风险调整指标
# ============================================================

def sharpe(returns: pd.Series, rf: float = 0.03, periods: int = 252) -> float:
    """夏普比率"""
    vol = volatility(returns, periods)
    if vol == 0:
        return 0.0
    excess_annual = returns.mean() * periods - rf
    return float(excess_annual / vol)


def smart_sharpe(returns: pd.Series, rf: float = 0.03, periods: int = 252) -> float:
    """
    智能夏普（Smart Sharpe），借鉴 QuantStats
    对收益的自相关性施加惩罚，避免序列相关导致夏普虚高
    """
    vol = volatility(returns, periods)
    if vol == 0 or len(returns) < 3:
        return 0.0
    # 自相关惩罚项: sqrt(1 + 2 * sum(rho_i))，i=1..k
    # 取 lag1..lag10 平均
    max_lag = min(10, len(returns) // 4)
    if max_lag < 1:
        return sharpe(returns, rf, periods)
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = returns.autocorr(lag)
        if pd.isna(rho):
            continue
        rho_sum += rho
    penalty = math.sqrt(max(0.0, 1.0 + 2.0 * rho_sum))
    excess_annual = returns.mean() * periods - rf
    denom = vol * penalty
    if denom == 0:
        return 0.0
    return float(excess_annual / denom)


def sortino(returns: pd.Series, rf: float = 0.03, periods: int = 252) -> float:
    """索提诺比率（仅惩罚下行风险）"""
    dd = downside_deviation(returns, 0.0, periods)
    if dd == 0:
        return 0.0
    excess_annual = returns.mean() * periods - rf
    return float(excess_annual / dd)


def calmar(equity: pd.Series, periods: int = 252) -> float:
    """Calmar 比率：年化收益 / |最大回撤|"""
    ann = cagr(equity, periods)
    mdd, _, _ = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(ann / abs(mdd))


def ulcer_index(equity: pd.Series) -> float:
    """Ulcer Index：衡量回撤深度与持续期"""
    if equity is None or len(equity) < 2:
        return 0.0
    cummax = equity.cummax()
    drawdown_pct = (equity - cummax) / cummax * 100.0
    return float(math.sqrt((drawdown_pct ** 2).mean()))


def ulcer_performance_index(equity: pd.Series, rf: float = 0.03, periods: int = 252) -> float:
    """Ulcer Performance Index：借鉴 QuantStats"""
    ui = ulcer_index(equity)
    if ui == 0:
        return 0.0
    excess = cagr(equity, periods) - rf
    return float(excess / ui)


def omega(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega 比率：收益分布的累计概率之比"""
    if returns is None or len(returns) < 2:
        return 0.0
    excess = returns - threshold
    gain = excess[excess > 0].sum()
    loss = -excess[excess < 0].sum()
    if loss == 0:
        return float('inf') if gain > 0 else 0.0
    return float(gain / loss)


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """总盈利 / |总亏损|，借鉴 QuantStats"""
    if returns is None or len(returns) == 0:
        return 0.0
    total_gain = returns[returns > 0].sum()
    total_loss = -returns[returns < 0].sum()
    if total_loss == 0:
        return float('inf') if total_gain > 0 else 0.0
    return float(total_gain / total_loss)


def profit_factor(returns: pd.Series) -> float:
    """Profit Factor: gross profit / gross loss"""
    return gain_to_pain_ratio(returns)


# ============================================================
# Section 3. 风险指标 (VaR / CVaR / Tail)
# ============================================================

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """VaR（历史法）"""
    if returns is None or len(returns) < 2:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall"""
    if returns is None or len(returns) < 2:
        return 0.0
    var_threshold = value_at_risk(returns, confidence)
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        return var_threshold
    return float(tail.mean())


def tail_ratio(returns: pd.Series, confidence: float = 0.95) -> float:
    """Tail Ratio: |right tail| / |left tail|"""
    if returns is None or len(returns) < 2:
        return 0.0
    right = np.percentile(returns, confidence * 100)
    left = np.percentile(returns, (1 - confidence) * 100)
    if left == 0:
        return 0.0
    return float(right / abs(left))


# ============================================================
# Section 4. 基准归因 (借鉴 Qlib + QuantStats)
# ============================================================

def alpha_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    rf: float = 0.03,
    periods: int = 252,
) -> Tuple[float, float]:
    """
    CAPM Alpha & Beta

    Returns:
        (alpha_annualized, beta)
    """
    if returns is None or benchmark is None or len(returns) < 2:
        return 0.0, 0.0
    # 对齐
    df = pd.concat([returns.rename("s"), benchmark.rename("b")], axis=1, join="inner").dropna()
    if len(df) < 2:
        return 0.0, 0.0
    s = df["s"] - rf / periods
    b = df["b"] - rf / periods
    cov = np.cov(s, b, ddof=1)
    var_b = cov[1, 1]
    if var_b == 0:
        return 0.0, 0.0
    beta = cov[0, 1] / var_b
    alpha_daily = s.mean() - beta * b.mean()
    alpha_annual = alpha_daily * periods
    return float(alpha_annual), float(beta)


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods: int = 252) -> float:
    """信息比率：active return / tracking error"""
    if returns is None or benchmark is None or len(returns) < 2:
        return 0.0
    df = pd.concat([returns.rename("s"), benchmark.rename("b")], axis=1, join="inner").dropna()
    if len(df) < 2:
        return 0.0
    active = df["s"] - df["b"]
    te = active.std() * math.sqrt(periods)
    if te == 0:
        return 0.0
    return float((active.mean() * periods) / te)


def treynor(returns: pd.Series, benchmark: pd.Series, rf: float = 0.03, periods: int = 252) -> float:
    """Treynor 比率：(年化收益 - rf) / beta"""
    _, beta = alpha_beta(returns, benchmark, rf, periods)
    if beta == 0:
        return 0.0
    ann = returns.mean() * periods
    return float((ann - rf) / beta)


def capture_ratios(
    returns: pd.Series,
    benchmark: pd.Series,
) -> Tuple[float, float]:
    """上行/下行捕获比率（借鉴 QuantStats）"""
    if returns is None or benchmark is None or len(returns) < 2:
        return 0.0, 0.0
    df = pd.concat([returns.rename("s"), benchmark.rename("b")], axis=1, join="inner").dropna()
    up = df[df["b"] > 0]
    dn = df[df["b"] < 0]
    if len(up) == 0 or len(dn) == 0:
        return 0.0, 0.0
    up_cap = up["s"].mean() / up["b"].mean() if up["b"].mean() != 0 else 0.0
    dn_cap = dn["s"].mean() / dn["b"].mean() if dn["b"].mean() != 0 else 0.0
    return float(up_cap), float(dn_cap)


# ============================================================
# Section 5. 因子分析指标 (借鉴 Alphalens + AlphaPurify)
# ============================================================

def factor_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    因子 IC 时间序列 (Information Coefficient)

    Args:
        factor: MultiIndex (date, code) 或 (date, code) 列 'factor' 的 DataFrame
        forward_returns: 同结构，列 'ret'
        method: 'spearman' | 'pearson'

    Returns:
        以 date 为索引的 IC Series
    """
    df = _merge_factor_return(factor, forward_returns)
    if df.empty:
        return pd.Series(dtype=float)

    ic_by_date = {}
    for dt, grp in df.groupby(level="date"):
        if len(grp) < 10:
            continue
        x = grp["factor"].values
        y = grp["ret"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 10:
            continue
        x, y = x[mask], y[mask]
        if method == "spearman":
            corr, _ = sp_stats.spearmanr(x, y)
        else:
            corr, _ = sp_stats.pearsonr(x, y)
        if not np.isnan(corr):
            ic_by_date[dt] = corr

    return pd.Series(ic_by_date).sort_index()


def factor_ic_decay(
    factor: pd.DataFrame,
    returns_df: pd.DataFrame,
    periods: List[int] = [1, 5, 10, 20],
    method: str = "spearman",
) -> pd.DataFrame:
    """
    因子 IC 衰减曲线：对每个持有期计算 IC_mean, IC_std, IC_IR, IC > 0 占比
    """
    out = []
    for p in periods:
        # 构造 p 日 forward return
        if "ret_1d" in returns_df.columns:
            ret_col = f"ret_{p}d_fwd"
            if ret_col not in returns_df.columns:
                # 用 shift 构造
                returns_df = returns_df.copy()
                returns_df[ret_col] = returns_df.groupby(level="code")["ret_1d"].shift(-p)
        else:
            ret_col = "ret"

        fwd = returns_df[[ret_col]].rename(columns={ret_col: "ret"}).reset_index()
        f = factor.reset_index()
        ic_series = factor_ic(f, fwd, method=method)
        if ic_series.empty:
            continue
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_pos = (ic_series > 0).mean()
        ic_t = ic_mean / (ic_std / math.sqrt(len(ic_series))) if ic_std > 0 else 0.0
        out.append({
            "period": p,
            "ic_mean": float(ic_mean),
            "ic_std": float(ic_std),
            "ic_ir": float(ic_ir),
            "ic_positive_ratio": float(ic_pos),
            "ic_t_stat": float(ic_t),
            "n_periods": len(ic_series),
        })
    return pd.DataFrame(out)


def factor_quantile_returns(
    factor: pd.DataFrame,
    returns_df: pd.DataFrame,
    n_quantiles: int = 5,
    period: int = 1,
) -> pd.DataFrame:
    """
    因子分位收益（借鉴 Alphalens / AlphaPurify）

    Args:
        factor: MultiIndex (date, code) 含 'factor' 列
        returns_df: MultiIndex (date, code) 含 'ret_1d' 列
        n_quantiles: 分位数（5=五分位）
        period: 持有期（1=次日收益）

    Returns:
        DataFrame，行=分位（1=最低, n=最高），列=统计量
    """
    f = factor.reset_index()
    fwd_col = f"ret_{period}d_fwd"
    if fwd_col not in returns_df.columns:
        returns_df = returns_df.copy()
        if "ret_1d" in returns_df.columns:
            returns_df[fwd_col] = returns_df.groupby(level="code")["ret_1d"].shift(-period)
        elif "ret" in returns_df.columns:
            # already forward returns, just rename
            returns_df[fwd_col] = returns_df["ret"]
        else:
            return pd.DataFrame()
    fwd = returns_df[[fwd_col]].rename(columns={fwd_col: "ret"}).reset_index()

    merged = f.merge(fwd, on=["date", "code"], how="inner").dropna()
    if merged.empty:
        return pd.DataFrame()

    merged["quantile"] = merged.groupby("date")["factor"].transform(
        lambda x: pd.qcut(x.rank(method="first"), n_quantiles, labels=False, duplicates="drop")
    ) + 1

    grp = merged.groupby("quantile")["ret"]
    summary = pd.DataFrame({
        "mean_return": grp.mean(),
        "std_return": grp.std(),
        "count": grp.count(),
    })
    summary["mean_annual"] = summary["mean_return"] * 252
    summary["std_annual"] = summary["std_return"] * math.sqrt(252)
    return summary


def factor_turnover(
    factor: pd.DataFrame,
    n_quantiles: int = 5,
) -> Dict[str, float]:
    """
    因子分位组合的换手率 (借鉴 Alphalens)
    """
    f = factor.reset_index()
    f["quantile"] = f.groupby("date")["factor"].transform(
        lambda x: pd.qcut(x.rank(method="first"), n_quantiles, labels=False, duplicates="drop")
    ) + 1

    turnover_by_q = {}
    for q, grp in f.groupby("quantile"):
        dates = sorted(grp["date"].unique())
        if len(dates) < 2:
            turnover_by_q[int(q)] = 0.0
            continue
        # 相邻期持有的股票交集变化率
        prev_set = set(grp[grp["date"] == dates[0]]["code"])
        rates = []
        for dt in dates[1:]:
            cur_set = set(grp[grp["date"] == dt]["code"])
            if not prev_set:
                continue
            diff = len(cur_set.symmetric_difference(prev_set))
            rates.append(diff / (2 * len(prev_set | cur_set) + 1e-9))
            prev_set = cur_set
        turnover_by_q[int(q)] = float(np.mean(rates)) if rates else 0.0

    return turnover_by_q


# ============================================================
# Section 6. 交易统计
# ============================================================

def trade_stats(trades: pd.DataFrame) -> Dict[str, float]:
    """
    从成交明细 (列: action, price, shares, amount, pnl) 提取交易统计
    """
    if trades is None or trades.empty:
        return {
            "total_trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_commission": 0.0,
            "total_tax": 0.0,
        }
    sell_trades = trades[trades["action"] == "sell"] if "action" in trades.columns else trades
    pnl = sell_trades["pnl"] if "pnl" in sell_trades.columns else pd.Series(dtype=float)
    return {
        "total_trades": int(len(trades)),
        "buy_trades": int((trades["action"] == "buy").sum()) if "action" in trades.columns else 0,
        "sell_trades": int((trades["action"] == "sell").sum()) if "action" in trades.columns else 0,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "total_commission": float(trades["commission"].sum()) if "commission" in trades.columns else 0.0,
        "total_tax": float(trades["tax"].sum()) if "tax" in trades.columns else 0.0,
    }


# ============================================================
# Section 7. 一站式计算
# ============================================================

def compute_all_metrics(
    equity: pd.Series,
    returns: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    trades: Optional[pd.DataFrame] = None,
    rf: float = 0.03,
    periods: int = 252,
) -> Dict[str, float]:
    """
    一站式计算所有标准指标，借鉴 QuantStats full() 接口

    Args:
        equity: 权益曲线 pd.Series，索引日期
        returns: 收益序列 pd.Series（可由 equity 派生）
        benchmark_returns: 基准收益序列
        trades: 交易记录
        rf: 无风险利率（年化）
        periods: 年化交易日数

    Returns:
        包含 30+ 指标的字典
    """
    if returns is None:
        if equity is None or len(equity) < 2:
            return {}
        returns = equity.pct_change().dropna()

    out: Dict[str, float] = {}

    # Section 1: 基础
    out["total_return"] = total_return(equity)
    out["cagr"] = cagr(equity, periods)
    out["volatility"] = volatility(returns, periods)
    mdd, peak_d, trough_d = max_drawdown(equity)
    out["max_drawdown"] = mdd
    out["max_drawdown_duration"] = max_drawdown_duration(equity)
    out["mdd_peak_date"] = str(peak_d)[:10] if peak_d is not pd.Timestamp(0) else ""
    out["mdd_trough_date"] = str(trough_d)[:10] if trough_d is not pd.Timestamp(0) else ""

    # Section 2: 风险调整
    out["sharpe"] = sharpe(returns, rf, periods)
    out["smart_sharpe"] = smart_sharpe(returns, rf, periods)
    out["sortino"] = sortino(returns, rf, periods)
    out["calmar"] = calmar(equity, periods)
    out["ulcer_index"] = ulcer_index(equity)
    out["ulcer_performance_index"] = ulcer_performance_index(equity, rf, periods)
    out["omega"] = omega(returns)
    out["gain_to_pain_ratio"] = gain_to_pain_ratio(returns)
    out["profit_factor"] = profit_factor(returns)

    # Section 3: 风险
    out["var_95"] = value_at_risk(returns, 0.95)
    out["cvar_95"] = conditional_var(returns, 0.95)
    out["tail_ratio"] = tail_ratio(returns, 0.95)

    # Section 4: 基准
    if benchmark_returns is not None and len(benchmark_returns) >= 2:
        a, b = alpha_beta(returns, benchmark_returns, rf, periods)
        out["alpha"] = a
        out["beta"] = b
        out["information_ratio"] = information_ratio(returns, benchmark_returns, periods)
        out["treynor"] = treynor(returns, benchmark_returns, rf, periods)
        up_cap, dn_cap = capture_ratios(returns, benchmark_returns)
        out["up_capture"] = up_cap
        out["down_capture"] = dn_cap
        # 基准自身的指标
        out["benchmark_total_return"] = total_return(_to_equity(benchmark_returns))
        out["benchmark_cagr"] = cagr(_to_equity(benchmark_returns), periods)
        out["benchmark_volatility"] = volatility(benchmark_returns, periods)
        out["benchmark_sharpe"] = sharpe(benchmark_returns, rf, periods)
        out["benchmark_max_drawdown"] = max_drawdown(_to_equity(benchmark_returns))[0]

    # Section 6: 交易
    if trades is not None:
        out.update({f"trade_{k}": v for k, v in trade_stats(trades).items()})

    return out


# ============================================================
# 内部工具
# ============================================================

def _merge_factor_return(factor: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.DataFrame:
    """合并因子与未来收益，要求 MultiIndex(date, code)"""
    f = factor.copy()
    if not isinstance(f.index, pd.MultiIndex):
        if {"date", "code"}.issubset(f.columns):
            f = f.set_index(["date", "code"])
        else:
            return pd.DataFrame()
    r = forward_returns.copy()
    if not isinstance(r.index, pd.MultiIndex):
        if {"date", "code"}.issubset(r.columns):
            r = r.set_index(["date", "code"])
        else:
            return pd.DataFrame()
    return f.join(r, how="inner").dropna()


def _to_equity(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """收益序列转权益曲线"""
    return (1 + returns.fillna(0)).cumprod() * start


# 暴露的主要 API
__all__ = [
    # 基础
    "cagr", "total_return", "volatility", "downside_deviation",
    "max_drawdown", "max_drawdown_duration",
    # 风险调整
    "sharpe", "smart_sharpe", "sortino", "calmar",
    "ulcer_index", "ulcer_performance_index", "omega",
    "gain_to_pain_ratio", "profit_factor",
    # 风险
    "value_at_risk", "conditional_var", "tail_ratio",
    # 基准归因
    "alpha_beta", "information_ratio", "treynor", "capture_ratios",
    # 因子分析
    "factor_ic", "factor_ic_decay", "factor_quantile_returns", "factor_turnover",
    # 交易
    "trade_stats",
    # 一站式
    "compute_all_metrics",
]
