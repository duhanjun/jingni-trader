"""
扩展风险指标（Extended Risk Metrics）

借鉴来源：
- VectorBT Portfolio.stats()：50+ 内置指标，含 VaR / CVaR / Information Ratio /
  Beta / Alpha / Turnover / Profit Factor / Expectancy
- Qlib backtest evaluator：IC / RankIC / Annualized Excess Return / Information Ratio
- youngju.dev VaR/CVaR 实战指南：Historical / Parametric / Monte Carlo 三种 VaR

对照 jingni-trader backtest-engine 既有 BaseBacktestMetrics 的缺口：
- 既有：total_return / annual_return / volatility / sharpe / max_drawdown /
  calmar / sortino / win_rate / total_trades
- 缺失：VaR / CVaR / Information Ratio / Beta / Alpha / Turnover /
  Profit Factor / Expectancy / Downside Deviation

本模块提供上述缺失指标的向量化实现，可作为 BaseBacktestMetrics 的扩展。
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def calc_var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    历史 VaR（Value at Risk）

    参数:
        returns: 日收益率序列
        confidence: 置信水平，如 0.95 表示 95% 置信

    返回:
        VaR 值（正数，表示最大损失比例）
    """
    if len(returns) < 2:
        return 0.0
    pct = (1 - confidence) * 100
    return float(-np.percentile(returns, pct))


def calc_cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    历史 CVaR / Expected Shortfall

    CVaR = 在收益低于 VaR 的条件下的平均损失
    """
    if len(returns) < 2:
        return 0.0
    var = calc_var_historical(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


def calc_var_parametric(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """
    参数法 VaR（假设正态分布）

    VaR = -(mean + z * std)
    """
    from scipy.stats import norm
    if len(returns) < 2:
        return 0.0
    z = norm.ppf(1 - confidence)  # 负数
    return float(-(returns.mean() + z * returns.std()))


def calc_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    信息比率 = 超额收益年化 / 跟踪误差年化

    参数:
        returns: 策略日收益率
        benchmark_returns: 基准日收益率
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    # 对齐索引
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return 0.0
    r = returns.loc[common]
    b = benchmark_returns.loc[common]
    excess = r - b
    tracking_error = excess.std() * np.sqrt(trading_days)
    if tracking_error == 0:
        return 0.0
    annual_excess = excess.mean() * trading_days
    return float(annual_excess / tracking_error)


def calc_beta(
    returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Beta = Cov(r, b) / Var(b)"""
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return 0.0
    r = returns.loc[common].values
    b = benchmark_returns.loc[common].values
    var_b = np.var(b, ddof=1)
    if var_b == 0:
        return 0.0
    cov = np.cov(r, b, ddof=1)[0, 1]
    return float(cov / var_b)


def calc_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> float:
    """
    Jensen's Alpha（年化）

    Alpha = R_p - [R_f + Beta * (R_b - R_f)]
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    beta = calc_beta(returns, benchmark_returns)
    ann_r = returns.mean() * trading_days
    ann_b = benchmark_returns.mean() * trading_days
    return float(ann_r - (risk_free + beta * (ann_b - risk_free)))


def calc_turnover(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
) -> float:
    """
    年化换手率 = 总成交额 / 平均持仓市值 / 年数

    参数:
        equity_curve: 含 date, market_value 列
        trades: 含 amount 列的成交记录
    """
    if trades.empty or equity_curve.empty:
        return 0.0
    total_amount = float(trades['amount'].sum())
    avg_mv = float(equity_curve['market_value'].mean()) if 'market_value' in equity_curve.columns else 0.0
    if avg_mv <= 0:
        return 0.0
    n_years = len(equity_curve) / 252
    if n_years <= 0:
        return 0.0
    return float(total_amount / avg_mv / n_years)


def calc_profit_factor(trades: pd.DataFrame) -> float:
    """盈亏比 = 总盈利 / 总亏损（绝对值）"""
    if trades.empty:
        return 0.0
    # 按 buy/sell 配对计算每笔交易盈亏较复杂，这里用 trades['pnl'] 近似
    # buy 的 pnl 为负（成本），sell 的 pnl 为正（收入）
    if 'pnl' not in trades.columns:
        return 0.0
    # 仅看 sell 单的 pnl（实现盈亏）
    sells = trades[trades.get('action', pd.Series()) == 'sell']
    if sells.empty:
        return 0.0
    gains = sells.loc[sells['pnl'] > 0, 'pnl'].sum()
    losses = abs(sells.loc[sells['pnl'] < 0, 'pnl'].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


def calc_expectancy(trades: pd.DataFrame) -> float:
    """期望收益 = 平均每笔交易盈亏"""
    if trades.empty:
        return 0.0
    sells = trades[trades.get('action', pd.Series()) == 'sell']
    if sells.empty or 'pnl' not in sells.columns:
        return 0.0
    return float(sells['pnl'].mean())


def calc_downside_deviation(
    returns: pd.Series, trading_days: int = 252
) -> float:
    """下行偏差（年化）"""
    neg = returns[returns < 0]
    if len(neg) < 2:
        return 0.0
    return float(neg.std() * np.sqrt(trading_days))


def calc_extended_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    trading_days: int = 252,
    var_confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    一次性计算扩展风险指标集

    在 BaseBacktestMetrics.calc_all_metrics 基础上增加：
    VaR / CVaR / Information Ratio / Beta / Alpha / Turnover /
    Profit Factor / Expectancy / Downside Deviation
    """
    if equity_curve.empty or 'equity' not in equity_curve.columns:
        return {}

    eq = equity_curve.set_index('date')['equity']
    returns = eq.pct_change().dropna()

    metrics: Dict[str, Any] = {
        "var_95": calc_var_historical(returns, var_confidence),
        "cvar_95": calc_cvar_historical(returns, var_confidence),
        "var_parametric_95": calc_var_parametric(returns, var_confidence),
        "downside_deviation": calc_downside_deviation(returns, trading_days),
        "turnover": calc_turnover(equity_curve, trades),
        "profit_factor": calc_profit_factor(trades),
        "expectancy": calc_expectancy(trades),
    }

    if benchmark_returns is not None and not benchmark_returns.empty:
        metrics["information_ratio"] = calc_information_ratio(
            returns, benchmark_returns, trading_days
        )
        metrics["beta"] = calc_beta(returns, benchmark_returns)
        metrics["alpha"] = calc_alpha(
            returns, benchmark_returns, risk_free, trading_days
        )

    return metrics
