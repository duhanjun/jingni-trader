"""
基准相对绩效指标模块

借鉴来源:
- Microsoft Qlib: portfolio_analysis 提供信息比率、跟踪误差、Alpha/Beta
- QuantConnect Lean: 完整的基准对比指标体系

优化点:
jingni-trader backtest-engine 的 _calc_metrics 仅计算绝对指标
(年化收益、夏普、最大回撤、胜率)，缺少相对基准的评估指标，
无法衡量策略的超额收益能力与风格暴露。

本模块补充:
- Information Ratio (信息比率)
- Tracking Error (跟踪误差)
- Alpha / Beta (CAPM)
- 超额收益曲线与最大回撤
- Up/Down Capture Ratio (牛熊捕获系数)
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("benchmark-metrics")


def calc_benchmark_metrics(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    计算策略相对基准的绩效指标

    参数:
        strategy_equity: 策略净值序列(index=date)
        benchmark_equity: 基准净值序列(index=date)
        risk_free_rate: 年化无风险利率
        trading_days: 年交易日数

    返回:
        包含信息比率、跟踪误差、Alpha、Beta 等指标的字典
    """
    # 对齐日期
    df = pd.DataFrame({"strategy": strategy_equity, "benchmark": benchmark_equity}).dropna()
    if len(df) < 2:
        return {}

    strat_ret = df["strategy"].pct_change().dropna()
    bench_ret = df["benchmark"].pct_change().dropna()
    excess_ret = strat_ret - bench_ret

    # 跟踪误差: 超额收益的年化波动率
    tracking_error = float(excess_ret.std() * np.sqrt(trading_days)) if len(excess_ret) > 1 else 0.0

    # 信息比率: 超额收益均值 / 跟踪误差
    excess_annual = float(excess_ret.mean() * trading_days)
    information_ratio = excess_annual / tracking_error if tracking_error > 0 else 0.0

    # Beta: cov(strategy, benchmark) / var(benchmark)
    cov = float(np.cov(strat_ret, bench_ret, ddof=1)[0, 1]) if len(strat_ret) > 1 else 0.0
    var_b = float(bench_ret.var(ddof=1)) if len(bench_ret) > 1 else 0.0
    beta = cov / var_b if var_b > 0 else 0.0

    # Alpha (CAPM): R_s - [R_f + Beta * (R_b - R_f)]
    rf_daily = risk_free_rate / trading_days
    strat_annual = float(strat_ret.mean() * trading_days)
    bench_annual = float(bench_ret.mean() * trading_days)
    alpha = strat_annual - (risk_free_rate + beta * (bench_annual - risk_free_rate))

    # 超额收益曲线与最大回撤
    excess_equity = (1 + excess_ret).cumprod()
    excess_mdd = float((excess_equity / excess_equity.cummax() - 1).min()) if len(excess_equity) > 1 else 0.0

    # 牛熊捕获系数
    up_days = bench_ret[bench_ret > 0].index
    down_days = bench_ret[bench_ret < 0].index
    up_capture = float(strat_ret.loc[up_days].mean() / bench_ret.loc[up_days].mean()) \
        if len(up_days) > 0 and bench_ret.loc[up_days].mean() != 0 else 0.0
    down_capture = float(strat_ret.loc[down_days].mean() / bench_ret.loc[down_days].mean()) \
        if len(down_days) > 0 and bench_ret.loc[down_days].mean() != 0 else 0.0

    # 相关系数
    correlation = float(strat_ret.corr(bench_ret)) if len(strat_ret) > 1 else 0.0

    return {
        "information_ratio": round(information_ratio, 4),
        "tracking_error": round(tracking_error, 4),
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "excess_return_annual": round(excess_annual, 4),
        "excess_max_drawdown": round(excess_mdd, 4),
        "up_capture_ratio": round(up_capture, 4),
        "down_capture_ratio": round(down_capture, 4),
        "correlation_with_benchmark": round(correlation, 4),
    }


def calc_full_metrics(
    strategy_equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """
    计算完整绩效指标(绝对 + 相对)

    参数:
        strategy_equity: 策略净值序列
        benchmark_equity: 基准净值序列(可选)
    """
    if len(strategy_equity) < 2:
        return {}

    returns = strategy_equity.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)
    n_years = len(returns) / trading_days
    annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    volatility = float(returns.std() * np.sqrt(trading_days)) if len(returns) > 1 else 0.0
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0.0
    max_drawdown = float((strategy_equity / strategy_equity.cummax() - 1).min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Sortino
    neg_ret = returns[returns < 0]
    downside_std = float(neg_ret.std() * np.sqrt(trading_days)) if len(neg_ret) > 1 else 0.0
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0

    metrics = {
        "total_return": round(total_return, 4),
        "annual_return": round(annual_return, 4),
        "volatility": round(volatility, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_drawdown, 4),
        "calmar_ratio": round(calmar, 4),
        "win_rate": round(float((returns > 0).mean()), 4),
    }

    if benchmark_equity is not None and len(benchmark_equity) > 1:
        bench_metrics = calc_benchmark_metrics(
            strategy_equity, benchmark_equity, risk_free_rate, trading_days,
        )
        metrics.update(bench_metrics)

    return metrics
