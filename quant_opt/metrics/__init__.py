"""
增强绩效指标
=============

借鉴来源：
- AkQuant 0.2.43 的 HTML 报告（基准对比区块：跟踪误差、信息比率、Alpha、Beta）
- Qlib 的 StandardMetrics + 风险调整指标
- quantstats 的 `qs.stats` 系列指标

补充 jingni-trader 原生 BaseBacktestMetrics 中缺失的：
- 跟踪误差 (Tracking Error)
- 信息比率 (Information Ratio, IR)
- Jensen's Alpha / Beta（vs 基准）
- 最大回撤持续期 (Max DD Duration)
- 换手率
- 基准相对收益 (Excess Return)
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def _series_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def calc_all_metrics(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """
    计算完整绩效指标

    参数:
        equity_curve:  策略净值 Series（index=date）
        trades:        成交记录 DataFrame
        benchmark:     基准净值 Series（index=date），可选
        risk_free:     无风险利率（年化）
        trading_days:  年交易日数

    返回:
        dict，含基础指标 + 风险调整指标 + （可选）基准对比
    """
    if equity_curve is None or len(equity_curve) < 2:
        return {}

    eq = equity_curve.dropna()
    if len(eq) < 2:
        return {}
    returns = _series_returns(eq)

    metrics: Dict[str, Any] = {}

    # ── 1. 基础指标 ──
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_years = len(eq) / trading_days
    annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    volatility = float(returns.std() * np.sqrt(trading_days)) if len(returns) else 0.0
    excess = annual_return - risk_free
    sharpe = excess / volatility if volatility > 0 else 0.0
    downside = returns[returns < 0]
    downside_std = float(downside.std() * np.sqrt(trading_days)) if len(downside) > 1 else 0.0
    sortino = excess / downside_std if downside_std > 0 else 0.0

    cum_max = eq.cummax()
    drawdown = (eq - cum_max) / cum_max
    mdd = float(drawdown.min())

    # 回撤持续期：以"距离新高"的最大天数衡量
    is_at_high = eq == cum_max
    high_groups = (~is_at_high).cumsum()
    underwater_periods = drawdown.groupby(high_groups).size()
    mdd_duration = int(underwater_periods.max()) if len(underwater_periods) else 0

    metrics.update({
        "total_return": round(total_return, 6),
        "annual_return": round(annual_return, 6),
        "volatility": round(volatility, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(mdd, 6),
        "max_drawdown_duration_days": mdd_duration,
        "calmar_ratio": round(annual_return / abs(mdd), 4) if mdd != 0 else 0.0,
        "n_periods": int(len(eq)),
        "start_date": str(eq.index.min()),
        "end_date": str(eq.index.max()),
    })

    # ── 2. 交易统计 ──
    if trades is not None:
        metrics["total_trades"] = int(len(trades))
        if not trades.empty and "pnl" in trades.columns:
            pnls = trades["pnl"]
            metrics["win_rate"] = round(float((pnls > 0).mean()), 4)
            metrics["avg_pnl"] = round(float(pnls.mean()), 4)
            metrics["total_pnl"] = round(float(pnls.sum()), 4)
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            if len(wins) > 0 and len(losses) > 0 and losses.sum() != 0:
                avg_win = float(wins.sum() / len(wins))
                avg_loss = float(losses.sum() / len(losses))
                metrics["profit_loss_ratio"] = round(avg_win / abs(avg_loss), 4)
            else:
                metrics["profit_loss_ratio"] = 0.0
        else:
            metrics["win_rate"] = 0.0
            metrics["profit_loss_ratio"] = 0.0

    # ── 3. 基准对比（核心：借鉴 AkQuant 报告）──
    if benchmark is not None and len(benchmark) > 2:
        bench = benchmark.reindex(eq.index).ffill().dropna()
        if len(bench) >= 2:
            bench_returns = bench.pct_change().dropna()
            common = returns.index.intersection(bench_returns.index)
            if len(common) >= 2:
                r_p = returns.loc[common]
                r_b = bench_returns.loc[common]
                excess_returns = r_p - r_b

                # Tracking Error
                te = float(excess_returns.std() * np.sqrt(trading_days))
                # Information Ratio
                ir = float(excess_returns.mean() * trading_days / te) if te > 0 else 0.0

                # Beta & Alpha（CAPM 回归）
                var_b = float(r_b.var())
                beta = float(r_p.cov(r_b) / var_b) if var_b > 0 else 0.0
                alpha = float(r_p.mean() - beta * r_b.mean()) * trading_days  # 日 alpha 年化

                # 累计超额
                cum_strat = (1 + r_p).cumprod()
                cum_bench = (1 + r_b).cumprod()
                cum_excess = float(cum_strat.iloc[-1] - cum_bench.iloc[-1])

                bench_total_return = float(bench.iloc[-1] / bench.iloc[0] - 1)
                bench_annual = float((1 + bench_total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

                metrics["benchmark"] = {
                    "total_return": round(bench_total_return, 6),
                    "annual_return": round(bench_annual, 6),
                }
                metrics["excess"] = {
                    "tracking_error": round(te, 6),
                    "information_ratio": round(ir, 4),
                    "alpha": round(alpha, 6),
                    "beta": round(beta, 4),
                    "cumulative_excess_return": round(cum_excess, 6),
                }

    return metrics


def drawdown_series(equity: pd.Series) -> pd.Series:
    """回撤序列（用于绘图）"""
    cum_max = equity.cummax()
    return (equity - cum_max) / cum_max
