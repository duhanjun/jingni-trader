"""
回测指标修正与扩展 - 优化验证模块

借鉴来源:
  - empyrical (Quantopian): 经典绩效指标实现
  - quantstats: 详细绩效报告
  - pyfolio: Tear Sheet 分析

针对 jingni-trader 现有 base_backtest.py 的问题:
  1. calc_annual_return 用 total_return ** (1/n_years) - 1, 当 equity 起点不是 init_capital 时会出错
  2. calc_sharpe 用 returns.mean() * trading_days 计算年化收益, 与 calc_annual_return 不一致
  3. calc_win_rate 基于 buy/sell 单笔 pnl, 但 buy 的 pnl 是负的 (买入支出现金), 胜率无意义
  4. 缺少 Information Ratio, Tracking Error, Benchmark 对比
  5. 缺少分月收益、滚动 Sharpe、盈亏比等

本模块实现:
  - 统一的年化算法 (几何年化, 与 total_return 一致)
  - Sharpe/Sortino/Calmar 一致性
  - Information Ratio (相对基准)
  - Tracking Error
  - 分月收益
  - 滚动 Sharpe
  - 盈亏比 (profit_loss_ratio)
  - 基于 trade pair 的真实胜率 (买入-卖出配对)

注意: 本文件仅用于优化验证, 不修改 main 分支任何代码。
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法"""
    if denominator == 0 or np.isnan(denominator):
        return default
    return float(numerator / denominator)


def total_return(equity: pd.Series) -> float:
    """累计收益率 = 末值/首值 - 1"""
    if len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start == 0:
        return 0.0
    return end / start - 1.0


def annual_return(equity: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """
    年化收益率 (几何)

    修正点: 用 (end/start) ** (252/n_days) - 1
    与 total_return 保持一致, 避免算术/几何年化混用
    """
    if len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    n_years = len(equity) / trading_days
    if n_years <= 0:
        return 0.0
    return (end / start) ** (1.0 / n_years) - 1.0


def annual_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """年化波动率"""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    equity: pd.Series,
    risk_free: float = 0.03,
    trading_days: int = TRADING_DAYS,
) -> float:
    """
    Sharpe 比率

    修正点: 用几何年化收益 - 无风险利率, 再除以年化波动率
    与 annual_return 保持一致
    """
    if len(equity) < 2:
        return 0.0
    rets = equity.pct_change().dropna()
    vol = annual_volatility(rets, trading_days)
    if vol == 0:
        return 0.0
    ann_ret = annual_return(equity, trading_days)
    return safe_div(ann_ret - risk_free, vol)


def sortino_ratio(
    equity: pd.Series,
    risk_free: float = 0.03,
    trading_days: int = TRADING_DAYS,
) -> float:
    """
    Sortino 比率 (下行风险调整)

    修正点: 用几何年化收益, 与 Sharpe 一致
    """
    if len(equity) < 2:
        return 0.0
    rets = equity.pct_change().dropna()
    neg = rets[rets < 0]
    if len(neg) < 2:
        return 0.0
    downside_std = float(neg.std(ddof=1) * np.sqrt(trading_days))
    if downside_std == 0:
        return 0.0
    ann_ret = annual_return(equity, trading_days)
    return safe_div(ann_ret - risk_free, downside_std)


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤 (负值)"""
    if len(equity) < 2:
        return 0.0
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Calmar 比率 = 年化收益 / |最大回撤|"""
    ann_ret = annual_return(equity, trading_days)
    mdd = abs(max_drawdown(equity))
    return safe_div(ann_ret, mdd)


def tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = TRADING_DAYS,
) -> float:
    """跟踪误差 (年化)"""
    aligned = pd.concat([returns, benchmark_returns], axis=1, keys=["r", "b"]).dropna()
    if len(aligned) < 2:
        return 0.0
    excess = aligned["r"] - aligned["b"]
    return float(excess.std(ddof=1) * np.sqrt(trading_days))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = TRADING_DAYS,
) -> float:
    """
    信息比率 = (策略收益 - 基准收益) / 跟踪误差

    修正点: 用日均超额收益 * 252 / 年化跟踪误差
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1, keys=["r", "b"]).dropna()
    if len(aligned) < 2:
        return 0.0
    excess = aligned["r"] - aligned["b"]
    te = tracking_error(returns, benchmark_returns, trading_days)
    if te == 0:
        return 0.0
    ann_excess = float(excess.mean() * trading_days)
    return safe_div(ann_excess, te)


def monthly_returns(equity: pd.Series) -> pd.Series:
    """分月收益率"""
    if len(equity) < 2:
        return pd.Series(dtype=float)
    monthly = equity.resample("ME").last()
    return monthly.pct_change().dropna()


def rolling_sharpe(
    equity: pd.Series,
    window: int = 63,
    trading_days: int = TRADING_DAYS,
    risk_free: float = 0.03,
) -> pd.Series:
    """滚动 Sharpe (window 日窗口)"""
    if len(equity) < window + 1:
        return pd.Series(dtype=float)
    rets = equity.pct_change().dropna()
    rolling_mean = rets.rolling(window).mean() * trading_days
    rolling_std = rets.rolling(window).std() * np.sqrt(trading_days)
    return ((rolling_mean - risk_free) / rolling_std.replace(0, np.nan)).dropna()


def win_rate_from_trades(trades: pd.DataFrame) -> float:
    """
    基于 trade pair 的真实胜率

    修正点: 把 buy/sell 配对, 计算每次完整交易的盈亏
    买入记录成本, 卖出回收现金, 盈亏 = 卖出收入 - 买入成本
    """
    if trades.empty or "action" not in trades.columns:
        return 0.0

    # 按代码分组, 配对 buy/sell
    profits = []
    for code, group in trades.sort_values(["date"]).groupby("code"):
        if "action" not in group.columns:
            continue
        buy_stack = []  # [(price, shares), ...]
        for _, row in group.iterrows():
            action = row["action"]
            price = float(row["price"])
            shares = int(row["shares"])
            if action == "buy":
                buy_stack.append([price, shares])
            elif action == "sell" and buy_stack:
                # FIFO 配对
                remaining = shares
                while remaining > 0 and buy_stack:
                    buy_price, buy_shares = buy_stack[0]
                    matched = min(remaining, buy_shares)
                    # 简化: 不计手续费 (已在 trades 中扣除)
                    profit = (price - buy_price) * matched
                    profits.append(profit)
                    buy_shares -= matched
                    remaining -= matched
                    if buy_shares <= 0:
                        buy_stack.pop(0)
                    else:
                        buy_stack[0][1] = buy_shares

    if not profits:
        return 0.0
    profits_arr = np.array(profits)
    return float((profits_arr > 0).mean())


def profit_loss_ratio(trades: pd.DataFrame) -> float:
    """盈亏比 = 平均盈利 / 平均亏损"""
    if trades.empty:
        return 0.0
    profits = []
    for code, group in trades.sort_values(["date"]).groupby("code"):
        buy_stack = []
        for _, row in group.iterrows():
            action = row["action"]
            price = float(row["price"])
            shares = int(row["shares"])
            if action == "buy":
                buy_stack.append([price, shares])
            elif action == "sell" and buy_stack:
                remaining = shares
                while remaining > 0 and buy_stack:
                    buy_price, buy_shares = buy_stack[0]
                    matched = min(remaining, buy_shares)
                    profit = (price - buy_price) * matched
                    profits.append(profit)
                    buy_shares -= matched
                    remaining -= matched
                    if buy_shares <= 0:
                        buy_stack.pop(0)
                    else:
                        buy_stack[0][1] = buy_shares
    if not profits:
        return 0.0
    profits_arr = np.array(profits)
    wins = profits_arr[profits_arr > 0]
    losses = profits_arr[profits_arr < 0]
    if len(losses) == 0 or losses.mean() == 0:
        return float("inf") if len(wins) > 0 else 0.0
    return float(abs(wins.mean() / losses.mean()))


def calc_all_metrics(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, Any]:
    """
    一次性计算所有绩效指标 (修正版)

    参数:
        equity: 净值序列 (index=date)
        trades: 成交记录 (可选)
        benchmark: 基准净值序列 (可选)
        risk_free: 无风险利率
        trading_days: 年交易日
    """
    if len(equity) < 2:
        return {}

    rets = equity.pct_change().dropna()

    metrics: Dict[str, Any] = {
        "total_return": total_return(equity),
        "annual_return": annual_return(equity, trading_days),
        "annual_volatility": annual_volatility(rets, trading_days),
        "sharpe_ratio": sharpe_ratio(equity, risk_free, trading_days),
        "sortino_ratio": sortino_ratio(equity, risk_free, trading_days),
        "max_drawdown": max_drawdown(equity),
        "calmar_ratio": calmar_ratio(equity, trading_days),
        "daily_win_rate": float((rets > 0).mean()) if len(rets) > 0 else 0.0,
        "n_days": len(equity),
    }

    # 基准对比指标
    if benchmark is not None and len(benchmark) >= 2:
        bench_rets = benchmark.pct_change().dropna()
        metrics["benchmark_total_return"] = total_return(benchmark)
        metrics["benchmark_annual_return"] = annual_return(benchmark, trading_days)
        metrics["tracking_error"] = tracking_error(rets, bench_rets, trading_days)
        metrics["information_ratio"] = information_ratio(rets, bench_rets, trading_days)
        metrics["excess_return"] = metrics["annual_return"] - metrics["benchmark_annual_return"]

    # 交易指标
    if trades is not None and not trades.empty:
        metrics["n_trades"] = len(trades)
        metrics["n_buys"] = int((trades["action"] == "buy").sum())
        metrics["n_sells"] = int((trades["action"] == "sell").sum())
        metrics["trade_win_rate"] = win_rate_from_trades(trades)
        metrics["profit_loss_ratio"] = profit_loss_ratio(trades)
        sell_amount = float(trades.loc[trades["action"] == "sell", "amount"].sum())
        avg_equity = float(equity.mean())
        metrics["turnover"] = safe_div(sell_amount, avg_equity)

    # 分月收益
    monthly = monthly_returns(equity)
    if len(monthly) > 0:
        metrics["monthly_mean_return"] = float(monthly.mean())
        metrics["monthly_std"] = float(monthly.std())
        metrics["monthly_win_rate"] = float((monthly > 0).mean())
        metrics["best_month"] = float(monthly.max())
        metrics["worst_month"] = float(monthly.min())

    return metrics
