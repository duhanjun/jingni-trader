"""
增强绩效指标计算

借鉴来源：
- VectorBT 的 57 项绩效指标体系
- Qlib 的绩效分析模块
- NautilusTrader 的风险指标设计

对照 jingni-trader 现有实现：
- skills/backtest-engine/engine.py 的 _calc_metrics 方法
  仅计算 7 项基础指标：total_return, annual_return, volatility,
  sharpe_ratio, max_drawdown, win_rate, calmar_ratio

本模块新增指标：
- Sortino Ratio（下行风险调整收益）
- Information Ratio（相对基准的超额收益/跟踪误差）
- Beta / Alpha（CAPM 模型）
- Turnover Ratio（换手率）
- Exposure（持仓暴露度）
- Profit Factor（盈亏比）
- Win/Loss Ratio（单笔盈亏比）
- Max Drawdown Duration（最大回撤持续期）
- Annual Volatility / Monthly Return
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


# 年化因子（A股交易日）
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
RISK_FREE_RATE = 0.03  # 默认无风险利率 3%


def calc_enhanced_metrics(
    equity_curve: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """
    计算增强绩效指标

    参数:
        equity_curve: 净值曲线，包含 date, equity 列
        trades: 交易记录，包含 date, code, action, price, shares, amount, commission, tax
        benchmark: 基准收益率序列（日频）
        risk_free_rate: 年化无风险利率
        periods_per_year: 年化因子

    返回:
        包含所有指标的字典
    """
    if equity_curve.empty or 'equity' not in equity_curve.columns:
        return {}

    eq = equity_curve.set_index('date')['equity'].sort_index()
    if len(eq) < 2:
        return {}

    returns = eq.pct_change().dropna()
    n = len(returns)
    if n == 0:
        return {}

    # ---- 基础收益指标 ----
    cumulative = (1 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)
    years = n / periods_per_year
    annual_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

    # ---- 风险指标 ----
    volatility = float(returns.std() * np.sqrt(periods_per_year))
    downside_returns = returns[returns < 0]
    downside_deviation = float(
        np.sqrt((downside_returns ** 2).mean() * periods_per_year)
    ) if len(downside_returns) > 0 else 0.0

    # 最大回撤
    running_max = eq.cummax()
    drawdown = (eq / running_max - 1)
    max_drawdown = float(drawdown.min())

    # 最大回撤持续期（天数）
    in_drawdown = drawdown < 0
    max_dd_duration = 0
    current_dd = 0
    for d in in_drawdown:
        if d:
            current_dd += 1
            max_dd_duration = max(max_dd_duration, current_dd)
        else:
            current_dd = 0

    # ---- 风险调整收益指标 ----
    sharpe = float((annual_return - risk_free_rate) / volatility) if volatility > 0 else 0.0
    sortino = float((annual_return - risk_free_rate) / downside_deviation) if downside_deviation > 0 else 0.0
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # ---- 相对基准指标 ----
    info_ratio = 0.0
    beta = 0.0
    alpha = 0.0
    tracking_error = 0.0
    if benchmark is not None and len(benchmark) > 0:
        # 对齐日期
        aligned = pd.concat([returns, benchmark], axis=1, join='inner').dropna()
        if len(aligned) > 1:
            strat_ret = aligned.iloc[:, 0]
            bench_ret = aligned.iloc[:, 1]
            excess = strat_ret - bench_ret
            tracking_error = float(excess.std() * np.sqrt(periods_per_year))
            info_ratio = float(
                (excess.mean() * periods_per_year) / tracking_error
            ) if tracking_error > 0 else 0.0
            # Beta / Alpha (CAPM)
            cov_matrix = np.cov(strat_ret, bench_ret)
            var_bench = cov_matrix[1, 1]
            if var_bench > 0:
                beta = float(cov_matrix[0, 1] / var_bench)
                alpha = float(
                    (strat_ret.mean() - beta * bench_ret.mean()) * periods_per_year
                )

    # ---- 交易指标 ----
    win_rate = 0.0
    profit_factor = 0.0
    win_loss_ratio = 0.0
    n_trades = 0
    turnover = 0.0
    avg_trade_return = 0.0

    if trades is not None and not trades.empty:
        # 计算每笔交易的盈亏（配对买卖）
        pnl_list = _compute_trade_pnl(trades)
        if pnl_list:
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]
            n_trades = len(pnl_list)
            win_rate = float(len(wins) / n_trades) if n_trades > 0 else 0.0
            gross_profit = float(sum(wins))
            gross_loss = float(abs(sum(losses)))
            profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
            avg_win = float(np.mean(wins)) if wins else 0.0
            avg_loss = float(abs(np.mean(losses))) if losses else 0.0
            win_loss_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 0.0
            avg_trade_return = float(np.mean(pnl_list))

        # 换手率（单边）
        buy_amount = trades[trades['action'] == 'buy']['amount'].sum() if 'action' in trades.columns else 0
        avg_capital = eq.mean()
        turnover = float(buy_amount / avg_capital / years) if avg_capital > 0 and years > 0 else 0.0

    # ---- 月度收益 ----
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    best_month = float(monthly_returns.max()) if len(monthly_returns) > 0 else 0.0
    worst_month = float(monthly_returns.min()) if len(monthly_returns) > 0 else 0.0
    positive_months = float((monthly_returns > 0).mean()) if len(monthly_returns) > 0 else 0.0

    return {
        # ---- 收益指标 ----
        "total_return": total_return,
        "annual_return": annual_return,
        "best_month": best_month,
        "worst_month": worst_month,
        "positive_month_ratio": positive_months,
        # ---- 风险指标 ----
        "volatility": volatility,
        "downside_deviation": downside_deviation,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_days": int(max_dd_duration),
        # ---- 风险调整收益 ----
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        # ---- 相对基准 ----
        "information_ratio": info_ratio,
        "beta": beta,
        "alpha": alpha,
        "tracking_error": tracking_error,
        # ---- 交易指标 ----
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "win_loss_ratio": win_loss_ratio,
        "avg_trade_pnl": avg_trade_return,
        "turnover_per_year": turnover,
    }


def _compute_trade_pnl(trades: pd.DataFrame) -> list:
    """配对计算每笔完整交易的盈亏"""
    if trades.empty or 'code' not in trades.columns:
        return []

    pnl_list = []
    # 按股票代码分组，按时间排序，配对买卖
    for code, group in trades.sort_values('date').groupby('code'):
        position_shares = 0
        cost_basis = 0.0  # 累计买入成本（含手续费）

        for _, t in group.iterrows():
            action = t.get('action', '')
            shares = int(t.get('shares', 0))
            amount = float(t.get('amount', 0))
            commission = float(t.get('commission', 0))
            tax = float(t.get('tax', 0))

            if action == 'buy':
                position_shares += shares
                cost_basis += amount + commission
            elif action == 'sell':
                if position_shares <= 0:
                    continue
                # 按比例计算本次卖出对应的成本
                ratio = shares / position_shares if position_shares > 0 else 0
                cost_portion = cost_basis * ratio
                net_proceeds = amount - commission - tax
                pnl = net_proceeds - cost_portion
                pnl_list.append(pnl)
                position_shares -= shares
                cost_basis -= cost_portion

    return pnl_list


def calc_basic_metrics(equity_curve: pd.DataFrame, init_capital: float = 1e6) -> Dict[str, float]:
    """
    复刻 jingni-trader 现有的基础指标计算（用于对比验证）

    对照: skills/backtest-engine/engine.py 的 _calc_metrics 方法
    """
    if equity_curve.empty or 'equity' not in equity_curve.columns:
        return {}
    eq = equity_curve.set_index('date')['equity']
    if len(eq) < 2:
        return {}
    returns = eq.pct_change().dropna()
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    max_drawdown = (eq / eq.cummax() - 1).min()
    sharpe = (annual_return - 0.03) / volatility if volatility != 0 else 0
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "calmar_ratio": float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0,
    }
