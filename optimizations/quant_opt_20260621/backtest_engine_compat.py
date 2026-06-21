"""
兼容层: 复用 main 分支已有的 BaseBacktestMetrics, 不修改原代码

通过 sys.path 注入, 直接 import 现有 skills/backtest-engine 的绩效计算逻辑,
保证向量化回测与 native_adapter 使用同一套指标计算口径, 便于对比.
"""
import os
import sys
from typing import Dict, Any

import pandas as pd

# 把项目根目录加入 path, 以便 import skills.backtest-engine 的模块
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 注入 backtest-engine 的 scripts 路径
_BT_ENGINE_SCRIPTS = os.path.join(_PROJECT_ROOT, "skills", "backtest-engine", "scripts")
if _BT_ENGINE_SCRIPTS not in sys.path:
    sys.path.insert(0, _BT_ENGINE_SCRIPTS)


def calc_all_metrics_compat(equity_curve: pd.Series, trades: pd.DataFrame,
                             risk_free: float = 0.03, trading_days: int = 252) -> Dict[str, Any]:
    """
    调用 main 分支的 BaseBacktestMetrics.calc_all_metrics

    保证向量化回测与 native_adapter 指标口径完全一致.
    """
    try:
        from base.base_backtest import BaseBacktestMetrics
        return BaseBacktestMetrics.calc_all_metrics(equity_curve, trades, risk_free, trading_days)
    except ImportError:
        # 降级: 内置最小实现 (与原实现保持一致)
        return _fallback_metrics(equity_curve, trades, risk_free, trading_days)


def _fallback_metrics(equity_curve: pd.Series, trades: pd.DataFrame,
                       risk_free: float = 0.03, trading_days: int = 252) -> Dict[str, Any]:
    """降级绩效计算 (与 BaseBacktestMetrics 算法一致)"""
    import numpy as np
    from datetime import datetime

    if len(equity_curve) < 2:
        return {}

    returns = equity_curve.pct_change().dropna()

    # 累计收益
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    # 年化收益
    n_years = len(equity_curve) / trading_days
    annual_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # 波动率
    volatility = float(returns.std() * np.sqrt(trading_days)) if len(returns) >= 2 else 0.0

    # 夏普
    sharpe = float((returns.mean() * trading_days - risk_free) / volatility) if volatility > 0 else 0.0

    # 最大回撤
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    max_dd = float(drawdown.min())

    # Calmar
    calmar = float(annual_return / abs(max_dd)) if max_dd != 0 else 0.0

    # Sortino
    neg_returns = returns[returns < 0]
    downside_std = float(neg_returns.std() * np.sqrt(trading_days)) if len(neg_returns) >= 2 else 0.0
    sortino = float((returns.mean() * trading_days - risk_free) / downside_std) if downside_std > 0 else 0.0

    # 胜率
    if not trades.empty and "pnl" in trades.columns:
        win_rate = float((trades["pnl"] > 0).sum() / len(trades))
    else:
        win_rate = 0.0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "sortino_ratio": sortino,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
