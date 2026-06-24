"""
胜率计算修复（验证用）
======================
OPTIMIZATION 3 (part 2): 修复 BaseBacktestMetrics.calc_win_rate 的 bug。

原始 bug（backtest-engine base/base_backtest.py）:
    winning = (trades["pnl"] > 0).sum()
    total = len(trades)
    return winning / total
问题：把“买入”成交也计入胜率分母。买入 trade 的 pnl = -buy_amount - commission 恒为负，
导致胜率被严重低估。

修复思路（借鉴 QuantConnect LEAN / 主流回测框架的 round-trip 统计）：
胜率应只统计“卖出”成交（action == 'sell'），即按平仓盈亏计算：
    win_rate = (卖出且 pnl>0 的笔数) / (卖出总笔数)

提供：
- calc_win_rate_original:  复刻原始有 bug 的实现（对比基准）
- calc_win_rate_corrected: 仅统计 sell 成交的修正实现
"""
from __future__ import annotations
import pandas as pd


def calc_win_rate_original(trades: pd.DataFrame) -> float:
    """
    原始（有 bug）胜率：统计全部 trade（含买入）。
    买入 pnl 恒为负 -> 胜率被低估。
    """
    if trades is None or trades.empty:
        return 0.0
    winning = (trades["pnl"] > 0).sum()
    total = len(trades)
    return float(winning / total) if total > 0 else 0.0


def calc_win_rate_corrected(trades: pd.DataFrame) -> float:
    """
    修正胜率：仅统计 action == 'sell' 的成交（平仓盈亏）。
    win_rate = (sell 且 pnl>0 笔数) / (sell 总笔数)
    """
    if trades is None or trades.empty:
        return 0.0
    if "action" not in trades.columns:
        # 无 action 列时退化为原始口径
        return calc_win_rate_original(trades)
    sells = trades[trades["action"] == "sell"]
    if sells.empty:
        return 0.0
    winning = (sells["pnl"] > 0).sum()
    total = len(sells)
    return float(winning / total) if total > 0 else 0.0


if __name__ == "__main__":
    # 构造合成 trades：3 笔买入(pnl恒负) + 4 笔卖出(2 正 2 负)
    trades = pd.DataFrame([
        {"action": "buy", "pnl": -10050},
        {"action": "buy", "pnl": -20300},
        {"action": "buy", "pnl": -5100},
        {"action": "sell", "pnl": 1200},
        {"action": "sell", "pnl": -800},
        {"action": "sell", "pnl": 3500},
        {"action": "sell", "pnl": -200},
    ])
    print("original (bug):", calc_win_rate_original(trades))      # 2/7 ≈ 0.2857
    print("corrected     :", calc_win_rate_corrected(trades))     # 2/4 = 0.5
