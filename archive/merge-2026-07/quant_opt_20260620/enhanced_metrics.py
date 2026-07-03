"""
增强绩效指标模块

借鉴来源:
- VectorBT: Sortino / Profit Factor / Turnover / Information Ratio 的向量化计算公式
- 通用量化实践: 补充 jingni-trader 现有 _calc_metrics 缺失的指标

对照原实现 skills/backtest-engine/engine.py 的 _calc_metrics (L84-107):
  原实现仅含: total_return, annual_return, volatility, sharpe, max_drawdown,
              win_rate, calmar
  缺失: sortino, profit_factor, information_ratio, avg_turnover, longest_dd_days

本模块提供增强版指标, 可独立应用于任意 equity_curve, 不修改原代码。
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_enhanced_metrics(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    trades: Optional[pd.DataFrame] = None,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    计算增强版绩效指标

    参数:
        equity: 净值曲线 (index=date, value=equity)
        benchmark: 基准净值曲线 (可选, 用于信息比率)
        trades: 成交记录 (可选, 用于换手率/盈亏比)
        risk_free: 无风险利率(年化)
        trading_days: 年交易日
    """
    if equity is None or len(equity) < 2:
        return {}
    equity = equity.sort_index()
    returns = equity.pct_change().dropna()
    if len(returns) == 0:
        return {}

    ann_factor = trading_days
    n_years = len(equity) / trading_days

    # ── 基础收益指标 ──
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    annual_return = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    volatility = float(returns.std() * np.sqrt(ann_factor))

    # ── 风险调整指标 ──
    sharpe = float((returns.mean() * ann_factor - risk_free) / volatility) if volatility > 0 else 0.0

    # Sortino: 仅对下行收益计算标准差
    downside = returns[returns < 0]
    downside_std = float(downside.std() * np.sqrt(ann_factor)) if len(downside) > 1 else 0.0
    sortino = float((returns.mean() * ann_factor - risk_free) / downside_std) if downside_std > 0 else 0.0

    # ── 回撤指标 ──
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # 最长回撤持续天数
    in_dd = drawdown < 0
    longest_dd_days = 0
    cur = 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        longest_dd_days = max(longest_dd_days, cur)

    # ── 盈亏指标 ──
    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    profit_factor = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
    win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "longest_drawdown_days": int(longest_dd_days),
    }

    # ── 信息比率 (相对基准) ──
    if benchmark is not None and len(benchmark) >= 2:
        bench = benchmark.sort_index().pct_change().dropna()
        # 对齐
        common = returns.index.intersection(bench.index)
        if len(common) > 1:
            excess = returns.loc[common] - bench.loc[common]
            tracking_err = float(excess.std() * np.sqrt(ann_factor))
            ir = float((excess.mean() * ann_factor) / tracking_err) if tracking_err > 0 else 0.0
            metrics["information_ratio"] = ir
            metrics["tracking_error"] = tracking_err
            metrics["excess_return_annual"] = float(excess.mean() * ann_factor)

    # ── 换手率 (基于成交记录) ──
    if trades is not None and not trades.empty and "amount" in trades.columns:
        # 简化: 双边换手 = sum(成交额) / 平均资产
        avg_equity = float(equity.mean())
        if avg_equity > 0:
            total_amount = float(trades["amount"].sum())
            metrics["total_turnover_ratio"] = float(total_amount / avg_equity)
            metrics["avg_turnover_per_day"] = float(total_amount / len(equity))

    return metrics


def compare_metrics(baseline: Dict[str, float], enhanced: Dict[str, float]) -> pd.DataFrame:
    """对比两套指标, 输出表格"""
    all_keys = sorted(set(list(baseline.keys()) + list(enhanced.keys())))
    rows = []
    for k in all_keys:
        rows.append({
            "metric": k,
            "baseline": baseline.get(k, "—"),
            "enhanced": enhanced.get(k, "—"),
            "new_in_enhanced": k not in baseline,
        })
    return pd.DataFrame(rows)
