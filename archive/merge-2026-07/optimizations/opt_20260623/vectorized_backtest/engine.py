"""
向量化回测引擎（借鉴 VectorBT 的向量化回测思想）

核心优化点：
    现有 backtest-engine 的 native_adapter.py 通过 `for dt in dates` + iterrows()
    逐日事件驱动回测，在 1000+ 交易日 × 多股票场景下性能瓶颈明显。

    本模块针对"等权多头 + 定期调仓"这类常见研究场景，用矩阵运算替代事件循环，
    在保证 A 股 T+1、涨跌停、印花税等规则的前提下大幅提速。

借鉴来源：
    - VectorBT: Portfolio.from_signals 的向量化实现思路
    - Qlib: TopkDropoutStrategy 的截面选股 + 调仓逻辑

适用范围与限制：
    - 适合：等权多头、定期调仓、因子分组的快速研究回测
    - 不适合：复杂路径依赖策略（如金字塔加仓、条件止损链）——这类仍应用事件驱动

与现有 backtest-engine 的关系：
    不修改 main 分支代码；作为独立优化验证模块。
    输出与 native_adapter.run_backtest() 的 metrics 结构等价，可对比。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("vectorized-backtest")

# A 股交易成本默认值（与现有 config 对齐）
DEFAULT_COMMISSION_RATE = 0.00025
DEFAULT_STAMP_TAX_RATE = 0.001
DEFAULT_SLIPPAGE = 0.001
DEFAULT_INIT_CAPITAL = 1e6
TRADING_DAYS = 252


def _build_price_matrix(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将长表 OHLCV 转为宽表（date × code），便于矩阵运算。

    返回:
        close_matrix, is_limit_up_matrix, is_limit_down_matrix
    """
    close = data.pivot(index='date', columns='code', values='close').sort_index()
    limit_up = data.pivot(index='date', columns='code', values='is_limit_up').sort_index() \
        if 'is_limit_up' in data.columns else pd.DataFrame(False, index=close.index, columns=close.columns)
    limit_down = data.pivot(index='date', columns='code', values='is_limit_down').sort_index() \
        if 'is_limit_down' in data.columns else pd.DataFrame(False, index=close.index, columns=close.columns)
    return close, limit_up.fillna(False), limit_down.fillna(False)


def _signals_to_target_weights(
    signals: pd.DataFrame,
    close_matrix: pd.DataFrame,
    top_k: int = 20,
    equal_weight: bool = True,
) -> pd.DataFrame:
    """
    将信号转为目标权重矩阵（date × code）。

    策略：每个调仓日选 signal 最高的 top_k 只股票，等权配置。

    参数:
        signals: 含 code, date, signal 的 DataFrame
        close_matrix: 价格宽表（用于对齐列）
        top_k: 持仓数量
        equal_weight: True=等权

    返回:
        目标权重矩阵（date × code），未持仓为 0
    """
    sig_wide = signals.pivot(index='date', columns='code', values='signal').sort_index()
    # 对齐到 close_matrix 的日期与股票
    sig_wide = sig_wide.reindex(index=close_matrix.index, columns=close_matrix.columns)

    # 每日选 top_k（signal 最大的 k 只）
    # 用 rank 降序，取前 k
    ranks = sig_wide.rank(axis=1, method='first', ascending=False)
    held = ranks <= top_k

    if equal_weight:
        weights = held.astype(float) / held.sum(axis=1).replace(0, np.nan).values[:, None]
        weights = weights.fillna(0)
    else:
        weights = held.astype(float)
        row_sum = weights.sum(axis=1).replace(0, np.nan)
        weights = weights.div(row_sum, axis=0).fillna(0)

    return weights


def vectorized_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    init_capital: float = DEFAULT_INIT_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
    top_k: int = 20,
    rebalance_every: int = 1,
    t_plus_1: bool = True,
    price_limit: bool = True,
) -> Dict[str, Any]:
    """
    向量化回测（等权多头 + 定期调仓）。

    关键设计（借鉴 VectorBT 的信号对齐纪律，防前视偏差）：
        - t 日收盘生成信号 → t+1 日开盘/收盘执行（T+1 规则）
        - 买入价 = t+1 收盘 × (1 + slippage)
        - 卖出价 = t+1 收盘 × (1 - slippage)
        - 涨停日不买入、跌停日不卖出

    参数:
        data: OHLCV 长表，含 code, date, close, [is_limit_up, is_limit_down]
        signals: 含 code, date, signal 的 DataFrame
        init_capital: 初始资金
        top_k: 持仓数
        rebalance_every: 调仓频率（每 N 个交易日调仓一次）
        t_plus_1: 是否遵守 T+1
        price_limit: 是否遵守涨跌停限制

    返回:
        {
            "equity_curve": DataFrame(date, equity, cash, market_value, position_count),
            "trades": DataFrame,
            "metrics": Dict[str, float],
        }
    """
    if data.empty or signals.empty:
        return _empty_result()

    close, limit_up, limit_down = _build_price_matrix(data)
    target_weights = _signals_to_target_weights(signals, close, top_k=top_k, equal_weight=True)

    dates = close.index
    n_dates = len(dates)
    if n_dates < 2:
        return _empty_result()

    codes = close.columns
    n_codes = len(codes)

    # 前向填充价格（停牌时沿用前价），避免 NaN 导致计算中断
    close_ffill = close.ffill()

    # T+1：t 日信号 → t+1 日执行。把目标权重向后移 1 日
    if t_plus_1:
        exec_weights = target_weights.shift(1).fillna(0)
    else:
        exec_weights = target_weights.copy()

    # 调仓频率控制：仅在 rebalance 日更新权重，其余日保持持仓
    if rebalance_every > 1:
        mask = pd.Series(False, index=dates)
        mask.iloc[::rebalance_every] = True
        # 非调仓日权重设为 NaN，后续用 ffill 填充
        exec_weights = exec_weights.where(mask.values[:, None], np.nan).ffill().fillna(0)

    # 涨跌停限制：涨停日该股目标权重置 0（不买入），跌停日卖出权重仍允许（但实际成交受限）
    if price_limit:
        buy_blocked = limit_up.reindex_like(exec_weights).fillna(False)
        exec_weights = exec_weights.where(~buy_blocked, 0)

    # 向量化净值计算
    # 每日收益率矩阵：pct_change
    daily_returns = close_ffill.pct_change().fillna(0)

    # 权重制回测：跟踪组合净值，每日组合收益 = sum(持仓权重 × 个股收益) - 交易成本
    equity_records: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    prev_holdings = pd.Series(0.0, index=codes)
    equity = init_capital

    for i, dt in enumerate(dates):
        tgt = exec_weights.loc[dt]

        # 1) 持仓收益：用昨日持仓权重 × 今日个股收益
        if i > 0:
            port_return = (prev_holdings * daily_returns.loc[dt]).sum()
            equity *= (1 + port_return)

        # 2) 调仓成本：从 prev_holdings 调整到 tgt
        # 换手率 = sum(|tgt - prev|) / 2 （单边换手）
        turnover = (tgt - prev_holdings).abs().sum() / 2

        # 涨跌停限制
        if price_limit:
            # 涨停日不买入：tgt > prev 且涨停的部分不计入
            buy_blocked = limit_up.loc[dt].reindex(codes).fillna(False) if dt in limit_up.index else pd.Series(False, index=codes)
            effective_tgt = tgt.where(~((tgt > prev_holdings) & buy_blocked), prev_holdings)
            # 跌停日不卖出：prev > tgt 且跌停的部分不计入
            sell_blocked = limit_down.loc[dt].reindex(codes).fillna(False) if dt in limit_down.index else pd.Series(False, index=codes)
            effective_tgt = effective_tgt.where(~((prev_holdings > tgt) & sell_blocked), prev_holdings)
            turnover = (effective_tgt - prev_holdings).abs().sum() / 2
            tgt = effective_tgt

        # 交易成本：佣金（双边）+ 印花税（卖出）+ 滑点
        # 简化：成本率 ≈ turnover × (commission×2 + slippage×2 + stamp_tax)
        cost_rate = turnover * (commission_rate * 2 + slippage * 2 + stamp_tax_rate)
        cost = equity * cost_rate
        equity -= cost

        # 记录交易（汇总）
        if turnover > 0:
            trades.append({
                'date': dt, 'action': 'rebalance',
                'turnover': float(turnover),
                'amount': float(equity * turnover),
                'cost': float(cost),
            })

        # 更新持仓为目标权重
        prev_holdings = tgt.copy()

        position_count = int((prev_holdings > 0).sum())
        equity_records.append({
            'date': dt,
            'equity': float(equity),
            'cash': 0.0,  # 权重制下全仓投入
            'market_value': float(equity),
            'position_count': position_count,
        })

    equity_curve = pd.DataFrame(equity_records)

    if equity_curve.empty:
        return _empty_result()

    metrics = _calc_metrics(equity_curve['equity'])
    return {
        "equity_curve": equity_curve,
        "trades": pd.DataFrame(trades),
        "metrics": metrics,
        "backend": "vectorized",
    }


def _calc_metrics(equity: pd.Series, risk_free: float = 0.03) -> Dict[str, float]:
    """计算绩效指标（与现有 BaseBacktestMetrics 对齐）"""
    if len(equity) < 2:
        return {}
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {}
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n = len(returns)
    annual_return = (1 + total_return) ** (TRADING_DAYS / n) - 1 if n > 0 else 0
    volatility = returns.std() * np.sqrt(TRADING_DAYS)
    max_drawdown = (equity / equity.cummax() - 1).min()
    sharpe = (annual_return - risk_free) / volatility if volatility != 0 else 0
    win_rate = (returns > 0).mean() if n > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    # Sortino
    neg = returns[returns < 0]
    downside_std = neg.std() * np.sqrt(TRADING_DAYS) if len(neg) >= 2 else 0
    sortino = (annual_return - risk_free) / downside_std if downside_std != 0 else 0
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "calmar_ratio": float(calmar),
        "sortino_ratio": float(sortino),
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "equity_curve": pd.DataFrame(),
        "trades": pd.DataFrame(),
        "metrics": {},
        "backend": "vectorized",
    }


def benchmark_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    对比向量化回测与现有事件驱动回测的性能。

    返回性能与指标对比结果。
    """
    # 向量化版
    t0 = time.perf_counter()
    vec_res = vectorized_backtest(data, signals, top_k=top_k)
    t1 = time.perf_counter()
    vec_time = t1 - t0

    # 事件驱动版（复刻 native_adapter 逻辑，避免依赖 main 分支）
    t2 = time.perf_counter()
    loop_res = _event_driven_backtest_baseline(data, signals, top_k=top_k)
    t3 = time.perf_counter()
    loop_time = t3 - t2

    speedup = loop_time / vec_time if vec_time > 0 else float('inf')

    return {
        "vectorized_time_sec": round(vec_time, 4),
        "loop_time_sec": round(loop_time, 4),
        "speedup": round(speedup, 2),
        "vectorized_metrics": vec_res.get("metrics", {}),
        "loop_metrics": loop_res.get("metrics", {}),
        "n_dates": len(vec_res.get("equity_curve", [])),
    }


def _event_driven_backtest_baseline(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    top_k: int = 20,
    init_capital: float = DEFAULT_INIT_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
) -> Dict[str, Any]:
    """事件驱动回测基线（复刻 native_adapter 逻辑，用于性能对比）"""
    if data.empty or signals.empty:
        return _empty_result()

    data = data.sort_values(['date', 'code']).reset_index(drop=True)
    signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

    # 每日选 top_k
    signals['rank'] = signals.groupby('date')['signal'].rank(ascending=False, method='first')
    signals = signals[signals['rank'] <= top_k].copy()
    signals['signal'] = 1  # 选中的均为买入信号

    dates = sorted(signals['date'].unique())
    if not dates:
        return _empty_result()

    cash = init_capital
    positions: Dict[str, int] = {}
    equity_records: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    for dt in dates:
        day_signal = signals[signals['date'] == dt]
        day_data = data[data['date'] == dt]
        if day_data.empty:
            continue
        day_data_map = day_data.set_index('code')

        # 卖出不在当日信号的持仓
        sell_codes = [c for c in list(positions.keys()) if positions[c] > 0 and c not in set(day_signal['code'])]
        buy_codes = [c for c in day_signal['code'] if c in day_data_map.index]

        for code in sell_codes:
            if code not in day_data_map.index:
                continue
            price_row = day_data_map.loc[code]
            if price_row.get('is_limit_down', False):
                continue
            price = price_row['close']
            shares = positions[code]
            sell_amount = price * shares
            commission = max(sell_amount * commission_rate, 5)
            tax = sell_amount * stamp_tax_rate
            cash += sell_amount - commission - tax
            trades.append({'date': dt, 'code': code, 'action': 'sell', 'amount': sell_amount})
            positions[code] = 0

        if buy_codes:
            n_buy = len(buy_codes)
            budget_per_stock = cash * 0.95 / n_buy
            for code in buy_codes:
                price_row = day_data_map.loc[code]
                if price_row.get('is_limit_up', False):
                    continue
                price = price_row['close'] * (1 + slippage)
                shares = int(budget_per_stock / price / 100) * 100
                if shares <= 0:
                    continue
                buy_amount = price * shares
                commission = max(buy_amount * commission_rate, 5)
                cost = buy_amount + commission
                if cost > cash:
                    shares = int((cash * 0.98) / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * commission_rate, 5)
                    cost = buy_amount + commission
                cash -= cost
                positions[code] = positions.get(code, 0) + shares
                trades.append({'date': dt, 'code': code, 'action': 'buy', 'amount': buy_amount})

        market_value = sum(
            shares * day_data_map.loc[c, 'close']
            for c, shares in positions.items()
            if shares > 0 and c in day_data_map.index
        )
        equity_records.append({
            'date': dt,
            'equity': cash + market_value,
            'cash': cash,
            'market_value': market_value,
            'position_count': sum(1 for s in positions.values() if s > 0),
        })

    equity_curve = pd.DataFrame(equity_records)
    if equity_curve.empty:
        return _empty_result()

    return {
        "equity_curve": equity_curve,
        "trades": pd.DataFrame(trades),
        "metrics": _calc_metrics(equity_curve['equity']),
        "backend": "event_driven",
    }
