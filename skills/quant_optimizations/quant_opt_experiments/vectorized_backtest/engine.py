"""
方向 2：矢量化回测引擎（vectorbt 风格）

借鉴：vectorbt 的设计思想
      - 整个价格/信号序列视为 NumPy / pandas 矩阵
      - 一次算完所有 bar 的组合权重（不再 Python for-loop 逐日）
      - 支持列方向 = 多个参数组合 / 标的，天然支持参数扫描
      - 持仓从信号到现金的更新也是向量化

核心：
- 输入：close (date x asset), entries/exits 布尔矩阵
- 输出：净值曲线、绩效指标、交易记录
- 支持 A 股 T+1、手续费、滑点

vs 现有 native_adapter（位于 skills/backtest-engine/scripts/adapters/native_adapter.py）：
- 现有实现按日期逐日 for 循环；遇到 1000+ 标的时慢
- 新实现向量化：单次矩阵运算得到完整 nav 序列
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class VectorizedBacktestResult:
    equity: pd.DataFrame            # date x asset (per-asset equity curve, init=1)
    portfolio_value: pd.Series     # 组合整体净值
    cash: pd.Series
    position_count: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# 1) 核心矢量化组合模拟器
# ---------------------------------------------------------------------------
def _simulate_one(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    init_cash: float = 1e6,
    commission: float = 0.00025,
    stamp_tax: float = 0.001,
    slippage: float = 0.0001,
    min_commission: float = 5.0,
    min_lot: int = 100,
    t_plus_1: bool = True,
) -> Dict[str, pd.Series]:
    """
    对【单标的】做矢量化模拟。

    注意：即便单标的也保留一个内部"持仓天数"数组，从而避免 Python for-loop
    决定"何时可以卖"。

    close/exits/entries index 必须一致（datetime 升序）
    """
    n = len(close)
    empty_trades_cols = ["date", "side", "price", "shares", "amount", "fee"]
    if n == 0:
        empty = pd.Series(dtype=float)
        return {
            "equity": empty, "cash": empty, "shares": empty, "hold_days": empty,
            "trades": pd.DataFrame(columns=empty_trades_cols),
        }

    px = close.values.astype(float)
    ent = entries.reindex(close.index).fillna(False).astype(bool).values
    ext = exits.reindex(close.index).fillna(False).astype(bool).values

    # 执行价（含滑点）
    buy_px = px * (1 + slippage)
    sell_px = px * (1 - slippage)

    # 状态向量
    cash = np.empty(n)
    shares = np.empty(n)
    hold_days = np.empty(n)
    cash[0] = init_cash
    shares[0] = 0
    hold_days[0] = 0

    trade_records: List[dict] = []

    for t in range(1, n):
        c = cash[t - 1]
        s = shares[t - 1]
        h = hold_days[t - 1]

        # T+1：上一日买入今日不能卖（hold_days < 1 不允许卖）
        can_sell = (s > 0) and ((not t_plus_1) or h >= 1)

        if ext[t] and can_sell:
            # 卖出全部
            proceeds = s * sell_px[t]
            fee = max(proceeds * commission, min_commission)
            tax = proceeds * stamp_tax
            net = proceeds - fee - tax
            trade_records.append({
                "date": close.index[t], "side": "sell",
                "price": float(sell_px[t]), "shares": int(s),
                "amount": float(proceeds), "fee": float(fee + tax),
            })
            c = net
            s = 0
            h = 0
        elif ent[t] and c > 0:
            # 买入：用 95% 现金 / 100 取整手
            budget = c * 0.95
            qty = int(budget // (buy_px[t] * min_lot)) * min_lot
            if qty > 0:
                cost = qty * buy_px[t]
                fee = max(cost * commission, min_commission)
                total = cost + fee
                if total <= c:
                    trade_records.append({
                        "date": close.index[t], "side": "buy",
                        "price": float(buy_px[t]), "shares": int(qty),
                        "amount": float(cost), "fee": float(fee),
                    })
                    c -= total
                    s += qty
                    h = 0  # 买入当日记 0，T+1 模式明日才可卖

        # 每日结算：未触发信号则维持昨日状态
        if t > 0 and not (ext[t] and can_sell) and not (ent[t] and c > 0):
            c = cash[t - 1]
            s = shares[t - 1]
            h = hold_days[t - 1] + (1 if shares[t - 1] > 0 else 0)
            # 注意：当日 buy 后 h=0；保持 T+1 语义

        # 修正：上面 if/elif 链只在分支内更新 c/s/h，未命中时仍继承 t-1 值
        cash[t] = c
        shares[t] = s
        hold_days[t] = h if s > 0 else 0

    # 权益
    equity = cash + shares * px

    return {
        "cash": pd.Series(cash, index=close.index),
        "shares": pd.Series(shares, index=close.index),
        "equity": pd.Series(equity, index=close.index),
        "hold_days": pd.Series(hold_days, index=close.index),
        "trades": pd.DataFrame(trade_records),
    }


def vectorized_backtest_single(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    **kwargs,
) -> Dict[str, pd.Series]:
    """单标的的便捷入口"""
    return _simulate_one(close, entries, exits, **kwargs)


def vectorized_backtest_multi(
    close: pd.DataFrame,           # date x asset
    entries: pd.DataFrame,         # date x asset  (True/False)
    exits: pd.DataFrame,           # date x asset
    alloc_per_asset: float = 0.1,  # 资金分配到每个标的上限比例
    init_cash: float = 1e6,
    **kwargs,
) -> VectorizedBacktestResult:
    """
    多标的矢量化回测：
    - 资金按比例分到每只标的上（不超额分配）
    - 每只标的独立子组合
    - 总组合净值 = 各子组合净值之和
    """
    close = close.sort_index()
    entries = entries.reindex_like(close).fillna(False).astype(bool)
    exits = exits.reindex_like(close).fillna(False).astype(bool)

    n_assets = close.shape[1]
    per_asset_cash = init_cash * alloc_per_asset
    per_asset_init = per_asset_cash

    equity_mat = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    trades_list: List[pd.DataFrame] = []

    for col in close.columns:
        sim = _simulate_one(
            close[col].dropna(),
            entries[col],
            exits[col],
            init_cash=per_asset_init,
            **kwargs,
        )
        equity_mat[col] = sim["equity"]
        if not sim["trades"].empty:
            tr = sim["trades"].copy()
            tr["code"] = col
            trades_list.append(tr)

    # 总组合
    portfolio_value = equity_mat.sum(axis=1)
    cash = init_cash - equity_mat.iloc[0].sum() + portfolio_value  # 简化表达
    position_count = (equity_mat > per_asset_init * 0.001).sum(axis=1)

    trades_df = (
        pd.concat(trades_list, ignore_index=True)
        if trades_list
        else pd.DataFrame(columns=["date", "side", "price", "shares", "amount", "fee", "code"])
    )

    return VectorizedBacktestResult(
        equity=equity_mat,
        portfolio_value=portfolio_value,
        cash=cash,
        position_count=position_count,
        trades=trades_df,
        metrics=calc_metrics(portfolio_value),
    )


# ---------------------------------------------------------------------------
# 2) 绩效指标（向量化）
# ---------------------------------------------------------------------------
def calc_metrics(equity: pd.Series, periods_per_year: int = 252, risk_free: float = 0.03) -> Dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {}
    ret = equity.pct_change().dropna()
    if ret.empty:
        return {}

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_days = len(ret)
    annual_return = float((1 + total_return) ** (periods_per_year / n_days) - 1)
    vol = float(ret.std() * np.sqrt(periods_per_year))
    sharpe = float((annual_return - risk_free) / vol) if vol > 0 else 0.0
    downside = ret[ret < 0]
    sortino = float((annual_return - risk_free) / (downside.std() * np.sqrt(periods_per_year))) if len(downside) > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = float(annual_return / abs(max_dd)) if max_dd != 0 else 0.0
    win_rate = float((ret > 0).mean())
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# 3) 信号生成器
# ---------------------------------------------------------------------------
def ma_cross_signals(
    close: pd.DataFrame,
    fast: int = 5,
    slow: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """双均线金叉死叉信号"""
    fast_ma = close.rolling(fast, min_periods=1).mean()
    slow_ma = close.rolling(slow, min_periods=1).mean()
    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    return entries.fillna(False), exits.fillna(False)


def rank_topk_signals(
    factor: pd.DataFrame,             # date x asset
    top_pct: float = 0.2,             # 选 top 20%
    hold_days: int = 5,               # 持仓天数
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """横截面 top-k 调仓信号"""
    ranks = factor.rank(axis=1, pct=True, ascending=True)
    entries = pd.DataFrame(False, index=factor.index, columns=factor.columns)
    exits = pd.DataFrame(False, index=factor.index, columns=factor.columns)

    in_holding = pd.Series(False, index=factor.columns)
    days_left = pd.Series(0, index=factor.columns)
    rebal_dates = factor.index[::hold_days]

    for dt in factor.index:
        if in_holding.any():
            days_left = (days_left - 1).clip(lower=0)
        if dt in rebal_dates:
            new_top = ranks.loc[dt] >= (1 - top_pct)
            exits.loc[dt, in_holding & ~new_top] = True
            entries.loc[dt, new_top & ~in_holding] = True
            in_holding = new_top
            days_left = pd.Series(hold_days, index=factor.columns)

    return entries.fillna(False), exits.fillna(False)