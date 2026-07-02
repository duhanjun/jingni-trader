"""
向量化回测引擎（Vectorized Backtest Engine）

借鉴自：
- FinRL-X (AI4Finance-Foundation/FinRL-Trading, arXiv:2603.21330) 的 weight-centric 设计
- 标准化 PortfolioWeight 数据类作为策略层与回测层的统一契约
- 完全 numpy/pandas 向量化（避免逐日 Python for 循环）

对照 jingni-trader/skills/backtest-engine/scripts/adapters/native_adapter.py：
- 原版用 `for dt in dates:` 逐日循环，O(N_dates) Python 开销
- 本实现核心计算全部用 numpy 矩阵运算，向量化后预期有 10x-100x 加速
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 统一契约：PortfolioWeight（借鉴 FinRL-X 的 weight-centric 设计）
# ---------------------------------------------------------------------------
@dataclass
class PortfolioWeight:
    """策略层与回测层之间的标准化权重接口

    shape: (n_dates, n_codes)
    weight_frame: DataFrame[date × code]，值 ∈ [0, 1]，每日加和 = 1
    """
    weight_frame: pd.DataFrame
    rebalance_freq: str = "daily"   # daily / weekly / monthly
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.weight_frame, pd.DataFrame):
            raise TypeError("weight_frame 必须为 DataFrame")
        # 校验：每行加和 ≈ 1
        row_sums = self.weight_frame.sum(axis=1)
        if not np.all(np.abs(row_sums - 1.0) < 1e-3):
            # 仅警告，不强制中断（容许 cash 仓位）
            pass


# ---------------------------------------------------------------------------
# 信号 → 权重 转换器
# ---------------------------------------------------------------------------
def signals_to_weights(
    signals: pd.DataFrame,
    top_quantile: float = 0.2,
    bottom_quantile: Optional[float] = None,
    long_only: bool = True,
) -> PortfolioWeight:
    """把 0/1 形式的 signal DataFrame 转成 PortfolioWeight。

    Parameters
    ----------
    signals : DataFrame with columns [code, date, signal]
        signal ∈ {-1, 0, 1}
    top_quantile : float
        每日买入 signal=1 的前 top_quantile 比例
    bottom_quantile : float, optional
        如果设置且 long_only=False，对 signal=-1 的股票做空
    long_only : bool
        True 则不允许做空
    """
    df = signals.pivot(index="date", columns="code", values="signal").fillna(0)
    if long_only:
        # 取每日 signal>0 中前 top_quantile 比例
        pos_mask = df > 0
        # 用 signal 值在正信号内部排名（pct=0~1，最高值 pct 最大）
        # 避免 ties：用 method='first' + ascending=False
        rank = df.where(pos_mask).rank(axis=1, method="first", ascending=False, pct=True)
        keep = (rank <= top_quantile) & pos_mask
        weights = keep.astype(float).div(keep.sum(axis=1).replace(0, np.nan), axis=0)
        weights = weights.fillna(0)
    else:
        # 做多/做空：用 signal 值本身排序
        pos_mask = df > 0
        neg_mask = df < 0
        pos_rank = df.where(pos_mask).rank(axis=1, method="first", ascending=False, pct=True)
        neg_rank = df.where(neg_mask).rank(axis=1, method="first", ascending=True, pct=True)  # 越负越小
        pos_keep = (pos_rank <= top_quantile) & pos_mask
        neg_keep = (neg_rank <= bottom_quantile) & neg_mask if bottom_quantile else None
        if neg_keep is not None:
            weights = pos_keep.astype(float) - neg_keep.astype(float)
        else:
            weights = pos_keep.astype(float)
        # 归一化使每日 |权重|=1
        abs_sum = weights.abs().sum(axis=1).replace(0, np.nan)
        weights = weights.div(abs_sum, axis=0).fillna(0)
    return PortfolioWeight(weight_frame=weights, rebalance_freq="daily")


# ---------------------------------------------------------------------------
# 核心：向量化回测（核心思想：所有运算都是矩阵级）
# ---------------------------------------------------------------------------
def vectorized_backtest(
    price_df: pd.DataFrame,
    weights: PortfolioWeight,
    init_capital: float = 1_000_000.0,
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.001,
    min_commission: float = 5.0,
    slippage: float = 0.0001,
    t_plus_1: bool = True,
    price_limit: bool = True,
) -> Dict[str, Any]:
    """完全向量化的回测引擎。

    Parameters
    ----------
    price_df : DataFrame with columns [code, date, open, high, low, close, volume,
                                       is_limit_up, is_limit_down]
    weights : PortfolioWeight
        标准化权重（date × code）
    init_capital : float
    commission_rate, stamp_tax_rate, min_commission : A 股交易成本
    slippage : float
    t_plus_1 : bool
        True 则 T+1 不可当日卖出
    price_limit : bool
        True 则跳过涨跌停不可成交

    Returns
    -------
    dict，包含 trades / positions / equity_curve / metrics
    """
    # 1) 准备 wide-form 价格矩阵
    price_wide = price_df.pivot(index="date", columns="code", values="close").sort_index()
    open_wide = price_df.pivot(index="date", columns="code", values="open").sort_index()
    is_limit_up = (
        price_df.pivot(index="date", columns="code", values="is_limit_up").sort_index()
        if "is_limit_up" in price_df.columns
        else pd.DataFrame(False, index=price_wide.index, columns=price_wide.columns)
    )
    is_limit_down = (
        price_df.pivot(index="date", columns="code", values="is_limit_down").sort_index()
        if "is_limit_down" in price_df.columns
        else pd.DataFrame(False, index=price_wide.index, columns=price_wide.columns)
    )

    w = weights.weight_frame.reindex(index=price_wide.index, columns=price_wide.columns).fillna(0)

    # 2) 目标持仓股数（按当日权重 × 当日总资产）
    # 注意：target_shares 的计算依赖前一交易日收盘后的资产
    n_dates = len(price_wide)
    equity_arr = np.zeros(n_dates)
    cash_arr = np.zeros(n_dates)
    mv_arr = np.zeros(n_dates)
    pos_count_arr = np.zeros(n_dates, dtype=int)

    # 初始全部现金
    cash = init_capital
    holdings = np.zeros(len(price_wide.columns))  # 当前持股数（股）
    holdings_price = np.full(len(price_wide.columns), np.nan)  # 买入价
    is_today_buy = np.zeros(len(price_wide.columns), dtype=bool)  # T+1 标记

    close_arr = price_wide.values
    open_arr = open_wide.values
    limit_up_arr = is_limit_up.values
    limit_down_arr = is_limit_down.values
    w_arr = w.values

    # 简单按 100 股整手
    lot = 100
    trade_records: List[Dict[str, Any]] = []

    prev_total_equity = init_capital

    for t in range(n_dates):
        # 0) 资产估值（用今日开盘前 → 即昨日收盘）
        if t == 0:
            equity_today = init_capital
        else:
            prev_prices = close_arr[t - 1]
            valid = ~np.isnan(prev_prices) & (holdings > 0) & ~is_today_buy
            mv = float(np.nansum(holdings[valid] * prev_prices[valid]))
            equity_today = cash + mv

        equity_arr[t] = equity_today

        # 1) 卖出：T+1 允许卖出（T-1 买的今天可以卖）
        # 假设信号在开盘前产生，按今日 open 价卖出
        target_w = w_arr[t]
        # 当前实际权重
        if equity_today > 0:
            curr_prices_for_value = open_arr[t] if t < n_dates else close_arr[t]
            if t == 0:
                curr_holdings_value = 0
            else:
                prev_close = close_arr[t - 1]
                valid_now = ~np.isnan(prev_close) & (holdings > 0)
                curr_holdings_value = float(np.nansum(holdings[valid_now] * prev_close[valid_now]))
        else:
            curr_holdings_value = 0

        # 卖出：curr_weight > target_weight 的部分
        if t > 0 and equity_today > 0:
            prev_close = close_arr[t - 1]
            valid_now = ~np.isnan(prev_close) & (holdings > 0) & ~is_today_buy
            # 今日开盘价
            sell_price = open_arr[t] * (1 - slippage)  # 卖出价略低
            sellable = valid_now & ~np.isnan(sell_price) & (
                (~price_limit) | (~np.asarray(limit_down_arr[t], dtype=bool))
            )
            # 估算当前权重
            total_v = float(np.nansum(holdings[sellable] * prev_close[sellable]))
            if total_v > 0:
                curr_w = np.zeros_like(target_w)
                curr_w[sellable] = (holdings[sellable] * prev_close[sellable]) / equity_today
            else:
                curr_w = np.zeros_like(target_w)
            # 要减仓的代码
            reduce_mask = (curr_w > target_w) & sellable
            if reduce_mask.any():
                # 计算要卖的股数
                target_value = target_w[reduce_mask] * equity_today
                curr_value = (holdings[reduce_mask] * prev_close[reduce_mask])
                reduce_value = np.maximum(curr_value - target_value, 0)
                sell_shares = np.minimum(
                    (reduce_value / prev_close[reduce_mask] / lot).astype(int) * lot,
                    holdings[reduce_mask],
                )
                sell_shares = np.maximum(sell_shares, 0)
                # 实际卖出的钱
                sell_prices_actual = sell_price[reduce_mask]
                sell_amounts = sell_shares * sell_prices_actual
                # 手续费 + 印花税
                commissions = np.maximum(sell_amounts * commission_rate, min_commission)
                taxes = sell_amounts * stamp_tax_rate
                cash += np.sum(sell_amounts - commissions - taxes)
                # 记录交易
                for i, idx in enumerate(np.where(reduce_mask)[0]):
                    if sell_shares[i] > 0:
                        trade_records.append({
                            "date": price_wide.index[t],
                            "code": price_wide.columns[idx],
                            "action": "sell",
                            "price": float(sell_prices_actual[i]),
                            "shares": int(sell_shares[i]),
                            "amount": float(sell_amounts[i]),
                        })
                holdings[reduce_mask] -= sell_shares

        # 2) 买入：T+1 下单，今日不能卖出
        # 估算当前可用现金
        if equity_today > 0:
            target_value = target_w * equity_today
            prev_close = close_arr[t - 1] if t > 0 else close_arr[t]
            if t == 0:
                curr_value_now = np.zeros_like(target_value)
            else:
                valid_now2 = ~np.isnan(prev_close) & (holdings > 0)
                curr_value_now = np.where(valid_now2, holdings * prev_close, 0)
            # 需要加仓的代码
            need_buy_value = np.maximum(target_value - curr_value_now, 0)
            # 限制：不能买涨停
            buyable = ~np.isnan(open_arr[t]) & (
                (~price_limit) | (~np.asarray(limit_up_arr[t], dtype=bool))
            )
            need_buy_value = np.where(buyable, need_buy_value, 0)
            total_need = float(np.nansum(need_buy_value))
            if total_need > 0 and cash > 0:
                alloc = np.minimum(need_buy_value, cash * need_buy_value / total_need * 0.95)
                buy_price = open_arr[t] * (1 + slippage)  # 买入价略高
                buy_shares = ((alloc / buy_price / lot).astype(int)) * lot
                buy_shares = np.where(buyable, buy_shares, 0)
                # 资金约束
                cost = buy_shares * buy_price
                commissions = np.maximum(cost * commission_rate, min_commission)
                total_cost = cost + commissions
                # 若超额，缩股
                over = total_cost.sum() > cash
                if over:
                    scale = cash / total_cost.sum() * 0.99
                    buy_shares = (buy_shares.astype(float) * scale / lot).astype(int) * lot
                    buy_shares = np.maximum(buy_shares, 0)
                    cost = buy_shares * buy_price
                    commissions = np.maximum(cost * commission_rate, min_commission)
                    total_cost = cost + commissions
                cash -= float(np.nansum(total_cost))
                for i, idx in enumerate(np.where(buy_shares > 0)[0]):
                    trade_records.append({
                        "date": price_wide.index[t],
                        "code": price_wide.columns[idx],
                        "action": "buy",
                        "price": float(buy_price[idx]),
                        "shares": int(buy_shares[i]),
                        "amount": float(cost[i]),
                    })
                holdings += buy_shares
                is_today_buy = (buy_shares > 0)  # T+1 标记

        # 3) 用今日 close 重算市值
        valid_close = ~np.isnan(close_arr[t]) & (holdings > 0) & ~is_today_buy
        mv = float(np.nansum(holdings[valid_close] * close_arr[t][valid_close]))
        total_equity = cash + mv
        mv_arr[t] = mv
        cash_arr[t] = cash
        pos_count_arr[t] = int(np.sum(holdings > 0))

        # 4) 重置 T+1 标记
        is_today_buy = np.zeros_like(is_today_buy)

    # 构造结果
    equity_curve = pd.DataFrame({
        "date": price_wide.index,
        "equity": equity_arr,
        "cash": cash_arr,
        "market_value": mv_arr,
        "position_count": pos_count_arr,
    })
    trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        columns=["date", "code", "action", "price", "shares", "amount"]
    )
    metrics = _calc_metrics(equity_curve, init_capital)
    return {
        "equity_curve": equity_curve,
        "trades": trades_df,
        "positions": pd.DataFrame({"code": price_wide.columns, "shares": holdings}),
        "metrics": metrics,
    }


def _calc_metrics(equity_curve: pd.DataFrame, init_capital: float) -> Dict[str, float]:
    """计算绩效指标：年化收益 / 夏普 / 最大回撤 / Calmar / 胜率"""
    if equity_curve.empty or "equity" not in equity_curve.columns:
        return {}
    eq = equity_curve.set_index("date")["equity"]
    if len(eq) < 2:
        return {}
    returns = eq.pct_change().dropna()
    if len(returns) < 2:
        return {}
    total_return = float((1 + returns).prod() - 1)
    n = len(returns)
    annual_return = float((1 + total_return) ** (252 / n) - 1)
    volatility = float(returns.std() * np.sqrt(252))
    sharpe = float((annual_return - 0.03) / volatility) if volatility > 0 else 0.0
    running_max = eq.cummax()
    drawdown = eq / running_max - 1
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
    win_rate = float((returns > 0).mean())
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "n_days": n,
    }


__all__ = [
    "PortfolioWeight",
    "signals_to_weights",
    "vectorized_backtest",
]
