"""
A 股模拟交易所 (exchange_simulator)
====================================

借鉴来源:
  - Qlib (microsoft/qlib) `qlib/backtest/exchange.py`
    * exchange_kwargs 配置风格（limit_threshold / open_cost / close_cost / impact_cost / min_cost / trade_unit）
    * 价格涨跌停校验
    * 涨跌停订单撮合逻辑
  - jingni-trader/skills/backtest-engine/scripts/adapters/native_adapter.py
    * T+1 规则、印花税（仅卖出）
    * 最小交易单位 100 股

设计目标:
  1. 把交易成本/规则参数化到 `ExchangeConfig`，便于场景压测
  2. 涨跌停校验逻辑更精细（含停牌/一字板）
  3. 与 native_adapter 接口兼容，可平替现有回测适配器
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class ExchangeConfig:
    """
    借鉴 Qlib `exchange_kwargs`，集中描述 A 股交易成本与规则

    字段说明:
      - limit_threshold: 涨跌幅限制（0.095 = 9.5%）
      - open_cost: 买入成本（含佣金 + 过户费）
      - close_cost: 卖出成本（佣金 + 过户费，不含印花税）
      - stamp_tax: 印花税（仅卖出，按成交额）
      - impact_cost: 冲击成本（按成交额，模拟滑点）
      - min_cost: 最低佣金（元/笔）
      - trade_unit: 最小交易单位（股）
      - t_plus_1: T+1 规则
      - lot_size_enforce: 是否强制整百股
    """
    limit_threshold: float = 0.095
    open_cost: float = 0.00025      # 买入佣金 万2.5
    close_cost: float = 0.00025     # 卖出佣金 万2.5
    stamp_tax: float = 0.001        # 印花税 千1
    impact_cost: float = 0.0001     # 滑点 万1
    min_cost: float = 5.0
    trade_unit: int = 100
    t_plus_1: bool = True
    lot_size_enforce: bool = True

    def total_buy_cost_rate(self) -> float:
        """买入端总成本率（不含滑点）"""
        return self.open_cost + self.impact_cost

    def total_sell_cost_rate(self) -> float:
        """卖出端总成本率（不含滑点）"""
        return self.close_cost + self.impact_cost + self.stamp_tax


@dataclass
class Order:
    """交易订单"""
    code: str
    action: str            # 'buy' / 'sell'
    target_shares: int     # 目标持仓股数（正数）
    reason: str = ""

    def __post_init__(self):
        if self.action not in ("buy", "sell"):
            raise ValueError(f"action 必须是 'buy' 或 'sell'，收到: {self.action}")
        if self.target_shares < 0:
            raise ValueError("target_shares 必须为非负数")


@dataclass
class Fill:
    """订单成交回报"""
    code: str
    action: str
    shares: int
    price: float
    gross_amount: float    # 成交总额
    commission: float
    tax: float
    impact_cost: float
    net_cash_flow: float   # 净现金变动（买入为负，卖出为正）


class LimitUpDownError(Exception):
    """价格触及涨跌停，无法成交"""
    pass


class SuspendedError(Exception):
    """股票停牌"""
    pass


class InsufficientCashError(Exception):
    """资金不足"""
    pass


class TPlus1Error(Exception):
    """T+1 限制：当日买入次日才可卖出"""
    pass


class AShareExchange:
    """
    A 股模拟交易所

    用法:
        ex = AShareExchange(ExchangeConfig())
        ex.update_market(daily_bars_df)
        fill, error = ex.submit_order(order, current_positions, cash, current_date)
    """

    def __init__(self, config: Optional[ExchangeConfig] = None):
        self.config = config or ExchangeConfig()
        self._market: Dict[str, pd.DataFrame] = {}   # code -> DataFrame(date-indexed)
        self._t1_holding: Dict[str, set] = {}        # code -> {date} 当日买入的日期

    # ------------------------------------------------------------------
    # 数据注入
    # ------------------------------------------------------------------

    def update_market(self, daily_bars: pd.DataFrame) -> None:
        """
        注入行情数据

        期望列: code, date, open, high, low, close, volume, is_limit_up, is_limit_down
        """
        required = {"code", "date", "close"}
        missing = required - set(daily_bars.columns)
        if missing:
            raise ValueError(f"daily_bars 缺少必需列: {missing}")

        df = daily_bars.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"])

        for code, grp in df.groupby("code"):
            self._market[code] = grp.set_index("date")

    def get_bar(self, code: str, date: pd.Timestamp) -> Optional[pd.Series]:
        bar = self._market.get(code)
        if bar is None or date not in bar.index:
            return None
        return bar.loc[date]

    def get_prev_close(self, code: str, date: pd.Timestamp) -> Optional[float]:
        bar = self._market.get(code)
        if bar is None:
            return None
        idx = bar.index.get_indexer([date], method="ffill")[0]
        if idx <= 0:
            return None
        return float(bar.iloc[idx - 1]["close"])

    # ------------------------------------------------------------------
    # 撮合核心
    # ------------------------------------------------------------------

    def check_tradable(self, code: str, date: pd.Timestamp, action: str) -> Tuple[bool, str]:
        """
        检查股票在指定日期是否可交易，返回 (tradable, reason)
        """
        bar = self.get_bar(code, date)
        if bar is None:
            return False, f"{code}@{date} 无行情数据"
        if bar.get("volume", 0) <= 0:
            return False, f"{code}@{date} 停牌（成交量=0）"
        if action == "buy" and bool(bar.get("is_limit_up", False)):
            return False, f"{code}@{date} 涨停（一字板）"
        if action == "sell" and bool(bar.get("is_limit_down", False)):
            return False, f"{code}@{date} 跌停（一字板）"
        return True, "ok"

    def _deal_price(self, bar: pd.Series, action: str) -> float:
        """计算成交价（含 impact_cost 滑点）"""
        price = float(bar["close"])
        if action == "buy":
            price *= 1 + self.config.impact_cost
        else:  # sell
            price *= 1 - self.config.impact_cost
        return price

    def _calc_costs(self, action: str, amount: float) -> Tuple[float, float, float]:
        """返回 (commission, tax, impact_cost_value)"""
        cfg = self.config
        if action == "buy":
            commission = max(amount * cfg.open_cost, cfg.min_cost)
            tax = 0.0
            impact = amount * cfg.impact_cost
        else:
            commission = max(amount * (cfg.close_cost), cfg.min_cost)
            tax = amount * cfg.stamp_tax
            impact = amount * cfg.impact_cost
        return commission, tax, impact

    def _round_lots(self, shares: int) -> int:
        """按最小交易单位向下取整"""
        if not self.config.lot_size_enforce:
            return shares
        unit = self.config.trade_unit
        return (shares // unit) * unit

    def submit_order(
        self,
        order: Order,
        current_positions: Dict[str, int],     # code -> shares
        cash: float,
        current_date: pd.Timestamp,
    ) -> Tuple[Optional[Fill], Optional[str]]:
        """
        提交订单，返回 (fill, error_message)
        """
        if order.target_shares <= 0:
            return None, "target_shares 必须为正"

        tradable, reason = self.check_tradable(order.code, current_date, order.action)
        if not tradable:
            return None, reason

        # T+1 校验
        if self.config.t_plus_1 and order.action == "sell":
            bought_dates = self._t1_holding.get(order.code, set())
            if current_date in bought_dates:
                return None, f"T+1 限制：{order.code} 当日买入当日不可卖出"

        bar = self.get_bar(order.code, current_date)
        price = self._deal_price(bar, order.action)

        # 1) 卖出：直接按当前持仓处理
        if order.action == "sell":
            held = current_positions.get(order.code, 0)
            if held <= 0:
                return None, f"无 {order.code} 持仓可卖"
            shares = min(order.target_shares, held)
            shares = self._round_lots(shares)
            if shares <= 0:
                return None, "可卖股数 < 1 手"

            gross = price * shares
            commission, tax, impact = self._calc_costs("sell", gross)
            net_cash = gross - commission - tax
            return Fill(
                code=order.code,
                action="sell",
                shares=shares,
                price=price,
                gross_amount=gross,
                commission=commission,
                tax=tax,
                impact_cost=impact,
                net_cash_flow=net_cash,
            ), None

        # 2) 买入：按可用现金处理
        commission0, tax0, impact0 = self._calc_costs("buy", price * self.config.trade_unit)
        cost_per_lot = price * self.config.trade_unit + commission0 + impact0
        max_lots = int(cash / cost_per_lot)
        shares = self._round_lots(min(order.target_shares, max_lots * self.config.trade_unit))
        if shares <= 0:
            return None, f"资金不足，最多可买 {max_lots} 手"

        gross = price * shares
        commission, tax, impact = self._calc_costs("buy", gross)
        total_cost = gross + commission + impact
        if total_cost > cash:
            return None, f"资金不足：需要 {total_cost:.2f}，可用 {cash:.2f}"

        return Fill(
            code=order.code,
            action="buy",
            shares=shares,
            price=price,
            gross_amount=gross,
            commission=commission,
            tax=tax,
            impact_cost=impact,
            net_cash_flow=-total_cost,
        ), None

    def mark_buy(self, code: str, date: pd.Timestamp) -> None:
        """T+1 记录：当日买入"""
        if not self.config.t_plus_1:
            return
        self._t1_holding.setdefault(code, set()).add(date)

    def reset_t1(self) -> None:
        """重置 T+1 记录（新交易日开始时调用）"""
        self._t1_holding.clear()


# ============================================================
# 简化回测器：使用 AShareExchange 执行
# ============================================================

@dataclass
class StrategyOutput:
    """每日策略输出"""
    date: pd.Timestamp
    target_holdings: Dict[str, int] = field(default_factory=dict)   # code -> 目标股数


def run_exchange_backtest(
    daily_bars: pd.DataFrame,
    strategy: List[StrategyOutput],
    config: Optional[ExchangeConfig] = None,
    init_cash: float = 1_000_000.0,
) -> Dict[str, Any]:
    """
    使用 AShareExchange 跑回测

    Args:
        daily_bars: 行情数据 (code, date, OHLCV, is_limit_up, is_limit_down)
        strategy: 每日目标持仓
        config: 交易所配置
        init_cash: 初始资金

    Returns:
        {
            "equity_curve": DataFrame(date, equity, cash, market_value, position_count),
            "trades": DataFrame(每笔成交),
            "metrics": dict,
            "config": ExchangeConfig,
            "rejected_orders": list
        }
    """
    cfg = config or ExchangeConfig()
    ex = AShareExchange(cfg)
    ex.update_market(daily_bars)

    cash = init_cash
    positions: Dict[str, int] = {}
    equity_records: List[Dict[str, Any]] = []
    trades_records: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for output in strategy:
        dt = output.date
        # 1) 收盘时按目标持仓做调仓
        target = dict(output.target_holdings)

        # 卖出：当前持仓中不在目标里的
        for code, shares in list(positions.items()):
            if shares <= 0:
                continue
            tgt = target.get(code, 0)
            if tgt < shares:
                diff = shares - tgt
                order = Order(code=code, action="sell", target_shares=diff, reason="rebalance")
                fill, err = ex.submit_order(order, positions, cash, dt)
                if fill is None:
                    rejected.append({"date": dt, "code": code, "action": "sell", "reason": err})
                else:
                    cash += fill.net_cash_flow
                    positions[code] = positions.get(code, 0) - fill.shares
                    trades_records.append({
                        "date": dt, "code": code, "action": "sell",
                        "price": fill.price, "shares": fill.shares,
                        "amount": fill.gross_amount,
                        "commission": fill.commission,
                        "tax": fill.tax,
                        "pnl": fill.net_cash_flow,
                    })
                    # 卖出不触发 T+1 标记

        # 买入：目标持仓中当前没有的
        for code, tgt in target.items():
            tgt = max(0, tgt)
            cur = positions.get(code, 0)
            if tgt > cur:
                diff = tgt - cur
                order = Order(code=code, action="buy", target_shares=diff, reason="rebalance")
                fill, err = ex.submit_order(order, positions, cash, dt)
                if fill is None:
                    rejected.append({"date": dt, "code": code, "action": "buy", "reason": err})
                else:
                    cash += fill.net_cash_flow
                    positions[code] = positions.get(code, 0) + fill.shares
                    trades_records.append({
                        "date": dt, "code": code, "action": "buy",
                        "price": fill.price, "shares": fill.shares,
                        "amount": fill.gross_amount,
                        "commission": fill.commission,
                        "tax": fill.tax,
                        "pnl": -fill.gross_amount - fill.commission,
                    })
                    ex.mark_buy(code, dt)

        # 2) 按当日收盘价计算权益
        market_value = 0.0
        for code, shares in positions.items():
            if shares <= 0:
                continue
            bar = ex.get_bar(code, dt)
            if bar is not None:
                market_value += shares * float(bar["close"])
        total_equity = cash + market_value
        equity_records.append({
            "date": dt,
            "equity": total_equity,
            "cash": cash,
            "market_value": market_value,
            "position_count": sum(1 for s in positions.values() if s > 0),
        })

    equity_df = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trades_records)

    return {
        "equity_curve": equity_df,
        "trades": trades_df,
        "rejected_orders": rejected,
        "config": cfg,
    }


__all__ = [
    "ExchangeConfig",
    "Order",
    "Fill",
    "AShareExchange",
    "StrategyOutput",
    "run_exchange_backtest",
    "LimitUpDownError", "SuspendedError", "InsufficientCashError", "TPlus1Error",
]
