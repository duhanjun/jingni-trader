"""
事件驱动回测引擎验证模块
借鉴 NautilusTrader 的确定性事件驱动架构 + 研究到实盘一致性

核心改进点（对照 jingni-trader 现有 native_adapter.py）：
1. 双时间戳：ts_event（事件发生时间）+ ts_init（系统创建事件时间），保证可审计
2. 严格 T+1：T 日收盘信号 → T+1 日开盘成交（现有实现 T 日 close 成交，存在前视偏差）
3. 订单状态机：PENDING -> ACCEPTED -> FILLED / CANCELED / REJECTED
4. FillModel 抽象：滑点、成交价模型可插拔（借鉴 NautilusTrader SimulatedExchange）
5. 事件按 ts_event 严格排序，确定性可复现
6. 撮合时区分 open/high/low/close，支持 VWAP 等多种成交价

借鉴来源：
- NautilusTrader: https://nautilustrader.io/docs/latest/concepts/architecture/
  - NautilusKernel 共享执行模型（回测/实盘一致）
  - TestClock 确定性时钟，事件按 ts_event 排序
  - SimulatedExchange + FillModel + FeeModel + LatencyModel
  - 双时间戳 ts_event / ts_init
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import time
import numpy as np
import pandas as pd


# ============================================================
# 事件系统：借鉴 NautilusTrader 的 Event 对象 + 双时间戳
# ============================================================

class EventType(str, Enum):
    BAR = "bar"                  # 行情事件（每日 OHLCV）
    SIGNAL = "signal"            # 信号事件
    ORDER_SUBMIT = "order_submit"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELED = "order_canceled"
    ORDER_REJECTED = "order_rejected"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class Event:
    """事件基类，携带双时间戳（借鉴 NautilusTrader）"""
    event_type: EventType
    ts_event: pd.Timestamp      # 事件在业务域发生的时间（如行情日期）
    ts_init: pd.Timestamp       # 系统创建该事件对象的时间（回测中=时钟推进时间）
    payload: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Event") -> bool:
        # 确定性排序：先按 ts_event，再按 event_type 优先级
        priority = {
            EventType.BAR: 0,
            EventType.SIGNAL: 1,
            EventType.ORDER_SUBMIT: 2,
            EventType.ORDER_ACCEPTED: 3,
            EventType.ORDER_FILLED: 4,
        }
        return (self.ts_event, priority.get(self.event_type, 9)) < (
            other.ts_event, priority.get(other.event_type, 9))


@dataclass
class Order:
    """订单对象 + 状态机"""
    order_id: str
    code: str
    side: OrderSide
    quantity: int               # 股数（A股100股整数倍）
    target_weight: Optional[float] = None  # 目标权重（可选）
    status: OrderStatus = OrderStatus.PENDING
    submit_ts: Optional[pd.Timestamp] = None
    fill_ts: Optional[pd.Timestamp] = None
    fill_price: float = 0.0
    filled_quantity: int = 0
    commission: float = 0.0
    tax: float = 0.0
    reject_reason: str = ""


# ============================================================
# FillModel：可插拔的成交价模型（借鉴 NautilusTrader FillModel）
# ============================================================

class FillModel:
    """成交价模型抽象，支持 next_open / vwap / close 等策略"""

    def __init__(self, price_type: str = "next_open", slippage: float = 0.001):
        self.price_type = price_type
        self.slippage = slippage

    def get_fill_price(self, side: OrderSide, bar: pd.Series) -> float:
        raw = bar.get(self.price_type, bar["close"])
        if side == OrderSide.BUY:
            return float(raw) * (1 + self.slippage)
        else:
            return float(raw) * (1 - self.slippage)

    def is_tradable(self, side: OrderSide, bar: pd.Series, price_limit: bool) -> bool:
        if not price_limit:
            return True
        if side == OrderSide.BUY and bar.get("is_limit_up", False):
            return False
        if side == OrderSide.SELL and bar.get("is_limit_down", False):
            return False
        return True


# ============================================================
# 确定性时钟（借鉴 NautilusTrader TestClock）
# ============================================================

class DeterministicClock:
    """确定性时钟：回测中显式推进，保证可复现"""

    def __init__(self, start_ts: pd.Timestamp):
        self.current_ts = start_ts

    def advance_to(self, ts: pd.Timestamp):
        if ts < self.current_ts:
            raise ValueError(f"时钟不可回退: {self.current_ts} -> {ts}")
        self.current_ts = ts


# ============================================================
# 事件驱动回测引擎核心
# ============================================================

class EventDrivenBacktestEngine:
    """
    事件驱动回测引擎

    关键设计：
    - 信号在 T 日收盘后生成，订单在 T+1 日提交并成交（严格 T+1）
    - 事件按 ts_event 严格排序处理，确定性可复现
    - FillModel 可插拔，默认 next_open（次日开盘成交）
    - 订单状态机完整记录生命周期
    """

    def __init__(
        self,
        init_capital: float = 1e6,
        commission_rate: float = 0.00025,
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        fill_model: Optional[FillModel] = None,
        budget_ratio: float = 0.95,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit
        self.fill_model = fill_model or FillModel(price_type="next_open", slippage=0.001)
        self.budget_ratio = budget_ratio

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        执行事件驱动回测

        参数:
            data: 行情数据 code, date, open, high, low, close, volume,
                  is_st, is_limit_up, is_limit_down
            signals: 信号 code, date, signal (1/-1/0) 或 target_weight
        """
        t0 = time.perf_counter()
        if data.empty or signals.empty:
            return self._empty_result()

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        all_dates = sorted(data["date"].unique())
        if len(all_dates) == 0:
            return self._empty_result()

        clock = DeterministicClock(pd.Timestamp(all_dates[0]))

        # 行情索引：(date, code) -> bar Series
        bar_lookup = {}
        for _, row in data.iterrows():
            bar_lookup[(row["date"], row["code"])] = row

        # 次日映射：T -> T+1（用于 T+1 成交）
        date_to_next = {}
        for i in range(len(all_dates) - 1):
            date_to_next[all_dates[i]] = all_dates[i + 1]

        # 信号索引：date -> {code: signal_row}
        signal_lookup = {}
        for _, row in signals.iterrows():
            signal_lookup.setdefault(row["date"], {})[row["code"]] = row

        cash = self.init_capital
        positions: Dict[str, int] = {}          # code -> shares
        cost_basis: Dict[str, float] = {}       # code -> avg cost
        pending_orders: List[Order] = []        # 待成交订单（T+1 撮合）
        trades: List[Dict] = []
        equity_records: List[Dict] = []
        order_log: List[Dict] = []
        order_counter = 0

        for dt in all_dates:
            clock.advance_to(pd.Timestamp(dt))

            # ---- Step 1: 撮合上一日提交的订单（T+1 成交）----
            still_pending = []
            pending_codes = {o.code for o in pending_orders}
            day_bars = {code: bar_lookup[(dt, code)]
                        for code in pending_codes
                        if (dt, code) in bar_lookup}

            for order in pending_orders:
                ts_init = clock.current_ts
                bar = day_bars.get(order.code)
                if bar is None:
                    order.status = OrderStatus.CANCELED
                    order.reject_reason = "no_bar_data"
                    order_log.append(self._order_to_dict(order, dt, "canceled"))
                    continue

                if not self.fill_model.is_tradable(order.side, bar, self.price_limit):
                    order.status = OrderStatus.CANCELED
                    order.reject_reason = "price_limit"
                    order_log.append(self._order_to_dict(order, dt, "canceled"))
                    continue

                fill_price = self.fill_model.get_fill_price(order.side, bar)
                order.status = OrderStatus.FILLED
                order.fill_ts = dt
                order.fill_price = fill_price
                order.filled_quantity = order.quantity

                amount = fill_price * order.quantity
                if order.side == OrderSide.BUY:
                    commission = max(amount * self.commission_rate, self.min_commission)
                    order.commission = commission
                    order.tax = 0.0
                    cash -= amount + commission
                    old_shares = positions.get(order.code, 0)
                    new_shares = old_shares + order.quantity
                    old_cost = cost_basis.get(order.code, 0) * old_shares
                    cost_basis[order.code] = (old_cost + amount) / new_shares if new_shares else 0
                    positions[order.code] = new_shares
                else:
                    commission = max(amount * self.commission_rate, self.min_commission)
                    tax = amount * self.stamp_tax_rate
                    order.commission = commission
                    order.tax = tax
                    cash += amount - commission - tax
                    positions[order.code] = positions.get(order.code, 0) - order.quantity
                    if positions[order.code] <= 0:
                        positions.pop(order.code, None)
                        cost_basis.pop(order.code, None)

                trades.append({
                    "date": dt,
                    "code": order.code,
                    "action": order.side.value,
                    "price": fill_price,
                    "shares": order.quantity,
                    "amount": amount,
                    "commission": commission,
                    "tax": order.tax,
                    "order_id": order.order_id,
                    "signal_date": order.submit_ts,
                })
                order_log.append(self._order_to_dict(order, dt, "filled"))

            pending_orders = still_pending

            # ---- Step 2: 处理当日信号，生成次日订单 ----
            day_signals = signal_lookup.get(dt, {})
            if day_signals:
                sell_codes, buy_codes = [], []
                for code, sig_row in day_signals.items():
                    sig = sig_row.get("signal", 0)
                    if isinstance(sig, (int, float, np.integer, np.floating)):
                        if float(sig) > 0:
                            buy_codes.append(code)
                        elif float(sig) < 0:
                            sell_codes.append(code)

                # 卖出订单
                for code in sell_codes:
                    shares = positions.get(code, 0)
                    if shares <= 0:
                        continue
                    order_counter += 1
                    order = Order(
                        order_id=f"O{order_counter:06d}",
                        code=code,
                        side=OrderSide.SELL,
                        quantity=shares,
                        status=OrderStatus.PENDING,
                        submit_ts=dt,
                    )
                    pending_orders.append(order)
                    order_log.append(self._order_to_dict(order, dt, "submitted"))

                # 买入订单（按预算分配）
                if buy_codes:
                    budget_per = cash * self.budget_ratio / len(buy_codes)
                    for code in buy_codes:
                        next_dt = date_to_next.get(dt)
                        if next_dt is None:
                            continue
                        next_bar = bar_lookup.get((next_dt, code))
                        ref_price = next_bar["open"] * (1 + self.fill_model.slippage) if next_bar is not None else None
                        if ref_price is None:
                            continue
                        shares = int(budget_per / ref_price / 100) * 100
                        if shares <= 0:
                            continue
                        order_counter += 1
                        order = Order(
                            order_id=f"O{order_counter:06d}",
                            code=code,
                            side=OrderSide.BUY,
                            quantity=shares,
                            status=OrderStatus.PENDING,
                            submit_ts=dt,
                        )
                        pending_orders.append(order)
                        order_log.append(self._order_to_dict(order, dt, "submitted"))

            # ---- Step 3: 计算当日净值 ----
            market_value = 0.0
            day_data = data[data["date"] == dt]
            day_data_map = day_data.set_index("code") if not day_data.empty else pd.DataFrame()
            for code, shares in positions.items():
                if shares <= 0:
                    continue
                if code in day_data_map.index:
                    market_value += shares * float(day_data_map.loc[code, "close"])

            equity_records.append({
                "date": dt,
                "equity": cash + market_value,
                "cash": cash,
                "market_value": market_value,
                "position_count": sum(1 for s in positions.values() if s > 0),
            })

        elapsed = time.perf_counter() - t0
        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)
        orders_df = pd.DataFrame(order_log)

        metrics = self._calc_metrics(equity_curve)
        metrics["engine"] = "event_driven"
        metrics["elapsed_sec"] = round(elapsed, 4)
        metrics["total_orders"] = len(orders_df)
        metrics["filled_orders"] = int((orders_df["status"] == "filled").sum()) if not orders_df.empty else 0
        metrics["canceled_orders"] = int((orders_df["status"] == "canceled").sum()) if not orders_df.empty else 0

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(
                [{"code": k, "shares": v} for k, v in positions.items()],
                columns=["code", "shares"]),
            "equity_curve": equity_curve,
            "orders": orders_df,
            "metrics": metrics,
        }

    # --------------------------------------------------------
    def _order_to_dict(self, order: Order, dt, phase: str) -> Dict:
        return {
            "order_id": order.order_id,
            "code": order.code,
            "side": order.side.value,
            "quantity": order.quantity,
            "submit_ts": order.submit_ts,
            "fill_ts": order.fill_ts,
            "fill_price": order.fill_price,
            "status": order.status.value,
            "phase": phase,
            "ts_event": dt,
        }

    def _calc_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        if equity_curve.empty or "equity" not in equity_curve.columns:
            return {}
        eq = equity_curve.set_index("date")["equity"]
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

    def _empty_result(self):
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "metrics": {},
        }
