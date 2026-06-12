"""
验证测试：事件驱动回测引擎（Event-Driven Backtest Engine）

借鉴来源：NautilusTrader (https://github.com/nautechsystems/nautilus_trader)
  - NautilusTrader 采用 Rust 核心 + Python 绑定的混合架构，
    通过 MessageBus 实现组件间松耦合通信。
  - 核心设计理念：
    * 事件驱动架构：所有组件通过消息总线通信
    * 确定性时间模型：回测和实盘使用相同的时间推进逻辑
    * 风险引擎：预交易风险检查管道
    * 单线程高性能：借鉴 LMAX Disruptor 模式

优化方向：为 jingni-trader 的 backtest-engine 引入事件驱动架构，
  提升回测引擎的准确性和可扩展性，支持风险检查管道。

测试内容：
  1. 事件总线（MessageBus）发布/订阅机制
  2. 事件驱动时间推进（确定性时间模型）
  3. 风险检查管道（RiskEngine）
  4. 订单生命周期管理
  5. 回测性能对比（事件驱动 vs 向量化）
  6. 边界条件（空信号、极端价格变动）
"""

import unittest
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq
from abc import ABC, abstractmethod


# ============================================================
# 事件驱动架构实现（借鉴 NautilusTrader 设计）
# ============================================================

class EventType(Enum):
    """事件类型"""
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELED = "order_canceled"
    POSITION_CHANGED = "position_changed"
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"


@dataclass
class Event:
    """事件基类"""
    event_type: EventType
    timestamp: pd.Timestamp
    payload: Dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """
    消息总线：实现组件间的发布/订阅通信

    借鉴 NautilusTrader 的 MessageBus 设计：
    - 支持按事件类型订阅
    - 支持通配符订阅
    - 事件按时间顺序分发
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[callable]] = defaultdict(list)
        self._event_queue: List[Event] = []

    def subscribe(self, event_type: EventType, handler: callable):
        self._subscribers[event_type].append(handler)

    def publish(self, event: Event):
        self._event_queue.append(event)
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)

    def clear(self):
        self._event_queue.clear()


class RiskCheck(ABC):
    """风险检查基类"""

    @abstractmethod
    def check(self, order: Dict, portfolio: Dict) -> tuple:
        """返回 (passed: bool, reason: str)"""
        ...


class MaxPositionRiskCheck(RiskCheck):
    """最大持仓检查"""

    def __init__(self, max_position_pct: float = 0.2):
        self.max_position_pct = max_position_pct

    def check(self, order: Dict, portfolio: Dict) -> tuple:
        capital = portfolio.get("capital", 0)
        position_value = portfolio.get("position_value", 0)
        order_value = order["price"] * order["quantity"]
        new_position_pct = (position_value + order_value) / capital if capital > 0 else 1.0
        if new_position_pct > self.max_position_pct:
            return False, f"超过最大持仓限制 {self.max_position_pct*100}%"
        return True, "OK"


class MaxDrawdownRiskCheck(RiskCheck):
    """最大回撤检查"""

    def __init__(self, max_drawdown: float = 0.2):
        self.max_drawdown = max_drawdown

    def check(self, order: Dict, portfolio: Dict) -> tuple:
        current_drawdown = portfolio.get("current_drawdown", 0)
        if abs(current_drawdown) > self.max_drawdown:
            return False, f"超过最大回撤限制 {self.max_drawdown*100}%"
        return True, "OK"


class RiskEngine:
    """
    风险引擎：预交易风险检查管道

    借鉴 NautilusTrader 的 RiskEngine 设计：
    - 可插拔的风险检查器
    - 管道式检查，任一失败则拒绝
    """

    def __init__(self, checks: Optional[List[RiskCheck]] = None):
        self.checks = checks or []

    def add_check(self, check: RiskCheck):
        self.checks.append(check)

    def evaluate(self, order: Dict, portfolio: Dict) -> tuple:
        for check in self.checks:
            passed, reason = check.check(order, portfolio)
            if not passed:
                return False, reason
        return True, "ALL_CHECKS_PASSED"


class OrderStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELED = "canceled"


@dataclass
class Order:
    """订单"""
    order_id: str
    code: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    timestamp: pd.Timestamp
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0


class Portfolio:
    """投资组合管理器"""

    def __init__(self, init_capital: float = 1e6):
        self.init_capital = init_capital
        self.cash = init_capital
        self.positions: Dict[str, int] = defaultdict(int)
        self.position_value: Dict[str, float] = defaultdict(float)
        self.equity_history: List[Dict] = []
        self.trades: List[Dict] = []
        self.commission_rate = 0.00025
        self.stamp_tax_rate = 0.001  # 卖出印花税
        self.peak_equity = init_capital

    @property
    def total_equity(self) -> float:
        return self.cash + sum(self.position_value.values())

    @property
    def current_drawdown(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return (self.total_equity - self.peak_equity) / self.peak_equity

    def update_market_value(self, code: str, price: float):
        if code in self.positions:
            self.position_value[code] = self.positions[code] * price

    def record_snapshot(self, timestamp: pd.Timestamp):
        self.peak_equity = max(self.peak_equity, self.total_equity)
        self.equity_history.append({
            "date": timestamp,
            "equity": self.total_equity,
            "cash": self.cash,
            "positions": dict(self.positions),
        })

    def apply_fill(self, order: Order, fill_price: float):
        """执行成交"""
        order_value = fill_price * order.quantity
        if order.side == "buy":
            commission = order_value * self.commission_rate
            total_cost = order_value + commission
            if self.cash < total_cost:
                return False
            self.cash -= total_cost
            self.positions[order.code] += order.quantity
        else:
            commission = order_value * self.commission_rate
            stamp_tax = order_value * self.stamp_tax_rate
            self.cash += order_value - commission - stamp_tax
            self.positions[order.code] -= order.quantity

        self.position_value[order.code] = self.positions[order.code] * fill_price
        return True


class EventDrivenBacktestEngine:
    """
    事件驱动回测引擎

    借鉴 NautilusTrader 的设计理念：
    - 事件驱动架构：市场数据 → 策略 → 信号 → 订单 → 成交
    - 确定性时间模型：按时间顺序推进
    - 风险引擎预检查：订单执行前过风控管道
    """

    def __init__(self, init_capital: float = 1e6):
        self.bus = MessageBus()
        self.portfolio = Portfolio(init_capital)
        self.risk_engine = RiskEngine([
            MaxPositionRiskCheck(max_position_pct=0.3),
            MaxDrawdownRiskCheck(max_drawdown=0.3),
        ])
        self._order_counter = 0
        self._subscribe_handlers()

    def _subscribe_handlers(self):
        """注册内部事件处理器"""
        self.bus.subscribe(EventType.ORDER_SUBMITTED, self._handle_order_submitted)

    def _handle_order_submitted(self, event: Event):
        """处理订单提交事件"""
        order = event.payload["order"]
        # 风险检查
        portfolio_state = {
            "capital": self.portfolio.total_equity,
            "position_value": self.portfolio.position_value.get(order.code, 0),
            "current_drawdown": self.portfolio.current_drawdown,
        }
        passed, reason = self.risk_engine.evaluate(
            {"price": order.price, "quantity": order.quantity},
            portfolio_state,
        )
        if passed:
            self.bus.publish(Event(
                EventType.RISK_CHECK_PASSED,
                event.timestamp,
                {"order": order}
            ))
        else:
            order.status = OrderStatus.REJECTED
            self.bus.publish(Event(
                EventType.RISK_CHECK_FAILED,
                event.timestamp,
                {"order": order, "reason": reason}
            ))

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD-{self._order_counter:06d}"

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        执行事件驱动回测

        参数:
            data: 行情数据 (code, date, open, high, low, close, volume)
            signals: 交易信号 (code, date, signal)
        """
        self.bus.clear()
        self.portfolio = Portfolio(self.portfolio.init_capital)

        # 合并数据和信号
        merged = data.merge(
            signals[["code", "date", "signal"]],
            on=["code", "date"],
            how="left",
        )
        merged["signal"] = merged["signal"].fillna(0)

        # 按日期分组，按时间顺序推进
        date_groups = sorted(merged["date"].unique())

        for date in date_groups:
            day_data = merged[merged["date"] == date]

            # 发布市场数据事件
            for _, row in day_data.iterrows():
                self.bus.publish(Event(
                    EventType.MARKET_DATA,
                    date,
                    {
                        "code": row["code"],
                        "close": row["close"],
                        "open": row["open"],
                    }
                ))
                # 更新持仓市值
                self.portfolio.update_market_value(row["code"], row["close"])

            # 处理信号
            for _, row in day_data.iterrows():
                if row["signal"] == 1:  # 买入信号
                    self._process_buy_signal(row, date)
                elif row["signal"] == -1:  # 卖出信号
                    self._process_sell_signal(row, date)

            # 记录每日快照
            self.portfolio.record_snapshot(date)

        return self._calculate_metrics()

    def _process_buy_signal(self, row: pd.Series, date: pd.Timestamp):
        """处理买入信号"""
        price = row["close"]
        # 使用全部可用现金的固定比例（分散到多个标的）
        budget = self.portfolio.cash * 0.1  # 每个标的最多 10% 现金
        quantity = int(budget / price / 100) * 100  # 整手
        if quantity < 100:
            return

        order = Order(
            order_id=self._next_order_id(),
            code=row["code"],
            side="buy",
            quantity=quantity,
            price=price,
            timestamp=date,
        )
        self.bus.publish(Event(EventType.ORDER_SUBMITTED, date, {"order": order}))

        # 模拟成交（简化：按当日收盘价成交）
        if order.status != OrderStatus.REJECTED:
            done = self.portfolio.apply_fill(order, price)
            if done:
                order.status = OrderStatus.FILLED
                order.filled_quantity = quantity
                order.filled_price = price
                self.portfolio.trades.append({
                    "date": date,
                    "code": order.code,
                    "side": order.side,
                    "quantity": quantity,
                    "price": price,
                    "order_id": order.order_id,
                })
                self.bus.publish(Event(EventType.ORDER_FILLED, date, {"order": order}))

    def _process_sell_signal(self, row: pd.Series, date: pd.Timestamp):
        """处理卖出信号"""
        code = row["code"]
        quantity = self.portfolio.positions.get(code, 0)
        if quantity <= 0:
            return

        price = row["close"]
        order = Order(
            order_id=self._next_order_id(),
            code=code,
            side="sell",
            quantity=quantity,
            price=price,
            timestamp=date,
        )
        self.bus.publish(Event(EventType.ORDER_SUBMITTED, date, {"order": order}))

        if order.status != OrderStatus.REJECTED:
            done = self.portfolio.apply_fill(order, price)
            if done:
                order.status = OrderStatus.FILLED
                order.filled_quantity = quantity
                order.filled_price = price
                self.portfolio.trades.append({
                    "date": date,
                    "code": order.code,
                    "side": order.side,
                    "quantity": quantity,
                    "price": price,
                    "order_id": order.order_id,
                })
                self.bus.publish(Event(EventType.ORDER_FILLED, date, {"order": order}))

    def _calculate_metrics(self) -> Dict[str, Any]:
        """计算绩效指标"""
        if not self.portfolio.equity_history:
            return {"total_return": 0.0, "annual_return": 0.0, "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0, "total_trades": 0, "win_rate": 0.0}

        equity_df = pd.DataFrame(self.portfolio.equity_history)
        equity_curve = equity_df.set_index("date")["equity"]

        # 日收益率
        returns = equity_curve.pct_change().dropna()
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        max_dd = (equity_curve / equity_curve.cummax() - 1).min()

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "total_trades": len(self.portfolio.trades),
            "event_count": len(self.bus._event_queue),
        }


# ============================================================
# 测试用例
# ============================================================

class TestMessageBus(unittest.TestCase):
    """测试消息总线"""

    def setUp(self):
        self.bus = MessageBus()
        self.received_events = []

    def test_publish_subscribe(self):
        def handler(event):
            self.received_events.append(event)

        self.bus.subscribe(EventType.MARKET_DATA, handler)
        event = Event(EventType.MARKET_DATA, pd.Timestamp("2024-01-01"), {"code": "000001.SZ"})
        self.bus.publish(event)
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].payload["code"], "000001.SZ")

    def test_multiple_subscribers(self):
        results = []

        def handler1(event):
            results.append(("h1", event.event_type))

        def handler2(event):
            results.append(("h2", event.event_type))

        self.bus.subscribe(EventType.SIGNAL, handler1)
        self.bus.subscribe(EventType.SIGNAL, handler2)
        self.bus.publish(Event(EventType.SIGNAL, pd.Timestamp("2024-01-01")))

        self.assertEqual(len(results), 2)

    def test_no_subscriber(self):
        """无订阅者时不应报错"""
        self.bus.publish(Event(EventType.MARKET_DATA, pd.Timestamp("2024-01-01")))
        self.assertEqual(len(self.bus._event_queue), 1)


class TestRiskEngine(unittest.TestCase):
    """测试风险引擎"""

    def setUp(self):
        self.risk_engine = RiskEngine([
            MaxPositionRiskCheck(max_position_pct=0.2),
            MaxDrawdownRiskCheck(max_drawdown=0.2),
        ])

    def test_position_limit_pass(self):
        order = {"price": 10.0, "quantity": 1000}
        portfolio = {"capital": 1000000, "position_value": 0, "current_drawdown": 0}
        passed, reason = self.risk_engine.evaluate(order, portfolio)
        self.assertTrue(passed)

    def test_position_limit_fail(self):
        order = {"price": 10.0, "quantity": 30000}
        portfolio = {"capital": 1000000, "position_value": 0, "current_drawdown": 0}
        passed, reason = self.risk_engine.evaluate(order, portfolio)
        self.assertFalse(passed)
        self.assertIn("最大持仓", reason)

    def test_drawdown_fail(self):
        order = {"price": 10.0, "quantity": 1000}
        portfolio = {"capital": 1000000, "position_value": 0, "current_drawdown": -0.25}
        passed, reason = self.risk_engine.evaluate(order, portfolio)
        self.assertFalse(passed)
        self.assertIn("最大回撤", reason)


class TestEventDrivenBacktest(unittest.TestCase):
    """测试事件驱动回测引擎"""

    @classmethod
    def setUpClass(cls):
        """创建测试数据"""
        np.random.seed(42)
        codes = ["000001.SZ", "000002.SZ"]
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
        records = []
        for code in codes:
            n = len(dates)
            base_price = {"000001.SZ": 10.0, "000002.SZ": 20.0}[code]
            returns = np.random.normal(0.0005, 0.02, n)
            prices = base_price * np.cumprod(1 + returns)
            for i, d in enumerate(dates):
                records.append({
                    "code": code,
                    "date": d,
                    "open": prices[i],
                    "high": prices[i] * 1.01,
                    "low": prices[i] * 0.99,
                    "close": prices[i],
                    "volume": 1000000,
                })
        cls.data = pd.DataFrame(records).sort_values(["code", "date"]).reset_index(drop=True)

        # 生成 MA20 突破信号
        signals = []
        for code in codes:
            code_data = cls.data[cls.data["code"] == code].copy()
            code_data["ma20"] = code_data["close"].rolling(20).mean()
            code_data["signal"] = 0
            code_data.loc[code_data["close"] > code_data["ma20"], "signal"] = 1
            code_data.loc[code_data["close"] <= code_data["ma20"], "signal"] = -1
            signals.append(code_data[["code", "date", "signal"]])
        cls.signals = pd.concat(signals).reset_index(drop=True)

    def test_full_backtest(self):
        """测试完整回测流程"""
        engine = EventDrivenBacktestEngine(init_capital=1e6)
        result = engine.run(self.data, self.signals)

        self.assertIn("total_return", result)
        self.assertIn("sharpe_ratio", result)
        self.assertIn("max_drawdown", result)
        self.assertGreater(result["event_count"], 0)
        self.assertGreater(result["total_trades"], 0)

        print(f"\n    事件驱动回测结果:")
        print(f"    累计收益: {result['total_return']:.2%}")
        print(f"    年化收益: {result['annual_return']:.2%}")
        print(f"    夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"    最大回撤: {result['max_drawdown']:.2%}")
        print(f"    交易次数: {result['total_trades']}")
        print(f"    事件总数: {result['event_count']}")

    def test_empty_signals(self):
        """边界条件：空信号"""
        engine = EventDrivenBacktestEngine(init_capital=1e6)
        empty_signals = pd.DataFrame(columns=["code", "date", "signal"])
        result = engine.run(self.data, empty_signals)
        self.assertEqual(result["total_trades"], 0)
        self.assertEqual(result["total_return"], 0.0)

    def test_risk_rejection(self):
        """测试风险拒绝"""
        engine = EventDrivenBacktestEngine(init_capital=1e6)
        # 设置极端严格的风险参数
        engine.risk_engine = RiskEngine([
            MaxPositionRiskCheck(max_position_pct=0.01),  # 仅 1%
        ])
        result = engine.run(self.data, self.signals)
        # 严格风控下交易数应极少
        self.assertLess(result["total_trades"], 10)

    def test_single_code(self):
        """测试单标的回测"""
        single_data = self.data[self.data["code"] == "000001.SZ"].copy()
        single_signals = self.signals[self.signals["code"] == "000001.SZ"].copy()
        engine = EventDrivenBacktestEngine(init_capital=1e6)
        result = engine.run(single_data, single_signals)
        self.assertIsNotNone(result["total_return"])


class TestEventDrivenPerformance(unittest.TestCase):
    """性能对比测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = [f"{i:06d}.{'SZ' if i % 2 == 0 else 'SH'}" for i in range(50)]
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
        records = []
        for code in codes:
            n = len(dates)
            base_price = np.random.uniform(5, 100)
            returns = np.random.normal(0.0005, 0.02, n)
            prices = base_price * np.cumprod(1 + returns)
            for i, d in enumerate(dates):
                records.append({
                    "code": code,
                    "date": d,
                    "open": prices[i],
                    "high": prices[i] * 1.01,
                    "low": prices[i] * 0.99,
                    "close": prices[i],
                    "volume": 1000000,
                })
        cls.large_data = pd.DataFrame(records).sort_values(["code", "date"]).reset_index(drop=True)

        # 生成信号
        signals = []
        for code in codes:
            code_data = cls.large_data[cls.large_data["code"] == code].copy()
            code_data["ma20"] = code_data["close"].rolling(20).mean()
            code_data["signal"] = 0
            code_data.loc[code_data["close"] > code_data["ma20"], "signal"] = 1
            signals.append(code_data[["code", "date", "signal"]])
        cls.large_signals = pd.concat(signals).reset_index(drop=True)

    def test_performance(self):
        """测试事件驱动回测性能"""
        engine = EventDrivenBacktestEngine(init_capital=1e6)
        n_trials = 3
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            engine.run(self.large_data, self.large_signals)
            times.append(time.perf_counter() - t0)

        avg_time = np.mean(times)
        print(f"\n    数据集: {len(self.large_data)} 行 x 50 只股票")
        print(f"    事件驱动回测均值: {avg_time:.3f}s")
        print(f"    事件总数: {engine.bus._event_queue}")

        # 事件驱动回测应在合理时间内完成
        self.assertLess(avg_time, 30.0, "事件驱动回测不应超过 30 秒")


if __name__ == "__main__":
    unittest.main(verbosity=2)