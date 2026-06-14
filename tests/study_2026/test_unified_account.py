"""
优化方向: 统一账户/仓位模型 (Unified Account & Position Model)
借鉴来源: QUANTAXIS (https://github.com/yutiansut/QUANTAXIS)
  - QUANTAXIS 的 QIFI (QUANTAXIS Interactive Financial Interface) 协议
    定义了统一的账户模型，使得回测和实盘使用相同的账户结构
  - 核心价值: 回测和实盘账户一致性，减少回测到实盘的部署风险
  - 参考文件: QUANTAXIS/QIFI/QifiAccount.py, QUANTAXIS/QARSBridge/
对比对象: jingni-trader 当前回测引擎中直接使用 dict 存储持仓
          (skills/backtest-engine/scripts/adapters/native_adapter.py positions = {})
          portfolio-risk-engine 中缺乏统一的仓位管理
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from copy import deepcopy
import json


# ============================================================
# 1. 统一账户模型核心实现
# ============================================================

class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """订单"""
    order_id: str
    code: str
    side: OrderSide
    price: float
    quantity: int
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    reason: str = ""  # 下单原因

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "code": self.code,
            "side": self.side.value,
            "price": self.price,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "commission": self.commission,
            "tax": self.tax,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "reason": self.reason,
        }


@dataclass
class Trade:
    """成交记录"""
    trade_id: str
    order_id: str
    code: str
    side: OrderSide
    price: float
    quantity: int
    amount: float
    commission: float = 0.0
    tax: float = 0.0
    trade_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "code": self.code,
            "side": self.side.value,
            "price": self.price,
            "quantity": self.quantity,
            "amount": self.amount,
            "commission": self.commission,
            "tax": self.tax,
            "trade_time": self.trade_time.isoformat(),
        }


@dataclass
class Position:
    """持仓"""
    code: str
    side: PositionSide = PositionSide.LONG
    quantity: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_cost: float = 0.0  # 含手续费的总成本

    def update_market_value(self, current_price: float):
        """更新市值和未实现盈亏"""
        self.market_value = self.quantity * current_price
        if self.quantity > 0:
            self.unrealized_pnl = self.market_value - self.total_cost

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "side": self.side.value,
            "quantity": self.quantity,
            "avg_cost": round(self.avg_cost, 4),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_cost": round(self.total_cost, 2),
        }


@dataclass
class AccountSnapshot:
    """账户快照（用于风控和审计）"""
    timestamp: datetime
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_equity: float = 0.0
    total_market_value: float = 0.0
    total_pnl: float = 0.0
    position_count: int = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash": round(self.cash, 2),
            "total_equity": round(self.total_equity, 2),
            "total_market_value": round(self.total_market_value, 2),
            "total_pnl": round(self.total_pnl, 2),
            "position_count": self.position_count,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
        }


class UnifiedAccount:
    """
    统一账户模型 - 借鉴 QIFI 协议设计

    核心设计原则:
    1. 回测和实盘使用相同的账户结构
    2. 订单→成交→持仓的完整生命周期管理
    3. 内置风控检查
    4. 支持账户快照（用于审计和回放）
    5. 序列化支持（JSON 兼容）

    与 QIFI 的对应关系:
    - QIFI_Account → UnifiedAccount (账户主类)
    - QIFI Position → Position (持仓)
    - QIFI Order → Order (订单)
    - QIFI Trade → Trade (成交)
    """

    def __init__(
        self,
        account_id: str = "default",
        init_cash: float = 1_000_000.0,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.001,
        risk_limits: Dict[str, float] = None,
    ):
        self.account_id = account_id
        self.initial_cash = init_cash
        self.cash = init_cash
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.risk_limits = risk_limits or {
            "max_position_pct": 0.05,  # 单票最大仓位
            "max_daily_loss": 0.02,     # 单日最大亏损
            "max_positions": 20,        # 最大持仓数
        }

        # 持仓
        self.positions: Dict[str, Position] = {}

        # 订单和成交记录
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.trades: List[Trade] = []
        self._order_counter = 0
        self._trade_counter = 0

        # 快照历史
        self.snapshots: List[AccountSnapshot] = []

        # 当日统计
        self._day_start_equity = init_cash
        self._current_date = None

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD{self._order_counter:08d}"

    def _next_trade_id(self) -> str:
        self._trade_counter += 1
        return f"TRD{self._trade_counter:08d}"

    # ---- 订单管理 ----

    def submit_order(
        self,
        code: str,
        side: OrderSide,
        price: float,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        reason: str = "",
    ) -> Order:
        """提交订单"""
        order = Order(
            order_id=self._next_order_id(),
            code=code,
            side=side,
            price=price,
            quantity=quantity,
            order_type=order_type,
            reason=reason,
        )

        # 风控检查
        reject_reason = self._risk_check(order)
        if reject_reason:
            order.status = OrderStatus.REJECTED
            order.reason = f"风控拒绝: {reject_reason}"
            self.filled_orders.append(order)
            return order

        self.pending_orders.append(order)
        return order

    def fill_order(self, order: Order, fill_price: float, fill_quantity: int = None,
                   current_date: Any = None) -> Trade:
        """成交订单"""
        if fill_quantity is None:
            fill_quantity = order.quantity

        fill_quantity = min(fill_quantity, order.quantity - order.filled_quantity)

        if fill_quantity <= 0:
            return None

        # 计算成本
        amount = fill_price * fill_quantity
        commission = max(amount * self.commission_rate, self.min_commission)
        tax = amount * self.stamp_tax_rate if order.side == OrderSide.SELL else 0

        # 更新订单
        order.filled_quantity += fill_quantity
        order.filled_price = (
            (order.filled_price * (order.filled_quantity - fill_quantity) + fill_price * fill_quantity)
            / order.filled_quantity
        )
        order.commission += commission
        order.tax += tax
        order.filled_at = datetime.now()

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            self.pending_orders = [o for o in self.pending_orders if o.order_id != order.order_id]
            self.filled_orders.append(order)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        # 创建成交记录
        trade = Trade(
            trade_id=self._next_trade_id(),
            order_id=order.order_id,
            code=order.code,
            side=order.side,
            price=fill_price,
            quantity=fill_quantity,
            amount=amount,
            commission=commission,
            tax=tax,
        )
        self.trades.append(trade)

        # 更新现金和持仓
        if order.side == OrderSide.BUY:
            self.cash -= (amount + commission)
            self._update_position(order.code, fill_quantity, fill_price, amount + commission)
        else:
            self.cash += (amount - commission - tax)
            self._update_position(order.code, -fill_quantity, fill_price, 0)

        return trade

    def cancel_order(self, order: Order):
        """撤销订单"""
        if order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
            order.status = OrderStatus.CANCELLED
            self.pending_orders = [o for o in self.pending_orders if o.order_id != order.order_id]
            self.filled_orders.append(order)

    # ---- 持仓管理 ----

    def _update_position(self, code: str, delta_quantity: int,
                         price: float, cost: float):
        """更新持仓"""
        if code not in self.positions:
            self.positions[code] = Position(code=code, side=PositionSide.LONG)

        pos = self.positions[code]

        if delta_quantity > 0:  # 加仓
            new_total_qty = pos.quantity + delta_quantity
            pos.total_cost += cost
            pos.avg_cost = pos.total_cost / new_total_qty if new_total_qty > 0 else 0
            pos.quantity = new_total_qty
        else:  # 减仓
            sell_qty = -delta_quantity
            if sell_qty > pos.quantity:
                sell_qty = pos.quantity

            realized_pnl = (price - pos.avg_cost) * sell_qty
            pos.realized_pnl += realized_pnl
            pos.quantity -= sell_qty
            pos.total_cost -= pos.avg_cost * sell_qty

            if pos.quantity <= 0:
                del self.positions[code]

    def update_market_prices(self, prices: Dict[str, float]):
        """更新所有持仓的市值"""
        for code, pos in self.positions.items():
            if code in prices:
                pos.update_market_value(prices[code])

    def get_position(self, code: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(code)

    # ---- 风控 ----

    def _risk_check(self, order: Order) -> Optional[str]:
        """风控检查"""
        limits = self.risk_limits

        # 持仓数量检查
        if order.side == OrderSide.BUY:
            current_positions = len([p for p in self.positions.values() if p.quantity > 0])
            if order.code not in self.positions and current_positions >= limits.get("max_positions", 20):
                return f"持仓数量已达上限 {limits['max_positions']}"

        # 单票仓位检查
        if order.side == OrderSide.BUY:
            current_eq = self.total_equity
            if current_eq > 0:
                order_amount = order.price * order.quantity
                position_pct = order_amount / current_eq
                if position_pct > limits.get("max_position_pct", 0.05):
                    return f"单票仓位 {position_pct:.2%} 超过上限 {limits['max_position_pct']:.2%}"

        # 现金检查
        if order.side == OrderSide.BUY:
            estimated_cost = order.price * order.quantity * (1 + self.commission_rate)
            if estimated_cost > self.cash:
                return "现金不足"

        return None

    # ---- 账户快照 ----

    def take_snapshot(self, current_date: Any = None) -> AccountSnapshot:
        """创建账户快照"""
        total_mv = sum(p.market_value for p in self.positions.values())
        total_equity = self.cash + total_mv
        total_pnl = total_equity - self.initial_cash

        snapshot = AccountSnapshot(
            timestamp=datetime.now(),
            cash=self.cash,
            positions=deepcopy(self.positions),
            total_equity=total_equity,
            total_market_value=total_mv,
            total_pnl=total_pnl,
            position_count=len([p for p in self.positions.values() if p.quantity > 0]),
        )
        self.snapshots.append(snapshot)
        return snapshot

    # ---- 属性 ----

    @property
    def total_equity(self) -> float:
        total_mv = sum(p.market_value for p in self.positions.values())
        return self.cash + total_mv

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.total_equity - self.initial_cash

    @property
    def position_count(self) -> int:
        return len([p for p in self.positions.values() if p.quantity > 0])

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "total_equity": round(self.total_equity, 2),
            "total_market_value": round(self.total_market_value, 2),
            "total_pnl": round(self.total_pnl, 2),
            "position_count": self.position_count,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "pending_orders": [o.to_dict() for o in self.pending_orders],
            "trades": [t.to_dict() for t in self.trades[-10:]],  # 最近10笔
            "risk_limits": self.risk_limits,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)


# ============================================================
# 2. 测试用例
# ============================================================

class TestOrderLifecycle(unittest.TestCase):
    """订单生命周期测试"""

    def setUp(self):
        self.account = UnifiedAccount(account_id="test_001", init_cash=1_000_000.0)

    def test_submit_and_fill_buy_order(self):
        """测试提交和成交买入订单"""
        order = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=2000, reason="测试买入"  # 降低数量避免风控拒绝
        )
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.assertEqual(len(self.account.pending_orders), 1)

        trade = self.account.fill_order(order, fill_price=10.0)
        self.assertIsNotNone(trade)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(len(self.account.pending_orders), 0)
        self.assertEqual(len(self.account.filled_orders), 1)

        # 检查持仓
        pos = self.account.get_position("000001.SZ")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.quantity, 2000)
        self.assertAlmostEqual(pos.avg_cost, 10.0, delta=0.01)

        # 检查现金
        expected_cost = 10.0 * 2000 + max(10.0 * 2000 * 0.00025, 5.0)
        self.assertAlmostEqual(self.account.cash, 1_000_000.0 - expected_cost, delta=0.01)

    def test_submit_and_fill_sell_order(self):
        """测试提交和成交卖出订单"""
        # 先买入
        order_buy = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=2000
        )
        self.account.fill_order(order_buy, fill_price=10.0)

        # 再卖出
        order_sell = self.account.submit_order(
            code="000001.SZ", side=OrderSide.SELL,
            price=12.0, quantity=2000
        )
        trade = self.account.fill_order(order_sell, fill_price=12.0)

        self.assertIsNotNone(trade)
        # 卖出后持仓应为0
        pos = self.account.get_position("000001.SZ")
        self.assertIsNone(pos)

        # 卖出产生印花税
        self.assertGreater(trade.tax, 0)

    def test_partial_fill(self):
        """测试部分成交"""
        order = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=2000
        )

        trade1 = self.account.fill_order(order, fill_price=10.0, fill_quantity=1000)
        self.assertEqual(order.status, OrderStatus.PARTIALLY_FILLED)
        self.assertEqual(order.filled_quantity, 1000)

        trade2 = self.account.fill_order(order, fill_price=10.5, fill_quantity=1000)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.filled_quantity, 2000)

        # 均价应为 (10*1000 + 10.5*1000) / 2000 = 10.25
        pos = self.account.get_position("000001.SZ")
        self.assertAlmostEqual(pos.avg_cost, 10.25, delta=0.02)

    def test_cancel_order(self):
        """测试撤销订单"""
        order = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=2000  # 降低数量以避免风控拒绝
        )
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.account.cancel_order(order)
        self.assertEqual(order.status, OrderStatus.CANCELLED)
        self.assertEqual(len(self.account.pending_orders), 0)

    def test_order_rejection_due_to_cash(self):
        """测试现金不足时的订单拒绝"""
        order = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=1000000.0, quantity=2000
        )
        self.assertEqual(order.status, OrderStatus.REJECTED)


class TestPositionManagement(unittest.TestCase):
    """持仓管理测试"""

    def setUp(self):
        self.account = UnifiedAccount(account_id="test_002", init_cash=1_000_000.0)

    def test_buy_multiple_stocks(self):
        """测试买入多只股票"""
        codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
        for code in codes:
            order = self.account.submit_order(
                code=code, side=OrderSide.BUY,
                price=10.0, quantity=1000
            )
            self.account.fill_order(order, fill_price=10.0)

        self.assertEqual(self.account.position_count, 3)

    def test_update_market_prices(self):
        """测试更新市值"""
        order = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=2000
        )
        self.account.fill_order(order, fill_price=10.0)

        # 价格上涨
        self.account.update_market_prices({"000001.SZ": 12.0})
        pos = self.account.get_position("000001.SZ")
        self.assertEqual(pos.market_value, 24000.0)
        self.assertGreater(pos.unrealized_pnl, 0)

    def test_realized_pnl(self):
        """测试已实现盈亏"""
        order_buy = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=2000
        )
        self.account.fill_order(order_buy, fill_price=10.0)

        # 卖出1000股，价格12
        order_sell = self.account.submit_order(
            code="000001.SZ", side=OrderSide.SELL,
            price=12.0, quantity=1000
        )
        self.account.fill_order(order_sell, fill_price=12.0)

        pos = self.account.get_position("000001.SZ")
        self.assertEqual(pos.quantity, 1000)
        # 已实现盈亏 = (12-10)*1000 = 2000，减去佣金和税费
        self.assertGreater(pos.realized_pnl, 0)

    def test_position_removal_on_zero(self):
        """测试持仓清零后删除"""
        order_buy = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=2000
        )
        self.account.fill_order(order_buy, fill_price=10.0)
        self.assertIn("000001.SZ", self.account.positions)

        order_sell = self.account.submit_order(
            code="000001.SZ", side=OrderSide.SELL,
            price=10.0, quantity=2000
        )
        self.account.fill_order(order_sell, fill_price=10.0)
        self.assertNotIn("000001.SZ", self.account.positions)


class TestRiskManagement(unittest.TestCase):
    """风控测试"""

    def setUp(self):
        self.account = UnifiedAccount(
            account_id="test_003",
            init_cash=1_000_000.0,
            risk_limits={
                "max_position_pct": 0.1,
                "max_daily_loss": 0.05,
                "max_positions": 3,
            }
        )

    def test_position_limit(self):
        """测试持仓数量限制"""
        codes = [f"{i:06d}.SZ" for i in range(1, 6)]
        for code in codes[:3]:
            order = self.account.submit_order(
                code=code, side=OrderSide.BUY,
                price=10.0, quantity=1000
            )
            self.account.fill_order(order, fill_price=10.0)

        # 第4个应该被拒绝
        order = self.account.submit_order(
            code=codes[3], side=OrderSide.BUY,
            price=10.0, quantity=1000
        )
        self.assertEqual(order.status, OrderStatus.REJECTED)

    def test_single_position_limit(self):
        """测试单票仓位限制"""
        # 买入超过10%仓位
        order = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=15000  # 150k > 10% of 1M
        )
        self.assertEqual(order.status, OrderStatus.REJECTED)

    def test_snapshot(self):
        """测试账户快照"""
        order = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=2000
        )
        self.account.fill_order(order, fill_price=10.0)
        self.account.update_market_prices({"000001.SZ": 10.5})

        snapshot = self.account.take_snapshot()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.position_count, 1)
        self.assertIn("000001.SZ", snapshot.positions)
        self.assertEqual(len(self.account.snapshots), 1)

    def test_multiple_snapshots(self):
        """测试多次快照"""
        for i in range(10):
            self.account.take_snapshot()
        self.assertEqual(len(self.account.snapshots), 10)


class TestSerialization(unittest.TestCase):
    """序列化测试"""

    def setUp(self):
        self.account = UnifiedAccount(account_id="test_004", init_cash=1_000_000.0)
        order = self.account.submit_order(
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, quantity=2000
        )
        self.account.fill_order(order, fill_price=10.0)

    def test_to_dict(self):
        """测试转换为字典"""
        d = self.account.to_dict()
        self.assertEqual(d["account_id"], "test_004")
        self.assertIn("positions", d)
        self.assertIn("trades", d)
        self.assertIn("risk_limits", d)

    def test_to_json(self):
        """测试转换为JSON"""
        j = self.account.to_json()
        self.assertIsInstance(j, str)
        parsed = json.loads(j)
        self.assertEqual(parsed["account_id"], "test_004")

    def test_position_serialization(self):
        """测试持仓序列化"""
        pos = self.account.get_position("000001.SZ")
        d = pos.to_dict()
        self.assertEqual(d["code"], "000001.SZ")
        self.assertEqual(d["quantity"], 2000)
        self.assertIn("avg_cost", d)
        self.assertIn("market_value", d)


class TestAccountIntegration(unittest.TestCase):
    """集成测试：模拟完整回测流程"""

    def test_full_backtest_simulation(self):
        """模拟完整回测流程"""
        np.random.seed(42)
        account = UnifiedAccount(
            account_id="backtest_001",
            init_cash=1_000_000.0,
        )

        n_dates = 100
        codes = [f"{i:06d}.SZ" for i in range(1, 11)]
        dates = pd.date_range('2023-01-01', periods=n_dates, freq='B')

        # 生成模拟价格
        prices = {}
        for code in codes:
            base = np.random.uniform(5, 50)
            prices[code] = base + np.cumsum(np.random.randn(n_dates) * 0.3)

        equity_curve = []

        for t in range(n_dates):
            dt = dates[t]

            # 更新市值
            current_prices = {code: prices[code][t] for code in codes}
            account.update_market_prices(current_prices)

            # 每10天调仓
            if t % 10 == 0:
                # 卖出所有
                for code, pos in list(account.positions.items()):
                    order = account.submit_order(
                        code=code, side=OrderSide.SELL,
                        price=current_prices[code], quantity=pos.quantity,
                        reason="定期调仓"
                    )
                    account.fill_order(order, fill_price=current_prices[code])

                # 买入前3只
                selected = codes[:3]
                budget = account.cash * 0.3 / len(selected)
                for code in selected:
                    price = current_prices[code]
                    qty = int(budget / price / 100) * 100
                    if qty > 0:
                        order = account.submit_order(
                            code=code, side=OrderSide.BUY,
                            price=price, quantity=qty, reason="定期调仓"
                        )
                        account.fill_order(order, fill_price=price)

            # 记录快照
            snapshot = account.take_snapshot()
            equity_curve.append(snapshot.total_equity)

        # 验证
        self.assertEqual(len(account.snapshots), n_dates)
        self.assertGreater(len(account.trades), 0)

        # 检查净值曲线变化
        self.assertNotEqual(equity_curve[0], equity_curve[-1])

        print(f"\n  初始权益: {equity_curve[0]:.2f}")
        print(f"  最终权益: {equity_curve[-1]:.2f}")
        print(f"  总收益: {(equity_curve[-1]/equity_curve[0] - 1)*100:.2f}%")
        print(f"  总成交笔数: {len(account.trades)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)