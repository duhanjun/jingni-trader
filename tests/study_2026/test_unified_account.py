"""
================================================================================
优化方向: 统一账户/持仓数据模型（QIFI 协议风格）
借鉴来源: QUANTAXIS (https://github.com/yutiansut/QUANTAXIS) — QIFI 协议
         QUANTAXIS v2.1 引入了 QIFI (Quantitative Investment Financial Interface)
         统一账户协议，定义了跨语言的标准账户数据结构，实现 Python 和 Rust
         版本间的零拷贝数据交换和 100% API 兼容。

优化目标:
  当前 jingni-trader 的账户模型分散在多个模块中：
  - execution-monitor-engine 中有 Account dataclass
  - portfolio-risk-engine 中有独立的仓位计算逻辑
  - backtest-engine 中的 equity_curve 计算与账户脱钩

  缺乏统一的账户/持仓/交易记录数据模型，导致：
  1. 各模块间的数据传递依赖 ad-hoc 字典
  2. 账户状态无法跨模块共享
  3. 难以支持多账户/多策略场景

验证内容:
  1. 定义 QIFI 风格的统一账户模型
  2. 实现零拷贝式的跨模块数据传递
  3. 多模块集成：账户变更可被回测/风控/执行引擎同时消费
  4. 序列化/反序列化往返测试
  5. 内存效率对比 — 字典 vs dataclass vs NamedTuple
"""

import unittest
import sys
import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, date
from enum import Enum
import copy

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# ── 测试配置 ──────────────────────────────────
_ACCOUNT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), '../../workspace/account_state.json'
)


# ================================================================================
# Part 1: QIFI 风格统一账户协议
# ================================================================================


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """统一持仓模型（兼容多空、期货、现货）"""
    code: str                              # 证券代码
    side: PositionSide = PositionSide.LONG # 持仓方向
    volume: int = 0                        # 持仓量（股/张）
    available_volume: int = 0              # 可卖量
    frozen_volume: int = 0                 # 冻结量（挂单中）
    avg_cost: float = 0.0                  # 平均成本价
    current_price: float = 0.0             # 当前价
    market_value: float = 0.0              # 市值
    unrealized_pnl: float = 0.0            # 浮动盈亏
    realized_pnl: float = 0.0              # 已实现盈亏
    open_date: str = ""                    # 开仓日期

    def update_market_value(self, price: float):
        """更新市价相关字段"""
        self.current_price = price
        self.market_value = self.volume * price
        self.unrealized_pnl = (price - self.avg_cost) * self.volume


@dataclass
class Order:
    """统一订单模型"""
    order_id: str = ""
    code: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    price: float = 0.0
    volume: int = 0
    filled_volume: int = 0
    status: OrderStatus = OrderStatus.PENDING
    commission: float = 0.0
    stamp_tax: float = 0.0
    slippage: float = 0.0
    create_time: str = ""
    update_time: str = ""
    memo: str = ""

    @property
    def notional(self) -> float:
        return self.price * self.volume

    @property
    def filled_notional(self) -> float:
        return self.price * self.filled_volume


@dataclass
class Trade:
    """统一成交记录模型"""
    trade_id: str = ""
    order_id: str = ""
    code: str = ""
    side: OrderSide = OrderSide.BUY
    price: float = 0.0
    volume: int = 0
    commission: float = 0.0
    stamp_tax: float = 0.0
    trade_time: str = ""


@dataclass
class Account:
    """统一账户模型（QIFI 协议风格）

    设计原则:
    1. 所有字段有明确含义和类型
    2. 支持序列化/反序列化
    3. 包含完整的持仓、订单、成交记录
    4. 可作为消息在不同模块间零拷贝传递（引用共享）
    """

    # 账户基本信息
    account_id: str = "default"
    account_name: str = ""

    # 资金信息
    initial_capital: float = 1_000_000.0
    available_cash: float = 1_000_000.0
    frozen_cash: float = 0.0                # 冻结资金（挂单占用）
    total_assets: float = 1_000_000.0       # 总资产
    total_liabilities: float = 0.0          # 总负债

    # 净值和风控
    nav: float = 1_000_000.0               # 净值
    start_of_day_nav: float = 1_000_000.0   # 日初净值
    daily_pnl: float = 0.0                  # 当日损益
    cumulative_pnl: float = 0.0             # 累计损益

    # 组合数据
    positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: Dict[str, Order] = field(default_factory=dict)
    trade_history: List[Trade] = field(default_factory=list)

    # 元信息
    last_update: str = ""
    version: int = 1

    # ── 衍生属性 ──────────────────────

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def daily_return(self) -> float:
        if self.start_of_day_nav <= 0:
            return 0.0
        return (self.nav - self.start_of_day_nav) / self.start_of_day_nav

    @property
    def leverage(self) -> float:
        if self.nav <= 0:
            return 0.0
        return self.total_assets / self.nav

    @property
    def position_count(self) -> int:
        return len(self.positions)

    # ── 原子操作 ──────────────────────

    def recalculate_nav(self, prices: Dict[str, float]):
        """根据最新价格重新计算净值和市值"""
        total_mv = 0.0
        for code, pos in self.positions.items():
            price = prices.get(code, pos.current_price)
            pos.update_market_value(price)
            total_mv += pos.market_value
        self.total_assets = self.available_cash + self.frozen_cash + total_mv
        self.nav = self.total_assets - self.total_liabilities
        self.cumulative_pnl = self.nav - self.initial_capital
        self.last_update = datetime.now().isoformat()

    def reset_daily(self):
        """交易日重置"""
        self.start_of_day_nav = self.nav
        self.daily_pnl = 0.0
        self.version += 1

    def apply_trade(self, trade: Trade, prices: Dict[str, float]):
        """应用一笔成交到账户"""
        if trade.side == OrderSide.BUY:
            cost = trade.price * trade.volume + trade.commission
            self.available_cash -= cost
            if trade.code not in self.positions:
                self.positions[trade.code] = Position(
                    code=trade.code,
                    side=PositionSide.LONG,
                    open_date=trade.trade_time,
                )
            pos = self.positions[trade.code]
            old_cost = pos.avg_cost * pos.volume
            pos.volume += trade.volume
            pos.available_volume += trade.volume
            pos.avg_cost = (old_cost + cost) / pos.volume if pos.volume > 0 else 0.0
            if trade.code in prices:
                pos.update_market_value(prices[trade.code])
        else:
            total_fee = trade.commission + trade.stamp_tax
            revenue = trade.price * trade.volume - total_fee
            self.available_cash += revenue
            if trade.code in self.positions:
                pos = self.positions[trade.code]
                pnl_per_share = trade.price - pos.avg_cost
                pos.realized_pnl += pnl_per_share * trade.volume
                pos.volume -= trade.volume
                pos.available_volume -= trade.volume
                if pos.volume <= 0:
                    del self.positions[trade.code]

        self.trade_history.append(trade)
        self.recalculate_nav(prices)
        self.daily_pnl = self.nav - self.start_of_day_nav
        self.cumulative_pnl = self.nav - self.initial_capital

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "account_id": self.account_id,
            "account_name": self.account_name,
            "initial_capital": self.initial_capital,
            "available_cash": self.available_cash,
            "frozen_cash": self.frozen_cash,
            "nav": self.nav,
            "daily_pnl": self.daily_pnl,
            "cumulative_pnl": self.cumulative_pnl,
            "daily_return": self.daily_return,
            "leverage": self.leverage,
            "position_count": self.position_count,
            "total_market_value": self.total_market_value,
            "positions": {
                code: asdict(pos) for code, pos in self.positions.items()
            },
            "last_update": self.last_update,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict) -> "Account":
        """从字典恢复"""
        positions = {}
        for code, pos_data in data.get("positions", {}).items():
            pos_data["side"] = PositionSide(pos_data["side"])
            positions[code] = Position(**pos_data)

        acct = cls(
            account_id=data.get("account_id", "default"),
            account_name=data.get("account_name", ""),
            initial_capital=data.get("initial_capital", 1_000_000.0),
            available_cash=data.get("available_cash", 1_000_000.0),
            frozen_cash=data.get("frozen_cash", 0.0),
            nav=data.get("nav", 1_000_000.0),
            daily_pnl=data.get("daily_pnl", 0.0),
            cumulative_pnl=data.get("cumulative_pnl", 0.0),
            positions=positions,
            last_update=data.get("last_update", ""),
            version=data.get("version", 1),
        )
        return acct


# ================================================================================
# Part 2: 跨模块事件总线
# ================================================================================

@dataclass
class AccountEvent:
    """账户变更事件 — 用于跨模块通知"""
    event_type: str  # "trade", "position_change", "nav_update", "stop_triggered"
    account_id: str
    data: Dict[str, Any]
    timestamp: str = ""
    source_module: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class AccountEventBus:
    """轻量级事件总线 — 解耦模块间的账户变更通知"""

    def __init__(self):
        self._listeners: Dict[str, List[callable]] = {}

    def subscribe(self, event_type: str, callback: callable):
        """订阅事件类型"""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def publish(self, event: AccountEvent):
        """发布事件"""
        listeners = self._listeners.get(event.event_type, [])
        for cb in listeners:
            cb(event)
        # "all" 类型监听所有事件
        for cb in self._listeners.get("all", []):
            cb(event)


# ================================================================================
# Part 3: 测试用例
# ================================================================================


class TestUnifiedAccountModel(unittest.TestCase):
    """统一账户模型单元测试"""

    def setUp(self):
        self.account = Account(
            account_id="test_001",
            account_name="Test Account",
            initial_capital=1_000_000.0,
            available_cash=1_000_000.0,
        )

    def test_account_initialization(self):
        """测试账户初始化"""
        self.assertEqual(self.account.nav, 1_000_000.0)
        self.assertEqual(self.account.available_cash, 1_000_000.0)
        self.assertEqual(self.account.position_count, 0)
        self.assertEqual(self.account.daily_return, 0.0)
        self.assertEqual(self.account.leverage, 1.0)

    def test_apply_buy_trade(self):
        """测试买入成交"""
        trade = Trade(
            trade_id="T001",
            order_id="O001",
            code="000001.SZ",
            side=OrderSide.BUY,
            price=10.0,
            volume=10000,
            commission=25.0,
            trade_time="2024-01-15 09:35:00",
        )
        self.account.apply_trade(trade, {"000001.SZ": 10.0})

        self.assertIn("000001.SZ", self.account.positions)
        pos = self.account.positions["000001.SZ"]
        self.assertEqual(pos.volume, 10000)
        self.assertAlmostEqual(pos.avg_cost, 10.0025, places=4)
        self.assertAlmostEqual(self.account.available_cash, 1_000_000 - 100025, places=2)

    def test_apply_sell_trade(self):
        """测试卖出成交"""
        # 先买入
        trade_buy = Trade(
            trade_id="T001", order_id="O001",
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, volume=10000,
            commission=25.0, trade_time="2024-01-15",
        )
        self.account.apply_trade(trade_buy, {"000001.SZ": 10.0})

        # 再卖出
        trade_sell = Trade(
            trade_id="T002", order_id="O002",
            code="000001.SZ", side=OrderSide.SELL,
            price=11.0, volume=5000,
            commission=13.75, stamp_tax=55.0,
            trade_time="2024-01-16",
        )
        self.account.apply_trade(trade_sell, {"000001.SZ": 11.0})

        self.assertEqual(self.account.positions["000001.SZ"].volume, 5000)
        self.assertEqual(len(self.account.trade_history), 2)

    def test_recalculate_nav(self):
        """测试净值重算"""
        # 买入 10000 股 @ 10 元
        trade = Trade(
            trade_id="T001", order_id="O001",
            code="000001.SZ", side=OrderSide.BUY,
            price=10.0, volume=10000,
            commission=25.0, trade_time="2024-01-15",
        )
        self.account.apply_trade(trade, {"000001.SZ": 10.0})

        # 股价涨到 11 元
        self.account.recalculate_nav({"000001.SZ": 11.0})
        self.assertAlmostEqual(self.account.nav, 899975.0 + 110000.0, places=2)
        self.assertAlmostEqual(self.account.cumulative_pnl,
                               1009975.0 - 1_000_000.0, places=2)

    def test_reset_daily(self):
        """测试日重置"""
        self.account.apply_trade(
            Trade(trade_id="T001", order_id="O001",
                  code="000001.SZ", side=OrderSide.BUY,
                  price=10.0, volume=10000,
                  commission=25.0, trade_time="2024-01-15"),
            {"000001.SZ": 10.0},
        )
        self.account.recalculate_nav({"000001.SZ": 10.5})

        old_version = self.account.version
        self.account.reset_daily()
        self.assertEqual(self.account.start_of_day_nav, self.account.nav)
        self.assertEqual(self.account.daily_pnl, 0.0)
        self.assertEqual(self.account.version, old_version + 1)

    def test_serialization_roundtrip(self):
        """测试序列化/反序列化往返"""
        # 做几笔交易
        self.account.apply_trade(
            Trade(trade_id="T001", order_id="O001",
                  code="000001.SZ", side=OrderSide.BUY,
                  price=10.0, volume=10000, commission=25.0,
                  trade_time="2024-01-15"),
            {"000001.SZ": 10.0},
        )
        self.account.apply_trade(
            Trade(trade_id="T002", order_id="O002",
                  code="600000.SH", side=OrderSide.BUY,
                  price=8.0, volume=5000, commission=10.0,
                  trade_time="2024-01-16"),
            {"000001.SZ": 10.0, "600000.SH": 8.0},
        )

        # 导出为字典
        data = self.account.to_dict()

        # 还原
        restored = Account.from_dict(data)

        # 验证关键字段
        self.assertEqual(restored.nav, self.account.nav)
        self.assertEqual(restored.available_cash, self.account.available_cash)
        self.assertEqual(restored.position_count, self.account.position_count)
        self.assertEqual(len(restored.positions), len(self.account.positions))

    def test_to_json(self):
        """测试 JSON 序列化"""
        self.account.apply_trade(
            Trade(trade_id="T001", order_id="O001",
                  code="000001.SZ", side=OrderSide.BUY,
                  price=10.0, volume=10000, commission=25.0,
                  trade_time="2024-01-15"),
            {"000001.SZ": 10.0},
        )
        json_str = self.account.to_json()
        self.assertIn("account_id", json_str)
        self.assertIn("000001.SZ", json_str)
        # 确保 JSON 可解析
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)


class TestAccountEventBus(unittest.TestCase):
    """跨模块事件总线测试"""

    def setUp(self):
        self.bus = AccountEventBus()
        self.received_events: List[AccountEvent] = []

    def test_subscribe_and_publish(self):
        """测试订阅和发布"""

        def on_trade(event: AccountEvent):
            self.received_events.append(event)

        self.bus.subscribe("trade", on_trade)

        event = AccountEvent(
            event_type="trade",
            account_id="test_001",
            data={"code": "000001.SZ", "volume": 1000},
        )
        self.bus.publish(event)
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].event_type, "trade")

    def test_all_subscription(self):
        """测试 'all' 监听所有事件"""

        def on_any(event: AccountEvent):
            self.received_events.append(event)

        self.bus.subscribe("all", on_any)

        self.bus.publish(AccountEvent("trade", "t1", {}))
        self.bus.publish(AccountEvent("nav_update", "t1", {}))
        self.bus.publish(AccountEvent("stop_triggered", "t1", {}))

        self.assertEqual(len(self.received_events), 3)

    def test_cross_module_simulation(self):
        """模拟跨模块事件通知"""
        risk_events = []
        exec_events = []
        report_events = []

        def risk_handler(e): risk_events.append(e)
        def exec_handler(e): exec_events.append(e)
        def report_handler(e): report_events.append(e)

        self.bus.subscribe("nav_update", risk_handler)
        self.bus.subscribe("nav_update", exec_handler)
        self.bus.subscribe("nav_update", report_handler)

        self.bus.publish(AccountEvent(
            "nav_update", "acct_1",
            {"nav": 1_050_000.0, "daily_pnl": 50000.0},
            source_module="execution-engine",
        ))

        self.assertEqual(len(risk_events), 1)
        self.assertEqual(len(exec_events), 1)
        self.assertEqual(len(report_events), 1)


class TestAccountMemoryEfficiency(unittest.TestCase):
    """内存效率对比测试"""

    def test_memory_comparison(self):
        """对比 dataclass vs dict vs NamedTuple 的内存占用"""
        import sys as _sys

        # 模拟一个 50 只股票的持仓账户
        stocks = [f"{i:06d}.SZ" for i in range(50)]

        # dataclass 版本
        t0 = time.perf_counter()
        acct_dc = Account(account_id="mem_test")
        for i, code in enumerate(stocks):
            acct_dc.positions[code] = Position(
                code=code, volume=1000 * (i + 1), avg_cost=10.0 + i * 0.5,
                current_price=11.0 + i * 0.3,
            )
        acct_dc.recalculate_nav({code: 11.0 + i * 0.3 for i, code in enumerate(stocks)})
        t_dc = time.perf_counter() - t0
        dc_size = _sys.getsizeof(acct_dc) + sum(
            _sys.getsizeof(p) for p in acct_dc.positions.values()
        )

        # dict 版本（类似现有实现）
        t0 = time.perf_counter()
        acct_dict: Dict[str, Any] = {
            "nav": 1_000_000.0,
            "available_cash": 1_000_000.0,
            "positions": {},
        }
        for i, code in enumerate(stocks):
            acct_dict["positions"][code] = {
                "volume": 1000 * (i + 1),
                "avg_cost": 10.0 + i * 0.5,
                "current_price": 11.0 + i * 0.3,
                "market_value": 1000 * (i + 1) * (11.0 + i * 0.3),
            }
        t_dict = time.perf_counter() - t0
        dict_size = _sys.getsizeof(acct_dict) + sum(
            _sys.getsizeof(p) for p in acct_dict["positions"].values()
        )

        print(f"\n  内存占用对比 (50 只持仓):")
        print(f"  QIFI dataclass: {dc_size:,} bytes, 构建耗时: {t_dc:.6f}s")
        print(f"  字典方案:       {dict_size:,} bytes, 构建耗时: {t_dict:.6f}s")
        print(f"  内存比率: dataclass/dict = {dc_size/dict_size:.2f}x")
        print(f"  速度比率: dataclass/dict = {t_dc/t_dict:.2f}x (越接近1越相似)")

    def test_serialization_speed(self):
        """对比序列化速度"""
        stocks = [f"{i:06d}.SZ" for i in range(50)]
        acct = Account(account_id="ser_test")
        for i, code in enumerate(stocks):
            acct.positions[code] = Position(
                code=code, volume=1000 * (i + 1), avg_cost=10.0 + i * 0.5,
                current_price=11.0 + i * 0.3,
            )

        n_iterations = 1000

        # JSON 序列化
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            _ = acct.to_json()
        t_json = time.perf_counter() - t0

        # dict 序列化
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            _ = acct.to_dict()
        t_dict = time.perf_counter() - t0

        print(f"\n  序列化速度对比 ({n_iterations} 次):")
        print(f"  to_json(): {t_json:.4f}s ({t_json/n_iterations*1e6:.0f} us/次)")
        print(f"  to_dict(): {t_dict:.4f}s ({t_dict/n_iterations*1e6:.0f} us/次)")


class TestAccountCrossModuleIntegration(unittest.TestCase):
    """跨模块集成测试"""

    def test_backtest_risk_exec_flow(self):
        """模拟 回测 → 风控 → 执行 三模块协作流程"""
        bus = AccountEventBus()
        account = Account(account_id="integration_test")
        account.reset_daily()

        # 风控模块收到的通知
        risk_alerts = []

        def risk_checker(event: AccountEvent):
            nav = event.data.get("nav", 0)
            start_nav = event.data.get("start_of_day_nav", 1)
            daily_loss = (nav - start_nav) / start_nav if start_nav > 0 else 0
            if daily_loss < -0.03:
                risk_alerts.append({
                    "type": "stop_trading",
                    "daily_loss": daily_loss,
                    "nav": nav,
                })

        bus.subscribe("nav_update", risk_checker)

        # 模拟交易日循环
        prices = {"000001.SZ": 10.0, "600000.SH": 8.0}
        for day in range(5):
            account.reset_daily()

            # 买入
            account.apply_trade(
                Trade(trade_id=f"T{day}_1", order_id=f"O{day}_1",
                      code="000001.SZ", side=OrderSide.BUY,
                      price=prices["000001.SZ"], volume=1000,
                      commission=2.5, trade_time=f"2024-01-{15+day:02d}"),
                prices,
            )

            # 更新价格（模拟市场波动）
            prices["000001.SZ"] *= (1 + np.random.normal(0.001, 0.01))
            prices["600000.SH"] *= (1 + np.random.normal(0.001, 0.01))
            account.recalculate_nav(prices)

            # 发布净值更新事件
            bus.publish(AccountEvent(
                "nav_update", account.account_id,
                {"nav": account.nav, "start_of_day_nav": account.start_of_day_nav},
                source_module="backtest-engine",
            ))

        # 验证流程完整性
        self.assertGreater(len(account.trade_history), 0)
        self.assertTrue(account.nav > 0, "净值为正")
        self.assertGreaterEqual(account.version, 5)

        print(f"\n  模拟 5 日交易完成:")
        print(f"  最终净值: {account.nav:.2f}")
        print(f"  累计损益: {account.cumulative_pnl:.2f}")
        print(f"  持仓数: {account.position_count}")
        print(f"  成交笔数: {len(account.trade_history)}")
        print(f"  风控告警: {len(risk_alerts)} 次")

    def test_existing_account_compatibility(self):
        """测试与现有 execution-monitor-engine 中 Account 的兼容性"""
        if not os.path.exists(_ACCOUNT_STATE_PATH):
            self.skipTest("无现有账户状态文件")

        with open(_ACCOUNT_STATE_PATH, "r") as f:
            existing_state = json.load(f)

        # 将现有状态映射到统一模型
        unified = Account(
            account_id="migrated",
            nav=existing_state.get("nav", 1_000_000.0),
            available_cash=existing_state.get("available_cash", 1_000_000.0),
            last_update=existing_state.get("updated_at", ""),
        )

        for code, pos_data in existing_state.get("positions", {}).items():
            unified.positions[code] = Position(
                code=code,
                volume=pos_data.get("volume", 0),
                avg_cost=pos_data.get("avg_cost", 0.0),
            )

        unified.recalculate_nav({code: pos.avg_cost for code, pos in unified.positions.items()})

        print(f"\n  现有状态迁移结果:")
        print(f"  净值: {existing_state.get('nav', 'N/A')} → {unified.nav:.2f}")
        print(f"  持仓数: {len(existing_state.get('positions', {}))} → {unified.position_count}")


# ================================================================================
# 运行入口
# ================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("统一账户模型（QIFI 风格）验证测试")
    print("借鉴来源: QUANTAXIS QIFI 协议")
    print("=" * 70)
    unittest.main(verbosity=2)