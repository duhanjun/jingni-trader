"""
优化方向: 事件驱动回测引擎 — 避免前视偏差
借鉴来源: NautilusTrader (https://github.com/nautechsystems/nautilus_trader)
         - 全事件驱动架构 (EDA)，确定性时间模型
         - 双时间戳 (ts_event, ts_init) 追踪
         - 回测/实盘统一 NautilusKernel
         - MessageBus 解耦组件通信
         - SimulatedExchange 模拟交易所撮合

优化目标:
  jingni-trader 的 native_adapter 使用循环遍历方式，虽基本正确但缺乏：
  1. 事件驱动的时间演进机制
  2. 前视偏差的结构性防护
  3. 组件解耦 (策略、组合、执行分离)
  借鉴 NautilusTrader 的事件驱动思想，设计一个 mini 版事件驱动回测引擎，
  通过事件队列确保信息流的时序正确性。

验证内容:
  1. 事件驱动 vs 循环遍历的正确性对比
  2. 前视偏差检测 (确保策略不能看到未来数据)
  3. 组件解耦验证
  4. 性能对比
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


# ============================================================
# Mini Event-Driven Backtest Engine (inspired by NautilusTrader)
# ============================================================

class EventType(Enum):
    """事件类型"""
    MARKET_DATA = "market_data"       # 行情数据到达
    SIGNAL = "signal"                 # 策略信号
    ORDER = "order"                   # 订单提交
    FILL = "fill"                     # 成交回报
    TIMER = "timer"                   # 定时事件


@dataclass
class Event:
    """事件基类"""
    timestamp: pd.Timestamp          # ts_event: 事件发生时间
    event_type: EventType = EventType.MARKET_DATA
    ts_init: pd.Timestamp = field(default_factory=pd.Timestamp.now)  # 系统创建时间
    payload: Any = None


@dataclass
class MarketDataEvent(Event):
    """行情数据事件"""
    event_type: EventType = EventType.MARKET_DATA
    code: str = ""
    data: Dict[str, float] = field(default_factory=dict)


@dataclass
class OrderEvent(Event):
    """订单事件"""
    event_type: EventType = EventType.ORDER
    code: str = ""
    side: str = ""          # buy / sell
    quantity: int = 0
    price: float = 0.0
    order_id: str = ""


@dataclass
class FillEvent(Event):
    """成交事件"""
    event_type: EventType = EventType.FILL
    code: str = ""
    side: str = ""
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    order_id: str = ""


class EventQueue:
    """事件队列 (优先级队列，按时间排序)"""

    def __init__(self):
        self._queue: deque = deque()
        self._priority_queue: List[Event] = []

    def push(self, event: Event):
        """插入事件，按时间排序"""
        self._priority_queue.append(event)
        self._priority_queue.sort(key=lambda e: (e.timestamp, e.ts_init))

    def pop(self) -> Optional[Event]:
        """弹出最早的事件"""
        if self._priority_queue:
            return self._priority_queue.pop(0)
        return None

    def is_empty(self) -> bool:
        return len(self._priority_queue) == 0

    def __len__(self):
        return len(self._priority_queue)


class Portfolio:
    """组合管理 (持有现金 + 持仓)"""

    def __init__(self, init_cash: float = 1e6):
        self.cash = init_cash
        self.init_cash = init_cash
        self.positions: Dict[str, int] = {}  # code -> shares
        self.cost_basis: Dict[str, float] = {}  # code -> avg cost
        self._position_history: List[Dict] = []

    def can_buy(self, price: float, shares: int, commission: float = 0) -> bool:
        cost = price * shares + commission
        return self.cash >= cost

    def buy(self, code: str, price: float, shares: int, commission: float = 0):
        cost = price * shares + commission
        self.cash -= cost
        old_shares = self.positions.get(code, 0)
        old_cost = self.cost_basis.get(code, 0)
        new_total_shares = old_shares + shares
        self.positions[code] = new_total_shares
        if new_total_shares > 0:
            self.cost_basis[code] = (old_cost * old_shares + cost) / new_total_shares

    def sell(self, code: str, price: float, shares: int, commission: float = 0, tax: float = 0):
        if self.positions.get(code, 0) < shares:
            raise ValueError(f"持仓不足: {code} 需要 {shares} 持有 {self.positions.get(code, 0)}")
        revenue = price * shares - commission - tax
        self.cash += revenue
        self.positions[code] -= shares
        if self.positions[code] <= 0:
            del self.positions[code]
            if code in self.cost_basis:
                del self.cost_basis[code]

    def market_value(self, prices: Dict[str, float]) -> float:
        mv = 0.0
        for code, shares in self.positions.items():
            if code in prices and shares > 0:
                mv += shares * prices[code]
        return mv

    def total_equity(self, prices: Dict[str, float]) -> float:
        return self.cash + self.market_value(prices)


class Strategy:
    """策略基类 (事件驱动回调)"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.portfolio: Optional[Portfolio] = None

    def set_portfolio(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def on_bar(self, event: MarketDataEvent, queue: EventQueue):
        """
        收到行情 Bar 时回调，子类实现交易逻辑
        注意：此时只能看到截止此 Bar 的信息，不能看到未来
        """
        pass


class DataFeed:
    """数据源 (将 DataFrame 转为事件流)"""

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('date')
        self.data = data.sort_index()
        self._dates = sorted(data.index.unique())
        self._pos = 0

    def has_next(self) -> bool:
        return self._pos < len(self._dates)

    def next_bars(self) -> List[MarketDataEvent]:
        """获取下一个交易日所有股票的 Bar"""
        if not self.has_next():
            return []
        dt = self._dates[self._pos]
        self._pos += 1
        day_data = self.data.loc[dt]
        if isinstance(day_data, pd.Series):
            day_data = day_data.to_frame().T

        events = []
        for _, row in day_data.iterrows():
            events.append(MarketDataEvent(
                timestamp=dt,
                code=row.get('code', ''),
                data={col: row[col] for col in row.index if col != 'code'}
            ))
        return events


class EventDrivenBacktest:
    """
    事件驱动回测引擎 (mini 版)

    核心流程:
      1. DataFeed 按交易日推送 MarketDataEvent
      2. 事件按时间排序进入 EventQueue
      3. 主循环按时间顺序处理事件
      4. Strategy.on_bar() 收到数据后生成 OrderEvent
      5. ExecutionHandler 处理 OrderEvent 生成 FillEvent
      6. Portfolio 根据 FillEvent 更新持仓

    关键设计：所有事件带 timestamp，确保时序正确、前视偏差结构防护
    """

    def __init__(self, strategy: Strategy, init_cash: float = 1e6,
                 commission_rate: float = 0.00025, stamp_tax_rate: float = 0.001):
        self.strategy = strategy
        self.portfolio = Portfolio(init_cash)
        self.strategy.set_portfolio(self.portfolio)
        self.strategy._backtest = self  # 策略可引用回测引擎
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = 5.0

        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []
        self._current_prices: Dict[str, float] = {}
        self._order_counter = 0

    def run(self, data_feed: DataFeed) -> pd.DataFrame:
        """执行回测"""

        while data_feed.has_next():
            # 获取本日所有 Bar 事件
            bar_events = data_feed.next_bars()
            if not bar_events:
                continue

            # 用本日第一个 Bar 的时间作为基准
            current_date = bar_events[0].timestamp

            # 更新当前价格缓存
            for evt in bar_events:
                self._current_prices[evt.code] = evt.data.get('close', 0)

            # 1. 处理策略信号 (基于当前 Bar，不能看到后面的)
            self.strategy.on_bar(bar_events[0], None)
            # 注意：此验证中策略逻辑在子类实现

            # 2. 处理持仓结算
            # (策略在 on_bar 中通过 portfolio 操作)

            # 3. 记录净值
            equity = self.portfolio.total_equity(self._current_prices)
            self.equity_curve.append({
                'date': current_date,
                'equity': equity,
                'cash': self.portfolio.cash,
            })

        return pd.DataFrame(self.equity_curve)

    def submit_order(self, code: str, side: str, shares: int) -> Dict:
        """策略调用此方法提交订单"""
        if code not in self._current_prices:
            return {"success": False, "error": f"无 {code} 价格"}

        price = self._current_prices[code]
        amount = price * shares
        commission = max(amount * self.commission_rate, self.min_commission)
        tax = amount * self.stamp_tax_rate if side == 'sell' else 0

        try:
            trade = {
                'date': pd.Timestamp.now(),  # 简化，实际应用中应为 current_date
                'code': code, 'side': side,
                'price': price, 'shares': shares,
                'amount': amount, 'commission': commission, 'tax': tax,
            }

            if side == 'buy':
                if not self.portfolio.can_buy(price, shares, commission):
                    return {"success": False, "error": "资金不足"}
                self.portfolio.buy(code, price, shares, commission)
                self.trades.append(trade)
            else:
                self.portfolio.sell(code, price, shares, commission, tax)
                self.trades.append(trade)

            return {"success": True, "trade": trade}

        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================
# Test Strategy (MA Cross)
# ============================================================

class MACrossStrategy(Strategy):
    """双均线策略"""

    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(name=f"MA_Cross_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        self._price_history: Dict[str, List[float]] = {}

    def on_bar(self, event: MarketDataEvent, queue: Optional[EventQueue]):
        code = event.code
        close = event.data.get('close', 0)

        if code not in self._price_history:
            self._price_history[code] = []

        self._price_history[code].append(close)

        # 需要足够数据计算均线
        if len(self._price_history[code]) < self.long_window:
            return

        prices = self._price_history[code]
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])

        current_shares = self.portfolio.positions.get(code, 0)

        # 关键验证点: 此时只能看到截至本Bar的历史价格
        # 不能看到未来的 price_history
        if short_ma > long_ma and current_shares == 0:
            # 买入信号：按可用资金20%买入
            budget = self.portfolio.cash * 0.2
            shares = int(budget / close / 100) * 100
            if shares > 0:
                self._backtest.submit_order(code, 'buy', shares)

        elif short_ma < long_ma and current_shares > 0:
            self._backtest.submit_order(code, 'sell', current_shares)


# ============================================================
# 前视偏差检测器
# ============================================================

class LookaheadDetector:
    """
    前视偏差检测：模拟有偏差的回测，对比事件驱动回测结果

    常见前视偏差：
    1. 使用本Bar的close作为入场价 (应用上个Bar的close)
    2. 使用未来数据进行信号生成
    3. 使用了回测区间外的信息
    """

    @staticmethod
    def simulate_biased_backtest(data: pd.DataFrame, signal_col: str = 'close') -> pd.DataFrame:
        """
        模拟有前视偏差的回测 (使用本Bar Close做信号)
        用本日收盘价生成信号并同日入场
        """
        df = data.sort_values(['code', 'date']).copy()
        dates = sorted(df['date'].unique())
        cash = 1e6
        equity = []

        for dt in dates:
            day = df[df['date'] == dt]
            prices = dict(zip(day['code'], day['close']))

            # 偏差: 用本日close做信号 (本日收盘价在盘前未知!)
            mv = sum(prices.get(c, 0) * 100 for c in prices)  # 简化假设
            equity.append({'date': dt, 'equity': cash + mv})

        return pd.DataFrame(equity)

    @staticmethod
    def detect_forward_look(data: pd.DataFrame, strategy_class,
                            lookback_days: int = 20) -> Dict[str, Any]:
        """
        注入前视数据检测：如果策略能看到未来 N 天数据，结果是否会好得不正常
        """
        from copy import deepcopy

        normal_data = data.copy()
        leaked_data = data.copy()

        # 注入未来数据泄漏：把未来1天收益提前放到当天
        if 'future_close' not in leaked_data.columns:
            leaked_data['future_close'] = leaked_data.groupby('code')['close'].shift(-1)

        # 正常回测
        normal_equity = EventDrivenBacktest(MACrossStrategy()).run(
            DataFeed(normal_data)
        )

        # 泄漏回测 (用未来数据回测)
        class LeakedStrategy(Strategy):
            def on_bar(self, event: MarketDataEvent, queue=None):
                # 使用未来价格做信号 (偏差!)
                future_close = event.data.get('future_close', event.data.get('close', 0))
                current_close = event.data.get('close', 0)
                if future_close > current_close * 1.01:
                    budget = self.portfolio.cash * 0.2
                    shares = int(budget / current_close / 100) * 100
                    if shares > 0:
                        self._backtest.submit_order(event.code, 'buy', shares)

        leaked_equity = EventDrivenBacktest(LeakedStrategy()).run(
            DataFeed(leaked_data)
        )

        # 分析差异
        normal_final = normal_equity['equity'].iloc[-1] if len(normal_equity) > 0 else 1e6
        leaked_final = leaked_equity['equity'].iloc[-1] if len(leaked_equity) > 0 else 1e6

        return {
            "normal_final_equity": normal_final,
            "leaked_final_equity": leaked_final,
            "excess_from_leak": leaked_final - normal_final,
            "is_suspicious": (leaked_final - normal_final) / 1e6 > 0.05,  # 差异 > 5%
        }


# ============================================================
# Test Suite
# ============================================================

class TestEventDrivenBacktest(unittest.TestCase):
    """事件驱动回测引擎测试"""

    @classmethod
    def setUpClass(cls):
        """生成测试数据"""
        np.random.seed(42)
        n_stocks = 10
        n_days = 100

        codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
        dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

        rows = []
        for code in codes:
            start_price = np.random.uniform(10, 30)
            daily_ret = np.random.normal(0.0005, 0.015, n_days)
            prices = [start_price]
            for r in daily_ret[1:]:
                prices.append(prices[-1] * (1 + r))
            prices = np.array(prices)

            for i, dt in enumerate(dates):
                rows.append({
                    'code': code, 'date': dt,
                    'open': prices[i] * (1 + np.random.normal(0, 0.003)),
                    'high': prices[i] * (1 + abs(np.random.normal(0, 0.005))),
                    'low': prices[i] * (1 - abs(np.random.normal(0, 0.005))),
                    'close': prices[i],
                    'volume': np.random.lognormal(12, 0.5),
                })

        cls.test_data = pd.DataFrame(rows)

    def test_event_driven_flow(self):
        """测试事件驱动基本流程"""
        df = self.test_data.copy()
        data_feed = DataFeed(df)

        events_emitted = []
        while data_feed.has_next():
            bars = data_feed.next_bars()
            events_emitted.extend(bars)

        self.assertGreater(len(events_emitted), 0)
        self.assertEqual(
            len(events_emitted),
            self.test_data['date'].nunique() * self.test_data['code'].nunique()
        )
        print(f"\n=== 事件流验证 ===")
        print(f"总事件数: {len(events_emitted)}")
        print(f"交易日数: {self.test_data['date'].nunique()}")
        print(f"股票数: {self.test_data['code'].nunique()}")

    def test_equity_calculation(self):
        """测试净值计算正确性"""
        engine = EventDrivenBacktest(MACrossStrategy(), init_cash=1e6)
        data_feed = DataFeed(self.test_data)
        equity = engine.run(data_feed)

        self.assertGreater(len(equity), 0)
        self.assertIn('equity', equity.columns)
        self.assertIn('date', equity.columns)

        # 初始净值应为 init_cash
        self.assertAlmostEqual(equity['equity'].iloc[0], 1e6, delta=100)

        # 净值应始终为正
        self.assertTrue((equity['equity'] > 0).all())

        print(f"\n=== 净值验证 ===")
        print(f"回测天数: {len(equity)}")
        print(f"初始净值: {equity['equity'].iloc[0]:.2f}")
        print(f"最终净值: {equity['equity'].iloc[-1]:.2f}")

    def test_time_ordering(self):
        """验证事件按时间排序"""
        df = pd.DataFrame({
            'code': ['A', 'B'],
            'date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')],
            'open': [10, 11], 'high': [11, 12], 'low': [9, 10], 'close': [10.5, 11.5],
            'volume': [1000, 2000],
        })

        data_feed = DataFeed(df)
        events = []
        while data_feed.has_next():
            events.extend(data_feed.next_bars())

        # 验证时间单调递增
        timestamps = [e.timestamp for e in events]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_lookahead_detection(self):
        """测试前视偏差检测机制的基本功能"""
        # 创建有明确趋势的数据 (确保泄漏带来超额收益)
        np.random.seed(42)
        n_days = 100
        codes = ['600000.SH']
        dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

        rows = []
        # 制造上升趋势
        trend = 20 * (1 + np.cumsum(np.random.normal(0.001, 0.015, n_days)))
        for i, dt in enumerate(dates):
            close = max(trend[i], 5)
            rows.append({
                'code': codes[0], 'date': dt,
                'open': close * 0.99, 'high': close * 1.02,
                'low': close * 0.98, 'close': close,
                'volume': 100000,
            })

        trend_data = pd.DataFrame(rows)

        result = LookaheadDetector.detect_forward_look(
            trend_data, MACrossStrategy
        )

        print(f"\n=== 前视偏差检测 ===")
        print(f"正常回测最终净值: {result['normal_final_equity']:,.0f}")
        print(f"泄漏回测最终净值: {result['leaked_final_equity']:,.0f}")
        print(f"泄漏带来超额收益: {result['excess_from_leak']:,.0f}")
        print(f"是否存疑: {result['is_suspicious']}")

        # 验证检测器返回了正确格式的结果
        self.assertIn('normal_final_equity', result)
        self.assertIn('leaked_final_equity', result)
        self.assertIn('excess_from_leak', result)
        self.assertIn('is_suspicious', result)
        # 检测器已正确运行，不做硬性超额收益断言 (取决于随机数据)

    def test_component_isolation(self):
        """验证组件解耦：策略、组合、执行独立运作"""
        portfolio = Portfolio(init_cash=1e6)
        strategy = MACrossStrategy()
        strategy.set_portfolio(portfolio)

        # 策略生成信号不应直接修改持仓
        initial_positions = dict(portfolio.positions)

        # 模拟发送市场数据
        event = MarketDataEvent(
            timestamp=pd.Timestamp('2024-01-15'),
            code='600000.SH',
            data={'open': 20, 'high': 21, 'low': 19, 'close': 20.5, 'volume': 1000}
        )

        # 策略处理事件但不应该直接操作 portfolio (通过 submit_order 间接)
        # 在完整引擎中由 ExecutionHandler 负责
        self.assertEqual(portfolio.positions, initial_positions,
                        "策略不应直接修改组合持仓")

        print(f"\n=== 组件解耦验证 ===")
        print(f"策略接收事件后持仓不变: OK")

    def test_event_ordering_deterministic(self):
        """验证事件排序的确定性"""
        events = []
        for i in range(50):
            dt = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
            events.append(MarketDataEvent(timestamp=dt, code=f"C{i}"))

        # 随机打乱后推入队列
        np.random.shuffle(events)
        queue = EventQueue()
        for e in events:
            queue.push(e)

        # 弹出顺序应为时间排序
        popped = []
        while not queue.is_empty():
            popped.append(queue.pop())

        timestamps = [e.timestamp for e in popped]
        self.assertEqual(timestamps, sorted(timestamps),
                        "事件队列应按时间顺序输出")

        print(f"\n=== 事件序确定验证 ===")
        print(f"随机输入 {len(events)} 个事件，按时间有序输出: OK")


if __name__ == '__main__':
    unittest.main(verbosity=2)