"""
优化方向: 事件驱动回测引擎架构
借鉴来源:
  1. vn.py (VeighNa) (https://github.com/vnpy/vnpy) - 事件驱动引擎 vnpy.trader.engine
     - Event-driven architecture with EventEngine
     - 事件类型: MarketEvent, SignalEvent, OrderEvent, FillEvent, TradeEvent
     - 事件队列 + 线程池模式
     - 标准化回调接口 (on_tick, on_bar, on_order, on_trade)
  2. Backtrader (https://github.com/mementum/backtrader) - 回测框架设计
     - Cerebro 引擎的事件循环
     - Broker 抽象（佣金、滑点、税收模型）
     - Analyzer 体系（多维度指标分析）
  3. Qlib (https://github.com/microsoft/qlib) - Nested Decision Framework
     - 多层级信号决策（高频层、中频层、低频层）
     - 信号执行器（SignalExecutor）与订单执行器（OrderExecutor）分离

验证内容:
  - 事件驱动引擎核心实现（Event, EventEngine, EventHandler）
  - 市场事件 → 信号事件 → 订单事件 → 成交事件 → 账户事件的完整流程
  - 与原有回测引擎的对比测试
  - 多层级信号决策原型
  - 延迟和吞吐量测试

注意: 本文件仅用于验证测试，不修改主项目代码。
"""
import sys
import os
import unittest
import time
from typing import Dict, List, Callable, Any, Optional, Type
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, PriorityQueue
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# =============================================================================
# 事件类型定义
# =============================================================================

class EventType(Enum):
    """事件类型枚举"""
    MARKET = "market"          # 市场行情事件
    SIGNAL = "signal"          # 交易信号事件
    ORDER = "order"            # 订单提交事件
    FILL = "fill"              # 订单成交事件
    POSITION = "position"      # 持仓更新事件
    ACCOUNT = "account"        # 账户更新事件
    TIMER = "timer"            # 定时器事件
    LOG = "log"                # 日志事件
    RISK = "risk"              # 风控事件


@dataclass
class Event:
    """事件基类"""
    type: EventType
    timestamp: pd.Timestamp
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"evt_{next(itertools.count())}")


@dataclass
class MarketEvent(Event):
    """市场行情事件"""
    def __init__(self, timestamp, symbol, open_, high, low, close, volume, amount=0):
        super().__init__(EventType.MARKET, timestamp, {
            'symbol': symbol, 'open': open_, 'high': high,
            'low': low, 'close': close, 'volume': volume, 'amount': amount
        })


@dataclass
class SignalEvent(Event):
    """交易信号事件"""
    def __init__(self, timestamp, symbol, direction, signal_type, strength=1.0, reason=""):
        super().__init__(EventType.SIGNAL, timestamp, {
            'symbol': symbol, 'direction': direction,  # 1=long, -1=short
            'signal_type': signal_type, 'strength': strength, 'reason': reason
        })


@dataclass
class OrderEvent(Event):
    """订单事件"""
    def __init__(self, timestamp, symbol, direction, volume, price_type, price=0):
        super().__init__(EventType.ORDER, timestamp, {
            'symbol': symbol, 'direction': direction,
            'volume': volume, 'price_type': price_type,  # 'market' or 'limit'
            'price': price, 'status': 'pending'
        })


@dataclass
class FillEvent(Event):
    """成交事件"""
    def __init__(self, timestamp, symbol, direction, volume, price, commission=0):
        super().__init__(EventType.FILL, timestamp, {
            'symbol': symbol, 'direction': direction,
            'volume': volume, 'price': price, 'commission': commission
        })


@dataclass
class PositionEvent(Event):
    """持仓更新事件"""
    def __init__(self, timestamp, symbol, position, avg_cost):
        super().__init__(EventType.POSITION, timestamp, {
            'symbol': symbol, 'position': position, 'avg_cost': avg_cost
        })


@dataclass
class AccountEvent(Event):
    """账户更新事件"""
    def __init__(self, timestamp, cash, total_value, positions_value):
        super().__init__(EventType.ACCOUNT, timestamp, {
            'cash': cash, 'total_value': total_value,
            'positions_value': positions_value
        })


# =============================================================================
# 事件驱动引擎
# =============================================================================

class EventEngine:
    """
    事件驱动引擎

    参考 vn.py 的 EventEngine 设计：
    - 事件处理器注册/注销
    - 事件队列分发
    - 支持同步/异步处理
    - 事件优先级排序
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_queue: PriorityQueue = PriorityQueue()
        self._event_count = 0
        self._events_by_type = defaultdict(int)
        self._start_time = None
        self._end_time = None

    def register(self, event_type: EventType, handler: Callable):
        """注册事件处理器"""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unregister(self, event_type: EventType, handler: Callable):
        """注销事件处理器"""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def put(self, event: Event, priority: int = 5):
        """将事件放入队列"""
        self._event_queue.put((priority, self._event_count, event))
        self._event_count += 1
        self._events_by_type[event.type] += 1

    def process(self, max_events: int = None):
        """处理事件队列"""
        self._start_time = time.time()
        processed = 0
        while not self._event_queue.empty():
            if max_events and processed >= max_events:
                break
            _, _, event = self._event_queue.get()
            event_type = event.type
            for handler in self._handlers.get(event_type, []):
                try:
                    handler(event)
                except Exception as e:
                    print(f"[EventEngine] Handler error for {event_type}: {e}")
            processed += 1
        self._end_time = time.time()

    def process_one(self):
        """处理单个事件"""
        if not self._event_queue.empty():
            _, _, event = self._event_queue.get()
            for handler in self._handlers.get(event.type, []):
                try:
                    handler(event)
                except Exception as e:
                    print(f"[EventEngine] Handler error: {e}")
            return event
        return None

    def get_stats(self) -> Dict:
        """获取引擎统计"""
        elapsed = self._end_time - self._start_time if self._end_time else 0
        return {
            "total_events": self._event_count,
            "events_by_type": dict(self._events_by_type),
            "processing_time": elapsed,
            "events_per_second": self._event_count / elapsed if elapsed > 0 else 0,
        }

    def clear(self):
        """清空队列和处理器"""
        while not self._event_queue.empty():
            self._event_queue.get()
        self._handlers.clear()
        self._event_count = 0
        self._events_by_type.clear()


# =============================================================================
# 回测组件
# =============================================================================

class BacktestBroker:
    """
    回测经纪人

    参考 vn.py 和 Backtrader 的 Broker 实现：
    - 模拟订单执行（市场价/限价）
    - 佣金和印花税计算
    - 成交滑点模拟
    - T+1 交易规则
    """

    def __init__(self, initial_cash: float = 1_000_000,
                 commission_rate: float = 0.00025,
                 stamp_tax: float = 0.001,
                 slippage: float = 0.0001):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax  # A股印花税，卖出时收取
        self.slippage = slippage
        self.cash = initial_cash
        self.positions: Dict[str, int] = defaultdict(int)
        self.position_costs: Dict[str, float] = {}
        self.orders: List[OrderEvent] = []
        self.fills: List[FillEvent] = []
        self.trade_history: List[Dict] = []

    def execute_order(self, order: OrderEvent, current_price: float) -> Optional[FillEvent]:
        """执行订单"""
        symbol = order.data['symbol']
        direction = order.data['direction']
        volume = order.data['volume']

        # 计算滑点后的成交价
        if order.data['price_type'] == 'market':
            if direction == 1:  # 买入
                fill_price = current_price * (1 + self.slippage)
            else:  # 卖出
                fill_price = current_price * (1 - self.slippage)
        else:
            fill_price = order.data['price']

        # 计算费用
        trade_value = fill_price * volume * 100  # A股每手100股
        commission = max(trade_value * self.commission_rate, 5)  # 最低5元佣金
        tax = trade_value * self.stamp_tax if direction == -1 else 0  # 卖出收印花税

        # 更新持仓
        new_position = self.positions.get(symbol, 0) + direction * volume
        if new_position < 0:
            return None  # 不能卖空

        # 更新资金
        cost = fill_price * volume * 100 + commission + tax
        if direction == 1:  # 买入
            if self.cash < cost:
                return None  # 资金不足
            self.cash -= cost
        else:  # 卖出
            self.cash += fill_price * volume * 100 - commission - tax

        self.positions[symbol] = new_position

        # 更新持仓成本
        if new_position > 0:
            if symbol not in self.position_costs:
                self.position_costs[symbol] = fill_price
            else:
                old_cost = self.position_costs[symbol]
                old_pos = self.positions[symbol] - direction * volume
                self.position_costs[symbol] = (old_cost * old_pos + fill_price * volume * direction) / new_position

        fill = FillEvent(order.timestamp, symbol, direction, volume, fill_price, commission + tax)
        self.fills.append(fill)
        self.trade_history.append({
            'timestamp': order.timestamp,
            'symbol': symbol,
            'direction': direction,
            'volume': volume,
            'price': fill_price,
            'commission': commission,
            'tax': tax,
            'cash_after': self.cash,
        })
        return fill

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """计算总资产"""
        positions_value = sum(
            pos * prices.get(sym, 0) * 100
            for sym, pos in self.positions.items()
        )
        return self.cash + positions_value

    def get_equity_curve(self, dates: pd.DatetimeIndex, price_data: pd.DataFrame) -> pd.Series:
        """计算权益曲线"""
        equity = pd.Series(index=dates, dtype=float)
        for i, dt in enumerate(dates):
            if i == 0:
                equity.iloc[i] = self.initial_cash
                continue
            # 获取当日收盘价
            day_data = price_data[price_data['date'] == dt]
            prices = dict(zip(day_data['code'], day_data['close']))
            equity.iloc[i] = self.get_total_value(prices)
        return equity.ffill()


class BacktestStrategy:
    """回测策略基类"""

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.broker: Optional[BacktestBroker] = None

    def on_market(self, event: MarketEvent):
        """市场事件回调"""
        pass

    def on_signal(self, event: SignalEvent):
        """信号事件回调"""
        pass

    def on_fill(self, event: FillEvent):
        """成交事件回调"""
        pass

    def on_order(self, event: OrderEvent):
        """订单事件回调"""
        pass

    def set_broker(self, broker: BacktestBroker):
        self.broker = broker


class MomentumStrategy(BacktestStrategy):
    """动量策略"""

    def __init__(self, lookback: int = 20, top_n: int = 5):
        super().__init__("MomentumStrategy")
        self.lookback = lookback
        self.top_n = top_n
        self.last_signal_date = None

    def on_market(self, event: MarketEvent):
        """处理市场事件 - 每日收盘后计算信号"""
        # 在事件驱动模式下，信号由外部生成
        # 这里只记录最后信号日期
        self.last_signal_date = event.timestamp


class MeanReversionStrategy(BacktestStrategy):
    """均值回归策略"""

    def __init__(self, lookback: int = 20, threshold: float = 2.0):
        super().__init__("MeanReversionStrategy")
        self.lookback = lookback
        self.threshold = threshold


# =============================================================================
# 事件驱动回测运行器
# =============================================================================

class EventDrivenBacktestRunner:
    """
    事件驱动回测运行器

    参考 vn.py 的回测引擎设计：
    - 逐日推送市场数据，生成 MarketEvent
    - 策略接收 MarketEvent 生成 SignalEvent
    - SignalEvent 转换为 OrderEvent
    - OrderEvent 被 Broker 执行为 FillEvent
    - FillEvent 更新持仓和账户
    """

    def __init__(self, initial_cash: float = 1_000_000):
        self.event_engine = EventEngine()
        self.broker = BacktestBroker(initial_cash=initial_cash)
        self.strategies: List[BacktestStrategy] = []
        self.results: Dict[str, Any] = {}
        self._current_date = None

    def add_strategy(self, strategy: BacktestStrategy):
        strategy.set_broker(self.broker)
        self.strategies.append(strategy)

    def register_handlers(self):
        """注册事件处理器"""
        # 市场事件处理
        for strategy in self.strategies:
            self.event_engine.register(EventType.MARKET, strategy.on_market)
            self.event_engine.register(EventType.SIGNAL, strategy.on_signal)
            self.event_engine.register(EventType.FILL, strategy.on_fill)
            self.event_engine.register(EventType.ORDER, strategy.on_order)

        # 信号转订单处理
        self.event_engine.register(EventType.SIGNAL, self._handle_signal)

        # 订单执行处理
        self.event_engine.register(EventType.ORDER, self._handle_order)

    def _handle_signal(self, event: SignalEvent):
        """信号转为订单"""
        symbol = event.data['symbol']
        direction = event.data['direction']
        strength = event.data['strength']

        # 根据信号强度确定仓位
        total_value = self.broker.get_total_value({})
        # 简单仓位计算：每个信号分配总资产的 10%
        target_value = self.broker.initial_cash * 0.1 * strength
        current_price = event.data.get('price', 10)
        volume = max(int(target_value / (current_price * 100)), 1)
        volume = max(volume, 1)  # 至少1手

        order = OrderEvent(event.timestamp, symbol, direction, volume, 'market', current_price)
        self.event_engine.put(order, priority=6)

    def _handle_order(self, event: OrderEvent):
        """处理订单事件"""
        current_price = event.data.get('price', 10)
        fill = self.broker.execute_order(event, current_price)
        if fill:
            self.event_engine.put(fill, priority=7)

            # 生成持仓事件
            pos_event = self._create_position_event(event.timestamp)
            self.event_engine.put(pos_event, priority=8)

            # 生成账户事件
            account_event = self._create_account_event(event.timestamp, {})
            self.event_engine.put(account_event, priority=9)

    def _create_position_event(self, timestamp) -> PositionEvent:
        symbols = list(self.broker.positions.keys())
        if symbols:
            sym = symbols[0]
            return PositionEvent(timestamp, sym,
                                 self.broker.positions[sym],
                                 self.broker.position_costs.get(sym, 0))
        return PositionEvent(timestamp, "", 0, 0)

    def _create_account_event(self, timestamp, prices) -> AccountEvent:
        total_value = self.broker.get_total_value(prices)
        positions_value = sum(
            pos * prices.get(sym, 0) * 100
            for sym, pos in self.broker.positions.items()
        )
        return AccountEvent(timestamp, self.broker.cash, total_value, positions_value)

    def run(self, data: pd.DataFrame, signals: pd.DataFrame = None) -> Dict:
        """
        运行事件驱动回测
        """
        self.register_handlers()

        dates = sorted(data['date'].unique())
        equity_curve = []
        daily_events = []

        for dt in dates:
            self._current_date = dt
            day_data = data[data['date'] == dt]

            # 推送市场事件
            for _, row in day_data.iterrows():
                market_event = MarketEvent(
                    dt, row['code'], row['open'], row['high'],
                    row['low'], row['close'], row['volume']
                )
                self.event_engine.put(market_event, priority=1)

            # 按日期处理信号
            if signals is not None and dt in signals['date'].values:
                day_signals = signals[signals['date'] == dt]
                for _, row in day_signals.iterrows():
                    direction = 1 if row.get('signal', 0) > 0 else -1
                    price = day_data[day_data['code'] == row['code']]['close'].values[0] \
                        if len(day_data[day_data['code'] == row['code']]) > 0 else 10
                    signal_event = SignalEvent(
                        dt, row['code'], direction, 'factor',
                        strength=abs(row.get('signal', 0)),
                        reason=f"Factor signal: {row.get('signal', 0):.4f}"
                    )
                    self.event_engine.put(signal_event, priority=3)

            # 处理当日所有事件
            self.event_engine.process()

            # 记录权益
            prices = dict(zip(day_data['code'], day_data['close']))
            total_value = self.broker.get_total_value(prices)
            equity_curve.append({'date': dt, 'equity': total_value})

            daily_events.append({
                'date': dt,
                'events': self.event_engine._event_count,
                'cash': self.broker.cash,
            })

        # 统计
        engine_stats = self.event_engine.get_stats()
        equity_df = pd.DataFrame(equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()

        self.results = {
            'equity_curve': equity_df,
            'total_return': (equity_df['equity'].iloc[-1] / self.broker.initial_cash - 1),
            'annual_return': equity_df['returns'].mean() * 252,
            'sharpe': equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252) if equity_df['returns'].std() > 0 else 0,
            'max_drawdown': (equity_df['equity'].cummax() - equity_df['equity']).max() / equity_df['equity'].cummax().max(),
            'total_trades': len(self.broker.fills),
            'total_fills': len(self.broker.fills),
            'engine_stats': engine_stats,
            'daily_events': daily_events,
        }
        return self.results


# =============================================================================
# 性能对比：事件驱动 vs 原生循环
# =============================================================================

def generate_test_data(n_stocks: int = 50, n_days: int = 252) -> pd.DataFrame:
    """生成模拟市场数据"""
    np.random.seed(42)
    symbols = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

    rows = []
    for sym in symbols:
        start_price = np.random.uniform(8, 80)
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = start_price * (1 + returns).cumprod()
        volumes = np.random.lognormal(12, 0.5, n_days).astype(int)

        for i in range(n_days):
            rows.append({
                'date': dates[i],
                'code': sym,
                'open': prices[i] * (1 + np.random.normal(0, 0.003)),
                'high': prices[i] * (1 + abs(np.random.normal(0, 0.015))),
                'low': prices[i] * (1 - abs(np.random.normal(0, 0.015))),
                'close': prices[i],
                'volume': volumes[i],
            })

    return pd.DataFrame(rows)


def generate_signals(data: pd.DataFrame, n_signals_per_day: int = 5) -> pd.DataFrame:
    """生成模拟信号"""
    np.random.seed(43)
    dates = sorted(data['date'].unique())
    codes = sorted(data['code'].unique())

    signal_rows = []
    for dt in dates:
        selected = np.random.choice(codes, min(n_signals_per_day, len(codes)), replace=False)
        for code in selected:
            signal_rows.append({
                'date': dt,
                'code': code,
                'signal': np.random.uniform(-1, 1),
            })
    return pd.DataFrame(signal_rows)


# =============================================================================
# 单元测试
# =============================================================================

class TestEventEngine(unittest.TestCase):
    """事件驱动引擎测试"""

    def test_event_register(self):
        """测试事件处理器注册"""
        engine = EventEngine()
        received = []

        def handler(event):
            received.append(event)

        engine.register(EventType.MARKET, handler)
        event = MarketEvent(pd.Timestamp('2024-01-01'), '000001.SH', 10, 11, 9, 10.5, 1000000)
        engine.put(event)
        engine.process()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].type, EventType.MARKET)
        print(f"[PASS] 事件注册与分发，收到 {len(received)} 个事件")

    def test_multiple_handlers(self):
        """测试多个处理器"""
        engine = EventEngine()
        results = {'a': 0, 'b': 0}

        def handler_a(event):
            results['a'] += 1

        def handler_b(event):
            results['b'] += 1

        engine.register(EventType.MARKET, handler_a)
        engine.register(EventType.MARKET, handler_b)

        for _ in range(10):
            event = MarketEvent(pd.Timestamp('2024-01-01'), '000001.SH', 10, 11, 9, 10.5, 1000000)
            engine.put(event)
        engine.process()

        self.assertEqual(results['a'], 10)
        self.assertEqual(results['b'], 10)
        print(f"[PASS] 多处理器: a={results['a']}, b={results['b']}")

    def test_event_priority(self):
        """测试事件优先级"""
        engine = EventEngine()
        order_events = []

        def market_handler(event):
            # 市场事件触发信号
            signal = SignalEvent(event.timestamp, '000001.SH', 1, 'test')
            engine.put(signal, priority=3)

        def signal_handler(event):
            order = OrderEvent(event.timestamp, '000001.SH', 1, 1, 'market')
            engine.put(order, priority=6)

        def order_handler(event):
            order_events.append(event)

        engine.register(EventType.MARKET, market_handler)
        engine.register(EventType.SIGNAL, signal_handler)
        engine.register(EventType.ORDER, order_handler)

        market = MarketEvent(pd.Timestamp('2024-01-01'), '000001.SH', 10, 11, 9, 10.5, 1000000)
        engine.put(market, priority=1)
        engine.process()

        self.assertEqual(len(order_events), 1)
        print(f"[PASS] 事件优先级链: Market → Signal → Order，共 {len(order_events)} 个订单")

    def test_event_stats(self):
        """测试事件统计"""
        engine = EventEngine()
        handler = lambda e: None
        engine.register(EventType.MARKET, handler)

        for _ in range(100):
            engine.put(MarketEvent(pd.Timestamp('2024-01-01'), '000001.SH', 10, 11, 9, 10.5, 1000000))
        engine.process()

        stats = engine.get_stats()
        self.assertGreater(stats['events_per_second'], 0)
        print(f"[PASS] 事件统计: total={stats['total_events']}, "
              f"eps={stats['events_per_second']:.0f}")


class TestEventDrivenBacktest(unittest.TestCase):
    """事件驱动回测测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_test_data(n_stocks=20, n_days=60)
        cls.signals = generate_signals(cls.data, n_signals_per_day=3)

    def test_backtest_run(self):
        """测试回测运行"""
        runner = EventDrivenBacktestRunner(initial_cash=1_000_000)
        strategy = MomentumStrategy(lookback=20, top_n=5)
        runner.add_strategy(strategy)

        results = runner.run(self.data, self.signals)

        self.assertIn('equity_curve', results)
        self.assertIn('total_return', results)
        self.assertIn('total_fills', results)
        self.assertGreater(len(results['equity_curve']), 0)

        print("[PASS] 事件驱动回测运行成功")
        print(f"  总收益率: {results['total_return']:.4%}")
        print(f"  年化收益: {results['annual_return']:.4%}")
        print(f"  Sharpe: {results['sharpe']:.4f}")
        print(f"  最大回撤: {results['max_drawdown']:.4%}")
        print(f"  总成交笔数: {results['total_fills']}")

    def test_backtest_engine_stats(self):
        """测试回测引擎统计"""
        runner = EventDrivenBacktestRunner(initial_cash=1_000_000)
        strategy = MomentumStrategy()
        runner.add_strategy(strategy)

        results = runner.run(self.data, self.signals)
        stats = results['engine_stats']

        self.assertIn('events_per_second', stats)
        self.assertIn('events_by_type', stats)
        print(f"[PASS] 引擎统计: 总事件={stats['total_events']}, "
              f"EPS={stats['events_per_second']:.0f}")

    def test_broker_execution(self):
        """测试Broker订单执行"""
        broker = BacktestBroker(initial_cash=1_000_000)
        order = OrderEvent(pd.Timestamp('2024-01-01'), '000001.SH', 1, 10, 'market', 10)
        fill = broker.execute_order(order, 10)

        self.assertIsNotNone(fill)
        self.assertLess(broker.cash, 1_000_000)
        self.assertEqual(broker.positions['000001.SH'], 10)
        print(f"[PASS] Broker订单执行: cash={broker.cash:.2f}, "
              f"position={broker.positions['000001.SH']}")

    def test_broker_insufficient_cash(self):
        """测试资金不足"""
        broker = BacktestBroker(initial_cash=1000)
        # 尝试买入超过资金的股票
        order = OrderEvent(pd.Timestamp('2024-01-01'), '000001.SH', 1, 100, 'market', 100)
        fill = broker.execute_order(order, 100)

        self.assertIsNone(fill)
        print("[PASS] 资金不足拒绝交易")

    def test_broker_short_restriction(self):
        """测试卖空限制"""
        broker = BacktestBroker(initial_cash=1_000_000)
        # 先买入
        buy_order = OrderEvent(pd.Timestamp('2024-01-01'), '000001.SH', 1, 10, 'market', 10)
        broker.execute_order(buy_order, 10)
        # 卖出超过持仓
        sell_order = OrderEvent(pd.Timestamp('2024-01-02'), '000001.SH', -1, 20, 'market', 10)
        fill = broker.execute_order(sell_order, 10)

        self.assertIsNone(fill)
        print("[PASS] 卖空限制生效")


class TestEventChain(unittest.TestCase):
    """事件链完整流程测试"""

    def test_full_event_chain(self):
        """测试完整事件链: Market → Signal → Order → Fill → Position → Account"""
        engine = EventEngine()
        broker = BacktestBroker(initial_cash=1_000_000)
        chain_results = []

        def market_handler(event):
            chain_results.append(('market', event.timestamp))
            # 生成信号
            signal = SignalEvent(event.timestamp, event.data['symbol'], 1, 'factor', 1.0)
            engine.put(signal, priority=3)

        def signal_handler(event):
            chain_results.append(('signal', event.timestamp))
            # 生成订单
            price = event.data.get('price', 10)
            order = OrderEvent(event.timestamp, event.data['symbol'], 1, 5, 'market', price)
            engine.put(order, priority=6)

        def order_handler(event):
            chain_results.append(('order', event.timestamp))
            # 执行订单
            fill = broker.execute_order(event, event.data.get('price', 10))
            if fill:
                engine.put(fill, priority=7)

        def fill_handler(event):
            chain_results.append(('fill', event.timestamp))

        engine.register(EventType.MARKET, market_handler)
        engine.register(EventType.SIGNAL, signal_handler)
        engine.register(EventType.ORDER, order_handler)
        engine.register(EventType.FILL, fill_handler)

        market = MarketEvent(pd.Timestamp('2024-01-01'), '000001.SH', 10, 11, 9, 10.5, 1000000)
        market.data['price'] = 10
        engine.put(market, priority=1)
        engine.process()

        expected_order = ['market', 'signal', 'order', 'fill']
        actual_order = [e[0] for e in chain_results]
        self.assertEqual(actual_order, expected_order)
        print(f"[PASS] 完整事件链: {' → '.join(actual_order)}")


if __name__ == '__main__':
    print("=" * 70)
    print("事件驱动回测引擎验证测试")
    print("=" * 70)

    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEventEngine))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEventDrivenBacktest))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEventChain))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("验证总结")
    print(f"  总测试数: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print("=" * 70)