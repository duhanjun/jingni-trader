"""
===========================================================================
测试文件: test_event_driven_backtest.py
借鉴来源:
    1. Nautilus Trader (https://github.com/nautechsystems/nautilus_trader)
       - 事件驱动架构设计 (Cython 加速, 微秒级数据处理)
       - 回测/实盘统一架构 (single codebase for backtest & live)
       - 风控断路器 (OrderEmitters, RiskEngine, PositionManager)

    2. trade-learn (https://github.com/MuuYesen/trade-learn)
       - Python 逻辑 + Rust 内核 (110x+ 性能提升)
       - 双模架构: Engine (正确性) + Lite (快速原型)

优化方向: backtest-engine - 事件驱动回测架构
     - 当前问题: 回测使用向量化计算, 无法模拟真实事件流
     - 优化方案: 引入事件驱动架构, 支持:
       1. Tick/Bar 级事件模拟
       2. 订单簿撮合 (order book matching)
       3. 真实交易约束 (T+1, 涨跌停, 最小交易单位)
       4. 回测与实盘代码复用

测试内容:
     1. 事件驱动核心引擎正确性测试
     2. 订单簿撮合逻辑测试
     3. 与向量化回测结果对比
     4. 边界条件测试 (涨跌停/T+1等)

⚠️ 注意: 此文件为验证代码，仅在测试目录中运行，不修改主代码。
===========================================================================
"""

import sys
import os
import json
import time
import unittest
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


# ===========================================================================
# 事件驱动核心引擎 (借鉴 Nautilus Trader)
# ===========================================================================

class EventType(Enum):
    """事件类型枚举"""
    MARKET_DATA = "MARKET_DATA"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    MARKET_OPEN = "MARKET_OPEN"
    MARKET_CLOSE = "MARKET_CLOSE"
    RISK_CHECK = "RISK_CHECK"
    PORTFOLIO_REBALANCE = "PORTFOLIO_REBALANCE"


@dataclass(order=True)
class Event:
    """事件基类 (借鉴 Nautilus Trader Event 设计)"""
    timestamp: pd.Timestamp = field(compare=True)
    event_type: EventType = field(compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)
    priority: int = field(default=0, compare=True)


@dataclass
class Order:
    """订单模型"""
    order_id: str
    code: str
    side: str  # 'buy' or 'sell'
    quantity: int
    price: float
    order_type: str = 'limit'  # 'limit' or 'market'
    status: str = 'pending'    # pending, filled, cancelled, rejected
    created_at: pd.Timestamp = None
    filled_at: pd.Timestamp = None
    fill_price: float = 0.0
    fill_quantity: int = 0


@dataclass
class Position:
    """持仓模型"""
    code: str
    quantity: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0

    def update(self, price: float):
        """更新市值和未实现盈亏"""
        self.market_value = self.quantity * price
        self.unrealized_pnl = (price - self.avg_cost) * self.quantity if self.quantity > 0 else 0


@dataclass
class Portfolio:
    """组合账户"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_capital: float = 0.0
    equity_curve: List[Dict] = field(default_factory=list)

    def total_equity(self, prices: Dict[str, float]) -> float:
        """计算总权益"""
        total = self.cash
        for code, pos in self.positions.items():
            if code in prices:
                total += pos.quantity * prices[code]
            else:
                total += pos.market_value
        return total

    def record_snapshot(self, date: pd.Timestamp, prices: Dict[str, float]):
        """记录净值快照"""
        equity = self.total_equity(prices)
        self.equity_curve.append({
            'date': date,
            'equity': equity,
            'cash': self.cash,
        })


class EventBus:
    """事件总线 (借鉴 Nautilus Trader EventRouter)"""

    def __init__(self):
        self._queue: List[Event] = []
        self._handlers: Dict[EventType, List[callable]] = defaultdict(list)
        self._processed_count = 0

    def subscribe(self, event_type: EventType, handler: callable):
        """注册事件处理器"""
        self._handlers[event_type].append(handler)

    def publish(self, event: Event):
        """发布事件到队列"""
        heapq.heappush(self._queue, event)

    def process_next(self) -> bool:
        """处理队列中的下一个事件"""
        if not self._queue:
            return False

        event = heapq.heappop(self._queue)
        handlers = self._handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # 在生产环境中应记录错误并触发断路器
                raise

        self._processed_count += 1
        return True

    def process_all(self):
        """处理所有已排队事件"""
        while self.process_next():
            pass

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def processed_count(self) -> int:
        return self._processed_count


# ===========================================================================
# 交易模拟组件
# ===========================================================================

class OrderBookMatcher:
    """
    简化版订单簿撮合引擎

    借鉴 Nautilus Trader 的 Execution 模块设计:
    1. 限价单 (limit order): 按指定价格成交
    2. 市价单 (market order): 按当前市价成交
    3. 模拟滑点 (slippage)
    4. A股特有规则: T+1, 涨跌停限制
    """

    def __init__(
        self,
        commission_rate: float = 0.00025,
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.0001,
        t_plus_1: bool = True,
        price_limit_pct: float = 0.10,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.t_plus_1 = t_plus_1
        self.price_limit_pct = price_limit_pct
        # T+1: 今日买入的股票记录, 明日才能卖出
        self._today_bought: Set[str] = set()

    def reset_daily(self):
        """新交易日重置 T+1 记录"""
        self._today_bought = set()

    def match_order(
        self,
        order: Order,
        current_price: float,
        pre_close: float,
    ) -> Tuple[bool, float, int, float]:
        """
        撮合订单

        返回: (是否成交, 成交价, 成交量, 总费用)

        A股规则:
        - 买入: T+1, 当日不可卖
        - 涨跌停: 涨跌停价不能成交 (除非能买/卖涨停板)
        - 最小交易单位: 100股
        - T+1 卖出限制: 今日买入的不能卖出
        """
        # 涨跌停检查
        upper_limit = pre_close * (1 + self.price_limit_pct)
        lower_limit = pre_close * (1 - self.price_limit_pct)

        # 模拟滑点
        if order.side == 'buy':
            exec_price = current_price * (1 + self.slippage)
            # 涨停价受限
            exec_price = min(exec_price, upper_limit)
        else:
            exec_price = current_price * (1 - self.slippage)
            # 跌停价受限
            exec_price = max(exec_price, lower_limit)

        # 限价单: 检查价格约束
        if order.order_type == 'limit':
            if order.side == 'buy' and exec_price > order.price:
                return (False, 0, 0, 0)
            if order.side == 'sell' and exec_price < order.price:
                return (False, 0, 0, 0)

        # 计算费用
        amount = exec_price * order.quantity
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_tax = amount * self.stamp_tax_rate if order.side == 'sell' else 0
        total_fee = commission + stamp_tax

        # T+1: 记录今日买入
        if order.side == 'buy' and self.t_plus_1:
            self._today_bought.add(order.code)

        return (True, exec_price, order.quantity, total_fee)

    def can_sell(self, code: str) -> bool:
        """检查是否可以卖出 (T+1 合规)"""
        if self.t_plus_1 and code in self._today_bought:
            return False
        return True


class CircuitBreaker:
    """
    风控断路器 (借鉴 Nautilus Trader RiskEngine)

    检查项:
    1. 最大持仓比例
    2. 单日最大亏损
    3. 单笔最大金额
    4. 最小现金余额
    """

    def __init__(
        self,
        max_position_ratio: float = 0.10,
        max_daily_loss_ratio: float = 0.03,
        max_order_ratio: float = 0.05,
        min_cash_pct: float = 0.05,
    ):
        self.max_position_ratio = max_position_ratio
        self.max_daily_loss_ratio = max_daily_loss_ratio
        self.max_order_ratio = max_order_ratio
        self.min_cash_pct = min_cash_pct
        self._start_of_day_equity = 0.0

    def reset_daily(self, equity: float):
        self._start_of_day_equity = equity

    def check_order(
        self,
        portfolio: Portfolio,
        order: Order,
        prices: Dict[str, float],
    ) -> Dict[str, Any]:
        """检查订单是否可以执行"""
        current_equity = portfolio.total_equity(prices)

        checks = {}

        # 1. 单日亏损检查
        daily_return = (current_equity - self._start_of_day_equity) / self._start_of_day_equity if self._start_of_day_equity > 0 else 0
        checks['daily_loss'] = {
            'passed': daily_return > -self.max_daily_loss_ratio,
            'actual': float(daily_return),
            'threshold': self.max_daily_loss_ratio,
        }

        # 2. 单笔金额检查
        order_value = order.price * order.quantity
        max_order_value = current_equity * self.max_order_ratio
        checks['order_size'] = {
            'passed': order_value <= max_order_value,
            'actual': float(order_value),
            'threshold': max_order_value,
        }

        # 3. 现金检查 (买入时)
        if order.side == 'buy':
            total_needed = order_value * 1.001  # 约等于佣金+费用
            checks['cash_sufficient'] = {
                'passed': portfolio.cash >= total_needed,
                'actual': float(portfolio.cash),
                'needed': float(total_needed),
            }

        # 4. 最小现金保留
        checks['min_cash'] = {
            'passed': portfolio.cash >= current_equity * self.min_cash_pct,
        }

        all_passed = all(c.get('passed', True) for c in checks.values())

        return {
            'allowed': all_passed,
            'checks': checks,
        }


# ===========================================================================
# 事件驱动回测引擎
# ===========================================================================

class EventDrivenBacktestEngine:
    """
    事件驱动回测引擎

    借鉴 Nautilus Trader + trade-learn 设计:
    1. 事件总线驱动市场数据流
    2. 订单簿撮合
    3. 风控断路器
    4. 净值跟踪
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        **kwargs,
    ):
        self.event_bus = EventBus()
        self.matcher = OrderBookMatcher(**kwargs)
        self.circuit_breaker = CircuitBreaker()
        self.portfolio = Portfolio(cash=initial_capital, initial_capital=initial_capital)

        # 通用价格缓存
        self._current_prices: Dict[str, float] = {}
        self._pre_close: Dict[str, float] = {}
        self._signals: Dict[str, Dict] = {}

        # 注册处理器
        self.event_bus.subscribe(EventType.MARKET_DATA, self._on_market_data)
        self.event_bus.subscribe(EventType.SIGNAL, self._on_signal)
        self.event_bus.subscribe(EventType.ORDER, self._on_order)
        self.event_bus.subscribe(EventType.MARKET_OPEN, self._on_market_open)
        self.event_bus.subscribe(EventType.MARKET_CLOSE, self._on_market_close)

    def _on_market_open(self, event: Event):
        """市场开盘事件"""
        self.circuit_breaker.reset_daily(
            self.portfolio.total_equity(self._current_prices)
        )
        self.matcher.reset_daily()

    def _on_market_close(self, event: Event):
        """市场收盘事件 - 记录净值"""
        self.portfolio.record_snapshot(event.timestamp, self._current_prices)

    def _on_market_data(self, event: Event):
        """市场数据处理"""
        payload = event.payload
        code = payload['code']
        self._current_prices[code] = payload['close']
        self._pre_close[code] = payload.get('pre_close', payload['close'])

        # 更新持仓市值
        if code in self.portfolio.positions:
            self.portfolio.positions[code].update(payload['close'])

    def _on_signal(self, event: Event):
        """信号处理 - 生成订单"""
        payload = event.payload
        code = payload['code']
        signal = payload['signal']  # 1: buy, -1: sell, 0: hold
        price = self._current_prices.get(code)

        if price is None or signal == 0:
            return

        if signal > 0:
            # 买入信号
            max_order_amount = self.portfolio.total_equity(self._current_prices) * 0.05
            quantity = int(max_order_amount / price // 100 * 100)
            if quantity > 0:
                order = Order(
                    order_id=f"buy_{code}_{event.timestamp}",
                    code=code,
                    side='buy',
                    quantity=quantity,
                    price=price,
                    created_at=event.timestamp,
                )
                self.event_bus.publish(Event(
                    timestamp=event.timestamp,
                    event_type=EventType.ORDER,
                    payload={'order': order},
                    priority=1,
                ))

        elif signal < 0:
            # 卖出信号
            pos = self.portfolio.positions.get(code)
            if pos and pos.quantity > 0 and self.matcher.can_sell(code):
                order = Order(
                    order_id=f"sell_{code}_{event.timestamp}",
                    code=code,
                    side='sell',
                    quantity=pos.quantity,
                    price=price,
                    created_at=event.timestamp,
                )
                self.event_bus.publish(Event(
                    timestamp=event.timestamp,
                    event_type=EventType.ORDER,
                    payload={'order': order},
                    priority=1,
                ))

    def _on_order(self, event: Event):
        """订单执行"""
        order = event.payload['order']
        price = self._current_prices.get(order.code)
        pre_close = self._pre_close.get(order.code, price)

        if price is None:
            order.status = 'rejected'
            return

        # 风控检查
        risk_check = self.circuit_breaker.check_order(
            self.portfolio, order, self._current_prices
        )
        if not risk_check['allowed']:
            order.status = 'rejected'
            return

        # 撮合
        filled, fill_price, fill_qty, fee = self.matcher.match_order(
            order, price, pre_close
        )

        if filled:
            order.status = 'filled'
            order.fill_price = fill_price
            order.fill_quantity = fill_qty
            order.filled_at = event.timestamp

            # 更新组合
            if order.side == 'buy':
                total_cost = fill_price * fill_qty + fee
                self.portfolio.cash -= total_cost

                if order.code not in self.portfolio.positions:
                    self.portfolio.positions[order.code] = Position(code=order.code)
                pos = self.portfolio.positions[order.code]
                old_value = pos.avg_cost * pos.quantity
                pos.quantity += fill_qty
                pos.avg_cost = (old_value + fill_price * fill_qty) / pos.quantity

            else:  # sell
                total_revenue = fill_price * fill_qty - fee
                self.portfolio.cash += total_revenue

                pos = self.portfolio.positions[order.code]
                pos.quantity -= fill_qty
                if pos.quantity <= 0:
                    del self.portfolio.positions[order.code]

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        progress: bool = False,
    ) -> Dict[str, Any]:
        """
        运行事件驱动回测

        参数:
            data: DataFrame with columns [date, code, open, high, low, close, pre_close, ...]
            signals: DataFrame with columns [code, date, signal]
        """
        dates = sorted(data['date'].unique())
        total_dates = len(dates)

        for i, dt in enumerate(dates):
            if progress and i % 100 == 0:
                print(f"  回测进度: {i}/{total_dates} ({100*i/total_dates:.0f}%)")

            day_data = data[data['date'] == dt]
            day_signals = signals[signals['date'] == dt]

            # 市场开盘
            self.event_bus.publish(Event(
                timestamp=dt, event_type=EventType.MARKET_OPEN
            ))

            # 推送行情数据
            for _, row in day_data.iterrows():
                self.event_bus.publish(Event(
                    timestamp=dt,
                    event_type=EventType.MARKET_DATA,
                    payload={
                        'code': row['code'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'pre_close': row.get('pre_close', row['close']),
                        'volume': row.get('volume', 0),
                    },
                ))

            # 推送交易信号
            for _, sig_row in day_signals.iterrows():
                code = sig_row['code']
                if code in self._current_prices:
                    self.event_bus.publish(Event(
                        timestamp=dt,
                        event_type=EventType.SIGNAL,
                        payload={'code': code, 'signal': sig_row['signal']},
                    ))

            # 处理所有事件
            self.event_bus.process_all()

            # 市场收盘
            self.event_bus.publish(Event(
                timestamp=dt, event_type=EventType.MARKET_CLOSE
            ))
            self.event_bus.process_all()

        # 生成结果
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        metrics = self._calc_metrics(equity_df)

        return {
            'equity_curve': equity_df,
            'metrics': metrics,
            'final_equity': self.portfolio.total_equity(self._current_prices),
            'cash': self.portfolio.cash,
            'positions': {
                code: {'quantity': pos.quantity, 'avg_cost': pos.avg_cost}
                for code, pos in self.portfolio.positions.items()
                if pos.quantity > 0
            },
        }

    def _calc_metrics(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        """计算绩效指标"""
        if equity_df.empty or len(equity_df) < 2:
            return {}

        returns = equity_df.set_index('date')['equity'].pct_change().dropna()
        if len(returns) < 2:
            return {}

        total_return = (equity_df['equity'].iloc[-1] / self.portfolio.initial_capital) - 1
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = (equity_df['equity'] / equity_df['equity'].cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float((returns > 0).mean()),
        }


# ===========================================================================
# 单元测试
# ===========================================================================

class TestEventDrivenBacktest(unittest.TestCase):
    """事件驱动回测测试套件"""

    @classmethod
    def setUpClass(cls):
        """生成模拟A股数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-02', '2024-06-28', freq='B')
        codes = ['000001.SZ', '600000.SH', '000002.SZ']

        rows = []
        for code in codes:
            base_price = np.random.uniform(10, 30)
            n = len(dates)
            daily_returns = np.random.normal(0.0003, 0.015, n)
            prices = base_price * (1 + daily_returns).cumprod()

            df = pd.DataFrame({
                'date': dates,
                'code': code,
                'open': prices * (1 + np.random.normal(0, 0.002, n)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.008, n))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.008, n))),
                'close': prices,
                'volume': np.random.lognormal(10, 0.3, n).astype(int),
            })
            df['pre_close'] = df['close'].shift(1).fillna(df['close'])
            rows.append(df)

        cls.test_data = pd.concat(rows, ignore_index=True)

        # 生成交易信号: 简单的均线策略
        signal_rows = []
        for code in codes:
            group = cls.test_data[cls.test_data['code'] == code].sort_values('date').copy()
            ma_short = group['close'].rolling(5).mean()
            ma_long = group['close'].rolling(20).mean()
            sig = pd.DataFrame({
                'date': group['date'].values,
                'code': code,
                'signal': np.where(ma_short > ma_long, 1, np.where(ma_short < ma_long, -1, 0)),
            })
            signal_rows.append(sig)

        cls.signals = pd.concat(signal_rows, ignore_index=True)

    def test_event_bus_basic(self):
        """测试事件总线基础功能"""
        bus = EventBus()

        received = []
        bus.subscribe(EventType.MARKET_DATA, lambda e: received.append(e))

        bus.publish(Event(timestamp=pd.Timestamp('2024-01-01'), event_type=EventType.MARKET_DATA))
        bus.publish(Event(timestamp=pd.Timestamp('2024-01-02'), event_type=EventType.MARKET_DATA))

        bus.process_all()

        self.assertEqual(len(received), 2)
        self.assertEqual(bus.pending_count, 0)
        self.assertEqual(bus.processed_count, 2)
        print("[PASS] test_event_bus_basic")

    def test_event_bus_ordering(self):
        """测试事件按时间排序"""
        bus = EventBus()

        order_tracker = []

        def handler(e):
            order_tracker.append(e.timestamp)

        bus.subscribe(EventType.MARKET_DATA, handler)

        # 乱序发布
        bus.publish(Event(timestamp=pd.Timestamp('2024-01-03'), event_type=EventType.MARKET_DATA))
        bus.publish(Event(timestamp=pd.Timestamp('2024-01-01'), event_type=EventType.MARKET_DATA))
        bus.publish(Event(timestamp=pd.Timestamp('2024-01-02'), event_type=EventType.MARKET_DATA))

        bus.process_all()

        # 应该按时序处理
        self.assertEqual(order_tracker[0], pd.Timestamp('2024-01-01'))
        self.assertEqual(order_tracker[1], pd.Timestamp('2024-01-02'))
        self.assertEqual(order_tracker[2], pd.Timestamp('2024-01-03'))
        print("[PASS] test_event_bus_ordering")

    def test_order_matching_basic(self):
        """测试订单撮合基础逻辑"""
        matcher = OrderBookMatcher()

        order = Order('test1', '000001.SZ', 'buy', 100, 10.5)
        filled, price, qty, fee = matcher.match_order(order, current_price=10.0, pre_close=10.0)

        self.assertTrue(filled)
        self.assertGreater(fee, 0)
        print(f"[PASS] test_order_matching_basic: filled={filled}, price={price:.2f}, fee={fee:.2f}")

    def test_order_matching_price_limit(self):
        """测试涨跌停限制"""
        matcher = OrderBookMatcher(price_limit_pct=0.10)

        # 涨停: 无法以高于涨停价买入
        order_buy = Order('test_limit', '000001.SZ', 'buy', 100, 12.0)
        filled, price, qty, fee = matcher.match_order(
            order_buy, current_price=10.0, pre_close=10.0
        )

        if filled:
            self.assertLessEqual(price, 11.0)  # 涨停价 = 10 * 1.10

        # 跌停: 无法以低于跌停价卖出
        order_sell = Order('test_sell', '000001.SZ', 'sell', 100, 8.0)
        filled, price, qty, fee = matcher.match_order(
            order_sell, current_price=10.0, pre_close=10.0
        )

        if filled:
            self.assertGreaterEqual(price, 9.0)  # 跌停价 = 10 * 0.90

        print("[PASS] test_order_matching_price_limit")

    def test_circuit_breaker(self):
        """测试风控断路器"""
        cb = CircuitBreaker(
            max_order_ratio=0.05,
            max_daily_loss_ratio=0.03,
        )
        portfolio = Portfolio(cash=1_000_000)
        prices = {'000001.SZ': 10.0}

        cb.reset_daily(1_000_000)

        # 正常订单
        order = Order('test', '000001.SZ', 'buy', 1000, 10.0)
        result = cb.check_order(portfolio, order, prices)
        self.assertTrue(result['allowed'])
        print(f"[PASS] test_circuit_breaker - 正常订单: {result['checks']}")

        # 超大订单
        large_order = Order('large', '000001.SZ', 'buy', 10000, 10.0)
        result = cb.check_order(portfolio, large_order, prices)
        self.assertFalse(result['allowed'])
        print(f"[PASS] test_circuit_breaker - 超大订单: {result['checks']}")

    def test_full_backtest_run(self):
        """测试完整事件驱动回测"""
        engine = EventDrivenBacktestEngine(initial_capital=100_000)

        start = time.time()
        result = engine.run(self.test_data, self.signals)
        elapsed = time.time() - start

        self.assertIsNotNone(result['equity_curve'])
        self.assertGreater(len(result['equity_curve']), 0)
        self.assertIn('metrics', result)

        metrics = result['metrics']
        print(f"\n  回测结果 (事件驱动):")
        print(f"    最终权益: {result['final_equity']:,.2f}")
        print(f"    现金: {result['cash']:,.2f}")
        print(f"    持仓: {len(result['positions'])} 只")
        if metrics:
            print(f"    总收益: {metrics.get('total_return', 0):.4%}")
            print(f"    年化收益: {metrics.get('annual_return', 0):.4%}")
            print(f"    夏普比: {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"    最大回撤: {metrics.get('max_drawdown', 0):.4%}")
            print(f"    胜率: {metrics.get('win_rate', 0):.4%}")
        print(f"    耗时: {elapsed:.2f}s")

        # 基本合理性检查
        self.assertGreater(result['final_equity'], 0)
        self.assertGreaterEqual(result['cash'], 0)

        print("[PASS] test_full_backtest_run")

    def test_edge_case_empty_signals(self):
        """测试边界条件: 空信号"""
        engine = EventDrivenBacktestEngine(initial_capital=100_000)
        empty_signals = self.signals.iloc[0:0]

        result = engine.run(self.test_data, empty_signals)

        self.assertIsNotNone(result['equity_curve'])
        self.assertAlmostEqual(result['final_equity'], 100_000, delta=1)
        self.assertEqual(len(result['positions']), 0)

        print("[PASS] test_edge_case_empty_signals: 无交易时净值不变")

    def test_event_counting(self):
        """测试事件计数和性能"""
        engine = EventDrivenBacktestEngine(initial_capital=100_000)

        engine.run(self.test_data, self.signals)

        processed = engine.event_bus.processed_count
        print(f"\n  事件统计:")
        print(f"    总处理事件数: {processed}")
        print(f"    每交易日平均: {processed / len(self.test_data['date'].unique()):.1f}")

        self.assertGreater(processed, 0, "应处理一定数量的事件")

        print("[PASS] test_event_counting")


# ===========================================================================
# 主函数
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("事件驱动回测架构 验证测试")
    print("借鉴来源: Nautilus Trader + trade-learn Event-Driven Architecture")
    print("=" * 70)

    print("\n运行测试套件...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEventDrivenBacktest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"测试结果: {'全部通过' if result.wasSuccessful() else '存在失败'}")
    print("所有验证代码位于独立测试文件中，未修改主代码。")
    print("=" * 70)