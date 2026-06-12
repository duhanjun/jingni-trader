"""
验证代码：事件驱动回测架构
============================================
借鉴来源: vn.py / VeighNa (https://github.com/vnpy/vnpy)
优化方向: backtest-engine —— 回测引擎的准确性与可维护性
核心思路: vn.py 使用 Event-Driven 架构，将行情/订单/成交/账户状态统一为事件流，
         通过 pub/sub 解耦各组件。策略只关心 on_bar/on_tick 回调，引擎负责
         事件循环、撮合、风控等。事件可序列化记录，支持回放和复盘。
日期: 2026-06-12

约束: 仅验证可行性，不可直接修改主代码，不可执行 git commit/merge。
"""

import unittest
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import uuid


# ═══════════════════════════════════════════
# 1. 事件定义（借鉴 vn.py 事件系统）
# ═══════════════════════════════════════════

class EventType(Enum):
    """事件类型枚举"""
    TICK = "tick"                   # Tick 行情
    BAR = "bar"                     # Bar 行情
    ORDER = "order"                 # 订单状态更新
    TRADE = "trade"                 # 成交回报
    POSITION = "position"           # 持仓更新
    ACCOUNT = "account"             # 账户更新
    LOG = "log"                     # 日志
    TIMER = "timer"                 # 定时器
    RISK_WARNING = "risk_warning"   # 风控告警


@dataclass
class Event:
    """事件对象"""
    type: EventType
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BarData:
    """Bar 数据"""
    code: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0


@dataclass
class OrderData:
    """订单数据"""
    order_id: str = ""
    code: str = ""
    side: str = ""           # buy / sell
    price: float = 0.0
    volume: int = 0
    filled_volume: int = 0
    status: str = "pending"  # pending / filled / cancelled / rejected
    order_time: datetime = field(default_factory=datetime.now)


@dataclass
class TradeData:
    """成交数据"""
    trade_id: str = ""
    order_id: str = ""
    code: str = ""
    side: str = ""
    price: float = 0.0
    volume: int = 0
    commission: float = 0.0
    stamp_tax: float = 0.0
    trade_time: datetime = field(default_factory=datetime.now)


@dataclass
class AccountData:
    """账户数据"""
    nav: float = 1_000_000.0
    available_cash: float = 1_000_000.0
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    daily_pnl: float = 0.0


# ═══════════════════════════════════════════
# 2. 事件引擎（借鉴 vn.py EventEngine）
# ═══════════════════════════════════════════

class EventEngine:
    """
    事件驱动引擎

    核心功能：
    - 事件注册/注销
    - 事件分发（同步模式）
    - 事件记录（支持回放）
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_log: List[Event] = []
        self._record_mode = False

    def register(self, event_type: EventType, handler: Callable) -> None:
        """注册事件处理器"""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unregister(self, event_type: EventType, handler: Callable) -> None:
        """注销事件处理器"""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def put(self, event: Event) -> None:
        """将事件放入队列并立即处理"""
        if self._record_mode:
            self._event_log.append(event)
        self._process(event)

    def _process(self, event: Event) -> None:
        """处理单个事件"""
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"事件处理异常 [{event.type}]: {e}")

    def start_recording(self) -> None:
        """开始记录事件"""
        self._record_mode = True
        self._event_log.clear()

    def stop_recording(self) -> None:
        """停止记录"""
        self._record_mode = False

    def replay(self, target_engine: 'EventEngine' = None) -> None:
        """
        回放已记录的事件

        可在另一个引擎中重放（如回测引擎），用于：
        - 策略复盘
        - 事故重现
        - 回归测试
        """
        eng = target_engine or self
        for event in self._event_log:
            eng.put(event)

    def get_event_log(self) -> List[Event]:
        """获取事件日志"""
        return list(self._event_log)


# ═══════════════════════════════════════════
# 3. 策略基类（借鉴 vn.py CtaTemplate）
# ═══════════════════════════════════════════

class StrategyBase:
    """策略基类"""

    def __init__(self, name: str = "strategy"):
        self.name = name
        self.bar_buffer: Dict[str, List[BarData]] = defaultdict(list)

    def on_bar(self, bar: BarData) -> Optional[Dict[str, Any]]:
        """
        Bar 行情回调

        返回:
            交易信号 dict（可选），含 code, side, volume, price
        """
        raise NotImplementedError

    def on_tick(self, tick_data: Dict[str, Any]) -> None:
        """Tick 行情回调（可选）"""
        pass

    def on_trade(self, trade: TradeData) -> None:
        """成交回报回调（可选）"""
        pass

    def on_order(self, order: OrderData) -> None:
        """订单状态回调（可选）"""
        pass

    def on_stop(self) -> None:
        """策略停止时回调（可选）"""
        pass


class MACrossoverStrategy(StrategyBase):
    """双均线交叉策略"""

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        super().__init__(name="MA_Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_bar(self, bar: BarData) -> Optional[Dict[str, Any]]:
        self.bar_buffer[bar.code].append(bar)

        prices = [b.close for b in self.bar_buffer[bar.code]]

        if len(prices) < self.slow_period + 1:
            return None

        fast_ma = np.mean(prices[-self.fast_period:])
        fast_ma_prev = np.mean(prices[-self.fast_period - 1:-1])
        slow_ma = np.mean(prices[-self.slow_period:])
        slow_ma_prev = np.mean(prices[-self.slow_period - 1:-1])

        # 金叉买入
        if fast_ma_prev <= slow_ma_prev and fast_ma > slow_ma:
            return {"code": bar.code, "side": "buy", "volume": 100, "price": bar.close}
        # 死叉卖出
        elif fast_ma_prev >= slow_ma_prev and fast_ma < slow_ma:
            return {"code": bar.code, "side": "sell", "volume": 100, "price": bar.close}
        return None


# ═══════════════════════════════════════════
# 4. 回测引擎（事件驱动的简版实现）
# ═══════════════════════════════════════════

class EventDrivenBacktestEngine:
    """
    事件驱动回测引擎

    组件:
        1. EventEngine   — 事件总线
        2. StrategyBase  — 策略逻辑
        3. 撮合器         — 订单执行模拟
        4. 风控断路器      — 硬止损
    """

    def __init__(
        self,
        init_capital: float = 1_000_000,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.0001,
    ):
        self.event_engine = EventEngine()
        self.account = AccountData(nav=init_capital, available_cash=init_capital)
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage

        # 状态
        self.daily_nav: List[Dict[str, Any]] = []
        self.trades: List[TradeData] = []
        self.orders: Dict[str, OrderData] = {}
        self.current_date: Optional[datetime] = None

        # 风控
        self.start_of_day_nav = init_capital

        # 注册事件处理器
        self._register_handlers()

    def _register_handlers(self) -> None:
        """注册内部事件处理器"""
        self.event_engine.register(EventType.BAR, self._on_new_bar)
        self.event_engine.register(EventType.ORDER, self._on_order_update)
        self.event_engine.register(EventType.TRADE, self._on_trade)

    # ─── 事件处理器 ─────────────────────────

    def _on_new_bar(self, event: Event) -> None:
        """新 Bar 到达时触发"""
        bar: BarData = event.data
        self.current_date = bar.date

        # 调用策略
        signal = self.strategy.on_bar(bar)

        # 处理信号
        if signal:
            self._execute_signal(signal, bar.date)

        # 记录每日净值
        nav = self._calc_nav({bar.code: bar.close} if bar.code else {})
        self.daily_nav.append({
            "date": bar.date,
            "equity": nav,
            "code": bar.code,
        })

        # 检查日亏损止损
        self._check_daily_stop(nav)

    def _on_order_update(self, event: Event) -> None:
        """订单状态更新"""
        order: OrderData = event.data
        self.orders[order.order_id] = order

    def _on_trade(self, event: Event) -> None:
        """成交回报"""
        trade: TradeData = event.data
        self.trades.append(trade)
        self._update_account_position(trade)

    # ─── 交易执行 ──────────────────────────

    def _execute_signal(self, signal: Dict[str, Any], timestamp: datetime) -> None:
        """执行交易信号"""
        code = signal["code"]
        side = signal["side"]
        volume = signal.get("volume", 100)
        price = signal["price"]

        # 应用滑点
        if side == "buy":
            exec_price = price * (1 + self.slippage)
        else:
            exec_price = price * (1 - self.slippage)

        # 计算费用
        amount = exec_price * volume
        commission = max(amount * self.commission_rate, 5.0)
        stamp_tax = amount * self.stamp_tax_rate if side == "sell" else 0

        # 检查资金（买入时）
        if side == "buy":
            total_cost = amount + commission
            if total_cost > self.account.available_cash:
                return  # 资金不足
            self.account.available_cash -= total_cost
        else:
            # 检查持仓
            pos = self.account.positions.get(code, {"volume": 0})
            if pos["volume"] < volume:
                return  # 持仓不足
            self.account.available_cash += amount - commission - stamp_tax

        # 更新持仓
        if code not in self.account.positions:
            self.account.positions[code] = {"volume": 0, "avg_cost": 0.0}
        pos = self.account.positions[code]
        if side == "buy":
            pos["avg_cost"] = ((pos["avg_cost"] * pos["volume"] + amount) /
                                (pos["volume"] + volume)) if pos["volume"] + volume > 0 else exec_price
            pos["volume"] += volume
        else:
            pos["volume"] -= volume
            if pos["volume"] <= 0:
                del self.account.positions[code]

        # 发出成交事件
        trade = TradeData(
            trade_id=str(uuid.uuid4())[:12],
            order_id=str(uuid.uuid4())[:12],
            code=code, side=side,
            price=exec_price, volume=volume,
            commission=commission, stamp_tax=stamp_tax,
            trade_time=timestamp,
        )
        self.event_engine.put(Event(EventType.TRADE, trade))

    # ─── 净值计算 ──────────────────────────

    def _calc_nav(self, prices: Dict[str, float]) -> float:
        """计算当前净值"""
        total = self.account.available_cash
        for code, pos in self.account.positions.items():
            price = prices.get(code, 0)
            total += pos["volume"] * price
        return total

    def _update_account_position(self, trade: TradeData) -> None:
        """更新账户头寸（成交后）"""
        self.account.nav = self._calc_nav({trade.code: trade.price})
        self.account.daily_pnl = self.account.nav - self.start_of_day_nav

    def _check_daily_stop(self, nav: float) -> None:
        """检查日亏损止损"""
        daily_return = (nav - self.start_of_day_nav) / self.start_of_day_nav
        if daily_return <= -0.03:
            event = Event(EventType.RISK_WARNING, {
                "reason": "日亏损超过3%",
                "daily_return": daily_return,
                "nav": nav,
            })
            self.event_engine.put(event)

    # ─── 回测主流程 ────────────────────────

    def run_backtest(
        self,
        strategy: StrategyBase,
        data: pd.DataFrame,
        benchmark: str = "000300.SH",
    ) -> Dict[str, Any]:
        """执行事件驱动回测"""
        self.strategy = strategy
        self.event_engine.start_recording()

        bars = self._prepare_bars(data)
        for bar in bars:
            self.start_of_day_nav = self.account.nav
            self.account.daily_pnl = 0
            event = Event(EventType.BAR, bar)
            self.event_engine.put(event)

        self.strategy.on_stop()

        metrics = self._calc_metrics(benchmark)
        return {
            "equity_curve": pd.DataFrame(self.daily_nav),
            "trades": self.trades,
            "metrics": metrics,
            "event_count": len(self.event_engine.get_event_log()),
        }

    def _prepare_bars(self, data: pd.DataFrame) -> List[BarData]:
        """将 DataFrame 转为 BarData 列表"""
        bars = []
        for _, row in data.iterrows():
            bars.append(BarData(
                code=row['code'],
                date=row['date'],
                open=row.get('open', row['close']),
                high=row.get('high', row['close']),
                low=row.get('low', row['close']),
                close=row['close'],
                volume=row.get('volume', row.get('vol', 0)),
                amount=row.get('amount', 0),
            ))
        return sorted(bars, key=lambda b: b.date)

    def _calc_metrics(self, benchmark: str) -> Dict[str, float]:
        """计算绩效指标"""
        if not self.daily_nav:
            return {}
        eq = pd.DataFrame(self.daily_nav)
        eq = eq.set_index('date')['equity']
        returns = eq.pct_change().dropna()
        if len(returns) < 2:
            return {}

        total_return = eq.iloc[-1] / eq.iloc[0] - 1
        n_days = len(returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        volatility = returns.std() * np.sqrt(252)
        max_dd = (eq / eq.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(annual_return / abs(max_dd)) if max_dd != 0 else 0,
            "total_trades": len(self.trades),
        }


# ═══════════════════════════════════════════
# 5. 测试代码
# ═══════════════════════════════════════════

class TestEventDrivenBacktest(unittest.TestCase):
    """事件驱动回测正确性测试"""

    @classmethod
    def setUpClass(cls):
        """创建测试数据"""
        np.random.seed(42)
        codes = ['000001.SZ', '000002.SH', '600000.SH']
        dates = pd.date_range('2024-01-01', periods=100, freq='B')

        rows = []
        for code in codes:
            base_price = np.random.uniform(10, 50)
            prices = base_price * np.cumprod(1 + np.random.normal(0.0002, 0.015, len(dates)))
            for dt, close in zip(dates, prices):
                rows.append({
                    'code': code,
                    'date': dt,
                    'open': close * 0.99,
                    'high': close * 1.02,
                    'low': close * 0.98,
                    'close': close,
                    'volume': np.random.lognormal(10, 0.5),
                })
        cls.test_data = pd.DataFrame(rows).sort_values(['date', 'code'])

    def test_event_registration(self):
        """测试事件注册和分发"""
        engine = EventEngine()
        received = []

        def handler(event: Event):
            received.append(event)

        engine.register(EventType.BAR, handler)
        engine.put(Event(EventType.BAR, BarData(code="000001.SZ", date=datetime.now(),
                                                  open=10, high=11, low=9, close=10.5, volume=10000)))
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0].data, BarData)

    def test_event_recording_and_replay(self):
        """测试事件记录和回放"""
        engine1 = EventEngine()
        engine2 = EventEngine()

        received = []
        def handler(event: Event):
            received.append(event)

        engine2.register(EventType.BAR, handler)

        engine1.start_recording()
        for i in range(5):
            engine1.put(Event(EventType.BAR,
                              BarData(code="000001.SZ", date=datetime.now(),
                                      open=10, high=11, low=9, close=10 + i, volume=10000)))
        engine1.stop_recording()

        # 回放到 engine2
        engine1.replay(engine2)

        self.assertEqual(len(received), 5)
        self.assertEqual(received[-1].data.close, 14)

    def test_ma_crossover_strategy(self):
        """测试双均线策略信号生成"""
        strategy = MACrossoverStrategy(fast_period=5, slow_period=20)

        # 模拟明确的金叉信号：前段价格下降（快线在慢线下），后段反转上涨
        np.random.seed(123)
        bars = []
        base = 30.0
        # 前25天：持续下跌（快线 < 慢线）
        for i in range(25):
            base -= 0.3
            bars.append(BarData(
                code="000001.SZ", date=datetime(2024, 1, 1) + pd.Timedelta(days=i),
                open=base * 0.99, high=base * 1.01, low=base * 0.98,
                close=base, volume=10000,
            ))
        # 后35天：持续上涨（快线将会上穿慢线）
        for i in range(35):
            base += 0.6
            bars.append(BarData(
                code="000001.SZ", date=datetime(2024, 1, 1) + pd.Timedelta(days=25 + i),
                open=base * 0.99, high=base * 1.01, low=base * 0.98,
                close=base, volume=10000,
            ))

        signals = []
        for bar in bars:
            signal = strategy.on_bar(bar)
            if signal:
                signals.append(signal)

        self.assertGreater(len(signals), 0, "应产生至少一个交易信号")

    def test_full_backtest_flow(self):
        """测试完整事件驱动回测流程"""
        strategy = MACrossoverStrategy(fast_period=5, slow_period=20)

        engine = EventDrivenBacktestEngine(
            init_capital=1_000_000,
            commission_rate=0.00025,
            stamp_tax_rate=0.001,
            slippage=0.0001,
        )

        result = engine.run_backtest(strategy, self.test_data)

        # 验证返回结果完整性
        self.assertIn('equity_curve', result)
        self.assertIn('metrics', result)
        self.assertIn('total_trades', result['metrics'])
        self.assertGreater(result['event_count'], 0)

        # 验证净值曲线
        equity = result['equity_curve']
        self.assertGreater(len(equity), 0)
        self.assertIn('equity', equity.columns)

        # 验证绩效指标合理性
        metrics = result['metrics']
        self.assertGreaterEqual(metrics['annual_return'], -1.0)  # 不跌破 100%
        self.assertGreaterEqual(metrics['max_drawdown'], -1.0)

        print(f"\n回测结果:")
        print(f"  年化收益:  {metrics['annual_return']:.2%}")
        print(f"  夏普比率:  {metrics['sharpe_ratio']:.3f}")
        print(f"  最大回撤:  {metrics['max_drawdown']:.2%}")
        print(f"  成交笔数:  {metrics['total_trades']}")
        print(f"  事件数量:  {result['event_count']}")

    def test_risk_circuit_breaker(self):
        """测试风控断路器"""
        engine = EventDrivenBacktestEngine(init_capital=1_000_000)
        warnings_received = []

        def on_risk_warning(event: Event):
            warnings_received.append(event.data)

        engine.event_engine.register(EventType.RISK_WARNING, on_risk_warning)

        # 模拟触发止损
        engine.start_of_day_nav = 1_000_000
        engine._check_daily_stop(950_000)  # 跌 5%

        self.assertEqual(len(warnings_received), 1)
        self.assertIn("日亏损超过3%", warnings_received[0]["reason"])
        self.assertAlmostEqual(warnings_received[0]["daily_return"], -0.05, delta=0.01)

    def test_commission_calculation(self):
        """测试手续费计算"""
        engine = EventDrivenBacktestEngine(
            init_capital=1_000_000,
            commission_rate=0.00025,
            stamp_tax_rate=0.001,
            slippage=0.0001,  # 默认滑点
        )

        # 买入
        engine._execute_signal(
            {"code": "000001.SZ", "side": "buy", "volume": 1000, "price": 20.0},
            datetime.now()
        )

        self.assertLess(engine.account.available_cash, 1_000_000)
        self.assertIn("000001.SZ", engine.account.positions)

        # 验证买入费用（含滑点）
        # exec_price = 20 * (1 + slippage) = 20.002
        # amount = 20.002 * 1000 = 20002
        # commission = max(20002 * 0.00025, 5) = 5.0005
        # expected_cash = 1000000 - 20002 - 5.0005 ≈ 979992.9995
        exec_price = 20.0 * (1 + engine.slippage)
        amount = exec_price * 1000
        commission = max(amount * engine.commission_rate, 5.0)
        expected_cash = 1_000_000 - amount - commission
        self.assertAlmostEqual(engine.account.available_cash, expected_cash, delta=0.01)

    def test_event_driven_vs_procedural(self):
        """
        对比测试：事件驱动 vs 过程式回测

        核心验证点：
        1. 事件驱动模式下净值曲线计算是否正确
        2. 事件驱动与过程式计算是否一致
        """
        import time

        strategy = MACrossoverStrategy(fast_period=5, slow_period=20)

        # 事件驱动模式
        t0 = time.time()
        engine_ed = EventDrivenBacktestEngine(init_capital=1_000_000)
        result_ed = engine_ed.run_backtest(strategy, self.test_data)
        t_ed = time.time() - t0

        # 过程式模式（模拟当前 engine.py 的方式）
        t0 = time.time()
        result_proc = self._procedural_backtest(strategy, self.test_data)
        t_proc = time.time() - t0

        print(f"\n架构对比:")
        print(f"  事件驱动耗时:  {t_ed:.4f}s (事件数: {result_ed['event_count']})")
        print(f"  过程式耗时:    {t_proc:.4f}s")
        print(f"  性能比率:      {t_ed / max(t_proc, 0.001):.2f}x")

        # 聚合事件驱动的净值到日级别
        eq_ed = result_ed['equity_curve']
        # 取每日最后一条净值记录
        eq_ed_daily = eq_ed.groupby('date')['equity'].last().reset_index()
        eq_proc = result_proc['equity_curve']

        # 对齐日期比较净值
        merged = pd.merge(
            eq_ed_daily[['date', 'equity']].rename(columns={'equity': 'eq_ed'}),
            eq_proc[['date', 'equity']].rename(columns={'equity': 'eq_proc'}),
            on='date', how='inner'
        )
        if len(merged) > 1:
            corr = merged['eq_ed'].corr(merged['eq_proc'])
            print(f"  净值曲线相关性:  {corr:.4f}")
            self.assertGreater(corr, 0.7, f"净值曲线相关性应 > 0.7，实际: {corr:.4f}")

    def _procedural_backtest(self, strategy, data):
        """过程式回测（模拟当前代码风格）"""
        data = data.sort_values(['date', 'code'])
        capital = 1_000_000
        cash = capital
        positions = {}
        equity_list = []
        trades = 0

        for dt in data['date'].unique():
            day_data = data[data['date'] == dt]

            for _, row in day_data.iterrows():
                bar = BarData(
                    code=row['code'], date=row['date'],
                    open=row['open'], high=row['high'],
                    low=row['low'], close=row['close'],
                    volume=row['volume'],
                )
                signal = strategy.on_bar(bar)
                if signal:
                    trades += 1
                    code = signal['code']
                    price = signal['price']
                    vol = signal['volume'] if signal['side'] == 'buy' else min(
                        positions.get(code, {}).get('volume', 0), signal['volume']
                    )
                    amount = price * vol

                    if signal['side'] == 'buy':
                        commission = max(amount * 0.00025, 5)
                        if cash >= amount + commission:
                            cash -= amount + commission
                            if code not in positions:
                                positions[code] = {'volume': 0, 'cost': 0}
                            old_cost = positions[code]['cost'] * positions[code]['volume']
                            positions[code]['volume'] += vol
                            positions[code]['cost'] = (old_cost + amount) / positions[code]['volume']
                    else:
                        if code in positions and positions[code]['volume'] >= vol:
                            commission = max(amount * 0.00025, 5)
                            stamp = amount * 0.001
                            cash += amount - commission - stamp
                            positions[code]['volume'] -= vol
                            if positions[code]['volume'] <= 0:
                                del positions[code]

            # 记录每日净值
            stock_value = 0
            for code, pos in positions.items():
                stock_day = day_data[day_data['code'] == code]
                if not stock_day.empty:
                    stock_value += pos['volume'] * stock_day['close'].iloc[0]
            nav = cash + stock_value
            equity_list.append({'date': dt, 'equity': nav})

        return {
            'equity_curve': pd.DataFrame(equity_list),
            'metrics': {'total_trades': trades},
        }


class TestEventDrivenEdgeCases(unittest.TestCase):
    """边界条件测试"""

    def test_empty_data(self):
        """空数据测试"""
        engine = EventDrivenBacktestEngine()
        strategy = MACrossoverStrategy()
        result = engine.run_backtest(strategy, pd.DataFrame())
        self.assertEqual(len(result['equity_curve']), 0)
        self.assertEqual(len(result['trades']), 0)

    def test_single_bar(self):
        """单条数据测试"""
        data = pd.DataFrame([{
            'code': '000001.SZ',
            'date': pd.Timestamp('2024-01-01'),
            'open': 10, 'high': 11, 'low': 9,
            'close': 10.5, 'volume': 10000,
        }])
        engine = EventDrivenBacktestEngine()
        strategy = MACrossoverStrategy()
        result = engine.run_backtest(strategy, data)
        self.assertEqual(len(result['equity_curve']), 1)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__('__main__'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)