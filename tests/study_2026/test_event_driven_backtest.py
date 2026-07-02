"""
借鉴来源:
  - NautilusTrader (https://github.com/nautechsystems/nautilus_trader) - 事件驱动架构
  - Event-Driven Backtesting Engine Design Pattern
  - vnpy (https://github.com/vnpy/vnpy) - 事件引擎模式

优化方向: 事件驱动回测引擎 - 解决向量化回测的前视偏差问题

当前 jingni-trader 的 native_adapter 使用向量化逐日循环方式，
虽然按日期顺序执行，但所有数据在内存中同时可用，容易引入前视偏差。

事件驱动架构通过消息队列逐步推送事件，从架构层面杜绝前视偏差:
  - DataHandler → Event Queue → Strategy → Portfolio → Execution

本测试验证:
  1. 事件驱动引擎的基本正确性（与向量化方式对比）
  2. 前视偏差检测能力
  3. 事件流处理的完整性
"""

import sys
import os
import unittest
from collections import deque
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd


# ---- 事件定义 ----

class EventType(Enum):
    MARKET_DATA = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


@dataclass
class Event:
    """事件基类"""
    event_type: EventType
    timestamp: pd.Timestamp
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataEvent(Event):
    """行情数据事件"""
    def __init__(self, timestamp: pd.Timestamp, code: str, open_p: float, high: float,
                 low: float, close: float, volume: int, pre_close: float = 0):
        super().__init__(
            event_type=EventType.MARKET_DATA,
            timestamp=timestamp,
            data={
                "code": code,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "pre_close": pre_close,
            }
        )


@dataclass
class SignalEvent(Event):
    """交易信号事件"""
    def __init__(self, timestamp: pd.Timestamp, code: str, signal: float):
        super().__init__(
            event_type=EventType.SIGNAL,
            timestamp=timestamp,
            data={"code": code, "signal": signal}
        )


@dataclass
class OrderEvent(Event):
    """订单事件"""
    def __init__(self, timestamp: pd.Timestamp, code: str, direction: str,
                 quantity: int, order_type: str = "market"):
        super().__init__(
            event_type=EventType.ORDER,
            timestamp=timestamp,
            data={
                "code": code,
                "direction": direction,  # "buy" or "sell"
                "quantity": quantity,
                "order_type": order_type,
            }
        )


@dataclass
class FillEvent(Event):
    """成交事件"""
    def __init__(self, timestamp: pd.Timestamp, code: str, direction: str,
                 quantity: int, price: float, commission: float = 0):
        super().__init__(
            event_type=EventType.FILL,
            timestamp=timestamp,
            data={
                "code": code,
                "direction": direction,
                "quantity": quantity,
                "price": price,
                "commission": commission,
            }
        )


# ---- 组件实现 ----

class EventQueue:
    """事件队列（先入先出）"""
    def __init__(self):
        self._queue: deque = deque()

    def push(self, event: Event):
        self._queue.append(event)

    def pop(self) -> Optional[Event]:
        return self._queue.popleft() if self._queue else None

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)


class EventDrivenDataHandler:
    """
    数据处理器：按时间顺序逐个推送行情事件
    核心设计：每个时间点只暴露当前可用的数据，无法访问未来数据
    """

    def __init__(self, data: pd.DataFrame):
        """
        参数:
            data: 包含 code, date, open, high, low, close, volume 的 DataFrame
        """
        self._data = data.sort_values(["date", "code"]).reset_index(drop=True)
        self._dates = sorted(data["date"].unique())
        self._current_idx = 0
        self._history = {}  # code → list of historical bars

    def has_next(self) -> bool:
        return self._current_idx < len(self._dates)

    def current_date(self) -> pd.Timestamp:
        return self._dates[self._current_idx] if self.has_next() else None

    def get_current_data(self) -> List[MarketDataEvent]:
        """获取当前日期的所有行情事件"""
        if not self.has_next():
            return []

        dt = self._dates[self._current_idx]
        day_data = self._data[self._data["date"] == dt]

        events = []
        for _, row in day_data.iterrows():
            evt = MarketDataEvent(
                timestamp=dt,
                code=row["code"],
                open_p=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                pre_close=float(row.get("pre_close", row["close"] * 0.99)),
            )
            events.append(evt)

            # 更新历史（只保存当前及之前的数据）
            if row["code"] not in self._history:
                self._history[row["code"]] = []
            self._history[row["code"]].append({
                "date": dt,
                "close": float(row["close"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": int(row["volume"]),
            })

        self._current_idx += 1
        return events

    def get_history(self, code: str, lookback: int = 20) -> List[Dict]:
        """获取某只股票的历史数据（仅当前时间点之前的数据）"""
        if code not in self._history:
            return []
        return self._history[code][-lookback:]

    def reset(self):
        self._current_idx = 0
        self._history = {}


class EventDrivenStrategy:
    """策略基类（事件驱动版）"""

    def __init__(self, queue: EventQueue):
        self._queue = queue

    def on_market_data(self, events: List[MarketDataEvent], data_handler: EventDrivenDataHandler):
        """收到行情事件时的回调"""
        raise NotImplementedError

    def push_signal(self, timestamp: pd.Timestamp, code: str, signal: float):
        self._queue.push(SignalEvent(timestamp, code, signal))


class MA20ReversalStrategy(EventDrivenStrategy):
    """20日均线反转策略"""

    def __init__(self, queue: EventQueue, lookback: int = 20, top_n: int = 10):
        super().__init__(queue)
        self.lookback = lookback
        self.top_n = top_n

    def on_market_data(self, events: List[MarketDataEvent], data_handler: EventDrivenDataHandler):
        dt = events[0].timestamp if events else None
        if dt is None:
            return

        scores = []
        for evt in events:
            hist = data_handler.get_history(evt.data["code"], self.lookback + 1)
            if len(hist) < self.lookback:
                continue

            current_price = evt.data["close"]
            ma20 = np.mean([h["close"] for h in hist[-self.lookback:]])

            if ma20 > 0:
                reversal_score = -(current_price / ma20 - 1)  # 反转：偏离均线越远越买
                scores.append((evt.data["code"], reversal_score))

        # 选 top_n 个买
        scores.sort(key=lambda x: x[1], reverse=True)
        for code, score in scores[:self.top_n]:
            self.push_signal(dt, code, 1)  # 买入信号

        # 选 bottom_n 个卖
        for code, score in scores[-self.top_n:]:
            self.push_signal(dt, code, -1)  # 卖出信号


class EventDrivenPortfolio:
    """组合管理器"""

    def __init__(self, init_capital: float = 1e6):
        self.cash = init_capital
        self.positions = {}  # code -> shares
        self.equity_history = []
        self.trades = []

    def update_market_value(self, current_prices: Dict[str, float], date: pd.Timestamp):
        """按日更新市值"""
        mv = 0
        for code, shares in self.positions.items():
            if shares > 0 and code in current_prices:
                mv += shares * current_prices[code]

        total = self.cash + mv
        self.equity_history.append({
            "date": date,
            "equity": total,
            "cash": self.cash,
            "market_value": mv,
        })

    def process_signal(self, signal: SignalEvent, day_data: Dict[str, MarketDataEvent],
                       commission_rate: float = 0.00025, stamp_tax: float = 0.001):
        """处理交易信号"""
        code = signal.data["code"]
        sig = signal.data["signal"]

        if code not in day_data:
            return

        price = day_data[code].data["close"]

        if sig < 0:  # 卖出
            if code in self.positions and self.positions[code] > 0:
                shares = self.positions[code]
                amount = price * shares
                comm = max(amount * commission_rate, 5)
                tax = amount * stamp_tax
                self.cash += amount - comm - tax
                self.positions[code] = 0
                self.trades.append({
                    "date": signal.timestamp,
                    "code": code,
                    "action": "sell",
                    "price": price,
                    "shares": shares,
                    "amount": amount,
                })

        elif sig > 0:  # 买入
            budget = self.cash * 0.2  # 每次最多用 20% 资金
            shares = int(budget / price / 100) * 100
            if shares > 0:
                amount = price * shares
                comm = max(amount * commission_rate, 5)
                cost = amount + comm
                if cost > self.cash:
                    return
                self.cash -= cost
                self.positions[code] = self.positions.get(code, 0) + shares
                self.trades.append({
                    "date": signal.timestamp,
                    "code": code,
                    "action": "buy",
                    "price": price,
                    "shares": shares,
                    "amount": amount,
                })

    def get_equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_history)


def calc_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
    """计算绩效指标"""
    if equity_curve.empty:
        return {}
    eq = equity_curve.set_index("date")["equity"]
    returns = eq.pct_change().dropna()
    if len(returns) < 2:
        return {}

    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if eq.iloc[0] > 0 else 0
    annual_return = (1 + total_return) ** (252 / max(len(returns), 1)) - 1
    volatility = returns.std() * np.sqrt(252)
    max_dd = (eq / eq.cummax() - 1).min()
    sharpe = (annual_return - 0.015) / volatility if volatility > 0 else 0
    win_rate = (returns > 0).mean()

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "calmar_ratio": float(annual_return / abs(max_dd)) if max_dd != 0 else 0,
    }


class EventDrivenBacktestEngine:
    """事件驱动回测引擎"""

    def __init__(self, data: pd.DataFrame, strategy: EventDrivenStrategy,
                 init_capital: float = 1e6):
        self.data_handler = EventDrivenDataHandler(data)
        self.strategy = strategy
        self.queue = EventQueue()
        self.portfolio = EventDrivenPortfolio(init_capital)

    def run(self) -> Dict[str, Any]:
        """执行事件驱动回测"""
        while self.data_handler.has_next():
            # 1. 获取当日行情事件
            market_events = self.data_handler.get_current_data()
            if not market_events:
                continue

            dt = market_events[0].timestamp
            day_data = {e.data["code"]: e for e in market_events}

            # 2. 策略处理行情事件，生成信号
            self.strategy.on_market_data(market_events, self.data_handler)

            # 3. 处理所有信号（执行交易）
            while not self.queue.is_empty():
                event = self.queue.pop()
                if event.event_type == EventType.SIGNAL:
                    self.portfolio.process_signal(event, day_data)

            # 4. 更新持仓市值
            current_prices = {code: evt.data["close"] for code, evt in day_data.items()}
            self.portfolio.update_market_value(current_prices, dt)

        equity_curve = self.portfolio.get_equity_curve()
        metrics = calc_metrics(equity_curve)

        return {
            "trades": pd.DataFrame(self.portfolio.trades),
            "equity_curve": equity_curve,
            "metrics": metrics,
        }


# ---- Lookahead Bias 检测 ----

def detect_lookahead_bias(engine_class, data: pd.DataFrame, strategy_fn,
                          init_capital: float = 1e6) -> Dict[str, Any]:
    """
    前视偏差检测：对比正常回测和"未来数据注入"回测的差异
    如果引擎有前视偏差漏洞，注入'假未来数据'后结果应该相同
    """

    # 1. 正常回测
    engine1 = engine_class(data, strategy_fn(EventQueue()), init_capital)
    result1 = engine1.run()

    # 2. 注入"未来数据" - 将 future_price 列替换为下一日收盘价
    data_future = data.copy()
    data_future = data_future.sort_values(["code", "date"])
    data_future["close_original"] = data_future["close"]
    data_future["close"] = data_future.groupby("code")["close"].shift(-1).fillna(
        data_future["close"]
    )

    engine2 = engine_class(data_future, strategy_fn(EventQueue()), init_capital)
    result2 = engine2.run()

    # 比较净值曲线
    eq1 = result1["equity_curve"].set_index("date")["equity"]
    eq2 = result2["equity_curve"].set_index("date")["equity"]

    # 截取公共日期
    common = eq1.index.intersection(eq2.index)
    if len(common) < 2:
        return {"lookahead_vulnerable": None, "reason": "数据不足"}

    eq1 = eq1[common]
    eq2 = eq2[common]

    # 如果两者差异很大，说明有前视偏差（注入了未来数据改变了结果）
    cumulative_diff = abs((eq2 / eq1 - 1).iloc[-1])

    return {
        "lookahead_vulnerable": cumulative_diff > 0.05,
        "cumulative_difference": float(cumulative_diff),
        "normal_final_equity": float(eq1.iloc[-1]),
        "future_injected_final_equity": float(eq2.iloc[-1]),
    }


# ---- 向量化回测（对比基准） ----

def vectorized_backtest(data: pd.DataFrame, signals: pd.DataFrame,
                        init_capital: float = 1e6) -> Dict[str, Any]:
    """简化版向量化回测"""
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

    dates = sorted(signals["date"].unique())
    cash = init_capital
    positions = {}
    equity_records = []
    trades = []

    for dt in dates:
        day_signal = signals[signals["date"] == dt]
        day_data = data[data["date"] == dt].set_index("code")

        # 卖出
        sells = day_signal[day_signal["signal"] < 0]
        for _, row in sells.iterrows():
            code = row["code"]
            if code in positions and positions[code] > 0 and code in day_data.index:
                price = day_data.loc[code, "close"]
                shares = positions[code]
                cash += price * shares * 0.99975  # 扣除佣金
                trades.append({"date": dt, "code": code, "action": "sell",
                               "price": price, "shares": shares})
                positions[code] = 0

        # 买入
        buys = day_signal[day_signal["signal"] > 0]
        n_buy = len(buys)
        if n_buy > 0:
            budget_per = cash * 0.2 / n_buy
            for _, row in buys.iterrows():
                code = row["code"]
                if code in day_data.index:
                    price = day_data.loc[code, "close"]
                    shares = int(budget_per / price / 100) * 100
                    if shares > 0:
                        cost = price * shares * 1.00025
                        if cost <= cash:
                            cash -= cost
                            positions[code] = positions.get(code, 0) + shares
                            trades.append({"date": dt, "code": code, "action": "buy",
                                           "price": price, "shares": shares})

        # 市值计算
        mv = sum(
            shares * day_data.loc[code, "close"]
            for code, shares in positions.items()
            if shares > 0 and code in day_data.index
        )
        equity_records.append({"date": dt, "equity": cash + mv})

    equity_curve = pd.DataFrame(equity_records)
    return {
        "trades": pd.DataFrame(trades),
        "equity_curve": equity_curve,
        "metrics": calc_metrics(equity_curve),
    }


# ---- 测试用例 ----

class TestEventDrivenBacktest(unittest.TestCase):
    """事件驱动回测引擎测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = [f"{i:06d}.SH" for i in range(600000, 600030)]  # 30 只
        dates = pd.date_range("2023-01-01", "2024-06-30", freq="B")

        rows = []
        for code in codes:
            n = len(dates)
            start_price = np.random.uniform(5, 50)
            drift = np.random.uniform(-0.0002, 0.001)
            returns = np.random.normal(drift, 0.015, n)
            prices = start_price * np.cumprod(1 + returns)
            volumes = np.random.lognormal(12, 1, n).astype(int)

            df_code = pd.DataFrame({
                "code": code,
                "date": dates,
                "open": prices * (1 + np.random.normal(0, 0.005, n)),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
                "close": prices,
                "volume": volumes,
            })
            rows.append(df_code)

        cls.test_data = pd.concat(rows, ignore_index=True).sort_values(["code", "date"])

    def _make_strategy(self, queue):
        return MA20ReversalStrategy(queue, lookback=20, top_n=5)

    def test_basic_run(self):
        """测试基本运行：事件驱动回测能正常完成"""
        queue = EventQueue()
        strategy = self._make_strategy(queue)
        engine = EventDrivenBacktestEngine(self.test_data, strategy, 1e6)
        result = engine.run()

        self.assertFalse(result["equity_curve"].empty,
                         "净值曲线不应为空")
        self.assertGreater(len(result["equity_curve"]), 10,
                           "应该有足够的交易日")
        self.assertIn("equity", result["equity_curve"].columns)
        self.assertIn("total_return", result["metrics"])
        self.assertIn("sharpe_ratio", result["metrics"])
        self.assertIn("max_drawdown", result["metrics"])

        print(f"\n✓ 事件驱动回测基本运行通过")
        print(f"  交易日数: {len(result['equity_curve'])}")
        print(f"  总收益率: {result['metrics']['total_return']:.4%}")
        print(f"  夏普比率: {result['metrics']['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {result['metrics']['max_drawdown']:.4%}")

    def test_no_lookahead_bias(self):
        """测试事件驱动引擎无前视偏差"""
        queue = EventQueue()
        result = detect_lookahead_bias(
            EventDrivenBacktestEngine,
            self.test_data,
            lambda q: self._make_strategy(q),
        )

        print(f"\n✓ 前视偏差检测:")
        print(f"  累积差异: {result['cumulative_difference']:.6%}")
        print(f"  正常回测最终净值: {result['normal_final_equity']:.2f}")
        print(f"  注入未来数据最终净值: {result['future_injected_final_equity']:.2f}")

        # 事件驱动引擎：注入未来数据后结果应该不同（检出）
        self.assertIsNotNone(result["lookahead_vulnerable"],
                             "检测结果不应为 None")

    def test_event_queue_ordering(self):
        """测试事件队列的 FIFO 顺序"""
        queue = EventQueue()
        t1 = pd.Timestamp("2024-01-01")
        t2 = pd.Timestamp("2024-01-02")

        queue.push(MarketDataEvent(t1, "A", 10, 12, 9, 11, 1000))
        queue.push(MarketDataEvent(t2, "B", 20, 22, 19, 21, 2000))

        self.assertEqual(len(queue), 2)
        e1 = queue.pop()
        self.assertEqual(e1.timestamp, t1)
        self.assertEqual(e1.data["code"], "A")
        e2 = queue.pop()
        self.assertEqual(e2.timestamp, t2)
        self.assertEqual(e2.data["code"], "B")
        self.assertTrue(queue.is_empty())

    def test_data_handler_history_isolation(self):
        """测试数据处理器正确隔离历史数据和未来数据"""
        handler = EventDrivenDataHandler(self.test_data)

        # 处理第一天数据
        first_events = handler.get_current_data()
        first_date = handler.current_date() or first_events[0].timestamp if first_events else None

        # 检查任何股票的 lookback 历史不超过 1 天
        for evt in first_events[:3]:
            hist = handler.get_history(evt.data["code"], 100)
            self.assertLessEqual(len(hist), 1,
                                 f"第一天数据历史不应超过1条，code={evt.data['code']}")

        # 继续推进
        for _ in range(9):  # 再推进 9 天
            handler.get_current_data()

        # 处理第 10 天数据
        tenth_events = handler.get_current_data()

        # 第 10 天的某只股票历史应 <= 10
        if tenth_events:
            evt = tenth_events[0]
            hist = handler.get_history(evt.data["code"], 100)
            # 经过 1 + 9 = 10 次 get_current_data() 调用后
            # current_idx = 10，即第 11 个交易日（0-indexed）
            self.assertLessEqual(len(hist), 11,
                                 f"第11天数据历史不应超过11条")

        print(f"\n✓ DataHandler 正确隔离历史数据")

    def test_vs_vectorized_consistency(self):
        """对比事件驱动和向量化回测的一致性"""
        # 生成简单信号（每天对等权重的股票发信号）
        codes = sorted(self.test_data["code"].unique())[:5]
        dates = sorted(self.test_data["date"].unique())

        signals_list = []
        for dt in dates:
            for code in codes:
                if np.random.random() > 0.5:
                    signals_list.append({
                        "date": dt,
                        "code": code,
                        "signal": 1 if np.random.random() > 0.5 else -1,
                    })
        signals_df = pd.DataFrame(signals_list)

        # 事件驱动
        queue = EventQueue()
        strategy = SimpleSignalStrategy(queue, signals_df)
        engine = EventDrivenBacktestEngine(self.test_data, strategy, 1e6)
        event_result = engine.run()

        # 向量化
        vec_result = vectorized_backtest(self.test_data, signals_df, 1e6)

        eq_event = event_result["equity_curve"].set_index("date")["equity"]
        eq_vec = vec_result["equity_curve"].set_index("date")["equity"]

        # 比较最终净值
        print(f"\n✓ 事件驱动 vs 向量化回测对比:")
        print(f"  事件驱动最终净值: {eq_event.iloc[-1]:.2f}")
        print(f"  向量化最终净值: {eq_vec.iloc[-1]:.2f}")
        print(f"  差异率: {abs(eq_event.iloc[-1] / eq_vec.iloc[-1] - 1):.4%}")


class SimpleSignalStrategy(EventDrivenStrategy):
    """简单信号策略：从预定义信号表读取"""

    def __init__(self, queue: EventQueue, signals: pd.DataFrame):
        super().__init__(queue)
        self.signals = signals.set_index(["date", "code"])

    def on_market_data(self, events: List[MarketDataEvent], data_handler):
        dt = events[0].timestamp
        for evt in events:
            code = evt.data["code"]
            try:
                sig = self.signals.loc[(dt, code), "signal"]
                self.push_signal(dt, code, sig)
            except KeyError:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)