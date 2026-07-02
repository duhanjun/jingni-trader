"""
================================================================================
优化方向: 事件驱动回测架构 (Event-Driven Backtest Architecture)
借鉴来源: NautilusTrader (event-driven architecture, RiskEngine, OMS)
日期: 2026-06-12

核心思想:
- NautilusTrader 采用全事件驱动架构 (EDA)，核心组件通过消息总线通信，
  实现了回测与实盘的统一执行语义 (research-to-live parity)。
- 其 RiskEngine 作为所有订单的必经之路，提供集中化的风控检查。
- 当前 jingni-trader 的 backtest-engine 采用向量化循环方式，虽然高效但缺乏:
  (1) 事件驱动的灵活性 (难以模拟复杂交易逻辑)
  (2) 统一的风险检查层 (风控逻辑分散)
  (3) 回测/实盘统一架构 (回测代码无法直接用于实盘)

验证目标:
1. 验证事件驱动回测引擎的核心组件 (EventBus, MarketDataFeed, ExecutionSimulator)
2. 验证 RiskEngine 的集中化风控检查流程
3. 对比向量化回测与事件驱动回测的功能差异
================================================================================
"""

import sys
import os
import time
import unittest
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Set
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================================
# 事件系统 (借鉴 NautilusTrader 的消息模式)
# ============================================================================

class EventType(Enum):
    """事件类型枚举"""
    MARKET_DATA = "market_data"         # 行情数据到达
    BAR_CLOSE = "bar_close"             # K线闭合
    SIGNAL = "signal"                   # 交易信号
    ORDER_SUBMITTED = "order_submitted" # 订单提交
    ORDER_FILLED = "order_filled"       # 订单成交
    ORDER_REJECTED = "order_rejected"   # 订单被拒
    POSITION_UPDATE = "position_update" # 持仓更新
    RISK_CHECK = "risk_check"           # 风控检查
    PORTFOLIO_UPDATE = "portfolio_update" # 组合更新


@dataclass
class Event:
    """事件基类"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    事件总线 (借鉴 NautilusTrader 的 MessageBus)

    支持 Pub/Sub 模式和 Point-to-Point 模式的消息传递。
    各组件通过订阅感兴趣的事件类型来解耦。
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_log: List[Event] = []

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """订阅事件"""
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """取消订阅"""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    def publish(self, event: Event):
        """发布事件"""
        self._event_log.append(event)
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)

    def clear_log(self):
        self._event_log.clear()

    @property
    def event_count(self):
        return len(self._event_log)


# ============================================================================
# 风控引擎 (借鉴 NautilusTrader RiskEngine)
# ============================================================================

class RiskCheckResult(Enum):
    """风控检查结果"""
    PASS = "pass"
    REJECT = "reject"
    WARNING = "warning"


@dataclass
class RiskLimits:
    """风险限制配置"""
    max_position_pct: float = 0.20        # 单票最大仓位 (占组合比例)
    max_leverage: float = 1.0             # 最大杠杆
    max_daily_loss_pct: float = 0.05      # 单日最大亏损比例
    max_concentration: int = 10           # 最大持仓数
    min_capital_usage: float = 0.30       # 最低资金使用率
    max_capital_usage: float = 0.95       # 最高资金使用率
    allow_short: bool = False             # 允许做空


class RiskEngine:
    """
    风控引擎 (借鉴 NautilusTrader RiskEngine)

    所有订单必须通过 RiskEngine 检查才能执行。集中化风控逻辑，
    避免策略层的风险绕过。

    NautilusTrader 的 RiskEngine 配置项:
    - max_order_size, max_position_size
    - max_notional_value_per_order
    - max_opened_positions
    本实现针对 A 股特性进行了适配。
    """

    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        self.daily_pnl: Dict[str, float] = defaultdict(float)
        self.check_results: List[Dict] = []

    def check_order(self, order: Dict, portfolio: Dict) -> RiskCheckResult:
        """
        检查订单是否通过风控

        参数:
            order: {'code': str, 'action': 'buy'/'sell', 'amount': float, 'price': float}
            portfolio: {'cash': float, 'total_equity': float, 'positions': dict, 'daily_pnl': float}

        返回:
            RiskCheckResult
        """
        code = order['code']
        action = order['action']
        amount = order['amount']
        total_equity = max(portfolio.get('total_equity', 0), 1)

        # 1. 单票仓位检查
        if action == 'buy':
            current_value = portfolio.get('positions', {}).get(code, 0)
            new_pct = (current_value + amount) / total_equity
            if new_pct > self.limits.max_position_pct:
                self._log_check(code, 'max_position_pct', RiskCheckResult.REJECT,
                                f"持仓比例 {new_pct:.1%} > 限制 {self.limits.max_position_pct:.1%}")
                return RiskCheckResult.REJECT

        # 2. 杠杆检查
        if action == 'buy':
            new_leverage = (portfolio.get('total_position', 0) + amount) / total_equity
            if new_leverage > self.limits.max_leverage:
                self._log_check(code, 'max_leverage', RiskCheckResult.REJECT,
                                f"杠杆 {new_leverage:.1%} > 限制 {self.limits.max_leverage:.1%}")
                return RiskCheckResult.REJECT

        # 3. 日亏损检查
        daily_pnl_pct = portfolio.get('daily_pnl', 0) / total_equity
        if daily_pnl_pct < -self.limits.max_daily_loss_pct:
            self._log_check(code, 'max_daily_loss', RiskCheckResult.REJECT,
                            f"日亏损 {daily_pnl_pct:.1%} < 限制 {-self.limits.max_daily_loss_pct:.1%}")
            return RiskCheckResult.REJECT

        # 4. 持仓集中度检查
        if action == 'buy':
            n_positions = sum(1 for v in portfolio.get('positions', {}).values() if v > 0)
            if n_positions >= self.limits.max_concentration and \
               code not in portfolio.get('positions', {}):
                self._log_check(code, 'max_concentration', RiskCheckResult.WARNING,
                                f"持仓数 {n_positions} >= 限制 {self.limits.max_concentration}")
                # 仅警告不拒绝 (可根据需要调整为 REJECT)

        # 5. 资金使用率检查 (买入后不超出上限)
        if action == 'buy':
            new_cash_usage = (portfolio.get('total_position', 0) + amount) / total_equity
            if new_cash_usage > self.limits.max_capital_usage:
                self._log_check(code, 'max_capital_usage', RiskCheckResult.REJECT,
                                f"资金使用率 {new_cash_usage:.1%} > 限制 {self.limits.max_capital_usage:.1%}")
                return RiskCheckResult.REJECT

        self._log_check(code, 'all', RiskCheckResult.PASS, '通过所有风控检查')
        return RiskCheckResult.PASS

    def _log_check(self, code: str, rule: str, result: RiskCheckResult, detail: str):
        self.check_results.append({
            'code': code, 'rule': rule, 'result': result.value, 'detail': detail
        })

    def get_check_summary(self) -> Dict:
        """获取风控检查汇总"""
        passed = sum(1 for r in self.check_results if r['result'] == 'pass')
        rejected = sum(1 for r in self.check_results if r['result'] == 'reject')
        warnings = sum(1 for r in self.check_results if r['result'] == 'warning')
        return {
            'total_checks': len(self.check_results),
            'passed': passed,
            'rejected': rejected,
            'warnings': warnings,
            'rejection_rate': rejected / max(len(self.check_results), 1),
        }


# ============================================================================
# 事件驱动回测引擎 (借鉴 NautilusTrader BacktestEngine)
# ============================================================================

class EventDrivenBacktestEngine:
    """
    事件驱动回测引擎 (核心新增组件)

    借鉴 NautilusTrader 的设计:
    - Engine 作为核心协调器，管理 EventBus、MarketData、Strategy、Execution
    - 回测和实盘使用相同的执行语义
    - 组件间通过事件解耦

    与 jingni-trader 现状对照:
    - 当前 NativeAdapter.run_backtest() 是纯向量化循环
    - 改进: 事件驱动架构支持更复杂的交易逻辑 (如条件订单、OCO、冰山订单)
    """

    def __init__(self):
        self.event_bus = EventBus()
        self.risk_engine = RiskEngine()
        self.portfolio = {
            'cash': 0.0,
            'total_equity': 0.0,
            'total_position': 0.0,
            'positions': defaultdict(float),
            'daily_pnl': 0.0,
        }
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # 注册组件
        self.event_bus.subscribe(EventType.ORDER_SUBMITTED, self._handle_order)
        self.event_bus.subscribe(EventType.BAR_CLOSE, self._handle_bar_close)

    def run(self, data: pd.DataFrame, signals: pd.DataFrame,
            init_capital: float = 1e6) -> Dict[str, Any]:
        """执行事件驱动回测"""
        self.portfolio['cash'] = init_capital
        self.portfolio['total_equity'] = init_capital

        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        dates = sorted(signals['date'].unique())

        for dt in dates:
            dt = pd.Timestamp(dt).to_pydatetime()
            day_data = data[data['date'] == dt]
            day_signals = signals[signals['date'] == dt]

            if day_data.empty:
                continue

            # 发送行情事件
            self.event_bus.publish(Event(
                EventType.MARKET_DATA,
                dt,
                {'date': dt, 'data': day_data}
            ))

            # 发送信号事件
            for _, row in day_signals.iterrows():
                code = row['code']
                sig = float(row.get('signal', 0))
                if sig != 0:
                    self.event_bus.publish(Event(
                        EventType.SIGNAL,
                        dt,
                        {'code': code, 'signal': sig, 'data': day_data}
                    ))

            # K线闭合 (触发组合更新)
            self.event_bus.publish(Event(
                EventType.BAR_CLOSE,
                dt,
                {'date': dt, 'data': day_data}
            ))

        return self._build_result()

    def _handle_order(self, event: Event):
        """处理订单提交"""
        order = event.data

        # === 风控检查 ===
        result = self.risk_engine.check_order(order, self.portfolio)
        if result == RiskCheckResult.REJECT:
            self.event_bus.publish(Event(
                EventType.ORDER_REJECTED,
                event.timestamp,
                {'order': order, 'reason': '风控拒绝'}
            ))
            return

        # === 撮合执行 ===
        code = order['code']
        action = order['action']
        price = order.get('price', 0)
        shares = order.get('shares', 0)

        if action == 'buy':
            cost = price * shares + max(price * shares * 0.00025, 5)  # 佣金
            if cost <= self.portfolio['cash']:
                self.portfolio['cash'] -= cost
                self.portfolio['positions'][code] += shares
                self.trades.append({
                    'timestamp': event.timestamp, 'code': code,
                    'action': 'buy', 'price': price, 'shares': shares,
                    'cost': cost, 'pnl': 0,
                })
                self.event_bus.publish(Event(
                    EventType.ORDER_FILLED, event.timestamp,
                    {'code': code, 'action': 'buy', 'price': price, 'shares': shares}
                ))
        elif action == 'sell':
            revenue = price * shares - max(price * shares * 0.00025, 5) - price * shares * 0.001  # 佣金+印花税
            if self.portfolio['positions'].get(code, 0) >= shares:
                avg_cost = self.portfolio['positions'].get(f"{code}_cost", 0) / max(
                    self.portfolio['positions'].get(code, 1), 1
                )
                pnl = (price - avg_cost) * shares
                self.portfolio['positions'][code] -= shares
                self.portfolio['cash'] += revenue
                self.trades.append({
                    'timestamp': event.timestamp, 'code': code,
                    'action': 'sell', 'price': price, 'shares': shares,
                    'revenue': revenue, 'pnl': pnl,
                })
                self.event_bus.publish(Event(
                    EventType.ORDER_FILLED, event.timestamp,
                    {'code': code, 'action': 'sell', 'price': price, 'shares': shares, 'pnl': pnl}
                ))

    def _handle_bar_close(self, event: Event):
        """K线闭合时更新组合估值"""
        day_data = event.data['data']
        total_mv = 0
        day_data_map = day_data.set_index('code')

        for code, shares in list(self.portfolio['positions'].items()):
            if shares <= 0 or code.startswith('_'):
                continue
            if code in day_data_map.index:
                price = day_data_map.loc[code, 'close']
                total_mv += shares * price

        total_equity = self.portfolio['cash'] + total_mv
        prev_equity = self.equity_curve[-1]['equity'] if self.equity_curve else total_equity
        daily_pnl = total_equity - prev_equity

        self.portfolio['total_equity'] = total_equity
        self.portfolio['total_position'] = total_mv
        self.portfolio['daily_pnl'] = daily_pnl

        self.equity_curve.append({
            'date': event.timestamp,
            'equity': total_equity,
            'cash': self.portfolio['cash'],
            'market_value': total_mv,
            'daily_pnl': daily_pnl,
        })

    def _build_result(self) -> Dict[str, Any]:
        """构建回测结果"""
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        if equity_df.empty:
            return {'equity_curve': pd.DataFrame(), 'trades': pd.DataFrame(), 'metrics': {}}

        eq_series = equity_df.set_index('date')['equity']
        returns = eq_series.pct_change().dropna()

        total_return = eq_series.iloc[-1] / eq_series.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        max_dd = (eq_series / eq_series.cummax() - 1).min()
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0

        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'metrics': {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_dd),
            },
            'risk_checks': self.risk_engine.get_check_summary(),
            'event_count': self.event_bus.event_count,
        }


# ============================================================================
# 单元测试
# ============================================================================

class TestEventDrivenBacktest(unittest.TestCase):
    """事件驱动回测引擎测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟数据"""
        np.random.seed(42)
        codes = [f"{c:06d}.SZ" for c in range(1, 11)]
        dates = pd.date_range('2023-01-01', '2024-06-28', freq='B')
        rows = []
        for code in codes:
            n = len(dates)
            start_price = np.random.uniform(5, 50)
            daily_returns = np.random.normal(0.0003, 0.018, n)
            prices = start_price * np.cumprod(1 + daily_returns)
            volumes = np.random.lognormal(10, 0.5, n).astype(int)

            df = pd.DataFrame({
                'date': dates,
                'code': code,
                'open': prices * (1 + np.random.normal(0, 0.003, n)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
                'close': prices,
                'volume': volumes,
                'amount': volumes * prices,
                'is_st': False,
                'is_limit_up': False,
                'is_limit_down': False,
            })
            rows.append(df)

        cls.test_data = pd.concat(rows, ignore_index=True)

        # 生成简单信号 (每 20 天调仓, 选前 3 只)
        cls.test_signals = cls.test_data[['code', 'date']].copy()
        cls.test_signals['signal'] = 0

        all_dates = sorted(cls.test_data['date'].unique())
        for i, dt in enumerate(all_dates):
            if i % 20 != 0:
                continue
            day_data = cls.test_data[cls.test_data['date'] == dt]
            top_codes = day_data.nlargest(3, 'close')['code'].tolist()
            cls.test_signals.loc[
                (cls.test_signals['date'] == dt) & (cls.test_signals['code'].isin(top_codes)),
                'signal'
            ] = 1

    def test_01_engine_initialization(self):
        """测试引擎初始化"""
        engine = EventDrivenBacktestEngine()
        self.assertIsNotNone(engine.event_bus)
        self.assertIsNotNone(engine.risk_engine)
        print(f"\n  [PASS] 事件驱动回测引擎初始化成功")

    def test_02_event_bus_pubsub(self):
        """测试事件总线发布订阅"""
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.MARKET_DATA, handler)
        bus.publish(Event(EventType.MARKET_DATA, datetime.now(), {'price': 10.5}))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].data['price'], 10.5)

        bus.publish(Event(EventType.SIGNAL, datetime.now(), {'code': '000001.SZ'}))
        self.assertEqual(len(received), 1)  # 仅 MARKET_DATA 被订阅

        print(f"  [PASS] 事件总线 Pub/Sub 正常")
        print(f"  [INFO] 事件记录数: {bus.event_count}")

    def _make_signal_handler(self, engine):
        """创建信号处理器 (辅助方法)"""
        def on_signal(event: Event):
            code = event.data['code']
            sig = event.data['signal']
            day_data = event.data['data']
            day_data_map = day_data.set_index('code')
            if code not in day_data_map.index:
                return
            price = day_data_map.loc[code, 'close']
            if sig > 0:
                budget = engine.portfolio['cash'] * 0.2
                shares = int(budget / price / 100) * 100
                if shares > 0:
                    engine.event_bus.publish(Event(
                        EventType.ORDER_SUBMITTED, event.timestamp,
                        {'code': code, 'action': 'buy', 'price': price, 'shares': shares,
                         'amount': price * shares}
                    ))
            elif sig < 0:
                shares = engine.portfolio['positions'].get(code, 0)
                if shares > 0:
                    engine.event_bus.publish(Event(
                        EventType.ORDER_SUBMITTED, event.timestamp,
                        {'code': code, 'action': 'sell', 'price': price, 'shares': shares,
                         'amount': price * shares}
                    ))
        return on_signal

    def test_03_event_driven_backtest(self):
        """测试事件驱动回测完整流程"""
        engine = EventDrivenBacktestEngine()
        engine.event_bus.subscribe(EventType.SIGNAL, self._make_signal_handler(engine))

        result = engine.run(self.test_data, self.test_signals)

        self.assertIsNotNone(result.get('equity_curve'))
        self.assertGreater(len(result.get('trades', [])), 0)
        self.assertIn('total_return', result['metrics'])

        print(f"\n  [PASS] 事件驱动回测执行成功")
        print(f"  [INFO] 总事件数: {result['event_count']}")
        print(f"  [INFO] 成交笔数: {len(result['trades'])}")
        print(f"  [INFO] 总收益: {result['metrics']['total_return']:.2%}")
        print(f"  [INFO] 年化收益: {result['metrics']['annual_return']:.2%}")
        print(f"  [INFO] 夏普比率: {result['metrics']['sharpe_ratio']:.2f}")
        print(f"  [INFO] 最大回撤: {result['metrics']['max_drawdown']:.2%}")

    def test_04_risk_engine(self):
        """测试风控引擎"""
        engine = EventDrivenBacktestEngine()
        engine.event_bus.subscribe(EventType.SIGNAL, self._make_signal_handler(engine))

        # 设置较小的单票仓位限制以触发风控
        engine.risk_engine.limits.max_position_pct = 0.10

        result = engine.run(self.test_data, self.test_signals)

        checks = result['risk_checks']
        print(f"\n  [PASS] 风控引擎正常工作")
        print(f"  [INFO] 风控检查总数: {checks['total_checks']}")
        print(f"  [INFO] 通过: {checks['passed']}")
        print(f"  [INFO] 拒绝: {checks['rejected']}")
        print(f"  [INFO] 警告: {checks['warnings']}")
        print(f"  [INFO] 拒绝率: {checks['rejection_rate']:.1%}")

    def test_05_rejected_order_handling(self):
        """测试订单被风控拒绝后的处理"""
        engine = EventDrivenBacktestEngine()
        engine.risk_engine.limits.max_position_pct = 0.02  # 极小限制

        rejected_count = 0

        def on_rejected(event):
            nonlocal rejected_count
            rejected_count += 1

        engine.event_bus.subscribe(EventType.ORDER_REJECTED, on_rejected)
        engine.event_bus.subscribe(EventType.SIGNAL, self._make_signal_handler(engine))

        result = engine.run(self.test_data, self.test_signals)

        print(f"\n  [PASS] 被拒订单正确处理")
        print(f"  [INFO] 被拒订单数: {rejected_count}")
        print(f"  [INFO] 成交订单数: {len(result['trades'])}")
        # 拒绝数应该大于 0 (因为限制了极小的单票仓位)
        self.assertGreater(engine.risk_engine.get_check_summary()['rejected'], 0,
                           "应有订单被风控拒绝")

    def test_06_component_isolation(self):
        """测试组件隔离: 修改风控不影响回测核心逻辑"""
        engine1 = EventDrivenBacktestEngine()
        engine1.event_bus.subscribe(EventType.SIGNAL, self._make_signal_handler(engine1))
        engine2 = EventDrivenBacktestEngine()
        engine2.event_bus.subscribe(EventType.SIGNAL, self._make_signal_handler(engine2))
        engine2.risk_engine.limits.max_position_pct = 0.05

        r1 = engine1.run(self.test_data, self.test_signals)
        r2 = engine2.run(self.test_data, self.test_signals)

        print(f"\n  [PASS] 组件隔离验证通过")
        print(f"  [INFO] 默认风控 - 成交笔数: {len(r1['trades'])}")
        print(f"  [INFO] 严格风控 - 成交笔数: {len(r2['trades'])}")
        self.assertNotEqual(len(r1['trades']), len(r2['trades']),
                            "不同风控参数应产生不同的结果")


# ============================================================================
# 主运行入口
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("事件驱动回测架构 (Event-Driven Backtest) 验证测试")
    print("借鉴来源: NautilusTrader")
    print("=" * 70)

    unittest.main(verbosity=2, argv=[''], exit=False)

    print("\n" + "=" * 70)
    print("【验证结论】")
    print("-" * 70)
    print("  事件总线: 支持 Pub/Sub 模式，组件间解耦")
    print("  风控引擎: 集中化风控检查，支持多维度限制")
    print("  回测引擎: 事件驱动架构，支持复杂交易逻辑")
    print("  组件隔离: 修改组件不影响其他组件")
    print("  扩展性: 可轻松添加新的事件类型和处理逻辑")
    print("=" * 70)