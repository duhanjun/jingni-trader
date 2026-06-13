"""
验证测试：事件驱动回测引擎
借鉴来源：qf-lib (https://github.com/quarkfin/qf-lib) - 事件驱动架构设计
        Qlib (https://github.com/microsoft/qlib) - 模块化 Alpha/风险/执行分层设计
优化方向：将 jingni-trader 的回测引擎从简单的信号驱动模式升级为事件驱动架构，
        支持 Alpha 模型、风控模型、仓位管理、执行模型的模块化组合。

设计思路：
  - qf-lib 将回测拆分为 Alpha Models、Risk Management、Position Sizing、Execution 四个独立模块
  - 每个模块通过事件总线（EventBus）通信，支持灵活组合
  - 策略从回测切换到实盘无需修改代码
  - 本测试验证事件驱动模型在功能正确性和扩展性方面的表现
"""
import sys
import os
import unittest
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================
# 事件驱动架构核心组件
# ============================================================

class EventType(Enum):
    """事件类型枚举"""
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ALERT = "risk_alert"


@dataclass
class Event:
    """事件基类"""
    event_type: EventType
    timestamp: pd.Timestamp
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """事件总线（发布-订阅模式）"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[callable]] = defaultdict(list)
        self._event_log: List[Event] = []
    
    def subscribe(self, event_type: EventType, handler: callable):
        """订阅事件"""
        self._subscribers[event_type].append(handler)
    
    def publish(self, event: Event):
        """发布事件"""
        self._event_log.append(event)
        for handler in self._subscribers.get(event.event_type, []):
            handler(event)
    
    def clear_log(self):
        self._event_log.clear()


# ============================================================
# 数据馈送器
# ============================================================

class DataFeed:
    """行情数据馈送器，按时间顺序推送 MarketData 事件"""
    
    def __init__(self, data: pd.DataFrame, event_bus: EventBus):
        """
        参数:
            data: 含 code, date, open, high, low, close, volume 等列的 DataFrame
            event_bus: 事件总线
        """
        self.data = data.sort_values(['date', 'code'])
        self.event_bus = event_bus
        self._dates = sorted(self.data['date'].unique())
        self._current_idx = 0
    
    def __iter__(self):
        self._current_idx = 0
        return self
    
    def __next__(self):
        if self._current_idx >= len(self._dates):
            raise StopIteration
        
        current_date = self._dates[self._current_idx]
        daily_data = self.data[self.data['date'] == current_date]
        self._current_idx += 1
        
        event = Event(
            event_type=EventType.MARKET_DATA,
            timestamp=pd.Timestamp(current_date),
            data={"daily_data": daily_data, "date": current_date}
        )
        return event
    
    def __len__(self):
        return len(self._dates)


# ============================================================
# Alpha 模型（信号生成）
# ============================================================

class AlphaModel(ABC):
    """Alpha 模型抽象基类 - 借鉴 qf-lib 的 AlphaModel 设计"""
    
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        根据行情数据生成交易信号
        
        返回:
            DataFrame with columns: code, signal_weight (-1 to 1)
        """
        pass


class ReversalAlphaModel(AlphaModel):
    """反转因子 Alpha 模型"""
    
    def __init__(self, window: int = 20, top_pct: float = 0.2):
        self.window = window
        self.top_pct = top_pct
        self._factor_cache = {}
    
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        if market_data.empty:
            return pd.DataFrame(columns=['code', 'signal_weight'])
        
        codes = market_data['code'].unique()
        signals = []
        
        for code in codes:
            code_data = market_data[market_data['code'] == code].sort_values('date')
            if len(code_data) < self.window:
                signals.append({'code': code, 'signal_weight': 0.0})
                continue
            
            returns = code_data['close'].pct_change(self.window).iloc[-1]
            
            # 反转：跌得多买，涨得多卖
            signal = -np.clip(returns * 5, -1, 1) if not np.isnan(returns) else 0
            signals.append({'code': code, 'signal_weight': float(signal)})
        
        result = pd.DataFrame(signals)
        
        # 选 top/bottom 各 20% 为强信号
        if len(result) > 0 and 'signal_weight' in result.columns:
            threshold_high = result['signal_weight'].quantile(1 - self.top_pct)
            threshold_low = result['signal_weight'].quantile(self.top_pct)
            result.loc[result['signal_weight'] < threshold_low, 'signal_weight'] = 0
            result.loc[result['signal_weight'] > threshold_high, 'signal_weight'] = result.loc[result['signal_weight'] > threshold_high, 'signal_weight']
        
        return result


# ============================================================
# 风险模型
# ============================================================

class RiskModel(ABC):
    """风险模型抽象基类"""
    
    @abstractmethod
    def check_risk(self, portfolio: Dict[str, Any], proposed_orders: List[Dict]) -> Tuple[bool, str]:
        """
        检查风险并返回是否允许执行
        
        返回:
            (is_allowed, reason)
        """
        pass


class BasicRiskModel(RiskModel):
    """基础风控模型：单票上限、单日亏损限制"""
    
    def __init__(self, 
                 max_single_stock_weight: float = 0.10,
                 max_daily_loss_ratio: float = 0.03,
                 max_positions: int = 20):
        self.max_single_stock_weight = max_single_stock_weight
        self.max_daily_loss_ratio = max_daily_loss_ratio
        self.max_positions = max_positions
        self._daily_pnl = 0.0
        self._daily_reset_date = None
    
    def check_risk(self, portfolio: Dict[str, Any], proposed_orders: List[Dict]) -> Tuple[bool, str]:
        equity = portfolio.get('equity', 1000000)
        
        # 检查单票权重上限
        for order in proposed_orders:
            order_value = abs(order.get('value', 0))
            if order_value / equity > self.max_single_stock_weight:
                return False, f"单票权重超限: {order_value/equity:.2%} > {self.max_single_stock_weight:.2%}"
        
        # 检查持仓数量上限
        current_positions = portfolio.get('positions', {})
        buy_count = sum(1 for o in proposed_orders if o.get('side') == 'buy')
        if len(current_positions) + buy_count > self.max_positions:
            return False, f"持仓数量超限: {len(current_positions) + buy_count} > {self.max_positions}"
        
        # 检查单日亏损限制
        if abs(self._daily_pnl) / equity > self.max_daily_loss_ratio:
            return False, f"单日亏损超限: {abs(self._daily_pnl)/equity:.2%} > {self.max_daily_loss_ratio:.2%}"
        
        return True, "risk check passed"
    
    def update_daily_pnl(self, pnl: float, date):
        if self._daily_reset_date != date:
            self._daily_pnl = 0.0
            self._daily_reset_date = date
        self._daily_pnl += pnl


# ============================================================
# 仓位管理模型
# ============================================================

class PositionSizer(ABC):
    """仓位管理抽象基类"""
    
    @abstractmethod
    def size_positions(self, 
                       signals: pd.DataFrame, 
                       portfolio: Dict[str, Any],
                       market_data: pd.DataFrame) -> List[Dict]:
        """
        根据信号计算每笔交易的目标仓位
        
        返回:
            List of order dicts: {code, side, quantity, value, price}
        """
        pass


class EqualWeightSizer(PositionSizer):
    """等权仓位管理器"""
    
    def __init__(self, max_positions: int = 20, max_single_weight: float = 0.05):
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight
    
    def size_positions(self, 
                       signals: pd.DataFrame, 
                       portfolio: Dict[str, Any],
                       market_data: pd.DataFrame) -> List[Dict]:
        equity = portfolio.get('equity', 1000000)
        current_positions = portfolio.get('positions', {})
        
        # 只选取信号最强的 top N
        strong_signals = signals[signals['signal_weight'] != 0].copy()
        if strong_signals.empty:
            return []
        
        strong_signals['abs_weight'] = strong_signals['signal_weight'].abs()
        strong_signals = strong_signals.sort_values('abs_weight', ascending=False)
        strong_signals = strong_signals.head(self.max_positions)
        
        orders = []
        single_value = equity * self.max_single_weight
        
        for _, row in strong_signals.iterrows():
            code = row['code']
            weight = row['signal_weight']
            
            if code in current_positions:
                current_weight = current_positions[code]['weight']
                if abs(weight) < 0.01:
                    # 平仓
                    orders.append({
                        'code': code,
                        'side': 'sell',
                        'quantity': current_positions[code]['quantity'],
                        'value': current_positions[code]['value'],
                        'price': 0  # 由执行引擎填充
                    })
                continue
            
            if weight > 0:
                orders.append({
                    'code': code,
                    'side': 'buy',
                    'quantity': 0,
                    'value': single_value,
                    'price': 0
                })
            elif weight < 0:
                orders.append({
                    'code': code,
                    'side': 'sell',
                    'quantity': 0,
                    'value': 0,
                    'price': 0
                })
        
        return orders


# ============================================================
# 执行引擎
# ============================================================

class ExecutionEngine:
    """执行引擎 - 模拟订单执行，考虑滑点、佣金、税费"""
    
    def __init__(self,
                 commission_rate: float = 0.00025,
                 stamp_tax_rate: float = 0.001,
                 slippage: float = 0.001,
                 min_commission: float = 5.0,
                 t_plus_1: bool = True):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission
        self.t_plus_1 = t_plus_1
        self._pending_sells = {}  # T+1 卖出暂存
    
    def execute_orders(self, 
                       orders: List[Dict], 
                       market_data: pd.DataFrame,
                       portfolio: Dict[str, Any]) -> List[Dict]:
        """执行订单并返回成交记录"""
        fills = []
        
        for order in orders:
            code = order['code']
            code_data = market_data[market_data['code'] == code]
            if code_data.empty:
                continue
            
            price_row = code_data.iloc[-1]
            base_price = price_row['close']
            
            # 涨跌停检查
            if 'is_limit_up' in code_data.columns and price_row.get('is_limit_up', False):
                continue
            if 'is_limit_down' in code_data.columns and price_row.get('is_limit_down', False):
                continue
            
            # T+1 卖出暂存检查
            if self.t_plus_1 and order['side'] == 'sell':
                if code in self._pending_sells:
                    continue  # 今天刚买的不能卖
            
            # 计算滑点
            if order['side'] == 'buy':
                exec_price = base_price * (1 + self.slippage)
            else:
                exec_price = base_price * (1 - self.slippage)
            
            # 计算数量
            if order['value'] > 0:
                quantity = int(order['value'] / exec_price / 100) * 100  # 整手
                if quantity == 0:
                    continue
            else:
                quantity = order.get('quantity', 0)
            
            if quantity <= 0:
                continue
            
            value = quantity * exec_price
            commission = max(value * self.commission_rate, self.min_commission)
            stamp_tax = value * self.stamp_tax_rate if order['side'] == 'sell' else 0
            total_cost = value + commission + stamp_tax
            
            fill = {
                'code': code,
                'side': order['side'],
                'quantity': quantity,
                'price': exec_price,
                'value': value,
                'commission': commission,
                'stamp_tax': stamp_tax,
                'total_cost': total_cost
            }
            fills.append(fill)
            
            # 更新持仓
            if order['side'] == 'buy':
                portfolio['cash'] -= total_cost
                if code not in portfolio['positions']:
                    portfolio['positions'][code] = {'quantity': 0, 'value': 0, 'weight': 0}
                portfolio['positions'][code]['quantity'] += quantity
                portfolio['positions'][code]['value'] += value
                if self.t_plus_1:
                    self._pending_sells[code] = True
            else:
                portfolio['cash'] += total_cost - stamp_tax
                if code in portfolio['positions']:
                    portfolio['positions'][code]['quantity'] -= quantity
                    portfolio['positions'][code]['value'] -= value
                    if portfolio['positions'][code]['quantity'] <= 0:
                        del portfolio['positions'][code]
        
        # 更新组合权益
        position_value = sum(p['value'] for p in portfolio['positions'].values())
        portfolio['equity'] = portfolio['cash'] + position_value
        
        # 清理 T+1 暂存
        self._pending_sells.clear()
        
        return fills


# ============================================================
# 事件驱动回测引擎
# ============================================================

class EventDrivenBacktestEngine:
    """事件驱动回测引擎 - 整合所有模块"""
    
    def __init__(self,
                 alpha_model: AlphaModel,
                 risk_model: RiskModel,
                 position_sizer: PositionSizer,
                 execution_engine: ExecutionEngine,
                 init_capital: float = 1e6):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.position_sizer = position_sizer
        self.execution_engine = execution_engine
        self.event_bus = EventBus()
        
        self.portfolio = {
            'cash': init_capital,
            'equity': init_capital,
            'positions': {},
            'initial_capital': init_capital
        }
        
        self.equity_curve = []
        self.all_fills = []
    
    def run(self, data: pd.DataFrame, benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """运行事件驱动回测"""
        data_feed = DataFeed(data, self.event_bus)
        
        for event in data_feed:
            date = event.timestamp
            market_data = event.data['daily_data']
            
            # 1. Alpha 模型生成信号
            signals = self.alpha_model.generate_signals(market_data)
            
            if not signals.empty:
                self.event_bus.publish(Event(
                    EventType.SIGNAL, date, {"signals": signals}
                ))
                
                # 2. 仓位管理计算目标仓位
                proposed_orders = self.position_sizer.size_positions(
                    signals, self.portfolio, market_data
                )
                
                # 3. 风控检查
                allowed, reason = self.risk_model.check_risk(self.portfolio, proposed_orders)
                
                if allowed:
                    # 4. 执行订单
                    fills = self.execution_engine.execute_orders(
                        proposed_orders, market_data, self.portfolio
                    )
                    
                    for f in fills:
                        f['date'] = date
                    self.all_fills.extend(fills)
                else:
                    self.event_bus.publish(Event(
                        EventType.RISK_ALERT, date, {"reason": reason}
                    ))
            
            # 更新持仓市值（按收盘价）
            self._mark_to_market(market_data, date)
            
            # 记录净值曲线
            benchmark_value = 1.0
            if benchmark_data is not None:
                bench_row = benchmark_data[benchmark_data['date'] == date]
                if not bench_row.empty:
                    benchmark_value = bench_row.iloc[0].get('close', benchmark_value)
            
            self.equity_curve.append({
                'date': date,
                'equity': self.portfolio['equity'],
                'cash': self.portfolio['cash'],
                'benchmark': benchmark_value
            })
        
        return {
            'equity_curve': pd.DataFrame(self.equity_curve),
            'fills': pd.DataFrame(self.all_fills) if self.all_fills else pd.DataFrame(),
            'portfolio': self.portfolio,
            'event_log': self.event_bus._event_log
        }
    
    def _mark_to_market(self, market_data: pd.DataFrame, date):
        """按市价更新持仓市值"""
        total_position_value = 0
        for code, position in list(self.portfolio['positions'].items()):
            code_data = market_data[market_data['code'] == code]
            if not code_data.empty:
                close_price = code_data.iloc[-1]['close']
                position['value'] = position['quantity'] * close_price
                total_position_value += position['value']
        
        self.portfolio['equity'] = self.portfolio['cash'] + total_position_value
        total = self.portfolio['equity']
        for pos in self.portfolio['positions'].values():
            pos['weight'] = pos['value'] / total if total > 0 else 0


# ============================================================
# 测试用例
# ============================================================

class TestEventDrivenBacktest(unittest.TestCase):
    """事件驱动回测引擎测试"""
    
    @classmethod
    def setUpClass(cls):
        """生成模拟测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
        codes = [f'{i:06d}.SH' for i in range(600001, 600021)]  # 20 stocks
        
        rows = []
        for date in dates:
            for code in codes:
                # 模拟价格走势（带反转特性）
                code_idx = codes.index(code)
                trend = np.sin(code_idx * 0.3 + date.dayofyear * 0.05) * 0.3
                noise = np.random.randn() * 0.02
                price = 10 + trend + noise
                
                rows.append({
                    'code': code,
                    'date': date,
                    'open': price * (1 + np.random.randn() * 0.005),
                    'high': price * (1 + abs(np.random.randn() * 0.01)),
                    'low': price * (1 - abs(np.random.randn() * 0.01)),
                    'close': price,
                    'volume': np.random.randint(100000, 1000000),
                    'is_limit_up': False,
                    'is_limit_down': False,
                    'is_st': False,
                })
        
        cls.test_data = pd.DataFrame(rows)
        
        # 生成基准数据
        bench_rows = []
        bench_price = 1.0
        for date in dates:
            bench_price *= (1 + np.random.randn() * 0.01)
            bench_rows.append({'date': date, 'close': bench_price})
        cls.benchmark_data = pd.DataFrame(bench_rows)
    
    def test_basic_event_flow(self):
        """测试事件驱动回测基本流程"""
        engine = EventDrivenBacktestEngine(
            alpha_model=ReversalAlphaModel(window=20, top_pct=0.3),
            risk_model=BasicRiskModel(max_single_stock_weight=0.10, max_positions=20),
            position_sizer=EqualWeightSizer(max_positions=10, max_single_weight=0.05),
            execution_engine=ExecutionEngine(commission_rate=0.00025, slippage=0.001),
            init_capital=1e6
        )
        
        result = engine.run(self.test_data, self.benchmark_data)
        
        # 验证基本结构
        self.assertIsNotNone(result)
        self.assertIn('equity_curve', result)
        self.assertIn('fills', result)
        
        equity_curve = result['equity_curve']
        self.assertFalse(equity_curve.empty)
        self.assertIn('equity', equity_curve.columns)
        self.assertIn('date', equity_curve.columns)
        
        # 验证起始资金
        self.assertAlmostEqual(equity_curve['equity'].iloc[0], 1e6, delta=1000)
        
        # 验证事件日志
        self.assertGreater(len(result['event_log']), 0)
        
        print(f"\n[事件驱动回测] 交易日数: {len(equity_curve)}")
        print(f"[事件驱动回测] 成交笔数: {len(result['fills'])}")
        print(f"[事件驱动回测] 事件总数: {len(result['event_log'])}")
        print(f"[事件驱动回测] 最终权益: {equity_curve['equity'].iloc[-1]:.2f}")
    
    def test_risk_model_intervention(self):
        """测试风控模型干预"""
        strict_risk = BasicRiskModel(max_single_stock_weight=0.01, max_positions=3)
        
        # 模拟高权重订单
        portfolio = {'equity': 1e6, 'cash': 1e6, 'positions': {}}
        large_order = [{'code': '600001.SH', 'side': 'buy', 'value': 200000}]
        
        allowed, reason = strict_risk.check_risk(portfolio, large_order)
        self.assertFalse(allowed)
        self.assertIn("单票权重超限", reason)
        
        # 正常订单应通过
        small_order = [{'code': '600001.SH', 'side': 'buy', 'value': 5000}]
        allowed, reason = strict_risk.check_risk(portfolio, small_order)
        self.assertTrue(allowed)
        print(f"\n[风控测试] 大单拒绝: {reason}")
        print(f"[风控测试] 小单通过: {allowed}")
    
    def test_slippage_and_commission(self):
        """测试滑点和佣金计算"""
        engine = ExecutionEngine(
            commission_rate=0.00025,
            stamp_tax_rate=0.001,
            slippage=0.001,
            min_commission=5.0
        )
        
        portfolio = {'cash': 1e6, 'equity': 1e6, 'positions': {}}
        market_data = pd.DataFrame([{
            'code': '600001.SH', 'close': 10.0,
            'is_limit_up': False, 'is_limit_down': False
        }])
        
        orders = [{'code': '600001.SH', 'side': 'buy', 'value': 100000, 'quantity': 0, 'price': 0}]
        fills = engine.execute_orders(orders, market_data, portfolio)
        
        self.assertEqual(len(fills), 1)
        fill = fills[0]
        
        # 买入价应含滑点: 10 * (1 + 0.001) = 10.01
        self.assertAlmostEqual(fill['price'], 10.01, delta=0.001)
        
        # 数量应为整手: floor(100000 / 10.01 / 100) * 100
        expected_qty = int(100000 / 10.01 / 100) * 100
        self.assertEqual(fill['quantity'], expected_qty)
        
        # 佣金 >= 最低佣金 5 元
        self.assertGreaterEqual(fill['commission'], 5.0)
        
        # 买入不收印花税
        self.assertEqual(fill['stamp_tax'], 0.0)
        
        print(f"\n[滑点佣金测试] 买入价: {fill['price']:.3f}, 数量: {fill['quantity']}, "
              f"佣金: {fill['commission']:.2f}, 印花税: {fill['stamp_tax']:.2f}")
    
    def test_position_sizer_equal_weight(self):
        """测试等权仓位管理"""
        sizer = EqualWeightSizer(max_positions=5, max_single_weight=0.05)
        
        # 生成信号
        signals = pd.DataFrame([
            {'code': c, 'signal_weight': w}
            for c, w in zip(
                [f'{i:06d}.SH' for i in range(600001, 600011)],
                [0.8, 0.7, 0.6, 0.5, -0.4, -0.3, -0.2, 0.1, -0.1, 0.05]
            )
        ])
        
        portfolio = {'equity': 1e6, 'cash': 1e6, 'positions': {}}
        orders = sizer.size_positions(signals, portfolio, pd.DataFrame())
        
        # 最多 max_positions=5 个订单
        self.assertLessEqual(len(orders), 5)
        
        # 验证每笔订单值
        single_value = 1e6 * 0.05
        for order in orders:
            if order['side'] == 'buy':
                self.assertAlmostEqual(order['value'], single_value, delta=1)
        
        print(f"\n[仓位管理测试] 信号数: {len(signals)}, 订单数: {len(orders)}")
        for o in orders:
            print(f"  {o['code']}: {o['side']}, value={o['value']:.0f}")
    
    def test_modular_composition(self):
        """测试模块可替换性"""
        # 使用不同的 Alpha 模型
        class MomentumAlpha(AlphaModel):
            def generate_signals(self, market_data):
                if market_data.empty:
                    return pd.DataFrame(columns=['code', 'signal_weight'])
                codes = market_data['code'].unique()
                result = []
                for code in codes:
                    cd = market_data[market_data['code'] == code].sort_values('date')
                    if len(cd) < 5:
                        result.append({'code': code, 'signal_weight': 0.0})
                    else:
                        ret = cd['close'].pct_change(5).iloc[-1]
                        result.append({'code': code, 'signal_weight': float(np.clip(ret * 5, -1, 1)) if not np.isnan(ret) else 0.0})
                return pd.DataFrame(result)
        
        engine = EventDrivenBacktestEngine(
            alpha_model=MomentumAlpha(),  # 替换 Alpha 模型
            risk_model=BasicRiskModel(),
            position_sizer=EqualWeightSizer(),
            execution_engine=ExecutionEngine(),
            init_capital=1e6
        )
        
        result = engine.run(self.test_data.iloc[:500])  # 用前500行加速测试
        self.assertFalse(result['equity_curve'].empty)
        print(f"\n[模块替换测试] 动量Alpha回测, 最终权益: {result['equity_curve']['equity'].iloc[-1]:.2f}")
    
    def test_event_bus_subscription(self):
        """测试事件总线发布-订阅机制"""
        bus = EventBus()
        received = []
        
        def handler(event):
            received.append(event.event_type)
        
        bus.subscribe(EventType.MARKET_DATA, handler)
        bus.subscribe(EventType.SIGNAL, handler)
        
        bus.publish(Event(EventType.MARKET_DATA, pd.Timestamp('2024-01-01')))
        bus.publish(Event(EventType.SIGNAL, pd.Timestamp('2024-01-01')))
        bus.publish(Event(EventType.RISK_ALERT, pd.Timestamp('2024-01-01')))
        
        # 只有订阅的事件类型才会被接收
        self.assertEqual(received, [EventType.MARKET_DATA, EventType.SIGNAL])
        self.assertEqual(len(bus._event_log), 3)  # 所有事件都记录在日志中
        
        print(f"\n[事件总线测试] 接收到的事件: {received}")


if __name__ == '__main__':
    unittest.main(verbosity=2)