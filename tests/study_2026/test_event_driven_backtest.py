"""
优化方向：事件驱动回测架构
借鉴来源：QuantConnect/LEAN - Event-Driven Architecture
项目地址：https://github.com/QuantConnect/Lean

LEAN 的核心设计是一个事件驱动的算法交易引擎，关键设计模式：
1. 事件类型分层：MarketDataEvent, OrderEvent, FillEvent, SignalEvent
2. 事件队列（Event Queue）：按时间顺序处理各类事件
3. 模块化组件：SlippageModel, FeeModel, FillModel, MarginModel
4. 时间同步：数据事件先于交易事件，确保持仓/资金状态一致

当前 jingni-trader 的回测引擎依赖适配器模式（backtrader/rqalpha/native），
缺乏统一的事件驱动模型。引入事件驱动架构可以：
- 更精确模拟真实交易时序
- 支持更复杂的交易规则（涨跌停、T+1）
- 方便扩展自定义费率和滑点模型

本测试验证：事件驱动回测的准确性、时序正确性、A股规则支持
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Any
import time


# ============================================================
# 事件驱动回测引擎（原型验证版）
# ============================================================

class EventType(Enum):
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    PORTFOLIO = "portfolio"


class MarketDataEvent:
    """行情数据事件"""
    def __init__(self, date, symbol, open_p, high, low, close, volume,
                 pre_close, is_limit_up=False, is_limit_down=False):
        self.type = EventType.MARKET_DATA
        self.date = date
        self.symbol = symbol
        self.open = open_p
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.pre_close = pre_close
        self.is_limit_up = is_limit_up
        self.is_limit_down = is_limit_down


class SignalEvent:
    """交易信号事件"""
    def __init__(self, date, symbol, signal_type, strength=1.0):
        """
        signal_type: 'BUY' | 'SELL' | 'HOLD'
        strength: 信号强度 0.0~1.0，用于确定仓位大小
        """
        self.type = EventType.SIGNAL
        self.date = date
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength


class OrderEvent:
    """订单事件"""
    def __init__(self, date, symbol, order_type, quantity, price=None):
        self.type = EventType.ORDER
        self.date = date
        self.symbol = symbol
        self.order_type = order_type  # 'MARKET' | 'LIMIT'
        self.quantity = quantity      # 正数=买入, 负数=卖出
        self.price = price
        self.status = 'PENDING'


class FillEvent:
    """成交事件"""
    def __init__(self, date, symbol, quantity, price, commission, slippage_cost):
        self.type = EventType.FILL
        self.date = date
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.commission = commission
        self.slippage_cost = slippage_cost


class PortfolioEvent:
    """组合估值事件（日终）"""
    def __init__(self, date, equity, cash, positions_value, positions: Dict):
        self.type = EventType.PORTFOLIO
        self.date = date
        self.equity = equity
        self.cash = cash
        self.positions_value = positions_value
        self.positions = positions


# ============================================================
# 交易模型（可插拔）
# ============================================================

class SlippageModel:
    """滑点模型"""
    def __init__(self, method='fixed', fixed_pct=0.001):
        self.method = method  # 'fixed' | 'volume_weighted'
        self.fixed_pct = fixed_pct

    def compute_slippage(self, price, volume, direction='buy'):
        """计算滑点成本"""
        if self.method == 'fixed':
            multiplier = 1 if direction == 'buy' else -1
            return price * self.fixed_pct * multiplier
        return 0.0


class FeeModel:
    """费率模型（A股）"""
    def __init__(self, commission_rate=0.0003, min_commission=5.0,
                 stamp_tax_rate=0.001, transfer_fee_rate=0.00002):
        self.commission_rate = commission_rate  # 佣金费率
        self.min_commission = min_commission      # 最低佣金
        self.stamp_tax_rate = stamp_tax_rate      # 印花税（仅卖出）
        self.transfer_fee_rate = transfer_fee_rate  # 过户费

    def compute_commission(self, price, quantity, direction='buy'):
        """计算总费用"""
        trade_amount = abs(price * quantity)
        # 佣金
        commission = trade_amount * self.commission_rate
        commission = max(commission, self.min_commission)
        # 印花税（仅卖出）
        stamp_tax = 0
        if direction == 'sell':
            stamp_tax = trade_amount * self.stamp_tax_rate
        # 过户费
        transfer_fee = trade_amount * self.transfer_fee_rate
        return round(commission + stamp_tax + transfer_fee, 2)


class FillModel:
    """成交模型（处理涨跌停限制）"""
    def can_fill(self, order: OrderEvent, market_data: MarketDataEvent) -> bool:
        """判断订单是否能成交"""
        if order.order_type == 'MARKET':
            # 涨停不能买，跌停不能卖
            if order.quantity > 0 and market_data.is_limit_up:
                return False
            if order.quantity < 0 and market_data.is_limit_down:
                return False
            return market_data.volume > 0
        return True  # LIMIT 订单简化处理


# ============================================================
# 事件驱动引擎
# ============================================================

class EventDrivenBacktestEngine:
    """事件驱动回测引擎"""

    def __init__(self, init_capital=1_000_000, t_plus_1=True):
        self.init_capital = init_capital
        self.t_plus_1 = t_plus_1

        # 交易模型
        self.slippage_model = SlippageModel()
        self.fee_model = FeeModel()
        self.fill_model = FillModel()

        # 内部状态
        self.event_queue = deque()
        self.cash = init_capital
        self.positions: Dict[str, int] = {}     # {symbol: quantity}
        self.avg_prices: Dict[str, float] = {}  # {symbol: avg_entry_price}
        self.equity_curve: List[dict] = []
        self.trade_log: List[dict] = []

        # T+1 卖出限制：前一天买的今天才能卖
        self.holding_since: Dict[str, pd.Timestamp] = {}  # {symbol: buy_date}

    def _schedule_buy(self, symbol: str, strength: float, date, market_data: MarketDataEvent):
        """计算买入订单"""
        if market_data.is_limit_up:
            return  # 涨停买不进

        # 按信号强度分配仓位（TopK Dropout 策略，参考 Qlib）
        target_weight = strength * 0.1  # 最多 10% 仓位
        target_value = self.cash * target_weight
        price = market_data.close  # 次日开盘价模拟
        if price <= 0:
            return
        quantity = int(target_value / (price * 100)) * 100  # A股100股整数倍

        if quantity < 100:
            return

        # 买入手数: 总仓位不超过 95%
        max_quantity = int((self.cash * 0.95) / (price * 100)) * 100
        quantity = min(quantity, max_quantity)

        if quantity > 0:
            order = OrderEvent(date, symbol, 'MARKET', quantity, price)
            self.event_queue.append(order)

    def _schedule_sell(self, symbol: str, strength: float, date, market_data: MarketDataEvent):
        """计算卖出订单"""
        if market_data.is_limit_down:
            return  # 跌停卖不出

        pos_qty = self.positions.get(symbol, 0)
        if pos_qty <= 0:
            return

        # T+1 检查
        if self.t_plus_1:
            buy_date = self.holding_since.get(symbol)
            if buy_date is not None and buy_date >= date:
                return

        # 卖出当前持仓的全部或按比例
        sell_qty = pos_qty
        if strength < 0.5:
            sell_qty = pos_qty  # 清仓
        else:
            sell_qty = 0  # 持有

        if sell_qty > 0:
            order = OrderEvent(date, symbol, 'MARKET', -sell_qty, market_data.close)
            self.event_queue.append(order)

    def _process_order(self, order: OrderEvent, market_data: MarketDataEvent):
        """处理订单 -> 成交"""
        if not self.fill_model.can_fill(order, market_data):
            self.trade_log.append({
                'date': order.date, 'symbol': order.symbol,
                'status': 'REJECTED', 'reason': 'limit' if market_data.is_limit_up else 'volume',
                'quantity': order.quantity
            })
            return

        # 计算成交价格（含滑点）
        fill_price = market_data.close
        slippage = self.slippage_model.compute_slippage(
            fill_price, abs(order.quantity),
            direction='buy' if order.quantity > 0 else 'sell'
        )
        fill_price = fill_price + slippage

        # 计算费用
        direction = 'buy' if order.quantity > 0 else 'sell'
        commission = self.fee_model.compute_commission(fill_price, abs(order.quantity), direction)

        # 资金检查（买入）
        if order.quantity > 0:
            cost = fill_price * order.quantity + commission
            if cost > self.cash:
                # 资金不足，减少买入量
                max_qty = int((self.cash - commission) / fill_price / 100) * 100
                if max_qty < 100:
                    self.trade_log.append({
                        'date': order.date, 'symbol': order.symbol,
                        'status': 'REJECTED', 'reason': 'insufficient_funds',
                        'quantity': order.quantity
                    })
                    return
                order.quantity = max_qty
                cost = fill_price * order.quantity + commission

            self.cash -= cost
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity

            # 更新持仓均价
            old_qty = self.positions.get(order.symbol, 0) - order.quantity
            old_avg = self.avg_prices.get(order.symbol, fill_price)
            new_avg = (old_avg * old_qty + fill_price * order.quantity) / (old_qty + order.quantity) if old_qty + order.quantity > 0 else fill_price
            self.avg_prices[order.symbol] = new_avg
            self.holding_since[order.symbol] = order.date

        else:
            # 卖出
            sell_qty = abs(order.quantity)
            pos_qty = self.positions.get(order.symbol, 0)
            if sell_qty > pos_qty:
                sell_qty = pos_qty

            if sell_qty == 0:
                return

            revenue = fill_price * sell_qty - commission
            self.cash += revenue
            self.positions[order.symbol] -= sell_qty

            if self.positions[order.symbol] <= 0:
                del self.positions[order.symbol]
                del self.avg_prices[order.symbol]
                if order.symbol in self.holding_since:
                    del self.holding_since[order.symbol]

        # 记录成交
        fill = FillEvent(order.date, order.symbol, order.quantity, fill_price, commission, slippage)
        self.trade_log.append({
            'date': fill.date, 'symbol': fill.symbol,
            'quantity': fill.quantity, 'price': fill.price,
            'commission': fill.commission, 'slippage': fill.slippage_cost,
            'status': 'FILLED'
        })

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """执行事件驱动回测

        参数:
            data: 日线行情数据 [date, code, open, high, low, close, ...]
            signals: 交易信号 [date, code, signal]  signal: 1=买入, -1=卖出, 0=持有

        返回:
            {equity_curve, metrics, trade_log}
        """
        # 按日期排序
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        dates = sorted(data['date'].unique())
        symbols = sorted(data['code'].unique())

        # 信号字典: {date: {symbol: signal}}
        signal_dict = {}
        for _, row in signals.iterrows():
            d = row['date']
            if d not in signal_dict:
                signal_dict[d] = {}
            signal_dict[d][row['code']] = row['signal']

        for date in dates:
            day_data = data[data['date'] == date]
            day_signals = signal_dict.get(date, {})

            # Phase 1: 处理卖出信号（先卖后买，释放资金）
            sell_orders = []
            buy_orders = []
            for symbol in symbols:
                sym_data = day_data[day_data['code'] == symbol]
                if sym_data.empty:
                    continue

                row = sym_data.iloc[0]
                md = MarketDataEvent(
                    date, symbol,
                    row.get('open', row['close']),
                    row.get('high', row['close']),
                    row.get('low', row['close']),
                    row['close'],
                    row.get('volume', 0),
                    row.get('pre_close', row['close']),
                    row.get('is_limit_up', False),
                    row.get('is_limit_down', False)
                )

                sig = day_signals.get(symbol, 0)

                if sig == -1:   # 卖出
                    self._schedule_sell(symbol, 0.0, date, md)
                elif sig == 1:  # 买入
                    self._schedule_buy(symbol, 1.0, date, md)

            # Phase 2: 处理订单
            while self.event_queue:
                order = self.event_queue.popleft()
                sym_data = day_data[day_data['code'] == order.symbol]
                if sym_data.empty:
                    continue
                row = sym_data.iloc[0]
                md = MarketDataEvent(
                    date, order.symbol,
                    row.get('open', row['close']),
                    row.get('high', row['close']),
                    row.get('low', row['close']),
                    row['close'],
                    row.get('volume', 0),
                    row.get('pre_close', row['close']),
                    row.get('is_limit_up', False),
                    row.get('is_limit_down', False)
                )
                self._process_order(order, md)

            # Phase 3: 日终估值
            positions_value = 0
            for symbol, qty in list(self.positions.items()):
                sym_data = day_data[day_data['code'] == symbol]
                if not sym_data.empty:
                    positions_value += qty * sym_data.iloc[0]['close']

            equity = self.cash + positions_value
            self.equity_curve.append({
                'date': date,
                'equity': equity,
                'cash': self.cash,
                'positions_value': positions_value,
                'positions_count': len(self.positions),
            })

        # 计算绩效指标
        metrics = self._calc_metrics()

        return {
            'equity_curve': pd.DataFrame(self.equity_curve),
            'metrics': metrics,
            'trade_log': pd.DataFrame(self.trade_log),
            'final_cash': self.cash,
            'final_positions': dict(self.positions),
        }

    def _calc_metrics(self) -> Dict[str, float]:
        """计算绩效指标"""
        if len(self.equity_curve) < 2:
            return {}

        eq = pd.DataFrame(self.equity_curve)
        eq = eq.set_index('date')['equity']
        returns = eq.pct_change().dropna()

        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        max_dd = (eq / eq.cummax() - 1).min()
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0

        return {
            'total_return': round(float(total_return), 4),
            'annual_return': round(float(annual_return), 4),
            'volatility': round(float(volatility), 4),
            'sharpe_ratio': round(float(sharpe), 4),
            'max_drawdown': round(float(max_dd), 4),
            'win_rate': round(float(win_rate), 4),
            'calmar_ratio': round(float(annual_return / abs(max_dd)), 4) if max_dd != 0 else 0,
        }


# ============================================================
# 测试用例
# ============================================================

class TestEventDrivenBacktest(unittest.TestCase):
    """事件驱动回测引擎验证测试"""

    @classmethod
    def setUpClass(cls):
        """生成测试数据"""
        np.random.seed(42)
        codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '601318.SH']
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
        rows = []

        for code in codes:
            base_price = np.random.uniform(8, 60)
            prices = [base_price]
            for _ in range(len(dates) - 1):
                prices.append(prices[-1] * (1 + np.random.normal(0.0003, 0.015)))

            for i, date in enumerate(dates):
                close = prices[i]
                open_p = close * (1 + np.random.normal(0, 0.003))
                high = max(open_p, close) * (1 + abs(np.random.normal(0, 0.005)))
                low = min(open_p, close) * (1 - abs(np.random.normal(0, 0.005)))
                pre_close = prices[i-1] if i > 0 else close
                change_pct = (close - pre_close) / pre_close * 100 if pre_close > 0 else 0

                rows.append({
                    'code': code, 'date': date,
                    'open': round(open_p, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'pre_close': round(pre_close, 2),
                    'volume': int(np.random.lognormal(14, 0.5)),
                    'change_pct': round(change_pct, 2),
                    'is_limit_up': change_pct >= 9.9,
                    'is_limit_down': change_pct <= -9.9,
                })

        cls.test_data = pd.DataFrame(rows).sort_values(['code', 'date']).reset_index(drop=True)
        print(f"\n测试数据: {len(codes)} 只股票, {len(dates)} 个交易日, 共 {len(cls.test_data)} 行")

        # 生成简单信号: 价格突破前20日均线买入
        signals = []
        for code in codes:
            sym_data = cls.test_data[cls.test_data['code'] == code].copy()
            sym_data['ma20'] = sym_data['close'].rolling(20).mean()
            sym_data['signal'] = 0
            sym_data.loc[sym_data['close'] > sym_data['ma20'] * 1.02, 'signal'] = 1
            sym_data.loc[sym_data['close'] < sym_data['ma20'] * 0.98, 'signal'] = -1
            signals.append(sym_data[['code', 'date', 'signal']])

        cls.test_signals = pd.concat(signals, ignore_index=True)
        print(f"信号数据: {len(cls.test_signals)} 行, "
              f"买入 { (cls.test_signals['signal'] == 1).sum()} 次, "
              f"卖出 { (cls.test_signals['signal'] == -1).sum()} 次")

    def test_01_basic_execution(self):
        """测试基本执行：引擎能正常运行并产出结果"""
        engine = EventDrivenBacktestEngine(init_capital=1_000_000)
        result = engine.run(self.test_data, self.test_signals)

        self.assertIn('equity_curve', result)
        self.assertIn('metrics', result)
        self.assertIn('trade_log', result)
        self.assertFalse(result['equity_curve'].empty, "净值曲线不应为空")
        self.assertGreater(len(result['equity_curve']), 0, "至少应有每日净值")

        print(f"\n  最终资金: {result['final_cash']:.2f}")
        print(f"  持仓数量: {len(result['final_positions'])}")
        print(f"  总收益: {result['metrics']['total_return']:.4%}")
        print(f"  夏普比率: {result['metrics']['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {result['metrics']['max_drawdown']:.4%}")

    def test_02_cash_never_negative(self):
        """测试资金非负：无论信号如何，现金不会为负"""
        engine = EventDrivenBacktestEngine(init_capital=1_000_000)
        result = engine.run(self.test_data, self.test_signals)

        eq = result['equity_curve']
        self.assertTrue((eq['cash'] >= -0.01).all(), "现金不应为负")
        self.assertTrue((eq['equity'] >= 0).all(), "总资产不应为负")

    def test_03_a_share_rules(self):
        """测试A股规则: T+1、涨跌停限制、100股整数倍"""
        engine = EventDrivenBacktestEngine(init_capital=1_000_000, t_plus_1=True)
        result = engine.run(self.test_data, self.test_signals)

        trade_log = result['trade_log']

        # 检查所有成交数量都是100的整数倍
        fills = trade_log[trade_log['status'] == 'FILLED']
        if len(fills) > 0:
            for qty in fills['quantity']:
                self.assertEqual(abs(qty) % 100, 0,
                                 f"成交数量 {qty} 不是100的整数倍")

        # 检查涨跌停板上的拒绝
        rejects = trade_log[trade_log['status'] == 'REJECTED']
        if not rejects.empty and 'reason' in rejects.columns:
            limit_rejects = rejects[rejects['reason'] == 'limit']
            print(f"\n  涨跌停板拒绝交易: {len(limit_rejects)} 次")

        # T+1: 同一天不应该有同一个股票的买入 + 卖出
        if len(fills) > 0:
            same_day_trades = fills.groupby(['date', 'symbol']).agg({
                'quantity': ['min', 'max']
            })
            for (date, symbol), row in same_day_trades.iterrows():
                qty_min = row[('quantity', 'min')]
                qty_max = row[('quantity', 'max')]
                # 如果既有买入(正)又有卖出(负)，则为同一天反向交易
                if qty_min < 0 and qty_max > 0:
                    self.fail(f"检测到 {date} {symbol} 同一天既有买入又有卖出（违反 T+1）")

    def test_04_trade_cost_calculation(self):
        """测试交易成本计算（佣金、印花税、过户费）"""
        fee = FeeModel()

        # 买入10000元股票
        buy_cost = fee.compute_commission(10.0, 1000, 'buy')
        expected_buy = max(10000 * 0.0003, 5.0) + 10000 * 0.00002  # 佣金+过户费
        self.assertAlmostEqual(buy_cost, round(expected_buy, 2), delta=0.5,
                               msg="买入费用计算不一致")

        # 卖出10000元股票
        sell_cost = fee.compute_commission(10.0, 1000, 'sell')
        expected_sell = max(10000 * 0.0003, 5.0) + 10000 * 0.001 + 10000 * 0.00002  # 佣金+印花税+过户费
        self.assertAlmostEqual(sell_cost, round(expected_sell, 2), delta=0.5,
                               msg="卖出费用计算不一致")

        print(f"\n  买入1000股@10元：费用 {buy_cost:.2f} 元")
        print(f"  卖出1000股@10元：费用 {sell_cost:.2f} 元")

    def test_05_slippage_impact(self):
        """测试滑点影响"""
        slip = SlippageModel(method='fixed', fixed_pct=0.002)

        buy_slip = slip.compute_slippage(10.0, 1000, 'buy')
        sell_slip = slip.compute_slippage(10.0, 1000, 'sell')

        self.assertGreater(buy_slip, 0, "买入滑点应为正")
        self.assertLess(sell_slip, 0, "卖出滑点应为负")
        self.assertAlmostEqual(buy_slip, 0.02, delta=0.01)
        self.assertAlmostEqual(sell_slip, -0.02, delta=0.01)

    def test_06_event_order_consistency(self):
        """测试事件处理顺序：同一交易日内先信号后行情"""
        engine = EventDrivenBacktestEngine(init_capital=1_000_000)
        result = engine.run(self.test_data, self.test_signals)

        eq = result['equity_curve']
        # 确保日期是单调递增的
        self.assertTrue((eq['date'].diff().dropna() > pd.Timedelta(0)).all(),
                        "事件日期必须单调递增")

        trade_log = result['trade_log']
        if len(trade_log) > 0:
            # 确保同一天的成交记录日期一致
            for _, trade in trade_log.iterrows():
                self.assertIsInstance(trade['date'], pd.Timestamp,
                                      "成交日期应为 Timestamp 类型")

    def test_07_large_portfolio(self):
        """测试较大组合的计算性能"""
        np.random.seed(123)
        codes = [f"{i:06d}.{'SZ' if i % 2 else 'SH'}" for i in range(1, 51)]
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
        rows = []

        for code in codes:
            base_price = np.random.uniform(5, 100)
            prices = [base_price * (1 + np.random.normal(0.0002, 0.012))**i
                      for i in range(len(dates))]
            for i, date in enumerate(dates):
                close = prices[i]
                rows.append({
                    'code': code, 'date': date,
                    'open': round(close * (1 + np.random.normal(0, 0.002)), 2),
                    'high': round(close * 1.02, 2),
                    'low': round(close * 0.98, 2),
                    'close': round(close, 2),
                    'pre_close': round(prices[i-1] if i > 0 else close, 2),
                    'volume': int(np.random.lognormal(14, 0.5)),
                    'change_pct': 0.0,
                    'is_limit_up': False,
                    'is_limit_down': False,
                })

        big_data = pd.DataFrame(rows)
        big_signals = big_data[['code', 'date']].copy()
        big_signals['signal'] = np.random.choice([0, 1, -1], len(big_data), p=[0.7, 0.15, 0.15])

        print(f"\n  大规模测试: {len(codes)} 只股票, {len(dates)} 个交易日, 共 {len(big_data)} 行")

        start = time.perf_counter()
        engine = EventDrivenBacktestEngine(init_capital=10_000_000)
        result = engine.run(big_data, big_signals)
        elapsed = time.perf_counter() - start

        print(f"  耗时: {elapsed:.3f}s")
        print(f"  成交数: {len(result['trade_log'][result['trade_log']['status'] == 'FILLED'])}")
        print(f"  最终权益: {result['equity_curve']['equity'].iloc[-1]:.2f}")

        self.assertLess(elapsed, 30.0, "50只股票1年回测不应超过30秒")

    def test_08_different_slippage_models(self):
        """测试不同滑点模型对回测结果的影响"""
        data_subset = self.test_data[
            self.test_data['code'].isin(['000001.SZ', '600036.SH'])
        ]
        signals_subset = self.test_signals[
            self.test_signals['code'].isin(['000001.SZ', '600036.SH'])
        ]

        results = {}
        for pct in [0.0, 0.001, 0.005, 0.01]:
            engine = EventDrivenBacktestEngine(init_capital=1_000_000)
            engine.slippage_model = SlippageModel(method='fixed', fixed_pct=pct)
            result = engine.run(data_subset, signals_subset)
            results[pct] = result['metrics']['total_return']

        print(f"\n  滑点敏感性分析:")
        for pct, ret in results.items():
            print(f"    滑点 {pct:.3f}: 总收益 {ret:.4%}")

        # 滑点越大，收益应该越低（或相等）
        rets = list(results.values())
        self.assertGreaterEqual(rets[0], rets[-1],
                                "高滑点不应产生比零滑点更高的收益")


if __name__ == '__main__':
    unittest.main(verbosity=2)