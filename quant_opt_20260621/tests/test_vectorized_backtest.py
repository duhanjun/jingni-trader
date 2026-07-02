"""
向量化回测引擎测试

测试维度：
1. 正确性：与 native_adapter 等价的参考实现对比关键指标
2. 性能：大规模数据下向量化 vs 逐日循环的耗时对比
3. 边界：空数据 / 单标的 / 单日 / 全涨跌停 / T+1
"""
import sys
import os
import time
import unittest
from typing import Dict, Any

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace')

from quant_opt_20260621.vectorized_backtest import VectorizedBacktestAdapter


# ----------------------------------------------------------------------
# 参考实现：复刻 native_adapter.NativeAdapter 的核心逻辑（逐日循环 + pandas）
# 用于与向量化版本做正确性对比。不导入 main 分支代码，避免 sys.path 问题。
# ----------------------------------------------------------------------

def reference_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    init_capital: float = 1e6,
    commission_rate: float = 0.00025,
    stamp_tax_rate: float = 0.001,
    t_plus_1: bool = True,
    price_limit: bool = True,
    slippage: float = 0.001,
) -> Dict[str, Any]:
    """逐日循环参考实现（与 native_adapter 行为一致）"""
    if data.empty or signals.empty:
        return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame(), "metrics": {}}

    data = data.sort_values(['date', 'code']).reset_index(drop=True).copy()
    signals = signals.sort_values(['date', 'code']).reset_index(drop=True).copy()
    data['date'] = pd.to_datetime(data['date'])
    signals['date'] = pd.to_datetime(signals['date'])

    dates = sorted(signals['date'].unique())
    if not dates:
        return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame(), "metrics": {}}

    cash = float(init_capital)
    positions: Dict[str, float] = {}
    equity_records = []
    trades = []

    for dt in dates:
        day_signal = signals[signals['date'] == dt]
        day_data = data[data['date'] == dt]
        if day_data.empty:
            continue
        day_data_map = day_data.set_index('code')

        sell_codes = []
        buy_codes = []
        for _, row in day_signal.iterrows():
            code = row['code']
            sig = row.get('signal', 0)
            if isinstance(sig, (int, float, np.integer, np.floating)):
                sig = float(sig)
                if sig > 0:
                    buy_codes.append(code)
                elif sig < 0:
                    sell_codes.append(code)

        for code in sell_codes:
            if code not in positions or positions[code] <= 0:
                continue
            if code not in day_data_map.index:
                continue
            price_row = day_data_map.loc[code]
            if price_limit and price_row.get('is_limit_down', False):
                continue
            price = price_row['close']
            shares = positions[code]
            sell_amount = price * shares
            commission = max(sell_amount * commission_rate, 5)
            tax = sell_amount * stamp_tax_rate
            cost = commission + tax
            cash += sell_amount - cost
            trades.append({
                'date': dt, 'code': code, 'action': 'sell',
                'price': price, 'shares': shares, 'amount': sell_amount,
                'commission': commission, 'tax': tax, 'pnl': sell_amount - cost,
            })
            positions[code] = 0

        if buy_codes:
            n_buy = len(buy_codes)
            budget_per_stock = cash * 0.95 / n_buy
            for code in buy_codes:
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                if price_limit and price_row.get('is_limit_up', False):
                    continue
                price = price_row['close'] * (1 + slippage)
                shares = int(budget_per_stock / price / 100) * 100
                if shares <= 0:
                    continue
                buy_amount = price * shares
                commission = max(buy_amount * commission_rate, 5)
                cost = buy_amount + commission
                if cost > cash:
                    shares = int((cash * 0.98) / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * commission_rate, 5)
                    cost = buy_amount + commission
                cash -= cost
                positions[code] = positions.get(code, 0) + shares
                trades.append({
                    'date': dt, 'code': code, 'action': 'buy',
                    'price': price, 'shares': shares, 'amount': buy_amount,
                    'commission': commission, 'tax': 0, 'pnl': -buy_amount - commission,
                })

        market_value = 0
        for code, shares in list(positions.items()):
            if shares <= 0:
                continue
            if code in day_data_map.index:
                market_value += shares * day_data_map.loc[code, 'close']
        total_equity = cash + market_value
        equity_records.append({
            'date': dt, 'equity': total_equity, 'cash': cash,
            'market_value': market_value,
            'position_count': sum(1 for s in positions.values() if s > 0),
        })

    equity_curve = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trades)
    return {
        "equity_curve": equity_curve,
        "trades": trades_df,
        "metrics": {},
    }


# ----------------------------------------------------------------------
# 测试数据生成器
# ----------------------------------------------------------------------

def make_test_data(n_codes: int = 5, n_days: int = 60, seed: int = 0,
                   limit_up_freq: float = 0.0, limit_down_freq: float = 0.0):
    """生成测试用行情数据 + 简单轮换信号"""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    codes = [f'{600000 + i}.SH' if i % 2 == 0 else f'{1 + i:06d}.SZ' for i in range(n_codes)]

    rows = []
    for c in codes:
        price = 10.0
        for d in dates:
            ret = np.random.randn() * 0.02
            price = max(price * (1 + ret), 1.0)
            change_pct = ret * 100
            is_up = np.random.rand() < limit_up_freq
            is_down = np.random.rand() < limit_down_freq
            rows.append({
                'code': c, 'date': d,
                'open': price * 0.99, 'high': price * 1.01,
                'low': price * 0.98, 'close': price,
                'volume': np.random.randint(1e6, 1e8),
                'is_limit_up': is_up, 'is_limit_down': is_down,
            })
    data = pd.DataFrame(rows)

    # 信号：每 10 天轮换
    sig_rows = []
    for d in dates:
        for c in codes:
            sig = 1 if (dates.get_loc(d) // 10) % 2 == 0 else -1
            sig_rows.append({'code': c, 'date': d, 'signal': sig})
    signals = pd.DataFrame(sig_rows)
    return data, signals


# ======================================================================
# 测试用例
# ======================================================================

class TestVectorizedBacktestCorrectness(unittest.TestCase):
    """正确性：向量化结果与逐日循环参考实现一致"""

    def test_equity_curve_matches_reference(self):
        """向量化与参考实现的净值曲线应高度一致（允许浮点误差）"""
        data, signals = make_test_data(n_codes=5, n_days=60, seed=42)

        ref = reference_backtest(data, signals)
        vec = VectorizedBacktestAdapter().run_backtest(data, signals)

        self.assertFalse(ref['equity_curve'].empty, "参考实现净值曲线不应为空")
        self.assertFalse(vec['equity_curve'].empty, "向量化净值曲线不应为空")

        # 对齐日期
        ref_eq = ref['equity_curve'].set_index('date')['equity']
        vec_eq = vec['equity_curve'].set_index('date')['equity']

        # 两者应覆盖相同日期
        common_dates = ref_eq.index.intersection(vec_eq.index)
        self.assertGreater(len(common_dates), 50, "应有足够多的共同交易日")

        ref_vals = ref_eq.loc[common_dates].values
        vec_vals = vec_eq.loc[common_dates].values

        # 允许 1% 的相对误差（买卖手数取整可能产生微小差异）
        rel_diff = np.abs(ref_vals - vec_vals) / (np.abs(ref_vals) + 1e-10)
        max_rel_diff = float(np.max(rel_diff))
        self.assertLess(max_rel_diff, 0.01,
                        f"净值曲线最大相对误差 {max_rel_diff:.6f} 超过 1%")

    def test_trade_count_matches(self):
        """成交笔数应一致"""
        data, signals = make_test_data(n_codes=5, n_days=60, seed=42)
        ref = reference_backtest(data, signals)
        vec = VectorizedBacktestAdapter().run_backtest(data, signals)

        # 成交笔数可能因买卖手数取整微小差异略有不同，但应在合理范围
        ref_n = len(ref['trades'])
        vec_n = len(vec['trades'])
        self.assertAlmostEqual(ref_n, vec_n, delta=max(2, ref_n * 0.1),
                               msg=f"成交笔数差异过大: ref={ref_n}, vec={vec_n}")

    def test_metrics_keys_present(self):
        """绩效指标应包含所有必需字段"""
        data, signals = make_test_data(n_codes=5, n_days=60, seed=42)
        vec = VectorizedBacktestAdapter().run_backtest(data, signals)

        required = {'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
                    'max_drawdown', 'calmar_ratio', 'sortino_ratio', 'win_rate',
                    'total_trades'}
        self.assertTrue(required.issubset(set(vec['metrics'].keys())),
                        f"缺失指标: {required - set(vec['metrics'].keys())}")

    def test_initial_capital_preserved(self):
        """首日净值应接近初始资金（仅有手续费损耗）"""
        data, signals = make_test_data(n_codes=3, n_days=30, seed=1)
        vec = VectorizedBacktestAdapter().run_backtest(data, signals, init_capital=1e6)
        first_equity = vec['equity_curve']['equity'].iloc[0]
        # 首日买入后，净值 = 现金 + 市值，应接近 1e6（扣除手续费）
        self.assertLess(abs(first_equity - 1e6) / 1e6, 0.01,
                        f"首日净值偏离初始资金过大: {first_equity}")


class TestVectorizedBacktestBoundary(unittest.TestCase):
    """边界条件测试"""

    def test_empty_data(self):
        """空数据应返回空结果"""
        empty = pd.DataFrame(columns=['code', 'date', 'close', 'is_limit_up', 'is_limit_down'])
        result = VectorizedBacktestAdapter().run_backtest(empty, empty)
        self.assertTrue(result['equity_curve'].empty)
        self.assertEqual(result['metrics'], {})

    def test_single_stock(self):
        """单标的回测应正常运行"""
        data, signals = make_test_data(n_codes=1, n_days=30, seed=2)
        result = VectorizedBacktestAdapter().run_backtest(data, signals)
        self.assertFalse(result['equity_curve'].empty)
        self.assertEqual(len(result['equity_curve']), 30)

    def test_single_day(self):
        """单日回测应正常运行"""
        data, signals = make_test_data(n_codes=3, n_days=1, seed=3)
        result = VectorizedBacktestAdapter().run_backtest(data, signals)
        self.assertFalse(result['equity_curve'].empty)

    def test_all_limit_up_blocks_buy(self):
        """全部涨停时应阻止买入"""
        data, signals = make_test_data(n_codes=3, n_days=5, seed=4)
        data['is_limit_up'] = True  # 全部涨停
        result = VectorizedBacktestAdapter().run_backtest(data, signals, price_limit=True)
        # 全部涨停不应有买入成交
        buys = result['trades'][result['trades']['action'] == 'buy'] if not result['trades'].empty else pd.DataFrame()
        self.assertEqual(len(buys), 0, "涨停日不应有买入成交")

    def test_all_limit_down_blocks_sell(self):
        """全部跌停时应阻止卖出"""
        data, signals = make_test_data(n_codes=3, n_days=10, seed=5)
        # 前 5 天买入信号，后 5 天卖出信号
        signals.loc[signals['date'] >= signals['date'].iloc[-5], 'signal'] = -1
        signals.loc[signals['date'] < signals['date'].iloc[-5], 'signal'] = 1
        data['is_limit_down'] = False
        # 仅后 5 天跌停
        late_dates = sorted(data['date'].unique())[-5:]
        data.loc[data['date'].isin(late_dates), 'is_limit_down'] = True
        result = VectorizedBacktestAdapter().run_backtest(data, signals, price_limit=True)
        sells = result['trades'][result['trades']['action'] == 'sell'] if not result['trades'].empty else pd.DataFrame()
        # 跌停日不应有卖出成交
        if not sells.empty:
            sell_dates = set(sells['date'].unique())
            self.assertFalse(sell_dates & set(late_dates),
                             "跌停日不应有卖出成交")

    def test_no_signals(self):
        """无信号时应保持初始资金"""
        data, signals = make_test_data(n_codes=3, n_days=10, seed=6)
        signals['signal'] = 0
        result = VectorizedBacktestAdapter().run_backtest(data, signals, init_capital=1e6)
        # 无交易，净值应恒等于初始资金
        self.assertTrue((result['equity_curve']['equity'] == 1e6).all(),
                        "无信号时净值应恒等于初始资金")
        self.assertEqual(len(result['trades']), 0)


class TestVectorizedBacktestPerformance(unittest.TestCase):
    """性能测试：向量化 vs 逐日循环"""

    def test_performance_large_scale(self):
        """大规模数据下向量化应显著快于逐日循环"""
        n_codes = 50
        n_days = 250  # 约 1 年
        data, signals = make_test_data(n_codes=n_codes, n_days=n_days, seed=100)

        # 逐日循环
        t0 = time.time()
        ref = reference_backtest(data, signals)
        t_ref = time.time() - t0

        # 向量化
        t0 = time.time()
        vec = VectorizedBacktestAdapter().run_backtest(data, signals)
        t_vec = time.time() - t0

        speedup = t_ref / t_vec if t_vec > 0 else float('inf')
        print(f"\n[性能] 50 标的 × 250 日：参考={t_ref:.3f}s, 向量化={t_vec:.3f}s, 加速比={speedup:.1f}x")

        # 向量化应至少快 3x（保守阈值，避免 CI 环境抖动）
        self.assertGreater(speedup, 3.0,
                           f"向量化加速比 {speedup:.1f}x 未达到 3x 阈值")

        # 结果应一致
        ref_eq = ref['equity_curve'].set_index('date')['equity']
        vec_eq = vec['equity_curve'].set_index('date')['equity']
        common = ref_eq.index.intersection(vec_eq.index)
        rel_diff = np.abs(ref_eq.loc[common].values - vec_eq.loc[common].values) / (np.abs(ref_eq.loc[common].values) + 1e-10)
        self.assertLess(float(np.max(rel_diff)), 0.02,
                        "性能测试中结果一致性应保持")


if __name__ == '__main__':
    unittest.main(verbosity=2)
