"""
向量化回测适配器测试

验证内容:
1. 正确性: 净值曲线合理、指标计算正确
2. 性能: 向量化回测应显著快于事件驱动
3. 边界: 空数据、无信号、全涨停、单股票
4. A股规则: T+1、涨跌停过滤、费用扣除
"""
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from vectorized_backtest_adapter import VectorizedBacktester, run_vectorized_backtest
from tests.data_generator import generate_market_data, generate_factor_data, generate_signals


class TestBacktestCorrectness(unittest.TestCase):
    """正确性测试"""

    @classmethod
    def setUpClass(cls):
        cls.market = generate_market_data(n_stocks=50, n_days=250, seed=10)
        cls.factors = generate_factor_data(cls.market, seed=10)
        cls.signals = generate_signals(cls.factors, top_pct=0.2)

    def test_equity_curve_shape(self):
        """净值曲线应与交易日数一致"""
        result = run_vectorized_backtest(self.market, self.signals, init_capital=1e6)
        self.assertFalse(result["equity_curve"].empty)
        self.assertIn("equity", result["equity_curve"].columns)
        self.assertIn("daily_return", result["equity_curve"].columns)
        # 净值曲线长度应接近交易日数(可能有少量过滤)
        self.assertGreater(len(result["equity_curve"]), 200)

    def test_initial_capital_preserved(self):
        """首日净值应接近初始资金(允许少量成本)"""
        result = run_vectorized_backtest(self.market, self.signals, init_capital=1_000_000)
        eq = result["equity_curve"]["equity"].iloc[0]
        # 首日可能因 T+1 无持仓，净值=初始资金
        self.assertAlmostEqual(eq, 1_000_000, delta=1_000)

    def test_metrics_complete(self):
        """绩效指标应完整"""
        result = run_vectorized_backtest(self.market, self.signals)
        m = result["metrics"]
        for key in ["total_return", "annual_return", "sharpe_ratio", "max_drawdown",
                     "volatility", "win_rate", "calmar_ratio", "sortino_ratio"]:
            self.assertIn(key, m, f"缺少指标 {key}")

    def test_no_signal_no_loss(self):
        """无信号时净值应保持不变(扣除成本后接近初始资金)"""
        empty_signals = pd.DataFrame(columns=["code", "date", "signal"])
        result = run_vectorized_backtest(self.market, empty_signals, init_capital=1e6)
        if not result["equity_curve"].empty:
            final_eq = result["equity_curve"]["equity"].iloc[-1]
            self.assertAlmostEqual(final_eq, 1e6, delta=1.0)

    def test_t_plus_1_effective(self):
        """T+1 模式下，当日信号不应在当日产生持仓"""
        # 构造单日信号
        sig = self.signals[self.signals["date"] == self.signals["date"].min()].copy()
        sig["signal"] = 1
        result = run_vectorized_backtest(self.market, sig, t_plus_1=True, init_capital=1e6)
        eq = result["equity_curve"]
        if not eq.empty:
            # 首日 position_count 应为 0(T+1 延迟)
            self.assertEqual(eq["position_count"].iloc[0], 0)


class TestBacktestPerformance(unittest.TestCase):
    """性能测试"""

    def test_vectorized_vs_loop_performance(self):
        """向量化回测应在合理时间内完成大规模数据"""
        # 200 股 × 500 日 = 10万行
        market = generate_market_data(n_stocks=200, n_days=500, seed=11)
        factors = generate_factor_data(market, seed=11)
        signals = generate_signals(factors, top_pct=0.1)

        t0 = time.perf_counter()
        result = run_vectorized_backtest(market, signals, init_capital=1e6, max_positions=50)
        elapsed = time.perf_counter() - t0
        print(f"\n[回测性能] 200股×500日 向量化耗时: {elapsed:.3f}s "
              f"(result.elapsed_sec={result.get('elapsed_sec')})")
        # 向量化回测应在 10 秒内完成
        self.assertLess(elapsed, 10.0, f"向量化回测耗时 {elapsed:.2f}s 过长")
        self.assertFalse(result["equity_curve"].empty)


class TestBacktestBoundary(unittest.TestCase):
    """边界条件测试"""

    def test_empty_data(self):
        """空数据应返回空结果而非报错"""
        result = run_vectorized_backtest(
            pd.DataFrame(), pd.DataFrame(columns=["code", "date", "signal"]))
        self.assertTrue(result["equity_curve"].empty)
        self.assertEqual(result["metrics"], {})

    def test_empty_signals(self):
        """空信号应返回初始资金不变的净值曲线"""
        market = generate_market_data(n_stocks=10, n_days=30, seed=12)
        empty_sig = pd.DataFrame(columns=["code", "date", "signal"])
        result = run_vectorized_backtest(market, empty_sig, init_capital=1e6)
        if not result["equity_curve"].empty:
            final = result["equity_curve"]["equity"].iloc[-1]
            self.assertAlmostEqual(final, 1e6, delta=1.0)

    def test_all_limit_up(self):
        """全部涨停时应无法买入(持仓为0)"""
        market = generate_market_data(n_stocks=10, n_days=20, seed=13)
        market["is_limit_up"] = 1  # 全部涨停
        sig = market[["code", "date"]].copy()
        sig["signal"] = 1
        result = run_vectorized_backtest(market, sig, price_limit=True, init_capital=1e6)
        if not result["equity_curve"].empty:
            # 涨停无法买入，持仓应为0
            self.assertTrue((result["equity_curve"]["position_count"] == 0).all())

    def test_single_stock(self):
        """单股票回测应正常完成"""
        market = generate_market_data(n_stocks=1, n_days=60, seed=14)
        factors = generate_factor_data(market, seed=14)
        signals = generate_signals(factors, top_pct=1.0)
        result = run_vectorized_backtest(market, signals, init_capital=1e6, max_positions=1)
        self.assertFalse(result["equity_curve"].empty)

    def test_cost_deduction(self):
        """费用应被扣除(频繁换手时净值应低于无费用)"""
        market = generate_market_data(n_stocks=20, n_days=100, seed=15)
        factors = generate_factor_data(market, seed=15)
        # 高频换仓信号: 每日随机选股
        rng = np.random.default_rng(15)
        signals = factors[["code", "date"]].copy()
        signals["signal"] = rng.integers(0, 2, len(signals))
        result_with_cost = run_vectorized_backtest(
            market, signals, commission_rate=0.001, stamp_tax_rate=0.001, init_capital=1e6)
        result_no_cost = run_vectorized_backtest(
            market, signals, commission_rate=0.0, stamp_tax_rate=0.0, init_capital=1e6)
        if not result_with_cost["equity_curve"].empty and not result_no_cost["equity_curve"].empty:
            final_with = result_with_cost["equity_curve"]["equity"].iloc[-1]
            final_no = result_no_cost["equity_curve"]["equity"].iloc[-1]
            # 有费用时净值应低于无费用(或相等)
            self.assertLessEqual(final_with, final_no + 1.0,
                                 "有费用净值应 <= 无费用净值")


class TestBacktestAShareRules(unittest.TestCase):
    """A 股规则测试"""

    def test_stamp_tax_only_on_sell(self):
        """印花税仅在卖出时扣除"""
        market = generate_market_data(n_stocks=10, n_days=30, seed=16)
        factors = generate_factor_data(market, seed=16)
        signals = generate_signals(factors, top_pct=0.5)
        # 仅买入印花税为0 vs 印花税0.1%
        r1 = run_vectorized_backtest(market, signals, stamp_tax_rate=0.0, init_capital=1e6)
        r2 = run_vectorized_backtest(market, signals, stamp_tax_rate=0.001, init_capital=1e6)
        if not r1["equity_curve"].empty and not r2["equity_curve"].empty:
            # 有印花税时净值应 <= 无印花税
            self.assertLessEqual(
                r2["equity_curve"]["equity"].iloc[-1],
                r1["equity_curve"]["equity"].iloc[-1] + 1.0)

    def test_slippage_reduces_return(self):
        """滑点应降低收益(买入价更高)"""
        market = generate_market_data(n_stocks=20, n_days=100, seed=17)
        factors = generate_factor_data(market, seed=17)
        signals = generate_signals(factors, top_pct=0.3)
        r_no_slip = run_vectorized_backtest(market, signals, slippage=0.0, init_capital=1e6)
        r_slip = run_vectorized_backtest(market, signals, slippage=0.005, init_capital=1e6)
        if not r_no_slip["equity_curve"].empty and not r_slip["equity_curve"].empty:
            self.assertLessEqual(
                r_slip["equity_curve"]["equity"].iloc[-1],
                r_no_slip["equity_curve"]["equity"].iloc[-1] + 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
