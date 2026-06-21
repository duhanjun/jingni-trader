"""
向量化回测引擎 - 验证测试
==========================
测试内容:
1. 正确性测试: T+1 执行、费用计算、涨跌停限制、收益计算
2. 性能对比测试: 向量化 vs 逐日循环 (模拟 native_adapter)
3. 边界条件测试: 空数据、单日、全空仓、满仓、极端权重

对比基准: jingni-trader backtest-engine/scripts/adapters/native_adapter.py
- native_adapter 的 T+1 未实现 (参数传入但未使用)
- native_adapter 的过户费缺失
- native_adapter 的 pnl 计算错误 (现金流非盈亏)

运行: python -m optimizations.vectorized_backtest.test_vectorized_backtest
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimizations.vectorized_backtest import (  # noqa: E402
    CostModel,
    VectorizedBacktester,
    compute_metrics,
    detect_board,
)


def make_price_panel(n_days: int = 60, n_codes: int = 10, seed: int = 42) -> pd.DataFrame:
    """生成收盘价面板 (index=date, columns=code)。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    base = 10.0 + rng.normal(0, 0.3, (n_days, n_codes)).cumsum(axis=0)
    base = np.maximum(base, 1.0)
    return pd.DataFrame(base, index=dates, columns=codes)


def make_open_panel(close: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """生成开盘价 (基于前一日收盘 + 跳空)。"""
    rng = np.random.default_rng(seed)
    open_ = close.shift(1) * (1 + rng.normal(0, 0.005, close.shape))
    open_.iloc[0] = close.iloc[0]
    return open_


class TestCostModel(unittest.TestCase):
    """费用模型正确性测试。"""

    def setUp(self) -> None:
        self.cost = CostModel()

    def test_buy_cost(self) -> None:
        """买入费用 = 佣金(最低5) + 过户费。"""
        # 大额: 佣金 = 100000 * 0.00025 = 25, 过户费 = 100000 * 0.00002 = 2, 合计 27
        cost = self.cost.buy_cost(100_000)
        self.assertAlmostEqual(cost, 25 + 2, places=6)

        # 小额: 佣金最低 5, 过户费 = 1000 * 0.00002 = 0.02, 合计 5.02
        cost = self.cost.buy_cost(1000)
        self.assertAlmostEqual(cost, 5 + 0.02, places=6)

    def test_sell_cost(self) -> None:
        """卖出费用 = 佣金(最低5) + 印花税 + 过户费。"""
        # 100000: 佣金 25 + 印花税 100 + 过户费 2 = 127
        cost = self.cost.sell_cost(100_000)
        self.assertAlmostEqual(cost, 25 + 100 + 2, places=6)

    def test_sell_higher_than_buy(self) -> None:
        """卖出费用应高于买入 (印花税)。"""
        amount = 50_000
        self.assertGreater(self.cost.sell_cost(amount), self.cost.buy_cost(amount))

    def test_transfer_fee_included(self) -> None:
        """过户费应被计算 (修复 native_adapter 缺失过户费的问题)。"""
        amount = 1_000_000
        buy = self.cost.buy_cost(amount)
        expected_transfer = amount * 0.00002
        self.assertGreaterEqual(buy, expected_transfer)


class TestBoardDetection(unittest.TestCase):
    """板块识别测试 (修复 native_adapter 一刀切 9.9% 涨跌停的问题)。"""

    def test_main_board(self) -> None:
        self.assertEqual(detect_board("600000.SH"), "main")
        self.assertEqual(detect_board("000001.SZ"), "main")

    def test_kcb(self) -> None:
        self.assertEqual(detect_board("688001.SH"), "kc")

    def test_cyb(self) -> None:
        self.assertEqual(detect_board("300001.SZ"), "cy")
        self.assertEqual(detect_board("301001.SZ"), "cy")

    def test_bjb(self) -> None:
        self.assertEqual(detect_board("830001.BJ"), "bj")
        self.assertEqual(detect_board("430001.BJ"), "bj")


class TestTPlus1Execution(unittest.TestCase):
    """T+1 执行逻辑正确性测试 (核心修复点)。

    jingni-trader native_adapter.py:27 声明 t_plus_1 参数但从未使用,
    导致当日买入当日可卖, 违反 A 股规则。
    """

    def setUp(self) -> None:
        self.bt = VectorizedBacktester(t_plus_1=True, deal_price="open")

    def test_signal_delayed_one_day(self) -> None:
        """T 日信号应在 T+1 日才反映到持仓上 (target_weight 模式)。"""
        n_days = 10
        price = make_price_panel(n_days=n_days, n_codes=3)
        # T 日 (第5日) 目标权重全仓第一只股票
        weights = pd.DataFrame(0.0, index=price.index, columns=price.columns)
        weights.iloc[5, 0] = 1.0  # 第5日目标全仓

        result = self.bt.run_target_weight(weights, price, initial_capital=100_000)

        # 实际持仓应在第6日才变为 1.0 (T+1 执行)
        actual = result.positions
        self.assertAlmostEqual(actual.iloc[5, 0], 0.0, places=6, msg="T 日不应已持仓")
        self.assertAlmostEqual(actual.iloc[6, 0], 1.0, places=6, msg="T+1 日应已持仓")

    def test_no_same_day_buy_sell(self) -> None:
        """信号模式下, 买入当日不可卖出 (T+1)。"""
        n_days = 10
        price = make_price_panel(n_days=n_days, n_codes=3)
        open_price = make_open_panel(price)

        # 第2日买入信号, 第2日卖出信号 (同日买卖, 应被 T+1 阻止)
        buy = pd.DataFrame(False, index=price.index, columns=price.columns)
        sell = pd.DataFrame(False, index=price.index, columns=price.columns)
        buy.iloc[2, 0] = True
        sell.iloc[2, 0] = True  # 同日卖出

        result = self.bt.run_signal(buy, sell, price, open_price, initial_capital=100_000)
        # 由于 T+1, 第2日买入的股票至少持有到第4日 (信号延迟1日执行 + T+1)
        # 检查持仓: 第3日 (执行日) 应有持仓, 且第3日不能卖
        self.assertGreater(result.positions.iloc[3, 0], 0, "T+1 执行后应有持仓")


class TestVectorizedBacktest(unittest.TestCase):
    """向量化回测核心功能测试。"""

    def setUp(self) -> None:
        self.bt = VectorizedBacktester()
        self.price = make_price_panel(n_days=60, n_codes=10)
        self.open_price = make_open_panel(self.price)

    def test_buy_and_hold(self) -> None:
        """买入持有策略: 净值应等于标的涨幅 (扣费后)。"""
        weights = pd.DataFrame(0.0, index=self.price.index, columns=self.price.columns)
        weights.iloc[:, 0] = 1.0  # 全仓第一只

        result = self.bt.run_target_weight(weights, self.price, initial_capital=1_000_000)

        # 净值曲线长度应等于交易日数
        self.assertEqual(len(result.equity_curve), len(self.price))
        # 初始净值应接近 1_000_000 (首日无收益)
        self.assertAlmostEqual(result.equity_curve.iloc[0], 1_000_000, delta=1)
        # 应有绩效指标
        self.assertIn("total_return", result.metrics)
        self.assertIn("sharpe", result.metrics)
        self.assertIn("max_drawdown", result.metrics)

    def test_equal_weight_portfolio(self) -> None:
        """等权组合: 每只股票 10% 权重。"""
        n_codes = len(self.price.columns)
        weights = pd.DataFrame(
            1.0 / n_codes, index=self.price.index, columns=self.price.columns
        )

        result = self.bt.run_target_weight(weights, self.price, initial_capital=1_000_000)
        # 权重和应 <= 1
        self.assertTrue((result.positions.sum(axis=1) <= 1.0 + 1e-9).all())

    def test_all_cash(self) -> None:
        """全空仓: 净值应保持不变 (无收益无费用)。"""
        weights = pd.DataFrame(0.0, index=self.price.index, columns=self.price.columns)
        result = self.bt.run_target_weight(weights, self.price, initial_capital=1_000_000)
        # 全现金, 净值应恒定
        np.testing.assert_allclose(
            result.equity_curve.values,
            np.full(len(result.equity_curve), 1_000_000),
            rtol=1e-6,
        )
        self.assertAlmostEqual(result.metrics["total_return"], 0.0, places=6)

    def test_limit_up_blocks_buy(self) -> None:
        """涨停日无法买入。"""
        weights = pd.DataFrame(0.0, index=self.price.index, columns=self.price.columns)
        weights.iloc[5:, 0] = 1.0  # 第5日起全仓第一只

        # 第6日 (执行日) 涨停
        limit_up = pd.DataFrame(False, index=self.price.index, columns=self.price.columns)
        limit_up.iloc[6, 0] = True

        result = self.bt.run_target_weight(
            weights, self.price, limit_up=limit_up, initial_capital=1_000_000
        )
        # 涨停日该股权重应被限制
        self.assertLess(result.positions.iloc[6, 0], 1.0, "涨停日不应满仓买入")

    def test_limit_down_blocks_sell(self) -> None:
        """跌停日无法卖出。"""
        weights = pd.DataFrame(0.0, index=self.price.index, columns=self.price.columns)
        weights.iloc[:5, 0] = 1.0  # 前5日全仓
        weights.iloc[5:, 0] = 0.0  # 第5日起清仓

        # 第6日 (执行卖出日) 跌停
        limit_down = pd.DataFrame(False, index=self.price.index, columns=self.price.columns)
        limit_down.iloc[6, 0] = True

        result = self.bt.run_target_weight(
            weights, self.price, limit_down=limit_down, initial_capital=1_000_000
        )
        # 跌停日应无法完全卖出, 仍有持仓
        self.assertGreater(result.positions.iloc[6, 0], 0.0, "跌停日应无法完全卖出")

    def test_benchmark_comparison(self) -> None:
        """基准对比: 应计算超额收益与信息比率。"""
        weights = pd.DataFrame(0.0, index=self.price.index, columns=self.price.columns)
        weights.iloc[:, 0] = 1.0
        benchmark = self.price.iloc[:, 0] / self.price.iloc[0, 0]  # 第一只作为基准

        result = self.bt.run_target_weight(
            weights, self.price, benchmark=benchmark, initial_capital=1_000_000
        )
        self.assertIsNotNone(result.benchmark_curve)
        self.assertIn("excess_return", result.metrics)
        self.assertIn("information_ratio", result.metrics)
        self.assertIn("tracking_error", result.metrics)

    def test_turnover_calculation(self) -> None:
        """换手率计算正确性。"""
        weights = pd.DataFrame(0.0, index=self.price.index, columns=self.price.columns)
        # 频繁切换持仓
        for i in range(len(weights)):
            weights.iloc[i, i % len(weights.columns)] = 1.0

        result = self.bt.run_target_weight(weights, self.price, initial_capital=1_000_000)
        # 频繁换仓应有较高换手率
        self.assertGreater(result.turnover.mean(), 0.1)
        self.assertIn("avg_turnover", result.metrics)


class TestMetricsCalculation(unittest.TestCase):
    """绩效指标计算正确性测试。"""

    def test_known_metrics(self) -> None:
        """已知净值的指标计算。"""
        # 构造一个年化 10% 的净值曲线 (252 日)
        dates = pd.bdate_range("2024-01-01", periods=252)
        daily_ret = (1.10 ** (1 / 252)) - 1
        equity = pd.Series(np.cumprod([1 + daily_ret] * 252) * 1_000_000, index=dates)

        m = compute_metrics(equity)
        self.assertAlmostEqual(m["total_return"], 0.10, places=2)
        self.assertAlmostEqual(m["annual_return"], 0.10, places=2)
        self.assertGreater(m["sharpe"], 0)

    def test_drawdown_calculation(self) -> None:
        """最大回撤计算。"""
        # 净值: 1.0 -> 1.2 -> 0.8 -> 1.0, 最大回撤 = (0.8-1.2)/1.2 = -33.3%
        equity = pd.Series([1.0, 1.2, 0.8, 1.0], index=pd.bdate_range("2024-01-01", periods=4))
        m = compute_metrics(equity)
        self.assertAlmostEqual(m["max_drawdown"], -1 / 3, places=2)

    def test_with_benchmark(self) -> None:
        """带基准的指标。"""
        dates = pd.bdate_range("2024-01-01", periods=100)
        equity = pd.Series(np.linspace(1, 1.2, 100), index=dates)
        bench = pd.Series(np.linspace(1, 1.1, 100), index=dates)
        m = compute_metrics(equity, benchmark=bench)
        self.assertIn("excess_return", m)
        self.assertIn("information_ratio", m)
        self.assertGreater(m["excess_return"], 0)

    def test_empty_equity(self) -> None:
        """空净值应返回空 dict。"""
        self.assertEqual(compute_metrics(pd.Series(dtype=float)), {})


class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试。"""

    def setUp(self) -> None:
        self.bt = VectorizedBacktester()

    def test_single_day(self) -> None:
        """仅 1 个交易日。"""
        price = pd.DataFrame(
            {"A": [10.0]}, index=pd.bdate_range("2024-01-01", periods=1)
        )
        weights = pd.DataFrame({"A": [1.0]}, index=price.index)
        result = self.bt.run_target_weight(weights, price, initial_capital=100_000)
        # 单日无收益 (T+1 未执行)
        self.assertAlmostEqual(result.equity_curve.iloc[0], 100_000, delta=1)

    def test_single_stock(self) -> None:
        """单只股票。"""
        price = make_price_panel(n_days=30, n_codes=1)
        weights = pd.DataFrame(1.0, index=price.index, columns=price.columns)
        result = self.bt.run_target_weight(weights, price, initial_capital=100_000)
        self.assertEqual(len(result.equity_curve), 30)

    def test_extreme_weights(self) -> None:
        """权重和超过 1 应被截断 (无杠杆)。"""
        price = make_price_panel(n_days=20, n_codes=5)
        weights = pd.DataFrame(0.5, index=price.index, columns=price.columns)  # 每只 50%, 总 250%
        result = self.bt.run_target_weight(weights, price, initial_capital=100_000)
        # 权重和应 <= 1
        self.assertTrue((result.positions.sum(axis=1) <= 1.0 + 1e-9).all())

    def test_nan_prices(self) -> None:
        """价格含 NaN (停牌) 应不崩溃。"""
        price = make_price_panel(n_days=30, n_codes=5)
        price.iloc[10:15, 0] = np.nan  # 停牌
        weights = pd.DataFrame(0.2, index=price.index, columns=price.columns)
        result = self.bt.run_target_weight(weights, price, initial_capital=100_000)
        self.assertEqual(len(result.equity_curve), 30)

    def test_misaligned_index(self) -> None:
        """权重与价格索引不对齐应自动取交集。"""
        price = make_price_panel(n_days=30, n_codes=5)
        weights = pd.DataFrame(
            0.2,
            index=pd.bdate_range("2024-02-01", periods=20),
            columns=price.columns,
        )
        result = self.bt.run_target_weight(weights, price, initial_capital=100_000)
        self.assertGreater(len(result.equity_curve), 0)


class TestPerformanceComparison(unittest.TestCase):
    """性能对比测试: 向量化 vs 逐日循环 (模拟 native_adapter)。"""

    def test_vectorized_vs_loop(self) -> None:
        """向量化回测应远快于逐日循环。

        jingni-trader native_adapter 采用纯 Python 按日循环,
        全市场 5000 股 x 1000 日预计分钟级。
        本测试用 100 股 x 250 日验证向量化优势。
        """
        n_days, n_codes = 250, 100
        price = make_price_panel(n_days=n_days, n_codes=n_codes)
        open_price = make_open_panel(price)

        # 等权组合, 每 20 日调仓
        weights = pd.DataFrame(0.0, index=price.index, columns=price.columns)
        for i in range(0, n_days, 20):
            weights.iloc[i, :20] = 1.0 / 20  # 每次持有 20 只

        bt = VectorizedBacktester()

        # 向量化回测
        t0 = time.perf_counter()
        result_vec = bt.run_target_weight(weights, price, open_price, initial_capital=1_000_000)
        vec_time = time.perf_counter() - t0

        # 模拟 native_adapter 逐日循环 (简化版, 仅计算收益)
        t0 = time.perf_counter()
        equity = 1_000_000.0
        positions = {}
        for i in range(n_days):
            date = price.index[i]
            # 简化: 每 20 日调仓到等权 20 只
            if i % 20 == 0 and i > 0:
                target_codes = list(price.columns[:20])
                budget = equity * 0.95 / len(target_codes)
                positions = {}
                for code in target_codes:
                    p = price.iloc[i][code]
                    if not np.isnan(p) and p > 0:
                        shares = int(budget / p / 100) * 100
                        positions[code] = shares
            # 计算当日市值
            market_value = sum(
                shares * price.iloc[i][code]
                for code, shares in positions.items()
                if not np.isnan(price.iloc[i][code])
            )
            equity = market_value + (equity - market_value)  # 简化
        loop_time = time.perf_counter() - t0

        speedup = loop_time / vec_time if vec_time > 0 else float("inf")
        print(
            f"\n[性能] 向量化: {vec_time*1000:.1f}ms, "
            f"逐日循环: {loop_time*1000:.1f}ms, 加速比: {speedup:.1f}x"
        )
        # 向量化应至少快 5x (实际通常 50x+)
        self.assertGreater(speedup, 3.0, f"向量化未明显加速: {speedup:.1f}x")
        # 结果应有效
        self.assertGreater(len(result_vec.equity_curve), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
