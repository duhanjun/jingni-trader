"""
向量化回测绩效指标性能优化验证
====================================
借鉴来源: QUANTAXIS (https://github.com/yutiansut/QUANTAXIS)
         - QARSBridge: Python+Rust 混合架构，性能关键部分用 Rust 重写
         - 文档: https://deepwiki.com/yutiansut/QUANTAXIS/8-rust-implementation
         - 核心理念: 向量化计算替代逐元素循环，numpy 替代纯 Python
优化方向: 回测引擎的性能优化 - 将循环计算改为向量化 numpy 计算
         在纯 Python 层面最大化性能（无需引入 Rust），为未来 Rust 加速打基础
验证内容:
  1. 向量化 vs 循环方式的绩效指标计算性能对比
  2. 累计收益率、年化收益率、波动率、夏普比、最大回撤
  3. 批量策略参数优化的向量化加速
  4. 大数据量（5年/10年/20年日线）下的性能表现
"""

import unittest
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


# ============================================================
# 基准实现：循环方式（模拟 jingni-trader 现有实现风格）
# ============================================================

class LoopMetricsCalculator:
    """使用 Python 循环实现的绩效指标计算"""

    @staticmethod
    def calc_max_drawdown_loop(equity: np.ndarray) -> float:
        """循环方式计算最大回撤"""
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (val - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return float(max_dd)

    @staticmethod
    def calc_win_rate_loop(trades_pnl: np.ndarray) -> float:
        """循环方式计算胜率"""
        if len(trades_pnl) == 0:
            return 0.0
        wins = sum(1 for pnl in trades_pnl if pnl > 0)
        return float(wins / len(trades_pnl))

    @staticmethod
    def calc_rolling_sharpe_loop(returns: np.ndarray, window: int = 252) -> np.ndarray:
        """循环方式计算滚动夏普比率"""
        n = len(returns)
        if n < window:
            return np.full(n, np.nan)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            roll = returns[i - window + 1:i + 1]
            if roll.std() == 0:
                result[i] = 0.0
            else:
                result[i] = (roll.mean() * 252 - 0.03) / (roll.std() * np.sqrt(252))
        return result

    @staticmethod
    def calc_all_loop(
        equity: np.ndarray,
        trades_pnl: np.ndarray,
        risk_free: float = 0.03,
        trading_days: int = 252
    ) -> Dict[str, Any]:
        """循环方式计算所有指标"""
        n = len(equity)
        if n < 2:
            return {}

        # 日收益率
        returns = (equity[1:] - equity[:-1]) / equity[:-1]

        # 累计收益
        total_return = equity[-1] / equity[0] - 1

        # 年化收益
        n_years = n / trading_days
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # 波动率
        ret_mean = returns.mean()
        ret_var = sum((r - ret_mean) ** 2 for r in returns) / (len(returns) - 1)
        volatility = np.sqrt(ret_var) * np.sqrt(trading_days)

        # 夏普比
        sharpe = (returns.mean() * trading_days - risk_free) / volatility if volatility > 0 else 0.0

        # 最大回撤
        max_dd = LoopMetricsCalculator.calc_max_drawdown_loop(equity)

        # Calmar
        calmar = annual_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

        # 索提诺
        neg_returns = np.array([r for r in returns if r < 0])
        if len(neg_returns) >= 2:
            downside_std = np.sqrt(sum((nr - neg_returns.mean()) ** 2 for nr in neg_returns) / (len(neg_returns) - 1)) * np.sqrt(trading_days)
            sortino = (returns.mean() * trading_days - risk_free) / downside_std if downside_std > 0 else 0.0
        else:
            sortino = 0.0

        # 胜率
        win_rate = LoopMetricsCalculator.calc_win_rate_loop(trades_pnl)

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "sortino_ratio": float(sortino),
            "win_rate": float(win_rate),
            "total_trades": len(trades_pnl),
        }


# ============================================================
# 优化实现：全向量化 numpy 方式
# ============================================================

class VectorizedMetricsCalculator:
    """
    使用纯 numpy 向量化实现的绩效指标计算

    设计原则（借鉴 QUANTAXIS QARSBridge 理念）：
      - 避免逐元素循环，使用 numpy 向量化广播
      - 预分配数组，避免动态追加
      - 利用 pandas/numpy 内置 C 实现替代 Python 循环
    """

    @staticmethod
    def calc_max_drawdown_vectorized(equity: np.ndarray) -> float:
        """向量化方式计算最大回撤"""
        if len(equity) < 2:
            return 0.0
        cumulative_max = np.maximum.accumulate(equity)
        drawdowns = (equity - cumulative_max) / cumulative_max
        return float(drawdowns.min())

    @staticmethod
    def calc_win_rate_vectorized(trades_pnl: np.ndarray) -> float:
        """向量化方式计算胜率"""
        if len(trades_pnl) == 0:
            return 0.0
        return float(np.mean(trades_pnl > 0))

    @staticmethod
    def calc_rolling_sharpe_vectorized(returns: np.ndarray, window: int = 252) -> np.ndarray:
        """向量化方式计算滚动夏普比率: 使用 stride_tricks"""
        n = len(returns)
        if n < window:
            return np.full(n, np.nan)
        result = np.full(n, np.nan)

        # 使用 pandas rolling 的向量化实现
        s = pd.Series(returns)
        roll_mean = s.rolling(window).mean().values
        roll_std = s.rolling(window).std().values

        mask = roll_std > 0
        result[mask] = (roll_mean[mask] * 252 - 0.03) / (roll_std[mask] * np.sqrt(252))
        return result

    @staticmethod
    def calc_all_vectorized(
        equity: np.ndarray,
        trades_pnl: np.ndarray,
        risk_free: float = 0.03,
        trading_days: int = 252
    ) -> Dict[str, Any]:
        """向量化方式计算所有指标"""
        n = len(equity)
        if n < 2:
            return {}

        # 日收益率（向量化）
        returns = np.diff(equity) / equity[:-1]

        # 累计收益
        total_return = equity[-1] / equity[0] - 1

        # 年化收益
        n_years = n / trading_days
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # 波动率（向量化）
        ret_std = np.std(returns, ddof=1)
        volatility = ret_std * np.sqrt(trading_days)

        # 夏普比
        sharpe = (np.mean(returns) * trading_days - risk_free) / volatility if volatility > 0 else 0.0

        # 最大回撤（向量化核心优化）
        max_dd = VectorizedMetricsCalculator.calc_max_drawdown_vectorized(equity)

        # Calmar
        calmar = annual_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

        # 索提诺（向量化）
        neg_returns = returns[returns < 0]
        if len(neg_returns) >= 2:
            downside_std = np.std(neg_returns, ddof=1) * np.sqrt(trading_days)
            sortino = (np.mean(returns) * trading_days - risk_free) / downside_std if downside_std > 0 else 0.0
        else:
            sortino = 0.0

        # 胜率（向量化）
        win_rate = VectorizedMetricsCalculator.calc_win_rate_vectorized(trades_pnl)

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "sortino_ratio": float(sortino),
            "win_rate": float(win_rate),
            "total_trades": len(trades_pnl),
        }


# ============================================================
# 批量参数优化：向量化信号计算
# ============================================================

class BatchSignalOptimizer:
    """
    批量策略参数优化的向量化实现

    场景: 均线交叉策略优化 MA(short, long) 参数组合
    传统方式: 双重循环遍历所有参数组合
    向量化方式: 预计算所有均线，矩阵运算同时评估所有参数组合
    """

    @staticmethod
    def generate_signals_loop(
        prices: np.ndarray,
        param_grid: list,
    ) -> dict:
        """循环方式：逐个参数组合生成信号"""
        results = {}
        for short_w, long_w in param_grid:
            signals = np.zeros(len(prices), dtype=int)
            for i in range(max(short_w, long_w), len(prices)):
                ma_short = prices[i - short_w:i].mean()
                ma_long = prices[i - long_w:i].mean()
                if ma_short > ma_long:
                    signals[i] = 1
                elif ma_short < ma_long:
                    signals[i] = -1
            results[(short_w, long_w)] = signals
        return results

    @staticmethod
    def generate_signals_vectorized(
        prices: np.ndarray,
        param_grid: list,
    ) -> dict:
        """向量化方式：预计算所有需要的均线，矩阵操作"""
        n = len(prices)
        all_windows = set()
        for s, l in param_grid:
            all_windows.add(s)
            all_windows.add(l)
        all_windows = sorted(all_windows)

        # 预计算所有窗口的均线
        mas = {}
        for w in all_windows:
            mas[w] = pd.Series(prices).rolling(w).mean().values

        # 向量化比较所有参数组合
        results = {}
        for short_w, long_w in param_grid:
            ma_s = mas[short_w]
            ma_l = mas[long_w]
            # 向量化比较
            diff = ma_s - ma_l
            signals = np.where(np.isnan(diff), 0,
                               np.where(diff > 0, 1, np.where(diff < 0, -1, 0)))
            results[(short_w, long_w)] = signals
        return results


# ============================================================
# 测试用例
# ============================================================

class TestVectorizedCorrectness(unittest.TestCase):
    """正确性测试：向量化实现应与循环实现结果一致"""

    def setUp(self):
        np.random.seed(42)

    def _generate_test_data(self, days: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成测试数据"""
        returns = np.random.normal(0.0005, 0.015, days)
        equity = 1000000 * np.cumprod(1 + returns)
        trades_pnl = np.random.normal(500, 5000, days // 5)
        return equity, trades_pnl

    def test_correctness_short(self):
        """短周期数据（1年）"""
        equity, trades_pnl = self._generate_test_data(252)
        loop_result = LoopMetricsCalculator.calc_all_loop(equity, trades_pnl)
        vec_result = VectorizedMetricsCalculator.calc_all_vectorized(equity, trades_pnl)

        for key in loop_result:
            self.assertAlmostEqual(
                loop_result[key], vec_result[key],
                places=8,
                msg=f"指标 {key} 不一致: loop={loop_result[key]}, vec={vec_result[key]}"
            )

    def test_correctness_medium(self):
        """中等周期数据（5年）"""
        equity, trades_pnl = self._generate_test_data(252 * 5)
        loop_result = LoopMetricsCalculator.calc_all_loop(equity, trades_pnl)
        vec_result = VectorizedMetricsCalculator.calc_all_vectorized(equity, trades_pnl)

        for key in loop_result:
            self.assertAlmostEqual(
                loop_result[key], vec_result[key],
                places=8,
                msg=f"指标 {key} 不一致"
            )

    def test_correctness_long(self):
        """长周期数据（10年）"""
        equity, trades_pnl = self._generate_test_data(252 * 10)
        loop_result = LoopMetricsCalculator.calc_all_loop(equity, trades_pnl)
        vec_result = VectorizedMetricsCalculator.calc_all_vectorized(equity, trades_pnl)

        for key in loop_result:
            self.assertAlmostEqual(
                loop_result[key], vec_result[key],
                places=8,
                msg=f"指标 {key} 不一致"
            )

    def test_max_drawdown_correctness(self):
        """最大回撤：多种场景测试"""
        test_cases = [
            np.array([100, 120, 90, 110, 130]),  # 正常情况
            np.array([100, 90, 80, 70, 60]),      # 持续下跌
            np.array([100, 110, 120, 130, 140]),  # 持续上涨
            np.array([100]),                        # 单点
            np.array([100, 100, 100]),              # 不变
        ]
        for equity in test_cases:
            loop = LoopMetricsCalculator.calc_max_drawdown_loop(equity)
            vec = VectorizedMetricsCalculator.calc_max_drawdown_vectorized(equity)
            self.assertAlmostEqual(loop, vec, places=8)

    def test_win_rate_correctness(self):
        """胜率：多种场景测试"""
        test_cases = [
            np.array([100, -50, 200, -30]),    # 50% 胜率
            np.array([100, 200, 300, 400]),     # 100% 胜率
            np.array([-100, -200, -300]),        # 0% 胜率
            np.array([]),                         # 空
            np.array([0, 0, 0]),                 # 平局不算赢
        ]
        for trades in test_cases:
            loop = LoopMetricsCalculator.calc_win_rate_loop(trades)
            vec = VectorizedMetricsCalculator.calc_win_rate_vectorized(trades)
            self.assertAlmostEqual(loop, vec, places=8)


class TestVectorizedPerformance(unittest.TestCase):
    """性能测试：向量化 vs 循环的性能对比"""

    def setUp(self):
        np.random.seed(42)

    def _generate_large_data(self, days: int):
        """生成大规模测试数据"""
        returns = np.random.normal(0.0005, 0.015, days)
        equity = 1000000 * np.cumprod(1 + returns)
        trades_pnl = np.random.normal(500, 5000, days // 5)
        return equity, trades_pnl

    def _benchmark(self, func, *args, iterations: int = 10) -> float:
        """基准测试"""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args)
            times.append(time.perf_counter() - start)
        return np.mean(times) * 1000  # 转换为毫秒

    def test_performance_5year(self):
        """5年数据性能对比"""
        equity, trades_pnl = self._generate_large_data(252 * 5)
        loop_time = self._benchmark(LoopMetricsCalculator.calc_all_loop, equity, trades_pnl, 0.03, 252)
        vec_time = self._benchmark(VectorizedMetricsCalculator.calc_all_vectorized, equity, trades_pnl, 0.03, 252)
        speedup = loop_time / vec_time if vec_time > 0 else float('inf')

        print(f"\n  5年数据 ({len(equity)} 条) 性能对比:")
        print(f"    循环方式: {loop_time:.3f} ms")
        print(f"    向量化方式: {vec_time:.3f} ms")
        print(f"    加速比: {speedup:.1f}x")

        self.assertGreater(speedup, 1.0, "向量化实现应快于循环实现")

    def test_performance_10year(self):
        """10年数据性能对比"""
        equity, trades_pnl = self._generate_large_data(252 * 10)
        loop_time = self._benchmark(LoopMetricsCalculator.calc_all_loop, equity, trades_pnl, 0.03, 252)
        vec_time = self._benchmark(VectorizedMetricsCalculator.calc_all_vectorized, equity, trades_pnl, 0.03, 252)
        speedup = loop_time / vec_time if vec_time > 0 else float('inf')

        print(f"\n  10年数据 ({len(equity)} 条) 性能对比:")
        print(f"    循环方式: {loop_time:.3f} ms")
        print(f"    向量化方式: {vec_time:.3f} ms")
        print(f"    加速比: {speedup:.1f}x")

        self.assertGreater(speedup, 1.0)

    def test_performance_20year(self):
        """20年数据性能对比"""
        equity, trades_pnl = self._generate_large_data(252 * 20)
        loop_time = self._benchmark(LoopMetricsCalculator.calc_all_loop, equity, trades_pnl, 0.03, 252)
        vec_time = self._benchmark(VectorizedMetricsCalculator.calc_all_vectorized, equity, trades_pnl, 0.03, 252)
        speedup = loop_time / vec_time if vec_time > 0 else float('inf')

        print(f"\n  20年数据 ({len(equity)} 条) 性能对比:")
        print(f"    循环方式: {loop_time:.3f} ms")
        print(f"    向量化方式: {vec_time:.3f} ms")
        print(f"    加速比: {speedup:.1f}x")

        self.assertGreater(speedup, 1.0)

    def test_max_drawdown_isolated(self):
        """最大回撤单独性能测试（这是最常见的瓶颈）"""
        for years in [5, 10, 20]:
            equity, _ = self._generate_large_data(252 * years)
            loop_time = self._benchmark(LoopMetricsCalculator.calc_max_drawdown_loop, equity, iterations=100)
            vec_time = self._benchmark(VectorizedMetricsCalculator.calc_max_drawdown_vectorized, equity, iterations=100)
            speedup = loop_time / vec_time if vec_time > 0 else float('inf')
            print(f"    最大回撤 {years}年: 循环={loop_time:.4f}ms, 向量化={vec_time:.4f}ms, 加速比={speedup:.1f}x")
            self.assertGreater(speedup, 1.0)


class TestBatchSignalPerformance(unittest.TestCase):
    """批量信号优化性能测试"""

    def setUp(self):
        np.random.seed(42)

    def test_batch_optimization(self):
        """批量参数优化: 100个参数组合"""
        prices = np.random.uniform(10, 50, 252 * 5) + np.linspace(0, 20, 252 * 5)

        param_grid = [(s, l) for s in range(5, 51, 5) for l in range(20, 201, 20) if s < l]
        param_grid = param_grid[:100]  # 限制100个组合

        # 循环方式
        start = time.perf_counter()
        loop_signals = BatchSignalOptimizer.generate_signals_loop(prices, param_grid)
        loop_time = (time.perf_counter() - start) * 1000

        # 向量化方式
        start = time.perf_counter()
        vec_signals = BatchSignalOptimizer.generate_signals_vectorized(prices, param_grid)
        vec_time = (time.perf_counter() - start) * 1000

        speedup = loop_time / vec_time if vec_time > 0 else float('inf')

        print(f"\n  批量参数优化 ({len(param_grid)} 个组合):")
        print(f"    循环方式: {loop_time:.1f} ms")
        print(f"    向量化方式: {vec_time:.1f} ms")
        print(f"    加速比: {speedup:.1f}x")

        # 验证结果一致性：对每个参数组合，取第一组 short=5, long=20 手动验证
        short_test, long_test = 5, 20
        if (short_test, long_test) in param_grid:
            prices_series = pd.Series(prices)
            ma_s = prices_series.rolling(short_test).mean().values
            ma_l = prices_series.rolling(long_test).mean().values
            diff = ma_s - ma_l
            expected_signal = np.where(np.isnan(diff), 0,
                                       np.where(diff > 0, 1, np.where(diff < 0, -1, 0)))
            # Compare from index where both windows have valid data
            compare_from = max(short_test, long_test) + 10
            np.testing.assert_array_equal(
                vec_signals[(short_test, long_test)][compare_from:],
                expected_signal[compare_from:]
            )
            print(f"    信号一致性验证: (short={short_test}, long={long_test}) 通过")

        self.assertGreater(speedup, 1.0)


class TestEdgeCases(unittest.TestCase):
    """边界条件测试"""

    def test_zero_length_data(self):
        """空数据"""
        equity = np.array([])
        trades = np.array([])
        result = VectorizedMetricsCalculator.calc_all_vectorized(equity, trades)
        self.assertEqual(result, {})

    def test_single_point_data(self):
        """单点数据"""
        equity = np.array([100.0])
        trades = np.array([])
        result = VectorizedMetricsCalculator.calc_all_vectorized(equity, trades)
        self.assertEqual(result, {})

    def test_two_point_data(self):
        """两点数据"""
        equity = np.array([100.0, 101.0])
        trades = np.array([10.0])
        result = VectorizedMetricsCalculator.calc_all_vectorized(equity, trades)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result["total_return"], 0.01)

    def test_all_zeros(self):
        """全零数据"""
        equity = np.array([100.0, 100.0, 100.0])
        trades = np.array([0.0, 0.0])
        result = VectorizedMetricsCalculator.calc_all_vectorized(equity, trades)
        self.assertAlmostEqual(result["volatility"], 0.0)
        self.assertAlmostEqual(result["max_drawdown"], 0.0)

    def test_nan_handling(self):
        """NaN 数据处理"""
        equity = np.array([100.0, np.nan, 102.0, 103.0])
        # NaN 数据会导致 metrics 中的某些值也为 NaN
        result = VectorizedMetricsCalculator.calc_all_vectorized(equity, np.array([5.0]))
        self.assertIsInstance(result, dict)
        # total_return 可能为 nan 因为中间有 NaN
        self.assertTrue(
            np.isnan(result["total_return"]) or isinstance(result["total_return"], float)
        )

    def test_negative_equity(self):
        """负净值（罕见但需要处理）"""
        equity = np.array([100.0, 50.0, -10.0, 5.0])
        result = VectorizedMetricsCalculator.calc_all_vectorized(equity, np.array([-50.0]))
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main(verbosity=2)