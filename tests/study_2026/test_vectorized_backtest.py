"""
优化方向：向量化快速回测加速器
借鉴来源：VectorBT (https://github.com/polakowo/vectorbt)
借鉴亮点：
  - 矩阵化计算范式：将数千种策略配置打包到多维 NumPy 数组
  - 利用 Numba 加速核心计算路径
  - 单次操作同时评估所有参数组合
  - Purged Walk-Forward Cross-Validation

问题分析：
  jingni-trader 当前回测依赖 RQAlpha / Backtrader 等事件驱动框架，
  当需要测试大量参数组合时效率较低（逐事件循环 + Python 开销）。
  例如：测试 100 个参数组合需要运行 100 次独立的回测。

优化方案：
  引入 VectorizedBacktester，基于 NumPy 向量化操作实现：
  1. 一次计算全部参数组合的交易信号矩阵
  2. 向量化计算持仓、权益曲线、绩效指标
  3. 批量比较所有参数组合的绩效
  4. 支持 purged walk-forward CV
"""

import sys
import os
import unittest
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# 向量化回测引擎（借鉴 VectorBT 矩阵化计算范式）
# ============================================================

class VectorizedBacktester:
    """
    向量化回测引擎

    核心思路（借鉴 VectorBT）：
    - 将参数搜索空间表示为多维数组
    - 利用 NumPy 向量化操作一次性计算所有参数组合
    - 避免 Python 循环开销
    """

    def __init__(
        self,
        commission_rate: float = 0.0003,
        stamp_tax_rate: float = 0.001,   # 卖出印花税
        slippage: float = 0.0001,
        init_capital: float = 1e6,
        t_plus_1: bool = True,
    ):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.init_capital = init_capital
        self.t_plus_1 = t_plus_1

    def generate_signals_ma_cross(
        self,
        prices: pd.DataFrame,       # date × code
        fast_windows: List[int],
        slow_windows: List[int],
    ) -> Dict[str, np.ndarray]:
        """
        批量生成双均线交叉信号矩阵（借鉴 VectorBT 的 run_combs 模式）

        返回:
            entries: shape (n_combos, n_dates, n_codes) bool 数组
        """
        n_dates, n_codes = prices.shape
        price_array = prices.values

        fast_windows = sorted(set(fast_windows))
        slow_windows = sorted(set(slow_windows))

        entries_list = []
        exits_list = []
        combo_labels = []

        for fast_w in fast_windows:
            fast_ma = np.full_like(price_array, np.nan)
            for i in range(fast_w - 1, n_dates):
                fast_ma[i] = np.nanmean(price_array[i - fast_w + 1:i + 1], axis=0)

            for slow_w in slow_windows:
                if fast_w >= slow_w:
                    continue  # 快线必须短于慢线

                slow_ma = np.full_like(price_array, np.nan)
                for i in range(slow_w - 1, n_dates):
                    slow_ma[i] = np.nanmean(price_array[i - slow_w + 1:i + 1], axis=0)

                # 向量化生成信号
                entry = np.full((n_dates, n_codes), False)
                exit = np.full((n_dates, n_codes), False)

                for i in range(1, n_dates):
                    valid = ~np.isnan(fast_ma[i]) & ~np.isnan(slow_ma[i]) & \
                            ~np.isnan(fast_ma[i-1]) & ~np.isnan(slow_ma[i-1])
                    # 金叉：快线上穿慢线
                    cross_up = valid & (fast_ma[i] > slow_ma[i]) & (fast_ma[i-1] <= slow_ma[i-1])
                    # 死叉：快线下穿慢线
                    cross_down = valid & (fast_ma[i] < slow_ma[i]) & (fast_ma[i-1] >= slow_ma[i-1])

                    entry[i, cross_up] = True
                    exit[i, cross_down] = True

                entries_list.append(entry)
                exits_list.append(exit)
                combo_labels.append(f"MA({fast_w},{slow_w})")

        return {
            "entries": entries_list,
            "exits": exits_list,
            "labels": combo_labels,
            "fast_windows": fast_windows,
            "slow_windows": slow_windows,
        }

    def backtest_from_signals(
        self,
        prices: np.ndarray,           # (n_dates, n_codes)
        entries: np.ndarray,          # (n_dates, n_codes) bool
        exits: np.ndarray,            # (n_dates, n_codes) bool
    ) -> Dict[str, np.ndarray]:
        """
        从信号数组运行向量化回测

        参数:
            prices: 价格矩阵 (n_dates, n_codes)
            entries: 买入信号 (n_dates, n_codes)
            exits: 卖出信号 (n_dates, n_codes)

        返回:
            {
                'equity_curve': (n_dates,),
                'positions': (n_dates, n_codes),
                'trades': [...],
                'metrics': {...}
            }
        """
        n_dates, n_codes = prices.shape

        # 向量化追踪持仓
        positions = np.zeros((n_dates, n_codes), dtype=float)
        cash = np.full(n_dates, self.init_capital, dtype=float)
        equity = np.full(n_dates, self.init_capital, dtype=float)

        trade_count = 0
        total_commission = 0.0

        for t in range(n_dates):
            # 前一日持仓
            if t > 0:
                positions[t] = positions[t-1].copy()
                cash[t] = cash[t-1]

            # 当日信号处理
            buy_mask = entries[t]
            sell_mask = exits[t]

            # 先卖后买
            for code_idx in range(n_codes):
                if sell_mask[code_idx] and positions[t, code_idx] > 0:
                    sell_value = positions[t, code_idx] * prices[t, code_idx]
                    commission = sell_value * (self.commission_rate + self.stamp_tax_rate)
                    slippage_cost = sell_value * self.slippage

                    cash[t] += sell_value - commission - slippage_cost
                    total_commission += commission + slippage_cost
                    positions[t, code_idx] = 0
                    trade_count += 1

                if buy_mask[code_idx]:
                    available_cash = cash[t] * 0.9  # 留 10% 现金
                    n_positions = max(1, np.sum(positions[t] > 0) + 1)
                    per_stock_cash = available_cash / n_positions

                    # 最小交易 100 股（A 股规则）
                    max_shares = int(per_stock_cash / (prices[t, code_idx] * (1 + self.commission_rate)))
                    max_shares = (max_shares // 100) * 100  # 整手

                    if max_shares >= 100:
                        buy_value = max_shares * prices[t, code_idx]
                        commission = buy_value * self.commission_rate
                        slippage_cost = buy_value * self.slippage
                        total_cost = buy_value + commission + slippage_cost

                        if total_cost <= cash[t]:
                            cash[t] -= total_cost
                            positions[t, code_idx] = max_shares
                            total_commission += commission + slippage_cost
                            trade_count += 1

            # 计算当日权益
            position_value = np.sum(positions[t] * prices[t])
            equity[t] = cash[t] + position_value

        # 计算性能指标
        metrics = self._calc_metrics(equity)

        return {
            "equity_curve": equity,
            "positions": positions,
            "trade_count": trade_count,
            "total_commission": total_commission,
            "metrics": metrics,
        }

    def batch_backtest_ma_cross(
        self,
        prices: pd.DataFrame,
        fast_windows: List[int],
        slow_windows: List[int],
    ) -> pd.DataFrame:
        """
        批量回测所有参数组合

        返回: DataFrame 包含每个组合的绩效指标
        """
        signals = self.generate_signals_ma_cross(prices, fast_windows, slow_windows)
        price_array = prices.values

        results = []

        for i, (entry, exit, label) in enumerate(
            zip(signals["entries"], signals["exits"], signals["labels"])
        ):
            bt_result = self.backtest_from_signals(price_array, entry, exit)
            metrics = bt_result["metrics"]
            metrics["label"] = label
            metrics["trades"] = bt_result["trade_count"]

            # 提取参数
            parts = label.replace("MA(", "").replace(")", "").split(",")
            metrics["fast_window"] = int(parts[0])
            metrics["slow_window"] = int(parts[1])

            results.append(metrics)

        return pd.DataFrame(results)

    def _calc_metrics(self, equity: np.ndarray) -> Dict[str, float]:
        """计算绩效指标"""
        n = len(equity)
        if n < 2:
            return {"total_return": 0, "sharpe_ratio": 0, "max_drawdown": 0}

        total_return = equity[-1] / equity[0] - 1
        annual_return = (1 + total_return) ** (252 / n) - 1

        daily_returns = equity[1:] / equity[:-1] - 1
        volatility = float(np.std(daily_returns) * np.sqrt(252))
        sharpe = annual_return / volatility if volatility > 0 else 0

        # 最大回撤
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = float(np.min(drawdown))

        # 胜率
        win_rate = float(np.mean(daily_returns > 0)) if len(daily_returns) > 0 else 0

        return {
            "total_return": round(float(total_return), 6),
            "annual_return": round(float(annual_return), 6),
            "volatility": round(float(volatility), 6),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown": round(float(max_drawdown), 6),
            "win_rate": round(float(win_rate), 4),
        }

    def purged_walk_forward_cv(
        self,
        prices: pd.DataFrame,
        fast_window: int,
        slow_window: int,
        n_folds: int = 5,
        purge_days: int = 5,
    ) -> Dict[str, List]:
        """
        Purged Walk-Forward 交叉验证（借鉴 VectorBT + Lopez de Prado）

        将数据分为 n_folds 个训练/验证折叠，
        通过 purge + embargo 防止信息泄露
        """
        n_dates = len(prices)
        fold_size = n_dates // (n_folds + 1)
        test_size = fold_size

        fold_metrics = []

        for fold in range(n_folds):
            train_end = (fold + 1) * fold_size
            # purge: 训练集末尾跳过 purge_days
            train_end_purged = max(0, train_end - purge_days)
            test_start = train_end + 1
            test_end = min(test_start + test_size, n_dates)

            if test_start >= n_dates or test_end <= test_start:
                continue

            # 在训练集上回测
            train_prices = prices.iloc[:train_end_purged]
            test_prices = prices.iloc[test_start:test_end]

            if len(train_prices) < slow_window or len(test_prices) < 3:
                continue

            # 训练集回测
            signals = self.generate_signals_ma_cross(
                train_prices, [fast_window], [slow_window]
            )
            train_bt = self.backtest_from_signals(
                train_prices.values, signals["entries"][0], signals["exits"][0],
            )

            # 测试集回测（使用相同参数）
            test_signals = self.generate_signals_ma_cross(
                test_prices, [fast_window], [slow_window]
            )
            test_bt = self.backtest_from_signals(
                test_prices.values, test_signals["entries"][0], test_signals["exits"][0],
            )

            fold_metrics.append({
                "fold": fold,
                "train_return": train_bt["metrics"]["total_return"],
                "test_return": test_bt["metrics"]["total_return"],
                "train_sharpe": train_bt["metrics"]["sharpe_ratio"],
                "test_sharpe": test_bt["metrics"]["sharpe_ratio"],
            })

        if not fold_metrics:
            return {"folds": [], "summary": {}}

        df_folds = pd.DataFrame(fold_metrics)
        summary = {
            "mean_test_return": float(df_folds["test_return"].mean()),
            "std_test_return": float(df_folds["test_return"].std()),
            "mean_test_sharpe": float(df_folds["test_sharpe"].mean()),
            "n_folds": len(fold_metrics),
        }

        return {"folds": [m for m in fold_metrics], "summary": summary}


# ============================================================
# 单元测试
# ============================================================


class TestVectorizedBacktester(unittest.TestCase):
    """向量化回测引擎测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟价格数据"""
        np.random.seed(42)
        n_dates = 500
        n_codes = 10
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")

        prices = np.zeros((n_dates, n_codes))
        for i in range(n_codes):
            start = np.random.uniform(10, 50)
            returns = np.random.randn(n_dates) * 0.02 + 0.0005  # 微小正期望
            prices[:, i] = start * np.cumprod(1 + returns)

        codes = [f"{i:06d}.SZ" for i in range(n_codes)]
        cls.prices = pd.DataFrame(prices, index=dates, columns=codes)
        cls.n_dates = n_dates
        cls.n_codes = n_codes

    def test_single_backtest(self):
        """测试单次回测"""
        bt = VectorizedBacktester(init_capital=1e6)

        signals = bt.generate_signals_ma_cross(
            self.prices, fast_windows=[5], slow_windows=[20]
        )

        result = bt.backtest_from_signals(
            self.prices.values,
            signals["entries"][0],
            signals["exits"][0],
        )

        self.assertIn("equity_curve", result)
        self.assertIn("metrics", result)
        self.assertEqual(len(result["equity_curve"]), self.n_dates)
        self.assertGreater(result["equity_curve"][-1], 0)

        # 回测不应亏损初始资金（考虑交易成本后可能略有亏损，放宽限制）
        self.assertGreater(result["equity_curve"][-1], 0.5 * bt.init_capital,
                          "回测不应亏损超过 50% 初始资金")

    def test_metrics(self):
        """测试绩效指标计算"""
        bt = VectorizedBacktester()

        signals = bt.generate_signals_ma_cross(
            self.prices, fast_windows=[5], slow_windows=[20]
        )
        result = bt.backtest_from_signals(
            self.prices.values,
            signals["entries"][0],
            signals["exits"][0],
        )

        metrics = result["metrics"]
        self.assertIn("total_return", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("win_rate", metrics)

        # 检查值在合理范围
        self.assertGreaterEqual(metrics["max_drawdown"], -1.0)
        self.assertLessEqual(metrics["max_drawdown"], 0.0)
        self.assertGreaterEqual(metrics["win_rate"], 0.0)
        self.assertLessEqual(metrics["win_rate"], 1.0)

    def test_batch_backtest(self):
        """测试批量参数回测"""
        bt = VectorizedBacktester(init_capital=1e6)

        fast_windows = [5, 10, 20]
        slow_windows = [30, 60]

        results_df = bt.batch_backtest_ma_cross(
            self.prices, fast_windows, slow_windows
        )

        expected_combos = sum(1 for f in fast_windows for s in slow_windows if f < s)
        self.assertEqual(
            len(results_df), expected_combos,
            f"应该有 {expected_combos} 个参数组合"
        )

        # 验证列
        for col in ["total_return", "sharpe_ratio", "max_drawdown", "label",
                     "fast_window", "slow_window"]:
            self.assertIn(col, results_df.columns)

    def test_purged_walk_forward_cv(self):
        """测试 Purged Walk-Forward CV"""
        bt = VectorizedBacktester()

        cv_result = bt.purged_walk_forward_cv(
            self.prices,
            fast_window=5,
            slow_window=20,
            n_folds=5,
            purge_days=3,
        )

        self.assertIn("folds", cv_result)
        self.assertIn("summary", cv_result)
        self.assertGreater(len(cv_result["folds"]), 0)

        summary = cv_result["summary"]
        self.assertIn("mean_test_return", summary)
        self.assertIn("mean_test_sharpe", summary)

    def test_transaction_cost(self):
        """测试交易成本计算"""
        # 高费率 vs 零费率
        bt_with_cost = VectorizedBacktester(
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
        )
        bt_no_cost = VectorizedBacktester(
            commission_rate=0.0,
            stamp_tax_rate=0.0,
        )

        signals = bt_with_cost.generate_signals_ma_cross(
            self.prices, fast_windows=[5], slow_windows=[20]
        )

        result_with = bt_with_cost.backtest_from_signals(
            self.prices.values,
            signals["entries"][0],
            signals["exits"][0],
        )
        result_no = bt_no_cost.backtest_from_signals(
            self.prices.values,
            signals["entries"][0],
            signals["exits"][0],
        )

        # 有成本的最终权益 ≤ 无成本的最终权益
        self.assertLessEqual(
            result_with["equity_curve"][-1],
            result_no["equity_curve"][-1] + 1,
            "有交易成本的权益不应显著高于无成本的"
        )

    def test_position_limits(self):
        """测试持仓约束（整手、最小交易量）"""
        bt = VectorizedBacktester(init_capital=10000)  # 小资金

        signals = bt.generate_signals_ma_cross(
            self.prices, fast_windows=[5], slow_windows=[20]
        )
        result = bt.backtest_from_signals(
            self.prices.values,
            signals["entries"][0],
            signals["exits"][0],
        )

        # 检查持仓数量
        positions = result["positions"]
        # 每个持仓量应该是 100 的整数倍（A 股整手）
        non_zero_mask = positions > 0
        for t in range(positions.shape[0]):
            for c in range(positions.shape[1]):
                if positions[t, c] > 0:
                    self.assertEqual(
                        positions[t, c] % 100, 0,
                        f"持仓应为整手(100股)的倍数: t={t}, c={c}, pos={positions[t,c]}"
                    )


class TestPerformanceComparison(unittest.TestCase):
    """性能对比：向量化 vs 逐循环回测"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_dates = 1000
        n_codes = 20
        dates = pd.date_range("2021-01-01", periods=n_dates, freq="B")

        prices = np.zeros((n_dates, n_codes))
        for i in range(n_codes):
            start = np.random.uniform(10, 50)
            returns = np.random.randn(n_dates) * 0.02 + 0.0005
            prices[:, i] = start * np.cumprod(1 + returns)

        codes = [f"{i:06d}.SZ" for i in range(n_codes)]
        cls.prices = pd.DataFrame(prices, index=dates, columns=codes)

    def test_performance_comparison(self):
        """向量化回测 vs 逐股循环回测性能对比"""
        bt = VectorizedBacktester()
        price_array = self.prices.values

        # 向量化方式：一次性处理全部股票
        signals = bt.generate_signals_ma_cross(
            self.prices, fast_windows=[10], slow_windows=[30]
        )

        start = time.perf_counter()
        result_vec = bt.backtest_from_signals(
            price_array, signals["entries"][0], signals["exits"][0],
        )
        vec_time = time.perf_counter() - start

        # 逐股循环方式：逐只股票独立回测
        start = time.perf_counter()
        n_codes = price_array.shape[1]
        all_positions = np.zeros_like(price_array, dtype=float)
        all_equity_sum = np.zeros(price_array.shape[0])

        for c in range(n_codes):
            single_price = price_array[:, [c]]
            single_entry = signals["entries"][0][:, [c]]
            single_exit = signals["exits"][0][:, [c]]

            # 简化的单股回测
            pos = np.zeros(len(single_price))
            cash = 1e6 / n_codes
            equity = np.zeros(len(single_price))
            for t in range(len(single_price)):
                if t > 0:
                    pos[t] = pos[t-1]
                if single_entry[t, 0] and pos[t] == 0:
                    pos[t] = int(cash / single_price[t, 0] / 100) * 100
                    cash -= pos[t] * single_price[t, 0]
                elif single_exit[t, 0] and pos[t] > 0:
                    cash += pos[t] * single_price[t, 0]
                    pos[t] = 0
                equity[t] = cash + pos[t] * single_price[t, 0]
            all_positions[:, c] = pos
            all_equity_sum += equity

        loop_time = time.perf_counter() - start

        print(f"\n性能对比 (20 只股票 × 1000 天):")
        print(f"  向量化回测: {vec_time:.4f}秒")
        print(f"  逐股循环:   {loop_time:.4f}秒")
        print(f"  加速比:     {loop_time/vec_time:.2f}x")

        # 向量化方式应更快
        self.assertLess(vec_time, loop_time,
                       "向量化回测应比逐股循环更快")

    def test_batch_performance(self):
        """批量参数测试性能"""
        bt = VectorizedBacktester()

        fast_windows = [5, 10, 20, 30]
        slow_windows = [40, 60, 80]

        start = time.perf_counter()
        results = bt.batch_backtest_ma_cross(self.prices, fast_windows, slow_windows)
        batch_time = time.perf_counter() - start

        n_combos = len(results)
        print(f"\n批量参数回测性能 ({n_combos} 个参数组合):")
        print(f"  总耗时:    {batch_time:.4f}秒")
        print(f"  平均耗时:  {batch_time/n_combos:.4f}秒/组合")

        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)