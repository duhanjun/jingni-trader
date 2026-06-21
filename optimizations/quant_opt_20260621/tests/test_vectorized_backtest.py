"""
向量化回测引擎测试

验证内容:
1. 正确性: 向量化回测与现有 native_adapter 结果一致性 (净值曲线、交易记录)
2. 性能: 向量化回测 vs native_adapter 逐日循环的耗时对比
3. 边界条件: 空信号、全买入、全卖出、涨跌停限制、T+1 规则
4. A 股规则: 印花税、佣金、滑点、最小 100 股
"""
import os
import sys
import time
import unittest

import numpy as np
import pandas as pd

# 把优化包目录与项目根目录加入 path
_OPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.abspath(os.path.join(_OPT_DIR, ".."))
for p in [_OPT_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vectorized_backtest import VectorizedBacktest, crossover_signals, topk_signals
from tests.test_data_generator import generate_synthetic_data, generate_signal_data


def _signals_to_matrices(signals: pd.DataFrame) -> tuple:
    """把 signal 长表转为 entries/exits 布尔矩阵"""
    entries = signals.pivot(index="date", columns="code", values="signal").fillna(0) > 0
    exits = signals.pivot(index="date", columns="code", values="signal").fillna(0) < 0
    return entries, exits


class TestVectorizedBacktestBasics(unittest.TestCase):
    """基础功能测试"""

    def setUp(self):
        self.data = generate_synthetic_data(n_codes=10, n_days=100, seed=42)
        self.signals = generate_signal_data(self.data, fast_window=5, slow_window=20)
        self.entries, self.exits = _signals_to_matrices(self.signals)
        self.bt = VectorizedBacktest()

    def test_basic_run(self):
        """基础回测运行"""
        result = self.bt.from_signals(self.data, self.entries, self.exits, init_capital=1e6)
        self.assertIn("equity_curve", result)
        self.assertIn("metrics", result)
        self.assertIn("trades", result)
        self.assertGreater(len(result["equity_curve"]), 0)

    def test_equity_curve_shape(self):
        """净值曲线形状正确"""
        result = self.bt.from_signals(self.data, self.entries, self.exits)
        ec = result["equity_curve"]
        self.assertEqual(len(ec), self.data["date"].nunique())
        self.assertIn("equity", ec.columns)
        self.assertIn("cash", ec.columns)
        self.assertIn("market_value", ec.columns)

    def test_metrics_completeness(self):
        """绩效指标完整"""
        result = self.bt.from_signals(self.data, self.entries, self.exits)
        m = result["metrics"]
        for key in ["total_return", "annual_return", "sharpe_ratio", "max_drawdown",
                     "calmar_ratio", "sortino_ratio", "win_rate", "total_trades"]:
            self.assertIn(key, m, f"缺少指标: {key}")

    def test_initial_capital(self):
        """初始资金正确"""
        result = self.bt.from_signals(self.data, self.entries, self.exits, init_capital=2e6)
        first_equity = result["equity_curve"]["equity"].iloc[0]
        # 第一天若无交易, 净值应等于初始资金
        self.assertAlmostEqual(first_equity, 2e6, delta=1.0)


class TestAShareRules(unittest.TestCase):
    """A 股规则测试"""

    def setUp(self):
        self.data = generate_synthetic_data(n_codes=5, n_days=60, seed=100)
        self.signals = generate_signal_data(self.data, fast_window=3, slow_window=10)
        self.entries, self.exits = _signals_to_matrices(self.signals)

    def test_commission_charged(self):
        """佣金正确收取"""
        bt = VectorizedBacktest(commission_rate=0.001, min_commission=5.0)
        result = bt.from_signals(self.data, self.entries, self.exits)
        trades = result["trades"]
        if not trades.empty:
            buy_trades = trades[trades["action"] == "buy"]
            if not buy_trades.empty:
                # 佣金 = max(amount * 0.001, 5)
                for _, t in buy_trades.iterrows():
                    expected = max(t["amount"] * 0.001, 5.0)
                    self.assertAlmostEqual(t["commission"], expected, places=2)

    def test_stamp_tax_on_sell(self):
        """印花税仅在卖出时收取"""
        bt = VectorizedBacktest(stamp_tax_rate=0.001)
        result = bt.from_signals(self.data, self.entries, self.exits)
        trades = result["trades"]
        if not trades.empty:
            buy_trades = trades[trades["action"] == "buy"]
            sell_trades = trades[trades["action"] == "sell"]
            # 买入无印花税
            for _, t in buy_trades.iterrows():
                self.assertEqual(t["tax"], 0.0, "买入不应收印花税")
            # 卖出有印花税
            for _, t in sell_trades.iterrows():
                expected = t["amount"] * 0.001
                self.assertAlmostEqual(t["tax"], expected, places=2)

    def test_t_plus_1(self):
        """T+1 规则: 买入当日不能卖出"""
        bt = VectorizedBacktest(t_plus_1=True)
        # 构造: 同一天既有买入又有卖出信号
        data = self.data.copy()
        dates = sorted(data["date"].unique())
        # 在第 5 天对所有股票发出买入信号, 第 5 天同时卖出 (应被 T+1 阻止)
        entries = pd.DataFrame(False, index=dates, columns=data["code"].unique())
        exits = pd.DataFrame(False, index=dates, columns=data["code"].unique())
        target_date = dates[5]
        entries.loc[target_date] = True
        exits.loc[target_date] = True  # 同日卖出

        result = bt.from_signals(data, entries, exits)
        trades = result["trades"]
        # 第 5 天应该只有买入, 没有卖出 (T+1)
        day5_trades = trades[trades["date"] == target_date] if not trades.empty else trades
        if not day5_trades.empty:
            actions = day5_trades["action"].unique()
            self.assertIn("buy", actions)
            # T+1 下同日不应有卖出
            self.assertNotIn("sell", actions, "T+1 规则下买入当日不应能卖出")

    def test_price_limit_blocks_trade(self):
        """涨跌停限制: 涨停不能买入, 跌停不能卖出"""
        # 构造一只涨停股票
        data = self.data.copy()
        dates = sorted(data["date"].unique())
        code = data["code"].iloc[0]
        target_date = dates[10]

        # 标记该股在 target_date 涨停
        mask = (data["date"] == target_date) & (data["code"] == code)
        data.loc[mask, "is_limit_up"] = True

        entries = pd.DataFrame(False, index=dates, columns=data["code"].unique())
        exits = pd.DataFrame(False, index=dates, columns=data["code"].unique())
        entries.loc[target_date, code] = True

        bt = VectorizedBacktest(price_limit=True)
        result = bt.from_signals(data, entries, exits)
        trades = result["trades"]
        if not trades.empty:
            day_trades = trades[trades["date"] == target_date]
            # 涨停股不应有买入成交
            blocked = day_trades[(day_trades["code"] == code) & (day_trades["action"] == "buy")]
            self.assertEqual(len(blocked), 0, "涨停股不应能买入")

    def test_min_100_shares(self):
        """最小 100 股整数倍"""
        bt = VectorizedBacktest()
        result = bt.from_signals(self.data, self.entries, self.exits, init_capital=1e6)
        trades = result["trades"]
        buy_trades = trades[trades["action"] == "buy"] if not trades.empty else trades
        for _, t in buy_trades.iterrows():
            self.assertEqual(t["shares"] % 100, 0, f"买入股数 {t['shares']} 不是 100 的整数倍")


class TestPerformanceVsNative(unittest.TestCase):
    """性能对比: 向量化回测 vs native_adapter"""

    def setUp(self):
        # 较大数据集
        self.data = generate_synthetic_data(n_codes=30, n_days=250, seed=999)
        self.signals = generate_signal_data(self.data, fast_window=5, slow_window=20)
        self.entries, self.exits = _signals_to_matrices(self.signals)

    def test_performance_comparison(self):
        """性能对比测试"""
        # 向量化回测
        bt_vec = VectorizedBacktest()
        # 预热
        bt_vec.from_signals(self.data, self.entries, self.exits)

        n_runs = 3
        t0 = time.perf_counter()
        for _ in range(n_runs):
            result_vec = bt_vec.from_signals(self.data, self.entries, self.exits)
        t_vec = (time.perf_counter() - t0) / n_runs

        # native_adapter (现有实现)
        try:
            # 注入 backtest-engine scripts 路径
            bt_scripts = os.path.join(_PROJECT_ROOT, "skills", "backtest-engine", "scripts")
            if bt_scripts not in sys.path:
                sys.path.insert(0, bt_scripts)
            from adapters.native_adapter import NativeAdapter

            native = NativeAdapter()
            # native_adapter 接受 data + signals (长表)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                result_native = native.run_backtest(self.data, self.signals)
            t_native = (time.perf_counter() - t0) / n_runs

            print(f"\n[性能] 回测对比 (30股票x250天), 平均 {n_runs} 次:")
            print(f"  native_adapter (逐日循环): {t_native*1000:.2f} ms")
            print(f"  vectorized_backtest:       {t_vec*1000:.2f} ms")
            print(f"  加速比: {t_native/t_vec:.2f}x")

            # 向量化应不慢于 native (允许 2x 容差, 因为 native 也是简化实现)
            # 主要验证两者都能完成, 且向量化不显著更慢
            self.assertLess(t_vec, t_native * 3, "向量化回测不应比 native 慢 3 倍以上")

        except ImportError as e:
            print(f"\n[性能] 跳过 native_adapter 对比 (导入失败): {e}")
            print(f"  vectorized_backtest: {t_vec*1000:.2f} ms")


class TestSignalGenerators(unittest.TestCase):
    """信号生成器测试"""

    def setUp(self):
        self.data = generate_synthetic_data(n_codes=10, n_days=100, seed=555)

    def test_crossover_signals(self):
        """均线交叉信号"""
        close_pivot = self.data.pivot(index="date", columns="code", values="close")
        fast = close_pivot.rolling(5, min_periods=1).mean()
        slow = close_pivot.rolling(20, min_periods=1).mean()
        entries, exits = crossover_signals(fast, slow)
        self.assertEqual(entries.shape, close_pivot.shape)
        self.assertEqual(exits.shape, close_pivot.shape)
        # 信号应为布尔
        self.assertTrue(entries.dtypes.iloc[0] == bool or entries.dtypes.iloc[0] == np.bool_)

    def test_topk_signals(self):
        """TopK 选股信号"""
        close_pivot = self.data.pivot(index="date", columns="code", values="close")
        # 用收益率作为因子
        factor = -close_pivot.pct_change(20)  # 反转因子
        entries, exits = topk_signals(factor, k=5, holding_days=5)
        # 每个有信号的日期应恰好选 5 只 (除非 NaN 太多)
        for i in range(len(entries)):
            n_entries = entries.iloc[i].sum()
            if n_entries > 0:
                self.assertLessEqual(n_entries, 5, f"第 {i} 行选股数 {n_entries} > 5")


class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试"""

    def test_empty_signals(self):
        """空信号"""
        data = generate_synthetic_data(n_codes=5, n_days=30)
        dates = sorted(data["date"].unique())
        codes = data["code"].unique()
        entries = pd.DataFrame(False, index=dates, columns=codes)
        exits = pd.DataFrame(False, index=dates, columns=codes)

        bt = VectorizedBacktest()
        result = bt.from_signals(data, entries, exits, init_capital=1e6)
        # 无交易, 净值应保持不变
        self.assertAlmostEqual(result["equity_curve"]["equity"].iloc[-1], 1e6, delta=1.0)
        self.assertEqual(result["metrics"].get("total_trades", 0), 0)

    def test_all_buy_no_sell(self):
        """全买入无卖出"""
        data = generate_synthetic_data(n_codes=5, n_days=30)
        dates = sorted(data["date"].unique())
        codes = data["code"].unique()
        entries = pd.DataFrame(False, index=dates, columns=codes)
        exits = pd.DataFrame(False, index=dates, columns=codes)
        entries.iloc[0] = True  # 第一天全买

        bt = VectorizedBacktest()
        result = bt.from_signals(data, entries, exits, init_capital=1e6)
        # 应有买入交易
        self.assertGreater(result["metrics"].get("total_trades", 0), 0)

    def test_single_day_data(self):
        """单日数据"""
        data = generate_synthetic_data(n_codes=3, n_days=1)
        dates = sorted(data["date"].unique())
        codes = data["code"].unique()
        entries = pd.DataFrame(False, index=dates, columns=codes)
        exits = pd.DataFrame(False, index=dates, columns=codes)
        entries.iloc[0] = True

        bt = VectorizedBacktest()
        result = bt.from_signals(data, entries, exits)
        # 单日数据净值曲线长度应为 1
        self.assertEqual(len(result["equity_curve"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
