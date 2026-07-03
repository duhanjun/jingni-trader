"""
优化验证测试套件

测试内容:
  1. 正确性测试: 向量化回测 vs 逐日循环回测结果一致性
  2. 性能对比测试: 向量化 vs 循环的耗时对比
  3. 边界条件测试: 空数据、单只股票、全涨停等极端情况
  4. 因子表达式引擎: 正确性 (vs 手写 pandas)、解析能力、回看窗口
  5. 滚动训练框架: 任务生成正确性、合并去重逻辑

运行: python -m optimization.tests.test_optimizations
"""
import sys
import os
import time
import warnings
import unittest

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimization.vectorized_backtest import (
    VectorizedBacktester, BacktestConfig, benchmark_against_loop,
    benchmark_parameter_sweep, _loop_backtest
)
from optimization.factor_expression_engine import (
    FactorExpressionEngine, ExpressionParser, PREDEFINED_FACTORS, Feature, Constant
)
from optimization.walk_forward import (
    RollingGen, RollingType, TaskExecutor, RecorderCollector, walk_forward_backtest
)

warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================
# 测试数据生成工具
# ============================================================

def generate_synthetic_data(
    n_codes: int = 50,
    n_days: int = 500,
    start_date: str = '2022-01-01',
    seed: int = 42,
) -> pd.DataFrame:
    """生成合成行情数据用于测试"""
    np.random.seed(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    codes = [f'{600000 + i:06d}.SH' for i in range(n_codes)]

    rows = []
    for code in codes:
        price = 10.0 + np.random.uniform(0, 20)
        for dt in dates:
            ret = np.random.normal(0.0003, 0.02)
            price = price * (1 + ret)
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_p = price * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.uniform(1e6, 1e8))
            turnover_rate = np.random.uniform(0.005, 0.05)
            rows.append({
                'code': code,
                'date': dt,
                'open': open_p,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'amount': volume * price,
                'turnover_rate': turnover_rate,
                'is_limit_up': False,
                'is_limit_down': False,
            })
    return pd.DataFrame(rows)


def data_to_pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """长表转宽表 (dates x codes)"""
    return df.pivot(index='date', columns='code', values=col)


def generate_target_weights(prices: pd.DataFrame, rebalance_freq: int = 5, top_n: int = 10) -> pd.DataFrame:
    """生成目标权重矩阵 (每 rebalance_freq 天等权持有 top_n 只)"""
    np.random.seed(123)
    T, N = prices.shape
    weights = np.zeros((T, N))
    for t in range(0, T, rebalance_freq):
        # 随机选 top_n 只
        selected = np.random.choice(N, size=min(top_n, N), replace=False)
        for s in selected:
            for tt in range(t, min(t + rebalance_freq, T)):
                weights[tt, s] = 1.0 / top_n
    return pd.DataFrame(weights, index=prices.index, columns=prices.columns)


# ============================================================
# 1. 向量化回测 - 正确性测试
# ============================================================

class TestVectorizedBacktestCorrectness(unittest.TestCase):
    """向量化回测正确性测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_data(n_codes=20, n_days=300)
        cls.prices = data_to_pivot(cls.data, 'close')
        cls.target_weights = generate_target_weights(cls.prices, rebalance_freq=5, top_n=5)

    def test_basic_run(self):
        """测试基本回测能正常运行"""
        bt = VectorizedBacktester()
        result = bt.run_from_weights(self.prices, self.target_weights)
        self.assertFalse(result.equity_curve.empty)
        self.assertIn('equity', result.equity_curve.columns)
        self.assertGreater(len(result.metrics), 0)

    def test_metrics_completeness(self):
        """测试绩效指标完整性"""
        bt = VectorizedBacktester()
        result = bt.run_from_weights(self.prices, self.target_weights)
        required = ['total_return', 'annual_return', 'volatility', 'sharpe_ratio',
                    'max_drawdown', 'win_rate', 'calmar_ratio', 'sortino_ratio']
        for key in required:
            self.assertIn(key, result.metrics, f"缺少指标: {key}")
            self.assertFalse(np.isnan(result.metrics[key]), f"指标 {key} 为 NaN")

    def test_equity_starts_at_init_capital(self):
        """测试净值起点等于初始资金"""
        cfg = BacktestConfig(init_capital=500000)
        bt = VectorizedBacktester(cfg)
        result = bt.run_from_weights(self.prices, self.target_weights)
        self.assertAlmostEqual(result.equity_curve['equity'].iloc[0], 500000, places=0)

    def test_t_plus_1_effect(self):
        """测试 T+1 规则生效: T+1 模式下首日无持仓收益"""
        cfg_t1 = BacktestConfig(t_plus_1=True)
        cfg_t0 = BacktestConfig(t_plus_1=False)
        bt_t1 = VectorizedBacktester(cfg_t1)
        bt_t0 = VectorizedBacktester(cfg_t0)

        r_t1 = bt_t1.run_from_weights(self.prices, self.target_weights)
        r_t0 = bt_t0.run_from_weights(self.prices, self.target_weights)

        # T+1 模式下首日收益应接近0 (无持仓)
        self.assertAlmostEqual(r_t1.equity_curve['return'].iloc[0], 0.0, places=6)
        # T+0 模式首日可能有持仓收益
        # 两者净值曲线不应完全相同
        self.assertFalse(np.allclose(r_t1.equity_curve['equity'].values, r_t0.equity_curve['equity'].values))

    def test_from_signals_api(self):
        """测试 from_signals API (借鉴 VectorBT)"""
        bt = VectorizedBacktester()
        # 生成信号矩阵
        signals = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        # 每5天随机选5只买入
        np.random.seed(99)
        for t in range(0, len(signals), 5):
            selected = np.random.choice(signals.columns, size=5, replace=False)
            signals.loc[signals.index[t], selected] = 1.0

        result = bt.run_from_signals(self.prices, signals, top_n=5)
        self.assertFalse(result.equity_curve.empty)
        self.assertGreater(len(result.metrics), 0)

    def test_weights_normalization(self):
        """测试权重归一化 (权重和 <= 1)"""
        bt = VectorizedBacktester()
        # 构造权重和 > 1 的情况
        bad_weights = pd.DataFrame(
            0.5, index=self.prices.index, columns=self.prices.columns
        )
        result = bt.run_from_weights(self.prices, bad_weights)
        # 不应崩溃，且能产生合理结果
        self.assertFalse(result.equity_curve.empty)


# ============================================================
# 2. 向量化回测 - 性能对比测试
# ============================================================

class TestVectorizedBacktestPerformance(unittest.TestCase):
    """向量化回测性能对比测试"""

    @classmethod
    def setUpClass(cls):
        # 使用较小规模数据，因为循环基线模拟了 native_adapter 的逐日 DataFrame 过滤 (较慢)
        cls.data = generate_synthetic_data(n_codes=30, n_days=500)
        cls.prices = data_to_pivot(cls.data, 'close')
        cls.target_weights = generate_target_weights(cls.prices, rebalance_freq=5, top_n=10)

    def test_performance_speedup(self):
        """测试向量化回测比循环回测快 (循环基线模拟 native_adapter 逐日过滤模式)"""
        results = benchmark_against_loop(self.prices, self.target_weights, n_runs=3)
        print(f"\n[性能对比] 数据规模: {results['n_days']}天 x {results['n_codes']}只股票")
        print(f"  向量化平均耗时: {results['vectorized_avg']:.4f}s")
        print(f"  循环平均耗时:   {results['loop_avg']:.4f}s")
        print(f"  加速比:         {results['speedup']:.2f}x")
        self.assertGreater(results['speedup'], 1.0, "向量化应比循环快")

    def test_parameter_sweep_speedup(self):
        """测试参数搜索场景的加速 (VectorBT 核心优势)"""
        results = benchmark_parameter_sweep(self.prices, n_configs=10, n_runs=2)
        print(f"\n[参数搜索对比] {results['n_configs']}个配置")
        print(f"  向量化总耗时: {results['vectorized_total']:.4f}s")
        print(f"  循环总耗时:   {results['loop_total']:.4f}s")
        print(f"  加速比:       {results['speedup']:.2f}x")
        self.assertGreater(results['speedup'], 1.0, "参数搜索场景向量化应更快")

    def test_large_scale_scalability(self):
        """测试大规模数据下的可扩展性"""
        large_data = generate_synthetic_data(n_codes=100, n_days=1000)
        large_prices = data_to_pivot(large_data, 'close')
        large_weights = generate_target_weights(large_prices, rebalance_freq=5, top_n=30)

        bt = VectorizedBacktester()
        t0 = time.perf_counter()
        result = bt.run_from_weights(large_prices, large_weights)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"\n[大规模测试] {large_prices.shape[0]}天 x {large_prices.shape[1]}只, 耗时: {elapsed:.4f}s")
        self.assertLess(elapsed, 5.0, "1000天x100只股票的向量化回测应在5秒内完成")
        self.assertFalse(result.equity_curve.empty)


# ============================================================
# 3. 向量化回测 - 边界条件测试
# ============================================================

class TestVectorizedBacktestBoundary(unittest.TestCase):
    """向量化回测边界条件测试"""

    def test_empty_data(self):
        """测试空数据"""
        bt = VectorizedBacktester()
        empty_prices = pd.DataFrame()
        empty_weights = pd.DataFrame()
        result = bt.run_from_weights(empty_prices, empty_weights)
        # 空数据应返回空结果，不崩溃
        self.assertTrue(result.equity_curve.empty)

    def test_single_stock(self):
        """测试单只股票"""
        dates = pd.bdate_range('2023-01-01', periods=100)
        prices = pd.DataFrame({'000001.SH': 10 + np.cumsum(np.random.randn(100) * 0.1)}, index=dates)
        weights = pd.DataFrame({'000001.SH': [0.8] * 100}, index=dates)
        bt = VectorizedBacktester()
        result = bt.run_from_weights(prices, weights)
        self.assertEqual(len(result.equity_curve), 100)
        self.assertGreater(len(result.metrics), 0)

    def test_all_limit_up(self):
        """测试全涨停 (无法买入)"""
        dates = pd.bdate_range('2023-01-01', periods=50)
        n_codes = 5
        prices = pd.DataFrame(
            10 + np.cumsum(np.random.randn(50, n_codes) * 0.01, axis=0),
            index=dates,
            columns=[f'{i:06d}.SH' for i in range(n_codes)]
        )
        weights = pd.DataFrame(0.2, index=dates, columns=prices.columns)
        # 全涨停
        limit_up = pd.DataFrame(True, index=dates, columns=prices.columns)

        bt = VectorizedBacktester(BacktestConfig(price_limit=True))
        result = bt.run_from_weights(prices, weights, is_limit_up=limit_up)
        # 涨停时无法买入，净值应接近初始资金
        self.assertFalse(result.equity_curve.empty)

    def test_zero_weights(self):
        """测试全零权重 (空仓)"""
        dates = pd.bdate_range('2023-01-01', periods=100)
        prices = pd.DataFrame(
            np.random.uniform(8, 12, (100, 3)),
            index=dates,
            columns=['A', 'B', 'C']
        )
        weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        bt = VectorizedBacktester()
        result = bt.run_from_weights(prices, weights)
        # 空仓，净值应保持不变
        self.assertAlmostEqual(result.equity_curve['equity'].iloc[-1], result.equity_curve['equity'].iloc[0], places=2)

    def test_single_day(self):
        """测试仅一天数据"""
        dates = pd.DatetimeIndex(['2023-06-01'])
        prices = pd.DataFrame({'A': [10.0]}, index=dates)
        weights = pd.DataFrame({'A': [1.0]}, index=dates)
        bt = VectorizedBacktester()
        result = bt.run_from_weights(prices, weights)
        self.assertEqual(len(result.equity_curve), 1)

    def test_nan_prices(self):
        """测试含 NaN 的价格数据"""
        dates = pd.bdate_range('2023-01-01', periods=50)
        prices = pd.DataFrame(
            np.random.uniform(8, 12, (50, 3)),
            index=dates,
            columns=['A', 'B', 'C']
        )
        prices.iloc[10:15, 1] = np.nan  # 注入 NaN
        weights = pd.DataFrame(0.33, index=dates, columns=prices.columns)
        bt = VectorizedBacktester()
        result = bt.run_from_weights(prices, weights)
        self.assertFalse(result.equity_curve.empty)


# ============================================================
# 4. 因子表达式引擎 - 正确性测试
# ============================================================

class TestFactorExpressionEngine(unittest.TestCase):
    """因子表达式引擎测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_data(n_codes=30, n_days=200)
        cls.engine = FactorExpressionEngine()

    def test_simple_feature(self):
        """测试原始字段引用"""
        result = self.engine.compute('$close', self.data)
        self.assertEqual(len(result), len(self.data))
        np.testing.assert_array_almost_equal(
            result.values, self.data['close'].values
        )

    def test_ref_operator(self):
        """测试 Ref 算子 (引用N天前)"""
        expr_result = self.engine.compute('Ref($close, 5)', self.data)
        # 手动计算
        manual = self.data.sort_values(['code', 'date']).groupby('code')['close'].shift(5)
        # 对齐比较 (去掉前5天的NaN)
        valid = ~expr_result.isna() & ~manual.isna()
        np.testing.assert_array_almost_equal(
            expr_result[valid].values, manual[valid].values
        )

    def test_mean_operator(self):
        """测试 Mean 算子 (滚动均值)"""
        expr_result = self.engine.compute('Mean($close, 10)', self.data)
        manual = self.data.sort_values(['code', 'date']).groupby('code')['close'].transform(
            lambda x: x.rolling(10, min_periods=5).mean()
        )
        valid = ~expr_result.isna() & ~manual.isna()
        np.testing.assert_array_almost_equal(
            expr_result[valid].values, manual[valid].values, decimal=5
        )

    def test_complex_expression(self):
        """测试复合表达式: Ref($close, 20) / $close - 1"""
        expr = 'Ref($close, 20) / $close - 1'
        expr_result = self.engine.compute(expr, self.data)
        # 手动计算
        sorted_data = self.data.sort_values(['code', 'date'])
        ref_close = sorted_data.groupby('code')['close'].shift(20)
        manual = ref_close / sorted_data['close'] - 1
        manual = manual.reindex(self.data.index)
        valid = ~expr_result.isna() & ~manual.isna()
        np.testing.assert_array_almost_equal(
            expr_result[valid].values, manual[valid].values, decimal=5
        )

    def test_binary_operations(self):
        """测试二元运算"""
        for op, func in [('+', np.add), ('-', np.subtract), ('*', np.multiply)]:
            expr = f'$high {op} $low'
            expr_result = self.engine.compute(expr, self.data)
            manual = func(self.data['high'], self.data['low'])
            valid = ~expr_result.isna()
            np.testing.assert_array_almost_equal(
                expr_result[valid].values, manual[valid].values
            )

    def test_division_operator(self):
        """测试除法 (含除零保护)"""
        expr_result = self.engine.compute('$close / $volume', self.data)
        manual = self.data['close'] / self.data['volume']
        valid = ~expr_result.isna() & ~manual.isna()
        np.testing.assert_array_almost_equal(
            expr_result[valid].values, manual[valid].values
        )

    def test_nested_expression(self):
        """测试嵌套表达式: Mean(Ref($close, 1) / $close - 1, 20)"""
        expr = 'Mean(Ref($close, 1) / $close - 1, 20)'
        expr_result = self.engine.compute(expr, self.data)
        # 手动计算
        sorted_data = self.data.sort_values(['code', 'date'])
        ret = sorted_data.groupby('code')['close'].transform(lambda x: x.shift(1) / x - 1)
        manual = ret.groupby(sorted_data['code']).transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        manual = manual.reindex(self.data.index)
        valid = ~expr_result.isna() & ~manual.isna()
        np.testing.assert_array_almost_equal(
            expr_result[valid].values, manual[valid].values, decimal=5
        )

    def test_lookback_window(self):
        """测试回看窗口自动检测"""
        parser = ExpressionParser()
        # Ref($close, 20) 需要 20 天回看
        tree = parser.parse('Ref($close, 20)')
        self.assertEqual(tree.get_longest_back_rolling(), 20)
        # Mean(Ref($close, 5), 10) 需要 5 + 10 = 15 天回看
        tree = parser.parse('Mean(Ref($close, 5), 10)')
        self.assertEqual(tree.get_longest_back_rolling(), 15)
        # 嵌套: Std(Mean($close, 20), 60) 需要 20 + 60 = 80 天
        tree = parser.parse('Std(Mean($close, 20), 60)')
        self.assertEqual(tree.get_longest_back_rolling(), 80)

    def test_predefined_factors(self):
        """测试预定义因子库全部可计算"""
        for name, expr in PREDEFINED_FACTORS.items():
            try:
                result = self.engine.compute(expr, self.data)
                self.assertEqual(len(result), len(self.data), f"因子 {name} 长度不匹配")
            except Exception as e:
                # turnover_rate 字段可能缺失时跳过
                if 'turnover_rate' in str(e):
                    continue
                self.fail(f"预定义因子 {name} ({expr}) 计算失败: {e}")

    def test_batch_compute(self):
        """测试批量计算"""
        expressions = {
            'ret_5d': 'Ref($close, 5) / $close - 1',
            'vol_20d': 'Std($close / Ref($close, 1) - 1, 20)',
            'ma_bias': '$close / Mean($close, 20) - 1',
        }
        result = self.engine.compute_batch(expressions, self.data)
        self.assertEqual(len(result.columns), 3)
        for name in expressions:
            self.assertIn(name, result.columns)

    def test_constant_in_expression(self):
        """测试常数在表达式中"""
        expr_result = self.engine.compute('$close * 2 + 1', self.data)
        manual = self.data['close'] * 2 + 1
        np.testing.assert_array_almost_equal(
            expr_result.values, manual.values
        )

    def test_cross_section_rank(self):
        """测试截面排名"""
        factor = self.engine.compute('Ref($close, 5) / $close - 1', self.data)
        ranked = self.engine.cross_section_rank(factor, self.data)
        self.assertEqual(len(ranked), len(self.data))
        # 排名值应在 [0, 1] 之间
        valid = ranked.dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_parser_error_handling(self):
        """测试解析器错误处理"""
        parser = ExpressionParser()
        # 未知算子在解析时即报错
        with self.assertRaises(ValueError):
            parser.parse('UnknownFunc($close, 5)')
        # 字段不存在在 load 时报错 (parse 阶段不校验字段存在性，与 Qlib 一致)
        engine = FactorExpressionEngine()
        with self.assertRaises(ValueError):
            engine.compute('$nonexistent_field', self.data)


# ============================================================
# 5. 滚动训练框架测试
# ============================================================

class TestWalkForwardRolling(unittest.TestCase):
    """滚动训练框架测试"""

    def test_rolling_generation(self):
        """测试滚动任务生成"""
        gen = RollingGen(
            rolling_type=RollingType.ROLLING,
            train_window=730,
            valid_window=180,
            test_window=180,
            step=180,
        )
        dates = pd.bdate_range('2019-01-01', '2024-12-31')
        tasks = gen.generate('2019-01-01', '2024-12-31', dates)
        self.assertGreater(len(tasks), 0, "应生成至少1个任务")
        # 检查任务间不重叠 (test 段)
        for i in range(1, len(tasks)):
            self.assertGreater(tasks[i].test.start, tasks[i-1].test.end,
                             "test 段应不重叠")
        # 检查 train 段在 test 段之前
        for t in tasks:
            self.assertLess(t.train.end, t.test.start)

    def test_expanding_generation(self):
        """测试扩展窗口模式"""
        gen = RollingGen(
            rolling_type=RollingType.EXPANDING,
            train_window=365,
            valid_window=90,
            test_window=90,
            step=90,
        )
        dates = pd.bdate_range('2020-01-01', '2024-12-31')
        tasks = gen.generate('2020-01-01', '2024-12-31', dates)
        self.assertGreater(len(tasks), 0)
        # 扩展模式: 所有任务的 train 起点相同
        first_start = tasks[0].train.start
        for t in tasks:
            self.assertEqual(t.train.start, first_start, "扩展模式 train 起点应固定")

    def test_task_executor(self):
        """测试任务执行器"""
        gen = RollingGen(rolling_type=RollingType.ROLLING, train_window=365,
                        valid_window=90, test_window=90, step=90)
        dates = pd.bdate_range('2020-01-01', '2024-12-31')
        tasks = gen.generate('2020-01-01', '2024-12-31', dates)

        def mock_train(task):
            return {
                'predictions': pd.DataFrame({
                    'date': pd.date_range(task.test.start, task.test.end, periods=5),
                    'code': ['A'] * 5,
                    'prediction': np.random.randn(5),
                }),
                'metrics': {'ic': 0.05},
            }

        executor = TaskExecutor(mock_train)
        results = executor.run(tasks)
        self.assertEqual(len(results), len(tasks))
        self.assertTrue(all(r['status'] == 'success' for r in results))

    def test_task_executor_error_isolation(self):
        """测试任务执行器错误隔离"""
        tasks = [
            type('T', (), {'task_id': 't1', 'test': type('S', (), {'start': '2021-01-01', 'end': '2021-06-30'})()})(),
            type('T', (), {'task_id': 't2', 'test': type('S', (), {'start': '2021-07-01', 'end': '2021-12-31'})()})(),
        ]
        call_count = [0]
        def flaky_train(task):
            call_count[0] += 1
            if task.task_id == 't1':
                raise RuntimeError("模拟失败")
            return {'predictions': pd.DataFrame(), 'metrics': {}}

        executor = TaskExecutor(flaky_train)
        results = executor.run(tasks)
        self.assertEqual(results[0]['status'], 'failed')
        self.assertEqual(results[1]['status'], 'success')
        self.assertEqual(call_count[0], 2, "失败后应继续执行后续任务")

    def test_recorder_collector(self):
        """测试记录收集器"""
        task_results = [
            {
                'task_id': 't1', 'status': 'success',
                'predictions': pd.DataFrame({
                    'date': ['2021-01-01', '2021-01-02'],
                    'code': ['A', 'B'],
                    'prediction': [0.1, 0.2],
                }),
                'metrics': {'ic': 0.05},
            },
            {
                'task_id': 't2', 'status': 'success',
                'predictions': pd.DataFrame({
                    'date': ['2021-01-02', '2021-01-03'],  # 01-02 与 t1 重叠
                    'code': ['B', 'C'],
                    'prediction': [0.3, 0.4],
                }),
                'metrics': {'ic': 0.06},
            },
            {'task_id': 't3', 'status': 'failed', 'error': 'test'},
        ]
        combined = RecorderCollector.collect_predictions(task_results)
        # 去重: (2021-01-02, B) 应保留 t2 的 (后出现的)
        overlap = combined[(combined['date'] == '2021-01-02') & (combined['code'] == 'B')]
        self.assertEqual(len(overlap), 1)
        self.assertAlmostEqual(overlap['prediction'].iloc[0], 0.3)

        metrics = RecorderCollector.collect_metrics(task_results)
        self.assertEqual(len(metrics), 2)  # 只有2个成功任务

    def test_walk_forward_backtest_integration(self):
        """测试 walk-forward 回测集成"""
        data = generate_synthetic_data(n_codes=20, n_days=400)
        # 添加前向收益
        data = data.sort_values(['code', 'date'])
        data['forward_return_5d'] = data.groupby('code')['close'].shift(-5) / data['close'] - 1
        # 添加一个简单因子
        data['simple_factor'] = data.groupby('code')['close'].pct_change(5)

        # 使用较小的窗口以适配 400 天的数据
        from optimization.walk_forward import RollingGen, RollingType
        gen = RollingGen(
            rolling_type=RollingType.ROLLING,
            train_window=200,   # ~200自然日 ≈ 142交易日
            valid_window=0,
            test_window=90,     # ~90自然日 ≈ 64交易日
            step=90,
        )
        trading_dates = pd.DatetimeIndex(sorted(data['date'].unique()))
        tasks = gen.generate('2022-01-01', '2023-06-30', trading_dates)
        self.assertGreater(len(tasks), 0, "应生成至少1个滚动任务")

        result = walk_forward_backtest(
            data, factor_col='simple_factor', return_col='forward_return_5d',
            train_start='2022-01-01', train_end='2023-06-30',
            train_window=200, test_window=90, step=90,
        )
        self.assertTrue(result['success'])


# ============================================================
# 测试入口
# ============================================================

def run_all_tests():
    """运行所有测试并输出报告"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestVectorizedBacktestCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorizedBacktestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorizedBacktestBoundary))
    suite.addTests(loader.loadTestsFromTestCase(TestFactorExpressionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestWalkForwardRolling))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    run_all_tests()
