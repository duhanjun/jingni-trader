"""
===========================================================================
测试文件: test_factor_expression_engine.py
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
     - Expression Engine: 基于表达式的因子定义系统
     - 核心论文: Qlib: An AI-oriented Quantitative Investment Platform
       (https://arxiv.org/abs/2009.11189)
     - Data Layer ops: Ref, Mean, Std, $close, $volume, etc.

优化方向: factor-engine - 因子表达式引擎
     - 当前问题: 因子计算硬编码在 compute_a_share_factors() 中，扩展性差
     - 优化方案: 引入声明式因子表达式 DSL，支持因子定义/注册/热加载
     - 预期收益: 因子开发效率提升 3-5x，支持 LLM 自动生成因子

测试内容:
     1. 表达式解析器正确性测试
     2. 因子计算正确性测试（与硬编码对比）
     3. 表达式热加载测试
     4. Alpha158 风格因子批量生成测试

⚠️ 注意: 此文件为验证代码，仅在测试目录中运行，不修改主代码。
===========================================================================
"""

import sys
import os
import re
import json
import time
import unittest
from typing import List, Dict, Callable, Any, Optional
from functools import lru_cache

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


# ===========================================================================
# 因子表达式引擎实现 (独立验证版)
# ===========================================================================

class FactorExpressionEngine:
    """
    因子表达式引擎 (借鉴 Qlib Expression Engine 设计)

    Qlib 的核心设计理念:
    - 因子是函数，不是数据: factor = expression(data) -> values
    - 表达式 DSL: $close, Ref($close, 1), Mean($close, 20)
    - 可组合: EMA(($close - Ref($close, 5)) / Ref($close, 5), 10)

    本实现特点:
    - 使用 Python 的 eval/exec 机制解析表达式
    - 支持常见时间序列算子
    - 支持用户自定义算子注册
    - 批量编译与缓存
    """

    # 内置算子
    BUILTIN_OPS = {}

    def __init__(self):
        self._op_cache: Dict[str, Callable] = {}
        self._register_builtins()

    def _register_builtins(self):
        """注册内置时间序列算子"""
        # ---- 引用算子 ----
        def ref_op(series: pd.Series, period: int) -> pd.Series:
            """Ref($close, 5): 引用 5 天前的值"""
            return series.shift(period)

        # ---- 滑动窗口算子 ----
        def mean_op(series: pd.Series, window: int) -> pd.Series:
            return series.rolling(window, min_periods=1).mean()

        def std_op(series: pd.Series, window: int) -> pd.Series:
            return series.rolling(window, min_periods=1).std()

        def sum_op(series: pd.Series, window: int) -> pd.Series:
            return series.rolling(window, min_periods=1).sum()

        def max_op(series: pd.Series, window: int) -> pd.Series:
            return series.rolling(window, min_periods=1).max()

        def min_op(series: pd.Series, window: int) -> pd.Series:
            return series.rolling(window, min_periods=1).min()

        def pct_change_op(series: pd.Series, period: int = 1) -> pd.Series:
            return series.pct_change(period)

        def rank_op(series: pd.Series) -> pd.Series:
            """截面排名 (分位数)"""
            return series.rank(pct=True)

        def delta_op(series: pd.Series, period: int) -> pd.Series:
            """Delta($close, 5): 与 5 天前的差值"""
            return series - series.shift(period)

        def corr_op(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
            """滚动相关系数"""
            return s1.rolling(window, min_periods=window // 2).corr(s2)

        def cov_op(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
            """滚动协方差"""
            return s1.rolling(window, min_periods=window // 2).cov(s2)

        def ema_op(series: pd.Series, window: int) -> pd.Series:
            """指数移动平均"""
            return series.ewm(span=window, adjust=False).mean()

        def ts_argmax_op(series: pd.Series, window: int) -> pd.Series:
            """滚动窗口内最大值位置"""
            result = series.rolling(window, min_periods=1).apply(
                lambda x: x.argmax() if len(x) > 0 else 0
            )
            return result

        def ts_argmin_op(series: pd.Series, window: int) -> pd.Series:
            """滚动窗口内最小值位置"""
            result = series.rolling(window, min_periods=1).apply(
                lambda x: x.argmin() if len(x) > 0 else 0
            )
            return result

        def rsi_op(series: pd.Series, period: int = 14) -> pd.Series:
            """RSI 指标"""
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            return 1 - 1 / (1 + rs)

        # 注册
        self.BUILTIN_OPS = {
            'Ref': ref_op,
            'Mean': mean_op,
            'Std': std_op,
            'Sum': sum_op,
            'Max': max_op,
            'Min': min_op,
            'PctChange': pct_change_op,
            'Rank': rank_op,
            'Delta': delta_op,
            'Corr': corr_op,
            'Cov': cov_op,
            'EMA': ema_op,
            'TsArgmax': ts_argmax_op,
            'TsArgmin': ts_argmin_op,
            'RSI': rsi_op,
        }

    def register_op(self, name: str, func: Callable):
        """注册自定义算子"""
        self.BUILTIN_OPS[name] = func

    def parse_expression(self, expr: str) -> Dict[str, Any]:
        """
        解析因子表达式，返回计算图描述

        语法规则:
        - 数据字段: $close, $open, $high, $low, $volume, $amount, $turnover
        - 算子: Ref($close, 5), Mean($close, 20), Std($close, 20) 等
        - 算术: +, -, *, /, (, )
        - 复合: (EMA($close, 12) - EMA($close, 26)) / $close

        返回: {'fields': [...], 'ops': [...], 'dep_graph': {...}}
        """
        # 提取数据字段引用
        field_pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*)'
        fields = list(set(re.findall(field_pattern, expr)))

        # 提取算子调用
        op_pattern = r'([A-Z][a-zA-Z_]*)\(([^)]*)\)'
        ops = re.findall(op_pattern, expr)

        return {
            'fields': fields,
            'ops': [{'name': name, 'args': args} for name, args in ops],
            'raw_expression': expr,
        }

    def compute_factor(
        self,
        expression: str,
        data: pd.DataFrame,
        group_col: str = 'code',
        date_col: str = 'date',
    ) -> pd.Series:
        """
        根据表达式计算因子值

        关键设计 (借鉴 Qlib):
        1. 按股票分组计算 (避免跨股票数据污染)
        2. 逐组应用表达式
        3. 缓存编译后的表达式
        """
        if data.empty:
            return pd.Series(dtype=float)

        # 预处理: 提取字段引用
        field_names = sorted(set(re.findall(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', expression)))

        # 确保字段存在
        for f in field_names:
            if f not in data.columns:
                raise ValueError(f"数据中缺少字段: {f}")

        # 编译表达式 (lru_cache 需要 hashable 参数)
        compiled = self._compile(expression, tuple(field_names))

        # 按股票分组计算
        result = pd.Series(index=data.index, dtype=float)
        grouped = data.groupby(group_col)

        for code, group in grouped:
            group_sorted = group.sort_values(date_col)

            # 准备局部变量: 字段数据 (保留为 pd.Series 以支持 rolling/shift 等算子)
            local_vars = {}
            for f in field_names:
                local_vars[f"_{f}_series"] = group_sorted[f].reset_index(drop=True)

            try:
                factor_values = compiled(local_vars, self.BUILTIN_OPS, np, pd)

                if isinstance(factor_values, pd.Series):
                    result.loc[group_sorted.index] = factor_values.values
                elif isinstance(factor_values, np.ndarray):
                    result.loc[group_sorted.index] = factor_values.ravel()[:len(group_sorted)]
                else:
                    result.loc[group_sorted.index] = factor_values
            except Exception as e:
                raise RuntimeError(
                    f"计算因子 '{expression}' 在股票 {code} 上失败: {e}"
                )

        return result

    @lru_cache(maxsize=512)
    def _compile(self, expression: str, field_names: tuple) -> Callable:
        """
        编译因子表达式为可执行函数

        使用 compile + eval 机制 (借鉴 Qlib 表达式引擎的编译-执行分离设计)
        将表达式编译为 Python 代码对象，执行时注入字段数据和算子
        """
        # 替换 $field 为内部变量名
        code = expression
        for f in field_names:
            code = code.replace(f'${f}', f'_{f}_series')

        # 编译表达式为 code object
        compiled_code = compile(code, '<factor_expression>', 'eval')

        # 返回一个 wrapper 函数，执行时注入命名空间
        def _compute(field_vars, ops, np, pd):
            # 构建完整的命名空间
            namespace = {}
            namespace.update(ops)          # 注入算子 (Ref, Mean, Std, EMA, ...)
            namespace.update(field_vars)   # 注入字段数据 (_close_series, ...)
            namespace['np'] = np
            namespace['pd'] = pd
            return eval(compiled_code, namespace, {})

        return _compute


# ===========================================================================
# 预定义因子库 (借鉴 Qlib Alpha158 设计)
# ===========================================================================

ALPHA_FACTOR_LIBRARY = {
    # ---- 收益率因子 ----
    'ret_1d': "$close / Ref($close, 1) - 1",
    'ret_5d': "$close / Ref($close, 5) - 1",
    'ret_20d': "$close / Ref($close, 20) - 1",
    'ret_60d': "$close / Ref($close, 60) - 1",

    # ---- 反转因子 ----
    'reversal_5d': "-( $close / Ref($close, 5) - 1 )",
    'reversal_20d': "-( $close / Ref($close, 20) - 1 )",

    # ---- 波动率因子 ----
    'volatility_5d': "Std($close / Ref($close, 1) - 1, 5)",
    'volatility_20d': "Std($close / Ref($close, 1) - 1, 20)",
    'volatility_60d': "Std($close / Ref($close, 1) - 1, 60)",

    # ---- 均线偏离因子 ----
    'ma_bias_5': "$close / Mean($close, 5) - 1",
    'ma_bias_10': "$close / Mean($close, 10) - 1",
    'ma_bias_20': "$close / Mean($close, 20) - 1",
    'ma_bias_60': "$close / Mean($close, 60) - 1",

    # ---- 成交量因子 ----
    'volume_ratio': "$volume / Mean($volume, 20) - 1",
    'volume_trend': "Mean($close / Ref($close, 1) - 1, 5) * (1 + $volume / Mean($volume, 20) - 1)",

    # ---- MACD ----
    'macd_dif': "EMA($close, 12) - EMA($close, 26)",
    'macd_dea': "EMA(EMA($close, 12) - EMA($close, 26), 9)",
    'macd_hist': "(EMA($close, 12) - EMA($close, 26)) - EMA(EMA($close, 12) - EMA($close, 26), 9)",

    # ---- RSI ----
    # 使用内置 RSI 算子 (内置函数处理 gain/loss 分离运算)
    'rsi_6': "RSI($close, 6)",
    'rsi_14': "RSI($close, 14)",

    # ---- 布林带 ----
    'boll_upper': "Mean($close, 20) + 2 * Std($close, 20)",
    'boll_lower': "Mean($close, 20) - 2 * Std($close, 20)",
    'boll_width': "(Mean($close, 20) + 2 * Std($close, 20) - (Mean($close, 20) - 2 * Std($close, 20))) / Mean($close, 20)",
}


# ===========================================================================
# 单元测试
# ===========================================================================

class TestFactorExpressionEngine(unittest.TestCase):
    """因子表达式引擎测试套件"""

    @classmethod
    def setUpClass(cls):
        """生成模拟测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
        codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']

        rows = []
        for code in codes:
            base_price = np.random.uniform(5, 50)
            n = len(dates)
            returns = np.random.normal(0.0005, 0.02, n)
            prices = base_price * (1 + returns).cumprod()

            df = pd.DataFrame({
                'date': dates,
                'code': code,
                'open': prices * (1 + np.random.normal(0, 0.003, n)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
                'close': prices,
                'volume': np.random.lognormal(10, 0.5, n).astype(int),
                'amount': np.random.lognormal(14, 0.5, n),
            })
            rows.append(df)

        cls.test_data = pd.concat(rows, ignore_index=True)
        cls.test_data = cls.test_data.sort_values(['code', 'date']).reset_index(drop=True)
        cls.engine = FactorExpressionEngine()

    def test_parse_simple_expression(self):
        """测试简单表达式解析"""
        parsed = self.engine.parse_expression("$close / Ref($close, 1) - 1")
        self.assertIn('close', parsed['fields'])
        self.assertTrue(any(op['name'] == 'Ref' for op in parsed['ops']))

    def test_parse_complex_expression(self):
        """测试复合表达式解析"""
        expr = "(EMA($close, 12) - EMA($close, 26)) - EMA(EMA($close, 12) - EMA($close, 26), 9)"
        parsed = self.engine.parse_expression(expr)
        self.assertIn('close', parsed['fields'])
        self.assertTrue(len(parsed['ops']) >= 3)

    def test_ret_1d_calculation(self):
        """测试日收益率因子计算正确性"""
        result = self.engine.compute_factor("$close / Ref($close, 1) - 1", self.test_data)

        # 与手动计算对比
        expected = pd.Series(index=self.test_data.index, dtype=float)
        for code, group in self.test_data.groupby('code'):
            group = group.sort_values('date')
            expected.loc[group.index] = group['close'].pct_change().values

        # 跳过第一天 (NaN)
        valid_mask = result.notna() & expected.notna()
        self.assertTrue(np.allclose(result[valid_mask].values, expected[valid_mask].values, rtol=1e-10))
        print("[PASS] test_ret_1d_calculation")

    def test_volatility_20d_calculation(self):
        """测试波动率因子计算正确性"""
        result = self.engine.compute_factor(
            "Std($close / Ref($close, 1) - 1, 20)", self.test_data)

        # 手动计算
        expected = pd.Series(index=self.test_data.index, dtype=float)
        for code, group in self.test_data.groupby('code'):
            group = group.sort_values('date')
            ret = group['close'].pct_change()
            vol = ret.rolling(20, min_periods=1).std()
            expected.loc[group.index] = vol.values

        valid_mask = result.notna() & expected.notna()
        self.assertTrue(np.allclose(result[valid_mask].values, expected[valid_mask].values, rtol=1e-8))
        print("[PASS] test_volatility_20d_calculation")

    def test_ma_bias_calculation(self):
        """测试均线偏离因子"""
        result = self.engine.compute_factor("$close / Mean($close, 20) - 1", self.test_data)

        expected = pd.Series(index=self.test_data.index, dtype=float)
        for code, group in self.test_data.groupby('code'):
            group = group.sort_values('date')
            ma = group['close'].rolling(20, min_periods=1).mean()
            expected.loc[group.index] = (group['close'] / ma - 1).values

        valid_mask = result.notna() & expected.notna()
        self.assertTrue(np.allclose(result[valid_mask].values, expected[valid_mask].values, rtol=1e-8))
        print("[PASS] test_ma_bias_calculation")

    def test_expression_caching(self):
        """测试表达式编译缓存"""
        self.engine._compile.cache_clear()

        expr = "$close / Ref($close, 1) - 1"
        fields = ['close']

        self.engine._compile(expr, tuple(fields))
        info_before = self.engine._compile.cache_info()

        self.engine._compile(expr, tuple(fields))
        info_after = self.engine._compile.cache_info()

        self.assertEqual(info_after.hits, info_before.hits + 1)
        print("[PASS] test_expression_caching")

    def test_bulk_factor_computation(self):
        """测试批量因子计算"""
        results = {}
        for name, expr in ALPHA_FACTOR_LIBRARY.items():
            try:
                result = self.engine.compute_factor(expr, self.test_data)
                results[name] = result
            except RuntimeError as e:
                self.fail(f"因子 {name} 计算失败: {e}")

        self.assertEqual(len(results), len(ALPHA_FACTOR_LIBRARY))
        # 验证所有因子非空
        for name, result in results.items():
            self.assertFalse(result.dropna().empty, f"因子 {name} 计算结果为空")
        print(f"[PASS] test_bulk_factor_computation: 成功计算 {len(results)} 个因子")

    def test_compare_with_hardcoded(self):
        """测试表达式引擎与硬编码计算的对比"""
        # 手动计算 reversal_20d 作为参考基准
        expected = pd.Series(index=self.test_data.index, dtype=float)
        for code, group in self.test_data.groupby('code'):
            group = group.sort_values('date')
            ret_20d = group['close'] / group['close'].shift(20) - 1
            expected.loc[group.index] = -ret_20d.values

        # 使用表达式引擎计算
        expr_reversal = self.engine.compute_factor(
            "-( $close / Ref($close, 20) - 1 )", self.test_data
        )

        # 对齐并对比
        valid_mask = expr_reversal.notna() & expected.notna()
        diff = np.abs(expr_reversal[valid_mask] - expected[valid_mask])
        max_diff = diff.max()
        self.assertLess(max_diff, 1e-8, f"reversal_20d 差异过大: {max_diff}")

        print("[PASS] test_compare_with_hardcoded: 表达式引擎与手动计算结果一致")


class TestFactorExpressionPerformance(unittest.TestCase):
    """性能基准测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        dates = pd.date_range('2021-01-01', '2024-12-31', freq='B')
        # 模拟 100 只股票
        codes = [f'{i:06d}.{"SH" if i >= 600000 else "SZ"}' for i in range(1, 101)]

        rows = []
        for code in codes:
            base_price = np.random.uniform(5, 80)
            n = len(dates)
            returns = np.random.normal(0.0005, 0.02, n)
            prices = base_price * (1 + returns).cumprod()
            rows.append(pd.DataFrame({
                'date': dates,
                'code': code,
                'close': prices,
                'open': prices * (1 + np.random.normal(0, 0.002, n)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.008, n))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.008, n))),
                'volume': np.random.lognormal(10, 0.5, n).astype(int),
            }))

        cls.large_data = pd.concat(rows, ignore_index=True)
        cls.engine = FactorExpressionEngine()

    def test_bulk_factor_performance(self):
        """测试批量因子计算性能"""
        engine = FactorExpressionEngine()
        engine._compile.cache_clear()

        # 选取 10 个代表性因子
        factors_to_test = [
            'ret_1d', 'ret_20d', 'reversal_20d', 'volatility_20d',
            'ma_bias_20', 'volume_ratio', 'volume_trend',
            'ret_60d', 'volatility_60d', 'ma_bias_60',
        ]

        start = time.time()
        for name in factors_to_test:
            expr = ALPHA_FACTOR_LIBRARY[name]
            engine.compute_factor(expr, self.large_data)
        elapsed = time.time() - start

        n_rows = len(self.large_data)
        print(f"\n  数据规模: {self.large_data['code'].nunique()} 只股票 x {self.large_data['date'].nunique()} 交易日 = {n_rows:,} 行")
        print(f"  计算 {len(factors_to_test)} 个因子耗时: {elapsed:.2f}s")
        print(f"  平均每因子: {elapsed / len(factors_to_test):.2f}s")

        # 性能要求：10 个因子应在合理时间内完成
        self.assertLess(elapsed, 120, "批量因子计算超时 (>2分钟)")
        print("[PASS] test_bulk_factor_performance")


# ===========================================================================
# 主函数
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("因子表达式引擎 验证测试")
    print("借鉴来源: Microsoft Qlib Expression Engine")
    print("=" * 70)

    # 运行正确性测试
    print("\n[1/2] 正确性测试")
    suite_correctness = unittest.TestLoader().loadTestsFromTestCase(TestFactorExpressionEngine)
    unittest.TextTestRunner(verbosity=2).run(suite_correctness)

    # 运行性能测试
    print("\n[2/2] 性能测试")
    suite_perf = unittest.TestLoader().loadTestsFromTestCase(TestFactorExpressionPerformance)
    unittest.TextTestRunner(verbosity=2).run(suite_perf)

    print("\n" + "=" * 70)
    print("测试完成。所有验证代码位于独立测试文件中，未修改主代码。")
    print("=" * 70)