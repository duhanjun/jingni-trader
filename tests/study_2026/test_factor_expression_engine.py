"""
验证代码：声明式因子表达式引擎
============================================
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
优化方向: factor-engine —— 因子库可扩展性
核心思路: Qlib 的 Expression Engine 通过字符串表达式 "$close / Ref($close, 20) - 1"
         声明式定义因子，而非在当前代码中硬编码因子计算逻辑。
         这种模式大幅提升因子的可扩展性和用户友好度。
日期: 2026-06-12

约束: 仅验证可行性，不可直接修改主代码，不可执行 git commit/merge。
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from collections import OrderedDict
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════
# 1. 因子表达式引擎原型（借鉴 Qlib Expression Engine）
# ═══════════════════════════════════════════

class FactorExpressionEngine:
    """
    声明式因子表达式引擎

    支持的操作符:
        $close, $open, $high, $low, $volume, $amount, $vwap
        Ref(expr, N)          — 前 N 期值
        Mean(expr, N)         — N 期均值
        Std(expr, N)          — N 期标准差
        Rank(expr)            — 横截面排名 (分位数)
        Delay(expr, N)        — 延迟 N 期（避免前视偏差）
        Delta(expr, N)        — N 期差分
        +, -, *, /, 算术运算
        Log(expr), Abs(expr), Sign(expr)

    示例表达式:
        "($close / Ref($close, 20) - 1)"                        → 20日动量（反转因子）
        "Mean($volume, 20)"                                      → 20日均量
        "-1 * ($close / Ref($close, 60) - 1)"                    → 60日反转因子
        "($close / Ref($close, 20) - 1) / Std($close, 20)"      → 波动率调整动量
        "Log($amount)"                                           → 对数成交额
    """

    # 内置字段映射
    FIELD_MAP = {
        '$close': 'close',
        '$open': 'open',
        '$high': 'high',
        '$low': 'low',
        '$volume': 'volume',
        '$amount': 'amount',
    }

    def __init__(self):
        self._registered_factors: Dict[str, str] = OrderedDict()

    def register_factor(self, name: str, expression: str) -> None:
        """注册一个因子表达式"""
        self._registered_factors[name] = expression

    def register_factors(self, factors: Dict[str, str]) -> None:
        """批量注册因子表达式"""
        for name, expr in factors.items():
            self.register_factor(name, expr)

    def list_factors(self) -> List[str]:
        """列出所有已注册因子"""
        return list(self._registered_factors.keys())

    def evaluate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有已注册因子

        参数:
            data: 包含 OHLCV 的 DataFrame（必须按 code, date 排序，含 code, date 列）

        返回:
            factor_df: code, date, [各因子列]
        """
        if data.empty or 'code' not in data.columns:
            return pd.DataFrame()

        data = data.sort_values(['code', 'date']).copy()
        result = data[['code', 'date']].copy()

        for name, expr in self._registered_factors.items():
            try:
                result[name] = self._evaluate_expression(data, expr)
            except Exception as e:
                raise RuntimeError(f"计算因子 '{name}' 时出错 (表达式: {expr}): {e}")

        return result

    def _evaluate_expression(self, data: pd.DataFrame, expr: str) -> pd.Series:
        """递归求值表达式（简单版 AST 解析器）"""
        expr = expr.strip()

        # 处理函数调用:  Ref(expr, N), Mean(expr, N), Std(expr, N)
        # 先处理最外层函数
        for func_name in ['Ref', 'Mean', 'Std', 'Delay', 'Delta', 'Rank', 'Log', 'Abs', 'Sign']:
            if expr.startswith(func_name + '(') and expr.endswith(')'):
                return self._eval_function(data, func_name, expr)

        # 处理括号
        if expr.startswith('(') and expr.endswith(')'):
            # 找到匹配的括号
            depth = 0
            for i, ch in enumerate(expr):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                if depth == 0 and i == len(expr) - 1:
                    return self._evaluate_expression(data, expr[1:-1])
                elif depth == 0:
                    break

        # 处理加减法（优先级最低的二元运算）
        for op, func in [('+', lambda a, b: a + b), ('-', lambda a, b: a - b)]:
            parts = self._split_at_top_level(expr, op)
            if len(parts) == 2:
                left = self._evaluate_expression(data, parts[0])
                right = self._evaluate_expression(data, parts[1])
                return func(left, right)

        # 处理乘除法
        for op, func in [('*', lambda a, b: a * b), ('/', lambda a, b: a / b.replace(0, np.nan))]:
            parts = self._split_at_top_level(expr, op)
            if len(parts) == 2:
                left = self._evaluate_expression(data, parts[0])
                right = self._evaluate_expression(data, parts[1])
                return func(left, right)

        # 处理负号
        if expr.startswith('-'):
            inner = self._evaluate_expression(data, expr[1:])
            return -inner

        # 处理字面数字
        try:
            val = float(expr)
            return pd.Series(val, index=data.index)
        except ValueError:
            pass

        # 处理字段引用
        if expr in self.FIELD_MAP:
            col = self.FIELD_MAP[expr]
            if col not in data.columns:
                raise ValueError(f"数据中缺少列: {col}")
            return data[col]

        # 未知表达式
        raise ValueError(f"无法解析表达式: {expr}")

    def _eval_function(self, data: pd.DataFrame, func_name: str, expr: str) -> pd.Series:
        """求值函数调用"""
        # 提取参数: "Func(arg1, arg2)"
        inner = expr[len(func_name) + 1:-1]
        args = self._split_args(inner)

        if func_name == 'Ref':
            # Ref(expr, N) — 前 N 期值（按 code 分组 shift）
            if len(args) != 2:
                raise ValueError("Ref 需要 2 个参数: Ref(expr, N)")
            sub_expr, n_str = args
            n = int(n_str.strip())
            series = self._evaluate_expression(data, sub_expr.strip())
            return series.groupby(data['code']).shift(n) if 'code' in data.columns else series.shift(n)

        elif func_name == 'Mean':
            # Mean(expr, N) — N 期滚动均值
            if len(args) != 2:
                raise ValueError("Mean 需要 2 个参数: Mean(expr, N)")
            sub_expr, n_str = args
            n = int(n_str.strip())
            series = self._evaluate_expression(data, sub_expr.strip())
            return self._rolling_apply(data, series, n, 'mean')

        elif func_name == 'Std':
            # Std(expr, N) — N 期滚动标准差
            if len(args) != 2:
                raise ValueError("Std 需要 2 个参数: Std(expr, N)")
            sub_expr, n_str = args
            n = int(n_str.strip())
            series = self._evaluate_expression(data, sub_expr.strip())
            return self._rolling_apply(data, series, n, 'std')

        elif func_name == 'Delay':
            # Delay(expr, N) — 向后延迟 N 期
            if len(args) != 2:
                raise ValueError("Delay 需要 2 个参数: Delay(expr, N)")
            sub_expr, n_str = args
            n = int(n_str.strip())
            series = self._evaluate_expression(data, sub_expr.strip())
            return series.groupby(data['code']).shift(n) if 'code' in data.columns else series.shift(n)

        elif func_name == 'Delta':
            # Delta(expr, N) — N 期差分
            if len(args) != 2:
                raise ValueError("Delta 需要 2 个参数: Delta(expr, N)")
            sub_expr, n_str = args
            n = int(n_str.strip())
            series = self._evaluate_expression(data, sub_expr.strip())
            return series - series.groupby(data['code']).shift(n)

        elif func_name == 'Rank':
            # Rank(expr) — 横截面排名（分位数 0~1）
            series = self._evaluate_expression(data, args[0].strip())
            return series.groupby(data['date']).rank(pct=True) if 'date' in data.columns else series.rank(pct=True)

        elif func_name == 'Log':
            series = self._evaluate_expression(data, args[0].strip())
            return np.log(series.replace(0, np.nan))

        elif func_name == 'Abs':
            series = self._evaluate_expression(data, args[0].strip())
            return series.abs()

        elif func_name == 'Sign':
            series = self._evaluate_expression(data, args[0].strip())
            return np.sign(series)

        raise ValueError(f"未知函数: {func_name}")

    def _rolling_apply(self, data: pd.DataFrame, series: pd.Series, window: int, method: str) -> pd.Series:
        """按 code 分组做滚动窗口聚合"""
        min_p = max(2, window // 4)
        if 'code' in data.columns:
            def _group_apply(x):
                if method == 'mean':
                    return x.rolling(window, min_periods=min_p).mean()
                elif method == 'std':
                    return x.rolling(window, min_periods=min_p).std()
                return x
            return series.groupby(data['code']).transform(_group_apply)
        else:
            if method == 'mean':
                return series.rolling(window, min_periods=min_p).mean()
            elif method == 'std':
                return series.rolling(window, min_periods=min_p).std()
            return series

    def _split_at_top_level(self, expr: str, op: str) -> List[str]:
        """在顶层（括号外）按操作符分割表达式"""
        depth = 0
        for i in range(len(expr) - len(op) + 1):
            ch = expr[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif depth == 0 and expr[i:i + len(op)] == op:
                # 特殊处理减号：避免与负号/一元负混淆
                if op == '-' and (i == 0 or expr[i - 1] in '+-*/('):
                    continue
                return [expr[:i].strip(), expr[i + len(op):].strip()]
        return [expr]

    def _split_args(self, args_str: str) -> List[str]:
        """在顶层逗号处分割函数参数"""
        depth = 0
        parts = []
        current = []
        for ch in args_str:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if ch == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current).strip())
        return parts


# ═══════════════════════════════════════════
# 2. Alpha158 风格因子集注册器（借鉴 Qlib Alpha158）
# ═══════════════════════════════════════════

ALPHA158_STYLE_FACTORS = {
    # === 动量因子 ===
    "momentum_5d":   "$close / Ref($close, 5) - 1",
    "momentum_10d":  "$close / Ref($close, 10) - 1",
    "momentum_20d":  "$close / Ref($close, 20) - 1",
    "momentum_60d":  "$close / Ref($close, 60) - 1",

    # === 反转因子 ===
    "reversal_5d":   "-1 * ($close / Ref($close, 5) - 1)",
    "reversal_10d":  "-1 * ($close / Ref($close, 10) - 1)",
    "reversal_20d":  "-1 * ($close / Ref($close, 20) - 1)",

    # === 波动率因子 ===
    "volatility_5d":   "Std($close / Ref($close, 1) - 1, 5)",
    "volatility_20d":  "Std($close / Ref($close, 1) - 1, 20)",

    # === 均线偏离 ===
    "ma_divergence_5d":   "$close / Mean($close, 5) - 1",
    "ma_divergence_10d":  "$close / Mean($close, 10) - 1",
    "ma_divergence_20d":  "$close / Mean($close, 20) - 1",

    # === 成交量因子 ===
    "volume_mean_5d":   "Mean($volume, 5)",
    "volume_mean_20d":  "Mean($volume, 20)",
    "volume_ratio":     "$volume / Mean($volume, 20)",

    # === 波动率调整动量（Qlib 核心创新，必须自包含，不能引用其它因子名） ===
    "vol_adj_mom_20d":  "($close / Ref($close, 20) - 1) / Std($close / Ref($close, 1) - 1, 20)",
    "vol_adj_mom_60d":  "($close / Ref($close, 60) - 1) / Std($close / Ref($close, 1) - 1, 60)",
}


def create_alpha158_style_engine() -> FactorExpressionEngine:
    """创建预配置 Alpha158 风格因子引擎"""
    engine = FactorExpressionEngine()
    engine.register_factors(ALPHA158_STYLE_FACTORS)
    return engine


# ═══════════════════════════════════════════
# 3. 测试代码
# ═══════════════════════════════════════════

class TestFactorExpressionEngine(unittest.TestCase):
    """因子表达式引擎正确性测试"""

    @classmethod
    def setUpClass(cls):
        """创建测试数据：5只股票 × 100个交易日"""
        np.random.seed(42)
        codes = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        dates = pd.date_range('2024-01-01', periods=100, freq='B')

        rows = []
        for code in codes:
            base_price = np.random.uniform(10, 50)
            prices = base_price * np.cumprod(1 + np.random.normal(0.0002, 0.015, len(dates)))

            for i, (dt, close) in enumerate(zip(dates, prices)):
                daily_ret = np.random.normal(0, 0.015)
                rows.append({
                    'code': code,
                    'date': dt,
                    'open': close * (1 + np.random.normal(0, 0.003)),
                    'high': close * (1 + abs(np.random.normal(0, 0.008))),
                    'low': close * (1 - abs(np.random.normal(0, 0.008))),
                    'close': close,
                    'volume': np.random.lognormal(10, 0.5),
                    'amount': close * np.random.lognormal(10, 0.5),
                })

        cls.test_data = pd.DataFrame(rows).sort_values(['code', 'date']).reset_index(drop=True)

    def test_basic_field_reference(self):
        """测试基本字段引用"""
        engine = FactorExpressionEngine()
        engine.register_factor('close_copy', '$close')
        result = engine.evaluate_all(self.test_data)

        self.assertIn('close_copy', result.columns)
        pd.testing.assert_series_equal(
            result['close_copy'].reset_index(drop=True),
            self.test_data['close'].reset_index(drop=True),
            check_names=False
        )

    def test_simple_arithmetic(self):
        """测试简单算术运算"""
        engine = FactorExpressionEngine()
        engine.register_factor('return_1d', '$close / Ref($close, 1) - 1')
        result = engine.evaluate_all(self.test_data)

        expected = self.test_data.groupby('code')['close'].pct_change()
        pd.testing.assert_series_equal(
            result['return_1d'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_momentum_factors(self):
        """测试动量因子"""
        engine = FactorExpressionEngine()
        engine.register_factors({
            'mom_5d': '$close / Ref($close, 5) - 1',
            'mom_20d': '$close / Ref($close, 20) - 1',
        })
        result = engine.evaluate_all(self.test_data)

        expected_5d = self.test_data.groupby('code')['close'].transform(lambda x: x / x.shift(5) - 1)
        expected_20d = self.test_data.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)

        pd.testing.assert_series_equal(
            result['mom_5d'].reset_index(drop=True),
            expected_5d.reset_index(drop=True),
            check_names=False, atol=1e-10,
        )
        pd.testing.assert_series_equal(
            result['mom_20d'].reset_index(drop=True),
            expected_20d.reset_index(drop=True),
            check_names=False, atol=1e-10,
        )

    def test_reversal_factors(self):
        """测试反转因子（负号运算）"""
        engine = FactorExpressionEngine()
        engine.register_factor('reversal_20d', '-1 * ($close / Ref($close, 20) - 1)')
        result = engine.evaluate_all(self.test_data)

        expected = -self.test_data.groupby('code')['close'].transform(lambda x: x / x.shift(20) - 1)

        pd.testing.assert_series_equal(
            result['reversal_20d'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False, atol=1e-10,
        )

    def test_rolling_mean(self):
        """测试滚动均值"""
        engine = FactorExpressionEngine()
        engine.register_factor('ma_10', 'Mean($close, 10)')
        result = engine.evaluate_all(self.test_data)

        # 验证结果合理性：均值应在合理范围内
        valid = result['ma_10'].dropna()
        self.assertGreater(len(valid), 0, "应至少有一些非 NaN 的滚动均值")
        # 均值不应偏离收盘价太多
        ratio = (valid.abs() / self.test_data.loc[valid.index, 'close'].abs())
        self.assertTrue((ratio < 5).all(), "滚动均值不应偏离收盘价超过5倍")

    def test_rolling_std(self):
        """测试滚动标准差"""
        engine = FactorExpressionEngine()
        engine.register_factor('vol_20', 'Std($close / Ref($close, 1) - 1, 20)')
        result = engine.evaluate_all(self.test_data)

        # 验证结果合理：标准差应为非负数
        valid = result['vol_20'].dropna()
        self.assertGreater(len(valid), 0, "应至少有一些非 NaN 的标准差值")
        self.assertTrue((valid >= 0).all(), "标准差不应为负")
        # 验证有实际变动（不只是 0）
        self.assertGreater(valid.std(), 0, "标准差应有变化")

    def test_volume_ratio(self):
        """测试量比因子"""
        engine = FactorExpressionEngine()
        engine.register_factor('volume_ratio', '$volume / Mean($volume, 20)')
        result = engine.evaluate_all(self.test_data)

        # 验证量比合理范围（大部分应当在 0~3 之间）
        valid = result['volume_ratio'].dropna()
        self.assertGreater(len(valid), 0)
        reasonable_ratio = ((valid >= 0) & (valid <= 5)).mean()
        self.assertGreater(reasonable_ratio, 0.7, f"量比因子正常值比例过低: {reasonable_ratio:.2%}")

    def test_rank_function(self):
        """测试横截面排名"""
        engine = FactorExpressionEngine()
        engine.register_factor('close_rank', 'Rank($close)')
        result = engine.evaluate_all(self.test_data)

        # 验证每个日期上的排名值在 [0, 1] 范围内
        for dt in result['date'].unique():
            ranks = result[result['date'] == dt]['close_rank']
            self.assertTrue((ranks >= 0).all() and (ranks <= 1).all(),
                            f"日期 {dt} 的排名值超出 [0,1] 范围")
            self.assertAlmostEqual(ranks.max(), 1.0, delta=0.1,
                                   msg=f"日期 {dt} 的最大排名应接近 1.0")

    def test_alpha158_batch_computation(self):
        """测试 Alpha158 风格因子集批量计算"""
        engine = create_alpha158_style_engine()
        result = engine.evaluate_all(self.test_data)

        # 验证所有因子都被计算
        for name in ALPHA158_STYLE_FACTORS:
            self.assertIn(name, result.columns, f"因子 '{name}' 未被计算")

        # 验证无全 NaN 因子
        for name in ALPHA158_STYLE_FACTORS:
            nan_ratio = result[name].isna().mean()
            # 60d 因子允许更高 NaN 比例（因为需要更长的历史数据）
            max_nan = 0.7 if '60' in name else 0.5
            self.assertLess(
                nan_ratio, max_nan,
                f"因子 '{name}' NaN 比例过高: {nan_ratio:.2%} (允许上限: {max_nan:.0%})"
            )

        # 验证因子数据合理性
        for name in ['momentum_5d', 'reversal_5d', 'ma_divergence_5d', 'volume_ratio']:
            valid = result[name].dropna()
            if len(valid) > 0:
                # 检查极端值比例（量比因子允许较高比例因为量比天然就经常偏离均值）
                max_extreme = 0.6 if 'volume' in name else 0.3
                extreme_ratio = (abs(valid) > 1.0).mean()
                self.assertLess(extreme_ratio, max_extreme,
                                f"因子 '{name}' 极端值比例过高: {extreme_ratio:.2%}")

    def test_factor_registry(self):
        """测试因子注册和管理"""
        engine = FactorExpressionEngine()
        engine.register_factors({
            'factor_a': '$close / Ref($close, 5) - 1',
            'factor_b': 'Mean($volume, 10)',
        })

        self.assertEqual(set(engine.list_factors()), {'factor_a', 'factor_b'})

        engine.register_factor('factor_c', 'Rank($close)')
        self.assertEqual(len(engine.list_factors()), 3)

    def test_performance_comparison(self):
        """
        性能对比测试：表达式引擎 vs 硬编码计算

        对比原生硬编码因子的计算速度与表达式引擎的计算速度
        """
        import time

        # 硬编码方式（当前 FactorEngine 的方式）
        t0 = time.time()
        data = self.test_data.copy()
        expected_mom_20 = data.groupby('code')['close'].transform(
            lambda x: x / x.shift(20) - 1
        )
        expected_vol_20 = data.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=5).std()
        )
        t_hardcoded = time.time() - t0

        # 表达式引擎方式
        engine = FactorExpressionEngine()
        engine.register_factors({
            'momentum_20d': '$close / Ref($close, 20) - 1',
            'volatility_20d': 'Std($close / Ref($close, 1) - 1, 20)',
        })
        t0 = time.time()
        result = engine.evaluate_all(self.test_data)
        t_expression = time.time() - t0

        # 验证结果一致性
        pd.testing.assert_series_equal(
            result['momentum_20d'].reset_index(drop=True).fillna(0),
            expected_mom_20.reset_index(drop=True).fillna(0),
            check_names=False, atol=1e-9,
        )

        print(f"\n性能对比:")
        print(f"  硬编码计算耗时:  {t_hardcoded:.4f}s")
        print(f"  表达式引擎耗时:  {t_expression:.4f}s")
        print(f"  性能比率:        {t_expression / t_hardcoded:.2f}x")
        print(f"  (表达式引擎略慢但相差在 2-3x 以内属可接受范围)")

        # 断言性能在可接受范围内
        self.assertLess(t_expression / max(t_hardcoded, 0.001), 10,
                        "表达式引擎性能不应比硬编码慢超过 10 倍")


class TestFactorExpressionEngineEdgeCases(unittest.TestCase):
    """边界条件测试"""

    def test_empty_data(self):
        """空数据测试"""
        engine = FactorExpressionEngine()
        engine.register_factor('mom', '$close / Ref($close, 5) - 1')
        result = engine.evaluate_all(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_single_row(self):
        """单行数据测试"""
        data = pd.DataFrame([{
            'code': '000001.SZ', 'date': pd.Timestamp('2024-01-01'),
            'open': 10, 'high': 11, 'low': 9, 'close': 10.5,
            'volume': 10000, 'amount': 105000,
        }])
        engine = FactorExpressionEngine()
        engine.register_factor('close_copy', '$close')
        result = engine.evaluate_all(data)
        self.assertEqual(result['close_copy'].iloc[0], 10.5)

    def test_single_stock(self):
        """单只股票测试"""
        data = self._make_stock_data('000001.SZ', 30)
        engine = FactorExpressionEngine()
        engine.register_factors({
            'mom_5d': '$close / Ref($close, 5) - 1',
            'ma_10d': 'Mean($close, 10)',
        })
        result = engine.evaluate_all(data)
        self.assertEqual(result['code'].nunique(), 1)
        self.assertIn('mom_5d', result.columns)

    @staticmethod
    def _make_stock_data(code: str, n: int) -> pd.DataFrame:
        dates = pd.date_range('2024-01-01', periods=n, freq='B')
        rows = []
        price = 20.0
        for dt in dates:
            price *= (1 + np.random.normal(0.0002, 0.015))
            rows.append({
                'code': code,
                'date': dt,
                'open': price * (1 + np.random.normal(0, 0.003)),
                'high': price * 1.01,
                'low': price * 0.99,
                'close': price,
                'volume': np.random.lognormal(10, 0.5),
                'amount': price * np.random.lognormal(10, 0.5),
            })
        return pd.DataFrame(rows)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__('__main__'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)