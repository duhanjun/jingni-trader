"""
优化方向: 因子表达式引擎 (Factor Expression Engine)
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
  - Qlib 的 Expression Engine 使用声明式 DSL 定义因子，如 Ref($close, 60)/$close
  - 核心价值: 将因子从"硬编码"变为"表达式驱动"，大幅提升因子库可扩展性
  - 参考文件: qlib/data/ops.py, qlib/contrib/data/handler.py (Alpha158)
对比对象: jingni-trader skills/factor-engine/scripts/adapters/ (pandas_ta_calculator.py, talib_calculator.py)
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ============================================================
# 1. 表达式引擎核心实现
# ============================================================

class ExpressionError(Exception):
    """表达式错误"""
    pass


@dataclass
class FactorMeta:
    """因子元信息"""
    name: str
    expression: str
    direction: int = 1  # 1=正向因子, -1=反向因子
    description: str = ""
    category: str = ""  # 动量/反转/波动/流动性/...


class FactorRegistry:
    """因子注册中心 - 借鉴 Qlib 的 Alpha158 因子库设计"""

    def __init__(self):
        self._factors: Dict[str, FactorMeta] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, meta: FactorMeta):
        self._factors[meta.name] = meta
        if meta.category not in self._categories:
            self._categories[meta.category] = []
        self._categories[meta.category].append(meta.name)

    def get(self, name: str) -> Optional[FactorMeta]:
        return self._factors.get(name)

    def list_by_category(self, category: str) -> List[str]:
        return self._categories.get(category, [])

    def list_all(self) -> List[str]:
        return list(self._factors.keys())

    def get_categories(self) -> List[str]:
        return list(self._categories.keys())


class ExpressionEngine:
    """
    表达式引擎 - 借鉴 Qlib 的 ops.py 设计

    支持的运算:
    - 字段引用: $close, $open, $high, $low, $volume, $amount
    - 算术: +, -, *, /
    - 条件: if(cond, true_val, false_val)
    - 时序: ref(expr, N), mean(expr, N), std(expr, N), max(expr, N), min(expr, N)
    - 差值: delta(expr, N)
    - 排名: rank(expr)
    - 截面: scale(expr), cs_rank(expr)
    - 逻辑: expr > val, expr < val, expr >= val, expr <= val, expr == val
    - 组合: and(expr1, expr2), or(expr1, expr2)
    """

    # 支持的原始字段
    FIELD_ALIASES = {
        '$close': 'close', '$open': 'open', '$high': 'high',
        '$low': 'low', '$volume': 'volume', '$amount': 'amount',
        '$vwap': 'vwap', '$returns': 'returns',
    }

    def __init__(self):
        self._registry = FactorRegistry()
        self._cache: Dict[str, np.ndarray] = {}

    def evaluate(self, expression: str, data: pd.DataFrame) -> np.ndarray:
        """
        计算表达式

        参数:
            expression: 因子表达式字符串
            data: 包含 OHLCV 的 DataFrame, 按 code 分组, 按 date 排序

        返回:
            因子值 numpy array
        """
        self._data = data
        self._cache = {}
        result = self._parse(expression.strip())
        return result

    def _parse(self, expr: str) -> np.ndarray:
        """递归下降解析器"""
        expr = expr.strip()

        # 缓存命中
        if expr in self._cache:
            return self._cache[expr]

        # 数字常量 (包含正负号)
        try:
            val = float(expr)
            result = np.full(len(self._data), val)
            self._cache[expr] = result
            return result
        except ValueError:
            pass

        # 字段引用 (纯字段名，没有运算符)
        if expr in self.FIELD_ALIASES:
            field = self.FIELD_ALIASES[expr]
            if field in self._data.columns:
                result = self._data[field].values
                self._cache[expr] = result
                return result

        # 处理前缀负号: -expr
        if expr.startswith('-') and len(expr) > 1:
            inner = self._parse(expr[1:])
            result = -inner
            self._cache[expr] = result
            return result

        # 处理括号内的表达式
        if expr.startswith('(') and expr.endswith(')'):
            return self._parse(expr[1:-1])

        # 二元运算符: 从低优先级到高优先级
        for op in ['+', '-']:
            idx = self._find_operator(expr, op)
            if idx > 0:  # 只有 idx > 0 才是二元运算符
                left = self._parse(expr[:idx])
                right = self._parse(expr[idx + 1:])
                result = left + right if op == '+' else left - right
                self._cache[expr] = result
                return result

        for op in ['*', '/']:
            idx = self._find_operator(expr, op)
            if idx >= 0:
                left = self._parse(expr[:idx])
                right = self._parse(expr[idx + 1:])
                if op == '*':
                    result = left * right
                else:
                    result = np.divide(left, right, out=np.zeros_like(left), where=right != 0)
                self._cache[expr] = result
                return result

        # 比较运算符
        for op in ['>=', '<=', '>', '<', '==']:
            idx = self._find_operator(expr, op)
            if idx >= 0:
                left = self._parse(expr[:idx])
                right = self._parse(expr[idx + len(op):])
                if op == '>=': result = (left >= right).astype(float)
                elif op == '<=': result = (left <= right).astype(float)
                elif op == '>': result = (left > right).astype(float)
                elif op == '<': result = (left < right).astype(float)
                elif op == '==': result = (left == right).astype(float)
                self._cache[expr] = result
                return result

        # 函数调用: func_name(args...)  - 在所有运算符之后检查
        lp = expr.find('(')
        if lp > 0 and expr.endswith(')'):
            func_name = expr[:lp].strip()
            # 确保函数名是有效标识符 (不含空格/运算符)
            if func_name.replace('_', '').isalnum() and ' ' not in func_name:
                args_str = expr[lp + 1:-1].strip()
                args = self._split_args(args_str)
                return self._call_func(func_name, args)

        # 纯字段引用（兜底，不应该到这来）
        if expr.startswith('$'):
            field = self.FIELD_ALIASES.get(expr)
            if field and field in self._data.columns:
                result = self._data[field].values
                self._cache[expr] = result
                return result

        raise ExpressionError(f"无法解析表达式: {expr}")

    def _find_operator(self, expr: str, op: str) -> int:
        """在表达式中查找不在括号内的运算符"""
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == ')': depth += 1
            elif expr[i] == '(': depth -= 1
            elif depth == 0:
                ol = min(len(op), i + 1)
                if expr[i - ol + 1:i + 1] == op:
                    return i - ol + 1
        return -1

    def _split_args(self, args_str: str) -> List[str]:
        """按逗号分割参数，保护括号和引号"""
        if not args_str:
            return []
        args = []
        depth = 0
        current = []
        for ch in args_str:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            elif ch == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
                continue
            current.append(ch)
        if current:
            args.append(''.join(current).strip())
        return args

    def _call_func(self, func_name: str, args: List[str]) -> np.ndarray:
        """调用内置函数"""
        if func_name == 'ref':
            return self._func_ref(args)
        elif func_name == 'mean':
            return self._func_rolling(args, 'mean')
        elif func_name == 'std':
            return self._func_rolling(args, 'std')
        elif func_name == 'max':
            return self._func_rolling(args, 'max')
        elif func_name == 'min':
            return self._func_rolling(args, 'min')
        elif func_name == 'sum':
            return self._func_rolling(args, 'sum')
        elif func_name == 'delta':
            return self._func_delta(args)
        elif func_name == 'rank':
            return self._func_rank(args)
        elif func_name == 'abs':
            return np.abs(self._parse(args[0]))
        elif func_name == 'log':
            val = self._parse(args[0])
            return np.log(np.maximum(val, 1e-10))
        elif func_name == 'scale':
            return self._func_scale(args)
        elif func_name == 'if':
            return self._func_if(args)
        elif func_name == 'and':
            return self._func_and(args)
        elif func_name == 'or':
            return self._func_or(args)
        else:
            raise ExpressionError(f"未知函数: {func_name}")

    def _func_ref(self, args: List[str]) -> np.ndarray:
        """ref(expr, N) - 向前取N天前值"""
        if len(args) != 2:
            raise ExpressionError("ref 需要两个参数: ref(expr, N)")
        val = self._parse(args[0])
        n = int(self._parse(args[1])[0])  # 假定N是常量
        result = np.full_like(val, np.nan)
        result[n:] = val[:-n]
        return result

    def _func_rolling(self, args: List[str], method: str) -> np.ndarray:
        """mean/std/max/min/sum(expr, N) - 滚动窗口计算"""
        if len(args) != 2:
            raise ExpressionError(f"{method} 需要两个参数: {method}(expr, N)")
        val = self._parse(args[0])
        n = int(self._parse(args[1])[0])
        result = pd.Series(val).rolling(n, min_periods=1)
        if method == 'mean': return result.mean().values
        elif method == 'std': return result.std().values
        elif method == 'max': return result.max().values
        elif method == 'min': return result.min().values
        elif method == 'sum': return result.sum().values

    def _func_delta(self, args: List[str]) -> np.ndarray:
        """delta(expr, N) - expr - ref(expr, N)"""
        if len(args) != 2:
            raise ExpressionError("delta 需要两个参数: delta(expr, N)")
        val = self._parse(args[0])
        n = int(self._parse(args[1])[0])
        ref_val = np.full_like(val, np.nan)
        ref_val[n:] = val[:-n]
        return val - ref_val

    def _func_rank(self, args: List[str]) -> np.ndarray:
        """rank(expr) - 截面排名（百分比）"""
        val = self._parse(args[0])
        return pd.Series(val).rank(pct=True).values

    def _func_scale(self, args: List[str]) -> np.ndarray:
        """scale(expr) - 截面标准化"""
        val = self._parse(args[0])
        std = np.nanstd(val)
        return (val - np.nanmean(val)) / (std if std > 0 else 1)

    def _func_if(self, args: List[str]) -> np.ndarray:
        """if(cond, true_val, false_val)"""
        if len(args) != 3:
            raise ExpressionError("if 需要三个参数: if(cond, true_val, false_val)")
        cond = self._parse(args[0])
        true_val = self._parse(args[1])
        false_val = self._parse(args[2])
        return np.where(cond > 0, true_val, false_val)

    def _func_and(self, args: List[str]) -> np.ndarray:
        a = self._parse(args[0])
        b = self._parse(args[1])
        return ((a > 0) & (b > 0)).astype(float)

    def _func_or(self, args: List[str]) -> np.ndarray:
        a = self._parse(args[0])
        b = self._parse(args[1])
        return ((a > 0) | (b > 0)).astype(float)

    # ---- 因子注册与批量计算 ----

    def register_factor(self, meta: FactorMeta):
        """注册预定义因子"""
        self._registry.register(meta)

    def calculate_factor(self, name: str, data: pd.DataFrame) -> np.ndarray:
        """按名称计算已注册因子"""
        meta = self._registry.get(name)
        if not meta:
            raise ExpressionError(f"未注册因子: {name}")
        return self.evaluate(meta.expression, data)

    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """批量计算所有已注册因子"""
        result = data[['code', 'date']].copy()
        for name in self._registry.list_all():
            result[name] = self.calculate_factor(name, data)
        return result


# ============================================================
# 2. 预定义因子库（借鉴 Qlib Alpha158）
# ============================================================

def build_alpha158_factors() -> ExpressionEngine:
    """构建 Alpha158 风格的因子库"""
    engine = ExpressionEngine()

    factors = [
        # ---- 动量因子 ----
        FactorMeta("KMID", "delta($close, 5)", direction=1,
                   description="5日价格动量", category="momentum"),
        FactorMeta("KMID2", "delta($close, 10)", direction=1,
                   description="10日价格动量", category="momentum"),
        FactorMeta("KLEN", "($close - $open) / $open", direction=1,
                   description="当日涨跌幅", category="momentum"),
        FactorMeta("KLEN2", "ref($close, -2) / $close - 1", direction=1,
                   description="2日收益率", category="momentum"),

        # ---- 反转因子 ----
        FactorMeta("KREV", "-delta($close, 5)", direction=1,
                   description="5日反转", category="reversal"),
        FactorMeta("KREV2", "-delta($close, 20)", direction=1,
                   description="20日反转", category="reversal"),

        # ---- 波动因子 ----
        FactorMeta("KVOL", "std(delta($close, 1), 20)", direction=-1,
                   description="20日波动率", category="volatility"),
        FactorMeta("KVOL2", "mean(abs($close - $open) / $open, 5)", direction=-1,
                   description="5日平均振幅", category="volatility"),

        # ---- 流动性因子 ----
        FactorMeta("KLIQ", "mean($volume, 5) / mean($volume, 20)", direction=1,
                   description="5日/20日成交量比", category="liquidity"),
        FactorMeta("KLIQ2", "delta(mean($volume, 5), 20)", direction=1,
                   description="成交量变化", category="liquidity"),

        # ---- 均线因子 ----
        FactorMeta("KMA5", "$close / mean($close, 5) - 1", direction=1,
                   description="收盘价偏离5日均线", category="trend"),
        FactorMeta("KMA10", "$close / mean($close, 10) - 1", direction=1,
                   description="收盘价偏离10日均线", category="trend"),
        FactorMeta("KMA20", "$close / mean($close, 20) - 1", direction=1,
                   description="收盘价偏离20日均线", category="trend"),
    ]

    for f in factors:
        engine.register_factor(f)

    return engine


# ============================================================
# 3. 测试用例
# ============================================================

class TestExpressionEngine(unittest.TestCase):
    """表达式引擎测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟数据"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2023-01-01', periods=n, freq='B')
        cls.data = pd.DataFrame({
            'code': '000001.SZ',
            'date': dates,
            'open': np.cumsum(np.random.randn(n) * 0.5) + 10,
            'high': np.cumsum(np.random.randn(n) * 0.5) + 10.5,
            'low': np.cumsum(np.random.randn(n) * 0.5) + 9.5,
            'close': np.cumsum(np.random.randn(n) * 0.5) + 10,
            'volume': np.abs(np.random.randn(n) * 10000 + 50000),
            'amount': np.abs(np.random.randn(n) * 100000 + 500000),
        })
        cls.engine = ExpressionEngine()

    def test_field_reference(self):
        """测试字段引用"""
        result = self.engine.evaluate('$close', self.data)
        np.testing.assert_array_almost_equal(result, self.data['close'].values)

    def test_arithmetic(self):
        """测试基本算术运算"""
        result = self.engine.evaluate('$close / $open - 1', self.data)
        expected = self.data['close'].values / self.data['open'].values - 1
        np.testing.assert_array_almost_equal(result, expected)

    def test_ref_operator(self):
        """测试ref运算符"""
        result = self.engine.evaluate('ref($close, 1)', self.data)
        expected = np.roll(self.data['close'].values, 1)
        expected[0] = np.nan
        np.testing.assert_array_almost_equal(result, expected)

    def test_delta_operator(self):
        """测试delta运算符"""
        result = self.engine.evaluate('delta($close, 5)', self.data)
        close = self.data['close'].values
        ref = np.roll(close, 5)
        ref[:5] = np.nan
        expected = close - ref
        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_mean(self):
        """测试滚动均值"""
        result = self.engine.evaluate('mean($close, 20)', self.data)
        expected = self.data['close'].rolling(20, min_periods=1).mean().values
        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_std(self):
        """测试滚动标准差"""
        result = self.engine.evaluate('std($close, 20)', self.data)
        expected = self.data['close'].rolling(20, min_periods=1).std().values
        np.testing.assert_array_almost_equal(result, expected)

    def test_compound_expression(self):
        """测试复合表达式"""
        # 收盘价偏离20日均线的百分比
        result = self.engine.evaluate('$close / mean($close, 20) - 1', self.data)
        ma20 = self.data['close'].rolling(20, min_periods=1).mean().values
        expected = self.data['close'].values / ma20 - 1
        np.testing.assert_array_almost_equal(result, expected)

    def test_rank_operator(self):
        """测试截面排名"""
        result = self.engine.evaluate('rank($close)', self.data)
        expected = pd.Series(self.data['close'].values).rank(pct=True).values
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_operator(self):
        """测试截面标准化"""
        result = self.engine.evaluate('scale($close)', self.data)
        val = self.data['close'].values
        expected = (val - np.mean(val)) / np.std(val)
        np.testing.assert_array_almost_equal(result, expected)

    def test_if_operator(self):
        """测试if条件运算符"""
        result = self.engine.evaluate('if($close > $open, 1, -1)', self.data)
        expected = np.where(self.data['close'].values > self.data['open'].values, 1.0, -1.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_abs_operator(self):
        """测试绝对值运算符"""
        result = self.engine.evaluate('abs($close - $open)', self.data)
        expected = np.abs(self.data['close'].values - self.data['open'].values)
        np.testing.assert_array_almost_equal(result, expected)

    def test_log_operator(self):
        """测试对数运算符"""
        result = self.engine.evaluate('log($close)', self.data)
        expected = np.log(np.maximum(self.data['close'].values, 1e-10))
        np.testing.assert_array_almost_equal(result, expected)

    def test_comparison_operators(self):
        """测试比较运算符"""
        result = self.engine.evaluate('$close > $open', self.data)
        expected = (self.data['close'].values > self.data['open'].values).astype(float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_and_operator(self):
        """测试逻辑与运算符"""
        result = self.engine.evaluate('and($close > $open, $volume > 50000)', self.data)
        c1 = self.data['close'].values > self.data['open'].values
        c2 = self.data['volume'].values > 50000
        expected = (c1 & c2).astype(float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_or_operator(self):
        """测试逻辑或运算符"""
        result = self.engine.evaluate('or($close > $open, $volume > 50000)', self.data)
        c1 = self.data['close'].values > self.data['open'].values
        c2 = self.data['volume'].values > 50000
        expected = (c1 | c2).astype(float)
        np.testing.assert_array_almost_equal(result, expected)


class TestFactorRegistry(unittest.TestCase):
    """因子注册中心测试"""

    def test_register_and_retrieve(self):
        """测试因子注册和检索"""
        engine = build_alpha158_factors()
        self.assertIn("KMID", engine._registry.list_all())
        self.assertIn("KVOL", engine._registry.list_all())
        self.assertEqual(len(engine._registry.list_all()), 13)

    def test_category_filtering(self):
        """测试按类别筛选因子"""
        engine = build_alpha158_factors()
        momentum = engine._registry.list_by_category("momentum")
        self.assertEqual(len(momentum), 4)
        self.assertIn("KMID", momentum)
        self.assertIn("KLEN", momentum)

    def test_categories(self):
        """测试获取所有类别"""
        engine = build_alpha158_factors()
        categories = engine._registry.get_categories()
        self.assertIn("momentum", categories)
        self.assertIn("volatility", categories)
        self.assertIn("reversal", categories)

    def test_calculate_by_name(self):
        """测试按名称计算因子"""
        engine = build_alpha158_factors()
        result = engine.calculate_factor("KMID", TestExpressionEngine.data)
        close = TestExpressionEngine.data['close'].values
        ref = np.roll(close, 5)
        ref[:5] = np.nan
        expected = close - ref
        np.testing.assert_array_almost_equal(result, expected)

    def test_batch_calculate(self):
        """测试批量计算"""
        engine = build_alpha158_factors()
        result = engine.calculate_all(TestExpressionEngine.data)
        self.assertIn("KMID", result.columns)
        self.assertIn("KVOL", result.columns)
        self.assertEqual(len(result.columns), 2 + 13)  # code + date + 13 factors

    def test_factor_meta(self):
        """测试因子元信息"""
        engine = build_alpha158_factors()
        meta = engine._registry.get("KVOL")
        self.assertEqual(meta.direction, -1)
        self.assertEqual(meta.description, "20日波动率")
        self.assertEqual(meta.category, "volatility")


class TestExpressionPerformance(unittest.TestCase):
    """表达式引擎性能测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 5000
        dates = pd.date_range('2023-01-01', periods=n, freq='B')
        cls.data = pd.DataFrame({
            'code': '000001.SZ',
            'date': dates,
            'open': np.cumsum(np.random.randn(n) * 0.5) + 10,
            'high': np.cumsum(np.random.randn(n) * 0.5) + 10.5,
            'low': np.cumsum(np.random.randn(n) * 0.5) + 9.5,
            'close': np.cumsum(np.random.randn(n) * 0.5) + 10,
            'volume': np.abs(np.random.randn(n) * 10000 + 50000),
            'amount': np.abs(np.random.randn(n) * 100000 + 500000),
        })
        cls.engine = build_alpha158_factors()

    def test_expression_vs_direct_batch(self):
        """对比表达式引擎 vs 直接 pandas 计算的批量效率"""
        import time

        # 表达式引擎计算
        start = time.perf_counter()
        result_expr = self.engine.calculate_all(self.data)
        expr_time = time.perf_counter() - start

        # 直接 pandas 计算（等效）
        start = time.perf_counter()
        close = self.data['close']
        ma5 = close.rolling(5, min_periods=1).mean()
        ma10 = close.rolling(10, min_periods=1).mean()
        ma20 = close.rolling(20, min_periods=1).mean()
        result_direct = pd.DataFrame({
            'KMID': close.diff(5).values,
            'KMID2': close.diff(10).values,
            'KLEN': (close - self.data['open']) / self.data['open'],
            'KLEN2': close.shift(-2).values / close.values - 1,
            'KREV': -close.diff(5).values,
            'KREV2': -close.diff(20).values,
            'KVOL': close.diff(1).rolling(20, min_periods=1).std().values,
            'KVOL2': (abs(close - self.data['open']) / self.data['open']).rolling(5, min_periods=1).mean().values,
            'KLIQ': (self.data['volume'].rolling(5, min_periods=1).mean() / 
                     self.data['volume'].rolling(20, min_periods=1).mean()).values,
            'KLIQ2': self.data['volume'].rolling(5, min_periods=1).mean().diff(20).values,
            'KMA5': close / ma5 - 1,
            'KMA10': close / ma10 - 1,
            'KMA20': close / ma20 - 1,
        })
        direct_time = time.perf_counter() - start

        print(f"\n  表达式引擎耗时: {expr_time:.4f}s")
        print(f"  直接 pandas 耗时: {direct_time:.4f}s")
        print(f"  表达式引擎/直接 pandas 比率: {expr_time / direct_time:.2f}x")

        # 表达式引擎不应该比直接pandas慢太多（2x以内是合理的）
        self.assertLess(expr_time / direct_time, 3.0,
                        f"表达式引擎性能过慢: {expr_time/direct_time:.2f}x")

    def test_cache_effectiveness(self):
        """测试缓存有效性"""
        import time

        self.engine._cache = {}

        # 第一次计算（冷缓存）
        start = time.perf_counter()
        self.engine.evaluate('$close / mean($close, 20) - 1', self.data)
        cold_time = time.perf_counter() - start

        # 第二次计算（热缓存）
        start = time.perf_counter()
        self.engine.evaluate('$close / mean($close, 20) - 1', self.data)
        hot_time = time.perf_counter() - start

        print(f"\n  冷缓存耗时: {cold_time:.4f}s")
        print(f"  热缓存耗时: {hot_time:.4f}s")
        print(f"  缓存加速比: {cold_time / hot_time:.2f}x")

        # 热缓存应明显快于冷缓存
        self.assertLess(hot_time, cold_time,
                        "缓存未生效：热缓存耗时 >= 冷缓存耗时")


if __name__ == '__main__':
    unittest.main(verbosity=2)