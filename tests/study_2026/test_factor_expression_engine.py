"""
验证测试：因子表达式引擎（Factor Expression Engine）

借鉴来源：Microsoft Qlib (https://github.com/microsoft/qlib)
  - Qlib 的 Expression Engine 提供 DSL 语法定义因子，
    如 `Ref($close, 60) / $close - 1` 计算 60 日收益率，
    支持跨截面算子（CSRank, CSMean）和时序算子（Ref, Mean, Std）。
  - 设计理念：因子声明为函数而非数据，让 LLM 可直接生成因子表达式。

优化方向：为 jingni-trader 的 factor-engine 引入表达式引擎，
  支持用户用简洁的 DSL 定义因子，降低因子开发门槛，
  提高因子库的可扩展性和 AI 友好性。

测试内容：
  1. 表达式解析器（tokenizer + parser → AST）
  2. 运算符实现（时序算子、截面算子、元素算子）
  3. 表达式求值（多股票 DataFrame 上的批量计算）
  4. 性能对比测试（表达式引擎 vs 手动循环）
  5. 边界条件测试（空数据、NaN、极端值）
"""

import unittest
import pandas as pd
import numpy as np
import re
import time
from typing import Dict, List, Callable, Any, Union
from abc import ABC, abstractmethod


# ============================================================
# 表达式引擎实现（借鉴 Qlib 设计）
# ============================================================

class Expression(ABC):
    """表达式 AST 节点基类"""

    @abstractmethod
    def evaluate(self, data: pd.DataFrame, code_groups: Dict[str, pd.DataFrame]) -> pd.Series:
        ...


class FieldRef(Expression):
    """字段引用节点，如 $close, $open, $volume"""

    def __init__(self, field: str):
        self.field = field

    def evaluate(self, data: pd.DataFrame, code_groups: Dict[str, pd.DataFrame]) -> pd.Series:
        return data[self.field].copy()

    def __repr__(self):
        return f"FieldRef({self.field})"


class Literal(Expression):
    """字面量节点"""

    def __init__(self, value: float):
        self.value = value

    def evaluate(self, data: pd.DataFrame, code_groups: Dict[str, pd.DataFrame]) -> pd.Series:
        return pd.Series(self.value, index=data.index)

    def __repr__(self):
        return f"Literal({self.value})"


class Ref(Expression):
    """时序偏移算子 Ref($close, 5) → 5 天前的收盘价"""

    def __init__(self, expr: Expression, offset: int):
        self.expr = expr
        self.offset = offset

    def evaluate(self, data: pd.DataFrame, code_groups: Dict[str, pd.DataFrame]) -> pd.Series:
        series = self.expr.evaluate(data, code_groups)
        result = pd.Series(index=data.index, dtype=float)
        for code, grp in code_groups.items():
            idx = grp.index
            result.loc[idx] = series.loc[idx].shift(self.offset)
        return result

    def __repr__(self):
        return f"Ref({self.expr}, {self.offset})"


class RollingOp(Expression):
    """滚动窗口算子 Mean($close, 20), Std($close, 20)"""

    OP_MAP = {
        "Mean": "mean",
        "Std": "std",
        "Max": "max",
        "Min": "min",
        "Sum": "sum",
    }

    def __init__(self, op_name: str, expr: Expression, window: int):
        self.op_name = op_name
        self.expr = expr
        self.window = window

    def evaluate(self, data: pd.DataFrame, code_groups: Dict[str, pd.DataFrame]) -> pd.Series:
        series = self.expr.evaluate(data, code_groups)
        method = self.OP_MAP.get(self.op_name, "mean")
        result = pd.Series(index=data.index, dtype=float)
        for code, grp in code_groups.items():
            idx = grp.index
            result.loc[idx] = getattr(series.loc[idx].rolling(self.window, min_periods=1), method)()
        return result

    def __repr__(self):
        return f"{self.op_name}({self.expr}, {self.window})"


class BinaryOp(Expression):
    """二元运算 Add, Sub, Mul, Div, Gt, Lt"""

    OP_MAP = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b,
        "Div": lambda a, b: a / b,
        "Gt":  lambda a, b: (a > b).astype(float),
        "Lt":  lambda a, b: (a < b).astype(float),
    }

    def __init__(self, op_name: str, left: Expression, right: Expression):
        self.op_name = op_name
        self.left = left
        self.right = right

    def evaluate(self, data: pd.DataFrame, code_groups: Dict[str, pd.DataFrame]) -> pd.Series:
        lv = self.left.evaluate(data, code_groups)
        rv = self.right.evaluate(data, code_groups)
        if isinstance(rv, pd.Series) and not isinstance(self.right, Literal):
            return self.OP_MAP[self.op_name](lv, rv)
        else:
            return self.OP_MAP[self.op_name](lv, rv)

    def __repr__(self):
        return f"{self.op_name}({self.left}, {self.right})"


class CSRank(Expression):
    """截面排名算子 CSRank($close) → 在当日的截面内排名"""

    def __init__(self, expr: Expression):
        self.expr = expr

    def evaluate(self, data: pd.DataFrame, code_groups: Dict[str, pd.DataFrame]) -> pd.Series:
        series = self.expr.evaluate(data, code_groups)
        result = pd.Series(index=data.index, dtype=float)
        for date, grp in data.groupby("date"):
            idx = grp.index
            result.loc[idx] = series.loc[idx].rank(pct=True)
        return result

    def __repr__(self):
        return f"CSRank({self.expr})"


class FactorExpressionParser:
    """
    表达式解析器：将 DSL 字符串解析为 AST 表达式树

    支持语法:
      - $field          字段引用
      - 123.45          数字字面量
      - Ref(expr, N)     时序偏移
      - Mean(expr, N)    滚动均值
      - Std(expr, N)     滚动标准差
      - Add(a, b)        加法
      - Sub(a, b)        减法
      - Mul(a, b)        乘法
      - Div(a, b)        除法
      - CSRank(expr)     截面排名
    """

    TOKEN_PATTERN = re.compile(
        r'\$(\w+)|'           # $field
        r'(\d+\.?\d*)|'       # number
        r'([A-Za-z_]\w*)\(|'  # func_name(
        r'(,)|'               # comma
        r'(\))|'              # right paren
        r'(\s+)'              # whitespace
    )

    def __init__(self):
        pass

    def tokenize(self, expr: str) -> List[tuple]:
        tokens = []
        for m in self.TOKEN_PATTERN.finditer(expr):
            if m.group(1):   # $field
                tokens.append(("FIELD", m.group(1)))
            elif m.group(2):  # number
                tokens.append(("NUMBER", float(m.group(2))))
            elif m.group(3):  # func_name(
                tokens.append(("FUNC", m.group(3)))
                tokens.append(("LPAREN", "("))
            elif m.group(4):  # comma
                tokens.append(("COMMA", ","))
            elif m.group(5):  # )
                tokens.append(("RPAREN", ")"))
        return tokens

    def parse(self, expr_str: str) -> Expression:
        tokens = self.tokenize(expr_str)
        self.pos = 0
        self.tokens = tokens
        return self._parse_expr()

    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _parse_expr(self) -> Expression:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        if tok[0] == "FIELD":
            self._consume()
            return FieldRef(tok[1])
        elif tok[0] == "NUMBER":
            self._consume()
            return Literal(tok[1])
        elif tok[0] == "FUNC":
            func_name = self._consume()[1]
            self._consume()  # LPAREN
            args = []
            args.append(self._parse_expr())
            while self._peek() and self._peek()[0] == "COMMA":
                self._consume()  # COMMA
                args.append(self._parse_expr())
            self._consume()  # RPAREN

            # 时序算子
            if func_name in ("Ref",):
                return Ref(args[0], int(args[1].value))
            elif func_name in ("Mean", "Std", "Max", "Min", "Sum"):
                return RollingOp(func_name, args[0], int(args[1].value))
            elif func_name in ("Add", "Sub", "Mul", "Div", "Gt", "Lt"):
                if func_name == "Mul" and isinstance(args[0], Literal) and isinstance(args[1], FieldRef):
                    # 简化：Literal * expr 可直接在 BinaryOp 中处理
                    pass
                return BinaryOp(func_name, args[0], args[1])
            elif func_name == "CSRank":
                return CSRank(args[0])
            else:
                raise ValueError(f"Unknown function: {func_name}")
        else:
            raise ValueError(f"Unexpected token: {tok}")


class FactorExpressionEngine:
    """
    因子表达式引擎：编译表达式并批量求值

    借鉴 Qlib 的设计：
    - 表达式编译为 AST
    - 按股票分组求值，支持跨截面计算
    """

    def __init__(self):
        self.parser = FactorExpressionParser()
        self._compiled: Dict[str, Expression] = {}

    def compile(self, name: str, expr_str: str) -> Expression:
        ast = self.parser.parse(expr_str)
        self._compiled[name] = ast
        return ast

    def evaluate(self, name: str, data: pd.DataFrame) -> pd.Series:
        if name not in self._compiled:
            raise ValueError(f"Factor '{name}' not compiled. Call compile() first.")

        ast = self._compiled[name]
        code_groups = {code: grp for code, grp in data.groupby("code")}
        return ast.evaluate(data, code_groups)

    def calculate(self, data: pd.DataFrame, expressions: Dict[str, str]) -> pd.DataFrame:
        result = data[["code", "date"]].copy()
        for name, expr_str in expressions.items():
            self.compile(name, expr_str)
            result[name] = self.evaluate(name, data)
        return result


# ============================================================
# 测试用例
# ============================================================

class TestFactorExpressionParser(unittest.TestCase):
    """测试表达式解析器"""

    def setUp(self):
        self.parser = FactorExpressionParser()

    def test_field_ref(self):
        ast = self.parser.parse("$close")
        self.assertIsInstance(ast, FieldRef)
        self.assertEqual(ast.field, "close")

    def test_literal(self):
        ast = self.parser.parse("123.45")
        self.assertIsInstance(ast, Literal)
        self.assertEqual(ast.value, 123.45)

    def test_ref(self):
        ast = self.parser.parse("Ref($close, 5)")
        self.assertIsInstance(ast, Ref)
        self.assertEqual(ast.offset, 5)
        self.assertIsInstance(ast.expr, FieldRef)

    def test_rolling_mean(self):
        ast = self.parser.parse("Mean($close, 20)")
        self.assertIsInstance(ast, RollingOp)
        self.assertEqual(ast.op_name, "Mean")
        self.assertEqual(ast.window, 20)

    def test_binary_op(self):
        ast = self.parser.parse("Div(Sub($close, Ref($close, 1)), Ref($close, 1))")
        self.assertIsInstance(ast, BinaryOp)
        self.assertEqual(ast.op_name, "Div")

    def test_csrank(self):
        ast = self.parser.parse("CSRank($close)")
        self.assertIsInstance(ast, CSRank)

    def test_complex_nested(self):
        """测试复杂嵌套表达式：20日收益率的截面排名"""
        expr = "CSRank(Div(Sub($close, Ref($close, 20)), Ref($close, 20)))"
        ast = self.parser.parse(expr)
        self.assertIsInstance(ast, CSRank)
        print(f"    复杂表达式 AST: {ast}")

    def test_invalid_expression(self):
        with self.assertRaises(ValueError):
            self.parser.parse("UnknownFunc($close)")


class TestFactorExpressionEngine(unittest.TestCase):
    """测试因子表达式引擎求值"""

    @classmethod
    def setUpClass(cls):
        """创建多股票测试数据"""
        np.random.seed(42)
        codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="B")
        records = []
        for code in codes:
            n = len(dates)
            base_price = {"000001.SZ": 10.0, "000002.SZ": 20.0, "000003.SZ": 50.0}[code]
            returns = np.random.normal(0.0005, 0.02, n)
            prices = base_price * np.cumprod(1 + returns)
            for i, d in enumerate(dates):
                records.append({
                    "code": code,
                    "date": d,
                    "open": prices[i] * (1 + np.random.normal(0, 0.002)),
                    "high": prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                    "low": prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                    "close": prices[i],
                    "volume": np.random.lognormal(10, 0.5),
                })
        cls.data = pd.DataFrame(records).sort_values(["code", "date"]).reset_index(drop=True)

    def setUp(self):
        self.engine = FactorExpressionEngine()

    def test_simple_ret_1d(self):
        """测试 1 日收益率: (close - close_lag1) / close_lag1"""
        expr = "Div(Sub($close, Ref($close, 1)), Ref($close, 1))"
        self.engine.compile("ret_1d", expr)
        result = self.engine.evaluate("ret_1d", self.data)

        # 手动验证
        manual = self.data.groupby("code")["close"].pct_change()
        manual = manual.reset_index(drop=True)
        result = result.reset_index(drop=True)

        # 比较（忽略 NaN）
        mask = result.notna() & manual.notna()
        self.assertTrue(np.allclose(result[mask], manual[mask], rtol=1e-10))

    def test_ma20(self):
        """测试 20 日均线"""
        expr = "Mean($close, 20)"
        self.engine.compile("ma20", expr)
        result = self.engine.evaluate("ma20", self.data)

        manual = self.data.groupby("code")["close"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        result = result.reset_index(drop=True)
        manual = manual.reset_index(drop=True)
        mask = result.notna() & manual.notna()
        self.assertTrue(np.allclose(result[mask], manual[mask], rtol=1e-10))

    def test_csrank_close(self):
        """测试截面排名"""
        expr = "CSRank($close)"
        self.engine.compile("cs_rank", expr)
        result = self.engine.evaluate("cs_rank", self.data)

        # 手动验证：每天按 close 排名
        manual = self.data.groupby("date")["close"].rank(pct=True)
        result = result.reset_index(drop=True)
        manual = manual.reset_index(drop=True)
        mask = result.notna() & manual.notna()
        self.assertTrue(np.allclose(result[mask], manual[mask], rtol=1e-10))

    def test_complex_factor(self):
        """测试复合因子：20日反转因子的截面排名"""
        # 使用 Sub(0, ret_20) 来实现取负值（反转因子）
        expr = "CSRank(Sub(0, Div(Sub($close, Ref($close, 20)), Ref($close, 20))))"
        self.engine.compile("reversal_20_rank", expr)
        result = self.engine.evaluate("reversal_20_rank", self.data)

        # 验证结果在 [0, 1] 范围内
        valid = result.dropna()
        self.assertTrue(valid.between(0, 1).all())
        self.assertEqual(len(result), len(self.data))

    def test_batch_calculate(self):
        """测试批量计算多个因子"""
        expressions = {
            "ret_1d": "Div(Sub($close, Ref($close, 1)), Ref($close, 1))",
            "ma20": "Mean($close, 20)",
            "ma5": "Mean($close, 5)",
            "ret_20": "Div(Sub($close, Ref($close, 20)), Ref($close, 20))",
        }
        result = self.engine.calculate(self.data, expressions)
        expected_cols = {"code", "date", "ret_1d", "ma20", "ma5", "ret_20"}
        self.assertTrue(expected_cols.issubset(set(result.columns)))
        self.assertEqual(len(result), len(self.data))

    def test_empty_data(self):
        """边界条件：空数据"""
        empty_df = pd.DataFrame(columns=["code", "date", "open", "high", "low", "close", "volume"])
        self.engine.compile("ret_1d", "Div(Sub($close, Ref($close, 1)), Ref($close, 1))")
        result = self.engine.evaluate("ret_1d", empty_df)
        self.assertEqual(len(result), 0)

    def test_nan_handling(self):
        """边界条件：含 NaN 数据"""
        df = self.data.copy()
        df.loc[df.sample(10).index, "close"] = np.nan
        self.engine.compile("ret_1d", "Div(Sub($close, Ref($close, 1)), Ref($close, 1))")
        result = self.engine.evaluate("ret_1d", df)
        self.assertEqual(len(result), len(df))
        # 不应抛出异常


class TestExpressionEnginePerformance(unittest.TestCase):
    """性能对比测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = [f"{i:06d}.{'SZ' if i % 2 == 0 else 'SH'}" for i in range(100)]
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
        records = []
        for code in codes:
            n = len(dates)
            base_price = np.random.uniform(5, 100)
            returns = np.random.normal(0.0005, 0.02, n)
            prices = base_price * np.cumprod(1 + returns)
            for i, d in enumerate(dates):
                records.append({
                    "code": code,
                    "date": d,
                    "close": prices[i],
                })
        cls.large_data = pd.DataFrame(records).sort_values(["code", "date"]).reset_index(drop=True)

    def setUp(self):
        self.engine = FactorExpressionEngine()

    def test_performance_vs_manual(self):
        """对比表达式引擎与手动 pandas 实现的性能"""
        n_trials = 3

        # 表达式引擎
        self.engine.compile("ret_20", "Div(Sub($close, Ref($close, 20)), Ref($close, 20))")
        expr_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            self.engine.evaluate("ret_20", self.large_data)
            expr_times.append(time.perf_counter() - t0)
        avg_expr = np.mean(expr_times)

        # 手动 pandas
        manual_times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            self.large_data.groupby("code")["close"].pct_change(20)
            manual_times.append(time.perf_counter() - t0)
        avg_manual = np.mean(manual_times)

        print(f"\n    数据集: {len(self.large_data)} 行 x 100 只股票")
        print(f"    表达式引擎均值: {avg_expr*1000:.2f}ms")
        print(f"    手动 pandas 均值: {avg_manual*1000:.2f}ms")
        print(f"    比率: {avg_expr/avg_manual:.2f}x")

        # 表达式引擎因按股票分组循环，性能低于原生 pandas groupby
        # 但提供了 DSL 灵活性和可组合性。可通过向量化优化改善性能。
        # 当前目标：不应比手动实现慢 30 倍以上
        self.assertLess(avg_expr / avg_manual, 30.0,
                       f"表达式引擎不应比手动实现慢 30 倍以上，实际: {avg_expr/avg_manual:.1f}x")


if __name__ == "__main__":
    unittest.main(verbosity=2)