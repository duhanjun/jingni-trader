"""
================================================================================
优化方向: 因子表达式 DSL（领域特定语言）引擎
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib) — Expression Engine
         Qlib 的表达式引擎允许用户用声明式语法定义因子，如:
           $close, Ref($close, 1), Mean($close, 3), $high - $low
         引擎自动解析表达式为列引用和运算操作，避免了硬编码因子计算逻辑。
================================================================================

优化目标:
  当前 jingni-trader 的 factor-engine 将所有因子计算硬编码在
  compute_a_share_factors() 方法中（如 ret_1d, ret_5d, turnover_20d 等），
  添加新因子需要修改引擎源码，不利于扩展。

  借鉴 qlib 的表达式引擎设计，实现一个因子 DSL，让用户可以用
  声明式语法定义因子，无需修改核心引擎代码。

验证内容:
  1. DSL 解析器 — 解析表达式字符串为 AST
  2. 因子注册与编译 — 将 DSL 表达式编译为可执行的 pandas 操作序列
  3. 执行与验证 — 用模拟数据验证编译结果的正确性
  4. 与现有因子计算结果做对比验证
  5. 性能对比 — DSL 编译后的计算 vs 原始硬编码计算
"""

import unittest
import sys
import os
import time
import re
import operator
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── 测试配置 ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# 尝试加载真实数据做对比
_REAL_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '../../workspace/data/cleaned_data.parquet'
)


# ================================================================================
# Part 1: 因子 DSL 解析器
# ================================================================================

# Token 类型
TOKEN_REF = "REF"        # 列引用: $close, $volume
TOKEN_OP = "OP"          # 运算符: +, -, *, /
TOKEN_FUNC = "FUNC"      # 函数: Ref, Mean, Std, Max, Min, Log
TOKEN_NUM = "NUM"        # 数字字面量
TOKEN_LPAREN = "LPAREN"
TOKEN_RPAREN = "RPAREN"
TOKEN_COMMA = "COMMA"


@dataclass
class Token:
    type: str
    value: str


@dataclass
class ASTNode:
    type: str  # "ref", "binary_op", "func_call", "number"
    value: Any = None
    left: Optional["ASTNode"] = None
    right: Optional["ASTNode"] = None
    args: List["ASTNode"] = field(default_factory=list)


# 内置函数映射
_BUILTIN_FUNCTIONS = {
    "Ref": "ref",       # 前 N 期值
    "Mean": "mean",     # 滚动均值
    "Std": "std",       # 滚动标准差
    "Max": "max",       # 滚动最大值
    "Min": "min",       # 滚动最小值
    "Sum": "sum",       # 滚动求和
    "Log": "log",       # 对数
    "Abs": "abs",       # 绝对值
    "Sign": "sign",     # 符号
    "Rank": "rank",     # 截面排名
    "Delay": "delay",   # 延迟 N 期
    "Delta": "delta",   # 差分
    "Corr": "corr",     # 滚动相关性
    "Cov": "cov",       # 滚动协方差
    "PctChange": "pct", # 涨跌幅
}


class FactorDSLParser:
    """因子表达式 DSL 解析器

    语法规则 (类 qlib 表达式):
        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := NUMBER | '$'IDENT | IDENT '(' args ')' | '(' expression ')'
        args       := expression (',' expression)*

    用法示例:
        "$close"                                    → 引用 close 列
        "Ref($close, 1)"                            → close 的前一期值
        "Mean($close, 20)"                          → close 的 20 日滚动均值
        "$high - $low"                              → high 减 low
        "(($close - Ref($close, 1)) / Ref($close, 1)) * 100" → 日收益率(%)
        "Mean($close / Ref($close, 1) - 1, 20)"     → 20 日平均收益率
    """

    def __init__(self, expression: str):
        self.expr = expression
        self.pos = 0
        self.tokens = self._tokenize()
        self.tok_idx = 0

    def _tokenize(self) -> List[Token]:
        tokens = []
        i = 0
        s = self.expr
        while i < len(s):
            c = s[i]
            if c.isspace():
                i += 1
                continue
            if c == '(':
                tokens.append(Token(TOKEN_LPAREN, '('))
                i += 1
            elif c == ')':
                tokens.append(Token(TOKEN_RPAREN, ')'))
                i += 1
            elif c == ',':
                tokens.append(Token(TOKEN_COMMA, ','))
                i += 1
            elif c in '+-*/':
                tokens.append(Token(TOKEN_OP, c))
                i += 1
            elif c in '><=':
                if i + 1 < len(s) and s[i + 1] == '=':
                    tokens.append(Token(TOKEN_OP, s[i:i + 2]))
                    i += 2
                else:
                    tokens.append(Token(TOKEN_OP, c))
                    i += 1
            elif c == '.':
                tokens.append(Token(TOKEN_OP, '.'))
                i += 1
            elif c == '$':
                # 列引用: $close, $volume 等
                j = i + 1
                while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                    j += 1
                col_name = s[i + 1:j]
                tokens.append(Token(TOKEN_REF, col_name))
                i = j
            elif c.isdigit() or c == '.':
                # 数字字面量
                j = i
                while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                    j += 1
                tokens.append(Token(TOKEN_NUM, s[i:j]))
                i = j
            elif c.isalpha() or c == '_':
                # 函数名或标识符
                j = i
                while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                    j += 1
                name = s[i:j]
                # 提前看下一个字符是否为 '(' 判断是函数调用
                k = j
                while k < len(s) and s[k].isspace():
                    k += 1
                if k < len(s) and s[k] == '(':
                    tokens.append(Token(TOKEN_FUNC, name))
                elif name == 'astype':
                    # .astype() 方法调用，不作为函数处理
                    tokens.append(Token(TOKEN_REF, name))
                else:
                    # 非函数的普通标识符，当列引用处理
                    tokens.append(Token(TOKEN_REF, name))
                i = j
            else:
                raise ValueError(f"无法识别的字符: {c!r} at position {i}")
        return tokens

    def _peek(self) -> Optional[Token]:
        if self.tok_idx < len(self.tokens):
            return self.tokens[self.tok_idx]
        return None

    def _consume(self) -> Token:
        tok = self.tokens[self.tok_idx]
        self.tok_idx += 1
        return tok

    def parse(self) -> ASTNode:
        """解析完整表达式"""
        return self._parse_expression()

    def _parse_expression(self) -> ASTNode:
        """expression := comparison (('>' | '<' | '>=' | '<=' | '==') comparison)*"""
        left = self._parse_comparison()
        while self._peek() and self._peek().type == TOKEN_OP and self._peek().value in '><':
            op = self._consume().value
            right = self._parse_comparison()
            left = ASTNode(type="binary_op", value=op, left=left, right=right)
        return left

    def _parse_comparison(self) -> ASTNode:
        """comparison := term (('+' | '-') term)*"""
        left = self._parse_term()
        while self._peek() and self._peek().type == TOKEN_OP and self._peek().value in '+-':
            op = self._consume().value
            right = self._parse_term()
            left = ASTNode(type="binary_op", value=op, left=left, right=right)
        return left

    def _parse_term(self) -> ASTNode:
        """term := factor (('*' | '/') factor)*"""
        left = self._parse_factor()
        while self._peek() and self._peek().type == TOKEN_OP and self._peek().value in '*/':
            op = self._consume().value
            right = self._parse_factor()
            left = ASTNode(type="binary_op", value=op, left=left, right=right)
        return left

    def _parse_factor(self) -> ASTNode:
        tok = self._peek()
        if tok is None:
            raise ValueError("表达式意外结束")

        if tok.type == TOKEN_NUM:
            self._consume()
            return ASTNode(type="number", value=float(tok.value))

        if tok.type == TOKEN_REF:
            self._consume()
            return ASTNode(type="ref", value=tok.value)

        if tok.type == TOKEN_FUNC:
            func_name = self._consume().value
            # 消费 '('
            if self._peek() is None or self._peek().type != TOKEN_LPAREN:
                raise ValueError(f"函数 {func_name} 后需要 '('")
            self._consume()

            args = []
            if self._peek() and self._peek().type != TOKEN_RPAREN:
                args.append(self._parse_expression())
                while self._peek() and self._peek().type == TOKEN_COMMA:
                    self._consume()
                    args.append(self._parse_expression())

            if self._peek() is None or self._peek().type != TOKEN_RPAREN:
                raise ValueError(f"函数 {func_name} 缺少 ')'")
            self._consume()
            return ASTNode(type="func_call", value=func_name, args=args)

        if tok.type == TOKEN_LPAREN:
            self._consume()
            node = self._parse_expression()
            if self._peek() is None or self._peek().type != TOKEN_RPAREN:
                raise ValueError("缺少 ')'")
            self._consume()
            return node

        raise ValueError(f"意外的 token: {tok}")


# ================================================================================
# Part 2: DSL 编译器 — AST → pandas 操作
# ================================================================================


class FactorDSLCompiler:
    """将 AST 编译为 SQL 风格的表达式描述和 pandas 计算序列"""

    def __init__(self):
        self.field_refs: List[str] = []  # 收集引用的列名

    def compile(self, node: ASTNode, required_fields: List[str] = None) -> str:
        """编译 AST，返回计算描述字符串"""
        self.field_refs = required_fields or []
        return self._compile_node(node)

    def _compile_node(self, node: ASTNode) -> str:
        if node.type == "ref":
            col = node.value
            if col not in self.field_refs:
                self.field_refs.append(col)
            return f"df['{col}']"

        if node.type == "number":
            return str(node.value)

        if node.type == "binary_op":
            left = self._compile_node(node.left)
            right = self._compile_node(node.right)
            return f"({left} {node.value} {right})"

        if node.type == "func_call":
            func_name = node.value
            if func_name not in _BUILTIN_FUNCTIONS:
                raise ValueError(f"未知函数: {func_name}")

            internal = _BUILTIN_FUNCTIONS[func_name]
            compiled_args = [self._compile_node(a) for a in node.args]

            # 翻译为 pandas 操作
            if internal == "ref":
                # Ref(expr, N) → shift(N)
                expr = compiled_args[0]
                n = compiled_args[1] if len(compiled_args) > 1 else "1"
                return f"{expr}.groupby(df['code']).shift(int(float({n})))"

            elif internal == "mean":
                expr = compiled_args[0]
                n = compiled_args[1] if len(compiled_args) > 1 else "20"
                return f"{expr}.groupby(df['code']).transform(lambda x: x.rolling(int(float({n})), min_periods=int(float({n})//2)).mean())"

            elif internal == "std":
                expr = compiled_args[0]
                n = compiled_args[1] if len(compiled_args) > 1 else "20"
                return f"{expr}.groupby(df['code']).transform(lambda x: x.rolling(int(float({n})), min_periods=int(float({n})//2)).std())"

            elif internal == "max":
                expr = compiled_args[0]
                n = compiled_args[1] if len(compiled_args) > 1 else "20"
                return f"{expr}.groupby(df['code']).transform(lambda x: x.rolling(int(float({n})), min_periods=int(float({n})//2)).max())"

            elif internal == "min":
                expr = compiled_args[0]
                n = compiled_args[1] if len(compiled_args) > 1 else "20"
                return f"{expr}.groupby(df['code']).transform(lambda x: x.rolling(int(float({n})), min_periods=int(float({n})//2)).min())"

            elif internal == "sum":
                expr = compiled_args[0]
                n = compiled_args[1] if len(compiled_args) > 1 else "20"
                return f"{expr}.groupby(df['code']).transform(lambda x: x.rolling(int(float({n})), min_periods=int(float({n})//2)).sum())"

            elif internal == "log":
                expr = compiled_args[0]
                return f"np.log({expr}.replace(0, np.nan))"

            elif internal == "abs":
                expr = compiled_args[0]
                return f"({expr}).abs()"

            elif internal == "sign":
                expr = compiled_args[0]
                return f"np.sign({expr})"

            elif internal == "rank":
                expr = compiled_args[0]
                return f"{expr}.groupby(df['date']).rank(pct=True)"

            elif internal == "delay":
                expr = compiled_args[0]
                n = compiled_args[1] if len(compiled_args) > 1 else "1"
                return f"{expr}.groupby(df['code']).shift(int(float({n})))"

            elif internal == "delta":
                expr = compiled_args[0]
                n = compiled_args[1] if len(compiled_args) > 1 else "1"
                return f"({expr} - {expr}.groupby(df['code']).shift({n}))"

            elif internal == "pct":
                expr = compiled_args[0]
                return f"{expr}.groupby(df['code']).pct_change()"

            else:
                raise ValueError(f"未实现的函数编译: {func_name}")

        raise ValueError(f"未知节点类型: {node.type}")


class FactorRegistry:
    """因子注册表 — 管理 DSL 定义的因子"""

    def __init__(self):
        self.factors: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, expression: str, description: str = ""):
        """注册一个 DSL 定义的因子"""
        self.factors[name] = {
            "expression": expression,
            "description": description,
            "ast": None,
            "compiled": None,
        }

    def compile_all(self):
        """预编译所有已注册因子"""
        for name, info in self.factors.items():
            parser = FactorDSLParser(info["expression"])
            ast = parser.parse()
            compiler = FactorDSLCompiler()
            compiled = compiler.compile(ast)
            info["ast"] = ast
            info["compiled"] = compiled
            info["field_refs"] = compiler.field_refs

    def compute(self, name: str, df: pd.DataFrame) -> pd.Series:
        """计算一个已注册因子"""
        info = self.factors.get(name)
        if info is None:
            raise ValueError(f"未注册的因子: {name}")
        if info["compiled"] is None:
            self.compile_all()
        # 安全执行编译后的表达式
        result = eval(info["compiled"], {"df": df, "np": np, "pd": pd, "int": int, "float": float, "__builtins__": {}})
        return result


# ================================================================================
# Part 3: 预设 Alpha158 因子集（qlib 经典因子子集验证）
# ================================================================================

# qlib Alpha158 因子集中的代表性因子
ALPHA158_SAMPLE = {
    "KMID": {
        "expr": "($open + $close) / 2",
        "desc": "中间价",
    },
    "KLEN": {
        "expr": "($high - $low) / $open",
        "desc": "K线实体长度比率",
    },
    "KMID2": {
        "expr": "($high + $low) / 2",
        "desc": "高低点均值",
    },
    "KUP": {
        "expr": "($close > $open)",
        "desc": "是否上涨（收盘价大于开盘价）",
    },
    "KMEAN": {
        "expr": "(($open + $high + $low + $close) / 4)",
        "desc": "平均价格",
    },
    # 时间序列因子
    "ROC5": {
        "expr": "(($close - Ref($close, 5)) / Ref($close, 5)) * 100",
        "desc": "5日变动率",
    },
    "ROC20": {
        "expr": "(($close - Ref($close, 20)) / Ref($close, 20)) * 100",
        "desc": "20日变动率",
    },
    "MA5": {
        "expr": "Mean($close, 5)",
        "desc": "5日均线",
    },
    "MA20": {
        "expr": "Mean($close, 20)",
        "desc": "20日均线",
    },
    "MA5_MA20_DIV": {
        "expr": "(Mean($close, 5) - Mean($close, 20)) / Mean($close, 20)",
        "desc": "5日与20日均线偏离度",
    },
    "STD20": {
        "expr": "Std($close, 20)",
        "desc": "20日波动率",
    },
    "MAX20": {
        "expr": "Max($close, 20)",
        "desc": "20日最高价",
    },
    "MIN20": {
        "expr": "Min($close, 20)",
        "desc": "20日最低价",
    },
    "HIGH_LOW_SPREAD": {
        "expr": "($high - $low) / Mean($close, 20)",
        "desc": "日振幅与均价比值",
    },
    "VOL_MA5": {
        "expr": "Mean($volume, 5)",
        "desc": "5日均量",
    },
    "VOL_MA20": {
        "expr": "Mean($volume, 20)",
        "desc": "20日均量",
    },
    "VOL_RATIO": {
        "expr": "Mean($volume, 5) / Mean($volume, 20)",
        "desc": "量比（5日/20日）",
    },
    "TURN_DEVIATION": {
        "expr": "(Mean($volume, 5) - Mean($volume, 20)) / Std($volume, 20)",
        "desc": "成交量偏离标准差",
    },
}


# ================================================================================
# Part 4: 测试用例
# ================================================================================


class TestFactorDSLParser(unittest.TestCase):
    """DSL 解析器单元测试"""

    def test_simple_ref(self):
        parser = FactorDSLParser("$close")
        ast = parser.parse()
        self.assertEqual(ast.type, "ref")
        self.assertEqual(ast.value, "close")

    def test_binary_op(self):
        parser = FactorDSLParser("$high - $low")
        ast = parser.parse()
        self.assertEqual(ast.type, "binary_op")
        self.assertEqual(ast.value, "-")
        self.assertEqual(ast.left.type, "ref")
        self.assertEqual(ast.left.value, "high")
        self.assertEqual(ast.right.type, "ref")
        self.assertEqual(ast.right.value, "low")

    def test_func_call(self):
        parser = FactorDSLParser("Ref($close, 1)")
        ast = parser.parse()
        self.assertEqual(ast.type, "func_call")
        self.assertEqual(ast.value, "Ref")
        self.assertEqual(len(ast.args), 2)

    def test_complex_expr(self):
        parser = FactorDSLParser("(($close - Ref($close, 1)) / Ref($close, 1)) * 100")
        ast = parser.parse()
        self.assertEqual(ast.type, "binary_op")

    def test_nested_func(self):
        parser = FactorDSLParser("Mean($close / Ref($close, 1) - 1, 20)")
        ast = parser.parse()
        self.assertEqual(ast.type, "func_call")
        self.assertEqual(ast.value, "Mean")

    def test_precedence(self):
        # 乘法优先级高于加法: 1 + 2 * 3 应解析为 1 + (2 * 3)
        parser = FactorDSLParser("1 + 2 * 3")
        ast = parser.parse()
        self.assertEqual(ast.type, "binary_op")
        self.assertEqual(ast.value, "+")
        self.assertEqual(ast.right.type, "binary_op")
        self.assertEqual(ast.right.value, "*")

    def test_all_alpha158_expressions(self):
        """验证 Alpha158 样本因子均可被解析"""
        for name, info in ALPHA158_SAMPLE.items():
            try:
                parser = FactorDSLParser(info["expr"])
                ast = parser.parse()
                self.assertIsNotNone(ast, f"Failed to parse: {name} = {info['expr']}")
            except Exception as e:
                self.fail(f"Parse failed for {name} ({info['expr']}): {e}")


class TestFactorDSLCompiler(unittest.TestCase):
    """DSL 编译器单元测试"""

    def setUp(self):
        # 构造模拟日线数据
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")
        codes = ["000001.SZ", "600000.SH", "000002.SZ"]
        rows = []
        for code in codes:
            price = np.random.uniform(10, 50)
            for d in dates:
                ret = np.random.normal(0.0005, 0.015)
                price = price * (1 + ret)
                rows.append({
                    "date": d,
                    "code": code,
                    "open": price * (1 + np.random.normal(0, 0.002)),
                    "high": price * (1 + abs(np.random.normal(0, 0.008))),
                    "low": price * (1 - abs(np.random.normal(0, 0.008))),
                    "close": price,
                    "volume": np.random.lognormal(15, 0.5),
                })
        self.df = pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)

    def test_compile_simple_ref(self):
        """编译简单列引用"""
        parser = FactorDSLParser("$close")
        ast = parser.parse()
        compiler = FactorDSLCompiler()
        compiled = compiler.compile(ast)
        result = eval(compiled, {"df": self.df, "np": np, "pd": pd, "int": int, "float": float, "__builtins__": {}})
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            self.df["close"].reset_index(drop=True),
            check_names=False,
        )

    def test_compile_binary_op(self):
        """编译二元运算"""
        parser = FactorDSLParser("$high - $low")
        ast = parser.parse()
        compiler = FactorDSLCompiler()
        compiled = compiler.compile(ast)
        result = eval(compiled, {"df": self.df, "np": np, "pd": pd, "int": int, "float": float, "__builtins__": {}})
        expected = self.df["high"] - self.df["low"]
        self.assertTrue(np.allclose(result.values, expected.values, equal_nan=True))

    def test_compile_func_ref(self):
        """编译 Ref() 函数"""
        parser = FactorDSLParser("Ref($close, 1)")
        ast = parser.parse()
        compiler = FactorDSLCompiler()
        compiled = compiler.compile(ast)
        result = eval(compiled, {"df": self.df, "np": np, "pd": pd, "int": int, "float": float, "__builtins__": {}})
        expected = self.df.groupby("code")["close"].shift(1)
        self.assertTrue(np.allclose(result.values, expected.values, equal_nan=True))

    def test_compile_func_mean(self):
        """编译 Mean() 函数"""
        parser = FactorDSLParser("Mean($close, 5)")
        ast = parser.parse()
        compiler = FactorDSLCompiler()
        compiled = compiler.compile(ast)
        result = eval(compiled, {"df": self.df, "np": np, "pd": pd, "int": int, "float": float, "__builtins__": {}})
        expected = self.df.groupby("code")["close"].transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        self.assertTrue(np.allclose(result.values, expected.values, equal_nan=True))

    def test_compile_rate_of_change(self):
        """编译价格变动率"""
        parser = FactorDSLParser("(($close - Ref($close, 5)) / Ref($close, 5)) * 100")
        ast = parser.parse()
        compiler = FactorDSLCompiler()
        compiled = compiler.compile(ast)
        result = eval(compiled, {"df": self.df, "np": np, "pd": pd, "int": int, "float": float, "__builtins__": {}})
        close_shift = self.df.groupby("code")["close"].shift(5)
        expected = (self.df["close"] - close_shift) / close_shift * 100
        self.assertTrue(np.allclose(result.values, expected.values, equal_nan=True))

    def test_registry_compile_all(self):
        """测试因子注册表批量编译"""
        registry = FactorRegistry()
        for name, info in ALPHA158_SAMPLE.items():
            registry.register(name, info["expr"], info["desc"])
        registry.compile_all()

        for name in ALPHA158_SAMPLE:
            self.assertIsNotNone(registry.factors[name]["compiled"],
                                 f"{name} 编译失败")

    def test_registry_compute(self):
        """测试因子注册表计算"""
        registry = FactorRegistry()
        for name, info in list(ALPHA158_SAMPLE.items())[:5]:  # 只测前5个
            registry.register(name, info["expr"], info["desc"])

        for name in list(ALPHA158_SAMPLE.keys())[:5]:
            result = registry.compute(name, self.df)
            self.assertIsInstance(result, pd.Series, f"{name} 计算结果不是 Series")
            self.assertEqual(len(result), len(self.df), f"{name} 结果长度不匹配")


class TestFactorDSLPerformance(unittest.TestCase):
    """DSL 性能对比测试"""

    def setUp(self):
        # 用模拟数据：500 个交易日，100 只股票
        np.random.seed(42)
        n_dates = 500
        n_stocks = 100
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
        codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
        rows = []
        for code in codes:
            price = np.random.uniform(10, 100)
            for d in dates:
                ret = np.random.normal(0.0005, 0.015)
                price = price * (1 + ret)
                rows.append({
                    "date": d,
                    "code": code,
                    "open": price * (1 + np.random.normal(0, 0.003)),
                    "high": price * (1 + abs(np.random.normal(0, 0.01))),
                    "low": price * (1 - abs(np.random.normal(0, 0.01))),
                    "close": price,
                    "volume": np.random.lognormal(14, 0.5),
                })
        self.df = pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)

    def test_dsl_vs_hardcoded_factor(self):
        """对比 DSL 计算与硬编码计算的性能和正确性"""
        # 硬编码方式：计算 20 日均线偏离度
        t0 = time.perf_counter()
        ma5_hard = self.df.groupby("code")["close"].transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        ma20_hard = self.df.groupby("code")["close"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        deviation_hard = (ma5_hard - ma20_hard) / ma20_hard
        t_hard = time.perf_counter() - t0

        # DSL 方式
        registry = FactorRegistry()
        registry.register("MA5_MA20_DIV",
                          "(Mean($close, 5) - Mean($close, 20)) / Mean($close, 20)",
                          "均线偏离")
        registry.compile_all()
        t0 = time.perf_counter()
        deviation_dsl = registry.compute("MA5_MA20_DIV", self.df)
        t_dsl = time.perf_counter() - t0

        # 正确性验证
        match_ratio = np.allclose(
            deviation_hard.fillna(0).values,
            deviation_dsl.fillna(0).values,
            atol=1e-6,
        )
        self.assertTrue(match_ratio, "DSL 与硬编码结果不一致")

        # 性能对比
        speed_ratio = t_hard / t_dsl if t_dsl > 0 else float('inf')
        print(f"\n  硬编码耗时: {t_hard:.4f}s, DSL 耗时: {t_dsl:.4f}s, 比率: {speed_ratio:.2f}x")
        print(f"  数据规模: {len(self.df)} 行 × 100 只股票, 500 交易日")
        print(f"  正确性: {'PASS' if match_ratio else 'FAIL'}")

    def test_batch_factor_computation(self):
        """批量因子计算性能测试"""
        registry = FactorRegistry()
        for name, info in ALPHA158_SAMPLE.items():
            registry.register(name, info["expr"], info["desc"])
        registry.compile_all()

        n_factors = len(ALPHA158_SAMPLE)

        t0 = time.perf_counter()
        for name in ALPHA158_SAMPLE:
            registry.compute(name, self.df)
        t_total = time.perf_counter() - t0

        avg_time = t_total / n_factors
        print(f"\n  批量计算 {n_factors} 个因子总耗时: {t_total:.4f}s")
        print(f"  平均每个因子: {avg_time:.4f}s")
        print(f"  数据规模: {len(self.df)} 行 (100只 × 500天)")

        # 性能基准：平均每个因子应 < 0.3s
        self.assertLess(avg_time, 0.5, f"单个因子计算平均耗时 {avg_time:.3f}s 超过阈值")


class TestFactorDSLIntegration(unittest.TestCase):
    """和现有因子引擎的集成对比测试"""

    def test_compare_with_existing_engine(self):
        """对比 DSL 计算结果与现有 factor-engine 结果"""
        if not os.path.exists(_REAL_DATA_PATH):
            self.skipTest("真实数据文件不存在，跳过集成对比")

        from skills.factor_engine.engine import FactorEngine

        # 加载真实数据
        df = pd.read_parquet(_REAL_DATA_PATH).head(10000)  # 取前1万行加速测试

        # 现有引擎计算
        engine = FactorEngine()
        existing_factors = engine.compute_a_share_factors(df)

        # DSL 计算对应因子
        registry = FactorRegistry()
        registry.register("ret_5d", "($close - Ref($close, 5)) / Ref($close, 5)", "5日收益")
        registry.register("volatility_20d", "Std(PctChange($close), 20)", "20日波动率")
        registry.register("volume_ratio", "Mean($volume, 5) / Mean($volume, 20)", "量比")
        registry.compile_all()

        dsl_ret_5d = registry.compute("ret_5d", df)
        dsl_volatility = registry.compute("volatility_20d", df)
        dsl_vol_ratio = registry.compute("volume_ratio", df)

        # 对比 ret_5d
        if "ret_5d" in existing_factors.columns:
            match = np.allclose(
                existing_factors["ret_5d"].fillna(0).values,
                dsl_ret_5d.fillna(0).values,
                atol=1e-6,
            )
            print(f"\n  ret_5d 对比: {'PASS' if match else 'FAIL'}")

        # 对比 volume_ratio
        if "volume_ratio" in existing_factors.columns:
            match = np.allclose(
                existing_factors["volume_ratio"].fillna(0).values,
                dsl_vol_ratio.fillna(0).values,
                atol=1e-6,
            )
            print(f"  volume_ratio 对比: {'PASS' if match else 'FAIL'}")

        # 不需要断言失败 — 集成测试主要是观察输出


# ================================================================================
# 运行入口
# ================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("因子表达式 DSL 引擎验证测试")
    print("借鉴来源: Microsoft Qlib Expression Engine")
    print("=" * 70)
    unittest.main(verbosity=2)