"""
因子表达式引擎核心
==================

实现类 Qlib / AKQuant 的 Expression Engine：
- 词法分析 (Tokenizer)
- 语法分析 (Parser) — 递归下降，支持二元运算符
- 求值 (Evaluator) — 表达式编译为可调用对象

DSL 语法：

    # 基本变量与算术
    Mom_20 = ($close - Delay($close, 20)) / Delay($close, 20)

    # 时间序列算子
    Mom_5 = $close / Delay($close, 5) - 1
    Vol_20 = Ts_Std($close, 20)
    Rev_5 = Sub(0, Delta($close, 5))                # 5 日反转
    ZScore_5 = ZScore(Sub($close, Ts_Mean($close, 5)))

    # 横截面算子
    RankRev = Rank(Rev_5)
    Alpha_Cs = Scale(Add(ZScore($close), ZScore($volume)))

    # 逻辑与条件
    Bull = If(Greater($close, Ts_Mean($close, 20)), 1, 0)
    Alpha_Composite = If(Greater(RankRev, 0.5), RankRev, 0)
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .operators import OPERATORS, ARITY, _resolve_var


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_SPEC = [
    ("NUMBER", r"\d+(\.\d+)?"),
    ("VARIABLE", r"\$[A-Za-z_][A-Za-z0-9_]*"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP", r"[+\-*/]"),                      # 二元运算符
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("WS", r"\s+"),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))


class Token:
    __slots__ = ("kind", "value", "pos")

    def __init__(self, kind: str, value: str, pos: int):
        self.kind = kind
        self.value = value
        self.pos = pos

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r})"


def tokenize(expr: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise SyntaxError(f"无法识别的字符: {expr[pos]!r} (位置 {pos})")
        pos = m.end()
        kind = m.lastgroup
        if kind == "WS":
            continue
        tokens.append(Token(kind, m.group(kind), m.start()))
    return tokens


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

Scalar = Union[int, float]


class NumberNode:
    """常数节点。is_constant=True，evaluate 返回 Python 标量。"""

    is_constant = True

    __slots__ = ("value",)

    def __init__(self, value: Scalar):
        self.value = value

    def evaluate(self, df: pd.DataFrame):  # type: ignore[override]
        return self.value


class VarNode:
    """变量节点（$close 等）。"""

    is_constant = False

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return _resolve_var(self.name, df)


class CallNode:
    """函数调用节点。"""

    is_constant = False

    __slots__ = ("name", "args")

    def __init__(self, name: str, args: List):
        self.name = name
        self.args = args

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.name not in OPERATORS:
            raise NameError(f"未知算子: {self.name}")
        arity = ARITY[self.name]
        if arity != -1 and arity != len(self.args):
            raise TypeError(
                f"算子 {self.name} 需要 {arity} 个参数，实际传入 {len(self.args)} 个"
            )
        evaluated_args = [
            a.value if isinstance(a, NumberNode) else a.evaluate(df)
            for a in self.args
        ]
        result = OPERATORS[self.name](df, *evaluated_args)
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=df.index)
        if result.name is None:
            result.name = self.name
        return result


class BinOpNode:
    """二元运算符节点：+ - * /。"""

    is_constant = False

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.value if isinstance(self.left, NumberNode) else self.left.evaluate(df)
        r = self.right.value if isinstance(self.right, NumberNode) else self.right.evaluate(df)
        if self.op == "+":
            out = l + r
        elif self.op == "-":
            out = l - r
        elif self.op == "*":
            out = l * r
        elif self.op == "/":
            if isinstance(r, pd.Series):
                out = l / r.replace(0, np.nan)
            else:
                out = l / r if r != 0 else np.nan
        else:
            raise ValueError(f"未知二元运算符: {self.op}")
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=df.index)
        if out.name is None:
            out.name = f"BinOp_{self.op}"
        return out


# ---------------------------------------------------------------------------
# Parser (Recursive Descent, with binary operators)
# ---------------------------------------------------------------------------
#
#  grammar:
#    expression := term (('+' | '-') term)*
#    term       := factor (('*' | '/') factor)*
#    factor     := unary
#    unary      := '-' unary | primary
#    primary    := NUMBER | VARIABLE | call | '(' expression ')'
#    call       := IDENT '(' arglist? ')'
#    arglist    := expression (',' expression)*
#


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Optional[Token]:
        idx = self.pos + offset
        if 0 <= idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, kind: Optional[str] = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("意外的表达式末尾")
        if kind and tok.kind != kind:
            raise SyntaxError(f"期望 {kind}，得到 {tok.kind} ({tok.value!r})")
        self.pos += 1
        return tok

    def parse(self):
        node = self.parse_expression()
        if self.peek() is not None:
            raise SyntaxError(f"表达式末尾有多余 token: {self.peek()}")
        return node

    def parse_expression(self):
        node = self.parse_term()
        while self.peek() and self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op_tok = self.consume()
            right = self.parse_term()
            node = BinOpNode(op_tok.value, node, right)
        return node

    def parse_term(self):
        node = self.parse_unary()
        while self.peek() and self.peek().kind == "OP" and self.peek().value in ("*", "/"):
            op_tok = self.consume()
            right = self.parse_unary()
            node = BinOpNode(op_tok.value, node, right)
        return node

    def parse_unary(self):
        if self.peek() and self.peek().kind == "OP" and self.peek().value == "-":
            self.consume()
            inner = self.parse_unary()
            # -x  =  0 - x
            return BinOpNode("-", NumberNode(0), inner)
        return self.parse_primary()

    def parse_primary(self):
        tok = self.consume()
        if tok.kind == "NUMBER":
            return NumberNode(float(tok.value) if "." in tok.value else int(tok.value))
        if tok.kind == "VARIABLE":
            return VarNode(tok.value)
        if tok.kind == "LPAREN":
            inner = self.parse_expression()
            self.consume("RPAREN")
            return inner
        if tok.kind == "IDENT":
            return self.finish_call(tok)
        raise SyntaxError(f"意外的 token: {tok}")

    def finish_call(self, ident_tok: Token) -> CallNode:
        if self.peek() is None or self.peek().kind != "LPAREN":
            raise SyntaxError(
                f"标识符 {ident_tok.value!r} 必须紧跟括号作为函数调用"
            )
        self.consume("LPAREN")
        args = self.parse_arglist()
        self.consume("RPAREN")
        return CallNode(ident_tok.value, args)

    def parse_arglist(self) -> List:
        args = []
        if self.peek() and self.peek().kind == "RPAREN":
            return args
        args.append(self.parse_expression())
        while self.peek() and self.peek().kind == "COMMA":
            self.consume("COMMA")
            args.append(self.parse_expression())
        return args


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_formula(formula: str):
    tokens = tokenize(formula)
    if not tokens:
        raise SyntaxError("公式为空")
    return Parser(tokens).parse()


def compile_formula(formula: str) -> Callable[[pd.DataFrame], pd.Series]:
    ast = parse_formula(formula)
    return lambda df: ast.evaluate(df)


class FactorExpressionEngine:
    """批量编译 / 计算多因子表达式。"""

    def __init__(self) -> None:
        self._compiled: Dict[str, Callable] = {}

    def register(self, name: str, formula: str) -> None:
        self._compiled[name] = compile_formula(formula)

    def list_factors(self) -> List[str]:
        return list(self._compiled.keys())

    def compute(self, name: str, df: pd.DataFrame) -> pd.Series:
        if name not in self._compiled:
            raise KeyError(f"因子 {name} 未注册")
        return self._compiled[name](df)

    def compute_all(
        self, df: pd.DataFrame, factor_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        names = factor_names or self.list_factors()
        result = df[["code", "date"]].copy()
        for n in names:
            result[n] = self.compute(n, df).values
        return result

    def evaluate_formula(self, formula: str, df: pd.DataFrame) -> pd.Series:
        return compile_formula(formula)(df)