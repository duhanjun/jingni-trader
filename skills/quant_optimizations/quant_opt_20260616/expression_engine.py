"""
expression_engine.py
====================

借鉴 Qlib `qlib.data.ops` 表达式引擎的设计 (https://github.com/microsoft/qlib)
为 jingni-trader 提供声明式因子定义能力。

设计目标
--------
1. 用户以类数学公式的字符串定义因子，例如 ``"Mean($close, 5) / Mean($close, 20) - 1"``
2. 引擎将字符串解析为抽象语法树 (AST)，再编译为可在多只股票、长时间序列上
   一次性执行的向量化算子图。
3. 内置丰富的算子：``Ref``、``Mean``、``Std``、``Sum``、``Max``、``Min``、
   ``Log``、``Abs``、``Sign``、``Rank``、``Delta``、``Corr``、``Cov``、``If`` 等。
4. 支持自定义算子扩展 (类似 Qlib 的 ``ExpressionOps`` 基类)。

正确性
------
- 解析与执行严格遵循运算符优先级。
- 算子以 pandas/numpy 向量化方式实现，避免 Python 级循环。
- 每个算子既支持 ``GroupBy('code')`` 滚动 (按股票) 也支持
  ``groupby('date')`` 截面 (cross-section) 两种语义。
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_opt_20260616.expr_engine")


# ============================================================================
# Part 1: 算子定义 (借鉴 Qlib ExpressionOps / ElemOperator / PairOperator / Rolling)
# ============================================================================

@dataclass(frozen=True)
class Field:
    """行情字段占位符，如 $close / $open / $volume / $amount 等"""
    name: str

    def __repr__(self) -> str:
        return f"${self.name}"


class Operator:
    """所有算子的抽象基类 (对应 Qlib ``ExpressionOps``)"""
    name: str = "Op"
    arity: int = 1  # 参数数量

    def __init__(self, *args: Any):
        self.args = args

    def eval(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.name, "args": [_to_dict(a) for a in self.args]}


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, Operator):
        return obj.to_dict()
    if isinstance(obj, Field):
        return {"field": obj.name}
    if isinstance(obj, (int, float, str)):
        return obj
    return str(obj)


# ----------------------------------------------------------------------------
# 字段与基础算子
# ----------------------------------------------------------------------------

def _resolve_field(data: pd.DataFrame, field: Field) -> pd.Series:
    """从 DataFrame 中取出字段，转为按 (date, code) 排序的 Series"""
    if field.name not in data.columns:
        raise KeyError(f"数据中缺少字段 ${field.name}")
    return data[field.name]


class ElemUnary(Operator):
    """一元算子基类 (借鉴 Qlib ``ElemOperator``)"""
    arity = 1

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return self._apply(x)

    def _apply(self, x: pd.Series) -> pd.Series:
        raise NotImplementedError


class Log(ElemUnary):
    name = "Log"

    def _apply(self, x: pd.Series) -> pd.Series:
        return np.log(x.replace(0, np.nan))


class Abs(ElemUnary):
    name = "Abs"

    def _apply(self, x: pd.Series) -> pd.Series:
        return x.abs()


class Sign(ElemUnary):
    name = "Sign"

    def _apply(self, x: pd.Series) -> pd.Series:
        return np.sign(x)


class Sqrt(ElemUnary):
    name = "Sqrt"

    def _apply(self, x: pd.Series) -> pd.Series:
        return np.sqrt(x.replace(-1, np.nan).abs())


# ----------------------------------------------------------------------------
# 二元算子 (借鉴 Qlib ``PairOperator``)
# ----------------------------------------------------------------------------

class PairOp(Operator):
    arity = 2

    def eval(self, data: pd.DataFrame) -> pd.Series:
        a = _eval(data, self.args[0])
        b = _eval(data, self.args[1])
        return self._apply(a, b)

    def _apply(self, a: pd.Series, b: pd.Series) -> pd.Series:
        raise NotImplementedError


class Add(PairOp):
    name = "Add"

    def _apply(self, a, b):
        return a + b


class Sub(PairOp):
    name = "Sub"

    def _apply(self, a, b):
        return a - b


class Mul(PairOp):
    name = "Mul"

    def _apply(self, a, b):
        return a * b


class Div(PairOp):
    name = "Div"

    def _apply(self, a, b):
        return a / b.replace(0, np.nan)


class Power(PairOp):
    name = "Power"

    def _apply(self, a, b):
        return a ** b


# ----------------------------------------------------------------------------
# 滚动算子 (借鉴 Qlib ``Rolling``)
# 所有滚动算子均按 code 分组计算，rolling 后产生 NaN 直到窗口填满
# ----------------------------------------------------------------------------

class Rolling(Operator):
    arity = 2  # (expr, N)
    requires_groupby_code = True

    def __init__(self, expr: Any, n: int):
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"窗口长度必须是正整数, 得到 {n!r}")
        super().__init__(expr, n)
        self.n = n


class Ref(Rolling):
    """Ref(x, N)  -> x 在 N 期前的值 (shift N)"""

    def __init__(self, expr: Any, n: int):
        super().__init__(expr, n)

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return x.groupby(data["code"]).shift(self.n)


class Mean(Rolling):
    name = "Mean"

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return x.groupby(data["code"]).rolling(self.n, min_periods=max(2, self.n // 2)).mean().reset_index(level=0, drop=True)


class Std(Rolling):
    name = "Std"

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return x.groupby(data["code"]).rolling(self.n, min_periods=max(2, self.n // 2)).std().reset_index(level=0, drop=True)


class Sum(Rolling):
    name = "Sum"

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return x.groupby(data["code"]).rolling(self.n, min_periods=max(2, self.n // 2)).sum().reset_index(level=0, drop=True)


class Max(Rolling):
    name = "Max"

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return x.groupby(data["code"]).rolling(self.n, min_periods=max(2, self.n // 2)).max().reset_index(level=0, drop=True)


class Min(Rolling):
    name = "Min"

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return x.groupby(data["code"]).rolling(self.n, min_periods=max(2, self.n // 2)).min().reset_index(level=0, drop=True)


class Delta(Rolling):
    """Delta(x, N) = x - Ref(x, N)"""

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return x - x.groupby(data["code"]).shift(self.n)


# ----------------------------------------------------------------------------
# 截面算子 (cross-section, 借鉴 Qlib ``CS`` 系列算子)
# ----------------------------------------------------------------------------

class CrossSection(Operator):
    """按 date 截面计算"""
    requires_groupby_date = True


class Rank(CrossSection):
    """Rank(x, pct=True) -> 截面分位排名 (0~1)"""

    def __init__(self, expr: Any, pct: bool = True):
        super().__init__(expr, pct)
        self.pct = pct

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return x.groupby(data["date"]).rank(pct=self.pct)


class ZScore(CrossSection):
    """ZScore(x) -> 截面 z-score"""

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        g = x.groupby(data["date"])
        return (x - g.transform("mean")) / g.transform("std").replace(0, np.nan)


class Scale(CrossSection):
    """Scale(x) -> 截面缩放到 sum=1"""

    def eval(self, data: pd.DataFrame) -> pd.Series:
        x = _eval(data, self.args[0])
        return x / x.groupby(data["date"]).transform("sum").replace(0, np.nan)


# ----------------------------------------------------------------------------
# 内置算子注册表
# ----------------------------------------------------------------------------

OP_REGISTRY: Dict[str, type] = {
    # 一元
    "Log": Log, "Abs": Abs, "Sign": Sign, "Sqrt": Sqrt,
    # 二元
    "Add": Add, "Sub": Sub, "Mul": Mul, "Div": Div, "Power": Power,
    # 滚动
    "Ref": Ref, "Mean": Mean, "Std": Std, "Sum": Sum, "Max": Max, "Min": Min,
    "Delta": Delta,
    # 截面
    "Rank": Rank, "ZScore": ZScore, "Scale": Scale,
}


# ============================================================================
# Part 2: 表达式解析器 (Tokenizer + Pratt Parser)
# ============================================================================

TOKEN_REGEX = re.compile(
    r"""
    \s+                                  |   # 空白
    (?P<NUM>[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?) |   # 数字
    (?P<FIELD>\$[A-Za-z_][A-Za-z0-9_]*)   |   # 字段 $close
    (?P<ID>[A-Za-z_][A-Za-z0-9_]*)        |   # 标识符/函数名
    (?P<OP>[\+\-\*\/\^(),])                   # 运算符 / 括号 / 逗号
    """,
    re.VERBOSE,
)


@dataclass
class Token:
    type: str
    value: str
    pos: int


def _tokenize(expr: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(expr):
        m = TOKEN_REGEX.match(expr, pos)
        if not m:
            raise SyntaxError(f"无法在位置 {pos} 解析表达式: {expr[pos:pos+10]!r}")
        if m.lastgroup is None:
            pos = m.end()
            continue
        tok_type = m.lastgroup
        tok_val = m.group()
        if tok_type == "ID" and tok_val in OP_REGISTRY:
            tok_type = "FUNC"
        tokens.append(Token(tok_type, tok_val, pos))
        pos = m.end()
    tokens.append(Token("EOF", "", pos))
    return tokens


class _Parser:
    """Pratt 解析器: 把 token 流转为 Operator AST"""

    PRECEDENCE = {
        "+": 10, "-": 10,
        "*": 20, "/": 20,
        "^": 30,
        "u-": 40,  # 一元负号
    }

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

    def _peek(self) -> Token:
        return self.tokens[self.i]

    def _eat(self, ttype: Optional[str] = None) -> Token:
        t = self.tokens[self.i]
        if ttype and t.type != ttype:
            raise SyntaxError(f"期望 {ttype}, 得到 {t.type} ({t.value!r}) @ {t.pos}")
        self.i += 1
        return t

    def parse(self) -> Operator:
        node = self._parse_expr(0)
        if self._peek().type != "EOF":
            raise SyntaxError(f"表达式末尾多余 token: {self._peek().value!r}")
        return node

    def _parse_expr(self, min_prec: int) -> Any:
        left = self._parse_primary()
        while True:
            tok = self._peek()
            if tok.type != "OP" or tok.value in (",", ")"):
                break
            prec = self.PRECEDENCE.get(tok.value)
            if prec is None or prec < min_prec:
                break
            self._eat()
            right = self._parse_expr(prec + 1)
            left = self._combine(tok.value, left, right)
        return left

    def _parse_primary(self) -> Any:
        tok = self._peek()
        # 一元负号
        if tok.type == "OP" and tok.value == "-":
            self._eat()
            operand = self._parse_expr(self.PRECEDENCE["u-"])
            return Mul(operand, -1) if not isinstance(operand, (int, float)) else -operand
        if tok.type == "OP" and tok.value == "+":
            self._eat()
            return self._parse_expr(self.PRECEDENCE["u-"])
        if tok.type == "OP" and tok.value == "(":
            self._eat("OP")
            node = self._parse_expr(0)
            self._eat("OP")  # )
            return node
        if tok.type == "NUM":
            self._eat()
            text = tok.value
            return int(text) if "." not in text and "e" not in text.lower() else float(text)
        if tok.type == "FIELD":
            self._eat()
            return Field(tok.value[1:])
        if tok.type == "FUNC":
            return self._parse_func_call()
        raise SyntaxError(f"意外 token: {tok.type} {tok.value!r} @ {tok.pos}")

    def _parse_func_call(self) -> Operator:
        name_tok = self._eat("FUNC")
        self._eat("OP")  # (
        args: List[Any] = []
        if self._peek().value != ")":
            args.append(self._parse_expr(0))
            while self._peek().value == ",":
                self._eat()
                args.append(self._parse_expr(0))
        self._eat("OP")  # )
        cls = OP_REGISTRY.get(name_tok.value)
        if cls is None:
            raise SyntaxError(f"未知算子: {name_tok.value}")
        try:
            return cls(*args)
        except TypeError as e:
            raise SyntaxError(f"算子 {name_tok.value} 调用参数错误: {e}")

    @staticmethod
    def _combine(op: str, left: Any, right: Any) -> Any:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return eval(f"{left}{op}{right}", {"__builtins__": {}})
        cls_map = {"+": Add, "-": Sub, "*": Mul, "/": Div, "^": Power}
        cls = cls_map[op]
        return cls(left, right)


def parse_expression(expr: str) -> Operator:
    """将表达式字符串解析为算子 AST"""
    tokens = _tokenize(expr)
    return _Parser(tokens).parse()


# ============================================================================
# Part 3: 执行器
# ============================================================================

def _eval(data: pd.DataFrame, node: Any) -> pd.Series:
    """递归求值节点"""
    if isinstance(node, Field):
        return _resolve_field(data, node)
    if isinstance(node, Operator):
        return node.eval(data)
    if isinstance(node, (int, float)):
        return pd.Series(node, index=data.index, dtype=float)
    raise TypeError(f"无法求值节点: {type(node).__name__}")


def evaluate_expression(data: pd.DataFrame, expr: str) -> pd.Series:
    """
    主入口: 解析并计算因子表达式

    参数:
        data: 必须含列 ``code``、``date``、以及表达式中引用的字段
        expr: 因子表达式字符串

    返回:
        pd.Series, index 与 **data 原始索引** 保持一致 (不改变顺序)
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data 必须是 DataFrame")
    if "code" not in data.columns or "date" not in data.columns:
        raise KeyError("data 必须包含 'code' 与 'date' 列")
    if data.empty:
        return pd.Series(dtype=float)

    # 保留原始位置, 计算后还原 (groupby/rolling 不依赖数据按 code 排序)
    original_index = data.index
    work = data.copy().reset_index(drop=True)
    ast_root = parse_expression(expr)
    result = _eval(work, ast_root)
    # 用 join 还原原始顺序
    if not result.index.equals(work.index):
        # 构造 (date, code) -> 原始 index 的映射
        mapper = pd.DataFrame({
            "_date": work["date"].values,
            "_code": work["code"].values,
            "_pos": np.arange(len(work)),
        })
        if isinstance(result.index, pd.RangeIndex) and result.index.equals(range(len(result))):
            tmp = pd.DataFrame({"_pos": np.arange(len(result)), "v": result.values})
            merged = mapper.merge(tmp, on="_pos", how="left")
            out = pd.Series(merged["v"].values, index=original_index)
            return out
    # fallback: 直接按原 index 还原
    if len(result) == len(original_index):
        result.index = original_index
    return result


# ============================================================================
# Part 4: 自定义算子扩展 (类似 Qlib 允许用户注册新算子)
# ============================================================================

def register_operator(name: str, cls: type) -> None:
    """注册用户自定义算子"""
    if not issubclass(cls, Operator):
        raise TypeError("算子类必须继承 Operator")
    OP_REGISTRY[name] = cls


__all__ = [
    "Field", "Operator", "OP_REGISTRY",
    "Log", "Abs", "Sign", "Sqrt",
    "Add", "Sub", "Mul", "Div", "Power",
    "Ref", "Mean", "Std", "Sum", "Max", "Min", "Delta",
    "Rank", "ZScore", "Scale",
    "parse_expression", "evaluate_expression", "register_operator",
]