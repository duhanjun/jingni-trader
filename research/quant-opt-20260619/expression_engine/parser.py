"""
表达式解析器：Tokenizer + 递归下降 Parser

借鉴 Qlib qlib/data/ops.py 中 "operators implemented as callable classes" 的设计：
- 解析阶段把字符串转 AST（由 operators.py 中的类实例组成的树）
- 任何 operator 类都能被反复实例化使用
"""
from __future__ import annotations
import re
from typing import List, Optional, Tuple

from .operators import (
    ElemOperator, PairOperator, Rolling, Feature, Constant,
    Abs, Log, Sign, Sqrt, Power, Rank,
    Add, Sub, Mul, Div, Greater, Less, Equal, And, Or, Not, If, SumIf,
    Ref, Delta, Mean, Std, Sum, Max, Min, Med, Mad, Quantile,
    Slope, Rsquare, Resi, Corr, Cov,
)


# -----------------------------------------------------------------------------
# Token 类型
# -----------------------------------------------------------------------------
# re.match 已经默认锚定在 pos 位置（不需 ^）
TOK_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TOK_NUM = re.compile(r"-?\d+(\.\d+)?(e[-+]?\d+)?")
TOK_DOLLAR = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")


def tokenize(expr: str) -> List[Tuple[str, str]]:
    """
    将表达式字符串转为 token 列表
    返回 [(type, value), ...]
    """
    tokens = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "+-*/^(),&|<>=!":
            tokens.append(("OP", ch))
            i += 1
            continue
        # 美元符号开头：特征引用
        m = TOK_DOLLAR.match(expr, i)
        if m:
            tokens.append(("FEATURE", m.group()))
            i = m.end()
            continue
        # 数字
        m = TOK_NUM.match(expr, i)
        if m:
            tokens.append(("NUM", m.group()))
            i = m.end()
            continue
        # 标识符
        m = TOK_NAME.match(expr, i)
        if m:
            tokens.append(("NAME", m.group()))
            i = m.end()
            continue
        raise SyntaxError(f"无法识别的字符: {ch!r} at pos {i} in {expr!r}")
    return tokens


# -----------------------------------------------------------------------------
# 操作符元数据
# -----------------------------------------------------------------------------
# 二元操作符: (优先级, 名字, 类)
_BINARY_OPS = {
    "+": (1, "Add", Add),
    "-": (1, "Sub", Sub),
    "*": (2, "Mul", Mul),
    "/": (2, "Div", Div),
    "&": (0, "And", And),
    "|": (0, "Or", Or),
    ">": (0, "Greater", Greater),
    "<": (0, "Less", Less),
    "=": (0, "Equal", Equal),
    "^": (3, "Power", None),  # 特殊：Power(feat, n)
}


class _Parser:
    """递归下降 parser"""

    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Tuple[str, str]]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> Tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, ttype: str, tval: Optional[str] = None):
        tok = self.peek()
        if tok is None:
            raise SyntaxError(f"期望 {ttype}({tval})，但已到末尾")
        if tok[0] != ttype or (tval is not None and tok[1] != tval):
            raise SyntaxError(f"期望 {ttype}({tval})，但获得 {tok}")
        return self.consume()

    # ---- grammar ----
    def parse_expression(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.peek() == ("OP", "|"):
            self.consume()
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self):
        left = self.parse_comparison()
        while self.peek() == ("OP", "&"):
            self.consume()
            right = self.parse_comparison()
            left = And(left, right)
        return left

    def parse_comparison(self):
        left = self.parse_additive()
        while self.peek() and self.peek()[0] == "OP" and self.peek()[1] in (">", "<", "="):
            op = self.consume()[1]
            right = self.parse_additive()
            if op == ">":
                left = Greater(left, right)
            elif op == "<":
                left = Less(left, right)
            elif op == "=":
                left = Equal(left, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.peek() and self.peek()[0] == "OP" and self.peek()[1] in ("+", "-"):
            op = self.consume()[1]
            right = self.parse_multiplicative()
            if op == "+":
                left = Add(left, right)
            elif op == "-":
                left = Sub(left, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_power()
        while self.peek() and self.peek()[0] == "OP" and self.peek()[1] in ("*", "/"):
            op = self.consume()[1]
            right = self.parse_power()
            if op == "*":
                left = Mul(left, right)
            elif op == "/":
                left = Div(left, right)
        return left

    def parse_power(self):
        left = self.parse_unary()
        while self.peek() and self.peek()[0] == "OP" and self.peek()[1] == "^":
            self.consume()
            right = self.parse_unary()
            left = Power(left, right)
        return left

    def parse_unary(self):
        tok = self.peek()
        if tok and tok[0] == "OP" and tok[1] == "-":
            self.consume()
            operand = self.parse_unary()
            return Mul(Constant(-1.0), operand)
        if tok and tok[0] == "OP" and tok[1] == "!":
            self.consume()
            operand = self.parse_unary()
            return Not(operand)
        return self.parse_primary()

    def parse_primary(self):
        tok = self.peek()
        if tok is None:
            raise SyntaxError("意外结束：缺少主表达式")

        if tok[0] == "OP" and tok[1] == "(":
            self.consume()
            node = self.parse_expression()
            self.expect("OP", ")")
            return node

        if tok[0] == "NUM":
            self.consume()
            return Constant(float(tok[1]))

        if tok[0] == "FEATURE":
            self.consume()
            return Feature(tok[1])

        if tok[0] == "NAME":
            name = self.consume()[1]
            # 是否有参数列表
            if self.peek() and self.peek() == ("OP", "("):
                return self.parse_call(name)
            # 部分操作符名兼容 (无参)
            return self._nullary(name)

        raise SyntaxError(f"无法解析 token: {tok}")

    def _nullary(self, name: str):
        # 允许裸名 Only if registered in alias
        if name in ("Open", "High", "Low", "Close", "Volume", "Amount", "Vwap"):
            return Feature("$" + name.lower())
        raise SyntaxError(f"未知标识符: {name}")

    def parse_call(self, name: str):
        self.expect("OP", "(")
        args = []
        if not (self.peek() and self.peek() == ("OP", ")")):
            args.append(self.parse_expression())
            while self.peek() and self.peek() == ("OP", ","):
                self.consume()
                args.append(self.parse_expression())
        self.expect("OP", ")")
        return self._build_call(name, args)

    def _build_call(self, name: str, args):
        # 已知函数表
        single_elem = {
            "Abs": Abs, "Log": Log, "Sign": Sign, "Sqrt": Sqrt, "Rank": Rank,
        }
        single_rolling = {
            "Ref": Ref, "Mean": Mean, "Std": Std, "Sum": Sum,
            "Max": Max, "Min": Min, "Med": Med, "Mad": Mad, "Quantile": Quantile,
            "Delta": Delta, "Slope": Slope, "Rsquare": Rsquare, "Resi": Resi,
        }
        if name in single_elem:
            if len(args) != 1:
                raise SyntaxError(f"{name} 需要 1 个参数，得到 {len(args)}")
            return single_elem[name](args[0])
        if name in single_rolling:
            if len(args) != 2:
                raise SyntaxError(f"{name} 需要 2 个参数 (feature, window)，得到 {len(args)}")
            return single_rolling[name](args[0], int(float(args[1].value if hasattr(args[1], "value") else 0)))
        if name == "Power":
            if len(args) != 2:
                raise SyntaxError(f"Power 需要 2 个参数")
            return Power(args[0], args[1].value if isinstance(args[1], Constant) else float(args[1].value))
        if name == "Corr":
            if len(args) != 3:
                raise SyntaxError(f"Corr 需要 3 个参数 (a, b, window)")
            return Corr(args[0], args[1], int(float(args[2].value if isinstance(args[2], Constant) else 0)))
        if name == "Cov":
            if len(args) != 3:
                raise SyntaxError(f"Cov 需要 3 个参数 (a, b, window)")
            return Cov(args[0], args[1], int(float(args[2].value if isinstance(args[2], Constant) else 0)))
        if name == "If":
            if len(args) != 3:
                raise SyntaxError(f"If 需要 3 个参数 (cond, true, false)")
            return If(args[0], args[1], args[2])
        if name == "SumIf":
            if len(args) != 3:
                raise SyntaxError(f"SumIf 需要 3 个参数 (cond, feature, window)")
            return SumIf(args[0], args[1], int(float(args[2].value if isinstance(args[2], Constant) else 0)))
        if name == "Not":
            if len(args) != 1:
                raise SyntaxError(f"Not 需要 1 个参数")
            return Not(args[0])
        raise SyntaxError(f"未知函数: {name}")


class ExpressionParser:
    """对外的解析器接口"""

    def parse(self, expr: str):
        tokens = tokenize(expr)
        parser = _Parser(tokens)
        node = parser.parse_expression()
        if parser.pos != len(tokens):
            raise SyntaxError(f"解析未完成，剩余 tokens: {tokens[parser.pos:]}")
        return node

    @staticmethod
    def validate(expr: str) -> bool:
        try:
            ExpressionParser().parse(expr)
            return True
        except Exception:
            return False
