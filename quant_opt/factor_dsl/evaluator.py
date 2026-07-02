"""
Factor DSL (Expression Engine) -- Alphalens/AKQuant inspired
============================================================

借鉴 AKQuant ``akquant.factor`` 表达式引擎思想, 提供一个**轻量级**的因子表达式解析器:

  expr = "Rank(Delta(Close, 5))"
  expr = "Sign(Add(Sub(Ts_Mean(Volume, 20), Ts_Mean(Volume, 5)), 0))"

设计要点:
- **AST + 解释器**: 词法/语法分析 -> 递归求值
- **时序算子 (Ts_)**: 在 (code, date) MultiIndex 上按 code 分组滚动
- **截面算子 (Rank, Scale, Demean)**: 在每个 date 横截面排序/标准化
- **零依赖**: 仅依赖 numpy/pandas
- **沙箱安全**: 不使用 ``eval``, 通过白名单算子

References
----------
- AKQuant Factor Engine Guide:
  https://akquant.akfamily.xyz/en/guide/factor/
- WorldQuant Alpha101 formulas
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. AST 节点
# --------------------------------------------------------------------------------------

@dataclass
class Number:
    value: float


@dataclass
class Var:
    name: str


@dataclass
class Call:
    name: str
    args: List[Any]


# --------------------------------------------------------------------------------------
# 2. 词法 / 语法
# --------------------------------------------------------------------------------------

_TOKEN_REGEX = re.compile(r"\s*(?:([A-Za-z_][A-Za-z0-9_]*)|([(),])|(-?\d+(?:\.\d+)?))")


def tokenize(expr: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    i = 0
    while i < len(expr):
        m = _TOKEN_REGEX.match(expr, i)
        if not m:
            raise ValueError(f"Unexpected char at {i}: {expr[i]!r}")
        name, punct, num = m.groups()
        if name is not None:
            tokens.append(("NAME", name))
        elif punct is not None:
            tokens.append(("PUNCT", punct))
        elif num is not None:
            tokens.append(("NUM", num))
        i = m.end()
    return tokens


class _Parser:
    """递归下降解析器, 输出 AST."""

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

    def parse(self) -> Any:
        node = self.expr()
        if self.pos != len(self.tokens):
            raise ValueError(f"Trailing tokens at {self.pos}")
        return node

    def expr(self) -> Any:
        return self.primary()

    def primary(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected EOF")
        kind, val = tok
        if kind == "NUM":
            self.consume()
            return Number(float(val))
        if kind == "NAME":
            self.consume()
            nxt = self.peek()
            if nxt is not None and nxt == ("PUNCT", "("):
                self.consume()  # '('
                args: List[Any] = []
                if self.peek() != ("PUNCT", ")"):
                    while True:
                        args.append(self.expr())
                        nxt = self.peek()
                        if nxt == ("PUNCT", ","):
                            self.consume()
                            continue
                        break
                if self.peek() != ("PUNCT", ")"):
                    raise ValueError("Expected ')'")
                self.consume()
                return Call(val, args)
            return Var(val)
        raise ValueError(f"Unexpected token: {tok}")


def parse(expr: str) -> Any:
    return _Parser(tokenize(expr)).parse()


# --------------------------------------------------------------------------------------
# 3. 算子注册表
# --------------------------------------------------------------------------------------

# 类型:
#   ("ts", func)        -> 时序算子: func(s, **kwargs) -> pd.Series
#   ("cs", func)        -> 截面算子: func(s) -> pd.Series (按 date 分组后 transform)
#   ("elem", func)      -> 逐元素算子: func(s) -> pd.Series
#   ("var", name)       -> 内置变量: 映射 DataFrame 列

OPS: Dict[str, Tuple[str, Callable]] = {
    # --- 元素级 ---
    "Abs":    ("elem", lambda s: s.abs()),
    "Sign":   ("elem", lambda s: np.sign(s)),
    "Log":    ("elem", lambda s: np.log(s.where(s > 0))),
    "Sqrt":   ("elem", lambda s: np.sqrt(s.where(s >= 0))),
    "Add":    ("elem", lambda a, b: a + b),
    "Sub":    ("elem", lambda a, b: a - b),
    "Mul":    ("elem", lambda a, b: a * b),
    "Div":    ("elem", lambda a, b: a / b.replace(0, np.nan)),
    "Const":  ("elem_const", lambda v: v),  # 占位符; 实际在 _eval_call 中处理

    # --- 时序 ---
    "Ts_Mean":   ("ts",  lambda s, d: s.groupby(level="code").transform(lambda x: x.rolling(int(d), min_periods=1).mean())),
    "Ts_Sum":    ("ts",  lambda s, d: s.groupby(level="code").transform(lambda x: x.rolling(int(d), min_periods=1).sum())),
    "Ts_Std":    ("ts",  lambda s, d: s.groupby(level="code").transform(lambda x: x.rolling(int(d), min_periods=2).std())),
    "Ts_Min":    ("ts",  lambda s, d: s.groupby(level="code").transform(lambda x: x.rolling(int(d), min_periods=1).min())),
    "Ts_Max":    ("ts",  lambda s, d: s.groupby(level="code").transform(lambda x: x.rolling(int(d), min_periods=1).max())),
    "Delta":     ("ts",  lambda s, d: s.groupby(level="code").diff(int(d))),
    "Delay":     ("ts",  lambda s, d: s.groupby(level="code").shift(int(d))),

    # --- 截面 ---
    "Rank":      ("cs",  lambda s: s.groupby(level="date").rank(pct=True)),
    "Demean":    ("cs",  lambda s: s - s.groupby(level="date").transform("mean")),
    "Scale":     ("cs",  lambda s: s / s.groupby(level="date").transform(lambda x: x.abs().sum()).replace(0, np.nan)),

    # --- 条件 ---
    "If":        ("elem", lambda cond, a, b: np.where(cond, a, b)),
}


# --------------------------------------------------------------------------------------
# 4. 求值器
# --------------------------------------------------------------------------------------

class FactorEvaluator:
    """
    因子求值器.

    期望 DataFrame 拥有 MultiIndex ``(code, date)``;
    支持的内置列名: ``close, open, high, low, volume, amount`` (不区分大小写).
    """

    BUILTIN_VARS = {"close", "open", "high", "low", "volume", "amount"}

    def __init__(self, df: pd.DataFrame, lower_alias: bool = True) -> None:
        if "code" in df.columns and "date" in df.columns:
            df = df.set_index(["code", "date"])
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame must be indexed by (code, date)")
        # 标准化列名为小写
        df = df.rename(columns={c: c.lower() for c in df.columns})
        self.df = df.sort_index()
        self.cache: Dict[int, pd.Series] = {}

    def eval(self, expr: Any) -> pd.Series:
        key = id(expr)
        if key in self.cache:
            return self.cache[key]
        result = self._eval_node(expr)
        self.cache[key] = result
        return result

    def _eval_node(self, node: Any) -> pd.Series:
        if isinstance(node, Number):
            return pd.Series(node.value, index=self.df.index, dtype=float)
        if isinstance(node, Var):
            name = node.name.lower()
            if name in self.BUILTIN_VARS:
                return self.df[name].astype(float)
            raise KeyError(f"Unknown variable: {node.name}")
        if isinstance(node, Call):
            return self._eval_call(node)
        raise TypeError(f"Unknown AST node: {type(node)}")

    def _eval_call(self, call: Call) -> pd.Series:
        name = call.name
        if name == "Const":
            # Const(value) -> 全字段常数 Series, 与 df 同 index
            if len(call.args) != 1 or not isinstance(call.args[0], Number):
                raise ValueError("Const expects a single number argument")
            return pd.Series(call.args[0].value, index=self.df.index, dtype=float)
        if name not in OPS:
            raise KeyError(f"Unknown operator: {name}")
        kind, func = OPS[name]

        # 先看是否所有参数都是 Number 常量 (叶子求值)
        all_numeric_const = all(isinstance(a, Number) for a in call.args)
        if all_numeric_const and kind == "elem":
            # 纯常量表达式: 直接数值计算
            const_vals = [a.value for a in call.args]
            return func(*const_vals)

        # 常规情况: 递归求值
        args = [self._eval_node(a) for a in call.args]

        if kind == "elem":
            return func(*args)
        if kind == "ts":
            # 期望 func(s, d) 或 func(s)
            if len(args) == 2 and isinstance(call.args[1], Number):
                return func(args[0], call.args[1].value)
            return func(args[0])
        if kind == "cs":
            return func(args[0])
        raise ValueError(f"Bad op kind: {kind}")


# --------------------------------------------------------------------------------------
# 5. 顶层便捷 API
# --------------------------------------------------------------------------------------

def evaluate_factor(df: pd.DataFrame, expr: str) -> pd.Series:
    """
    一行调用: 解析 + 求值 + 返回与 df 对齐的因子 Series.
    """
    ast = parse(expr)
    return FactorEvaluator(df).eval(ast)


# --------------------------------------------------------------------------------------
# 6. 内置常用因子库 (Alphalens/AKQuant 风格)
# --------------------------------------------------------------------------------------

PRESET_FACTORS: Dict[str, str] = {
    # 5 日反转
    "reversal_5d":    "Mul(Const(-1), Delta(close, 5))",
    # 20 日动量
    "momentum_20d":   "Delta(close, 20)",
    # 换手率均值回复
    "turnover_z":     "Div(Sub(volume, Ts_Mean(volume, 20)), Ts_Std(volume, 20))",
    # 量价相关性 (简化)
    "volume_rank":    "Rank(volume)",
    # 20 日波动率
    "volatility_20d": "Ts_Std(close, 20)",
}


def list_preset_factors() -> List[str]:
    return list(PRESET_FACTORS.keys())


def eval_preset(name: str, df: pd.DataFrame) -> pd.Series:
    if name not in PRESET_FACTORS:
        raise KeyError(f"Unknown preset factor: {name}")
    return evaluate_factor(df, PRESET_FACTORS[name])
