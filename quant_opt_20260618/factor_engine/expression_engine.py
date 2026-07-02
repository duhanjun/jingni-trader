"""
因子表达式引擎 (Factor Expression Engine)

借鉴自：
  - Qlib 表达式引擎 (Ref / Mean / Std / Rank 等算子链)
  - AKQuant FactorEngine (Polars + 算子语义)
  - WorldQuant Alpha101 (公式风格)

设计目标：
  1. 让用户以字符串公式形式灵活定义 Alpha 因子，例如
     "Rank(Ts_Mean(Close, 5))" / "Delta(Close, 5) / Ts_Std(Return, 20)"
  2. 自动处理时间序列（按 code 分组）与截面（按 date）两种语义
  3. 在不改 jingni-trader 现有 factor-engine 内部实现的前提下，
     提供一个独立的轻量级验证模块，便于回归与对比

约定：
  - 时间序列算子 (Ts_*)  按 (code, date) 进行 rolling/expanding
  - 截面算子     (Rank / Quantile / Demean)  按 (date) 截面
  - 基础列      ($close, $open, $high, $low, $volume, $amount, ...)
  - 数值常量     直接写 0.05, 3 等
  - 数学/逻辑   + - * /  (  )  Abs  Log  Sign  If
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# 算子注册表
# ─────────────────────────────────────────────────────────────

@dataclass
class OperatorSpec:
    """算子规格说明"""
    name: str
    category: str        # "ts" / "cs" / "math" / "logic"
    arity: int           # 参数个数
    description: str


OPERATORS: Dict[str, OperatorSpec] = {
    # 时间序列
    "Ts_Mean":     OperatorSpec("Ts_Mean",     "ts", 2, "时间窗口均值"),
    "Ts_Sum":      OperatorSpec("Ts_Sum",      "ts", 2, "时间窗口求和"),
    "Ts_Std":      OperatorSpec("Ts_Std",      "ts", 2, "时间窗口标准差"),
    "Ts_Min":      OperatorSpec("Ts_Min",      "ts", 2, "时间窗口最小值"),
    "Ts_Max":      OperatorSpec("Ts_Max",      "ts", 2, "时间窗口最大值"),
    "Ts_Rank":     OperatorSpec("Ts_Rank",     "ts", 2, "时间窗口内排序分位"),
    "Ts_ArgMax":   OperatorSpec("Ts_ArgMax",   "ts", 2, "时间窗口内最大值的滞后位置"),
    "Ts_ArgMin":   OperatorSpec("Ts_ArgMin",   "ts", 2, "时间窗口内最小值的滞后位置"),
    "Delay":       OperatorSpec("Delay",       "ts", 2, "滞后 d 期"),
    "Delta":       OperatorSpec("Delta",       "ts", 2, "差分 x[t] - x[t-d]"),
    # 截面
    "Rank":        OperatorSpec("Rank",        "cs", 1, "截面排序分位 (0-1)"),
    "Quantile":    OperatorSpec("Quantile",    "cs", 1, "截面分箱 (5 等分)"),
    "Demean":      OperatorSpec("Demean",      "cs", 1, "截面去均值"),
    "Scale":       OperatorSpec("Scale",       "cs", 1, "截面缩放到 abs 和=1"),
    # 数学/逻辑
    "Abs":         OperatorSpec("Abs",         "math", 1, "绝对值"),
    "Log":         OperatorSpec("Log",         "math", 1, "自然对数"),
    "Sign":        OperatorSpec("Sign",        "math", 1, "符号函数"),
    "Sqrt":        OperatorSpec("Sqrt",        "math", 1, "平方根"),
    "If":          OperatorSpec("If",          "logic", 3, "If(cond, a, b)"),
}


# ─────────────────────────────────────────────────────────────
# 列引用
# ─────────────────────────────────────────────────────────────

COLUMN_ALIASES: Dict[str, str] = {
    "$close":  "close",
    "$open":   "open",
    "$high":   "high",
    "$low":    "low",
    "$volume": "volume",
    "$amount": "amount",
    "$vwap":   "vwap",
    "$turnover_rate": "turnover_rate",
    "$change_pct":    "change_pct",
    "$ret_1d":        "ret_1d",
    # 兼容无 $ 前缀写法
    "close":   "close",
    "open":    "open",
    "high":    "high",
    "low":     "low",
    "volume":  "volume",
    "amount":  "amount",
    "vwap":    "vwap",
    "turnover_rate": "turnover_rate",
    "change_pct":    "change_pct",
    "ret_1d":        "ret_1d",
}


def _resolve_column(token: str) -> str:
    """将 $close 等 token 解析为 DataFrame 的列名"""
    key = token.strip()
    if key in COLUMN_ALIASES:
        return COLUMN_ALIASES[key]
    if key in COLUMN_ALIASES.values():
        return key
    raise KeyError(f"未知列引用: {token}")


# ─────────────────────────────────────────────────────────────
# 各算子的 DataFrame 实现
# ─────────────────────────────────────────────────────────────

def _ts_mean(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("code")[col].transform(
        lambda s: s.rolling(window, min_periods=max(1, window // 2)).mean()
    )


def _ts_sum(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("code")[col].transform(
        lambda s: s.rolling(window, min_periods=max(1, window // 2)).sum()
    )


def _ts_std(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("code")[col].transform(
        lambda s: s.rolling(window, min_periods=max(2, window // 2)).std()
    )


def _ts_min(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("code")[col].transform(
        lambda s: s.rolling(window, min_periods=1).min()
    )


def _ts_max(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("code")[col].transform(
        lambda s: s.rolling(window, min_periods=1).max()
    )


def _ts_rank(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("code")[col].transform(
        lambda s: s.rolling(window, min_periods=window).rank(pct=True)
    )


def _ts_argmax(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("code")[col].transform(
        lambda s: s.rolling(window, min_periods=window).apply(
            lambda x: window - 1 - np.argmax(x[::-1]) if len(x) == window else np.nan,
            raw=True,
        )
    )


def _ts_argmin(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("code")[col].transform(
        lambda s: s.rolling(window, min_periods=window).apply(
            lambda x: window - 1 - np.argmin(x[::-1]) if len(x) == window else np.nan,
            raw=True,
        )
    )


def _delay(df: pd.DataFrame, col: str, d: int) -> pd.Series:
    return df.groupby("code")[col].shift(d)


def _delta(df: pd.DataFrame, col: str, d: int) -> pd.Series:
    return df[col] - _delay(df, col, d)


def _rank_cs(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True)


def _quantile_cs(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].transform(
        lambda s: pd.qcut(s, q=5, labels=False, duplicates="drop") + 1
    ).astype(float)


def _demean_cs(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("date")[col].transform("mean")


def _scale_cs(df: pd.DataFrame, col: str) -> pd.Series:
    abs_sum = df.groupby("date")[col].transform(lambda s: s.abs().sum())
    return df[col] / abs_sum.replace(0, np.nan)


def _math_abs(s: pd.Series) -> pd.Series:
    return s.abs()


def _math_log(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan))


def _math_sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


def _math_sqrt(s: pd.Series) -> pd.Series:
    return np.sqrt(s.replace(-1, np.nan).clip(lower=0))


def _logic_if(cond: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    return np.where(cond.fillna(False), a, b)


# ─────────────────────────────────────────────────────────────
# 公式解析（递归下降，支持嵌套函数）
# ─────────────────────────────────────────────────────────────

# 词法：identifier ( [ number , )  $ .  + - * /
# id 支持以 $ 开头的列别名，例如 $close / $volume
_TOKEN_RE = re.compile(
    r"""
    \s+                                       |   # 空白
    (?P<num>\d+\.?\d*)                            |   # 数字
    (?P<id>\$?[A-Za-z_][A-Za-z0-9_]*)             |   # 标识符（可带 $ 前缀）
    (?P<sym>[(),+\-*/])                              # 单字符符号
    """,
    re.VERBOSE,
)


class FormulaError(ValueError):
    pass


class _Parser:
    def __init__(self, s: str):
        self.tokens: List[Tuple[str, str]] = []
        for m in _TOKEN_RE.finditer(s):
            if m.group("num"):
                self.tokens.append(("num", m.group("num")))
            elif m.group("id"):
                self.tokens.append(("id", m.group("id")))
            elif m.group("sym"):
                self.tokens.append(("sym", m.group("sym")))
        self.pos = 0

    def peek(self) -> Optional[Tuple[str, str]]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> Tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str, val: str) -> None:
        tok = self.peek()
        if tok is None or tok != (kind, val):
            raise FormulaError(f"expected {kind}:{val}, got {tok}")
        self.pos += 1

    def parse(self):
        node = self.parse_expr()
        if self.pos != len(self.tokens):
            raise FormulaError(f"trailing tokens at pos {self.pos}: {self.tokens[self.pos:]}")
        return node

    # 表达式：加减
    def parse_expr(self):
        left = self.parse_term()
        while True:
            tok = self.peek()
            if tok and tok == ("sym", "+"):
                self.consume()
                right = self.parse_term()
                left = ("binop", "+", left, right)
            elif tok and tok == ("sym", "-"):
                self.consume()
                right = self.parse_term()
                left = ("binop", "-", left, right)
            else:
                return left

    # 项：乘除
    def parse_term(self):
        left = self.parse_factor()
        while True:
            tok = self.peek()
            if tok and tok == ("sym", "*"):
                self.consume()
                right = self.parse_factor()
                left = ("binop", "*", left, right)
            elif tok and tok == ("sym", "/"):
                self.consume()
                right = self.parse_factor()
                left = ("binop", "/", left, right)
            else:
                return left

    # 因子：数字 / 列 / 函数 / 括号 / 一元负号
    def parse_factor(self):
        tok = self.peek()
        if tok is None:
            raise FormulaError("unexpected end")
        if tok == ("sym", "-"):
            self.consume()
            child = self.parse_factor()
            return ("neg", child)
        if tok == ("sym", "+"):
            self.consume()
            return self.parse_factor()
        if tok[0] == "num":
            self.consume()
            return ("num", float(tok[1]))
        if tok[0] == "id":
            name = tok[1]
            self.consume()
            nxt = self.peek()
            if nxt == ("sym", "("):
                # 函数调用
                self.expect("sym", "(")
                args = []
                if self.peek() != ("sym", ")"):
                    args.append(self.parse_expr())
                    while self.peek() == ("sym", ","):
                        self.consume()
                        args.append(self.parse_expr())
                self.expect("sym", ")")
                return ("call", name, args)
            # 标识符（列名或已注册函数 0 参）
            if name in OPERATORS and OPERATORS[name].arity == 0:
                return ("call", name, [])
            # 保留原始 token（带或不带 $ 都允许）
            if name in COLUMN_ALIASES or name in COLUMN_ALIASES.values():
                return ("col", name)
            raise FormulaError(f"未知列引用: {name}")
        if tok == ("sym", "("):
            self.consume()
            node = self.parse_expr()
            self.expect("sym", ")")
            return node
        raise FormulaError(f"unexpected token {tok}")


def parse_formula(s: str):
    return _Parser(s).parse()


# ─────────────────────────────────────────────────────────────
# AST -> DataFrame 求值
# ─────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values(["code", "date"]).reset_index(drop=True)

    def eval(self, node) -> pd.Series:
        kind = node[0]
        if kind == "num":
            return pd.Series(node[1], index=self.df.index, dtype=float)
        if kind == "col":
            col = _resolve_column(node[1])
            if col not in self.df.columns:
                raise KeyError(f"DataFrame 缺少列: {col}")
            return self.df[col].astype(float)
        if kind == "neg":
            return -self.eval(node[1])
        if kind == "binop":
            op, a, b = node[1], node[2], node[3]
            va, vb = self.eval(a), self.eval(b)
            if op == "+": return va + vb
            if op == "-": return va - vb
            if op == "*": return va * vb
            if op == "/": return va / vb.replace(0, np.nan)
            raise FormulaError(f"unknown op {op}")
        if kind == "call":
            name, args = node[1], node[2]
            if name not in OPERATORS:
                raise FormulaError(f"未知算子: {name}")
            spec = OPERATORS[name]
            return self._apply_op(spec, args)
        raise FormulaError(f"unknown node {node}")

    def _apply_op(self, spec: OperatorSpec, args) -> pd.Series:
        # 数学/逻辑算子：纯 Series 入参
        if spec.category == "math":
            v = self.eval(args[0])
            if spec.name == "Abs":  return _math_abs(v)
            if spec.name == "Log":  return _math_log(v)
            if spec.name == "Sign": return _math_sign(v)
            if spec.name == "Sqrt": return _math_sqrt(v)
        if spec.category == "logic":
            if spec.name == "If":
                cond = self.eval(args[0])
                a = self.eval(args[1])
                b = self.eval(args[2])
                return _logic_if(cond, a, b)
        # 时间序列算子：第一个参数为列名 token 或任意表达式
        if spec.category == "ts":
            window = int(float(self._eval_const(args[1])))
            # 第一个参数：可能是 col 节点或任意表达式
            col_node = args[0]
            if col_node[0] == "col":
                col = _resolve_column(col_node[1])
                series = self.df[col].astype(float)
            else:
                series = self.eval(col_node)
            # 把 series 临时附到 df 上（对 _ts_rank 等使用 groupby('code') 的算子）
            tmp = self.df.copy()
            tmp["__x__"] = series.values
            if spec.name == "Ts_Mean":   return _ts_mean(tmp, "__x__", window)
            if spec.name == "Ts_Sum":    return _ts_sum(tmp, "__x__", window)
            if spec.name == "Ts_Std":    return _ts_std(tmp, "__x__", window)
            if spec.name == "Ts_Min":    return _ts_min(tmp, "__x__", window)
            if spec.name == "Ts_Max":    return _ts_max(tmp, "__x__", window)
            if spec.name == "Ts_Rank":   return _ts_rank(tmp, "__x__", window)
            if spec.name == "Ts_ArgMax": return _ts_argmax(tmp, "__x__", window)
            if spec.name == "Ts_ArgMin": return _ts_argmin(tmp, "__x__", window)
            if spec.name == "Delay":     return _delay(tmp, "__x__", window)
            if spec.name == "Delta":     return _delta(tmp, "__x__", window)
        # 截面算子
        if spec.category == "cs":
            v = self.eval(args[0])
            if spec.name == "Rank":     return _rank_cs(self._attach(v), "__x__")
            if spec.name == "Quantile": return _quantile_cs(self._attach(v), "__x__")
            if spec.name == "Demean":   return _demean_cs(self._attach(v), "__x__")
            if spec.name == "Scale":    return _scale_cs(self._attach(v), "__x__")
        raise FormulaError(f"算子未实现: {spec.name}")

    def _eval_const(self, node) -> float:
        if node[0] == "num":
            return node[1]
        if node[0] == "neg":
            return -self._eval_const(node[1])
        raise FormulaError(f"expected constant, got {node}")

    def _eval_col_token(self, node) -> str:
        if node[0] == "col":
            return _resolve_column(node[1])
        if node[0] == "call" and node[1] in {"Delta", "Delay"}:
            # 允许 Delay(Delta($close, 1), 1) 链式
            return self.eval(node).name  # type: ignore
        raise FormulaError(f"expected column token, got {node}")

    def _attach(self, s: pd.Series) -> pd.DataFrame:
        """把 Series 临时附到 df 上，便于使用 groupby('date')"""
        tmp = self.df.copy()
        tmp["__x__"] = s.values
        return tmp


# ─────────────────────────────────────────────────────────────
# 对外 API
# ─────────────────────────────────────────────────────────────

def calc_factor(df: pd.DataFrame, formula: str, name: Optional[str] = None) -> pd.Series:
    """
    计算单个因子。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 code / date / close 等列
    formula : str
        因子公式，例如 "Rank(Ts_Mean($close, 5))"
    name : str, optional
        返回 Series 的名字

    Returns
    -------
    pd.Series
    """
    if "code" not in df.columns or "date" not in df.columns:
        raise ValueError("DataFrame 必须包含 'code' 和 'date' 列")
    ast_tree = parse_formula(formula)
    s = Evaluator(df).eval(ast_tree)
    if name:
        s.name = name
    return s


def calc_factors(df: pd.DataFrame, formulas: Dict[str, str]) -> pd.DataFrame:
    """
    批量计算多个因子。返回与 df 同长度的 DataFrame，新增列为因子名。
    """
    if "code" not in df.columns or "date" not in df.columns:
        raise ValueError("DataFrame 必须包含 'code' 和 'date' 列")
    out = df.copy()
    for name, formula in formulas.items():
        out[name] = calc_factor(df, formula).values
    return out


def list_operators() -> List[OperatorSpec]:
    """列出已注册的所有算子规格"""
    return list(OPERATORS.values())


__all__ = [
    "OPERATORS",
    "COLUMN_ALIASES",
    "OperatorSpec",
    "FormulaError",
    "parse_formula",
    "calc_factor",
    "calc_factors",
    "list_operators",
]
