"""
因子表达式引擎 - 优化验证模块

借鉴来源:
  - AKQuant: Polars 驱动的因子表达式引擎, 支持 Rank(Ts_Mean(Close, 5)) 等 Alpha101 风格公式
  - Qlib: 表达式 DSL 设计, 自动处理并行计算与数据对齐
  - QuantsPlaybook: 因子表达式模块化设计

优化目标:
  jingni-trader 现有 factor-engine/engine.py 中因子全部硬编码在 compute_a_share_factors
  方法内, 新增因子需要修改源码, 可扩展性差。本模块实现一个轻量表达式引擎,
  支持用字符串公式声明式定义因子, 并向量化计算。

设计要点:
  1. 算子分为截面算子 (Cross-Section, 按 date 分组) 和时序算子 (Time-Series, 按 code 分组)
  2. 表达式解析为 AST, 后序遍历求值, 支持嵌套
  3. 全程使用 pandas groupby/transform 向量化, 避免 Python 循环
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 算子注册表
# ---------------------------------------------------------------------------

class OperatorRegistry:
    """算子注册表, 区分截面算子与时序算子"""

    def __init__(self) -> None:
        self.cross_ops: Dict[str, Tuple[int, Callable]] = {}
        self.ts_ops: Dict[str, Tuple[int, Callable]] = {}

    def register_cross(self, name: str, arity: int) -> Callable:
        def deco(func: Callable) -> Callable:
            self.cross_ops[name] = (arity, func)
            return func
        return deco

    def register_ts(self, name: str, arity: int) -> Callable:
        def deco(func: Callable) -> Callable:
            self.ts_ops[name] = (arity, func)
            return func
        return deco

    def get(self, name: str) -> Optional[Tuple[int, Callable, str]]:
        if name in self.cross_ops:
            arity, func = self.cross_ops[name]
            return arity, func, "cross"
        if name in self.ts_ops:
            arity, func = self.ts_ops[name]
            return arity, func, "ts"
        return None


registry = OperatorRegistry()


# ---- 截面算子 (按 date 分组) ----

@registry.register_cross("Rank", 1)
def _cross_rank(s: pd.Series, _g: pd.DataFrame) -> pd.Series:
    """截面百分位排名, 借鉴 Alpha101"""
    return s.rank(pct=True)


@registry.register_cross("ZScore", 1)
def _cross_zscore(s: pd.Series, _g: pd.DataFrame) -> pd.Series:
    """截面 Z-Score 标准化"""
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - s.mean()) / std


@registry.register_cross("Scale", 1)
def _cross_scale(s: pd.Series, _g: pd.DataFrame) -> pd.Series:
    """截面缩放使绝对值之和为 1"""
    total = s.abs().sum()
    if total == 0 or np.isnan(total):
        return s
    return s / total


@registry.register_cross("Winsorize", 1)
def _cross_winsorize(s: pd.Series, _g: pd.DataFrame, n_sigma: float = 3.0) -> pd.Series:
    """截面去极值 (MAD 法)"""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    bound = n_sigma * 1.4826 * mad
    return s.clip(lower=med - bound, upper=med + bound)


# ---- 时序算子 (按 code 分组) ----

@registry.register_ts("Ts_Mean", 2)
def _ts_mean(s: pd.Series, g: pd.DataFrame, window: int) -> pd.Series:
    """时序滚动均值"""
    return g.groupby("code")[s.name].transform(
        lambda x: x.rolling(window, min_periods=max(1, window // 2)).mean()
    )


@registry.register_ts("Ts_Std", 2)
def _ts_std(s: pd.Series, g: pd.DataFrame, window: int) -> pd.Series:
    """时序滚动标准差"""
    return g.groupby("code")[s.name].transform(
        lambda x: x.rolling(window, min_periods=max(1, window // 2)).std()
    )


@registry.register_ts("Ts_Max", 2)
def _ts_max(s: pd.Series, g: pd.DataFrame, window: int) -> pd.Series:
    return g.groupby("code")[s.name].transform(
        lambda x: x.rolling(window, min_periods=1).max()
    )


@registry.register_ts("Ts_Min", 2)
def _ts_min(s: pd.Series, g: pd.DataFrame, window: int) -> pd.Series:
    return g.groupby("code")[s.name].transform(
        lambda x: x.rolling(window, min_periods=1).min()
    )


@registry.register_ts("Ts_Rank", 2)
def _ts_rank(s: pd.Series, g: pd.DataFrame, window: int) -> pd.Series:
    """时序滚动百分位排名"""
    return g.groupby("code")[s.name].transform(
        lambda x: x.rolling(window, min_periods=max(1, window // 2)).rank(pct=True)
    )


@registry.register_ts("Delta", 2)
def _ts_delta(s: pd.Series, g: pd.DataFrame, window: int) -> pd.Series:
    """时序差分: x_t - x_{t-window}"""
    return g.groupby("code")[s.name].diff(window)


@registry.register_ts("Delay", 2)
def _ts_delay(s: pd.Series, g: pd.DataFrame, window: int) -> pd.Series:
    """时序滞后"""
    return g.groupby("code")[s.name].shift(window)


@registry.register_ts("Ts_Sum", 2)
def _ts_sum(s: pd.Series, g: pd.DataFrame, window: int) -> pd.Series:
    return g.groupby("code")[s.name].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )


@registry.register_ts("Ts_Correlation", 3)
def _ts_corr(s1: pd.Series, g: pd.DataFrame, s2_name: str, window: int) -> pd.Series:
    """时序滚动相关系数"""
    return g.groupby("code").apply(
        lambda x: x[s1.name].rolling(window, min_periods=max(2, window // 2))
        .corr(x[s2_name])
    ).reset_index(level=0, drop=True)


# ---------------------------------------------------------------------------
# 表达式解析器 (递归下降)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\s*(?:(\()|(\))|(,)|"
    r"([A-Za-z_][A-Za-z0-9_]*)|"
    r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)|"
    r"([+\-*/]))"
)


class _Parser:
    """递归下降解析器, 支持函数调用、字段引用、字面量与四则运算

    文法 (优先级从低到高):
        expr    := term (('+' | '-') term)*
        term    := factor (('*' | '/') factor)*
        factor  := '-' factor | atom
        atom    := num | field | call | '(' expr ')'
        call    := name '(' expr (',' expr)* ')'
    """

    def __init__(self, expr: str) -> None:
        self.expr = expr
        self.tokens: List[Tuple[str, str]] = []
        self.pos = 0
        self._tokenize()

    def _tokenize(self) -> None:
        idx = 0
        while idx < len(self.expr):
            m = _TOKEN_RE.match(self.expr, idx)
            if not m:
                if self.expr[idx].isspace():
                    idx += 1
                    continue
                raise ValueError(f"无法解析的字符: {self.expr[idx]!r} 于位置 {idx}")
            idx = m.end()
            groups = m.groups()
            typs = ("lp", "rp", "comma", "name", "num", "op")
            for typ, val in zip(typs, groups):
                if val is not None:
                    self.tokens.append((typ, val))
                    break

    def _peek(self) -> Optional[Tuple[str, str]]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> Tuple[str, str]:
        if self.pos >= len(self.tokens):
            raise ValueError("意外的表达式结尾 (token 不足)")
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> Tuple:
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError(f"表达式末尾存在未消费的 token: {self.tokens[self.pos:]}")
        return node

    def _parse_expr(self) -> Tuple:
        node = self._parse_term()
        while True:
            tok = self._peek()
            if tok and tok[0] == "op" and tok[1] in ("+", "-"):
                self._next()
                right = self._parse_term()
                node = ("binop", tok[1], node, right)
            else:
                break
        return node

    def _parse_term(self) -> Tuple:
        node = self._parse_factor()
        while True:
            tok = self._peek()
            if tok and tok[0] == "op" and tok[1] in ("*", "/"):
                self._next()
                right = self._parse_factor()
                node = ("binop", tok[1], node, right)
            else:
                break
        return node

    def _parse_factor(self) -> Tuple:
        tok = self._peek()
        if tok and tok[0] == "op" and tok[1] == "-":
            self._next()
            operand = self._parse_factor()
            return ("neg", operand)
        return self._parse_atom()

    def _parse_atom(self) -> Tuple:
        typ, val = self._next()
        if typ == "num":
            return ("num", float(val))
        if typ == "name":
            if self._peek() and self._peek()[0] == "lp":
                self._next()  # consume (
                args: List[Tuple] = []
                if self._peek() and self._peek()[0] != "rp":
                    args.append(self._parse_expr())
                    while self._peek() and self._peek()[0] == "comma":
                        self._next()
                        args.append(self._parse_expr())
                if not self._peek() or self._peek()[0] != "rp":
                    raise ValueError("缺少右括号 ')'")
                self._next()  # consume )
                return ("call", val, args)
            return ("field", val)
        if typ == "lp":
            node = self._parse_expr()
            if not self._peek() or self._peek()[0] != "rp":
                raise ValueError("缺少右括号 ')'")
            self._next()
            return node
        raise ValueError(f"意外的 token: {typ} {val!r}")


# ---------------------------------------------------------------------------
# 表达式引擎
# ---------------------------------------------------------------------------

class FactorExpressionEngine:
    """
    因子表达式引擎

    示例:
        engine = FactorExpressionEngine()
        engine.add_field("close", df["close"])
        engine.add_field("open", df["open"])
        engine.add_field("volume", df["volume"])

        # Alpha101 风格公式
        result = engine.evaluate("Rank(Ts_Mean(Close, 5))", df)
        result = engine.evaluate("Delta(Close, 5) / Ts_Std(Close, 20)", df)
    """

    # 字段名归一化映射 (大小写不敏感)
    _FIELD_ALIASES = {
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "volume", "amount": "amount", "vwap": "vwap",
        "returns": "returns", "turnover": "turnover_rate",
    }

    def __init__(self) -> None:
        self.fields: Dict[str, pd.Series] = {}

    def add_field(self, name: str, series: pd.Series) -> None:
        """注册一个字段 (列), 名字归一化为小写"""
        self.fields[name.lower()] = series

    def add_dataframe(self, df: pd.DataFrame) -> None:
        """批量注册 DataFrame 的所有列"""
        for col in df.columns:
            if df[col].dtype.kind in "iufcb":  # 数值类型
                self.fields[col.lower()] = df[col]

    def evaluate(self, expr: str, df: pd.DataFrame) -> pd.Series:
        """
        求值表达式

        参数:
            expr: 表达式字符串, 如 "Rank(Ts_Mean(Close, 5))"
            df: 包含 code, date 和原始字段的 DataFrame (用于 groupby 上下文)

        返回:
            与 df 索引对齐的 Series
        """
        if not self.fields:
            self.add_dataframe(df)
        parser = _Parser(expr)
        ast = parser.parse()
        result = self._eval_node(ast, df)
        # 对齐索引
        if isinstance(result, pd.Series):
            return result.reset_index(drop=True) if result.index.name is None else result
        return result

    def _eval_node(self, node: Tuple, df: pd.DataFrame) -> pd.Series:
        kind = node[0]

        if kind == "num":
            return pd.Series(node[1], index=df.index)

        if kind == "field":
            name = node[1].lower()
            if name in self.fields:
                return self.fields[name]
            if name in df.columns:
                return df[name]
            raise KeyError(f"未知字段: {node[1]}")

        if kind == "neg":
            operand = self._eval_node(node[1], df)
            return -operand

        if kind == "binop":
            op = node[1]
            left = self._eval_node(node[2], df)
            right = self._eval_node(node[3], df)
            # 标量对齐
            if not isinstance(left, pd.Series):
                left = pd.Series(left, index=df.index)
            if not isinstance(right, pd.Series):
                right = pd.Series(right, index=df.index)
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right.replace(0, np.nan)
            raise ValueError(f"未知二元运算符: {op}")

        if kind == "call":
            op_name, args = node[1], node[2]
            info = registry.get(op_name)
            if info is None:
                raise KeyError(f"未知算子: {op_name}")
            arity, func, op_type = info

            # 解析参数: 第一个参数是 Series, 后续参数可能是字面量或字段
            evaluated_args: List[Any] = []
            for i, arg in enumerate(args):
                if i == 0:
                    evaluated_args.append(self._eval_node(arg, df))
                else:
                    # 后续参数: 若是 num 则直接用, 若是 field 则取该列名
                    if arg[0] == "num":
                        val = arg[1]
                        evaluated_args.append(int(val) if val == int(val) else val)
                    elif arg[0] == "field":
                        evaluated_args.append(arg[1].lower())
                    else:
                        evaluated_args.append(self._eval_node(arg, df))

            if op_type == "cross":
                # 截面算子: 按 date 分组
                series = evaluated_args[0]
                tmp_df = df.copy()
                tmp_col = f"_op_{op_name}_{id(series)}"
                tmp_df[tmp_col] = series.values
                result = tmp_df.groupby("date")[tmp_col].transform(func, _g=tmp_df)
                return result

            if op_type == "ts":
                # 时序算子: 按 code 分组
                series = evaluated_args[0]
                tmp_df = df.copy()
                tmp_col = f"_op_{op_name}_{id(series)}"
                tmp_df[tmp_col] = series.values
                extra = evaluated_args[1:]
                return func(tmp_df[tmp_col], tmp_df, *extra)

        raise ValueError(f"无法求值的节点: {node}")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def evaluate_factor(expr: str, df: pd.DataFrame) -> pd.Series:
    """便捷函数: 一次性求值因子表达式"""
    engine = FactorExpressionEngine()
    engine.add_dataframe(df)
    return engine.evaluate(expr, df)


# 预置 Alpha101 风格因子公式 (借鉴 Qlib Alpha158 思路)
PRESET_FACTORS: Dict[str, str] = {
    "alpha_km_5d": "Rank(Ts_Mean(Close, 5))",
    "alpha_reversal_5d": "Rank(-Delta(Close, 5))",
    "alpha_reversal_20d": "Rank(-Delta(Close, 20))",
    "alpha_vol_20d": "Rank(Ts_Std(Close, 20))",
    "alpha_volume_ratio": "Rank(Volume / Ts_Mean(Volume, 20))",
    "alpha_price_momentum": "Rank(Delta(Close, 20) / Delay(Close, 20))",
    "alpha_range_pos": "Rank((Close - Ts_Min(Low, 20)) / (Ts_Max(High, 20) - Ts_Min(Low, 20) + 1e-10))",
}