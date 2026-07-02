"""
因子表达式引擎 (Factor Expression Engine)

借鉴来源: Qlib Expression Engine (https://qlib.readthedocs.io)
- 用字符串表达式声明式定义因子，如 "Ref($close, 20) / $close - 1"
- 支持算子: Ref, Mean, Std, Sum, Max, Min, Delta, Rank, Corr, Cov,
            Abs, Log, Sign, Greater, Less, If 等
- 基于注册表 (Registry) 扩展，新增因子无需修改引擎代码
- 全程 pandas groupby 向量化，避免逐股循环

与 jingni-trader 现状对比:
- 现状: compute_a_share_factors() 硬编码 ~12 个因子，新增因子需改源码
- 优化后: 因子定义在配置/字典中，引擎按表达式求值，零代码扩展

注意: 这是一个轻量实现，覆盖 Qlib 最常用算子的子集，
足以验证设计思路的可行性与性能。
"""
from __future__ import annotations
import re
from typing import Dict, List, Callable, Any, Optional
import numpy as np
import pandas as pd


# ── 算子注册表 ──────────────────────────────────────────────
class OperatorRegistry:
    """因子表达式算子注册表"""

    _operators: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def deco(fn: Callable) -> Callable:
            cls._operators[name] = fn
            return fn
        return deco

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        return cls._operators.get(name)

    @classmethod
    def list_operators(cls) -> List[str]:
        return sorted(cls._operators.keys())


# ── 算子实现 (均按 code 分组向量化) ─────────────────────────
# 约定: 每个算子接收 (df, *args)，df 是按 code 排序的 DataFrame，
#       返回与 df 等长的 Series

@OperatorRegistry.register("Ref")
def _op_ref(df: pd.DataFrame, col: str, n: float) -> pd.Series:
    """Ref($close, n): n 期前的值"""
    n = int(n)
    return df.groupby("code")[col].shift(n)


@OperatorRegistry.register("Mean")
def _op_mean(df: pd.DataFrame, col: str, n: float) -> pd.Series:
    """Mean($close, n): n 期滚动均值"""
    n = int(n)
    return df.groupby("code")[col].transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).mean())


@OperatorRegistry.register("Std")
def _op_std(df: pd.DataFrame, col: str, n: float) -> pd.Series:
    """Std($close, n): n 期滚动标准差"""
    n = int(n)
    return df.groupby("code")[col].transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).std())


@OperatorRegistry.register("Sum")
def _op_sum(df: pd.DataFrame, col: str, n: float) -> pd.Series:
    """Sum($volume, n): n 期滚动求和"""
    n = int(n)
    return df.groupby("code")[col].transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).sum())


@OperatorRegistry.register("Max")
def _op_max(df: pd.DataFrame, col: str, n: float) -> pd.Series:
    n = int(n)
    return df.groupby("code")[col].transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).max())


@OperatorRegistry.register("Min")
def _op_min(df: pd.DataFrame, col: str, n: float) -> pd.Series:
    n = int(n)
    return df.groupby("code")[col].transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).min())


@OperatorRegistry.register("Delta")
def _op_delta(df: pd.DataFrame, col: str, n: float) -> pd.Series:
    """Delta($close, n): 当期 - n 期前"""
    n = int(n)
    return df.groupby("code")[col].transform(lambda x: x - x.shift(n))


@OperatorRegistry.register("Rank")
def _op_rank(df: pd.DataFrame, col: str, n: float = 0) -> pd.Series:
    """Rank($factor): 截面排名 (pct rank by date)。
    n 参数保留兼容性，n>0 时为时序 rank，n==0 时为截面 rank。"""
    n = int(n)
    if n > 0:
        return df.groupby("code")[col].transform(
            lambda x: x.rolling(n, min_periods=max(1, n // 2)).rank(pct=True)
        )
    return df.groupby("date")[col].rank(pct=True)


@OperatorRegistry.register("Corr")
def _op_corr(df: pd.DataFrame, col1: str, col2: str, n: float) -> pd.Series:
    """Corr($close, $volume, n): n 期滚动相关系数"""
    n = int(n)
    def _rolling_corr(sub: pd.DataFrame) -> pd.Series:
        return sub[col1].rolling(n, min_periods=max(2, n // 2)).corr(sub[col2])
    return df.groupby("code", group_keys=False).apply(_rolling_corr)


@OperatorRegistry.register("Cov")
def _op_cov(df: pd.DataFrame, col1: str, col2: str, n: float) -> pd.Series:
    n = int(n)
    def _rolling_cov(sub: pd.DataFrame) -> pd.Series:
        return sub[col1].rolling(n, min_periods=max(2, n // 2)).cov(sub[col2])
    return df.groupby("code", group_keys=False).apply(_rolling_cov)


@OperatorRegistry.register("Abs")
def _op_abs(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].abs()


@OperatorRegistry.register("Log")
def _op_log(df: pd.DataFrame, col: str) -> pd.Series:
    return np.log(df[col].clip(lower=1e-12))


@OperatorRegistry.register("Sign")
def _op_sign(df: pd.DataFrame, col: str) -> pd.Series:
    return np.sign(df[col])


@OperatorRegistry.register("Greater")
def _op_greater(df: pd.DataFrame, col: str, n: float) -> pd.Series:
    """Greater($close, n): 取 col 和 n 的较大值"""
    return df[col].clip(lower=n)


@OperatorRegistry.register("Less")
def _op_less(df: pd.DataFrame, col: str, n: float) -> pd.Series:
    return df[col].clip(upper=n)


# ── 表达式解析器 (递归下降) ─────────────────────────────────
_TOKEN_RE = re.compile(r"""
    \s*(
        [A-Za-z_][A-Za-z0-9_]*       # 标识符 (函数名或字段名)
        | \$[A-Za-z_][A-Za-z0-9_]*    # $close 形式的字段引用
        | \d+\.?\d*                   # 数字
        | [(),+\-*/]                  # 标点与运算符
    )
""", re.VERBOSE)


def _tokenize(expr: str) -> List[str]:
    tokens = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m or m.start() != pos:
            if expr[pos].isspace():
                pos += 1
                continue
            raise ValueError(f"无法解析的字符: {expr[pos]!r} at {pos}")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


class _Parser:
    """递归下降解析器，构造求值树"""

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None or (expected and tok != expected):
            raise ValueError(f"期望 {expected!r}, 实际 {tok!r}")
        self.pos += 1
        return tok

    def parse(self):
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError(f"未消费的 token: {self.tokens[self.pos:]}")
        return node

    def _parse_expr(self):
        left = self._parse_term()
        while self.peek() in ("+", "-"):
            op = self.consume()
            right = self._parse_term()
            left = ("binop", op, left, right)
        return left

    def _parse_term(self):
        left = self._parse_factor()
        while self.peek() in ("*", "/"):
            op = self.consume()
            right = self._parse_factor()
            left = ("binop", op, left, right)
        return left

    def _parse_factor(self):
        tok = self.peek()
        if tok == "(":
            self.consume("(")
            node = self._parse_expr()
            self.consume(")")
            return node
        if tok == "-":
            self.consume("-")
            return ("neg", self._parse_factor())
        # 函数调用: Name ( arg, arg, ... )
        if tok and tok[0].isalpha() and self.tokens[self.pos + 1:self.pos + 2] == ["("]:
            name = self.consume()
            self.consume("(")
            args = [self._parse_expr()]
            while self.peek() == ",":
                self.consume(",")
                args.append(self._parse_expr())
            self.consume(")")
            return ("call", name, args)
        # 字段引用 $close
        if tok and tok.startswith("$"):
            return ("field", self.consume()[1:])
        # 数字常量
        if tok and (tok[0].isdigit() or tok == "."):
            return ("num", float(self.consume()))
        # 裸字段名 (无 $)
        if tok and tok[0].isalpha():
            return ("field", self.consume())
        raise ValueError(f"无法解析的 token: {tok!r}")


def _eval_node(node, df: pd.DataFrame) -> pd.Series:
    """递归求值"""
    kind = node[0]
    if kind == "num":
        n = node[1]
        return pd.Series(n, index=df.index)
    if kind == "field":
        col = node[1]
        if col not in df.columns:
            raise KeyError(f"字段不存在: {col}")
        return df[col]
    if kind == "neg":
        return -_eval_node(node[1], df)
    if kind == "binop":
        _, op, l, r = node
        lv = _eval_node(l, df)
        rv = _eval_node(r, df)
        if op == "+":
            return lv + rv
        if op == "-":
            return lv - rv
        if op == "*":
            return lv * rv
        if op == "/":
            return lv / rv.replace(0, np.nan)
    if kind == "call":
        _, name, args = node
        fn = OperatorRegistry.get(name)
        if fn is None:
            raise KeyError(f"未知算子: {name}")
        # 数字参数直接传值，字段/子表达式先求值
        evaluated = []
        for a in args:
            if a[0] == "num":
                evaluated.append(a[1])
            elif a[0] == "field":
                evaluated.append(a[1])
            else:
                # 子表达式结果作为临时列传入
                tmp_col = f"__tmp_{id(a)}"
                df = df.assign(**{tmp_col: _eval_node(a, df)})
                evaluated.append(tmp_col)
        return fn(df, *evaluated)
    raise ValueError(f"未知节点类型: {kind}")


# ── 公共 API ────────────────────────────────────────────────
class FactorExpressionEngine:
    """因子表达式引擎"""

    def __init__(self):
        self.registry = OperatorRegistry

    def compute(self, data: pd.DataFrame, expressions: Dict[str, str]) -> pd.DataFrame:
        """
        批量计算因子。

        参数:
            data: 必须列 code, date, close。可选 open/high/low/volume/amount/turnover_rate
            expressions: {因子名: 表达式}，如
                {"momentum_20": "Ref($close, 20) / $close - 1",
                 "vol_20": "Std($close / Ref($close, 1) - 1, 20)",
                 "reversal_5": "-(Ref($close, 5) / $close - 1)"}

        返回:
            DataFrame[code, date, <各因子列>]
        """
        if data.empty:
            return pd.DataFrame(columns=["code", "date"])

        df = data.sort_values(["code", "date"]).copy()
        result = df[["code", "date"]].copy()

        for name, expr in expressions.items():
            try:
                tokens = _tokenize(expr)
                ast = _Parser(tokens).parse()
                values = _eval_node(ast, df)
                result[name] = values.values
            except Exception as e:
                raise ValueError(f"因子 {name!r} 表达式 {expr!r} 求值失败: {e}") from e
        return result

    def list_operators(self) -> List[str]:
        return self.registry.list_operators()


# ── 预置因子库 (借鉴 Qlib Alpha158 思路) ───────────────────
PRESET_FACTORS: Dict[str, str] = {
    # 动量/反转
    "momentum_5": "Ref($close, 5) / $close - 1",
    "momentum_20": "Ref($close, 20) / $close - 1",
    "momentum_60": "Ref($close, 60) / $close - 1",
    "reversal_5": "-(Ref($close, 5) / $close - 1)",
    "reversal_20": "-(Ref($close, 20) / $close - 1)",
    # 波动率
    "vol_5": "Std($close / Ref($close, 1) - 1, 5)",
    "vol_20": "Std($close / Ref($close, 1) - 1, 20)",
    "vol_60": "Std($close / Ref($close, 1) - 1, 60)",
    # 均线偏离
    "ma_bias_20": "$close / Mean($close, 20) - 1",
    "ma_bias_60": "$close / Mean($close, 60) - 1",
    # 成交量
    "volume_ratio_20": "$volume / Mean($volume, 20)",
    "volume_std_20": "Std($volume, 20) / Mean($volume, 20)",
    # 振幅
    "range_20": "(Max($high, 20) - Min($low, 20)) / $close",
    # 换手 (若数据含 turnover_rate)
    "turnover_mean_20": "Mean($turnover_rate, 20)",
}
