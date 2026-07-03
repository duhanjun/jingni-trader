"""
Polars 向量化因子计算引擎 + 因子表达式 DSL

借鉴来源：
- AKQuant: Polars 驱动的高性能因子计算引擎，支持 Alpha101 风格公式
- Microsoft Qlib: Alpha158 因子库设计、Point-in-time 数据处理

核心改进点（对照 jingni-trader/skills/factor-engine/engine.py）：
1. 用 Polars 替代 pandas rolling + groupby.transform(lambda) 的逐列循环，
   利用 Polars 的列式向量化与表达式并行获得显著性能提升。
2. 引入因子表达式 DSL（如 "Rank(Ts_Mean(Close, 5))"），
   让因子定义可声明式配置，避免每加一个因子都要改 compute_a_share_factors。
3. Point-in-time 安全：所有时序算子强制 min_periods，杜绝未来函数。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# 因子表达式 DSL
# ---------------------------------------------------------------------------

# 算子签名：算子名 -> (参数个数, 实现函数)
# 实现函数签名: (col_expr: pl.Expr, *params: int|float) -> pl.Expr
# 时序算子在 groupby("code") 上下文中执行，因此内部使用 .over("code")

def _op_ref(col: pl.Expr, n: int) -> pl.Expr:
    """Ts_Ref(x, n): 取 n 期前的值"""
    return col.shift(n).over("code")


def _op_mean(col: pl.Expr, n: int) -> pl.Expr:
    """Ts_Mean(x, n): n 期滚动均值"""
    return col.rolling_mean(window_size=n, min_periods=max(1, n // 2)).over("code")


def _op_std(col: pl.Expr, n: int) -> pl.Expr:
    """Ts_Std(x, n): n 期滚动标准差"""
    return col.rolling_std(window_size=n, min_periods=max(2, n // 2)).over("code")


def _op_max(col: pl.Expr, n: int) -> pl.Expr:
    return col.rolling_max(window_size=n, min_periods=1).over("code")


def _op_min(col: pl.Expr, n: int) -> pl.Expr:
    return col.rolling_min(window_size=n, min_periods=1).over("code")


def _op_sum(col: pl.Expr, n: int) -> pl.Expr:
    return col.rolling_sum(window_size=n, min_periods=1).over("code")


def _op_delta(col: pl.Expr, n: int) -> pl.Expr:
    """Ts_Delta(x, n) = x - Ts_Ref(x, n)"""
    return (col - col.shift(n).over("code")).over("code")


def _op_rank(col: pl.Expr) -> pl.Expr:
    """横截面 Rank（按 date 分组，pct rank）"""
    return col.rank(method="average").over("date") / col.count().over("date")


def _op_zscore(col: pl.Expr) -> pl.Expr:
    """横截面 Z-score"""
    mean = col.mean().over("date")
    std = col.std().over("date")
    return (col - mean) / (std + 1e-12)


def _op_abs(col: pl.Expr) -> pl.Expr:
    return col.abs()


def _op_log(col: pl.Expr) -> pl.Expr:
    return col.log()


def _op_corr(a: pl.Expr, b: pl.Expr, n: int) -> pl.Expr:
    """Ts_Corr(a, b, n): n 期滚动相关系数"""
    # Polars rolling corr 需要手动构造
    a_mean = a.rolling_mean(window_size=n, min_periods=max(2, n // 2)).over("code")
    b_mean = b.rolling_mean(window_size=n, min_periods=max(2, n // 2)).over("code")
    a_std = a.rolling_std(window_size=n, min_periods=max(2, n // 2)).over("code")
    b_std = b.rolling_std(window_size=n, min_periods=max(2, n // 2)).over("code")
    cov = ((a - a_mean) * (b - b_mean)).rolling_mean(
        window_size=n, min_periods=max(2, n // 2)
    ).over("code")
    return cov / ((a_std * b_std) + 1e-12)


# 算子注册表：名称 -> (参数个数, 是否时序横截面混合, 实现函数)
OPERATORS: Dict[str, Tuple[int, Callable[..., pl.Expr]]] = {
    "Ts_Ref": (1, _op_ref),
    "Ts_Mean": (1, _op_mean),
    "Ts_Std": (1, _op_std),
    "Ts_Max": (1, _op_max),
    "Ts_Min": (1, _op_min),
    "Ts_Sum": (1, _op_sum),
    "Ts_Delta": (1, _op_delta),
    "Ts_Corr": (1, _op_corr),  # 接收两个列表达式 + 一个窗口数值参数
    "Rank": (0, _op_rank),
    "Zscore": (0, _op_zscore),
    "Abs": (0, _op_abs),
    "Log": (0, _op_log),
}

# 字段别名映射（DSL 中可用的字段名 -> DataFrame 列名）
FIELD_ALIASES: Dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Amount": "amount",
    "Turnover": "turnover_rate",
    "Vwap": "vwap",
}


class FactorExprParser:
    """
    因子表达式解析器

    支持 Alpha101 风格的嵌套表达式，例如：
        Rank(Ts_Mean(Close, 5))
        Ts_Delta(Log(Volume), 10)
        Ts_Corr(Close, Volume, 20)

    语法：
        expr     := func | field | number
        func     := IDENT '(' args ')'
        args     := expr (',' arg)*
        arg      := expr | number
        field    := IDENT  (必须在 FIELD_ALIASES 中)
        number   := [0-9]+ (.[0-9]+)?
    """

    _TOKEN_RE = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)|(?P<punct>[(),+\-*/]))")

    def __init__(self, expression: str):
        self.expr = expression
        self.tokens: List[Tuple[str, str]] = []
        self._tokenize()
        self.pos = 0

    def _tokenize(self) -> None:
        pos = 0
        while pos < len(self.expr):
            m = self._TOKEN_RE.match(self.expr, pos)
            if not m:
                if self.expr[pos].isspace():
                    pos += 1
                    continue
                raise ValueError(f"无法解析的字符: {self.expr[pos]!r} 于位置 {pos}")
            pos = m.end()
            if m.group("num"):
                self.tokens.append(("num", m.group("num")))
            elif m.group("ident"):
                self.tokens.append(("ident", m.group("ident")))
            elif m.group("punct"):
                self.tokens.append(("punct", m.group("punct")))

    def peek(self) -> Optional[Tuple[str, str]]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> Tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> pl.Expr:
        node = self._parse_additive()
        if self.pos != len(self.tokens):
            raise ValueError(f"表达式末尾有未消费的 token: {self.tokens[self.pos:]}")
        return node

    def _parse_additive(self) -> pl.Expr:
        """解析加减法（低优先级）"""
        left = self._parse_multiplicative()
        while True:
            tok = self.peek()
            if tok == ("punct", "+"):
                self.consume()
                right = self._parse_multiplicative()
                left = left + right
            elif tok == ("punct", "-"):
                self.consume()
                right = self._parse_multiplicative()
                left = left - right
            else:
                break
        return left

    def _parse_multiplicative(self) -> pl.Expr:
        """解析乘除法（高优先级）"""
        left = self._parse_expr()
        while True:
            tok = self.peek()
            if tok == ("punct", "*"):
                self.consume()
                right = self._parse_expr()
                left = left * right
            elif tok == ("punct", "/"):
                self.consume()
                right = self._parse_expr()
                left = left / right
            else:
                break
        return left

    def _parse_expr(self) -> pl.Expr:
        """解析原子：函数调用、字段、数值、括号"""
        tok = self.peek()
        if tok is None:
            raise ValueError("表达式意外结束")
        kind, val = tok
        if kind == "num":
            self.consume()
            n = float(val)
            return pl.lit(n)
        if kind == "ident":
            self.consume()
            # 函数调用？
            if self.peek() == ("punct", "("):
                self.consume()  # 吃掉 '('
                return self._parse_call(val)
            # 字段引用
            if val not in FIELD_ALIASES:
                raise ValueError(f"未知字段: {val}")
            return pl.col(FIELD_ALIASES[val])
        if tok == ("punct", "("):
            self.consume()  # 吃掉 '('
            node = self._parse_additive()
            if self.peek() != ("punct", ")"):
                raise ValueError(f"缺少右括号，当前 token: {self.peek()}")
            self.consume()  # 吃掉 ')'
            return node
        raise ValueError(f"意外的 token: {tok}")

    def _parse_call(self, name: str) -> pl.Expr:
        if name not in OPERATORS:
            raise ValueError(f"未知算子: {name}")
        n_params, impl = OPERATORS[name]
        args: List[Any] = []
        # 解析参数列表
        if self.peek() != ("punct", ")"):
            args.append(self._parse_arg())
            while self.peek() == ("punct", ","):
                self.consume()
                args.append(self._parse_arg())
        if self.peek() != ("punct", ")"):
            raise ValueError(f"缺少右括号，当前 token: {self.peek()}")
        self.consume()  # 吃掉 ')'

        # 区分列参数与数值参数
        col_args = [a for a in args if isinstance(a, pl.Expr)]
        num_args = [a for a in args if isinstance(a, (int, float))]
        if len(num_args) != n_params:
            raise ValueError(
                f"算子 {name} 需要 {n_params} 个数值参数，实际得到 {len(num_args)}"
            )
        if not col_args:
            raise ValueError(f"算子 {name} 至少需要一个列参数")
        if len(col_args) == 1:
            return impl(col_args[0], *num_args)
        # 多列算子（如 Ts_Corr）
        return impl(*col_args, *num_args)

    def _parse_arg(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise ValueError("参数解析意外结束")
        kind, val = tok
        if kind == "num":
            self.consume()
            # 整数参数更常见
            f = float(val)
            return int(f) if f.is_integer() else f
        # 列参数支持算术表达式（如 Ts_Ref(Close,0)/Ts_Ref(Close,1)-1）
        return self._parse_additive()


def compile_factor(expression: str) -> pl.Expr:
    """编译因子表达式为 Polars Expr"""
    return FactorExprParser(expression).parse()


# ---------------------------------------------------------------------------
# Polars 因子引擎
# ---------------------------------------------------------------------------


@dataclass
class FactorDef:
    """因子定义（声明式）"""
    name: str
    expression: str
    direction: int = 1  # 1 越大越看多, -1 越大越看空
    description: str = ""


# 预置因子库（借鉴 Qlib Alpha158 的精简子集）
BUILTIN_FACTORS: List[FactorDef] = [
    FactorDef("rev_5d", "Ts_Delta(Close, 5)", direction=-1, description="5日反转"),
    FactorDef("rev_20d", "Ts_Delta(Close, 20)", direction=-1, description="20日反转"),
    FactorDef("vol_20d", "Ts_Std(Ts_Ref(Close, 0) / Ts_Ref(Close, 1) - 1, 20)", description="20日波动率"),
    FactorDef("turnover_5d", "Ts_Mean(Turnover, 5)", description="5日均换手"),
    FactorDef("turnover_20d", "Ts_Mean(Turnover, 20)", description="20日均换手"),
    FactorDef("vol_ratio", "Volume / Ts_Mean(Volume, 20)", description="量比"),
    FactorDef("price_mom", "Close / Ts_Ref(Close, 20) - 1", description="20日动量"),
    FactorDef("amt_mom", "Amount / Ts_Mean(Amount, 20)", description="成交额动量"),
    FactorDef("high_low_corr", "Ts_Corr(High, Low, 20)", description="高低价相关性"),
]


class PolarsFactorEngine:
    """
    基于 Polars 的向量化因子引擎

    对照原 jingni-trader FactorEngine.compute_a_share_factors：
    - 原实现：逐列 groupby('code').transform(lambda x: x.rolling(...))，每次都触发 pandas 重排
    - 新实现：一次性构造 Polars LazyFrame，所有因子表达式并行下推执行
    """

    def __init__(self, factors: Optional[List[FactorDef]] = None):
        self.factors = factors if factors is not None else list(BUILTIN_FACTORS)
        # 预编译表达式
        self._compiled: List[Tuple[str, pl.Expr, int]] = [
            (f.name, compile_factor(f.expression), f.direction) for f in self.factors
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子

        参数:
            df: pandas DataFrame，必须包含 code, date, open, high, low, close,
                volume, amount, turnover_rate 列

        返回:
            pandas DataFrame，列为 code, date, [各因子]
        """
        if df.empty:
            return df[["code", "date"]].copy() if {"code", "date"}.issubset(df.columns) else df.copy()

        # 转 Polars，按 code/date 排序保证时序算子正确
        required = {"code", "date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

        pdf = pl.from_pandas(df).sort(["code", "date"])

        # 构造表达式列表
        exprs = []
        for name, expr, _direction in self._compiled:
            exprs.append(expr.alias(name))

        result = pdf.with_columns(exprs).select(
            ["code", "date"] + [name for name, _, _ in self._compiled]
        )
        return result.to_pandas()

    def compute_with_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算因子并按 direction 调整符号"""
        out = self.compute(df)
        for name, _expr, direction in self._compiled:
            if direction == -1:
                out[name] = -out[name]
        return out


# ---------------------------------------------------------------------------
# 向量化 IC 分析（替代原 engine.py 中的逐日期 Python 循环）
# ---------------------------------------------------------------------------


def vectorized_ic_analysis(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    ic_type: str = "spearman",
) -> Dict[str, Any]:
    """
    向量化 IC 分析

    对照原实现 _calc_ic：原版用 `for dt in dates: cross = data[data['date']==dt]`
    逐日循环 + scipy.stats.spearmanr，O(N_dates * N_stocks) 的 Python 开销很大。
    新实现用 Polars groupby("date").agg() 一次性算完所有日期 × 所有因子的 IC。
    """
    if factor_df.empty or forward_returns.empty:
        return {}

    # 只取实际存在的远期收益列
    fwd_cols = [c for c in ["ret_forward_1d", "ret_forward_5d", "ret_forward_20d"]
                if c in forward_returns.columns]
    if not fwd_cols:
        return {}

    merge = factor_df.merge(
        forward_returns[["code", "date"] + fwd_cols],
        on=["code", "date"],
        how="inner",
    )
    if merge.empty:
        return {}

    if factor_names is None:
        skip = {"code", "date", "industry"}
        factor_names = [c for c in factor_df.columns if c not in skip]

    pdf = pl.from_pandas(merge)

    results: Dict[str, Any] = {}
    for fwd_col in fwd_cols:
        if fwd_col not in pdf.columns:
            continue

        ic_rows = []
        for f in factor_names:
            if f not in pdf.columns:
                continue

            if ic_type == "spearman":
                # Spearman = Pearson on ranks
                grp = pdf.group_by("date").agg(
                    pl.corr(
                        pl.col(f).rank(method="average"),
                        pl.col(fwd_col).rank(method="average"),
                        method="pearson",
                    ).alias("ic")
                ).sort("date")
            else:
                grp = pdf.group_by("date").agg(
                    pl.corr(pl.col(f), pl.col(fwd_col), method="pearson").alias("ic")
                ).sort("date")

            ic_series = grp.to_pandas()["ic"].dropna()
            if ic_series.empty:
                continue

            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
            n = len(ic_series)
            ic_t = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0

            ic_rows.append({
                "factor": f,
                "forward_period": fwd_col,
                "ic_mean": round(ic_mean, 6),
                "ic_std": round(ic_std, 6),
                "ic_ir": round(ic_ir, 4),
                "ic_positive_ratio": round(float((ic_series > 0).mean()), 4),
                "ic_t_stat": round(ic_t, 4),
            })
        results[fwd_col] = ic_rows

    return results
