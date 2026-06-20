"""
因子表达式引擎 - 优化验证模块

借鉴来源:
  - Microsoft Qlib: Alpha158/Alpha101 因子表达式 DSL, 算子 (Ref/Mean/Std/Rank/Corr)
  - akquant: Polars 驱动的因子表达式引擎, 支持 `Rank(Ts_Mean(Close, 5))` 风格公式
  - WorldQuant Alpha101: 经典 101 个 alpha 公式定义

针对 jingni-trader 现有 factor-engine 的问题:
  1. 因子计算硬编码在 PandasTaCalculator._calc_factor 的 if-elif 链中
  2. 添加新因子需要修改源码, 可扩展性差
  3. 无法表达复合因子 (如 Rank(Ts_Mean(Close, 5)) - Ts_Std(Volume, 20))
  4. A股专有因子 (reversal_20d, turnover_change 等) 也是硬编码

本模块实现:
  - 表达式 Parser: 支持 +, -, *, /, () 运算符
  - 算子库: Ref, Ts_Mean, Ts_Std, Ts_Max, Ts_Min, Ts_Rank, Rank, Delta,
           Corr, Cov, Ts_Sum, Abs, Log, Sign, EMA
  - 字段: Open, High, Low, Close, Volume, Amount, Turnover, Vwap
  - 按股票分组并行计算 (groupby + transform)
  - 因子注册机制: 注册自定义算子/字段

注意: 本文件仅用于优化验证, 不修改 main 分支任何代码。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------- 算子定义 ----------------

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, dtype=float)


def op_ref(s: pd.Series, n: int) -> pd.Series:
    """Ref(s, n): s.shift(n)"""
    return s.shift(n)


def op_delta(s: pd.Series, n: int) -> pd.Series:
    """Delta(s, n): s - s.shift(n)"""
    return s - s.shift(n)


def op_ts_mean(s: pd.Series, n: int) -> pd.Series:
    """Ts_Mean(s, n): 滚动均值"""
    return s.rolling(n, min_periods=max(1, n // 2)).mean()


def op_ts_std(s: pd.Series, n: int) -> pd.Series:
    """Ts_Std(s, n): 滚动标准差"""
    return s.rolling(n, min_periods=max(1, n // 2)).std()


def op_ts_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n // 2)).max()


def op_ts_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n // 2)).min()


def op_ts_sum(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n // 2)).sum()


def op_ts_rank(s: pd.Series, n: int) -> pd.Series:
    """Ts_Rank(s, n): 当前值在过去 n 期的分位数"""
    def _rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]
    return s.rolling(n, min_periods=max(1, n // 2)).apply(_rank, raw=False)


def op_ema(s: pd.Series, n: int) -> pd.Series:
    """EMA(s, n): 指数移动平均"""
    return s.ewm(span=n, adjust=False).mean()


def op_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Corr(a, b, n): 滚动相关系数"""
    return a.rolling(n, min_periods=max(1, n // 2)).corr(b)


def op_cov(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Cov(a, b, n): 滚动协方差"""
    return a.rolling(n, min_periods=max(1, n // 2)).cov(b)


def op_abs(s: pd.Series) -> pd.Series:
    return s.abs()


def op_log(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan))


def op_sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


# 截面算子 (cross-sectional, 按 date 分组)
def op_rank(s: pd.Series, group: pd.Series) -> pd.Series:
    """Rank(s): 截面排名 (按 date 分组)"""
    df = pd.DataFrame({"s": s, "g": group})
    return df.groupby("g")["s"].rank(pct=True)


# ---------------- 表达式 AST ----------------

@dataclass
class Token:
    type: str  # 'NUM', 'FIELD', 'FUNC', 'OP', 'LPAREN', 'RPAREN', 'COMMA'
    value: str


@dataclass
class ASTNode:
    op: str
    args: List = field(default_factory=list)
    value: Optional[float] = None
    field: Optional[str] = None


class FactorExpressionParser:
    """
    因子表达式解析器

    语法:
        expr    := term (('+' | '-') term)*
        term    := factor (('*' | '/') factor)*
        factor  := NUMBER | FIELD | FUNC '(' args ')' | '(' expr ')'
        args    := expr (',' expr)*
        FUNC    := identifier
        FIELD   := identifier (大写开头, 如 Close, Volume)
        NUMBER  := [0-9]+(.[0-9]+)?

    示例:
        Close
        Ts_Mean(Close, 5)
        Rank(Ts_Mean(Close, 5))
        (Close - Ref(Close, 5)) / Ref(Close, 5)
        Corr(Volume, Close, 10)
    """

    TOKEN_RE = re.compile(
        r"\s*("
        r"(?P<NUM>[0-9]+(?:\.[0-9]+)?)"
        r"|(?P<FUNC>[A-Za-z_][A-Za-z0-9_]*)\s*(?=\()"
        r"|(?P<FIELD>[A-Z][A-Za-z0-9_]*)"
        r"|(?P<OP>[+\-*/])"
        r"|(?P<LPAREN>\()"
        r"|(?P<RPAREN>\))"
        r"|(?P<COMMA>,)"
        r")"
    )

    def __init__(self, custom_funcs: Optional[Dict[str, Callable]] = None):
        self.builtin_funcs: Dict[str, Callable] = {
            "Ref": op_ref,
            "Delta": op_delta,
            "Ts_Mean": op_ts_mean,
            "Ts_Std": op_ts_std,
            "Ts_Max": op_ts_max,
            "Ts_Min": op_ts_min,
            "Ts_Sum": op_ts_sum,
            "Ts_Rank": op_ts_rank,
            "EMA": op_ema,
            "Corr": op_corr,
            "Cov": op_cov,
            "Abs": op_abs,
            "Log": op_log,
            "Sign": op_sign,
        }
        if custom_funcs:
            self.builtin_funcs.update(custom_funcs)

    def tokenize(self, expr: str) -> List[Token]:
        tokens = []
        pos = 0
        while pos < len(expr):
            m = self.TOKEN_RE.match(expr, pos)
            if not m:
                if expr[pos:].strip() == "":
                    break
                raise SyntaxError(f"无法解析的表达式位置 {pos}: {expr[pos:pos+10]}")
            pos = m.end()
            for ttype in ("NUM", "FUNC", "FIELD", "OP", "LPAREN", "RPAREN", "COMMA"):
                if m.group(ttype):
                    tokens.append(Token(ttype, m.group(ttype)))
                    break
        return tokens

    def parse(self, expr: str) -> ASTNode:
        tokens = self.tokenize(expr)
        node, idx = self._parse_expr(tokens, 0)
        if idx != len(tokens):
            raise SyntaxError(f"未消费完的 token: {tokens[idx:]}")
        return node

    def _parse_expr(self, tokens: List[Token], i: int) -> Tuple[ASTNode, int]:
        node, i = self._parse_term(tokens, i)
        while i < len(tokens) and tokens[i].type == "OP" and tokens[i].value in ("+", "-"):
            op = tokens[i].value
            i += 1
            right, i = self._parse_term(tokens, i)
            node = ASTNode(op="BINOP", args=[node, right], value=op)
        return node, i

    def _parse_term(self, tokens: List[Token], i: int) -> Tuple[ASTNode, int]:
        node, i = self._parse_factor(tokens, i)
        while i < len(tokens) and tokens[i].type == "OP" and tokens[i].value in ("*", "/"):
            op = tokens[i].value
            i += 1
            right, i = self._parse_factor(tokens, i)
            node = ASTNode(op="BINOP", args=[node, right], value=op)
        return node, i

    def _parse_factor(self, tokens: List[Token], i: int) -> Tuple[ASTNode, int]:
        tok = tokens[i]
        # 一元负号: -expr
        if tok.type == "OP" and tok.value == "-":
            node, i = self._parse_factor(tokens, i + 1)
            return ASTNode(op="UNARY", args=[node], value="-"), i
        # 一元正号: +expr (直接返回子节点)
        if tok.type == "OP" and tok.value == "+":
            return self._parse_factor(tokens, i + 1)
        if tok.type == "NUM":
            return ASTNode(op="NUM", value=float(tok.value)), i + 1
        if tok.type == "FIELD":
            return ASTNode(op="FIELD", field=tok.value), i + 1
        if tok.type == "LPAREN":
            node, i = self._parse_expr(tokens, i + 1)
            if i >= len(tokens) or tokens[i].type != "RPAREN":
                raise SyntaxError("缺少右括号 )")
            return node, i + 1
        if tok.type == "FUNC":
            fname = tok.value
            i += 1
            if i >= len(tokens) or tokens[i].type != "LPAREN":
                raise SyntaxError(f"函数 {fname} 后必须跟左括号 (")
            i += 1
            args = []
            if tokens[i].type != "RPAREN":
                arg, i = self._parse_expr(tokens, i)
                args.append(arg)
                while i < len(tokens) and tokens[i].type == "COMMA":
                    i += 1
                    arg, i = self._parse_expr(tokens, i)
                    args.append(arg)
            if i >= len(tokens) or tokens[i].type != "RPAREN":
                raise SyntaxError(f"函数 {fname} 缺少右括号 )")
            return ASTNode(op="FUNC", args=args, field=fname), i + 1
        raise SyntaxError(f"意外的 token: {tok.type}={tok.value}")


# ---------------- 因子表达式引擎 ----------------

class FactorExpressionEngine:
    """
    因子表达式引擎

    用法:
        engine = FactorExpressionEngine()
        engine.register_field("Vwap", lambda df: df["amount"] / df["volume"])
        result = engine.compute(
            data=df,
            expressions={
                "alpha_001": "(Close - Ref(Close, 5)) / Ref(Close, 5)",
                "alpha_reversal_20": "-Ts_Mean(Return, 20)",
                "alpha_vol_corr": "Corr(Volume, Close, 10)",
            }
        )
    """

    # 默认字段映射 (字段名 -> DataFrame 列名)
    DEFAULT_FIELDS = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Amount": "amount",
        "Turnover": "turnover_rate",
    }

    def __init__(self, custom_fields: Optional[Dict[str, str]] = None):
        self.fields = dict(self.DEFAULT_FIELDS)
        if custom_fields:
            self.fields.update(custom_fields)
        self.parser = FactorExpressionParser()
        self._cache: Dict[str, ASTNode] = {}

    def register_field(self, name: str, column: str) -> None:
        """注册自定义字段"""
        self.fields[name] = column

    def register_func(self, name: str, func: Callable) -> None:
        """注册自定义算子"""
        self.parser.builtin_funcs[name] = func

    def compute(
        self,
        data: pd.DataFrame,
        expressions: Dict[str, str],
    ) -> pd.DataFrame:
        """
        批量计算因子表达式

        参数:
            data: 行情数据, 必须含列 code, date, open, high, low, close, volume
            expressions: {因子名: 表达式字符串}

        返回:
            DataFrame, 列为 code, date, [各因子列]
        """
        if data.empty:
            return pd.DataFrame()

        df = data.sort_values(["code", "date"]).reset_index(drop=True).copy()

        # 预计算常用派生字段 Return (每次都计算, 因为 df 是新的)
        if "close" in df.columns:
            df["_return"] = df.groupby("code")["close"].pct_change()
            self.fields["Return"] = "_return"

        result = df[["code", "date"]].copy()

        for factor_name, expr in expressions.items():
            try:
                values = self._eval_expression(expr, df)
                result[factor_name] = values
            except Exception as e:
                raise RuntimeError(f"计算因子 {factor_name} 失败 (expr={expr}): {e}") from e

        return result

    def _eval_expression(self, expr: str, df: pd.DataFrame) -> pd.Series:
        """求值单个表达式"""
        if expr not in self._cache:
            self._cache[expr] = self.parser.parse(expr)
        ast = self._cache[expr]
        return self._eval_node(ast, df)

    def _eval_node(self, node: ASTNode, df: pd.DataFrame) -> pd.Series:
        if node.op == "NUM":
            return pd.Series(float(node.value), index=df.index)
        if node.op == "FIELD":
            col = self.fields.get(node.field)
            if col is None:
                raise KeyError(f"未知字段: {node.field}")
            if col not in df.columns:
                raise KeyError(f"字段 {node.field} 映射的列 {col} 不在 DataFrame 中")
            return df[col].astype(float)
        if node.op == "UNARY":
            val = self._eval_node(node.args[0], df)
            if node.value == "-":
                return -val
            return val
        if node.op == "BINOP":
            left = self._eval_node(node.args[0], df)
            right = self._eval_node(node.args[1], df)
            op = node.value
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right.replace(0, np.nan)
        if node.op == "FUNC":
            fname = node.field

            # 截面算子特殊处理 (需要 date 分组, 不在 builtin_funcs 中)
            if fname == "Rank":
                args = [self._eval_node(a, df) for a in node.args]
                return op_rank(args[0], df["date"])

            func = self.parser.builtin_funcs.get(fname)
            if func is None:
                raise KeyError(f"未知算子: {fname}")
            # 求值参数
            args = [self._eval_node(a, df) for a in node.args]
            # 数值参数 (如 Ts_Mean(Close, 5) 中的 5) 会被求值为常量 Series
            # 提取为标量
            scalar_args = []
            for a in args:
                if isinstance(a, pd.Series) and a.nunique() == 1:
                    scalar_args.append(float(a.iloc[0]))
                else:
                    scalar_args.append(a)

            # 时序算子: 按股票分组应用
            return self._apply_ts_func(func, args, scalar_args, df, func_name=fname)

        raise RuntimeError(f"未知节点类型: {node.op}")

    # 二元时序算子 (两个 Series 参数): Corr, Cov
    BINARY_TS_FUNCS = {"Corr", "Cov"}

    def _apply_ts_func(
        self,
        func: Callable,
        args: List,
        scalar_args: List,
        df: pd.DataFrame,
        func_name: str = "",
    ) -> pd.Series:
        """应用时序算子 (按 code 分组)"""
        s = args[0]

        # 二元算子 (Corr, Cov): 两个 Series 参数 + 一个窗口期
        if func_name in self.BINARY_TS_FUNCS and len(args) >= 2 and isinstance(args[1], pd.Series):
            s2 = args[1]
            # 窗口期从 scalar_args 中找 (跳过第一个 Series 对应的常量)
            # scalar_args = [s 的常量值, s2 的常量值, n 的常量值]
            # 但 s/s2 是 Series, scalar_args 中对应位置是它们 (非标量)
            # 所以 n 是 scalar_args 中唯一的标量
            n = None
            for a in scalar_args[1:]:
                if isinstance(a, (int, float)):
                    n = int(a)
                    break
            if n is None:
                raise RuntimeError(f"二元算子 {func_name} 缺少窗口期参数")
            tmp = pd.DataFrame({"s1": s, "s2": s2, "code": df["code"]})
            result = tmp.groupby("code").apply(
                lambda g: func(g["s1"], g["s2"], n),
                include_groups=False,
            )
            if isinstance(result.index, pd.MultiIndex):
                result = result.reset_index(level=0, drop=True)
            return result.sort_index()

        # 一元时序算子 (Ts_Mean, Ref, ...): 一个 Series + 一个标量窗口期
        # 找出 scalar_args 中的窗口期 (跳过第一个, 它对应 s)
        n = None
        for a in scalar_args[1:]:
            if isinstance(a, (int, float)):
                n = int(a)
                break
        if n is not None:
            tmp = pd.DataFrame({"s": s, "code": df["code"]})
            return tmp.groupby("code")["s"].transform(lambda x: func(x, n))

        # 无参数算子 (Abs, Log, Sign): 仅一个 Series
        return func(s)
