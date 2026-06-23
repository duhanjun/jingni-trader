"""
优化方向 1：因子表达式引擎（Expression Engine）

借鉴来源：Microsoft Qlib 的表达式 DSL 设计
- Qlib 论文: https://arxiv.org/abs/2009.11189
- Qlib ops.py: https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py

问题分析（jingni-trader 现状）：
- skills/factor-engine/scripts/adapters/pandas_ta_calculator.py 使用 if-elif 硬编码
  每个因子，新增因子必须修改 _calc_factor 方法，扩展性差
- 不支持横截面算子（Rank/Quantile），无法表达 "先时间序列后横截面" 的复合因子
- 因子定义与计算逻辑耦合，无法序列化/共享/复现

优化方案：
- 实现表达式引擎，用字符串表达式定义因子，如 "Ref($close, 5) / $close"
- 支持时间序列算子：Ref / Mean / Std / Max / Min / Rank / Delta / Correlation
- 支持横截面算子：CSRank（横截面排名）/ CSZscore（横截面标准化）
- 表达式可解析、可序列化、可缓存，因子库可无限扩展而无需改代码
"""
from __future__ import annotations

import re
import operator as op
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 算子注册表
# ---------------------------------------------------------------------------

class Ops:
    """算子注册表：所有算子按 (时间序列 / 横截面 / 元素) 三类注册"""

    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, kind: str = "element"):
        """装饰器：注册算子

        kind:
            - "ts": 时间序列算子，作用于单只股票的时间维度（groupby code 后 rolling）
            - "cs": 横截面算子，作用于同一日期的所有股票
            - "element": 元素级算子，逐元素运算
        """

        def decorator(func: Callable):
            cls._registry[name] = {"func": func, "kind": kind}
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        return cls._registry.get(name)

    @classmethod
    def list_ops(cls) -> List[str]:
        return sorted(cls._registry.keys())


# ---------------------------------------------------------------------------
# 时间序列算子（输入: 单只股票的 Series，按 date 排序）
# ---------------------------------------------------------------------------

@Ops.register("Ref", "ts")
def _ref(s: pd.Series, n: int) -> pd.Series:
    """取 n 天前的值"""
    return s.shift(n)


@Ops.register("Mean", "ts")
def _mean(s: pd.Series, n: int) -> pd.Series:
    """n 日滚动均值"""
    return s.rolling(n, min_periods=max(1, n // 2)).mean()


@Ops.register("Std", "ts")
def _std(s: pd.Series, n: int) -> pd.Series:
    """n 日滚动标准差"""
    return s.rolling(n, min_periods=max(1, n // 2)).std()


@Ops.register("Max", "ts")
def _max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n // 2)).max()


@Ops.register("Min", "ts")
def _min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n // 2)).min()


@Ops.register("Delta", "ts")
def _delta(s: pd.Series, n: int) -> pd.Series:
    """与 n 天前的差"""
    return s - s.shift(n)


@Ops.register("Rank", "ts")
def _ts_rank(s: pd.Series, n: int) -> pd.Series:
    """时间序列排名：当前值在过去 n 天中的分位"""
    return s.rolling(n, min_periods=max(1, n // 2)).rank(pct=True)


@Ops.register("WMA", "ts")
def _wma(s: pd.Series, n: int) -> pd.Series:
    """加权移动平均（线性权重）"""
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()

    def _w(x):
        return np.dot(x, weights) if len(x) == n else np.nan

    return s.rolling(n).apply(_w, raw=True)


@Ops.register("Corr", "ts")
def _corr(s: pd.Series, n: int, other: pd.Series) -> pd.Series:
    """与另一序列的 n 日滚动相关系数"""
    return s.rolling(n, min_periods=max(2, n // 2)).corr(other)


# ---------------------------------------------------------------------------
# 横截面算子（输入: 单日所有股票的 Series）
# ---------------------------------------------------------------------------

@Ops.register("CSRank", "cs")
def _cs_rank(s: pd.Series) -> pd.Series:
    """横截面排名（百分位 0~1）"""
    return s.rank(pct=True)


@Ops.register("CSZscore", "cs")
def _cs_zscore(s: pd.Series) -> pd.Series:
    """横截面 Z-score 标准化"""
    mu = s.mean()
    sd = s.std()
    return (s - mu) / sd if sd and sd > 0 else s - mu


@Ops.register("CSQuantile", "cs")
def _cs_quantile(s: pd.Series) -> pd.Series:
    """横截面分位（0~1）"""
    return s.rank(method="average", pct=True)


# ---------------------------------------------------------------------------
# 元素级算子
# ---------------------------------------------------------------------------

@Ops.register("Abs", "element")
def _abs(s: pd.Series) -> pd.Series:
    return s.abs()


@Ops.register("Log", "element")
def _log(s: pd.Series) -> pd.Series:
    return np.log(s.clip(lower=1e-12))


@Ops.register("Sign", "element")
def _sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


# ---------------------------------------------------------------------------
# 表达式解析器（递归下降）
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.?\d*)|(?P<str>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[+\-*/(),$]))"
)


def _tokenize(expr: str) -> List[str]:
    tokens = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            if expr[pos].isspace():
                pos += 1
                continue
            raise ValueError(f"无法解析的字符: {expr[pos]!r} 于位置 {pos}")
        pos = m.end()
        for g in ("num", "str", "op"):
            v = m.group(g)
            if v is not None:
                tokens.append(v)
                break
    return tokens


class _Parser:
    """递归下降解析器，生成 AST（用嵌套 tuple 表示）

    文法:
        expr   := term (('+'|'-') term)*
        term   := factor (('*'|'/') factor)*
        factor := number | field | func '(' args ')' | '(' expr ')'
        args   := expr (',' expr)*
        field  := '$' name | name
    """

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.i = 0

    def _peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _next(self) -> str:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def parse(self):
        node = self._expr()
        if self.i != len(self.tokens):
            raise ValueError(f"未消费的 token: {self.tokens[self.i:]}")
        return node

    def _expr(self):
        node = self._term()
        while self._peek() in ("+", "-"):
            o = self._next()
            right = self._term()
            node = ("binop", o, node, right)
        return node

    def _term(self):
        node = self._factor()
        while self._peek() in ("*", "/"):
            o = self._next()
            right = self._factor()
            node = ("binop", o, node, right)
        return node

    def _factor(self):
        t = self._peek()
        if t == "(":
            self._next()
            node = self._expr()
            if self._next() != ")":
                raise ValueError("缺少右括号 )")
            return node
        if t == "-":
            self._next()
            return ("neg", self._factor())
        # 数字
        try:
            v = float(t)
            self._next()
            return ("num", v)
        except (TypeError, ValueError):
            pass
        # 标识符：可能是函数调用或字段
        if t and (t[0].isalpha() or t[0] == "_"):
            self._next()
            if self._peek() == "(":
                self._next()
                args = [self._expr()]
                while self._peek() == ",":
                    self._next()
                    args.append(self._expr())
                if self._next() != ")":
                    raise ValueError("函数调用缺少右括号 )")
                return ("call", t, args)
            # 字段引用（可带 $ 前缀，tokenize 时 $ 已被忽略或保留）
            return ("field", t)
        # $field 形式
        if t == "$":
            self._next()
            name = self._next()
            return ("field", name)
        raise ValueError(f"意外的 token: {t!r}")


# ---------------------------------------------------------------------------
# 表达式引擎
# ---------------------------------------------------------------------------

class ExpressionEngine:
    """因子表达式引擎

    用法:
        engine = ExpressionEngine()
        df = engine.evaluate("Ref($close, 5) / $close - 1", ohlcv_df)
        # df 包含 code, date, factor 列

    支持的字段: $open $high $low $close $volume $amount 等（OHLCV 列名）
    """

    def __init__(self):
        self._cache: Dict[str, tuple] = {}

    def evaluate(self, expr: str, data: pd.DataFrame) -> pd.DataFrame:
        """在 data 上计算表达式，返回 [code, date, factor] DataFrame"""
        ast = self._parse(expr)
        result = self._eval_node(ast, data)
        out = data[["code", "date"]].copy()
        out["factor"] = result.values
        return out

    def evaluate_many(
        self, expressions: Dict[str, str], data: pd.DataFrame
    ) -> pd.DataFrame:
        """批量计算多个表达式因子，返回 [code, date, *因子列]"""
        out = data[["code", "date"]].copy()
        for name, expr in expressions.items():
            ast = self._parse(expr)
            out[name] = self._eval_node(ast, data).values
        return out

    # -- 内部 --

    def _parse(self, expr: str):
        if expr not in self._cache:
            tokens = _tokenize(expr)
            self._cache[expr] = _Parser(tokens).parse()
        return self._cache[expr]

    def _eval_node(self, node, data: pd.DataFrame) -> pd.Series:
        kind = node[0]

        if kind == "num":
            return pd.Series(node[1], index=data.index)

        if kind == "field":
            col = node[1]
            if col not in data.columns:
                raise KeyError(f"字段不存在: {col}（可用: {list(data.columns)}）")
            return data[col].astype(float)

        if kind == "neg":
            return -self._eval_node(node[1], data)

        if kind == "binop":
            _, opstr, l, r = node
            lv = self._eval_node(l, data)
            rv = self._eval_node(r, data)
            ops = {"+": op.add, "-": op.sub, "*": op.mul, "/": op.truediv}
            return ops[opstr](lv, rv)

        if kind == "call":
            _, name, args = node
            info = Ops.get(name)
            if info is None:
                raise KeyError(f"未知算子: {name}（已注册: {Ops.list_ops()}）")
            # 标量参数（如 Ref($close, 5) 中的 5）直接传 Python 标量，避免被包成 Series
            evaluated = []
            for a in args:
                if a[0] == "num":
                    v = a[1]
                    # 整数参数转为 int（如 Ref($close, 5) 中的 5 供 shift 使用）
                    evaluated.append(int(v) if v == int(v) else v)
                else:
                    evaluated.append(self._eval_node(a, data))
            return self._apply_op(info, evaluated, data)

        raise ValueError(f"未知 AST 节点: {node}")

    def _apply_op(self, info: Dict[str, Any], args: List, data: pd.DataFrame) -> pd.Series:
        kind = info["kind"]
        func = info["func"]

        if kind == "element":
            return func(args[0])

        if kind == "ts":
            # 时间序列算子：按 code 分组，保持 date 顺序
            s = args[0]
            rest = args[1:]
            # 构造临时 df 以便分组（仅对 Series 类型的参数建列）
            tmp = pd.DataFrame({"_v": s, "code": data["code"], "date": data["date"]})
            series_idx = []
            for i, r in enumerate(rest):
                if isinstance(r, pd.Series):
                    tmp[f"_a{i}"] = r
                    series_idx.append(i)
            tmp = tmp.sort_values(["code", "date"])

            def _g(g):
                call_args = [g["_v"]]
                for i, r in enumerate(rest):
                    if isinstance(r, pd.Series):
                        call_args.append(g[f"_a{i}"])
                    else:
                        call_args.append(r)  # 标量
                return func(*call_args)

            res = tmp.groupby("code", group_keys=False).apply(_g)
            # 对齐回原索引
            return res.reindex(data.index)

        if kind == "cs":
            # 横截面算子：按 date 分组
            s = args[0]
            tmp = pd.DataFrame({"_v": s, "date": data["date"]})
            res = tmp.groupby("date")["_v"].transform(func)
            return res

        raise ValueError(f"未知算子类型: {kind}")
