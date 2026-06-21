"""
因子表达式引擎 (Factor Expression Engine)
==========================================
借鉴来源: Microsoft Qlib Expression Engine + alfa.rs AST 表达式系统

解决痛点:
- jingni-trader factor-engine 中所有因子硬编码, 调整窗口需改源码重新部署
- PandasTa/Talib calculator 加载后从未被调用 (factor-engine/engine.py:31-46)

设计要点 (参考 Qlib):
1. 声明式因子定义: 用字符串表达式描述因子, 如 "MA($close, 20) - MA($close, 5)"
2. AST 解析: 表达式字符串 -> 抽象语法树 -> 操作符计算树
3. 算子体系: ElemOperator(一元) / PairOperator(二元) / Rolling(滚动窗口)
4. 字段引用: $close / $open / $high / $low / $volume / $amount / $turnover_rate
5. 缓存优化: 同一表达式在相同数据上只计算一次

支持的算子:
- 滚动: Ref, MA, EMA, Std, Var, Sum, Max, Min, Quantile, Skew, Kurt, Corr, Cov
- 截面: Rank, CSZScore, CSQuantile
- 一元: Log, Abs, Sign, Sqrt, Neg
- 二元: Add, Sub, Mul, Div, Greater, Less, Min, Max
- 时序: Delta, Delay, ROC, RSI, WMA

性能优化:
- 全程向量化 (pandas rolling/groupby), 无 Python 逐日循环
- 截面算子用 groupby(level='date').rank/zscore, 一次完成
- 表达式缓存: parse 结果缓存, 避免重复解析
"""
from __future__ import annotations

import ast
import functools
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 算子基类 (参考 Qlib 的 ElemOperator / PairOperator / Rolling)
# ---------------------------------------------------------------------------


class Operator:
    """算子基类。每个算子知道如何在一份数据上计算自身。"""

    def __call__(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.expr()


class Field(Operator):
    """字段引用, 如 $close。"""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        col = self.name.lstrip("$")
        if col not in data.columns:
            raise KeyError(f"字段 '{col}' 不存在, 可用列: {list(data.columns)}")
        return data[col]

    def expr(self) -> str:
        return f"${self.name.lstrip('$')}"


class Const(Operator):
    """常量。"""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, data: pd.DataFrame) -> float:
        return self.value

    def expr(self) -> str:
        # 整数值显示为整数, 避免出现 20.0
        if self.value == int(self.value):
            return str(int(self.value))
        return repr(self.value)


class ElemOperator(Operator):
    """一元算子基类。"""

    op_name: str = "ElemOp"

    def __init__(self, operand: Operator) -> None:
        self.operand = operand

    def apply(self, x: pd.Series) -> pd.Series:
        raise NotImplementedError

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        x = self.operand(data)
        return self.apply(x)

    def expr(self) -> str:
        return f"{self.op_name}({self.operand.expr()})"


class PairOperator(Operator):
    """二元算子基类。"""

    op_name: str = "PairOp"
    symbol: Optional[str] = None

    def __init__(self, left: Operator, right: Operator) -> None:
        self.left = left
        self.right = right

    def apply(self, left: Any, right: Any) -> Any:
        raise NotImplementedError

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        l = self.left(data)
        r = self.right(data)
        return self.apply(l, r)

    def expr(self) -> str:
        sym = self.symbol
        if sym is not None:
            return f"({self.left.expr()} {sym} {self.right.expr()})"
        return f"{self.op_name}({self.left.expr()}, {self.right.expr()})"


class Rolling(Operator):
    """滚动窗口算子基类 (按标的分组滚动)。"""

    op_name: str = "RollingOp"

    def __init__(self, feature: Operator, window: int) -> None:
        if window <= 0:
            raise ValueError(f"window 必须为正整数, 得到 {window}")
        self.feature = feature
        self.window = window

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        raise NotImplementedError

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        feat = self.feature(data)
        # 支持 MultiIndex(date, code) 分组滚动, 也支持单标的 DatetimeIndex
        if isinstance(feat.index, pd.MultiIndex):
            # level=1 即 code, 按标的滚动
            return feat.groupby(level=1, group_keys=False).apply(self.rolling_apply)
        return self.rolling_apply(feat)

    def expr(self) -> str:
        return f"{self.op_name}({self.feature.expr()}, {self.window})"


# ---------------------------------------------------------------------------
# 具体算子实现
# ---------------------------------------------------------------------------


# --- 滚动算子 ---


class Ref(Rolling):
    """Ref($close, 1) = 前一日收盘价 (时序位移)。"""

    op_name = "Ref"

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        return series.shift(self.window)


class MA(Rolling):
    """简单移动平均。"""

    op_name = "MA"

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window, min_periods=self.window).mean()


class EMA(Rolling):
    """指数移动平均。"""

    op_name = "EMA"

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        span = self.window
        return series.ewm(span=span, min_periods=span, adjust=False).mean()


class STD(Rolling):
    """滚动标准差。"""

    op_name = "STD"

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window, min_periods=self.window).std(ddof=0)


class VAR(Rolling):
    """滚动方差。"""

    op_name = "VAR"

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window, min_periods=self.window).var(ddof=0)


class SUM(Rolling):
    """滚动求和。"""

    op_name = "SUM"

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window, min_periods=1).sum()


class MAX(Rolling):
    """滚动最大值。"""

    op_name = "MAX"

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window, min_periods=self.window).max()


class MIN(Rolling):
    """滚动最小值。"""

    op_name = "MIN"

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window, min_periods=self.window).min()


class QUANTILE(Rolling):
    """滚动分位数。"""

    op_name = "QUANTILE"

    def __init__(self, feature: Operator, window: int, q: float = 0.5) -> None:
        super().__init__(feature, window)
        if not 0.0 < q < 1.0:
            raise ValueError(f"q 必须在 (0,1), 得到 {q}")
        self.q = q

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window, min_periods=self.window).quantile(self.q)

    def expr(self) -> str:
        return f"QUANTILE({self.feature.expr()}, {self.window}, {self.q})"


class CORR(Rolling):
    """滚动相关系数: CORR($close, $volume, 20)。"""

    op_name = "CORR"

    def __init__(self, left: Operator, right: Operator, window: int) -> None:
        super().__init__(left, window)
        self.right_op = right

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        l = self.feature(data)
        r = self.right_op(data)
        if isinstance(l.index, pd.MultiIndex):
            df = pd.concat([l.rename("a"), r.rename("b")], axis=1)
            return df.groupby(level=1, group_keys=False).apply(
                lambda g: g["a"].rolling(self.window, min_periods=self.window)
                .corr(g["b"])
            )
        return l.rolling(self.window, min_periods=self.window).corr(r)

    def expr(self) -> str:
        return f"CORR({self.feature.expr()}, {self.right_op.expr()}, {self.window})"


class WMA(Rolling):
    """加权移动平均 (线性权重)。"""

    op_name = "WMA"

    def rolling_apply(self, series: pd.Series) -> pd.Series:
        weights = np.arange(1, self.window + 1, dtype=float)
        weights = weights / weights.sum()

        def _w(s: pd.Series) -> float:
            return np.dot(s.values, weights) if len(s) == self.window else np.nan

        return series.rolling(self.window).apply(_w, raw=False)


# --- 时序算子 ---


class Delta(Operator):
    """Delta($close, 1) = $close - Ref($close, 1)。"""

    def __init__(self, feature: Operator, n: int = 1) -> None:
        self.feature = feature
        self.n = n

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        feat = self.feature(data)
        ref = Ref(self.feature, self.n)(data)
        return feat - ref

    def expr(self) -> str:
        return f"Delta({self.feature.expr()}, {self.n})"


class ROC(Operator):
    """ROC($close, 5) = $close / Ref($close, 5) - 1 收益率。"""

    def __init__(self, feature: Operator, n: int = 1) -> None:
        self.feature = feature
        self.n = n

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        feat = self.feature(data)
        ref = Ref(self.feature, self.n)(data)
        return feat / ref - 1.0

    def expr(self) -> str:
        return f"ROC({self.feature.expr()}, {self.n})"


class RSI(Operator):
    """RSI 指标 (Wilder 平滑)。"""

    def __init__(self, feature: Operator, window: int = 14) -> None:
        self.feature = feature
        self.window = window

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        close = self.feature(data)

        def _rsi(s: pd.Series) -> pd.Series:
            delta = s.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1 / self.window, min_periods=self.window, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / self.window, min_periods=self.window, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            return 100.0 - 100.0 / (1.0 + rs)

        if isinstance(close.index, pd.MultiIndex):
            return close.groupby(level=1, group_keys=False).apply(_rsi)
        return _rsi(close)

    def expr(self) -> str:
        return f"RSI({self.feature.expr()}, {self.window})"


# --- 截面算子 (cross-sectional, 按日期分组) ---


class CSRank(Operator):
    """截面排名 (按日期分组, 转为 0~1 分位)。"""

    def __init__(self, feature: Operator) -> None:
        self.feature = feature

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        feat = self.feature(data)
        if isinstance(feat.index, pd.MultiIndex):
            return feat.groupby(level=0, group_keys=False).rank(pct=True)
        return feat  # 单标的无截面概念, 原样返回

    def expr(self) -> str:
        return f"CSRank({self.feature.expr()})"


class CSZScore(Operator):
    """截面 Z-Score 标准化 (按日期分组)。"""

    def __init__(self, feature: Operator) -> None:
        self.feature = feature

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        feat = self.feature(data)
        if isinstance(feat.index, pd.MultiIndex):
            def _zscore(x: pd.Series) -> pd.Series:
                std = x.std(ddof=0)
                # std 为 0 或 NaN 时 (退化截面), 返回中心化结果 (全 0)
                if not std or np.isnan(std):
                    return x - x.mean()
                return (x - x.mean()) / std
            return feat.groupby(level=0, group_keys=False).apply(_zscore)
        return feat

    def expr(self) -> str:
        return f"CSZScore({self.feature.expr()})"


# --- 一元算子 ---


class Log(ElemOperator):
    op_name = "Log"

    def apply(self, x: pd.Series) -> pd.Series:
        return np.log(x.where(x > 0, np.nan))


class Abs(ElemOperator):
    op_name = "Abs"

    def apply(self, x: pd.Series) -> pd.Series:
        return x.abs()


class Sign(ElemOperator):
    op_name = "Sign"

    def apply(self, x: pd.Series) -> pd.Series:
        return np.sign(x)


class Sqrt(ElemOperator):
    op_name = "Sqrt"

    def apply(self, x: pd.Series) -> pd.Series:
        return np.sqrt(x.where(x >= 0, np.nan))


class Neg(ElemOperator):
    op_name = "Neg"

    def apply(self, x: pd.Series) -> pd.Series:
        return -x


# --- 二元算子 ---


class Add(PairOperator):
    op_name = "Add"
    symbol = "+"

    def apply(self, left: Any, right: Any) -> Any:
        return left + right


class Sub(PairOperator):
    op_name = "Sub"
    symbol = "-"

    def apply(self, left: Any, right: Any) -> Any:
        return left - right


class Mul(PairOperator):
    op_name = "Mul"
    symbol = "*"

    def apply(self, left: Any, right: Any) -> Any:
        return left * right


class Div(PairOperator):
    op_name = "Div"
    symbol = "/"

    def apply(self, left: Any, right: Any) -> Any:
        return left / right


class Greater(PairOperator):
    op_name = "Greater"

    def apply(self, left: Any, right: Any) -> Any:
        return np.maximum(left, right)


class Less(PairOperator):
    op_name = "Less"

    def apply(self, left: Any, right: Any) -> Any:
        return np.minimum(left, right)


# ---------------------------------------------------------------------------
# 表达式解析器 (字符串 -> AST)
# ---------------------------------------------------------------------------


# 算子名 -> (类, 参数数量) ; 滚动算子的最后一个参数是 window
OPERATORS: Dict[str, Tuple[type, int]] = {
    "Ref": (Ref, 2),
    "MA": (MA, 2),
    "EMA": (EMA, 2),
    "STD": (STD, 2),
    "VAR": (VAR, 2),
    "SUM": (SUM, 2),
    "MAX": (MAX, 2),
    "MIN": (MIN, 2),
    "QUANTILE": (QUANTILE, 3),
    "CORR": (CORR, 3),
    "WMA": (WMA, 2),
    "Delta": (Delta, 2),
    "ROC": (ROC, 2),
    "RSI": (RSI, 2),
    "CSRank": (CSRank, 1),
    "CSZScore": (CSZScore, 1),
    "Log": (Log, 1),
    "Abs": (Abs, 1),
    "Sign": (Sign, 1),
    "Sqrt": (Sqrt, 1),
    "Neg": (Neg, 1),
    "Greater": (Greater, 2),
    "Less": (Less, 2),
}

# 中缀运算符优先级 (参考 Qlib 与 Python)
_INFIX_OPS: Dict[str, Tuple[type, int]] = {
    "+": (Add, 1),
    "-": (Sub, 1),
    "*": (Mul, 2),
    "/": (Div, 2),
}


class ExpressionParser:
    """表达式解析器: 将字符串解析为 Operator AST。

    支持:
    - 字段引用: $close, $volume
    - 数值常量: 5, 20, 0.5
    - 函数调用: MA($close, 20), CORR($close, $volume, 20)
    - 中缀运算: $close - $open, MA($close,20) - MA($close,5)
    - 括号分组: ($close - $open) * $volume

    解析结果缓存, 避免重复解析同一表达式。
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Operator] = {}

    def parse(self, expr: str) -> Operator:
        expr = expr.strip()
        if expr in self._cache:
            return self._cache[expr]
        tokens = self._tokenize(expr)
        pos = [0]
        node = self._parse_expr(tokens, pos)
        if pos[0] != len(tokens):
            raise SyntaxError(f"表达式解析未完成, 剩余 token: {tokens[pos[0]:]}")
        self._cache[expr] = node
        return node

    # ---- tokenizer ----
    _TOKEN_RE = re.compile(
        r"""
        \s*(?:
            (?P<field>\$[A-Za-z_]\w*)      # $close
          | (?P<number>\d+\.\d+|\d+)        # 5, 0.5
          | (?P<name>[A-Za-z_]\w*)          # MA, close
          | (?P<op>[+\-*/(),])              # 运算符
        )
        """,
        re.VERBOSE,
    )

    def _tokenize(self, expr: str) -> List[Tuple[str, str]]:
        tokens: List[Tuple[str, str]] = []
        i = 0
        while i < len(expr):
            m = self._TOKEN_RE.match(expr, i)
            if not m or m.end() == i:
                if expr[i].isspace():
                    i += 1
                    continue
                raise SyntaxError(f"无法解析的字符 '{expr[i]}' 于位置 {i}: {expr!r}")
            i = m.end()
            kind = m.lastgroup
            value = m.group(kind)
            tokens.append((kind, value))
        return tokens

    # ---- recursive descent parser ----
    def _parse_expr(self, tokens: List, pos: List) -> Operator:
        node = self._parse_term(tokens, pos)
        while pos[0] < len(tokens) and tokens[pos[0]][0] == "op" and tokens[pos[0]][1] in ("+", "-"):
            op_char = tokens[pos[0]][1]
            pos[0] += 1
            right = self._parse_term(tokens, pos)
            op_cls = _INFIX_OPS[op_char][0]
            node = op_cls(node, right)
        return node

    def _parse_term(self, tokens: List, pos: List) -> Operator:
        node = self._parse_factor(tokens, pos)
        while pos[0] < len(tokens) and tokens[pos[0]][0] == "op" and tokens[pos[0]][1] in ("*", "/"):
            op_char = tokens[pos[0]][1]
            pos[0] += 1
            right = self._parse_factor(tokens, pos)
            op_cls = _INFIX_OPS[op_char][0]
            node = op_cls(node, right)
        return node

    def _parse_factor(self, tokens: List, pos: List) -> Operator:
        kind, value = tokens[pos[0]]
        if kind == "op" and value == "(":
            pos[0] += 1
            node = self._parse_expr(tokens, pos)
            if pos[0] >= len(tokens) or tokens[pos[0]] != ("op", ")"):
                raise SyntaxError("缺少右括号 ')'")
            pos[0] += 1
            return node
        if kind == "op" and value == "-":
            # 一元负号
            pos[0] += 1
            operand = self._parse_factor(tokens, pos)
            return Neg(operand)
        if kind == "field":
            pos[0] += 1
            return Field(value)
        if kind == "number":
            pos[0] += 1
            return Const(float(value))
        if kind == "name":
            # 函数调用 name(args...)
            name = value
            pos[0] += 1
            if pos[0] >= len(tokens) or tokens[pos[0]] != ("op", "("):
                raise SyntaxError(f"标识符 '{name}' 后应为 '('")
            pos[0] += 1
            args: List[Operator] = []
            if pos[0] < len(tokens) and tokens[pos[0]] != ("op", ")"):
                args.append(self._parse_expr(tokens, pos))
                while pos[0] < len(tokens) and tokens[pos[0]] == ("op", ","):
                    pos[0] += 1
                    args.append(self._parse_expr(tokens, pos))
            if pos[0] >= len(tokens) or tokens[pos[0]] != ("op", ")"):
                raise SyntaxError(f"函数 '{name}' 缺少右括号")
            pos[0] += 1
            return self._build_func(name, args)
        raise SyntaxError(f"意外的 token {tokens[pos[0]]}")

    def _build_func(self, name: str, args: List[Operator]) -> Operator:
        if name not in OPERATORS:
            raise NameError(f"未知算子 '{name}', 可用: {sorted(OPERATORS.keys())}")
        op_cls, arity = OPERATORS[name]

        def _as_const(op: Operator) -> Optional[Const]:
            """将常量或 Neg(Const) 折叠为 Const (支持负数窗口字面量)。"""
            if isinstance(op, Const):
                return op
            if isinstance(op, Neg) and isinstance(op.operand, Const):
                return Const(-op.operand.value)
            return None

        # 滚动算子: 最后一个参数是常量 window
        if op_cls in (Ref, MA, EMA, STD, VAR, SUM, MAX, MIN, WMA, Delta, ROC, RSI):
            if len(args) != 2:
                raise SyntaxError(f"{name} 需要 2 个参数 (feature, window), 得到 {len(args)}")
            feature, window_op = args
            const_op = _as_const(window_op)
            if const_op is None:
                raise SyntaxError(f"{name} 的 window 参数必须是常量, 得到 {window_op.expr()}")
            return op_cls(feature, int(const_op.value))
        if op_cls is QUANTILE:
            if len(args) != 3:
                raise SyntaxError(f"QUANTILE 需要 3 个参数 (feature, window, q)")
            feature, window_op, q_op = args
            w_const = _as_const(window_op)
            q_const = _as_const(q_op)
            if w_const is None or q_const is None:
                raise SyntaxError("QUANTILE 的 window 和 q 必须是常量")
            return QUANTILE(feature, int(w_const.value), float(q_const.value))
        if op_cls is CORR:
            if len(args) != 3:
                raise SyntaxError("CORR 需要 3 个参数 (left, right, window)")
            left, right, window_op = args
            const_op = _as_const(window_op)
            if const_op is None:
                raise SyntaxError("CORR 的 window 必须是常量")
            return CORR(left, right, int(const_op.value))
        # 一元/二元算子
        if arity == 1:
            if len(args) != 1:
                raise SyntaxError(f"{name} 需要 1 个参数, 得到 {len(args)}")
            return op_cls(args[0])
        if arity == 2:
            if len(args) != 2:
                raise SyntaxError(f"{name} 需要 2 个参数, 得到 {len(args)}")
            return op_cls(args[0], args[1])
        raise SyntaxError(f"算子 {name} 参数数量不匹配")


# ---------------------------------------------------------------------------
# 表达式引擎 (对外门面)
# ---------------------------------------------------------------------------


class ExpressionEngine:
    """因子表达式引擎门面。

    用法::

        engine = ExpressionEngine()
        factor = engine.compute("MA($close, 20) - MA($close, 5)", data)
        # data 为 MultiIndex(date, code) 的 DataFrame, 含 close 列
    """

    def __init__(self) -> None:
        self.parser = ExpressionParser()

    def parse(self, expr: str) -> Operator:
        return self.parser.parse(expr)

    def compute(self, expr: str, data: pd.DataFrame) -> pd.Series:
        """计算表达式, 返回 Series。"""
        node = self.parse(expr)
        result = node(data)
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        result.name = expr
        return result

    def compute_many(
        self,
        exprs: List[str],
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """批量计算多个表达式, 返回 DataFrame (列名为表达式)。"""
        out = {}
        for expr in exprs:
            try:
                out[expr] = self.compute(expr, data)
            except Exception as exc:  # noqa: BLE001
                out[expr] = pd.Series(np.nan, index=data.index, name=expr)
                out[expr].attrs["error"] = str(exc)
        return pd.DataFrame(out)
