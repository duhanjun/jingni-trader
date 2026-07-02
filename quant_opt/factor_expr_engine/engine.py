"""
因子表达式引擎核心实现
=====================

实现了一套轻量级、AST-白名单的因子表达式求值器。
设计上参考 Qlib 的 ``qlib.data.ops``，但去除了依赖、仅保留核心算子。

支持的语法
----------
字段引用:    $open $high $low $close $volume $amount $vwap
时序算子:    Ref(x, n)         取 n 期前
            Delta(x, n)       x - Ref(x, n)
            Mean(x, n)        过去 n 期均值
            Std(x, n)         过去 n 期标准差
            Sum(x, n)         过去 n 期累加
            TsRank(x, n)      过去 n 期时序百分位排名
截面算子:    Rank(x)           截面百分位排名
            Scale(x)          截面缩放到 sum=1
一元运算:    -x  Abs(x)  Log(x+1)  Sign(x)
二元运算:    + - * /
括号:        ( )

表达式通过 ast 解析，所有节点必须在白名单中，杜绝任意代码执行。
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 算子基类
# ---------------------------------------------------------------------------


@dataclass
class _BaseOp:
    name: str
    func: Callable
    doc: str = ""


class TimeOp(_BaseOp):
    """时序算子，函数签名 ``func(series_by_code: pd.Series, n: int) -> pd.Series``"""


class CrossOp(_BaseOp):
    """截面算子，函数签名 ``func(panel: pd.DataFrame) -> pd.DataFrame``"""


class FactorField(_BaseOp):
    """行情字段算子，``func(panel) -> pd.DataFrame`` 返回对应列"""


# ---------------------------------------------------------------------------
# 时序算子实现
# ---------------------------------------------------------------------------


def _op_ref(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    """Ref(x, n): 取 n 周期前的值"""
    return panel.groupby(level="code").shift(n)


def _op_delta(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    """Delta(x, n): x - Ref(x, n)"""
    return panel - panel.groupby(level="code").shift(n)


def _op_mean(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    """Mean(x, n): 过去 n 期均值"""
    return panel.groupby(level="code").transform(
        lambda s: s.rolling(n, min_periods=max(2, n // 4)).mean()
    )


def _op_std(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    """Std(x, n): 过去 n 期标准差"""
    return panel.groupby(level="code").transform(
        lambda s: s.rolling(n, min_periods=max(2, n // 4)).std()
    )


def _op_sum(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sum(x, n): 过去 n 期累加"""
    return panel.groupby(level="code").transform(
        lambda s: s.rolling(n, min_periods=max(2, n // 4)).sum()
    )


def _op_ts_rank(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    """TsRank(x, n): 当前值在过去 n 期中的百分位排名"""
    return panel.groupby(level="code").transform(
        lambda s: s.rolling(n, min_periods=max(2, n // 4)).rank(pct=True)
    )


# ---------------------------------------------------------------------------
# 截面算子实现
# ---------------------------------------------------------------------------


def _op_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank(x): 截面百分位排名"""
    return panel.groupby(level="date").rank(pct=True)


def _op_scale(panel: pd.DataFrame) -> pd.DataFrame:
    """Scale(x): 截面缩放到 sum(|x|)=1 (不强制正值, 方便后续加权)"""
    abs_sum = panel.abs().groupby(level="date").sum()
    return panel / abs_sum.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 一元函数
# ---------------------------------------------------------------------------


def _fn_abs(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.abs()


def _fn_log1p(panel: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(panel)


def _fn_sign(panel: pd.DataFrame) -> pd.DataFrame:
    return np.sign(panel)


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class FactorExprEngine:
    """因子表达式求值引擎

    输入数据为 ``pd.DataFrame``，必须包含 ``code``、``date`` 与若干行情列；
    内部统一索引为 ``(date, code)`` MultiIndex。
    """

    # 内部别名: $xxx -> XXX  (因为 $ 不是合法 Python 标识符)
    FIELD_MAP = {
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
        "AMOUNT": "amount",
        "VWAP": "vwap",
    }
    # 用户可写的 $ 别名 -> 内部别名
    USER_FIELD_ALIAS = {
        "$open": "OPEN", "$high": "HIGH", "$low": "LOW", "$close": "CLOSE",
        "$volume": "VOLUME", "$amount": "AMOUNT", "$vwap": "VWAP",
    }

    # 默认算子表: key 与调用名一致
    _DEFAULT_OPS: Dict[str, _BaseOp] = {
        # 时序
        "Ref": TimeOp("Ref", _op_ref, "Ref(x, n)"),
        "Delta": TimeOp("Delta", _op_delta, "Delta(x, n)"),
        "Mean": TimeOp("Mean", _op_mean, "Mean(x, n)"),
        "Std": TimeOp("Std", _op_std, "Std(x, n)"),
        "Sum": TimeOp("Sum", _op_sum, "Sum(x, n)"),
        "TsRank": TimeOp("TsRank", _op_ts_rank, "TsRank(x, n)"),
        # 截面
        "Rank": CrossOp("Rank", _op_rank, "Rank(x)"),
        "Scale": CrossOp("Scale", _op_scale, "Scale(x)"),
        # 一元函数
        "Abs": CrossOp("Abs", _fn_abs, "Abs(x)"),
        "Log1p": CrossOp("Log1p", _fn_log1p, "Log1p(x)"),
        "Sign": CrossOp("Sign", _fn_sign, "Sign(x)"),
    }

    BINARY_OPS: Dict[type, Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    UNARY_OPS: Dict[type, Callable[[Any], Any]] = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def __init__(self, extra_ops: Optional[Dict[str, _BaseOp]] = None) -> None:
        self.ops: Dict[str, _BaseOp] = dict(self._DEFAULT_OPS)
        if extra_ops:
            self.ops.update(extra_ops)

    # -----------------------------------------------------------------
    # 公开 API
    # -----------------------------------------------------------------

    def compute(
        self,
        data: pd.DataFrame,
        expr: str,
        name: str = "factor",
    ) -> pd.DataFrame:
        """对单条表达式求值，返回 ``[code, date, name]`` 的 DataFrame"""
        if "code" not in data.columns or "date" not in data.columns:
            raise ValueError("data must contain 'code' and 'date' columns")
        panel = self._to_panel(data)
        value = self._eval(self._parse(expr), panel)
        if isinstance(value, pd.DataFrame):
            if value.shape[1] != 1:
                value = value.iloc[:, :1]
            value = value.iloc[:, 0]
        if not isinstance(value, pd.Series):
            value = pd.Series(value, index=panel.index)
        out = value.reset_index()
        out.columns = ["date", "code", name]
        return out[["code", "date", name]]

    def compute_batch(
        self,
        data: pd.DataFrame,
        expressions: Dict[str, str],
    ) -> pd.DataFrame:
        """批量计算多个因子并合并

        parameters
        ----------
        data:
            原始行情
        expressions:
            {列名: 表达式} 字典

        returns
        -------
        ``pd.DataFrame`` 包含 code/date 和所有因子列
        """
        if not expressions:
            return data[["code", "date"]].copy()
        panel = self._to_panel(data)
        frames = [data[["code", "date"]].copy()]
        for col, expr in expressions.items():
            value = self._eval(self._parse(expr), panel)
            if isinstance(value, pd.DataFrame):
                if value.shape[1] != 1:
                    value = value.iloc[:, :1]
                value = value.iloc[:, 0]
            if not isinstance(value, pd.Series):
                value = pd.Series(value, index=panel.index)
            series = value.reset_index()
            series.columns = ["date", "code", col]
            frames.append(series[[col]])
        merged = pd.concat(frames, axis=1)
        return merged

    @property
    def supported_operators(self) -> List[str]:
        return sorted(self.ops.keys())

    # -----------------------------------------------------------------
    # 内部: 解析 & 求值
    # -----------------------------------------------------------------

    @staticmethod
    def _parse(expr: str) -> ast.Expression:
        # 把 $field 翻译成合法的 Python 标识符 (FIELD_MAP 中的大写名)
        sanitized = expr
        for user_name, internal in FactorExprEngine.USER_FIELD_ALIAS.items():
            # 使用单词边界, 防止误替换
            import re as _re
            sanitized = _re.sub(
                _re.escape(user_name) + r"\b", internal, sanitized
            )
        try:
            tree = ast.parse(sanitized, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"invalid expression syntax: {e}")
        return tree

    def _to_panel(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "code"]).set_index(["date", "code"])
        return df

    def _eval(self, node: ast.AST, panel: pd.DataFrame):
        """递归求值，节点白名单"""
        if isinstance(node, ast.Expression):
            return self._eval(node.body, panel)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.BINARY_OPS:
                raise ValueError(f"unsupported binary op: {op_type.__name__}")
            left = self._eval(node.left, panel)
            right = self._eval(node.right, panel)
            return self._broadcast_apply(self.BINARY_OPS[op_type], left, right)
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in self.UNARY_OPS:
                raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
            operand = self._eval(node.operand, panel)
            return self.UNARY_OPS[type(node.op)](operand)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only simple function calls are allowed")
            name = node.func.id
            if name not in self.ops:
                raise ValueError(f"unknown operator: {name}")
            op = self.ops[name]
            args = [self._eval(a, panel) for a in node.args]
            return self._dispatch_op(op, args, panel)
        if isinstance(node, ast.Name):
            if node.id in self.FIELD_MAP:
                col = self.FIELD_MAP[node.id]
                if col not in panel.columns:
                    raise ValueError(f"data missing field for {node.id}: {col}")
                # 关键: 强制使用统一的列名 "__x__" 以确保两个 DataFrame 相减时
                # pandas 不会做列对齐而误判为 NaN
                s = panel[col]
                return s.to_frame("__x__")
            raise ValueError(f"unknown identifier: {node.id}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"unsupported constant: {node.value!r}")
        if isinstance(node, ast.Num):  # 兼容老版本 Python
            return node.n
        raise ValueError(f"unsupported AST node: {type(node).__name__}")

    def _dispatch_op(self, op: _BaseOp, args, panel):
        if isinstance(op, TimeOp):
            if len(args) != 2:
                raise ValueError(f"{op.name} expects (x, n)")
            base, n = self._to_series_frame(args[0]), int(args[1])
            return op.func(base, n)
        if isinstance(op, CrossOp):
            if len(args) != 1:
                raise ValueError(f"{op.name} expects 1 argument")
            return op.func(self._to_series_frame(args[0]))
        raise ValueError(f"unsupported op type: {op}")

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _to_series_frame(value) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, pd.Series):
            return value.to_frame(value.name or "x")
        # 标量
        return pd.DataFrame({"x": value})

    @staticmethod
    def _broadcast_apply(func, left, right):
        if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
            return func(left, right)
        if isinstance(left, pd.DataFrame):
            return left.apply(lambda c: func(c, right))
        if isinstance(right, pd.DataFrame):
            return right.apply(lambda c: func(left, c), axis=1)
        return func(left, right)
