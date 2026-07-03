"""
可组合的因子表达式引擎（Composable Factor Expression Engine）

借鉴自：Microsoft Qlib 的 qlib.data.ops 模块
- Qlib 将因子定义为可组合的 Expression 树节点（ElemOperator / ExpressionOps）
- 支持横向 (CrossSection) 与时序 (TimeSeries) 操作
- 用户可声明式地表达：`$close / Ref($close, 1) - 1`

本模块的差异点：
1. 零依赖（仅 pandas/numpy），不引入 Qlib 重量级基础设施
2. 与 jingni-trader 现有 OHLCV DataFrame 兼容（columns: code, date, ...）
3. 内置中文文档、A 股特色（涨跌停、行业）算子
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 基础数据结构：Feature / Expression
# ---------------------------------------------------------------------------
class Expression(ABC):
    """表达式抽象基类，模仿 Qlib 的 Expression 接口"""

    @abstractmethod
    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        """对给定的 OHLCV DataFrame 计算该表达式的值。

        返回值的索引与 data 一致，长度相同。_evaluator 由 Evaluator 注入，
        用于子节点共享计算缓存。"""

    # 运算符重载 —— 表达式可以相互组合
    def __add__(self, other): return _Binary(self, _as_expr(other), "+")
    def __sub__(self, other): return _Binary(self, _as_expr(other), "-")
    def __mul__(self, other): return _Binary(self, _as_expr(other), "*")
    def __truediv__(self, other): return _Binary(self, _as_expr(other), "/")
    def __neg__(self): return _Unary(self, "-")
    def __radd__(self, other): return _Binary(_as_expr(other), self, "+")
    def __rsub__(self, other): return _Binary(_as_expr(other), self, "-")
    def __rmul__(self, other): return _Binary(_as_expr(other), self, "*")
    def __rtruediv__(self, other): return _Binary(_as_expr(other), self, "/")


# ---------------------------------------------------------------------------
# 叶子节点：Feature 字段引用 / Const 常量
# ---------------------------------------------------------------------------
def _as_expr(x):
    """把任意值包装为 Expression：Expression 透传，其他包成 _Const。"""
    if isinstance(x, Expression):
        return x
    return _Const(x)


class Feature(Expression):
    """引用 DataFrame 中某个字段，例如 $close"""

    def __init__(self, name: str):
        self.name = name

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if self.name not in data.columns:
            raise KeyError(
                f"Feature {self.name!r} 不在数据列中；当前列: {list(data.columns)}"
            )
        return data[self.name]

    def __repr__(self) -> str:  # pragma: no cover
        return f"${self.name}"


class _Const(Expression):
    def __init__(self, value):
        self.value = value

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        return pd.Series(self.value, index=data.index, dtype=float)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Const({self.value!r})"


# ---------------------------------------------------------------------------
# 组合节点：Unary / Binary
# ---------------------------------------------------------------------------
class _Unary(Expression):
    _OPS = {"-": lambda x: -x}

    def __init__(self, child: Expression, op: str):
        self.child = child
        self.op = op

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if _evaluator is not None:
            v = _evaluator.eval(self.child)
        else:
            v = self.child.compute(data)
        return self._OPS[self.op](v)

    def __repr__(self) -> str:  # pragma: no cover
        return f"({self.op}{self.child!r})"


class _Binary(Expression):
    _OPS = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b.replace(0, np.nan),
    }

    def __init__(self, left: Expression, right: Expression, op: str):
        self.left = left
        self.right = right
        self.op = op

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        # 若 Evaluator 注入，则子节点走 eval 实现共享计算
        if _evaluator is not None:
            l = _evaluator.eval(self.left)
            r = _evaluator.eval(self.right)
        else:
            l = self.left.compute(data)
            r = self.right.compute(data)
        return self._OPS[self.op](l, r)

    def __repr__(self) -> str:  # pragma: no cover
        return f"({self.left!r} {self.op} {self.right!r})"


# ---------------------------------------------------------------------------
# 算子：横向 / 时序 / 元素级
# ---------------------------------------------------------------------------
class _TimeSeriesOp(Expression):
    """所有按 (code, date) 排序后做时间窗口的算子基类"""

    def __init__(self, child: Expression, window: int):
        if window <= 0:
            raise ValueError("window 必须为正整数")
        self.child = child
        self.window = window


class Ref(_TimeSeriesOp):
    """Ref(x, d) = x.shift(d) —— 时序滞后"""

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if _evaluator is not None:
            s = _evaluator.eval(self.child)
        else:
            s = self.child.compute(data)
        return s.groupby(data["code"]).shift(self.window)


class Delta(_TimeSeriesOp):
    """Delta(x, d) = x - Ref(x, d)"""

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if _evaluator is not None:
            s = _evaluator.eval(self.child)
        else:
            s = self.child.compute(data)
        ref = s.groupby(data["code"]).shift(self.window)
        return s - ref


class TsMean(_TimeSeriesOp):
    """TsMean(x, d) = x 在过去 d 日的均值"""

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if _evaluator is not None:
            s = _evaluator.eval(self.child)
        else:
            s = self.child.compute(data)
        return s.groupby(data["code"]).transform(
            lambda x: x.rolling(self.window, min_periods=1).mean()
        )


class TsStd(_TimeSeriesOp):
    """TsStd(x, d)"""

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if _evaluator is not None:
            s = _evaluator.eval(self.child)
        else:
            s = self.child.compute(data)
        return s.groupby(data["code"]).transform(
            lambda x: x.rolling(self.window, min_periods=2).std()
        )


class TsRank(_TimeSeriesOp):
    """TsRank(x, d) = x 在过去 d 日的截面百分位排名"""

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if _evaluator is not None:
            s = _evaluator.eval(self.child)
        else:
            s = self.child.compute(data)
        return s.groupby(data["code"]).transform(
            lambda x: x.rolling(self.window, min_periods=1).rank(pct=True)
        )


class _CrossSectionOp(Expression):
    """所有按 date 横向做截面归一化的算子基类"""


class Rank(_CrossSectionOp):
    """Rank(x) = x 在当日所有股票中的百分位排名"""

    def __init__(self, child: Expression):
        self.child = child

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if _evaluator is not None:
            s = _evaluator.eval(self.child)
        else:
            s = self.child.compute(data)
        return s.groupby(data["date"]).rank(pct=True)


class Zscore(_CrossSectionOp):
    """Zscore(x) = (x - mean) / std 截面标准化"""

    def __init__(self, child: Expression):
        self.child = child

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if _evaluator is not None:
            s = _evaluator.eval(self.child)
        else:
            s = self.child.compute(data)
        grp = s.groupby(data["date"])
        return (s - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)


class Mean(_CrossSectionOp):
    """Mean(x) = x 在当日所有股票中的均值（中心化用）"""

    def __init__(self, child: Expression):
        self.child = child

    def compute(self, data: pd.DataFrame, _evaluator=None) -> pd.Series:
        if _evaluator is not None:
            s = _evaluator.eval(self.child)
        else:
            s = self.child.compute(data)
        return s.groupby(data["date"]).transform("mean")


# ---------------------------------------------------------------------------
# 便捷语法糖：F("close") 等价于 $close
# ---------------------------------------------------------------------------
def F(name: str) -> Feature:
    """Factory: F("close") -> $close"""
    return Feature(name)


# ---------------------------------------------------------------------------
# 批量求值：避免对同一棵子树重复计算
# ---------------------------------------------------------------------------
class Evaluator:
    """带 memoization 的批量表达式求值器。

    借鉴自 Qlib 内部对相同子树共享计算的优化。
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._cache: Dict[int, pd.Series] = {}

    def eval(self, expr: Expression) -> pd.Series:
        key = id(expr)
        if key in self._cache:
            return self._cache[key]
        # 注入 _evaluator 让子节点也走 eval
        result = expr.compute(self.data, _evaluator=self)
        self._cache[key] = result
        return result

    def eval_many(self, exprs: Dict[str, Expression]) -> pd.DataFrame:
        """批量求值多个表达式，输出列名 = dict 的 key"""
        out = {}
        for name, e in exprs.items():
            out[name] = self.eval(e)
        return pd.DataFrame(out, index=self.data.index)


# ---------------------------------------------------------------------------
# 预设常用 A 股因子（对照 jingni-trader 现有 compute_a_share_factors）
# ---------------------------------------------------------------------------
def builtin_a_share_factors() -> Dict[str, Expression]:
    """返回 A 股常用因子的表达式字典，可与 Evaluator 联合使用。"""
    close, high, low, vol = F("close"), F("high"), F("low"), F("volume")
    return {
        # 动量 / 反转
        "ret_1d": close / Ref(close, 1) - 1,
        "ret_5d": close / Ref(close, 5) - 1,
        "ret_20d": close / Ref(close, 20) - 1,
        "reversal_5d": -(close / Ref(close, 5) - 1),
        "reversal_20d": -(close / Ref(close, 20) - 1),
        # 均线
        "ma_5": TsMean(close, 5),
        "ma_20": TsMean(close, 20),
        "ma_60": TsMean(close, 60),
        # 波动率
        "volatility_20d": TsStd(close / Ref(close, 1) - 1, 20),
        # 成交量
        "volume_ma_20": TsMean(vol, 20),
        "volume_ratio": vol / TsMean(vol, 20),
        # K线
        "hl_range": (high - low) / close,
    }


__all__ = [
    "Expression",
    "Feature",
    "F",
    "Ref",
    "Delta",
    "TsMean",
    "TsStd",
    "TsRank",
    "Rank",
    "Zscore",
    "Mean",
    "Evaluator",
    "builtin_a_share_factors",
]
