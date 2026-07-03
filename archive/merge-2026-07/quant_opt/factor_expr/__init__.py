"""
因子表达式引擎（Alpha101 风格）
==============================

借鉴来源：
- AkQuant 0.2.43 (2026-06-09 发布) 的因子表达式引擎
  `Rank(Ts_Mean(Close, 5))` 风格，支持 Alpha101 风格公式。
- Qlib 的 Alpha158 因子库（多算子 + 截面/时序函数分层）。
- AlphaBench (ICLR 2026) 的 FFO (Formulaic Factor Operator) 执行规范。

设计目标：
- 提供可解析的"公式 → 因子值"映射
- 同时支持 横截面 (cs_) 与 时序 (ts_) 算子
- 使用 pandas/numpy 即可运行（不依赖 polars/Rust 扩展）

支持的算子：
- 时序：ts_mean/ts_sum/ts_std/ts_max/ts_min/ts_rank/ts_delta/ts_delay/ts_argmax/ts_argmin
- 横截面：cs_rank/cs_mean/cs_std/cs_scale/cs_zscore
- 基础：log/abs/sign/rank/delta/mean/sum/std
- 信号：if_else（条件表达式）

公式示例：
  Rank(Delta(Close, 5))                         # 5日 momentum 排名
  -1 * CsRank(Ts_Mean(Volume, 20))              # 量能反转
  Ts_Mean(Close, 10) / Ts_Mean(Close, 60)       # 均线比
  Sign(Delta(Close, 1)) * Ts_Std(Returns, 20)   # 波动率调整方向
"""
from __future__ import annotations

import ast
import math
import operator as op
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# 基础算子
# ──────────────────────────────────────────────────────────────────

_BIN_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}

_UNARY_OPS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


# ──────────────────────────────────────────────────────────────────
# 时序 (time-series) 算子
# ──────────────────────────────────────────────────────────────────

def _ts_op_factory(window: int, reducer: str) -> Callable:
    """构造时序算子"""
    window = int(window)
    def fn(series: pd.Series) -> pd.Series:
        grouped = series.groupby(level="code") if "code" in series.index.names else series
        if reducer == "mean":
            return grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
        if reducer == "sum":
            return grouped.transform(lambda x: x.rolling(window, min_periods=1).sum())
        if reducer == "std":
            return grouped.transform(lambda x: x.rolling(window, min_periods=2).std())
        if reducer == "max":
            return grouped.transform(lambda x: x.rolling(window, min_periods=1).max())
        if reducer == "min":
            return grouped.transform(lambda x: x.rolling(window, min_periods=1).min())
        if reducer == "rank":
            return grouped.transform(lambda x: x.rolling(window, min_periods=1).rank(pct=True))
        raise ValueError(f"未知 reducer: {reducer}")
    return fn


def _ts_delta(series: pd.Series, window: int) -> pd.Series:
    window = int(window)
    return series.groupby(level="code").diff(window) if "code" in series.index.names else series.diff(window)


def _ts_delay(series: pd.Series, window: int) -> pd.Series:
    window = int(window)
    return series.groupby(level="code").shift(window) if "code" in series.index.names else series.shift(window)


def _ts_argmax(series: pd.Series, window: int) -> pd.Series:
    window = int(window)
    return series.groupby(level="code").transform(
        lambda x: x.rolling(window, min_periods=1).apply(np.argmax, raw=True)
    ) if "code" in series.index.names else series.rolling(window, min_periods=1).apply(np.argmax, raw=True)


def _ts_argmin(series: pd.Series, window: int) -> pd.Series:
    window = int(window)
    return series.groupby(level="code").transform(
        lambda x: x.rolling(window, min_periods=1).apply(np.argmin, raw=True)
    ) if "code" in series.index.names else series.rolling(window, min_periods=1).apply(np.argmin, raw=True)


# ──────────────────────────────────────────────────────────────────
# 横截面 (cross-section) 算子
# ──────────────────────────────────────────────────────────────────

def _cs_rank(series: pd.Series) -> pd.Series:
    if "date" in series.index.names:
        return series.groupby(level="date").rank(pct=True)
    return series.rank(pct=True)


def _cs_zscore(series: pd.Series) -> pd.Series:
    if "date" in series.index.names:
        g = series.groupby(level="date")
        return (series - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return (series - series.mean()) / (series.std() if series.std() else np.nan)


def _cs_scale(series: pd.Series) -> pd.Series:
    if "date" in series.index.names:
        return series.groupby(level="date").transform(lambda x: x / x.abs().sum() if x.abs().sum() else x)
    s = series.abs().sum()
    return series / s if s else series


# ──────────────────────────────────────────────────────────────────
# 基础数学
# ──────────────────────────────────────────────────────────────────

def _log(x):
    if isinstance(x, pd.Series):
        return np.log(x.replace(0, np.nan))
    return math.log(x) if x > 0 else float("nan")


def _abs(x):
    return x.abs() if isinstance(x, pd.Series) else abs(x)


def _sign(x):
    if isinstance(x, pd.Series):
        return np.sign(x)
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _if_else(cond, a, b):
    if isinstance(cond, pd.Series):
        return np.where(cond.fillna(False), a, b)
    return a if cond else b


# ──────────────────────────────────────────────────────────────────
# 解析器
# ──────────────────────────────────────────────────────────────────

class FactorExprError(Exception):
    """因子表达式错误"""


@dataclass
class FactorField:
    """字段引用：长格式下指向 (code,date) 的某列"""
    name: str


# 算子注册表
_OPS: Dict[str, Callable] = {
    # 时序
    "Ts_Mean": lambda s, w: _ts_op_factory(int(w), "mean")(s),
    "Ts_Sum":  lambda s, w: _ts_op_factory(int(w), "sum")(s),
    "Ts_Std":  lambda s, w: _ts_op_factory(int(w), "std")(s),
    "Ts_Max":  lambda s, w: _ts_op_factory(int(w), "max")(s),
    "Ts_Min":  lambda s, w: _ts_op_factory(int(w), "min")(s),
    "Ts_Rank": lambda s, w: _ts_op_factory(int(w), "rank")(s),
    "Ts_Delta": _ts_delta,
    "Ts_Delay": _ts_delay,
    "Ts_Argmax": _ts_argmax,
    "Ts_Argmin": _ts_argmin,
    # 横截面
    "Rank":     _cs_rank,
    "CsRank":   _cs_rank,
    "CsZscore": _cs_zscore,
    "CsScale":  _cs_scale,
    # 基础
    "Log":     _log,
    "Abs":     _abs,
    "Sign":    _sign,
    "IfElse":  _if_else,
}


class _FactorParser:
    """
    将 Alpha101 风格公式解析为可在 long-format DataFrame 上执行的算子链。
    表达式 AST 节点：
        Num / Constant / Name / BinOp / UnaryOp / Call / Compare
    Name 节点优先当作 字段名 (大小写不敏感地匹配列)。
    """

    def __init__(self, df: pd.DataFrame, extra: Optional[Dict[str, Any]] = None):
        self.df = df
        self.extra = extra or {}

    def get(self, name: str) -> pd.Series:
        """获取字段值"""
        # 1) 列名匹配
        if name in self.df.columns:
            return self.df[name]
        # 2) 不区分大小写匹配
        for c in self.df.columns:
            if c.lower() == name.lower():
                return self.df[c]
        # 3) 特殊派生字段
        if name.lower() in ("returns", "ret", "ret_1d"):
            return self.df.groupby("code")["close"].pct_change() if "code" in self.df.columns else self.df["close"].pct_change()
        # 4) 外部注入
        if name in self.extra:
            return self.extra[name]
        raise FactorExprError(f"未找到字段: {name}")

    def eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self.eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise FactorExprError(f"不支持的字面量: {node.value!r}")
        if isinstance(node, ast.Num):  # 兼容 Python<3.8
            return float(node.n)
        if isinstance(node, ast.Name):
            return self.get(node.id)
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPS[type(node.op)](self.eval(node.operand))
        if isinstance(node, ast.BinOp):
            l, r = self.eval(node.left), self.eval(node.right)
            return _BIN_OPS[type(node.op)](l, r)
        if isinstance(node, ast.BoolOp):
            values = [self.eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                result = values[0]
                for v in values[1:]:
                    result = result & v
                return result
            if isinstance(node.op, ast.Or):
                result = values[0]
                for v in values[1:]:
                    result = result | v
                return result
            raise FactorExprError(f"不支持的布尔操作: {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            left = self.eval(node.left)
            result = None
            for op_node, comp in zip(node.ops, node.comparators):
                right = self.eval(comp)
                if isinstance(op_node, ast.Gt):
                    part = left > right
                elif isinstance(op_node, ast.GtE):
                    part = left >= right
                elif isinstance(op_node, ast.Lt):
                    part = left < right
                elif isinstance(op_node, ast.LtE):
                    part = left <= right
                elif isinstance(op_node, ast.Eq):
                    part = left == right
                elif isinstance(op_node, ast.NotEq):
                    part = left != right
                else:
                    raise FactorExprError(f"不支持的比较操作: {type(op_node).__name__}")
                result = part if result is None else (result & part)
                left = right
            return result
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if fname is None or fname not in _OPS:
                raise FactorExprError(f"未知算子: {ast.dump(node.func)}")
            args = [self.eval(a) for a in node.args]
            try:
                return _OPS[fname](*args)
            except TypeError as e:
                raise FactorExprError(f"算子 {fname} 参数错误: {e}") from e
        raise FactorExprError(f"不支持的语法节点: {type(node).__name__}")


def compile_factor(formula: str, df: pd.DataFrame, extra: Optional[Dict[str, Any]] = None) -> pd.Series:
    """
    编译并执行一条 Alpha101 风格公式。

    参数:
        formula: 公式字符串
        df:      long-format DataFrame，需包含 code, date 列
        extra:   额外注入的字段名->Series 映射

    返回:
        pd.Series（索引与 df 对齐）
    """
    if not isinstance(df.index, pd.MultiIndex):
        if "code" in df.columns and "date" in df.columns:
            df = df.set_index(["code", "date"]).sort_index()
        else:
            raise FactorExprError("df 必须包含 code 和 date 列，或已是 (code,date) MultiIndex")

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise FactorExprError(f"公式语法错误: {e}") from e

    parser = _FactorParser(df, extra=extra)
    result = parser.eval(tree)
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=df.index)
    return result


# ──────────────────────────────────────────────────────────────────
# 预置因子库（参考 Qlib Alpha101/158 与社区常见信号）
# ──────────────────────────────────────────────────────────────────

PRESET_FACTORS: Dict[str, str] = {
    "momentum_5d":   "Close / Ts_Delay(Close, 5) - 1",
    "momentum_20d":  "Close / Ts_Delay(Close, 20) - 1",
    "reversal_5d":   "-(Close / Ts_Delay(Close, 5) - 1)",
    "volatility_20d": "Ts_Std(Returns, 20)",
    "turnover_20d":  "Ts_Mean(Volume, 20)",
    "amount_ma5":    "Ts_Mean(Amount, 5)",
    "hl_range":      "(High - Low) / Close",
    "close_ma_ratio": "Close / Ts_Mean(Close, 20)",
    "rsi_14":        "Ts_Rank(Sign(Returns) + Sign(Ts_Delay(Returns, 1)) + Sign(Ts_Delay(Returns, 2)), 14)",
    "rank_mom_5d":   "Rank(Close / Ts_Delay(Close, 5) - 1)",
}


def compute_preset(name: str, df: pd.DataFrame) -> pd.Series:
    """计算预置因子"""
    if name not in PRESET_FACTORS:
        raise FactorExprError(f"未知预置因子: {name}，可选: {list(PRESET_FACTORS.keys())}")
    return compile_factor(PRESET_FACTORS[name], df)
