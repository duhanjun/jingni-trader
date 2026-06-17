"""
因子表达式引擎（Factor Expression Engine）
===========================================

借鉴来源
--------
- microsoft/qlib 的 Data Layer Expression Engine（Ref, $close, $open 等）
- akquant 的 Polars 因子表达式引擎（WorldQuant Alpha101 风格）

设计目标
--------
为 jingni-trader 提供一个轻量级、可扩展的因子计算 DSL，让因子开发从
"手写 groupby + rolling 循环"升级为"声明式公式"。

为什么不用完整借鉴 qlib？
- qlib 的 Expression Engine 强依赖自家 .bin 数据格式和 DataHandler，
  侵入性过大，不适合在 jingni-trader 这种"保留多数据源、保留 parquet"
  的架构上直接套用。
- akquant 的 Polars 引擎需要 Rust 运行时，依赖较重。

这里采用"轻量 Python + pandas"实现：
- 输入：标准化的 (code, date, OHLCV...) DataFrame
- DSL：使用 qlib/Alpha101 风格的字符串公式，如 "Rank(Mean($close, 20))"
- 输出：与原 DataFrame 对齐的因子列
- 完全离线、无需额外数据格式

表达式语法（首批支持）
--------------------
字段引用：
    $open, $high, $low, $close, $volume, $amount

时间序列算子（按 code 分组）：
    Ref(x, n)        - 滞后 n 期
    Delta(x, n)      - x - Ref(x, n)
    Mean(x, n)       - 滚动均值
    Std(x, n)        - 滚动标准差
    Sum(x, n)        - 滚动求和
    Max(x, n)        - 滚动最大
    Min(x, n)        - 滚动最小
    TsRank(x, n)     - 时序百分位排名
    Ema(x, n)        - 指数移动平均
    DecayLinear(x, n) - 线性衰减加权

截面算子（按 date 分组）：
    Rank(x)          - 截面百分位排名
    Quantile(x, q)   - 截面分位数
    Mad(x)           - 截面去中位数
    Scale(x)         - 缩放到 abs 之和 = 1

数学/逻辑：
    +, -, *, /, **, %, //
    Abs, Log, Sign, Sqrt
    And, Or, Greater, Less, Equal
    If(cond, a, b)

算子名同时支持大小写：Ref/ref/REF 等价。

示例
----
>>> import pandas as pd
>>> from quant_opt_20260617.factor_expression_engine import FactorEngine
>>> engine = FactorEngine()
>>> df = pd.DataFrame({
...     'code': ['A']*5 + ['B']*5,
...     'date': pd.date_range('2024-01-01', periods=5).tolist()*2,
...     'close': [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
...     'volume': [100, 110, 120, 130, 140, 200, 210, 220, 230, 240],
... })
>>> result = engine.compute(df, [
...     "Mean($close, 3)",
...     "Rank(Delta($close, 1))",
...     "Std(Mean($close, 5), 3)",
... ])
"""
from __future__ import annotations

import ast
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ============================================================
# 上下文（让算子能访问父 DataFrame）
# ============================================================

# Thread-local 存储当前正在求值的 data
# 算子通过 _get_parent_data() 获取
_TLS = threading.local()


def _get_parent_data() -> Optional[pd.DataFrame]:
    return getattr(_TLS, "data", None)


def _set_parent_data(d: Optional[pd.DataFrame]):
    _TLS.data = d


def _group_by_code_or_global(series: pd.Series, op) -> pd.Series:
    """如果当前上下文有 data 且有 code 列，按 code 分组；否则全局"""
    data = _get_parent_data()
    if data is not None and "code" in data.columns:
        # 把 series 重索引回 data 的行（因为 series 可能来自中间计算）
        try:
            # 用临时列附加
            temp = data[["code"]].copy()
            temp["_v"] = series.values
            return temp.groupby("code")["_v"].transform(op)
        except Exception:
            pass
    return op(series)


# ============================================================
# 表达式解析（Tokenize + AST 解析 + 安全求值）
# ============================================================

# 算子注册表：算子名 -> 函数签名
# 函数签名接收 (data: DataFrame, **kwargs) -> Series
TS_OPERATORS: Dict[str, Callable[..., pd.Series]] = {}
CS_OPERATORS: Dict[str, Callable[..., pd.Series]] = {}
MATH_FUNCS: Dict[str, Callable[..., Any]] = {}


def _register_ts(name: str):
    def deco(fn):
        TS_OPERATORS[name.lower()] = fn
        return fn
    return deco


def _register_cs(name: str):
    def deco(fn):
        CS_OPERATORS[name.lower()] = fn
        return fn
    return deco


def _register_math(name: str):
    def deco(fn):
        MATH_FUNCS[name.lower()] = fn
        return fn
    return deco


# ---- 时间序列算子 ----

@_register_ts("ref")
def op_ref(x, n):
    """滞后 n 期"""
    return _group_by_code_or_global(x, lambda s: s.shift(int(n)))


@_register_ts("delta")
def op_delta(x, n):
    return x - op_ref(x, n)


@_register_ts("mean")
def op_mean(x, n):
    n = int(n)
    return _group_by_code_or_global(x, lambda s: s.rolling(n, min_periods=1).mean())


@_register_ts("std")
def op_std(x, n):
    n = int(n)
    return _group_by_code_or_global(x, lambda s: s.rolling(n, min_periods=2).std())


@_register_ts("sum")
def op_sum(x, n):
    n = int(n)
    return _group_by_code_or_global(x, lambda s: s.rolling(n, min_periods=1).sum())


@_register_ts("max")
def op_max(x, n):
    n = int(n)
    return _group_by_code_or_global(x, lambda s: s.rolling(n, min_periods=1).max())


@_register_ts("min")
def op_min(x, n):
    n = int(n)
    return _group_by_code_or_global(x, lambda s: s.rolling(n, min_periods=1).min())


@_register_ts("tsrank")
def op_tsrank(x, n):
    """时序百分位排名（滚动窗口）"""
    n = int(n)
    return _group_by_code_or_global(x, lambda s: s.rolling(n, min_periods=2).rank(pct=True))


@_register_ts("ema")
def op_ema(x, n):
    """指数移动平均（adjust=False，与业界默认一致）"""
    n = int(n)
    return _group_by_code_or_global(x, lambda s: s.ewm(span=n, adjust=False).mean())


@_register_ts("decaylinear")
def op_decaylinear(x, n):
    """线性衰减加权（最新数据权重最大）"""
    n = int(n)
    weights = np.arange(1, n + 1, dtype=float)
    weights = weights / weights.sum()

    def _decay(s):
        if len(s) < n:
            return pd.Series(np.full(len(s), np.nan), index=s.index)
        vals = s.values
        out = np.full_like(vals, np.nan, dtype=float)
        for i in range(n - 1, len(vals)):
            window = vals[i - n + 1: i + 1]
            out[i] = np.dot(window, weights)
        return pd.Series(out, index=s.index)

    return _group_by_code_or_global(x, _decay)


# ---- 截面算子 ----

def _group_by_date_or_global(series: pd.Series, op) -> pd.Series:
    """如果当前上下文有 data 且有 date 列，按 date 分组；否则全局"""
    data = _get_parent_data()
    if data is not None and "date" in data.columns:
        try:
            temp = data[["date"]].copy()
            temp["_v"] = series.values
            return temp.groupby("date")["_v"].transform(op)
        except Exception:
            pass
    return op(series)


@_register_cs("rank")
def op_rank(x):
    """截面百分位排名"""
    return _group_by_date_or_global(x, lambda s: s.rank(pct=True))


@_register_cs("quantile")
def op_quantile(x, q):
    """截面分位数（返回 mask，大于分位数为 1）"""
    q = float(q)
    return _group_by_date_or_global(x, lambda s: (s >= s.quantile(q)).astype(int))


@_register_cs("mad")
def op_mad(x):
    """截面去中位数"""
    return _group_by_date_or_global(x, lambda s: s - s.median())


@_register_cs("scale")
def op_scale(x):
    """缩放到 abs 之和 = 1（类 pypfopt.scale）"""
    total = x.abs().sum()
    if total == 0 or np.isnan(total):
        return x
    return x / total


# ---- 数学/逻辑函数 ----

@_register_math("abs")
def m_abs(x):
    return np.abs(x)


@_register_math("log")
def m_log(x):
    return np.log(x)


@_register_math("sign")
def m_sign(x):
    return np.sign(x)


@_register_math("sqrt")
def m_sqrt(x):
    return np.sqrt(x)


@_register_math("greater")
def m_greater(a, b):
    return (a > b).astype(int)


@_register_math("less")
def m_less(a, b):
    return (a < b).astype(int)


@_register_math("equal")
def m_equal(a, b):
    return (a == b).astype(int)


@_register_math("and")
def m_and(a, b):
    return ((a != 0) & (b != 0)).astype(int)


@_register_math("or")
def m_or(a, b):
    return ((a != 0) | (b != 0)).astype(int)


@_register_math("if")
def m_if(cond, a, b):
    return np.where(cond != 0, a, b)


# ============================================================
# 字段名映射（$close -> close）
# ============================================================

FIELD_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")


def _resolve_field(token: str, data: pd.DataFrame) -> pd.Series:
    """$close -> data['close']"""
    if not token.startswith("$"):
        raise ValueError(f"无法解析字段：{token}")
    name = token[1:].lower()
    # 大小写不敏感
    matches = [c for c in data.columns if c.lower() == name]
    if not matches:
        raise KeyError(f"数据中找不到字段: {name}（可用列: {list(data.columns)}）")
    return data[matches[0]]


# ============================================================
# 表达式求值
# ============================================================

# 我们用受限的 AST 节点类型（安全求值，避免 eval/ast.literal_eval 的不安全）
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.Call, ast.Name, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Load,
)


def _safe_eval(expr: str, data: pd.DataFrame) -> Any:
    """受限 AST 求值"""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"表达式中包含不支持的语法: {type(node).__name__}")
    return _eval_node(tree.body, data)


def _eval_node(node, data: pd.DataFrame) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        name = node.id.lower()
        if name in MATH_FUNCS:
            # 名字直接是函数（不带调用） - 不常见，直接报错
            raise ValueError(f"算子 {name} 必须带括号调用")
        raise NameError(f"未定义的名称: {node.id}")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, data)
        right = _eval_node(node.right, data)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, data)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Not):
            return (~operand.astype(bool)).astype(int)
    if isinstance(node, ast.Compare):
        # 只支持单比较
        if len(node.ops) > 1:
            raise ValueError("不支持链式比较")
        left = _eval_node(node.left, data)
        right = _eval_node(node.comparators[0], data)
        op = node.ops[0]
        if isinstance(op, ast.Gt):
            return (left > right).astype(int)
        if isinstance(op, ast.Lt):
            return (left < right).astype(int)
        if isinstance(op, ast.GtE):
            return (left >= right).astype(int)
        if isinstance(op, ast.LtE):
            return (left <= right).astype(int)
        if isinstance(op, ast.Eq):
            return (left == right).astype(int)
        if isinstance(op, ast.NotEq):
            return (left != right).astype(int)
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, data) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for v in values[1:]:
                result = ((result != 0) & (v != 0)).astype(int)
            return result
        if isinstance(node.op, ast.Or):
            result = values[0]
            for v in values[1:]:
                result = ((result != 0) | (v != 0)).astype(int)
            return result
    if isinstance(node, ast.Call):
        fname = node.func.id.lower()
        args = [_eval_node(a, data) for a in node.args]
        # 优先尝试 TS / CS / MATH 三类
        if fname in TS_OPERATORS:
            return TS_OPERATORS[fname](*args)
        if fname in CS_OPERATORS:
            return CS_OPERATORS[fname](*args)
        if fname in MATH_FUNCS:
            return MATH_FUNCS[fname](*args)
        raise NameError(f"未知算子: {fname}")
    raise ValueError(f"无法求值的节点: {type(node).__name__}")


def _expand_field_refs(expr: str) -> str:
    """把 $close 替换为 Python 标识符（避免被解析为 Name 节点）"""
    return FIELD_RE.sub(lambda m: f"_F_{m.group(1).upper()}_", expr)


def _wrap_field_lookups(expr: str, data: pd.DataFrame) -> Dict[str, pd.Series]:
    """提取所有 $field 引用，预先查表；返回 dict 作为闭包变量"""
    refs = FIELD_RE.findall(expr)
    return {f"_F_{r.upper()}_": _resolve_field(f"${r}", data) for r in set(refs)}


def _eval_with_lookups(expr: str, data: pd.DataFrame) -> Any:
    """带字段预解析的求值：把 $field 替换为 _F_FIELD_ 标识符，
    在求值时通过 globals 解析到预取的 Series。"""
    # 提取所有 $field
    field_map = _wrap_field_lookups(expr, data)
    # 替换为 Python 标识符
    expanded = _expand_field_refs(expr)

    # 把字段映射注入到 eval 命名空间
    g = dict(field_map)

    # 算子支持大小写不敏感：同时注册小写原名和大写别名
    def _wrap_ts(name):
        def fn(*args):
            return TS_OPERATORS[name](*args)
        return fn
    def _wrap_cs(name):
        def fn(*args):
            return CS_OPERATORS[name](*args)
        return fn

    for n in TS_OPERATORS:
        g[n] = _wrap_ts(n)
        g[n.upper()] = _wrap_ts(n)
        g[n.capitalize()] = _wrap_ts(n)
    for n in CS_OPERATORS:
        g[n] = _wrap_cs(n)
        g[n.upper()] = _wrap_cs(n)
        g[n.capitalize()] = _wrap_cs(n)
    for n in MATH_FUNCS:
        g[n] = MATH_FUNCS[n]
        g[n.upper()] = MATH_FUNCS[n]
        g[n.capitalize()] = MATH_FUNCS[n]

    # 白名单 AST 校验
    tree = ast.parse(expanded, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"表达式中包含不支持的语法: {type(node).__name__}")
    return eval(compile(tree, "<factor>", "eval"), {"__builtins__": {}}, g)


# ============================================================
# 引擎主类
# ============================================================

@dataclass
class FactorDef:
    name: str
    expr: str


class FactorEngine:
    """因子表达式引擎"""

    def __init__(self, code_col: str = "code", date_col: str = "date"):
        self.code_col = code_col
        self.date_col = date_col

    def compute(
        self,
        data: pd.DataFrame,
        formulas: Union[str, List[str], List[FactorDef]],
    ) -> pd.DataFrame:
        """
        计算一个或多个因子

        参数
        ----
        data: 标准化的 (code, date, OHLCV...) DataFrame
        formulas: 单个公式 / 公式列表 / FactorDef 列表

        返回
        ----
        原 data 副本，附加因子列
        """
        if isinstance(formulas, str):
            formulas = [FactorDef(name=_auto_name(formulas), expr=formulas)]
        elif isinstance(formulas, FactorDef):
            formulas = [formulas]
        else:
            formulas = [
                f if isinstance(f, FactorDef) else FactorDef(name=_auto_name(f), expr=f)
                for f in formulas
            ]

        result = data.copy()
        # 设置父数据上下文（让算子能按 code / date 分组）
        _set_parent_data(result)
        try:
            for fdef in formulas:
                try:
                    result[fdef.name] = _eval_with_lookups(fdef.expr, result)
                except Exception as e:
                    raise RuntimeError(f"因子 {fdef.name} ({fdef.expr}) 计算失败: {e}")
        finally:
            _set_parent_data(None)
        return result

    def compute_one(self, data: pd.DataFrame, expr: str, name: Optional[str] = None) -> pd.Series:
        """便捷方法：计算单个因子，返回 Series"""
        result = self.compute(data, FactorDef(name=name or _auto_name(expr), expr=expr))
        col = name or _auto_name(expr)
        return result[col]

    def list_operators(self) -> Dict[str, List[str]]:
        """返回所有可用算子（用于帮助提示）"""
        return {
            "time_series": sorted(TS_OPERATORS.keys()),
            "cross_section": sorted(CS_OPERATORS.keys()),
            "math": sorted(MATH_FUNCS.keys()),
        }


def _auto_name(expr: str) -> str:
    """把公式转换为合法的列名"""
    # 移除 $ 符号，把括号 / 逗号 / 空格 替换为下划线
    name = re.sub(r"[\s(),]+", "_", expr.replace("$", ""))
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    return name.strip("_")[:60] or "factor"
