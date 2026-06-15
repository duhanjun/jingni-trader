"""
安全因子表达式引擎
====================

**借鉴来源**：
- AKQuant (https://github.com/akfamily/akquant) 的 ``akquant.factor.FactorEngine``：
  字符串公式驱动的因子计算 ``Rank(Ts_Mean(Close, 5))``
- 国内多因子选股系统 ``FactorExpressionEngine`` 的白名单校验思路
- Python AST 解析安全求值器模式（拒绝 ``eval/exec``，基于白名单节点）

**当前 jingni-trader 的痛点**：
- ``factor-engine`` 只能使用硬编码的 A 股因子（``ret_1d``、``reversal_5d`` 等），
  无法让用户在不修改源码的情况下添加自定义因子
- 缺乏字符串层面的因子复用机制，研究人员写新因子要改 Python 源码
- 没有公式语法（类似 Qlib 字段表达式 / Alpha101 公式）

**优化目标**：
- 提供一个 ``SafeExpressionEngine``，使用 ``ast`` 模块做语法+节点双重白名单校验
- 拒绝 ``__import__``、``getattr``、``compile``、``exec`` 等危险节点
- 支持 Alpha101 风格算子：``Rank``、``Ts_Mean``、``Ts_Std``、``Delay``、``Delta``、
  ``Correlation``、``StdDev``、``If``、``Sign``、``Abs`` 等
- 支持在 ``DataFrame`` 上做时间序列和横截面计算，输出新因子列
- 零依赖（除 pandas/numpy），可作为 jingni-trader 可选插件

**安全保证**：
1. 仅允许 ``ast`` 表达式模式（``mode='eval'``），无法写语句
2. 白名单 ``SAFE_NODES``：``Expression/BinOp/UnaryOp/Name/Constant/Compare/Call/...``
3. ``Call`` 节点函数名必须在 ``SAFE_FUNCTIONS`` 中
4. ``Name`` 节点变量名必须以 ``$`` 前缀或为预注册字段
5. 无 ``builtins`` 注入，不支持 ``__xxx__`` 属性访问
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# 字段引用正则：$field_name -> _field_field_name
_FIELD_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def _normalize_fields(expr: str) -> Tuple[str, Dict[str, str]]:
    """
    将 ``$close`` 转成 ``_field_close`` 这种合法 Python 标识符

    Returns:
        (normalized_expr, alias_map)
        alias_map: 内部标识符 -> 原始字段名
    """
    alias_map: Dict[str, str] = {}

    def _replace(m: re.Match) -> str:
        field_name = m.group(1)
        alias = f"_field_{field_name}"
        alias_map[alias] = field_name
        return alias

    normalized = _FIELD_PATTERN.sub(_replace, expr)
    return normalized, alias_map


# ---------------------------------------------------------------------------
# 1. 安全白名单定义
# ---------------------------------------------------------------------------

# 允许的 AST 节点类型（严格白名单）
SAFE_NODES: Tuple[type, ...] = (
    ast.Expression,    # 根节点
    ast.Constant,      # 字面量：1, 1.5, True
    ast.Name,          # 变量名
    ast.BinOp,         # 二元运算
    ast.UnaryOp,       # 一元运算
    ast.BoolOp,        # and / or
    ast.Compare,       # 比较
    ast.Call,          # 函数调用（白名单函数）
    ast.IfExp,         # 三元表达式
    ast.Tuple,         # 元组
    ast.List,          # 列表
    # 上下文
    ast.Load, ast.Load,
    # 运算符
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or,
)

# 二元运算符映射
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_BOOL_OPS = {
    ast.And: lambda a, b: a & b,   # 数组场景用位运算
    ast.Or: lambda a, b: a | b,
}


# ---------------------------------------------------------------------------
# 2. Alpha101 风格时序/截面算子
# ---------------------------------------------------------------------------

def _to_series(x: Any, index: Any) -> pd.Series:
    """将 ndarray 包装为 Series，保持索引"""
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, index=index)


def ts_mean(series: pd.Series, window: int) -> pd.Series:
    """时序滚动均值（按 code 分组由调用方处理）"""
    return series.rolling(window, min_periods=1).mean()


def ts_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=2).std()


def ts_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).sum()


def ts_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).max()


def ts_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).min()


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """时序百分位排名"""
    return series.rolling(window, min_periods=1).rank(pct=True)


def delay(series: pd.Series, period: int) -> pd.Series:
    return series.shift(period)


def delta(series: pd.Series, period: int) -> pd.Series:
    return series - series.shift(period)


def correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """时序滚动相关系数"""
    return x.rolling(window, min_periods=2).corr(y)


def stddev(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=2).std()


def _cross_section_func_factory(group_keys: pd.Series, idx: pd.Index) -> Dict[str, Callable]:
    """生成横截面算子（按 group_keys 分组）"""
    def rank(series: pd.Series) -> pd.Series:
        return series.groupby(group_keys).rank(pct=True)

    def scale(series: pd.Series, a: float = 1.0) -> pd.Series:
        def _scale(g: pd.Series) -> pd.Series:
            s = g.abs().sum()
            return a * g / s if s > 0 else g * 0
        return series.groupby(group_keys).transform(_scale)

    def demean(series: pd.Series) -> pd.Series:
        return series.groupby(group_keys).transform(lambda x: x - x.mean())

    def neutralize_helper(series: pd.Series) -> pd.Series:
        return series.groupby(group_keys).transform(lambda x: x - x.mean())

    return {
        "Rank": rank,
        "Scale": scale,
        "Demean": demean,
        "Neutralize": neutralize_helper,
    }


# 基础算子（与时序无关）
def _abs(x: Any) -> Any:
    return np.abs(x) if isinstance(x, np.ndarray) else abs(x)


def _sign(x: Any) -> Any:
    return np.sign(x) if isinstance(x, np.ndarray) else (1 if x > 0 else (-1 if x < 0 else 0))


def _log(x: Any) -> Any:
    return np.log(x) if isinstance(x, np.ndarray) else math.log(x)


def _sqrt(x: Any) -> Any:
    return np.sqrt(x) if isinstance(x, np.ndarray) else math.sqrt(x)


def _if(cond: Any, a: Any, b: Any) -> Any:
    """三元 If 算子：condition ? a : b"""
    if isinstance(cond, (pd.Series, np.ndarray)):
        return np.where(cond, a, b)
    return a if cond else b


# 白名单函数（仅用于 parse 时白名单校验，真实实现在 compute 时注入）
# 这些占位实现仅在严格模式且未通过 compute 注册时被调用
def _placeholder_ts(*args, **kwargs):
    raise UnsafeExpressionError(
        "Time-series operator called outside SafeExpressionEngine.compute()"
    )

def _placeholder_cs(*args, **kwargs):
    raise UnsafeExpressionError(
        "Cross-section operator called outside SafeExpressionEngine.compute()"
    )


SAFE_FUNCTIONS: Dict[str, Callable] = {
    # 数学
    "abs": _abs, "sign": _sign, "log": _log, "sqrt": _sqrt,
    # 条件
    "If": _if,
    # 时序（占位，compute 时替换为分组实现）
    "Ts_Mean": _placeholder_ts, "Ts_Std": _placeholder_ts, "Ts_Sum": _placeholder_ts,
    "Ts_Max": _placeholder_ts, "Ts_Min": _placeholder_ts, "Ts_Rank": _placeholder_ts,
    "Delay": _placeholder_ts, "Delta": _placeholder_ts,
    "StdDev": _placeholder_ts, "Correlation": _placeholder_ts,
    # 横截面（占位，compute 时替换为分组实现）
    "Rank": _placeholder_cs, "Scale": _placeholder_cs,
    "Demean": _placeholder_cs, "Neutralize": _placeholder_cs,
    # 内置安全函数
    "min": min, "max": max,
}


# ---------------------------------------------------------------------------
# 3. 表达式校验与求值器
# ---------------------------------------------------------------------------

class UnsafeExpressionError(ValueError):
    """表达式包含不安全节点/函数/变量时抛出"""


@dataclass
class ExpressionContext:
    """
    表达式求值上下文

    Attributes:
        data: 包含基础字段的 DataFrame, 必须有 ``code`` 和 ``date`` 列
        fields: 字段名 -> DataFrame 列名映射
        group_keys: 横截面分组键（一般是 ``code``）
    """

    data: pd.DataFrame
    fields: Dict[str, str] = field(default_factory=dict)
    group_keys: Optional[pd.Series] = None
    extra_funcs: Dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        if "code" not in self.data.columns or "date" not in self.data.columns:
            raise ValueError("data must contain 'code' and 'date' columns")
        if self.group_keys is None:
            self.group_keys = self.data["code"]


class SafeExpressionEngine:
    """
    安全表达式求值器

    关键 API:
        - ``parse(expr)``: 编译表达式，返回 ``CompiledFactor``
        - ``compute(compiled, ctx)``: 在 ``ExpressionContext`` 上执行

    使用示例::

        engine = SafeExpressionEngine()
        compiled = engine.parse("Rank(Ts_Mean($close, 5) - $close)")
        ctx = ExpressionContext(data=df_with_close)
        result = engine.compute(compiled, ctx)  # pd.Series, 索引对齐 df
    """

    def __init__(
        self,
        strict: bool = True,
        max_expression_length: int = 1000,
        allow_cross_section: bool = True,
    ):
        self.strict = strict
        self.max_expression_length = max_expression_length
        self.allow_cross_section = allow_cross_section
        self._cache: Dict[str, "CompiledFactor"] = {}

    # ---------- 校验阶段 ----------

    def _validate(self, tree: ast.AST) -> None:
        """深度遍历 AST，拒绝非白名单节点"""
        for node in ast.walk(tree):
            if not isinstance(node, SAFE_NODES):
                raise UnsafeExpressionError(
                    f"Unsafe AST node: {type(node).__name__}"
                )

            # 拒绝 Name 是 dunder
            if isinstance(node, ast.Name):
                if node.id.startswith("__") and node.id.endswith("__"):
                    raise UnsafeExpressionError(
                        f"Dunder attribute access is forbidden: {node.id}"
                    )

            # Call 节点：函数名必须在白名单
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    fname = node.func.id
                    if fname not in SAFE_FUNCTIONS:
                        raise UnsafeExpressionError(
                            f"Function not in whitelist: {fname}"
                        )
                else:
                    raise UnsafeExpressionError(
                        f"Only direct function calls are allowed, got: "
                        f"{type(node.func).__name__}"
                    )
                # 不允许关键字参数（防止注入）
                if node.keywords:
                    raise UnsafeExpressionError(
                        "Keyword arguments are not allowed"
                    )

    # ---------- 解析阶段 ----------

    def parse(self, expr: str) -> "CompiledFactor":
        """
        编译表达式字符串

        Returns:
            ``CompiledFactor`` 对象，可重复使用
        """
        expr = expr.strip()
        if len(expr) > self.max_expression_length:
            raise UnsafeExpressionError(
                f"Expression too long ({len(expr)} > {self.max_expression_length})"
            )

        if expr in self._cache:
            return self._cache[expr]

        # 预处理: $field -> _field_field
        normalized, alias_map = _normalize_fields(expr)

        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as e:
            raise UnsafeExpressionError(f"Syntax error: {e}")

        self._validate(tree)
        compiled = CompiledFactor(
            expr=expr,
            normalized_expr=normalized,
            tree=tree,
            alias_map=alias_map,
            engine=self,
        )
        self._cache[expr] = compiled
        return compiled

    # ---------- 求值阶段 ----------

    def compute(
        self,
        compiled: "CompiledFactor",
        ctx: ExpressionContext,
    ) -> pd.Series:
        """在 ExpressionContext 上执行已编译的表达式"""
        # 构造横截面算子
        cross_section_funcs = _cross_section_func_factory(
            ctx.group_keys, ctx.data.index
        ) if self.allow_cross_section else {}

        # 构造"时序算子"，自动按 group_keys 分组
        ts_funcs = self._make_ts_funcs(ctx.group_keys)
        # Correlation 需要特殊处理（双 Series + group）
        def _corr_w(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
            return self._correlation_grouped(x, y, window, ctx.group_keys)
        ts_funcs["Correlation"] = _corr_w

        # 字段名 -> 列
        env: Dict[str, Any] = dict(SAFE_FUNCTIONS)
        env.update(ts_funcs)
        env.update(cross_section_funcs)
        env.update(ctx.extra_funcs)

        # $field -> pd.Series
        for alias, col in ctx.fields.items():
            if col in ctx.data.columns:
                env[alias] = ctx.data[col]

        # 未在 fields 中声明的 $xxx 直接读 DataFrame
        # 这里用闭包：未注册字段访问时尝试 ctx.data[xxx]
        result = self._eval_node(
            compiled.tree.body, env, ctx, compiled.alias_map
        )

        # 标量结果（如 abs(-1)）包装成 Series
        if not isinstance(result, (pd.Series, np.ndarray)):
            return pd.Series(
                [result] * len(ctx.data),
                index=ctx.data.index,
            )
        if isinstance(result, np.ndarray) and result.ndim == 0:
            return pd.Series(
                [result.item()] * len(ctx.data),
                index=ctx.data.index,
            )
        return pd.Series(result, index=ctx.data.index)

    @staticmethod
    def _make_ts_funcs(group_keys: pd.Series) -> Dict[str, Callable]:
        """构造时序算子，自动按 group_keys 分组"""

        def _grouped(func):
            def wrapped(series: pd.Series, *args, **kwargs):
                if not isinstance(series, pd.Series):
                    return func(series, *args, **kwargs)
                return series.groupby(group_keys).transform(
                    lambda x: func(x, *args, **kwargs)
                )
            wrapped.__name__ = func.__name__
            return wrapped

        return {
            "Ts_Mean": _grouped(ts_mean),
            "Ts_Std": _grouped(ts_std),
            "Ts_Sum": _grouped(ts_sum),
            "Ts_Max": _grouped(ts_max),
            "Ts_Min": _grouped(ts_min),
            "Ts_Rank": _grouped(ts_rank),
            "Delay": _grouped(delay),
            "Delta": _grouped(delta),
            "StdDev": _grouped(stddev),
        }

    @staticmethod
    def _correlation_grouped(x: pd.Series, y: pd.Series, window: int,
                             group_keys: pd.Series) -> pd.Series:
        """分组滚动相关系数：先合并 x, y, group，再 groupby rolling.corr"""
        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            return correlation(x, y, window)
        df = pd.DataFrame({"x": x, "y": y, "g": group_keys.values})
        def _corr(g: pd.DataFrame) -> pd.Series:
            return g["x"].rolling(window, min_periods=2).corr(g["y"])
        result = df.groupby("g", group_keys=False).apply(_corr)
        # 重新对齐索引
        if not result.index.equals(x.index):
            result = result.reindex(x.index)
        return result

    def _eval_node(
        self,
        node: ast.AST,
        env: Dict[str, Any],
        ctx: ExpressionContext,
        alias_map: Optional[Dict[str, str]] = None,
    ) -> Any:
        """递归求值"""
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            # $field 已被替换为 _field_<name>
            if alias_map and node.id in alias_map:
                field_name = alias_map[node.id]
                if field_name in ctx.fields:
                    col = ctx.fields[field_name]
                    if col in ctx.data.columns:
                        return ctx.data[col]
                if field_name in ctx.data.columns:
                    return ctx.data[field_name]
                if self.strict:
                    raise UnsafeExpressionError(
                        f"Field not found: {field_name}"
                    )
                return 0
            if self.strict:
                raise UnsafeExpressionError(f"Unknown name: {node.id}")
            return 0

        if isinstance(node, ast.BinOp):
            l = self._eval_node(node.left, env, ctx, alias_map)
            r = self._eval_node(node.right, env, ctx, alias_map)
            return _BIN_OPS[type(node.op)](l, r)

        if isinstance(node, ast.UnaryOp):
            v = self._eval_node(node.operand, env, ctx, alias_map)
            return _UNARY_OPS[type(node.op)](v)

        if isinstance(node, ast.BoolOp):
            result = self._eval_node(node.values[0], env, ctx, alias_map)
            for v in node.values[1:]:
                result = _BOOL_OPS[type(node.op)](
                    result, self._eval_node(v, env, ctx, alias_map)
                )
            return result

        if isinstance(node, ast.Compare):
            l = self._eval_node(node.left, env, ctx, alias_map)
            for op, comp in zip(node.ops, node.comparators):
                r = self._eval_node(comp, env, ctx, alias_map)
                if not _CMP_OPS[type(op)](l, r):
                    return False
                l = r
            return True

        if isinstance(node, ast.Call):
            fname = node.func.id
            if fname not in env:
                raise UnsafeExpressionError(f"Function not available: {fname}")
            fn = env[fname]
            args = [self._eval_node(a, env, ctx, alias_map) for a in node.args]
            return fn(*args)

        if isinstance(node, ast.IfExp):
            cond = self._eval_node(node.test, env, ctx, alias_map)
            return (
                self._eval_node(node.body, env, ctx, alias_map) if cond
                else self._eval_node(node.orelse, env, ctx, alias_map)
            )

        raise UnsafeExpressionError(f"Node not handled: {type(node).__name__}")


# ---------------------------------------------------------------------------
# 4. 已编译因子 + 便捷 FactorEngine 封装
# ---------------------------------------------------------------------------

@dataclass
class CompiledFactor:
    """已编译的安全因子表达式"""
    expr: str
    normalized_expr: str
    tree: ast.AST
    alias_map: Dict[str, str]
    engine: SafeExpressionEngine


class FactorEngine:
    """
    高级因子引擎

    封装 ``SafeExpressionEngine``，并按 ``(code, date)`` 对结果做时序对齐。
    这是 jingni-trader 用户最可能调用的入口。
    """

    def __init__(self, **kwargs):
        self.expr_engine = SafeExpressionEngine(**kwargs)

    def register_function(self, name: str, func: Callable) -> None:
        """注册额外的安全函数（会更新 SAFE_FUNCTIONS）"""
        SAFE_FUNCTIONS[name] = func

    def compute(
        self,
        data: pd.DataFrame,
        formula: str,
        name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        计算单个因子

        Args:
            data: 包含 ``code``, ``date`` 和公式引用的字段列
            formula: 因子公式，例如 ``"Rank(Delay($close, 1) / $close - 1)"``
            name: 输出列名，默认用 formula 哈希

        Returns:
            带有 ``code``, ``date`` 和新因子列的 DataFrame
        """
        if name is None:
            name = f"factor_{abs(hash(formula)) % 10**8}"

        data = data.sort_values(["code", "date"]).reset_index(drop=True)
        ctx = ExpressionContext(data=data)
        compiled = self.expr_engine.parse(formula)
        result = self.expr_engine.compute(compiled, ctx)

        out = data[["code", "date"]].copy()
        out[name] = pd.Series(result, index=data.index).values
        return out

    def compute_many(
        self,
        data: pd.DataFrame,
        formulas: Dict[str, str],
    ) -> pd.DataFrame:
        """
        批量计算多个因子，并按 (code, date) 合并到单个 DataFrame

        Args:
            data: 输入数据
            formulas: ``{因子名: 公式}`` 字典

        Returns:
            合并后的 DataFrame, 包含 ``code``, ``date`` 和所有因子列
        """
        data = data.sort_values(["code", "date"]).reset_index(drop=True)
        merged = data[["code", "date"]].copy()
        for name, formula in formulas.items():
            factor_df = self.compute(data, formula, name=name)
            merged = merged.merge(factor_df, on=["code", "date"], how="left")
        return merged


# ---------------------------------------------------------------------------
# 5. 导出
# ---------------------------------------------------------------------------

__all__ = [
    "SafeExpressionEngine",
    "FactorEngine",
    "ExpressionContext",
    "CompiledFactor",
    "UnsafeExpressionError",
    "SAFE_FUNCTIONS",
    "SAFE_NODES",
]
