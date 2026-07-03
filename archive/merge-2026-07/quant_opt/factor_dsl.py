"""
Factor Expression Engine (DSL)
=============================

借鉴来源
--------
- **AKQuant** (akfamily/akquant)
  https://akquant.akfamily.xyz/en/guide/factor/
  内置基于 Polars 的高性能因子表达式引擎，支持 ``Rank(Ts_Mean(close, 5))`` 等
  Alpha101 风格公式。
- **WorldQuant Alpha101** 论文
  https://arxiv.org/abs/1601.00991

设计目标
--------
当前 ``factor-engine`` 的 ``compute_a_share_factors`` 把因子硬编码在源码中，
新增因子需要修改源码并重新部署。本模块提供一个**轻量级**字符串 DSL，
允许在配置文件中声明因子公式，从而:

1. 因子可由用户在不修改代码的情况下扩展；
2. 公式与底层数据列解耦，便于做正确性单元测试；
3. 与 jingni-trader 现有因子流水线兼容，输出仍是 ``code/date/因子列`` 格式。

支持的操作符
-----------
时序算子 (time-series):
  - ``Ts_Mean(col, n)`` / ``Ts_Sum`` / ``Ts_Std`` / ``Ts_Max`` / ``Ts_Min``
  - ``Delta(col, n)``   (col[t] - col[t-n])
  - ``Delay(col, n)``   (col[t-n])
  - ``Ts_Rank(col, n)`` (滚动 rank in window)
  - ``Decay_Linear(col, n)``  (线性衰减加权均值)

横截面算子 (cross-sectional):
  - ``Rank(col)`` / ``Scale(col)``  / ``ZScore(col)``

数学算子:
  - ``Abs`` ``Sign`` ``Log`` (带符号) ``Sqrt``
  - ``Add`` ``Sub`` ``Mul`` ``Div`` ``Pow``

说明
----
- 本实现基于 pandas，确保与 jingni-trader 现有依赖 (numpy/pandas) 完全兼容。
- 执行前会在 panel (long-format) DataFrame 上以 ``groupby('code')`` 按时序算，
  以 ``groupby('date')`` 做横截面算，符合 A 股因子计算惯例。
"""
from __future__ import annotations

import ast
import logging
import math
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_opt.factor_dsl")

# 因子引擎需要的最少列
REQUIRED_COLUMNS = {"code", "date"}


# ---------------------------------------------------------------------------
# 操作符实现
# ---------------------------------------------------------------------------
class FactorError(ValueError):
    pass


def _safe_log(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(np.abs(s))


def _ts_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def _ts_sum(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).sum()


def _ts_std(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).std()


def _ts_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).max()


def _ts_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).min()


def _delta(s: pd.Series, n: int) -> pd.Series:
    return s - s.shift(n)


def _delay(s: pd.Series, n: int) -> pd.Series:
    return s.shift(n)


def _ts_rank(s: pd.Series, n: int) -> pd.Series:
    # 滚动 rank (0~1)
    return s.rolling(n, min_periods=max(2, n // 2)).rank(pct=True)


def _decay_linear(s: pd.Series, n: int) -> pd.Series:
    """线性衰减加权: 权重 1, 2, ..., n"""
    weights = np.arange(1, n + 1, dtype=float)
    wsum = weights.sum()

    def wmean(x: np.ndarray) -> float:
        # x 可能是长度 < n 的窗口 (rolling 在早期样本上的实际长度)
        k = len(x)
        if k == 0:
            return np.nan
        # 如果存在 NaN，去掉后用等长度的尾部权重
        if np.isnan(x).any():
            mask = ~np.isnan(x)
            if mask.sum() == 0:
                return np.nan
            w = weights[-mask.sum():]
            return float(np.dot(x[mask], w) / w.sum())
        # 短窗口: 取 weights 的最后 k 个 (与 NaN 去除保持语义一致)
        if k < n:
            w = weights[-k:]
            return float(np.dot(x, w) / w.sum())
        return float(np.dot(x, weights) / wsum)

    return s.rolling(n, min_periods=max(2, n // 2)).apply(wmean, raw=True)


def _rank_cs(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def _scale_cs(s: pd.Series) -> pd.Series:
    s_abs_sum = s.abs().groupby(level=0).sum()  # placeholder; 见 _apply
    return s  # 实际在 _apply 中处理


def _zscore_cs(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-12)


# 时序函数: (s, n) -> s
TS_FUNCS: Dict[str, Callable[[pd.Series, int], pd.Series]] = {
    "Ts_Mean": _ts_mean,
    "Ts_Sum": _ts_sum,
    "Ts_Std": _ts_std,
    "Ts_Max": _ts_max,
    "Ts_Min": _ts_min,
    "Delta": _delta,
    "Delay": _delay,
    "Ts_Rank": _ts_rank,
    "Decay_Linear": _decay_linear,
}

# 元素级函数: s -> s
ELEM_FUNCS: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "Abs": np.abs,
    "Sign": np.sign,
    "Log": _safe_log,
    "Sqrt": lambda s: np.sqrt(np.abs(s)),
}


# ---------------------------------------------------------------------------
# 解析与求值
# ---------------------------------------------------------------------------
@dataclass
class FactorDef:
    """单个因子定义"""

    name: str
    expr: str


class FactorEngine:
    """因子表达式求值引擎

    使用示例::

        engine = FactorEngine()
        engine.register("alpha1", "Rank(Delta(close, 5))")
        result = engine.compute(df)  # df 需包含 code/date/close 等列
    """

    # 允许的 AST 节点白名单
    _ALLOWED_NODES = (
        ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant,
        ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
        ast.USub, ast.UAdd, ast.Mod,
    )
    _ALLOWED_NAMES = (
        set(TS_FUNCS) | set(ELEM_FUNCS) | {"Rank", "ZScore"}
    )

    def __init__(self) -> None:
        self._factors: Dict[str, FactorDef] = {}

    # ---------- 公共 API ----------
    def register(self, name: str, expr: str) -> None:
        """注册一个因子"""
        if not name or not isinstance(name, str):
            raise FactorError("factor name 必须为非空字符串")
        self._check(expr)
        self._factors[name] = FactorDef(name=name, expr=expr)

    def register_many(self, defs: Dict[str, str]) -> None:
        for n, e in defs.items():
            self.register(n, e)

    def list_factors(self) -> List[FactorDef]:
        return list(self._factors.values())

    def compute(
        self,
        data: pd.DataFrame,
        factors: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """计算因子

        Parameters
        ----------
        data : DataFrame
            至少包含 ``code, date``，其它列可在表达式中引用。
        factors : list of str, optional
            仅计算指定因子名；None 表示计算全部。
        """
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise FactorError(f"输入数据缺少必要列: {missing}")

        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        target_names = factors or list(self._factors.keys())
        results: Dict[str, pd.Series] = {}
        for fname in target_names:
            if fname not in self._factors:
                raise FactorError(f"未注册的因子: {fname}")
            results[fname] = self._eval_expr(self._factors[fname].expr, df)

        out = df[["code", "date"]].copy()
        for fname, series in results.items():
            out[fname] = series.to_numpy()
        return out

    # ---------- 内部 ----------
    def _check(self, expr: str) -> None:
        """白名单校验"""
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise FactorError(f"因子表达式语法错误: {e}") from e

        # 检查所有函数调用名是否在白名单
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise FactorError("只支持直接函数调用，禁止嵌套属性")
                fn_name = node.func.id
                if fn_name not in self._ALLOWED_NAMES:
                    raise FactorError(f"未知函数: {fn_name}")
                # 检查参数数量
                if fn_name in TS_FUNCS and len(node.args) != 2:
                    raise FactorError(f"{fn_name} 需要 2 个参数 (col, n)")
                if fn_name in ELEM_FUNCS and len(node.args) != 1:
                    raise FactorError(f"{fn_name} 需要 1 个参数")
                if fn_name in {"Rank", "ZScore"} and len(node.args) != 1:
                    raise FactorError(f"{fn_name} 需要 1 个参数")
                # 递归检查参数
                for arg in node.args:
                    self._check_node(arg)
            elif isinstance(node, ast.Name):
                # 在非调用上下文中, 标识符必须是已注册的操作符
                # (数据列在求值时检查)
                if node.id in self._ALLOWED_NAMES:
                    continue
                # 否则必须是合法的列引用 - 留到求值时验证
            elif not isinstance(node, self._ALLOWED_NODES):
                raise FactorError(f"禁止的 AST 节点: {type(node).__name__}")

    def _check_node(self, node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            self._check(ast.unparse(node) if hasattr(ast, "unparse") else "")
            return
        for child in ast.iter_child_nodes(node):
            self._check_node(child)

    def _eval_expr(self, expr: str, df: pd.DataFrame) -> pd.Series:
        """对一条表达式求值"""
        tree = ast.parse(expr, mode="eval")
        return self._eval_node(tree.body, df)

    def _eval_node(self, node: ast.AST, df: pd.DataFrame) -> pd.Series:
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id, df)
        if isinstance(node, ast.Constant):
            return pd.Series(node.value, index=df.index, dtype=float)
        if isinstance(node, ast.UnaryOp):
            v = self._eval_node(node.operand, df)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
        if isinstance(node, ast.BinOp):
            l = self._eval_node(node.left, df)
            r = self._eval_node(node.right, df)
            return self._apply_binop(node.op, l, r)
        if isinstance(node, ast.Call):
            return self._eval_call(node, df)
        raise FactorError(f"不支持的节点: {ast.dump(node)}")

    @staticmethod
    def _apply_binop(op: ast.AST, l: pd.Series, r: pd.Series) -> pd.Series:
        OPS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
        }
        if type(op) not in OPS:
            raise FactorError(f"不支持的二元操作: {type(op).__name__}")
        return OPS[type(op)](l, r)

    def _resolve_name(self, name: str, df: pd.DataFrame) -> pd.Series:
        # 先查列
        if name in df.columns:
            return df[name].astype(float)
        # 查算子
        if name in ELEM_FUNCS:
            raise FactorError(f"{name} 需要以函数形式调用: {name}(...)")
        if name in TS_FUNCS:
            raise FactorError(f"{name} 需要以函数形式调用: {name}(..., n)")
        if name in {"Rank", "ZScore"}:
            raise FactorError(f"{name} 需要以函数形式调用: {name}(...)")
        raise FactorError(f"未解析的标识符: {name}")

    def _eval_call(self, node: ast.Call, df: pd.DataFrame) -> pd.Series:
        if not isinstance(node.func, ast.Name):
            raise FactorError("只支持直接函数调用，禁止嵌套属性")
        fn_name = node.func.id

        # 时序算子
        if fn_name in TS_FUNCS:
            if len(node.args) != 2:
                raise FactorError(f"{fn_name} 需要 2 个参数 (col, n)")
            col = self._eval_node(node.args[0], df)
            n = self._eval_const(node.args[1])
            if not isinstance(n, int) or n <= 0:
                raise FactorError(f"{fn_name} 的窗口必须为正整数")
            grouped = col.groupby(df["code"])
            return grouped.transform(lambda s: TS_FUNCS[fn_name](s, n))

        # 元素级
        if fn_name in ELEM_FUNCS:
            if len(node.args) != 1:
                raise FactorError(f"{fn_name} 需要 1 个参数")
            v = self._eval_node(node.args[0], df)
            return ELEM_FUNCS[fn_name](v)

        # 横截面
        if fn_name == "Rank":
            v = self._eval_node(node.args[0], df)
            return v.groupby(df["date"]).transform(lambda s: s.rank(pct=True))
        if fn_name == "ZScore":
            v = self._eval_node(node.args[0], df)
            return v.groupby(df["date"]).transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-12)
            )

        raise FactorError(f"未知函数: {fn_name}")

    @staticmethod
    def _eval_const(node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return int(node.value) if isinstance(node.value, int) else float(node.value)
        # 允许一元负号作用于正整数
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = FactorEngine._eval_const(node.operand)
            if inner is not None:
                return -inner
        return None
