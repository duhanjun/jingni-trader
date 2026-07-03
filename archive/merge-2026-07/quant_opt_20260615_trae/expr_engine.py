"""
Expression Engine for Alpha Factor Mining
=========================================

借鉴自 Microsoft Qlib (https://github.com/microsoft/qlib) 的表达式引擎设计。
Qlib 通过自定义表达式引擎 (qlib.data.ops) 实现了灵活的因子 DSL：
  - 数据引用: $close, $open, $high, $low, $volume, $amount
  - 算术运算: +, -, *, /
  - 时序算子: Ref(x, n), Mean(x, n), Std(x, n), Max(x, n), Min(x, n)
  - 横截面算子: Rank(x), Quantile(x, n)
  - 复合算子: (A + B) / C

本模块复现该设计，去掉 Qlib 重型注册表与缓存机制，
保留 DSL 的核心思想并用 Python eval 在安全命名空间下求值，
从而以约 200 行代码实现"声明式 + 可组合"的因子表达。

References:
  - Qlib 表达式引擎源码: qlib/data/ops.py
  - 文档: https://qlib.readthedocs.io/en/latest/component/data.html#feature

设计目标 (与现有 jingni-trader factor-engine 的差异):
  1. 现有实现: compute_a_share_factors 写死若干因子，缺乏可扩展性
  2. 本模块:    用户可写 $close / Ref($close, 1) 即可生成因子，0 代码改动添加
  3. 安全:      使用 AST 解析 + 白名单命名空间，杜绝任意代码执行
"""
from __future__ import annotations

import ast
import operator
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 安全求值：基于 AST 节点白名单
# ---------------------------------------------------------------------------
_BIN_OPS: Dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS: Dict[type, object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class FactorExpressionError(ValueError):
    """因子表达式解析或求值错误"""


# ---------------------------------------------------------------------------
# 时序算子 (T-N / Rolling Window)
# ---------------------------------------------------------------------------
def Ref(x: pd.Series, n: int) -> pd.Series:
    """时序位移: x[t-n]"""
    return x.shift(n)


def Mean(x: pd.Series, n: int) -> pd.Series:
    """滚动均值"""
    return x.rolling(window=n, min_periods=1).mean()


def Std(x: pd.Series, n: int) -> pd.Series:
    """滚动标准差"""
    return x.rolling(window=n, min_periods=2).std()


def Max(x: pd.Series, n: int) -> pd.Series:
    """滚动最大值"""
    return x.rolling(window=n, min_periods=1).max()


def Min(x: pd.Series, n: int) -> pd.Series:
    """滚动最小值"""
    return x.rolling(window=n, min_periods=1).min()


def Sum(x: pd.Series, n: int) -> pd.Series:
    """滚动求和"""
    return x.rolling(window=n, min_periods=1).sum()


def Delta(x: pd.Series, n: int = 1) -> pd.Series:
    """差分: x - Ref(x, n)"""
    return x - x.shift(n)


def Slope(x: pd.Series, n: int) -> pd.Series:
    """n 周期线性回归斜率（OLS 单变量时序斜率）"""
    arr = x.values
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < n:
        return pd.Series(out, index=x.index)
    # 用向量化 conv 提升速度
    y = arr
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    for i in range(n - 1, len(arr)):
        ys = y[i - n + 1: i + 1]
        if np.isnan(ys).any():
            continue
        y_mean = ys.mean()
        cov = ((t - t_mean) * (ys - y_mean)).sum()
        out[i] = cov / t_var
    return pd.Series(out, index=x.index)


# ---------------------------------------------------------------------------
# 横截面算子 (Cross-Sectional)
# ---------------------------------------------------------------------------
def Rank(x: pd.Series) -> pd.Series:
    """横截面百分位秩"""
    return x.rank(pct=True)


def Quantile(x: pd.Series, n: int = 5) -> pd.Series:
    """横截面等分位数桶"""
    return pd.qcut(x.rank(method="first"), n, labels=False, duplicates="drop") + 1


def ZScore(x: pd.Series) -> pd.Series:
    """横截面 Z-Score"""
    return (x - x.mean()) / x.std(ddof=0)


def Normalize(x: pd.Series) -> pd.Series:
    """横截面 Min-Max 归一化"""
    rng = x.max() - x.min()
    if rng == 0 or np.isnan(rng):
        return x * 0
    return (x - x.min()) / rng


# ---------------------------------------------------------------------------
# 表达式求值器
# ---------------------------------------------------------------------------
class ExpressionEvaluator:
    """
    在 DataFrame (index=date) 上对因子表达式求值。

    使用方法:
        df = pd.DataFrame({'close': ..., 'open': ..., ...}, index=date_index)
        ev = ExpressionEvaluator(df)
        result = ev.eval("$close / Ref($close, 1) - 1")
    """

    _ALLOWED_FUNCS = {
        "Ref": Ref, "Mean": Mean, "Std": Std, "Max": Max, "Min": Min, "Sum": Sum,
        "Delta": Delta, "Slope": Slope,
        "Rank": Rank, "Quantile": Quantile, "ZScore": ZScore, "Normalize": Normalize,
        "abs": np.abs, "log": np.log, "sqrt": np.sqrt, "sign": np.sign,
    }

    def __init__(self, data: pd.DataFrame, prefix: str = "$"):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data 必须是 pandas DataFrame")
        self.data = data
        self.prefix = prefix
        # 命名空间: $close -> data['close'] (同时提供带下划线前缀的别名)
        self._namespace: Dict[str, object] = {
            f"{prefix}{col}": data[col] for col in data.columns
        }
        self._namespace.update(self._ALLOWED_FUNCS)
        # 也允许不带前缀的列名（兼容）
        for col in data.columns:
            self._namespace.setdefault(col, data[col])
        # 同时把 $close 映射到 __close 给 sanitize 使用
        for col in data.columns:
            self._namespace.setdefault(f"__{col}", data[col])

    def eval(self, expr: str) -> pd.Series:
        """对单条表达式求值"""
        # 把 $close 这种 Qlib 风格引用替换成 Python 合法标识符 __close
        sanitized = self._sanitize(expr)
        try:
            tree = ast.parse(sanitized, mode="eval")
        except SyntaxError as e:
            raise FactorExpressionError(f"表达式语法错误: {expr!r}: {e}") from e
        return self._eval_node(tree.body)

    def _sanitize(self, expr: str) -> str:
        import re
        # 把 $xxx 替换为 __xxx (符合 Python 标识符)
        return re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', r'__\1', expr)

    def eval_batch(self, expressions: Dict[str, str]) -> pd.DataFrame:
        """批量求值，返回每个表达式对应一列的 DataFrame"""
        result: Dict[str, pd.Series] = {}
        for name, expr in expressions.items():
            try:
                result[name] = self.eval(expr)
            except FactorExpressionError as e:
                raise FactorExpressionError(f"因子 {name!r} 求值失败: {e}") from e
        return pd.DataFrame(result, index=self.data.index)

    def _eval_node(self, node: ast.AST):
        """递归求值 AST 节点"""
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self._namespace:
                raise FactorExpressionError(f"未定义符号: {node.id!r}")
            return self._namespace[node.id]
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise FactorExpressionError(f"不支持二元运算: {type(node.op).__name__}")
            return op(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise FactorExpressionError(f"不支持一元运算: {type(node.op).__name__}")
            return op(self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FactorExpressionError("只允许简单函数调用 (Name)")
            func = self._eval_node(node.func)
            if func not in self._ALLOWED_FUNCS.values():
                raise FactorExpressionError(f"函数未在白名单: {node.func.id!r}")
            args = [self._eval_node(a) for a in node.args]
            if not args:
                raise FactorExpressionError("函数调用至少需要一个参数")
            return func(*args)
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]
        raise FactorExpressionError(f"不支持的 AST 节点: {type(node).__name__}")


# ---------------------------------------------------------------------------
# 多资产批量求值
# ---------------------------------------------------------------------------
def evaluate_by_code(
    data: pd.DataFrame,
    expressions: Dict[str, str],
    code_col: str = "code",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    对长表 (含 code, date) 数据按股票分组求值多个因子。

    借鉴自 Qlib 的分组计算思想 (groupby by code)。

    参数:
        data: 必须包含 code, date 列以及表达式中引用的所有原始字段
        expressions: {factor_name: expression}
        code_col, date_col: 分组键列名

    返回:
        与 data 相同长度的 DataFrame，附加各因子列
    """
    if data.empty:
        return data.copy()

    if code_col not in data.columns or date_col not in data.columns:
        raise FactorExpressionError(f"输入数据必须包含 {code_col} 和 {date_col} 列")

    # 推断所需列
    required = _extract_columns(expressions.values(), prefix="$")
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise FactorExpressionError(f"表达式引用了缺失列: {missing}")

    # 预排序 + 设置索引以加速 groupby
    df_sorted = data.sort_values([code_col, date_col]).reset_index(drop=True)

    pieces: List[pd.DataFrame] = []
    for code, sub in df_sorted.groupby(code_col, sort=False):
        sub_indexed = sub.set_index(date_col).sort_index()
        ev = ExpressionEvaluator(sub_indexed)
        try:
            factor_df = ev.eval_batch(expressions)
        except FactorExpressionError:
            factor_df = pd.DataFrame(
                {name: np.nan for name in expressions.keys()},
                index=sub_indexed.index,
            )
        factor_df[code_col] = code
        factor_df.index.name = date_col
        pieces.append(factor_df.reset_index())

    if not pieces:
        out = df_sorted[[code_col, date_col]].copy()
        for name in expressions:
            out[name] = np.nan
        return out

    factors_long = pd.concat(pieces, ignore_index=True)
    out = df_sorted[[code_col, date_col]].merge(
        factors_long, on=[code_col, date_col], how="left"
    )
    return out


def _extract_columns(
    expressions, prefix: str = "$"
) -> List[str]:
    """从表达式集合中解析出所有被引用的列名（去掉 prefix）"""
    cols: set = set()
    for expr in expressions:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id.startswith(prefix):
                cols.add(node.id[len(prefix):])
    return sorted(cols)