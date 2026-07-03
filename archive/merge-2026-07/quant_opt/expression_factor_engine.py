"""
表达式因子引擎 —— 借鉴 Microsoft Qlib 表达式引擎

Qlib 通过表达式引擎让用户用公式字符串定义因子（如 "Ref($close, 20) / $close - 1"），
极大提升了因子库的可扩展性。jingni-trader 现有 factor-engine 将因子硬编码在
compute_a_share_factors 方法中，新增因子需修改源码，可扩展性差。

本模块实现一个轻量表达式引擎，支持：
- 字段引用：$close, $open, $high, $low, $volume, $amount, $turnover_rate
- 算子：Ref, Mean, Sum, Std, Max, Min, Rank, Corr, Cov, Delta, WMA, EMA
- 算术运算：+ - * / abs log sign
- 横截面算子：CSRank（cross-sectional rank）, CSZScore（cross-sectional z-score）

借鉴来源：
- Qlib 表达式引擎: https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py
- Qlib Alpha158 因子库: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

设计要点：
1. 表达式解析为 AST（抽象语法树），支持嵌套
2. 按股票分组计算时序算子，避免前视偏差（rolling 只用历史数据）
3. 横截面算子按日期分组
4. 支持缓存，同一表达式不重复计算
"""
from __future__ import annotations

import ast
import re
import logging
import operator
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("expression-factor-engine")


# ── 算子注册表 ──────────────────────────────────────────────
# 时序算子（按 code 分组，rolling 只用历史数据，无前视偏差）
def _ref(series: pd.Series, n: int) -> pd.Series:
    """Ref(x, n): n 期前的值"""
    return series.groupby(level="code", group_keys=False).shift(n)


def _delta(series: pd.Series, n: int) -> pd.Series:
    """Delta(x, n): x - Ref(x, n)"""
    grp = series.groupby(level="code", group_keys=False)
    return series - grp.shift(n)


def _mean(series: pd.Series, n: int) -> pd.Series:
    """Mean(x, n): n 期滚动均值"""
    return series.groupby(level="code", group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).mean()
    )


def _sum(series: pd.Series, n: int) -> pd.Series:
    """Sum(x, n): n 期滚动求和"""
    return series.groupby(level="code", group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).sum()
    )


def _std(series: pd.Series, n: int) -> pd.Series:
    """Std(x, n): n 期滚动标准差"""
    return series.groupby(level="code", group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).std()
    )


def _max(series: pd.Series, n: int) -> pd.Series:
    """Max(x, n): n 期滚动最大值"""
    return series.groupby(level="code", group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).max()
    )


def _min(series: pd.Series, n: int) -> pd.Series:
    """Min(x, n): n 期滚动最小值"""
    return series.groupby(level="code", group_keys=False).transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).min()
    )


def _wma(series: pd.Series, n: int) -> pd.Series:
    """WMA(x, n): 加权移动平均，权重为 1..n"""
    weights = np.arange(1, n + 1, dtype=float)
    weights = weights / weights.sum()

    def _w(s: pd.Series) -> pd.Series:
        return s.rolling(n, min_periods=max(1, n // 2)).apply(
            lambda x: np.dot(x, weights) if len(x) == n else np.nan, raw=True
        )

    return series.groupby(level="code", group_keys=False).transform(_w)


def _ema(series: pd.Series, n: int) -> pd.Series:
    """EMA(x, n): 指数移动平均"""
    alpha = 2.0 / (n + 1.0)
    return series.groupby(level="code", group_keys=False).transform(
        lambda x: x.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    )


def _corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Corr(x, y, n): n 期滚动相关系数"""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1)
    df = df.swaplevel().sort_index() if df.index.names == ["code", "date"] else df

    def _rolling_corr(g: pd.DataFrame) -> pd.Series:
        return g["a"].rolling(n, min_periods=max(2, n // 2)).corr(g["b"])

    # 按 code 分组
    if "code" in df.index.names:
        result = df.groupby(level="code", group_keys=False).apply(_rolling_corr)
    else:
        result = _rolling_corr(df)
    return result


def _cov(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Cov(x, y, n): n 期滚动协方差"""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1)

    def _rolling_cov(g: pd.DataFrame) -> pd.Series:
        return g["a"].rolling(n, min_periods=max(2, n // 2)).cov(g["b"])

    if "code" in df.index.names:
        result = df.groupby(level="code", group_keys=False).apply(_rolling_cov)
    else:
        result = _rolling_cov(df)
    return result


# 横截面算子（按 date 分组）
def _cs_rank(series: pd.Series) -> pd.Series:
    """CSRank(x): 横截面排名（pct）"""
    return series.groupby(level="date", group_keys=False).rank(pct=True)


def _cs_zscore(series: pd.Series) -> pd.Series:
    """CSZScore(x): 横截面 z-score"""
    grp = series.groupby(level="date", group_keys=False)
    return (series - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)


# 一元算术算子
_UNARY_OPS: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "abs": np.abs,
    "log": lambda x: np.log(x.replace(0, np.nan)),
    "sign": np.sign,
    "neg": lambda x: -x,
    "sqrt": np.sqrt,
    "square": np.square,
}

# 二元算术算子
_BINARY_OPS: Dict[str, Callable] = {
    "Add": operator.add,
    "Sub": operator.sub,
    "Mul": operator.mul,
    "Div": lambda a, b: a / b.replace(0, np.nan),
}

# 时序算子注册表（参数：series, n）
_TS_OPS: Dict[str, Callable] = {
    "Ref": _ref,
    "Delta": _delta,
    "Mean": _mean,
    "Sum": _sum,
    "Std": _std,
    "Max": _max,
    "Min": _min,
    "WMA": _wma,
    "EMA": _ema,
}

# 双序列时序算子（参数：a, b, n）
_TS_OPS_DUAL: Dict[str, Callable] = {
    "Corr": _corr,
    "Cov": _cov,
}

# 横截面算子
_CS_OPS: Dict[str, Callable] = {
    "CSRank": _cs_rank,
    "CSZScore": _cs_zscore,
}

# 支持的字段
FIELDS = {"$close", "$open", "$high", "$low", "$volume", "$amount", "$turnover_rate", "$vwap"}


class ExpressionEngine:
    """
    表达式因子引擎

    用法:
        engine = ExpressionEngine(data)  # data 含 code, date, close, ...
        factor = engine.compute("Ref($close, 20) / $close - 1")
        factor2 = engine.compute("CSRank(Mean($volume, 10))")
    """

    def __init__(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError("数据为空")
        required = {"code", "date"}
        if not required.issubset(set(data.columns)):
            raise ValueError(f"数据缺少必要列: {required}")

        # 构建 (code, date) 多级索引，便于 groupby
        self._df = data.sort_values(["code", "date"]).set_index(["code", "date"])
        self._cache: Dict[str, pd.Series] = {}

    def _get_field(self, field_name: str) -> pd.Series:
        """获取字段，$close -> close 列"""
        col = field_name.lstrip("$")
        if col == "vwap" and col not in self._df.columns:
            # VWAP 兜底：amount / volume
            if "amount" in self._df.columns and "volume" in self._df.columns:
                return (self._df["amount"] / self._df["volume"].replace(0, np.nan)).rename("vwap")
            raise KeyError(f"字段 {field_name} 不存在且无法推导")
        if col not in self._df.columns:
            raise KeyError(f"字段 {field_name} 不存在")
        return self._df[col]

    def compute(self, expression: str) -> pd.Series:
        """
        计算表达式，返回 (code, date) 索引的 Series

        参数:
            expression: 因子表达式，如 "Ref($close, 20) / $close - 1"
        """
        if expression in self._cache:
            return self._cache[expression]

        result = self._eval(expression)
        result.name = expression
        self._cache[expression] = result
        return result

    def compute_many(self, expressions: List[str]) -> pd.DataFrame:
        """批量计算多个表达式，返回 DataFrame（列名为表达式）"""
        out = {}
        for expr in expressions:
            try:
                out[expr] = self.compute(expr)
            except Exception as e:
                logger.warning(f"表达式 {expr} 计算失败: {e}")
        if not out:
            return pd.DataFrame()
        return pd.DataFrame(out)

    def _eval(self, expression: str) -> pd.Series:
        """解析并计算表达式"""
        # 规范化：字段名加 $ 前缀已被处理，这里用 Python ast 解析
        # 将 $close 转为合法标识符 _DOLLAR_close 供 ast 解析
        normalized = re.sub(r"\$(\w+)", r"_DOLLAR_\1", expression)
        try:
            tree = ast.parse(normalized, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"表达式语法错误: {expression}: {e}")

        return self._eval_node(tree.body)

    def _eval_node(self, node) -> pd.Series:
        """递归求值 AST 节点"""
        # 数字字面量
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            # 广播为与数据等长的 Series
            return pd.Series(node.value, index=self._df.index)

        # 字段引用 _DOLLAR_close -> $close
        if isinstance(node, ast.Name) and node.id.startswith("_DOLLAR_"):
            field = "$" + node.id[len("_DOLLAR_"):]
            return self._get_field(field)

        # 一元算子调用 abs(x), log(x)
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            args = [self._eval_node(a) for a in node.args]

            # 一元算术
            if func_name in _UNARY_OPS and len(args) == 1:
                return _UNARY_OPS[func_name](args[0])

            # 横截面算子
            if func_name in _CS_OPS and len(args) == 1:
                return _CS_OPS[func_name](args[0])

            # 时序单序列算子 Op(x, n)
            if func_name in _TS_OPS and len(args) == 2:
                n = self._get_int_arg(node.args[1])
                return _TS_OPS[func_name](args[0], n)

            # 时序双序列算子 Op(x, y, n)
            if func_name in _TS_OPS_DUAL and len(args) == 3:
                n = self._get_int_arg(node.args[2])
                return _TS_OPS_DUAL[func_name](args[0], args[1], n)

            raise ValueError(f"不支持的算子: {func_name}")

        # 二元运算 + - * /
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right.replace(0, np.nan)
            if isinstance(node.op, ast.Pow):
                return left ** right

        # 一元负号
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_node(node.operand)

        raise ValueError(f"不支持的表达式节点: {ast.dump(node)}")

    @staticmethod
    def _get_int_arg(node) -> int:
        """从 AST 节点提取整数参数"""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        raise ValueError(f"算子参数必须为整数，得到: {ast.dump(node)}")


# ── 预置因子库（借鉴 Qlib Alpha158 子集）──────────────────────
# 参考: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
ALPHA158_SUBSET: List[str] = [
    # ── 价格动量类 ──
    "Ref($close, 1) / $close - 1",            # 1日收益
    "Ref($close, 5) / $close - 1",            # 5日收益
    "Ref($close, 20) / $close - 1",           # 20日收益
    "Ref($close, 60) / $close - 1",           # 60日收益
    # ── 反转类 ──
    "-(Ref($close, 5) / $close - 1)",         # 5日反转
    "-(Ref($close, 20) / $close - 1)",        # 20日反转
    # ── 波动率类 ──
    "Std(Ref($close, 1) / $close - 1, 20)",   # 20日波动率
    "Std(Ref($close, 1) / $close - 1, 60)",   # 60日波动率
    # ── 成交量类 ──
    "Mean($volume, 5) / Mean($volume, 20)",   # 量比
    "Ref($volume, 1) / $volume",              # 量变化
    # ── 换手率类 ──
    "Mean($turnover_rate, 5)",                # 5日均换手
    "Mean($turnover_rate, 20)",               # 20日均换手
    "Mean($turnover_rate, 5) / Mean($turnover_rate, 20)",  # 换手率比
    # ── 价格形态类 ──
    "($close - Mean($close, 20)) / Std($close, 20)",       # 布林位置
    "($high - $close) / ($close - $low + 1e-10)",          # 上影线占比
    # ── 横截面类 ──
    "CSRank(Ref($close, 20) / $close - 1)",   # 横截面反转
    "CSRank(Mean($volume, 20))",              # 横截面成交量排名
    "CSZScore(Std(Ref($close, 1) / $close - 1, 20))",      # 横截面波动率z-score
]


def build_alpha158_subset(data: pd.DataFrame) -> pd.DataFrame:
    """
    基于 Alpha158 子集构建因子矩阵

    参数:
        data: 含 code, date, close, volume, turnover_rate 等列的 DataFrame

    返回:
        DataFrame，索引为 (code, date)，列为各因子表达式
    """
    engine = ExpressionEngine(data)
    return engine.compute_many(ALPHA158_SUBSET)
