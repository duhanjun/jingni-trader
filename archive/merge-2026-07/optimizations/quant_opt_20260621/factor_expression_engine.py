"""
因子表达式引擎 (Factor Expression Engine)
借鉴来源: Microsoft qlib 的 qlib/data/ops.py 表达式引擎

设计目标:
1. 用声明式字符串定义因子, 如 "Ref($close, 20) / $close - 1"
2. 解析为 AST 并编译为可调用函数, 支持表达式级缓存
3. 按股票分组向量化计算, 避免逐行 Python 循环
4. 与现有 factor-engine 的 if/elif 链对比, 验证正确性与性能

支持的操作符 (借鉴 qlib Alpha158):
- 字段引用: $close, $open, $high, $low, $volume, $amount, $turnover_rate
- 时序: Ref(field, n), Mean(field, n), Std(field, n), Max(field, n), Min(field, n)
       Sum(field, n), Var(field, n), Slope(field, n), Rsquare(field, n), Resi(field, n)
       Quantile(field, n, qscore), IdxMax(field, n), IdxMin(field, n)
- 截面: Rank(field)  (按日期截面排名)
- 二元: Corr(x, y, n), Cov(x, y, n), Greater(x, y), Less(x, y), If(cond, x, y)
- 算术: +, -, *, /, 支持括号
- 函数: Abs, Log, Sign, Power
"""
from __future__ import annotations

import ast
import re
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("factor_expression_engine")


# ============================================================
# 表达式词法与语法解析
# ============================================================

# 字段引用: $close, $open 等
_FIELD_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")

# 支持的二元运算符 (Python ast 模块直接支持 +, -, *, /, 比较等)
# 函数调用形式: Ref($close, 20), Mean($close, 20)
_SUPPORTED_FUNCS = {
    # 时序算子 (按股票分组, 沿时间轴滚动)
    "Ref", "Mean", "Std", "Max", "Min", "Sum", "Var",
    "Slope", "Rsquare", "Resi", "Quantile", "IdxMax", "IdxMin",
    "WMA",  # 加权移动平均
    # 截面算子 (按日期分组)
    "Rank",
    # 二元时序算子
    "Corr", "Cov",
    # 逐元素函数
    "Abs", "Log", "Sign", "Power",
    # 条件
    "If", "Greater", "Less",
}


class ExpressionParser:
    """
    将因子表达式字符串解析为 Python AST 节点

    借鉴 qlib 的做法: 先把 $field 替换为合法标识符, 再用 ast.parse 解析,
    最后遍历 AST 转换为可执行的计算图.
    """

    def __init__(self):
        self._field_names: List[str] = []

    def parse(self, expr: str) -> ast.AST:
        """解析表达式字符串, 返回 AST"""
        # 提取所有 $field 引用
        fields = _FIELD_RE.findall(expr)
        self._field_names = list(dict.fromkeys(fields))  # 去重保序

        # 替换 $field 为 _FIELD_field, 使其成为合法 Python 标识符
        py_expr = _FIELD_RE.sub(lambda m: f"_FIELD_{m.group(1)}", expr)

        try:
            tree = ast.parse(py_expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"表达式语法错误: {expr!r}: {e}") from e

        self._validate(tree)
        return tree

    def _validate(self, node: ast.AST) -> None:
        """校验 AST 只包含支持的节点类型"""
        if isinstance(node, ast.Expression):
            self._validate(node.body)
        elif isinstance(node, ast.BinOp):
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.UnaryOp):
            self._validate(node.operand)
        elif isinstance(node, ast.BoolOp):
            # 支持 and / or (用于 If 条件组合)
            for v in node.values:
                self._validate(v)
        elif isinstance(node, ast.Compare):
            # 支持比较运算: >, <, >=, <=, ==, !=
            self._validate(node.left)
            for comp in node.comparators:
                self._validate(comp)
        elif isinstance(node, ast.Num):  # Python < 3.8
            pass
        elif isinstance(node, ast.Constant):
            pass
        elif isinstance(node, ast.Name):
            # 必须是 _FIELD_xxx 形式
            if not node.id.startswith("_FIELD_"):
                raise ValueError(f"不支持的变量引用: {node.id}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("不支持的方法调用形式")
            if node.func.id not in _SUPPORTED_FUNCS:
                raise ValueError(f"不支持的函数: {node.func.id}")
            for arg in node.args:
                self._validate(arg)
        else:
            raise ValueError(f"不支持的语法节点: {type(node).__name__}")

    @property
    def field_names(self) -> List[str]:
        return list(self._field_names)


# ============================================================
# 向量化算子实现 (按股票分组, 沿时间轴滚动)
# ============================================================

def _group_apply(df: pd.DataFrame, field: pd.Series, func: Callable) -> pd.Series:
    """按 code 分组应用滚动函数, 返回与原索引对齐的 Series"""
    # field 需要与 df 的 code 列对齐
    tmp = pd.DataFrame({"code": df["code"].values, "val": field.values}, index=df.index)
    return tmp.groupby("code")["val"].transform(func)


def op_ref(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    """Ref(field, n): n 期前的值"""
    n = int(n)
    return _group_apply(df, field, lambda x: x.shift(n))


def op_mean(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    return _group_apply(df, field, lambda x: x.rolling(int(n), min_periods=max(1, int(n) // 2)).mean())


def op_std(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    return _group_apply(df, field, lambda x: x.rolling(int(n), min_periods=max(2, int(n) // 2)).std())


def op_max(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    return _group_apply(df, field, lambda x: x.rolling(int(n), min_periods=1).max())


def op_min(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    return _group_apply(df, field, lambda x: x.rolling(int(n), min_periods=1).min())


def op_sum(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    return _group_apply(df, field, lambda x: x.rolling(int(n), min_periods=1).sum())


def op_var(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    return _group_apply(df, field, lambda x: x.rolling(int(n), min_periods=max(2, int(n) // 2)).var())


def op_slope(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    """Slope(field, n): n 期线性回归斜率"""
    n = int(n)

    def _slope(x: pd.Series) -> pd.Series:
        y = x.rolling(n, min_periods=n)
        # 用最小二乘法计算斜率: slope = cov(x, t) / var(t)
        t = np.arange(len(x))
        t_series = pd.Series(t, index=x.index)
        t_mean = t_series.rolling(n, min_periods=n).mean()
        y_mean = y.mean()
        cov = (t_series * x).rolling(n, min_periods=n).mean() - t_mean * y_mean
        t_var = t_series.rolling(n, min_periods=n).var()
        slope = cov / t_var.replace(0, np.nan)
        return slope

    return _group_apply(df, field, _slope)


def op_rsquare(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    """Rsquare(field, n): n 期线性回归 R²"""
    n = int(n)

    def _rsq(x: pd.Series) -> pd.Series:
        t = np.arange(len(x))
        t_series = pd.Series(t, index=x.index)
        t_mean = t_series.rolling(n, min_periods=n).mean()
        y_mean = x.rolling(n, min_periods=n).mean()
        cov = (t_series * x).rolling(n, min_periods=n).mean() - t_mean * y_mean
        t_std = t_series.rolling(n, min_periods=n).std()
        y_std = x.rolling(n, min_periods=n).std()
        r = cov / (t_std * y_std).replace(0, np.nan)
        return r * r

    return _group_apply(df, field, _rsq)


def op_resi(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    """Resi(field, n): n 期线性回归残差 (最后一期)"""
    n = int(n)

    def _resi(x: pd.Series) -> pd.Series:
        t = np.arange(len(x))
        t_series = pd.Series(t, index=x.index)
        t_mean = t_series.rolling(n, min_periods=n).mean()
        y_mean = x.rolling(n, min_periods=n).mean()
        t_var = t_series.rolling(n, min_periods=n).var()
        cov = (t_series * x).rolling(n, min_periods=n).mean() - t_mean * y_mean
        slope = cov / t_var.replace(0, np.nan)
        intercept = y_mean - slope * t_mean
        y_pred = slope * t_series + intercept
        return x - y_pred

    return _group_apply(df, field, _resi)


def op_quantile(df: pd.DataFrame, field: pd.Series, n: float, qscore: float = 0.5) -> pd.Series:
    """Quantile(field, n, qscore): n 期滚动分位数"""
    n = int(n)
    return _group_apply(df, field, lambda x: x.rolling(n, min_periods=max(1, n // 2)).quantile(qscore))


def op_idxmax(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    """IdxMax(field, n): n 期内最大值出现的位置 (距当前的期数)"""
    n = int(n)

    def _idxmax(x: pd.Series) -> pd.Series:
        def _f(window: np.ndarray) -> float:
            if len(window) < 1:
                return np.nan
            return len(window) - 1 - np.argmax(window)
        return x.rolling(n, min_periods=1).apply(_f, raw=True)

    return _group_apply(df, field, _idxmax)


def op_idxmin(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    n = int(n)

    def _idxmin(x: pd.Series) -> pd.Series:
        def _f(window: np.ndarray) -> float:
            if len(window) < 1:
                return np.nan
            return len(window) - 1 - np.argmin(window)
        return x.rolling(n, min_periods=1).apply(_f, raw=True)

    return _group_apply(df, field, _idxmin)


def op_wma(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    """WMA(field, n): 加权移动平均, 权重为 1..n"""
    n = int(n)
    weights = np.arange(1, n + 1, dtype=float)

    def _wma(x: pd.Series) -> pd.Series:
        def _f(window: np.ndarray) -> float:
            if len(window) < n:
                w = weights[:len(window)]
                return np.sum(window * w) / np.sum(w)
            return np.sum(window * weights) / np.sum(weights)
        return x.rolling(n, min_periods=1).apply(_f, raw=True)

    return _group_apply(df, field, _wma)


def op_rank(df: pd.DataFrame, field: pd.Series) -> pd.Series:
    """Rank(field): 按日期截面排名 (pct)"""
    # 截面排名: 按 date 分组
    tmp = pd.DataFrame({"date": df["date"].values, "val": field.values}, index=df.index)
    return tmp.groupby("date")["val"].rank(pct=True)


def op_corr(df: pd.DataFrame, x: pd.Series, y: pd.Series, n: float) -> pd.Series:
    """Corr(x, y, n): n 期滚动相关系数"""
    n = int(n)
    tmp = pd.DataFrame({"code": df["code"].values, "x": x.values, "y": y.values}, index=df.index)

    def _corr(g: pd.DataFrame) -> pd.Series:
        return g["x"].rolling(n, min_periods=max(2, n // 2)).corr(g["y"])

    return tmp.groupby("code").apply(_corr).reset_index(level=0, drop=True).reindex(df.index)


def op_cov(df: pd.DataFrame, x: pd.Series, y: pd.Series, n: float) -> pd.Series:
    n = int(n)
    tmp = pd.DataFrame({"code": df["code"].values, "x": x.values, "y": y.values}, index=df.index)

    def _cov(g: pd.DataFrame) -> pd.Series:
        return g["x"].rolling(n, min_periods=max(2, n // 2)).cov(g["y"])

    return tmp.groupby("code").apply(_cov).reset_index(level=0, drop=True).reindex(df.index)


def op_abs(df: pd.DataFrame, field: pd.Series) -> pd.Series:
    return field.abs()


def op_log(df: pd.DataFrame, field: pd.Series) -> pd.Series:
    return np.log(field.replace(0, np.nan).clip(lower=1e-12))


def op_sign(df: pd.DataFrame, field: pd.Series) -> pd.Series:
    return np.sign(field)


def op_power(df: pd.DataFrame, field: pd.Series, n: float) -> pd.Series:
    return field ** float(n)


def op_if(df: pd.DataFrame, cond: pd.Series, x, y) -> pd.Series:
    """If(cond, x, y): 条件选择, x/y 可为 Series 或标量"""
    cond_bool = cond.fillna(False).values
    if isinstance(x, pd.Series):
        x_vals = x.values
    else:
        x_vals = np.full(len(df), float(x))
    if isinstance(y, pd.Series):
        y_vals = y.values
    else:
        y_vals = np.full(len(df), float(y))
    return pd.Series(np.where(cond_bool, x_vals, y_vals), index=df.index)


def op_greater(df: pd.DataFrame, x, y) -> pd.Series:
    """Greater(x, y): x > y, 返回 1.0/0.0"""
    if isinstance(x, pd.Series):
        x_vals = x.values
    else:
        x_vals = float(x)
    if isinstance(y, pd.Series):
        y_vals = y.values
    else:
        y_vals = float(y)
    return pd.Series((x_vals > y_vals).astype(float), index=df.index)


def op_less(df: pd.DataFrame, x, y) -> pd.Series:
    """Less(x, y): x < y, 返回 1.0/0.0"""
    if isinstance(x, pd.Series):
        x_vals = x.values
    else:
        x_vals = float(x)
    if isinstance(y, pd.Series):
        y_vals = y.values
    else:
        y_vals = float(y)
    return pd.Series((x_vals < y_vals).astype(float), index=df.index)


# 算子注册表
OPERATORS: Dict[str, Callable] = {
    "Ref": op_ref, "Mean": op_mean, "Std": op_std, "Max": op_max, "Min": op_min,
    "Sum": op_sum, "Var": op_var, "Slope": op_slope, "Rsquare": op_rsquare,
    "Resi": op_resi, "Quantile": op_quantile, "IdxMax": op_idxmax, "IdxMin": op_idxmin,
    "WMA": op_wma, "Rank": op_rank, "Corr": op_corr, "Cov": op_cov,
    "Abs": op_abs, "Log": op_log, "Sign": op_sign, "Power": op_power,
    "If": op_if, "Greater": op_greater, "Less": op_less,
}


# ============================================================
# AST 求值器
# ============================================================

class ASTEvaluator:
    """遍历 AST, 对每个节点求值, 返回 pd.Series"""

    def __init__(self, df: pd.DataFrame, field_map: Dict[str, pd.Series]):
        self.df = df
        self.field_map = field_map  # 字段名 -> Series (与 df 索引对齐)

    def eval(self, node: ast.AST) -> pd.Series:
        if isinstance(node, ast.Expression):
            return self.eval(node.body)
        if isinstance(node, ast.Constant):
            # 标量常量广播为 Series
            return pd.Series(node.value, index=self.df.index, dtype=float)
        if isinstance(node, ast.Num):  # 兼容老版本
            return pd.Series(node.n, index=self.df.index, dtype=float)
        if isinstance(node, ast.Name):
            # _FIELD_xxx
            field_name = node.id[len("_FIELD_"):]
            if field_name not in self.field_map:
                raise KeyError(f"字段未提供: {field_name}")
            return self.field_map[field_name]
        if isinstance(node, ast.UnaryOp):
            operand = self.eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not):
                return (~operand.astype(bool)).astype(float)
            raise ValueError(f"不支持的一元运算: {type(node.op).__name__}")
        if isinstance(node, ast.BoolOp):
            # and / or
            values = [self.eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                result = values[0].astype(bool)
                for v in values[1:]:
                    result = result & v.astype(bool)
                return result.astype(float)
            if isinstance(node.op, ast.Or):
                result = values[0].astype(bool)
                for v in values[1:]:
                    result = result | v.astype(bool)
                return result.astype(float)
            raise ValueError(f"不支持的布尔运算: {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            left = self.eval(node.left)
            result = None
            for op, comp in zip(node.ops, node.comparators):
                right = self.eval(comp)
                if isinstance(op, ast.Gt):
                    cmp = (left > right).astype(float)
                elif isinstance(op, ast.Lt):
                    cmp = (left < right).astype(float)
                elif isinstance(op, ast.GtE):
                    cmp = (left >= right).astype(float)
                elif isinstance(op, ast.LtE):
                    cmp = (left <= right).astype(float)
                elif isinstance(op, ast.Eq):
                    cmp = (left == right).astype(float)
                elif isinstance(op, ast.NotEq):
                    cmp = (left != right).astype(float)
                else:
                    raise ValueError(f"不支持的比较运算: {type(op).__name__}")
                result = cmp if result is None else (result.astype(bool) & cmp.astype(bool)).astype(float)
                left = right
            return result
        if isinstance(node, ast.BinOp):
            left = self.eval(node.left)
            right = self.eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right.replace(0, np.nan)
            raise ValueError(f"不支持的二元运算: {type(node.op).__name__}")
        if isinstance(node, ast.Call):
            func_name = node.func.id  # type: ignore
            args = [self.eval(a) if isinstance(a, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Call,
                                                     ast.Compare, ast.BoolOp))
                    else (float(a.value) if isinstance(a, ast.Constant) else self.eval(a))
                    for a in node.args]
            op_func = OPERATORS[func_name]
            return op_func(self.df, *args)
        raise ValueError(f"不支持的节点: {type(node).__name__}")


# ============================================================
# 表达式引擎主类
# ============================================================

class FactorExpressionEngine:
    """
    因子表达式引擎

    用法:
        engine = FactorExpressionEngine()
        result = engine.compute("Ref($close, 20) / $close - 1", data)
        # data 必须含列: code, date, close, ...

    特性:
        - 表达式级缓存: 相同表达式只解析一次
        - 字段级缓存: 同一次 compute 调用中, $close 只取一次
        - 向量化: 使用 pandas groupby + rolling, 无 Python 逐行循环
    """

    def __init__(self):
        self._ast_cache: Dict[str, ast.AST] = {}
        self._parser = ExpressionParser()

    def compute(self, expr: str, data: pd.DataFrame) -> pd.Series:
        """计算单个表达式, 返回与 data 索引对齐的 Series"""
        if data is None or data.empty:
            raise ValueError("数据为空, 无法计算因子表达式")

        if expr not in self._ast_cache:
            tree = self._parser.parse(expr)
            self._ast_cache[expr] = tree
        tree = self._ast_cache[expr]

        # 构建字段映射
        field_map: Dict[str, pd.Series] = {}
        for fname in self._parser.field_names:
            if fname not in data.columns:
                raise KeyError(f"数据中缺少字段: {fname}")
            field_map[fname] = data[fname].astype(float)

        evaluator = ASTEvaluator(data, field_map)
        result = evaluator.eval(tree)
        result.name = expr
        return result

    def compute_many(self, expressions: Dict[str, str], data: pd.DataFrame) -> pd.DataFrame:
        """批量计算多个表达式, 返回 DataFrame (含 code, date + 各因子列)"""
        result = data[["code", "date"]].copy()
        for col_name, expr in expressions.items():
            result[col_name] = self.compute(expr, data).values
        return result

    @staticmethod
    def list_supported_operators() -> List[str]:
        return sorted(OPERATORS.keys())


# ============================================================
# 预定义 Alpha158 风格因子库 (借鉴 qlib Alpha158)
# ============================================================

ALPHA158_FACTORS: Dict[str, str] = {
    # K 线形态因子
    "KMID": "($close - $open) / $open",
    "KLEN": "($high - $low) / $open",
    "KMID2": "($close - $open) / ($high - $low + 1e-12)",
    "KUP": "($high - Greater($open, $close)) / $open",
    "KLOW": "(Less($open, $close) - $low) / $open",
    "KSFT": "(($high + $low) / 2 - $close) / $open",
    # 价格因子
    "OPEN0": "$open",
    "HIGH0": "$high",
    "LOW0": "$low",
    "VWAP0": "$amount / ($volume + 1e-12)",
    # 滚动窗口因子 (5/10/20/30/60 日)
    "ROC5": "Ref($close, 5) / $close",
    "ROC10": "Ref($close, 10) / $close",
    "ROC20": "Ref($close, 20) / $close",
    "ROC30": "Ref($close, 30) / $close",
    "ROC60": "Ref($close, 60) / $close",
    "MA5": "Mean($close, 5) / $close",
    "MA10": "Mean($close, 10) / $close",
    "MA20": "Mean($close, 20) / $close",
    "MA30": "Mean($close, 30) / $close",
    "MA60": "Mean($close, 60) / $close",
    "STD5": "Std($close, 5) / $close",
    "STD10": "Std($close, 10) / $close",
    "STD20": "Std($close, 20) / $close",
    "STD30": "Std($close, 30) / $close",
    "STD60": "Std($close, 60) / $close",
    "BETA5": "Slope($close, 5) / $close",
    "BETA10": "Slope($close, 10) / $close",
    "BETA20": "Slope($close, 20) / $close",
    "BETA30": "Slope($close, 30) / $close",
    "BETA60": "Slope($close, 60) / $close",
    "RSQR5": "Rsquare($close, 5)",
    "RSQR10": "Rsquare($close, 10)",
    "RSQR20": "Rsquare($close, 20)",
    "RSQR30": "Rsquare($close, 30)",
    "RSQR60": "Rsquare($close, 60)",
    "RESI5": "Resi($close, 5) / $close",
    "RESI10": "Resi($close, 10) / $close",
    "RESI20": "Resi($close, 20) / $close",
    "RESI30": "Resi($close, 30) / $close",
    "RESI60": "Resi($close, 60) / $close",
    "MAX5": "Max($close, 5) / $close",
    "MAX10": "Max($close, 10) / $close",
    "MAX20": "Max($close, 20) / $close",
    "MAX30": "Max($close, 30) / $close",
    "MAX60": "Max($close, 60) / $close",
    "MIN5": "Min($close, 5) / $close",
    "MIN10": "Min($close, 10) / $close",
    "MIN20": "Min($close, 20) / $close",
    "MIN30": "Min($close, 30) / $close",
    "MIN60": "Min($close, 60) / $close",
    "QTLU5": "Quantile($close, 5, 0.8) / $close",
    "QTLU20": "Quantile($close, 20, 0.8) / $close",
    "QTLD5": "Quantile($close, 5, 0.2) / $close",
    "QTLD20": "Quantile($close, 20, 0.2) / $close",
    "RANK5": "Rank(Mean($close, 5) / $close)",
    "RANK20": "Rank(Mean($close, 20) / $close)",
    "RSV5": "($close - Min($low, 5)) / (Max($high, 5) - Min($low, 5) + 1e-12)",
    "RSV20": "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-12)",
    "IMAX5": "IdxMax($high, 5)",
    "IMAX20": "IdxMax($high, 20)",
    "IMIN5": "IdxMin($low, 5)",
    "IMIN20": "IdxMin($low, 20)",
    "IMXD5": "IdxMax($high, 5) - IdxMin($low, 5)",
    "IMXD20": "IdxMax($high, 20) - IdxMin($low, 20)",
    "CORR5": "Corr($close, Log($volume + 1), 5)",
    "CORR10": "Corr($close, Log($volume + 1), 10)",
    "CORR20": "Corr($close, Log($volume + 1), 20)",
    "CORR30": "Corr($close, Log($volume + 1), 30)",
    "CORR60": "Corr($close, Log($volume + 1), 60)",
    "CORD5": "Corr($close / Ref($close, 1), Log($volume + 1), 5)",
    "CORD10": "Corr($close / Ref($close, 1), Log($volume + 1), 10)",
    "CORD20": "Corr($close / Ref($close, 1), Log($volume + 1), 20)",
    "CORD30": "Corr($close / Ref($close, 1), Log($volume + 1), 30)",
    "CORD60": "Corr($close / Ref($close, 1), Log($volume + 1), 60)",
    "CNTP5": "Mean($close > Ref($close, 1), 5)",
    "CNTP20": "Mean($close > Ref($close, 1), 20)",
    "CNTN5": "Mean($close < Ref($close, 1), 5)",
    "CNTN20": "Mean($close < Ref($close, 1), 20)",
    "CNTD5": "Mean($close > Ref($close, 1), 5) - Mean($close < Ref($close, 1), 5)",
    "CNTD20": "Mean($close > Ref($close, 1), 20) - Mean($close < Ref($close, 1), 20)",
    "SUMP5": "Sum(If($close > Ref($close, 1), $close / Ref($close, 1) - 1, 0), 5)",
    "SUMP20": "Sum(If($close > Ref($close, 1), $close / Ref($close, 1) - 1, 0), 20)",
    "SUMN5": "Sum(If($close < Ref($close, 1), Ref($close, 1) / $close - 1, 0), 5)",
    "SUMN20": "Sum(If($close < Ref($close, 1), Ref($close, 1) / $close - 1, 0), 20)",
    "SUMD5": "Sum(If($close > Ref($close, 1), $close / Ref($close, 1) - 1, 0), 5) - Sum(If($close < Ref($close, 1), Ref($close, 1) / $close - 1, 0), 5)",
    "SUMD20": "Sum(If($close > Ref($close, 1), $close / Ref($close, 1) - 1, 0), 20) - Sum(If($close < Ref($close, 1), Ref($close, 1) / $close - 1, 0), 20)",
    "VMA5": "Mean($volume, 5) / ($volume + 1e-12)",
    "VMA10": "Mean($volume, 10) / ($volume + 1e-12)",
    "VMA20": "Mean($volume, 20) / ($volume + 1e-12)",
    "VMA30": "Mean($volume, 30) / ($volume + 1e-12)",
    "VMA60": "Mean($volume, 60) / ($volume + 1e-12)",
    "VSTD5": "Std($volume, 5) / ($volume + 1e-12)",
    "VSTD10": "Std($volume, 10) / ($volume + 1e-12)",
    "VSTD20": "Std($volume, 20) / ($volume + 1e-12)",
    "VSTD30": "Std($volume, 30) / ($volume + 1e-12)",
    "VSTD60": "Std($volume, 60) / ($volume + 1e-12)",
    "WVMA5": "Std(Abs($close / Ref($close, 1) - 1) * $volume, 5) / (Mean(Abs($close / Ref($close, 1) - 1) * $volume, 5) + 1e-12)",
    "WVMA20": "Std(Abs($close / Ref($close, 1) - 1) * $volume, 20) / (Mean(Abs($close / Ref($close, 1) - 1) * $volume, 20) + 1e-12)",
    "VSUMP5": "Sum(If($close > Ref($close, 1), $volume, 0), 5) / (Sum($volume, 5) + 1e-12)",
    "VSUMP20": "Sum(If($close > Ref($close, 1), $volume, 0), 20) / (Sum($volume, 20) + 1e-12)",
    "VSUMN5": "Sum(If($close < Ref($close, 1), $volume, 0), 5) / (Sum($volume, 5) + 1e-12)",
    "VSUMN20": "Sum(If($close < Ref($close, 1), $volume, 0), 20) / (Sum($volume, 20) + 1e-12)",
    "VSUMD5": "Sum(If($close > Ref($close, 1), $volume, 0), 5) / (Sum($volume, 5) + 1e-12) - Sum(If($close < Ref($close, 1), $volume, 0), 5) / (Sum($volume, 5) + 1e-12)",
    "VSUMD20": "Sum(If($close > Ref($close, 1), $volume, 0), 20) / (Sum($volume, 20) + 1e-12) - Sum(If($close < Ref($close, 1), $volume, 0), 20) / (Sum($volume, 20) + 1e-12)",
    "REVERSAL_5": "-1 * ($close / Ref($close, 5) - 1)",
    "REVERSAL_20": "-1 * ($close / Ref($close, 20) - 1)",
    "REVERSAL_60": "-1 * ($close / Ref($close, 60) - 1)",
}


def compute_alpha158(data: pd.DataFrame, engine: Optional[FactorExpressionEngine] = None) -> pd.DataFrame:
    """计算 Alpha158 风格因子库"""
    if engine is None:
        engine = FactorExpressionEngine()
    return engine.compute_many(ALPHA158_FACTORS, data)
