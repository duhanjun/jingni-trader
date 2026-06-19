"""
因子表达式 DSL 引擎
=====================

借鉴项目:
    1. microsoft/qlib (11k+ stars) - 表达式引擎 + Alpha158/360 因子库
       - 字符串公式 -> 表达式树 -> 计算
       - 操作符: Ref, Mean, Std, Rank, Corr, Delta 等
       - 截面 (CS) vs 时序 (TS) 操作
    2. KunQuant (2026) - 公式编译器, 比 pandas 快 170x
       - 设计理念: 公式表达式 -> 高效执行
    3. vectorbt - 向量化计算 + Numba JIT 思路

设计目标:
    - 提供类似 qlib 的字符串公式接口: "$close / Ref($close, 5) - 1"
    - 纯 Python 实现 (无 cython 依赖, 适配本项目)
    - 集成到现有 factor-engine 体系
    - 可选向量化 vs 分组时序两种执行模式
    - 与现有 pandas_ta / talib calculator 并存

本文件仅作为 PoC 验证, 不直接修改 main 分支代码.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. 字段引用 (类似 qlib 的 $close 语法)
# ----------------------------------------------------------------------------
class FieldRef:
    """表示 $close、$volume 等字段引用"""
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"${self.name}"


# ----------------------------------------------------------------------------
# 2. 内置算子 (Operator)
# ----------------------------------------------------------------------------
# 操作符分类
#  - 元素级 (element-wise): Abs, Log, Sign, Sqrt, Power
#  - 时序滚动 (time-series): Ref, Mean, Std, Sum, Max, Min, Delta
#  - 截面 (cross-section): Rank, Scale, Neutralize
#  - 双目 (binary): Add, Sub, Mul, Div, Greater, Less
#  - 逻辑: If, And, Or, Not

OPERATORS: Dict[str, Dict[str, Any]] = {
    # 元素级
    "Abs":     {"fn": lambda x: np.abs(x),                  "arity": 1, "kind": "element"},
    "Log":     {"fn": lambda x: np.log(np.where(x > 0, x, np.nan)), "arity": 1, "kind": "element"},
    "Sign":    {"fn": lambda x: np.sign(x),                 "arity": 1, "kind": "element"},
    "Sqrt":    {"fn": lambda x: np.sqrt(np.where(x >= 0, x, np.nan)), "arity": 1, "kind": "element"},
    "Power":   {"fn": lambda x, p: np.power(x, p),          "arity": 2, "kind": "element"},
    "Inv":     {"fn": lambda x: 1.0 / np.where(x != 0, x, np.nan), "arity": 1, "kind": "element"},
    # 时序滚动 (window 沿每只股票的时间序列滚动)
    "Ref":     {"fn": None, "arity": 2, "kind": "ts", "param_names": ["d"]},
    "Delta":   {"fn": None, "arity": 2, "kind": "ts", "param_names": ["d"]},
    "Mean":    {"fn": None, "arity": 2, "kind": "ts", "param_names": ["d"]},
    "Std":     {"fn": None, "arity": 2, "kind": "ts", "param_names": ["d"]},
    "Sum":     {"fn": None, "arity": 2, "kind": "ts", "param_names": ["d"]},
    "Max":     {"fn": None, "arity": 2, "kind": "ts", "param_names": ["d"]},
    "Min":     {"fn": None, "arity": 2, "kind": "ts", "param_names": ["d"]},
    # 截面 (按 date 分组)
    "Rank":    {"fn": None, "arity": 1, "kind": "cs"},
    "Scale":   {"fn": None, "arity": 1, "kind": "cs"},
    # 双目
    "Add":     {"fn": lambda a, b: a + b, "arity": 2, "kind": "element"},
    "Sub":     {"fn": lambda a, b: a - b, "arity": 2, "kind": "element"},
    "Mul":     {"fn": lambda a, b: a * b, "arity": 2, "kind": "element"},
    "Div":     {"fn": lambda a, b: a / np.where(b != 0, b, np.nan), "arity": 2, "kind": "element"},
    "Greater": {"fn": lambda a, b: (a > b).astype(float), "arity": 2, "kind": "element"},
    "Less":    {"fn": lambda a, b: (a < b).astype(float), "arity": 2, "kind": "element"},
    "And":     {"fn": lambda a, b: (a & b).astype(float) if hasattr(a, "__and__") else (a * b), "arity": 2, "kind": "element"},
    "Or":      {"fn": lambda a, b: (a | b).astype(float) if hasattr(a, "__or__") else ((a + b) > 0).astype(float), "arity": 2, "kind": "element"},
    "If":      {"fn": lambda c, a, b: np.where(c > 0, a, b), "arity": 3, "kind": "element"},
}


# ----------------------------------------------------------------------------
# 3. 时序滚动算子的实现 (按 code 分组)
# ----------------------------------------------------------------------------
def _ts_op(data: pd.DataFrame, op: str, d: int) -> pd.DataFrame:
    """对每只股票按时间序列执行 rolling/位移 操作"""
    if op == "Ref":
        return data.groupby("code", group_keys=False).shift(d)
    if op == "Delta":
        return data.groupby("code", group_keys=False).apply(
            lambda g: g - g.shift(d)
        ).reset_index(level=0, drop=True)
    agg_map = {
        "Mean": "mean", "Std": "std", "Sum": "sum",
        "Max": "max", "Min": "min",
    }
    if op in agg_map:
        # 只对数值列做 rolling, 避免 date 等非数值列报错
        value_col = [c for c in data.columns if c not in ("code", "date")][0]
        return data.groupby("code", group_keys=False)[value_col].rolling(
            d, min_periods=max(2, d // 2)
        ).agg(agg_map[op]).reset_index(level=0, drop=True).to_frame(value_col)
    raise ValueError(f"Unknown TS operator: {op}")


# ----------------------------------------------------------------------------
# 4. 截面算子的实现 (按 date 分组)
# ----------------------------------------------------------------------------
def _cs_op(data: pd.DataFrame, op: str, value: pd.DataFrame) -> pd.DataFrame:
    if op == "Rank":
        return value.groupby(data["date"]).rank(pct=True)
    if op == "Scale":
        # 截面归一化: x / sum(|x|)  (类似 qlib 的 scale)
        abs_sum = value.abs().groupby(data["date"]).transform("sum")
        return value / abs_sum.replace(0, np.nan)
    raise ValueError(f"Unknown CS operator: {op}")


# ----------------------------------------------------------------------------
# 5. 公式解析器
# ----------------------------------------------------------------------------
_FIELD_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")


def parse_formula(formula: str) -> ast.AST:
    """
    将 qlib 风格公式解析为 Python AST.
    转换: $close -> FieldRef("close"), 内置算子保持函数名.
    """
    # 把 $field 替换为占位符, 然后还原成 FieldRef(...) 调用
    placeholders = {}

    def _sub(m):
        key = f"__FIELD_{m.group(1)}__"
        placeholders[key] = m.group(1)
        return key

    rewritten = _FIELD_RE.sub(_sub, formula)
    tree = ast.parse(rewritten, mode="eval")

    class _FieldRefTransformer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name):
            if node.id in placeholders:
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="FieldRef", ctx=ast.Load()),
                        args=[ast.Constant(value=placeholders[node.id])],
                        keywords=[],
                    ),
                    node,
                )
            return node

    return _FieldRefTransformer().visit(tree)


# ----------------------------------------------------------------------------
# 6. AST 求值器
# ----------------------------------------------------------------------------
@dataclass
class EvalContext:
    data: pd.DataFrame          # 必须含 code, date, ... 字段
    code: str
    date: str


def _eval_node(node: ast.AST, ctx: EvalContext) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, ctx)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "FieldRef":
            return ctx.data[node.args[0].value].to_numpy(dtype=float)
        if isinstance(node.func, ast.Name) and node.func.id in OPERATORS:
            op = OPERATORS[node.func.id]
            args = [_eval_node(a, ctx) for a in node.args]
            kind = op["kind"]
            # 时序操作
            if kind == "ts":
                tmp_col = f"__tmp_{node.func.id}"
                if isinstance(args[0], np.ndarray):
                    df = pd.DataFrame({tmp_col: args[0], "code": ctx.data["code"].values, "date": ctx.data["date"].values})
                else:
                    df = pd.DataFrame({tmp_col: args[0], "code": ctx.data["code"].values, "date": ctx.data["date"].values})
                d = int(args[1]) if len(args) > 1 else 0
                return _ts_op(df, node.func.id, d)[tmp_col].to_numpy()
            # 截面操作
            if kind == "cs":
                arr = args[0]
                if not isinstance(arr, np.ndarray):
                    arr = np.asarray(arr, dtype=float)
                df = pd.DataFrame({"v": arr, "date": ctx.data["date"].values})
                return _cs_op(df, node.func.id, df[["v"]])["v"].to_numpy()
            # 元素级
            return op["fn"](*args)
        raise ValueError(f"未知函数: {ast.dump(node.func)}")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, ctx)
        right = _eval_node(node.right, ctx)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / np.where(right != 0, right, np.nan)
        if isinstance(node.op, ast.Pow): return np.power(left, right)
        raise ValueError(f"不支持的二元算子: {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, ctx)
        if isinstance(node.op, ast.USub): return -operand
        if isinstance(node.op, ast.UAdd): return operand
        raise ValueError(f"不支持的一元算子: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, ctx)
        right = _eval_node(node.comparators[0], ctx)
        if isinstance(node.ops[0], ast.Gt): return (left > right).astype(float)
        if isinstance(node.ops[0], ast.Lt): return (left < right).astype(float)
        if isinstance(node.ops[0], ast.GtE): return (left >= right).astype(float)
        if isinstance(node.ops[0], ast.LtE): return (left <= right).astype(float)
        if isinstance(node.ops[0], ast.Eq): return (left == right).astype(float)
        raise ValueError("不支持的比较符")
    raise ValueError(f"不支持的节点类型: {type(node).__name__}")


# ----------------------------------------------------------------------------
# 7. 引擎入口
# ----------------------------------------------------------------------------
def calc_factor(
    data: pd.DataFrame,
    formula: str,
    factor_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    根据字符串公式计算因子.

    参数:
        data: 包含 code, date, 以及公式中引用的所有字段 (close/open/high/low/volume 等)
        formula: qlib 风格公式, 如 "$close / Ref($close, 20) - 1"
        factor_name: 输出列名, 默认使用 formula 哈希

    返回:
        含 code, date, [factor_name] 的 DataFrame
    """
    if data.empty:
        return data.copy()
    data = data.sort_values(["code", "date"]).reset_index(drop=True)
    tree = parse_formula(formula)
    ctx = EvalContext(data=data, code="code", date="date")
    values = _eval_node(tree.body, ctx)
    name = factor_name or f"expr_{abs(hash(formula)) % 10**8}"
    out = pd.DataFrame({
        "code": data["code"].values,
        "date": data["date"].values,
        name: values,
    })
    return out


# ----------------------------------------------------------------------------
# 8. Alpha158 因子集子集 (来自 qlib contrib/data/handler.py)
# ----------------------------------------------------------------------------
ALPHA158_SUBSET: List[Dict[str, str]] = [
    # 基础价量
    {"name": "KMID",    "formula": "($close - $open) / $open",                              "grp": "price"},
    {"name": "KLEN",    "formula": "($high - $low) / $open",                                "grp": "price"},
    {"name": "KMID2",   "formula": "($close - $open) / ($high - $low + 1e-12)",              "grp": "price"},
    {"name": "KUP",     "formula": "($high - Greater($close, $open)) / $open",              "grp": "price"},
    {"name": "KUP2",    "formula": "($high - Greater($close, $open)) / ($high - $low + 1e-12)", "grp": "price"},
    {"name": "KLOW",    "formula": "(Less($close, $open) - $low) / $open",                  "grp": "price"},
    {"name": "KLOW2",   "formula": "(Less($close, $open) - $low) / ($high - $low + 1e-12)",  "grp": "price"},
    {"name": "KSFT",    "formula": "(2 * $close - $high - $low) / $open",                   "grp": "price"},
    {"name": "KSFT2",   "formula": "(2 * $close - $high - $low) / ($high - $low + 1e-12)",   "grp": "price"},
    # 时序类
    {"name": "OPEN0",   "formula": "$open / $close - 1",                                    "grp": "ts"},
    {"name": "RET_5",   "formula": "$close / Ref($close, 5) - 1",                           "grp": "ts"},
    {"name": "RET_20",  "formula": "$close / Ref($close, 20) - 1",                          "grp": "ts"},
    {"name": "RET_60",  "formula": "$close / Ref($close, 60) - 1",                          "grp": "ts"},
    {"name": "MA_5",    "formula": "Mean($close, 5) / $close",                              "grp": "ts"},
    {"name": "MA_20",   "formula": "Mean($close, 20) / $close",                             "grp": "ts"},
    {"name": "MA_60",   "formula": "Mean($close, 60) / $close",                             "grp": "ts"},
    {"name": "STD_5",   "formula": "Std($close, 5) / $close",                               "grp": "ts"},
    {"name": "STD_20",  "formula": "Std($close, 20) / $close",                              "grp": "ts"},
    {"name": "VOL_5",   "formula": "Std($close, 5) / Mean($close, 5)",                      "grp": "ts"},
    {"name": "VOL_20",  "formula": "Std($close, 20) / Mean($close, 20)",                    "grp": "ts"},
    # 量价
    {"name": "VROC_5",  "formula": "$volume / Mean($volume, 5) - 1",                        "grp": "vol"},
    {"name": "VROC_20", "formula": "$volume / Mean($volume, 20) - 1",                       "grp": "vol"},
    # 截面
    {"name": "RANK_RET_5",  "formula": "Rank(Ref($close, 5) / $close - 1)",                 "grp": "cs"},
    {"name": "RANK_VOL_20", "formula": "Rank(Std($close, 20) / $close)",                    "grp": "cs"},
    # 复合
    {"name": "RSI_14",  "formula": "If($close > Ref($close, 14), 1, -1) * Abs($close / Ref($close, 14) - 1)", "grp": "compound"},
    {"name": "BETA_20", "formula": "Delta($close, 20) / (Delta($close, 20) + Abs(Delta($close, 20)) + 1e-12)", "grp": "compound"},
]


def calc_alpha158(data: pd.DataFrame, factors: Optional[List[Dict[str, str]]] = None) -> pd.DataFrame:
    """
    一键计算 Alpha158 因子子集.
    输入: 标准 OHLCV DataFrame
    输出: code, date, <各因子>
    """
    factors = factors or ALPHA158_SUBSET
    if data.empty:
        return pd.DataFrame(columns=["code", "date"])
    out = data[["code", "date"]].copy().reset_index(drop=True)
    for f in factors:
        try:
            tmp = calc_factor(data, f["formula"], f["name"])
            out = out.merge(tmp, on=["code", "date"], how="left")
        except Exception as e:  # 跳过出错因子
            print(f"[WARN] {f['name']} 计算失败: {e}")
    return out


if __name__ == "__main__":
    # 自测
    np.random.seed(42)
    n = 100
    codes = [f"00000{i % 10}.SZ" for i in range(n)]
    dates = pd.date_range("2024-01-01", periods=10).repeat(10)
    df = pd.DataFrame({
        "code": codes,
        "date": dates,
        "open":  np.random.uniform(10, 20, n),
        "high":  np.random.uniform(11, 21, n),
        "low":   np.random.uniform(9, 19, n),
        "close": np.random.uniform(10, 20, n),
        "volume": np.random.uniform(1e6, 1e7, n),
    })
    out = calc_alpha158(df)
    print(out.head())
    print("Shape:", out.shape)
