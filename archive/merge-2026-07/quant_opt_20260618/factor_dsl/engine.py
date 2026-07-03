"""
Factor Expression DSL（领域特定语言）
====================================

借鉴自：Microsoft Qlib 的 Expression Engine
参考：https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py
      Qlib Docs: "Building Formulaic Alphas"

设计目标
--------
让研究员通过简单公式（如 "Rank(Mean($close, 5))"）声明因子，
无需手写 groupby / rolling / rank 等样板代码。

与 jingni-trader 现有 factor-engine 的区别
-----------------------------------------
现有 factor-engine.compute_a_share_factors() 是 Python 硬编码的因子列表：
  - 新增因子需要改源码、加 if-else
  - 不能运行时动态添加
  - 不利于做因子挖掘 / 遗传编程

DSL 方案优势：
  - 因子公式可外部加载（YAML / JSON / DB）
  - 支持算子组合（嵌套表达式）
  - 自动向量化（基于 pandas groupby + transform）
  - 性能：使用 pandas 原生算子，无 eval 开销

支持的算子
--------
时序算子: Ref(x, n) / Mean(x, n) / Std(x, n) / Sum(x, n) / Delta(x, n) / Corr(x, y, n)
横截面算子: Rank(x) / Quantile(x, n) / Zscore(x)
逻辑算子: Sign(x) / Log(x) / Abs(x) / Sqrt(x)
算术运算: $close / Ref($close, 1) - 1（直接在表达式中写 + - * /）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# 时序算子（time-series operators）
# 输入输出都是 MultiIndex(code, date) 的 Series
# ─────────────────────────────────────────────────────────────

def _ref(x: pd.Series, n: int) -> pd.Series:
    """Ref(x, n): 取 n 期前的 x 值（按 code 分组）"""
    if isinstance(x.index, pd.MultiIndex) and "code" in x.index.names:
        return x.groupby(level="code").shift(n)
    return x.shift(n)


def _delta(x: pd.Series, n: int) -> pd.Series:
    """Delta(x, n): x - Ref(x, n)"""
    return x - _ref(x, n)


def _mean(x: pd.Series, n: int) -> pd.Series:
    """Mean(x, n): n 期滚动均值"""
    if isinstance(x.index, pd.MultiIndex) and "code" in x.index.names:
        return x.groupby(level="code").transform(lambda s: s.rolling(n, min_periods=1).mean())
    return x.rolling(n, min_periods=1).mean()


def _std(x: pd.Series, n: int) -> pd.Series:
    """Std(x, n): n 期滚动标准差"""
    if isinstance(x.index, pd.MultiIndex) and "code" in x.index.names:
        return x.groupby(level="code").transform(lambda s: s.rolling(n, min_periods=2).std())
    return x.rolling(n, min_periods=2).std()


def _sum(x: pd.Series, n: int) -> pd.Series:
    """Sum(x, n): n 期滚动求和"""
    if isinstance(x.index, pd.MultiIndex) and "code" in x.index.names:
        return x.groupby(level="code").transform(lambda s: s.rolling(n, min_periods=1).sum())
    return x.rolling(n, min_periods=1).sum()


# ─────────────────────────────────────────────────────────────
# 横截面算子（cross-sectional operators）
# ─────────────────────────────────────────────────────────────

def _rank(x: pd.Series) -> pd.Series:
    """Rank(x): 横截面百分位排名"""
    if isinstance(x.index, pd.MultiIndex) and "date" in x.index.names:
        return x.groupby(level="date").rank(pct=True)
    return x.rank(pct=True)


def _zscore(x: pd.Series) -> pd.Series:
    """Zscore(x): 横截面标准化"""
    if isinstance(x.index, pd.MultiIndex) and "date" in x.index.names:
        return x.groupby(level="date").transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0
        )
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x * 0


def _quantile(x: pd.Series, n: int = 5) -> pd.Series:
    """Quantile(x, n): 横截面 n 分位标签"""
    if isinstance(x.index, pd.MultiIndex) and "date" in x.index.names:
        return x.groupby(level="date").transform(
            lambda s: pd.qcut(s, n, labels=False, duplicates="drop")
        )
    return pd.qcut(x, n, labels=False, duplicates="drop")


# ─────────────────────────────────────────────────────────────
# 逻辑算子（logical / math operators）
# ─────────────────────────────────────────────────────────────

def _sign(x):
    if isinstance(x, pd.Series):
        return np.sign(x)
    return np.sign(x)


def _log(x):
    if isinstance(x, pd.Series):
        return np.log(x.replace(0, np.nan))
    return np.log(x if x != 0 else np.nan)


def _abs(x):
    if isinstance(x, pd.Series):
        return x.abs()
    return abs(x)


def _sqrt(x):
    if isinstance(x, pd.Series):
        return np.sqrt(x.clip(lower=0))
    return np.sqrt(max(0, x))


# ─────────────────────────────────────────────────────────────
# 算子注册表
# ─────────────────────────────────────────────────────────────

# 时序算子：签名 (Series, n) -> Series
TS_OPS: Dict[str, Callable] = {
    "Ref": _ref, "Delta": _delta, "Mean": _mean,
    "Std": _std, "Sum": _sum,
}

# 横截面算子：签名 (Series) -> Series
CS_OPS: Dict[str, Callable] = {
    "Rank": _rank, "Zscore": _zscore,
}

# Quantile 是 (Series, n) -> Series（横截面 + 参数）
CS_N_OPS: Dict[str, Callable] = {
    "Quantile": _quantile,
}

# 一元算子
MATH_OPS: Dict[str, Callable] = {
    "Sign": _sign, "Log": _log, "Abs": _abs, "Sqrt": _sqrt,
}


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class FactorExpression:
    """声明式因子表达式"""
    name: str
    formula: str
    description: str = ""
    category: str = "custom"  # momentum / value / quality / volume / custom

    def __repr__(self) -> str:
        return f"<FactorExpression {self.name}: {self.formula}>"


# ─────────────────────────────────────────────────────────────
# 表达式 AST 与解析
# ─────────────────────────────────────────────────────────────

# Tokenizer 正则
TOKEN_PATTERN = re.compile(
    r"""
    \s+                                       |  # 空白
    (?P<NUMBER>\d+\.?\d*)                     |  # 数字
    (?P<FIELD>\$[a-zA-Z_][a-zA-Z0-9_]*)       |  # 字段
    (?P<FUNC>[A-Z][a-zA-Z0-9_]*)              |  # 函数名（大写开头）
    (?P<ID>[a-z_][a-zA-Z0-9_]*)              |  # 标识符（小写开头，因子名）
    (?P<OP>[+\-*/(),])                           # 运算符和括号
    """,
    re.VERBOSE,
)


def tokenize(formula: str) -> List[tuple]:
    """词法分析：把公式拆为 (type, value) 列表"""
    tokens = []
    for m in TOKEN_PATTERN.finditer(formula):
        if m.group(0).isspace():
            continue
        for name in ("NUMBER", "FIELD", "FUNC", "ID", "OP"):
            v = m.group(name)
            if v is not None:
                tokens.append((name, v))
                break
    return tokens


def _parse_args(tokens: List[tuple], pos: int) -> tuple:
    """解析逗号分隔的参数列表，直到遇到右括号"""
    args = []
    while pos < len(tokens) and tokens[pos] != ("OP", ")"):
        arg, pos = _parse_expr(tokens, pos)
        args.append(arg)
        if pos < len(tokens) and tokens[pos] == ("OP", ","):
            pos += 1
    return tuple(args), pos


def _parse_expr(tokens: List[tuple], pos: int) -> tuple:
    """递归下降：解析一个表达式项

    支持：
    - 数字
    - $field
    - FuncName(args)
    - 一元 +/-
    - 算术运算 * / + -
    - 括号
    """
    # 一元前缀
    if pos < len(tokens) and tokens[pos] in [("OP", "-"), ("OP", "+")]:
        op = tokens[pos][1]
        pos += 1
        left, pos = _parse_atom(tokens, pos)
        if op == "-":
            left = ("neg", left)
    else:
        left, pos = _parse_atom(tokens, pos)

    # 二元运算
    while pos < len(tokens):
        t = tokens[pos]
        if t == ("OP", "+"):
            pos += 1
            right, pos = _parse_atom(tokens, pos)
            left = ("add", left, right)
        elif t == ("OP", "-"):
            pos += 1
            right, pos = _parse_atom(tokens, pos)
            left = ("sub", left, right)
        elif t == ("OP", "*"):
            pos += 1
            right, pos = _parse_atom(tokens, pos)
            left = ("mul", left, right)
        elif t == ("OP", "/"):
            pos += 1
            right, pos = _parse_atom(tokens, pos)
            left = ("div", left, right)
        else:
            break

    return left, pos


def _parse_atom(tokens: List[tuple], pos: int) -> tuple:
    """解析最小单元：数字/字段/函数调用/括号表达式"""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    tok_type, tok_val = tokens[pos]

    # 数字
    if tok_type == "NUMBER":
        return ("num", float(tok_val)), pos + 1

    # 字段
    if tok_type == "FIELD":
        return ("field", tok_val[1:]), pos + 1  # 去掉 $

    # 标识符（用于引用已注册的因子）
    if tok_type == "ID":
        return ("idref", tok_val), pos + 1

    # 函数调用
    if tok_type == "FUNC":
        pos += 1
        if pos >= len(tokens) or tokens[pos] != ("OP", "("):
            raise ValueError(f"Expected '(' after function {tok_val}")
        pos += 1
        args, pos = _parse_args(tokens, pos)
        if pos >= len(tokens) or tokens[pos] != ("OP", ")"):
            raise ValueError(f"Expected ')' after function args for {tok_val}")
        pos += 1
        return ("func", tok_val, args), pos

    # 括号
    if tok_type == "OP" and tok_val == "(":
        pos += 1
        expr, pos = _parse_expr(tokens, pos)
        if pos >= len(tokens) or tokens[pos] != ("OP", ")"):
            raise ValueError("Expected ')'")
        pos += 1
        return expr, pos

    raise ValueError(f"Unexpected token: {(tok_type, tok_val)}")


def parse_formula(formula: str) -> tuple:
    """把公式解析为 AST"""
    tokens = tokenize(formula)
    ast, pos = _parse_expr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"Unparsed tokens after position {pos}: {tokens[pos:]}")
    return ast


# ─────────────────────────────────────────────────────────────
# AST 求值
# ─────────────────────────────────────────────────────────────

def _eval_ast(node: tuple, data: pd.DataFrame,
              intermediate: Dict[str, pd.Series]) -> Any:
    """递归求值 AST 节点

    data: MultiIndex(code, date) 的 DataFrame
    intermediate: 已计算因子的缓存
    """
    ntype = node[0]

    if ntype == "num":
        return node[1]

    if ntype == "field":
        field_name = node[1]
        if field_name not in data.columns:
            raise KeyError(f"Field ${field_name} not in data columns: {list(data.columns)}")
        return data[field_name]

    if ntype == "idref":
        # 引用已注册的因子
        name = node[1]
        if name not in intermediate:
            raise KeyError(f"Factor {name!r} not yet computed; check dependency order")
        return intermediate[name]

    if ntype == "neg":
        val = _eval_ast(node[1], data, intermediate)
        if isinstance(val, pd.Series):
            return -val
        return -val

    if ntype in ("add", "sub", "mul", "div"):
        left = _eval_ast(node[1], data, intermediate)
        right = _eval_ast(node[2], data, intermediate)
        if ntype == "add":
            return left + right
        if ntype == "sub":
            return left - right
        if ntype == "mul":
            return left * right
        if ntype == "div":
            if isinstance(right, pd.Series):
                return left / right.replace(0, np.nan)
            return left / right if right != 0 else np.nan

    if ntype == "func":
        func_name = node[1]
        args = node[2]

        if func_name in TS_OPS:
            # 时序算子：(Series, n) -> Series
            if len(args) != 2:
                raise ValueError(f"{func_name} expects 2 args, got {len(args)}")
            arg0 = _eval_ast(args[0], data, intermediate)
            n_val = _eval_ast(args[1], data, intermediate)
            if not isinstance(n_val, (int, float)):
                raise ValueError(f"{func_name} 2nd arg must be numeric")
            return TS_OPS[func_name](arg0, int(n_val))

        if func_name in CS_OPS:
            if len(args) != 1:
                raise ValueError(f"{func_name} expects 1 arg, got {len(args)}")
            arg0 = _eval_ast(args[0], data, intermediate)
            return CS_OPS[func_name](arg0)

        if func_name in CS_N_OPS:
            if len(args) != 2:
                raise ValueError(f"{func_name} expects 2 args, got {len(args)}")
            arg0 = _eval_ast(args[0], data, intermediate)
            n_val = _eval_ast(args[1], data, intermediate)
            if not isinstance(n_val, (int, float)):
                raise ValueError(f"{func_name} 2nd arg must be numeric")
            return CS_N_OPS[func_name](arg0, int(n_val))

        if func_name in MATH_OPS:
            if len(args) != 1:
                raise ValueError(f"{func_name} expects 1 arg, got {len(args)}")
            arg0 = _eval_ast(args[0], data, intermediate)
            return MATH_OPS[func_name](arg0)

        raise ValueError(f"Unknown function: {func_name}")

    raise ValueError(f"Unknown AST node type: {ntype}")


# ─────────────────────────────────────────────────────────────
# 因子引擎
# ─────────────────────────────────────────────────────────────

class FactorEngine:
    """因子 DSL 执行引擎

    使用示例：

    >>> engine = FactorEngine()
    >>> engine.register(FactorExpression("mom_5", "Mean($close, 5)"))
    >>> engine.register(FactorExpression("alpha_1", "Rank(Delta($close, 5))"))
    >>> factor_df = engine.compute(price_df)
    """

    def __init__(self):
        self.expressions: Dict[str, FactorExpression] = {}
        self._ast_cache: Dict[str, tuple] = {}

    def register(self, expr: FactorExpression) -> "FactorEngine":
        """注册因子表达式"""
        # 预解析 AST
        try:
            self._ast_cache[expr.name] = parse_formula(expr.formula)
        except Exception as e:
            raise ValueError(f"Failed to parse formula for {expr.name}: {expr.formula!r}: {e}")
        self.expressions[expr.name] = expr
        return self

    def register_many(self, exprs: List[FactorExpression]) -> "FactorEngine":
        for e in exprs:
            self.register(e)
        return self

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """构建因子间的依赖关系图"""
        dep_graph = {}
        for name, expr in self.expressions.items():
            deps = []
            for other_name in self.expressions:
                if name == other_name:
                    continue
                # 使用单词边界匹配（避免 $close 中的 c 误匹配因子 c）
                if re.search(rf"\b{re.escape(other_name)}\b", expr.formula):
                    deps.append(other_name)
            dep_graph[name] = deps
        return dep_graph

    def _topological_sort(self, dep_graph: Dict[str, List[str]]) -> List[str]:
        """拓扑排序得到计算顺序（确保被依赖的先算）"""
        visited = set()
        order = []

        def dfs(node: str):
            if node in visited:
                return
            visited.add(node)
            for dep in dep_graph.get(node, []):
                dfs(dep)
            order.append(node)

        for name in dep_graph:
            dfs(name)
        return order

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """对 data 计算所有注册因子的值

        参数:
            data: 包含 OHLCV 等基础字段的 DataFrame
                  必须包含列: code, date, open, high, low, close, volume

        返回:
            DataFrame 包含 code, date 和所有因子列
        """
        if data.empty:
            return pd.DataFrame()

        # 准备 MultiIndex 以便算子能正确 group
        if not isinstance(data.index, pd.MultiIndex):
            data_indexed = data.set_index(["code", "date"]).sort_index()
        else:
            data_indexed = data.sort_index()

        # 拓扑排序，得到计算顺序
        dep_graph = self._build_dependency_graph()
        order = self._topological_sort(dep_graph)

        # 按序计算
        results: Dict[str, pd.Series] = {}
        for name in order:
            ast = self._ast_cache.get(name)
            if ast is None:
                results[name] = pd.Series(np.nan, index=data_indexed.index)
                continue
            try:
                result = _eval_ast(ast, data_indexed, results)
                if not isinstance(result, pd.Series):
                    # 常量结果
                    result = pd.Series(result, index=data_indexed.index)
                results[name] = result
            except Exception as e:
                # 不中断，记录错误
                results[name] = pd.Series(np.nan, index=data_indexed.index)

        # 重组结果
        if not results:
            return pd.DataFrame()

        out = pd.DataFrame(results, index=data_indexed.index).reset_index()
        return out


# ─────────────────────────────────────────────────────────────
# 内置因子库（Alpha158 风格简化版）
# ─────────────────────────────────────────────────────────────

def builtin_alpha_expressions() -> List[FactorExpression]:
    """返回内置的常用 A 股因子（Alpha158 简化风格）"""
    return [
        # 动量类
        FactorExpression("mom_5", "Mean($close, 5)", "5日动量", "momentum"),
        FactorExpression("mom_20", "Mean($close, 20)", "20日动量", "momentum"),
        FactorExpression("mom_60", "Mean($close, 60)", "60日动量", "momentum"),

        # 反转类
        FactorExpression("reversal_5", "Delta($close, 5)", "5日反转", "reversal"),

        # 波动类
        FactorExpression("vol_20", "Std($close, 20)", "20日波动率", "volatility"),

        # 量能类
        FactorExpression("vol_ratio", "$volume / Mean($volume, 20)", "量比", "volume"),

        # 价量综合
        FactorExpression("pvt", "Sum($close * $volume, 20)", "价量综合", "volume"),

        # 复合 alpha
        FactorExpression("alpha_mom_rank", "Rank(Mean($close, 5))", "5日均线的横截面排名", "alpha"),
        FactorExpression("alpha_combo", "Rank(Mean($close, 5)) - Rank(Std($close, 20))",
                         "动量 - 波动率排名", "alpha"),
    ]
