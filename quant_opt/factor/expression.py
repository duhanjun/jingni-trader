"""
因子表达式 DSL (Domain-Specific Language) 与 AST 沙箱

借鉴来源:
  - Qlib Alpha158/Alpha360 (https://github.com/microsoft/qlib): 表达式因子引擎
  - Hubble (arXiv:2604.09601): AST 沙箱 + 白名单算子 + 复杂度限制
  - FactorMiner (arXiv:2602.14670): 模块化因子 Skill 架构
  - Kakushadze "101 Formulaic Alphas": 公式化因子库

核心优化思路:
  原生 factor-engine/engine.py 中因子全部硬编码在 compute_a_share_factors(),
  新增因子需修改源码并重新部署, 可扩展性差。

  本实现提供表达式 DSL:
    - 因子定义为字符串: "Mean($close, 20) / $close - 1"
    - 算子白名单 (借鉴 Hubble 安全沙箱): Ref, Mean, Std, Max, Min, Rank, Corr, ...
    - AST 解析 + 参数校验 + 复杂度限制, 保证计算安全
    - 支持截面算子 (Rank, Zscore) 与时序算子 (Mean, Std, Ref)

  用法:
      engine = ExpressionFactorEngine()
      engine.register_dataset(df)  # df 含 code, date, close, volume, ...
      factor = engine.compute("Mean($close, 20)")  # 返回 20日均价
      factor = engine.compute("Rank(-Mean($close, 5)) / Rank($volume)")  # 复合因子

  价值: 因子库可由配置文件/JSON 动态加载, 无需改代码; 也为后续 LLM 驱动
  因子挖掘 (FactorEngine/QuantaAlpha) 提供可执行的表达式后端。
"""
from __future__ import annotations
import ast
import re
import operator
from typing import Dict, Any, List, Callable, Optional
import numpy as np
import pandas as pd


class FactorExpressionError(Exception):
    """因子表达式解析/计算错误"""


# ---------- 算子注册表 (白名单, 借鉴 Hubble AST 沙箱) ----------
class OperatorRegistry:
    """算子白名单注册表, 限制可执行运算保证安全性"""

    def __init__(self):
        self._ops: Dict[str, Callable] = {}
        self._arity: Dict[str, int] = {}
        self._is_cross_section: Dict[str, bool] = {}

    def register(self, name: str, func: Callable, arity: int, cross_section: bool = False):
        self._ops[name] = func
        self._arity[name] = arity
        self._is_cross_section[name] = cross_section

    def get(self, name: str) -> Optional[Callable]:
        return self._ops.get(name)

    def arity(self, name: str) -> int:
        return self._arity.get(name, -1)

    def is_cross_section(self, name: str) -> bool:
        return self._is_cross_section.get(name, False)

    def names(self) -> List[str]:
        return sorted(self._ops.keys())


def _ts_mean(df: pd.Series, n: int) -> pd.Series:
    """时序均值 (按 code 分组 rolling)"""
    return df.groupby(level="code").transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).mean())


def _ts_std(df: pd.Series, n: int) -> pd.Series:
    return df.groupby(level="code").transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).std())


def _ts_max(df: pd.Series, n: int) -> pd.Series:
    return df.groupby(level="code").transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).max())


def _ts_min(df: pd.Series, n: int) -> pd.Series:
    return df.groupby(level="code").transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).min())


def _ts_ref(df: pd.Series, n: int) -> pd.Series:
    """Ref(x, n): x 的 n 期前的值"""
    return df.groupby(level="code").shift(n)


def _ts_rank(df: pd.Series, n: int) -> pd.Series:
    """时序排名 (在最近 n 期内的分位)"""
    return df.groupby(level="code").transform(
        lambda x: x.rolling(n, min_periods=max(1, n // 2)).apply(lambda w: pd.Series(w).rank().iloc[-1] / len(w), raw=False)
    )


def _ts_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """时序相关: a, b 在最近 n 期的相关系数"""
    df = pd.DataFrame({"a": a, "b": b})
    return df.groupby(level="code").apply(
        lambda g: g["a"].rolling(n, min_periods=max(2, n // 2)).corr(g["b"])
    ).reset_index(level=0, drop=True)


def _cs_rank(df: pd.Series) -> pd.Series:
    """截面排名 (按 date 分组 pct rank)"""
    return df.groupby(level="date").rank(pct=True)


def _cs_zscore(df: pd.Series) -> pd.Series:
    """截面标准化"""
    g = df.groupby(level="date")
    return (df - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def _cs_demean(df: pd.Series) -> pd.Series:
    return df - df.groupby(level="date").transform("mean")


def _build_default_registry() -> OperatorRegistry:
    reg = OperatorRegistry()
    # 时序算子
    reg.register("Mean", _ts_mean, 2, cross_section=False)
    reg.register("Std", _ts_std, 2, cross_section=False)
    reg.register("Max", _ts_max, 2, cross_section=False)
    reg.register("Min", _ts_min, 2, cross_section=False)
    reg.register("Ref", _ts_ref, 2, cross_section=False)
    reg.register("TsRank", _ts_rank, 2, cross_section=False)
    reg.register("Corr", _ts_corr, 3, cross_section=False)
    # 截面算子
    reg.register("Rank", _cs_rank, 1, cross_section=True)
    reg.register("Zscore", _cs_zscore, 1, cross_section=True)
    reg.register("Demean", _cs_demean, 1, cross_section=True)
    return reg


# ---------- 表达式解析与求值 ----------
class ExpressionFactorEngine:
    """
    因子表达式引擎

    将字符串表达式解析为 AST, 在白名单算子沙箱中求值。
    支持 $field 变量、二元运算 (+, -, *, /)、函数调用。
    """

    MAX_DEPTH = 8  # AST 最大深度 (借鉴 Hubble 复杂度限制)
    MAX_NODES = 50

    def __init__(self, registry: Optional[OperatorRegistry] = None):
        self.registry = registry or _build_default_registry()
        self._data: Optional[pd.DataFrame] = None
        self._fields: Dict[str, pd.Series] = {}

    def register_dataset(self, df: pd.DataFrame):
        """
        注册数据集 (含 code, date, 以及 OHLCV 等字段)

        要求 df 含 code, date 列; 内部会建立 MultiIndex (code, date) 以支持
        时序算子 (groupby code) 与截面算子 (groupby date)。
        """
        if "code" not in df.columns or "date" not in df.columns:
            raise FactorExpressionError("数据集必须含 code, date 列")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["code", "date"]).set_index(["code", "date"])
        self._data = df
        self._fields = {col: df[col] for col in df.columns}

    def compute(self, expression: str) -> pd.Series:
        """计算因子表达式, 返回 Series (index: code, date)"""
        if self._data is None:
            raise FactorExpressionError("未注册数据集, 请先调用 register_dataset")

        # 预处理: $field -> F("field")  (Qlib 风格 $ 语法非合法 Python 标识符)
        # 借鉴 Qlib Expression 解析: 将 $close 转为字段访问函数调用
        expr = re.sub(r'\$(\w+)', r'F("\1")', expression)

        tree = ast.parse(expr, mode="eval")
        node_count = self._count_nodes(tree)
        if node_count > self.MAX_NODES:
            raise FactorExpressionError(
                f"表达式复杂度超限: {node_count} 节点 > {self.MAX_NODES}"
            )
        depth = self._depth(tree.body)
        if depth > self.MAX_DEPTH:
            raise FactorExpressionError(
                f"表达式深度超限: {depth} > {self.MAX_DEPTH}"
            )
        result = self._eval(tree.body)
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=self._data.index)
        return result

    def compute_many(self, expressions: Dict[str, str]) -> pd.DataFrame:
        """批量计算多个因子表达式, 返回 DataFrame"""
        out = {}
        for name, expr in expressions.items():
            try:
                out[name] = self.compute(expr)
            except FactorExpressionError as e:
                out[name] = pd.Series(np.nan, index=self._data.index, name=name)
        return pd.DataFrame(out)

    # ---------- AST 求值 ----------
    def _eval(self, node) -> Any:
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            ops = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
            }
            op_fn = ops.get(type(node.op))
            if op_fn is None:
                raise FactorExpressionError(f"不支持的二元运算: {type(node.op).__name__}")
            try:
                return op_fn(left, right)
            except Exception as e:
                raise FactorExpressionError(f"二元运算失败: {e}")

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return +operand
            raise FactorExpressionError(f"不支持的一元运算: {type(node.op).__name__}")

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FactorExpressionError("仅支持简单函数调用")
            fname = node.func.id

            # F("field"): 字段访问函数 (由 $field 预处理生成)
            if fname == "F":
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
                    raise FactorExpressionError("F() 需要一个字符串常量参数")
                field = node.args[0].value
                if field not in self._fields:
                    raise FactorExpressionError(
                        f"未知字段: ${field} (可用: {list(self._fields.keys())})"
                    )
                return self._fields[field]

            func = self.registry.get(fname)
            if func is None:
                raise FactorExpressionError(
                    f"未注册算子: {fname} (白名单: {self.registry.names()})"
                )
            expected = self.registry.arity(fname)
            if expected >= 0 and len(node.args) != expected:
                raise FactorExpressionError(
                    f"算子 {fname} 参数数量错误: 期望 {expected}, 实际 {len(node.args)}"
                )
            args = [self._eval(a) for a in node.args]
            try:
                return func(*args)
            except Exception as e:
                raise FactorExpressionError(f"算子 {fname} 执行失败: {e}")

        elif isinstance(node, ast.Name):
            # 普通标识符: 仅允许布尔常量
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            raise FactorExpressionError(f"未知标识符: {node.id}")

        elif isinstance(node, ast.Constant):
            return node.value

        else:
            raise FactorExpressionError(f"不支持的语法节点: {type(node).__name__}")

    @staticmethod
    def _count_nodes(tree) -> int:
        return sum(1 for _ in ast.walk(tree))

    @staticmethod
    def _depth(node) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(ExpressionFactorEngine._depth(c) for c in children)
