"""
alpha_expression_engine - 声明式因子表达式引擎

借鉴来源:
  - Microsoft qlib (4.2k+ stars) 的表达式引擎
    参考: https://github.com/microsoft/qlib 的 $close, Ref($close, 1), Mean($close, 3) 设计
  - RD-Agent (microsoft/RD-Agent) 的因子-模型协同优化思想

目标:
  jingni-trader 当前 factor-engine/scripts/engine.py 中因子是硬编码的列表
  (ret_1d, ret_5d, reversal_20d, lncap, turnover_20d, volatility_20d ...) ,
  添加/修改因子需要改源码。本模块提供声明式表达方式,
  用户/上游智能体可通过 JSON / DSL 注册新因子,运行时动态求值。

设计原则:
  1. 极简: 仅依赖 pandas/numpy,不引入外部 DSL 解析器
  2. 安全: 沙箱化执行,只允许注册的算子
  3. 可扩展: 用户可注册自定义算子
  4. 可缓存: 表达式->结果可在 session 内缓存
"""
from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_opt.alpha_engine")


# ============================================================================
# 0. 表达式预处理
# ============================================================================

import re as _re

_DOLLAR_RE = _re.compile(r'\$([a-zA-Z_][a-zA-Z_0-9]*)')


def _preprocess_dollar(expr: str) -> str:
    """
    把 $name 转成 __field__('name'), 兼容 Python ast 解析
    """
    return _DOLLAR_RE.sub(r"__field__('\1')", expr)


# ============================================================================
# 1. 内置算子 (借鉴 qlib)
# ============================================================================

# 时序算子 - 输入为 DataFrame (按 code 已 sort), 返回 DataFrame
TS_OPS: Dict[str, Callable] = {
    # 引用算子
    "Ref": lambda x, n: x.shift(n),
    "RefFuture": lambda x, n: x.shift(-n),

    # 滚动算子
    "Mean": lambda x, n: x.rolling(n, min_periods=1).mean(),
    "Std":  lambda x, n: x.rolling(n, min_periods=1).std(),
    "Sum":  lambda x, n: x.rolling(n, min_periods=1).sum(),
    "Max":  lambda x, n: x.rolling(n, min_periods=1).max(),
    "Min":  lambda x, n: x.rolling(n, min_periods=1).min(),
    "Median": lambda x, n: x.rolling(n, min_periods=1).median(),
    "Skew": lambda x, n: x.rolling(n, min_periods=2).skew(),
    "Kurt": lambda x, n: x.rolling(n, min_periods=3).kurt(),

    # 动量 / 反转
    "Ret": lambda x, n: x.pct_change(n),
    "LogRet": lambda x, n: np.log(x / x.shift(n)),

    # 排名 (截面)
    "Rank": lambda x: x.rank(pct=True),
    "ZScore": lambda x: (x - x.mean()) / x.std().replace(0, np.nan),

    # 差分
    "Delta": lambda x, n: x.diff(n),

    # 条件
    "If": lambda cond, a, b: pd.Series(np.where(cond, a, b), index=cond.index),
}

# 基础二元算子
BINARY_OPS: Dict[str, Callable] = {
    "Add":  lambda a, b: a + b,
    "Sub":  lambda a, b: a - b,
    "Mul":  lambda a, b: a * b,
    "Div":  lambda a, b: a / b.replace(0, np.nan),
    "Mod":  lambda a, b: a % b.replace(0, np.nan),
    "And":  lambda a, b: a & b,
    "Or":   lambda a, b: a | b,
    "Gt":   lambda a, b: a > b,
    "Lt":   lambda a, b: a < b,
    "Eq":   lambda a, b: a == b,
}

# 数据字段映射 (内置)
DEFAULT_FIELDS = {
    "open", "high", "low", "close", "volume", "amount",
    "vwap", "turnover_rate", "change_pct", "is_limit_up", "is_limit_down",
}

# 内置可注册因子 (与 jingni-trader factor-engine 行为对齐)
BUILTIN_FACTORS: Dict[str, str] = {
    # 动量/反转
    "ret_1d":  "Ret($close, 1)",
    "ret_5d":  "Ret($close, 5)",
    "ret_20d": "Ret($close, 20)",
    "ret_60d": "Ret($close, 60)",
    "reversal_5d":  "-Ret($close, 5)",
    "reversal_20d": "-Ret($close, 20)",

    # 流动性/规模
    "lncap": "Log($amount / $turnover_rate * 100)",
    "turnover_20d": "Mean($turnover_rate, 20)",
    "turnover_5d":  "Mean($turnover_rate, 5)",
    "turnover_change": "Mean($turnover_rate, 5) / Mean($turnover_rate, 20) - 1",

    # 波动
    "volatility_20d": "Std(Ret($close, 1), 20)",
    "volume_20d": "Mean($volume, 20)",
    "volume_ratio": "$volume / Mean($volume, 20)",

    # 价量复合
    "hl_range": "($high - $low) / $close",
    "oc_change": "($close - $open) / $open",
    "money_flow_20d": "Sum(Ret($close, 1) * $amount, 20)",

    # 经典 alpha101 风格
    "alpha_001": "Rank(Delta(Log($close), 1))",
    "alpha_005": "Rank(Mean($close, 5))",
    "alpha_010": "Rank(If($change_pct > 0, Std($volume, 5), -Std($volume, 5)))",
    "alpha_026": "-Rank(Mean(Delta($close, 5), 2))",
    "alpha_038": "-Rank(Mean($close, 10)) * Rank(Std($close, 10))",
}


# ============================================================================
# 2. 表达式求值器
# ============================================================================

@dataclass
class EvalContext:
    """单次表达式求值的上下文"""
    data: pd.DataFrame                # 包含 [code, date, open, ...] 字段
    custom_ops: Dict[str, Callable] = field(default_factory=dict)
    cache: Dict[str, pd.Series] = field(default_factory=dict)
    eval_count: int = 0

    def get_field(self, name: str) -> pd.Series:
        """通过 $name 访问数据字段, 返回按 (code, date) 索引的 Series"""
        if name not in self.data.columns:
            raise KeyError(f"数据缺少字段: {name} (可用: {list(self.data.columns)})")
        return self.data.set_index(['code', 'date'])[name] if 'code' in self.data.columns else self.data[name]


class ExpressionEngine:
    """
    声明式因子表达式引擎

    支持的语法 (与 qlib 对齐):
        $close            - 数据字段
        Ret(x, n)         - 时序算子
        Rank(x)           - 截面算子
        Add(a, b)         - 二元算子
        -x, x + y         - 字面量运算符重写为对应函数调用
    """

    def __init__(self, custom_factors: Optional[Dict[str, str]] = None,
                 custom_ops: Optional[Dict[str, Callable]] = None):
        self.factors: Dict[str, str] = dict(BUILTIN_FACTORS)
        if custom_factors:
            self.factors.update(custom_factors)
        self.ops: Dict[str, Callable] = dict(TS_OPS)
        if custom_ops:
            self.ops.update(custom_ops)
        # 重命名以避免与字段冲突
        self._all_funcs = set(self.ops.keys()) | set(self.factors.keys()) | {"Log", "Abs", "Sign"}

    # 工具函数 ----------------------------------------------------
    @staticmethod
    def _maybe_scalar(x: Any) -> Any:
        """如果 x 是只含一个标量值的 Series, 返回标量; 否则返回原值"""
        if isinstance(x, pd.Series):
            if len(x) == 1:
                try:
                    v = x.item()
                    if np.isfinite(v) or isinstance(v, str):
                        return v
                except (ValueError, TypeError):
                    pass
        return x

    def _to_series(self, x: Any, ctx: EvalContext) -> pd.Series:
        """把任意值转成与 ctx.data 对齐的 Series"""
        if isinstance(x, pd.Series):
            return x
        if isinstance(x, (int, float, np.number)):
            return pd.Series(x, index=ctx.data.index)
        return pd.Series(x)

    # 解析与求值 --------------------------------------------------
    def parse(self, expr: str) -> ast.AST:
        """把表达式字符串解析为 AST"""
        # 把 $name 转成伪函数调用 __field__('name'), 让 ast 能解析
        pre = _preprocess_dollar(expr)
        try:
            tree = ast.parse(pre, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"无法解析因子表达式 '{expr}': {e}") from e
        self._validate_safety(tree)
        return tree

    def _validate_safety(self, node: ast.AST) -> None:
        """白名单安全校验: 禁止 import / 函数调用任意内置"""
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                raise ValueError("禁止在因子表达式中使用 import")
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    name = sub.func.id
                    if name not in (self._all_funcs | {"__field__"}):
                        raise ValueError(f"未注册算子/因子: {name}")
                elif isinstance(sub.func, ast.Attribute):
                    raise ValueError("禁止属性调用")

    def eval(self, expr: str, ctx: EvalContext) -> pd.Series:
        """对单条表达式求值"""
        if expr in ctx.cache:
            return ctx.cache[expr]
        ctx.eval_count += 1
        tree = self.parse(expr)
        result = self._eval_node(tree.body, ctx)
        ctx.cache[expr] = result
        return result

    def _eval_node(self, node: ast.AST, ctx: EvalContext) -> Any:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, ctx)
        if isinstance(node, ast.Constant):
            v = node.value
            # 标量值直接返回 (不要包装成 Series, 否则后续 ambiguous bool)
            if isinstance(v, (int, float, np.number)):
                return v
            return self._to_series(v, ctx)
        if isinstance(node, ast.Name):
            if node.id in self.factors:
                return self.eval(self.factors[node.id], ctx)
            if node.id in self.ops:
                raise ValueError(f"{node.id} 是算子, 需要参数")
            if node.id in DEFAULT_FIELDS:
                return ctx.get_field(node.id)
            if node.id == "Log":
                raise ValueError("Log 必须作为函数调用")
            if node.id == "__field__":
                raise ValueError("__field__ 不可直接求值 (内部算子)")
            raise ValueError(f"未识别的标识符: {node.id}")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, ctx)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise ValueError(f"不支持的一元操作: {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            left  = self._eval_node(node.left, ctx)
            right = self._eval_node(node.right, ctx)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right.replace(0, np.nan) if isinstance(right, pd.Series) else left / right
            raise ValueError(f"不支持的二元操作: {type(node.op).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("仅支持命名函数调用")
            name = node.func.id
            # 特殊处理 __field__ (来自 $name 预处理)
            if name == "__field__":
                arg = node.args[0]
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    raise ValueError("__field__ 参数必须是字符串")
                return ctx.get_field(arg.value)
            # 特殊处理 Log
            if name == "Log":
                args = [self._eval_node(a, ctx) for a in node.args]
                a0 = self._maybe_scalar(args[0])
                if isinstance(a0, pd.Series):
                    return np.log(a0.replace(0, np.nan))
                return np.log(a0) if a0 != 0 else np.nan
            if name == "Abs":
                args = [self._eval_node(a, ctx) for a in node.args]
                a0 = self._maybe_scalar(args[0])
                return abs(a0) if not isinstance(a0, pd.Series) else a0.abs()
            if name == "Sign":
                args = [self._eval_node(a, ctx) for a in node.args]
                a0 = self._maybe_scalar(args[0])
                return np.sign(a0)
            if name in self.factors:
                return self.eval(self.factors[name], ctx)
            if name in self.ops:
                args_raw = [self._eval_node(a, ctx) for a in node.args]
                # 标量参数解包 (例如 Ret($close, 5) 中的 5)
                args = [self._maybe_scalar(a) for a in args_raw]
                return self.ops[name](*args)
            raise ValueError(f"未注册算子: {name}")
        if isinstance(node, ast.Subscript):
            # 预留, 未实现
            raise NotImplementedError("下标访问暂不支持")
        raise ValueError(f"不支持的节点类型: {type(node).__name__}")


# ============================================================================
# 3. 批量计算 API
# ============================================================================

@dataclass
class FactorResult:
    name: str
    expression: str
    series: pd.Series
    elapsed_ms: float


def compute_factors(
    data: pd.DataFrame,
    expressions: Dict[str, str],
    custom_factors: Optional[Dict[str, str]] = None,
    custom_ops: Optional[Dict[str, Callable]] = None,
) -> Tuple[pd.DataFrame, List[FactorResult]]:
    """
    批量计算一组因子

    参数:
        data: 原始数据 DataFrame, 必须含 code, date 与 $name 引用的列
        expressions: {因子名: 表达式} 字典
        custom_factors, custom_ops: 自定义算子

    返回:
        (合并后的 DataFrame [code, date, factor_1, factor_2, ...], 各因子元信息列表)
    """
    if data.empty:
        return pd.DataFrame(columns=['code', 'date']), []

    engine = ExpressionEngine(custom_factors, custom_ops)
    ctx = EvalContext(data=data.sort_values(['code', 'date']).reset_index(drop=True))

    base = data[['code', 'date']].copy() if 'code' in data.columns else data.copy()
    out = base.copy()
    results: List[FactorResult] = []

    for name, expr in expressions.items():
        t0 = time.perf_counter()
        try:
            series = engine.eval(expr, ctx)
            elapsed = (time.perf_counter() - t0) * 1000
            if isinstance(series, pd.Series):
                # 如果 series 是 (code, date) 复合索引, 先 reset 到与 out 对齐
                if not series.index.equals(out.index):
                    series = series.reset_index(drop=True)
                if len(series) == len(out):
                    out[name] = series.values
                else:
                    logger.warning("因子 %s 长度 %d != 数据长度 %d, 填充 NaN",
                                   name, len(series), len(out))
                    out[name] = np.nan
            else:
                out[name] = np.nan
            results.append(FactorResult(name, expr, series, elapsed))
        except Exception as e:
            logger.exception("计算因子 %s 失败: %s", name, e)
            out[name] = np.nan
            results.append(FactorResult(name, expr, pd.Series(dtype=float), 0.0))

    return out, results


def list_builtin_factors() -> List[str]:
    """列出所有内置因子名, 供接口暴露"""
    return sorted(BUILTIN_FACTORS.keys())


def get_factor_expression(name: str) -> Optional[str]:
    return BUILTIN_FACTORS.get(name)


# ============================================================================
# 4. 入口 / CLI
# ============================================================================

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="alpha_expression_engine 自检")
    ap.add_argument("--list", action="store_true", help="列出所有内置因子")
    args = ap.parse_args()
    if args.list:
        for n, e in BUILTIN_FACTORS.items():
            print(f"  {n:20s} = {e}")


if __name__ == "__main__":
    _cli()
