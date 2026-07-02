"""
因子表达式引擎（验证模块 3）

借鉴来源：
- AKQuant factor engine (Polars 驱动)：
    https://akquant.akfamily.xyz/en/guide/factor/
    核心思想：用户用字符串表达因子，例如 "Rank(Mean($close, 5) / Mean($close, 20))"
    引擎解析、组合、执行
- Qlib 表达式引擎 (qlib/data/ops.py)：
    支持 $open/$close/$volume + 大量算子
- WorldQuant Alpha101 风格操作符

相对 jingni-trader 现有实现的改进：
- 原实现: skills/factor-engine/engine.py::compute_a_share_factors
         因子硬编码在 if/else 链中，无法扩展
- 本实现: 声明式表达式字符串 + 算子注册表，用户可任意组合
- 新增: 算子纯函数化、易于单元测试和组合

设计目标（参考 AKQuant 文档）：
- "Concise Syntax" - 用类 Alpha101 语法
- "Extensibility"  - 用户可加自定义算子
- "Safety"        - AST 解析，避免 eval 注入
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 算子库
# ---------------------------------------------------------------------------

# ----- 时间序列算子（按 code 分组后运算） -----

def _ts_mean(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level="code").transform(
        lambda x: x.rolling(window, min_periods=max(2, window // 4)).mean()
    )

def _ts_std(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level="code").transform(
        lambda x: x.rolling(window, min_periods=max(2, window // 4)).std()
    )

def _ts_sum(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level="code").transform(
        lambda x: x.rolling(window, min_periods=max(2, window // 4)).sum()
    )

def _ts_max(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level="code").transform(
        lambda x: x.rolling(window, min_periods=max(2, window // 4)).max()
    )

def _ts_min(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level="code").transform(
        lambda x: x.rolling(window, min_periods=max(2, window // 4)).min()
    )

def _delta(s: pd.Series, period: int) -> pd.Series:
    return s.groupby(level="code").diff(period)

def _delay(s: pd.Series, period: int) -> pd.Series:
    return s.groupby(level="code").shift(period)

def _return_n(s: pd.Series, n: int) -> pd.Series:
    return s.groupby(level="code").pct_change(n)

def _decay_linear(s: pd.Series, window: int) -> pd.Series:
    """线性衰减加权平均 (Alpha101 DECAYLINEAR)"""
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def _wavg(x: pd.Series) -> pd.Series:
        if len(x) < window:
            return pd.Series(np.nan, index=x.index)
        vals = x.values
        out = np.full_like(vals, np.nan, dtype=float)
        for i in range(window - 1, len(vals)):
            out[i] = np.dot(vals[i - window + 1: i + 1], weights)
        return pd.Series(out, index=x.index)

    return s.groupby(level="code").transform(_wavg)

def _ts_rank(s: pd.Series, window: int) -> pd.Series:
    """Time-series rank: 当前值在过去 window 内的分位"""
    return s.groupby(level="code").transform(
        lambda x: x.rolling(window, min_periods=max(2, window // 4)).rank(pct=True)
    )


# ----- 截面算子（按 date 分组后运算） -----

def _rank(s: pd.Series) -> pd.Series:
    return s.groupby(level="date").rank(pct=True)

def _scale(s: pd.Series, a: float = 1.0) -> pd.Series:
    return s.groupby(level="date").transform(
        lambda x: a * x / x.abs().sum() if x.abs().sum() > 0 else x
    )

def _zscore(s: pd.Series) -> pd.Series:
    return s.groupby(level="date").transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0
    )

def _normalize(s: pd.Series) -> pd.Series:
    return s.groupby(level="date").transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else x * 0
    )


# ----- 数学/逻辑算子 -----

def _sign(s: pd.Series) -> pd.Series:
    return np.sign(s)

def _log1p(s: pd.Series) -> pd.Series:
    return np.log1p(s.abs()) * np.sign(s)

def _abs(s: pd.Series) -> pd.Series:
    return s.abs()

def _signed_power(s: pd.Series, e: float) -> pd.Series:
    return np.power(s.abs(), e) * np.sign(s)

def _where(cond: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(np.where(cond, a, b), index=a.index)


# ---------------------------------------------------------------------------
# 表达式求值器
# ---------------------------------------------------------------------------

# 变量映射: $close, $open, $high, $low, $volume, $amount, $vwap
VAR_REGEX = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")

# 函数调用正则
FUNC_REGEX = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)")

# 已注册的算子
OPERATORS: Dict[str, Callable] = {
    # ts
    "Ts_Mean": lambda s, w: _ts_mean(s, int(w)),
    "Ts_Std":  lambda s, w: _ts_std(s, int(w)),
    "Ts_Sum":  lambda s, w: _ts_sum(s, int(w)),
    "Ts_Max":  lambda s, w: _ts_max(s, int(w)),
    "Ts_Min":  lambda s, w: _ts_min(s, int(w)),
    "Ts_Rank": lambda s, w: _ts_rank(s, int(w)),
    "Decay_Linear": lambda s, w: _decay_linear(s, int(w)),
    "Delta":  lambda s, w: _delta(s, int(w)),
    "Delay":  lambda s, w: _delay(s, int(w)),
    "Return": lambda s, w: _return_n(s, int(w)),
    # cross-section
    "Rank":  lambda s: _rank(s),
    "Scale": lambda s, a=1.0: _scale(s, float(a)),
    "ZScore": lambda s: _zscore(s),
    "Normalize": lambda s: _normalize(s),
    # math
    "Sign":  lambda s: _sign(s),
    "Log1p": lambda s: _log1p(s),
    "Abs":   lambda s: _abs(s),
    "SignedPower": lambda s, e: _signed_power(s, float(e)),
    "Where": lambda c, a, b: _where(c, a, b),
}


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class FactorExpressionEngine:
    """
    因子表达式引擎

    用法:
        engine = FactorExpressionEngine(data_df)
        f = engine.compute("Rank(Return($close, 20))")
        # 一次性计算多因子
        result = engine.compute_batch([
            "Rank(Ts_Mean($close, 5))",
            "Sign($close - Delay($close, 1))",
            "Decay_Linear($volume, 10)",
        ])
    """

    def __init__(self, data: pd.DataFrame):
        """
        参数:
            data: 必须包含 ['date', 'code', 'open', 'high', 'low', 'close', 'volume']
                  可选: 'amount', 'vwap'
        """
        self.data = data.copy()
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data = self.data.sort_values(["date", "code"]).set_index(
            ["date", "code"]
        ).sort_index()

        # 暴露变量
        self.variables: Dict[str, pd.Series] = {}
        for col in self.data.columns:
            self.variables[f"${col}"] = self.data[col]

    def compute(self, expression: str) -> pd.Series:
        """
        计算单个因子

        支持的语法:
            $close                    # 字段引用
            Ts_Mean($close, 5)        # 算子调用
            Rank(Return($close, 5))   # 嵌套
            Ts_Mean($close, 5) - Ts_Mean($close, 20)  # 算术
            -1 * Sign($close - Delay($close, 1))      # 复合
        """
        if not expression or not expression.strip():
            raise ValueError("表达式不能为空")

        # 安全检查：禁止 eval
        if any(kw in expression for kw in ["__", "import", "exec", "eval", "open("]):
            raise ValueError(f"表达式包含不安全关键字: {expression}")

        try:
            return self._eval(expression)
        except (ValueError, KeyError, TypeError) as e:
            raise
        except Exception as e:
            raise RuntimeError(f"求值失败 '{expression}': {e}") from e

    def compute_batch(self, expressions: List[str]) -> pd.DataFrame:
        """批量计算多个因子"""
        out = pd.DataFrame(index=self.data.index)
        out["date"] = out.index.get_level_values("date")
        out["code"] = out.index.get_level_values("code")
        for expr in expressions:
            name = self._expr_to_name(expr)
            out[name] = self.compute(expr).values
        return out.reset_index(drop=True)

    @staticmethod
    def _expr_to_name(expr: str) -> str:
        """把表达式转为安全列名"""
        s = re.sub(r"[^A-Za-z0-9_]", "_", expr)
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:60]  # 截断

    # ----- 内部求值：先预处理 $var 语法，再 AST -----

    # 预编译：$identifier → _VAR_identifier（让 Python AST 能解析）
    _VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

    @classmethod
    def _preprocess(cls, expr: str) -> str:
        """把 $close → _VAR_close，让其符合 Python 语法"""
        return cls._VAR_RE.sub(r"_VAR_\1", expr)

    def _eval(self, expr: str) -> pd.Series:
        """
        统一入口：先把 $var 转成 _VAR_var，再用 AST 解析
        """
        expr = expr.strip()
        if not expr:
            raise ValueError("空表达式")
        pre = self._preprocess(expr)
        try:
            tree = ast.parse(pre, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"表达式语法错误: {e}")
        return self._eval_ast(tree.body)

    def _resolve_var(self, token: str) -> pd.Series:
        token = token.strip()
        if token in self.variables:
            return self.variables[token]
        raise KeyError(f"未知变量: {token}")

    def _eval_ast(self, node) -> object:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return v
            raise ValueError(f"不支持的字面量: {v!r}")
        if isinstance(node, ast.Name):
            # _VAR_xxx → 还原为 $xxx
            name = node.id
            if name.startswith("_VAR_"):
                return self._resolve_var(f"${name[5:]}")
            # 算子也走 Name 节点
            if name in OPERATORS:
                # 单独引用算子没有参数，抛错
                raise ValueError(f"算子 {name} 缺少参数")
            # 否则视为变量
            return self._resolve_var(f"${name}")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_ast(node.operand)
            # 数字也支持一元运算
            if isinstance(operand, (int, float)):
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return operand
            if isinstance(operand, pd.Series):
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return operand
            raise ValueError(f"一元运算仅支持 Series/数字，当前 {type(operand).__name__}")
        if isinstance(node, ast.BinOp):
            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)
            # 标量 + Series 广播
            left_series = isinstance(left, pd.Series)
            right_series = isinstance(right, pd.Series)
            if not (left_series or right_series):
                raise ValueError("二元运算至少需要一边是 Series")
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError(f"不支持的二元运算: {type(node.op).__name__}")
        if isinstance(node, ast.Call):
            # 函数调用 → 找算子
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in OPERATORS:
                    args = [self._eval_ast(a) for a in node.args]
                    return OPERATORS[fname](*args)
                raise ValueError(f"未知算子: {fname}")
            raise ValueError(f"不支持的调用形式: {ast.dump(node.func)}")
        raise ValueError(f"AST 节点未支持: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# Alpha101 风格示例集
# ---------------------------------------------------------------------------

ALPHA101_DEMO: List[str] = [
    "Rank(Return($close, 20))",                                  # 20日反转
    "Rank(Ts_Mean($amount, 5) / Ts_Mean($amount, 20))",           # 量比
    "Sign($close - Delay($close, 1))",                           # 1日动量
    "Rank(-1 * Ts_Std($close, 20))",                             # 低波动
    "ZScore(Ts_Mean($close, 5) - Ts_Mean($close, 20))",          # 均线偏离
    "Rank(Decay_Linear($volume, 10))",                           # 衰减量能
    "Ts_Rank($close, 10) - 0.5",                                 # 时序分位
]