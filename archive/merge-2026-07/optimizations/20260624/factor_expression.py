"""
因子表达式引擎 + 算子注册表（借鉴 Qlib qlib/data/ops.py 设计）

解决 jingni-trader 现有 factor-engine 的核心缺陷：
  - 因子硬编码在 compute_a_share_factors() 的 Python 代码中
  - 不可序列化、不可缓存、不可审计、不可配置化

借鉴来源：
  - Qlib 表达式引擎: https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py
  - 算子基类层级: ElemOperator / PairOperator / Rolling / PairRolling
  - get_extended_window_size() 边界处理机制
  - Alpha158 因子族配置: qlib/contrib/data/handler.py

设计要点：
  1. 字符串表达式即因子：`Ref($close, 20) / $close` 可解析、可序列化、可缓存
  2. 算子注册表：算子按名称注册，表达式引擎按名查找
  3. get_extended_window_size()：每个算子声明需要向前回看多少天，保证滚动计算边界正确
  4. 因子族配置化：get_feature_config() 返回 {group: [expr_list]}，因子族可插拔
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("opt-factor-expr")


# ── 表达式 AST 节点基类（借鉴 Qlib ExpressionOps）──────────
class Expression(ABC):
    """表达式节点抽象基类"""

    @abstractmethod
    def load(self, data: pd.DataFrame, code_col: str = "code") -> pd.Series:
        """
        在数据上求值，返回 (code, date) -> value 的 Series

        参数:
            data: 含 code, date 及 OHLCV 列的 DataFrame（已按 code, date 排序）
        返回:
            pd.Series，索引为原始 DataFrame 的 index
        """

    @abstractmethod
    def get_extended_window_size(self) -> Tuple[int, int]:
        """
        声明该表达式需要向前/向后回看的天数（借鉴 Qlib）

        返回: (left_extend, right_extend)
          - left_extend: 需要向前回看的天数（滚动窗口大小）
          - right_extend: 需要向后回看的天数（通常为 0，标签类因子为负）

        引擎据此裁剪数据边界，保证滚动计算不会产生截断偏差。
        """

    def __str__(self) -> str:  # 序列化用
        return self._str()

    @abstractmethod
    def _str(self) -> str:
        ...


class Feature(Expression):
    """字段引用，如 $close / $open / $volume"""

    def __init__(self, field: str):
        # $close -> close
        self.field = field.lstrip("$")

    def load(self, data: pd.DataFrame, code_col: str = "code") -> pd.Series:
        if self.field not in data.columns:
            raise KeyError(f"字段 {self.field} 不在数据列中: {list(data.columns)}")
        return data[self.field]

    def get_extended_window_size(self) -> Tuple[int, int]:
        return (0, 0)

    def _str(self) -> str:
        return f"${self.field}"


class Constant(Expression):
    """常量"""

    def __init__(self, value: float):
        self.value = float(value)

    def load(self, data: pd.DataFrame, code_col: str = "code") -> pd.Series:
        return pd.Series(self.value, index=data.index)

    def get_extended_window_size(self) -> Tuple[int, int]:
        return (0, 0)

    def _str(self) -> str:
        return str(self.value)


# ── 算子基类层级（借鉴 Qlib ElemOperator / PairOperator / Rolling）──
class ElemOperator(Expression):
    """单目元素级算子：如 Abs, Sign, Neg"""

    def __init__(self, operand: Expression):
        self.operand = operand

    def get_extended_window_size(self) -> Tuple[int, int]:
        return self.operand.get_extended_window_size()

    def _str(self) -> str:
        return f"{self.op_name()}({self.operand})"

    @classmethod
    def op_name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def _apply(self, series: pd.Series) -> pd.Series:
        ...

    def load(self, data: pd.DataFrame, code_col: str = "code") -> pd.Series:
        return self._apply(self.operand.load(data, code_col))


class PairOperator(Expression):
    """双目算子：如 Add, Sub, Mul, Div"""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def get_extended_window_size(self) -> Tuple[int, int]:
        l_l, l_r = self.left.get_extended_window_size()
        r_l, r_r = self.right.get_extended_window_size()
        return (max(l_l, r_l), max(l_r, r_r))

    def _str(self) -> str:
        return f"{self.op_name()}({self.left}, {self.right})"

    @classmethod
    def op_name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def _apply(self, left: pd.Series, right: pd.Series) -> pd.Series:
        ...

    def load(self, data: pd.DataFrame, code_col: str = "code") -> pd.Series:
        return self._apply(self.left.load(data, code_col), self.right.load(data, code_col))


class Rolling(Expression):
    """
    时序滚动算子：如 Ref, Mean, Std, Max, Min, Quantile
    按 code 分组滚动计算
    """

    def __init__(self, operand: Expression, window):
        self.operand = operand
        # window 可能是 Constant（来自解析器）或 int，统一转为 int
        if isinstance(window, Constant):
            self.window = int(window.value)
        elif isinstance(window, (int, float)):
            self.window = int(window)
        else:
            raise TypeError(f"Rolling window 必须是常量, 实际: {type(window)}")

    def get_extended_window_size(self) -> Tuple[int, int]:
        l, r = self.operand.get_extended_window_size()
        return (l + self.window, r)

    def _str(self) -> str:
        return f"{self.op_name()}({self.operand}, {self.window})"

    @classmethod
    def op_name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def _rolling_apply(self, grouped: pd.core.groupby.SeriesGroupBy) -> pd.Series:
        ...

    def load(self, data: pd.DataFrame, code_col: str = "code") -> pd.Series:
        operand_series = self.operand.load(data, code_col)
        # 按 code 分组滚动
        grouped = operand_series.groupby(data[code_col])
        result = self._rolling_apply(grouped)
        # groupby 后结果按 code 分组重排，需 reindex 回原始顺序
        return result.reindex(data.index)


class PairRolling(Expression):
    """
    截面/双序列滚动算子：如 Corr, Cov, Slope
    两个序列按 code 分组滚动计算
    """

    def __init__(self, left: Expression, right: Expression, window):
        self.left = left
        self.right = right
        # window 可能是 Constant（来自解析器）或 int，统一转为 int
        if isinstance(window, Constant):
            self.window = int(window.value)
        elif isinstance(window, (int, float)):
            self.window = int(window)
        else:
            raise TypeError(f"PairRolling window 必须是常量, 实际: {type(window)}")

    def get_extended_window_size(self) -> Tuple[int, int]:
        l_l, l_r = self.left.get_extended_window_size()
        r_l, r_r = self.right.get_extended_window_size()
        return (max(l_l, r_l) + self.window, max(l_r, r_r))

    def _str(self) -> str:
        return f"{self.op_name()}({self.left}, {self.right}, {self.window})"

    @classmethod
    def op_name(cls) -> str:
        return cls.__name__

    @abstractmethod
    def _rolling_apply(self, left_grouped, right_grouped) -> pd.Series:
        ...

    def load(self, data: pd.DataFrame, code_col: str = "code") -> pd.Series:
        left_series = self.left.load(data, code_col)
        right_series = self.right.load(data, code_col)
        left_grouped = left_series.groupby(data[code_col])
        right_grouped = right_series.groupby(data[code_col])
        result = self._rolling_apply(left_grouped, right_grouped)
        # 重排回原始顺序
        return result.reindex(data.index)


# ── 具体算子实现 ───────────────────────────────────────────
class Ref(Rolling):
    """引用 N 天前的值：Ref($close, 5) = 5 日前的收盘价"""

    def _rolling_apply(self, grouped):
        return grouped.shift(self.window)


class Mean(Rolling):
    """滚动均值"""

    def _rolling_apply(self, grouped):
        return grouped.rolling(self.window, min_periods=max(1, self.window // 2)).mean().reset_index(level=0, drop=True)


class Std(Rolling):
    """滚动标准差"""

    def _rolling_apply(self, grouped):
        return grouped.rolling(self.window, min_periods=max(2, self.window // 2)).std().reset_index(level=0, drop=True)


class Max(Rolling):
    """滚动最大值"""

    def _rolling_apply(self, grouped):
        return grouped.rolling(self.window, min_periods=1).max().reset_index(level=0, drop=True)


class Min(Rolling):
    """滚动最小值"""

    def _rolling_apply(self, grouped):
        return grouped.rolling(self.window, min_periods=1).min().reset_index(level=0, drop=True)


class Sum(Rolling):
    """滚动求和"""

    def _rolling_apply(self, grouped):
        return grouped.rolling(self.window, min_periods=1).sum().reset_index(level=0, drop=True)


class Var(Rolling):
    """滚动方差"""

    def _rolling_apply(self, grouped):
        return grouped.rolling(self.window, min_periods=max(2, self.window // 2)).var().reset_index(level=0, drop=True)


class Rank(Rolling):
    """滚动时序排名（pct）"""

    def _rolling_apply(self, grouped):
        return grouped.rolling(self.window, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        ).reset_index(level=0, drop=True)


class Delta(ElemOperator):
    """差分：Delta(x) = x - Ref(x, 1)"""

    def __init__(self, operand: Expression):
        super().__init__(operand)
        self._ref = Ref(operand, 1)

    def get_extended_window_size(self) -> Tuple[int, int]:
        l, r = self._ref.get_extended_window_size()
        return (l, r)

    def _apply(self, series: pd.Series) -> pd.Series:
        # 由 load 覆盖，此处不会调用，仅为满足抽象基类
        return series

    def load(self, data: pd.DataFrame, code_col: str = "code") -> pd.Series:
        cur = self.operand.load(data, code_col)
        prev = self._ref.load(data, code_col)
        return cur - prev

    def _str(self) -> str:
        return f"Delta({self.operand})"


class Abs(ElemOperator):
    def _apply(self, series):
        return series.abs()


class Sign(ElemOperator):
    def _apply(self, series):
        return np.sign(series)


class Neg(ElemOperator):
    def _apply(self, series):
        return -series


class Add(PairOperator):
    def _apply(self, left, right):
        return left + right


class Sub(PairOperator):
    def _apply(self, left, right):
        return left - right


class Mul(PairOperator):
    def _apply(self, left, right):
        return left * right


class Div(PairOperator):
    def _apply(self, left, right):
        return left / right.replace(0, np.nan)


class Corr(PairRolling):
    """滚动相关系数"""

    def _rolling_apply(self, left_grouped, right_grouped):
        results = []
        for (code, l_series), (_, r_series) in zip(left_grouped, right_grouped):
            df = pd.DataFrame({"l": l_series, "r": r_series})
            res = df["l"].rolling(self.window, min_periods=max(2, self.window // 2)).corr(df["r"])
            results.append(res)
        return pd.concat(results)


class Cov(PairRolling):
    """滚动协方差"""

    def _rolling_apply(self, left_grouped, right_grouped):
        results = []
        for (code, l_series), (_, r_series) in zip(left_grouped, right_grouped):
            df = pd.DataFrame({"l": l_series, "r": r_series})
            res = df["l"].rolling(self.window, min_periods=max(2, self.window // 2)).cov(df["r"])
            results.append(res)
        return pd.concat(results)


class Slope(PairRolling):
    """滚动回归斜率（y 对 x）"""

    def _rolling_apply(self, left_grouped, right_grouped):
        results = []
        x_idx = np.arange(self.window)
        x_mean = x_idx.mean()
        x_var = ((x_idx - x_mean) ** 2).sum()
        for (code, l_series), (_, r_series) in zip(left_grouped, right_grouped):
            df = pd.DataFrame({"y": l_series, "x": r_series}).reset_index(drop=True)
            def _slope(window):
                if len(window) < self.window:
                    return np.nan
                y = window["y"].values
                x = window["x"].values
                if np.std(x) == 0:
                    return np.nan
                return np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x)
            res = df.rolling(self.window).apply(_slope, raw=False)
            results.append(res.set_axis(l_series.index))
        return pd.concat(results)


# ── 算子注册表 ─────────────────────────────────────────────
class OperatorRegistry:
    """
    算子注册表（借鉴 Qlib OpsWrangler）

    算子按名称注册，表达式解析器按名查找。
    支持运行时扩展新算子。
    """

    _operators: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, op_class: type):
        if not issubclass(op_class, Expression):
            raise TypeError(f"{op_class} 必须是 Expression 子类")
        cls._operators[name] = op_class
        logger.debug(f"注册算子: {name} -> {op_class.__name__}")

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        return cls._operators.get(name)

    @classmethod
    def list_operators(cls) -> List[str]:
        return sorted(cls._operators.keys())

    @classmethod
    def register_defaults(cls):
        """注册内置算子集"""
        defaults = {
            "Ref": Ref, "Mean": Mean, "Std": Std, "Max": Max, "Min": Min,
            "Sum": Sum, "Var": Var, "Rank": Rank, "Delta": Delta,
            "Abs": Abs, "Sign": Sign, "Neg": Neg,
            "Add": Add, "Sub": Sub, "Mul": Mul, "Div": Div,
            "Corr": Corr, "Cov": Cov, "Slope": Slope,
        }
        for name, op_cls in defaults.items():
            cls.register(name, op_cls)


# 启动时注册默认算子
OperatorRegistry.register_defaults()


# ── 表达式解析器 ───────────────────────────────────────────
class ExpressionParser:
    """
    表达式解析器（借鉴 Qlib 字符串表达式语法）

    支持语法：
      - 字段引用: $close, $open, $volume
      - 常量: 5, 20, 0.5
      - 函数调用: Ref($close, 20), Mean($volume, 10)
      - 二元运算: $close / Ref($close, 20)  (自动转 Div)
      - 嵌套: Mean(Ref($close, 5), 20)

    示例:
      "Ref($close, 20) / $close"  → 20 日收益率
      "Mean($volume, 10)"          → 10 日均量
      "Corr($close, $volume, 20)"  → 20 日价量相关性
    """

    # 二元运算符映射
    BINARY_OPS = {
        "+": Add, "-": Sub, "*": Mul, "/": Div,
    }

    def __init__(self, registry: type = OperatorRegistry):
        self.registry = registry

    def parse(self, expr: str) -> Expression:
        """解析表达式字符串为 AST"""
        self._tokens = self._tokenize(expr)
        self._pos = 0
        node = self._parse_expr()
        if self._pos < len(self._tokens):
            raise SyntaxError(f"未消费的 token: {self._tokens[self._pos:]}")
        return node

    def _tokenize(self, expr: str) -> List[str]:
        """词法分析"""
        tokens = []
        i = 0
        while i < len(expr):
            c = expr[i]
            if c.isspace():
                i += 1
                continue
            if c in "()+-*/,":
                tokens.append(c)
                i += 1
            elif c == "$":
                # 字段引用 $close
                j = i + 1
                while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                    j += 1
                tokens.append(expr[i:j])
                i = j
            elif c.isalpha() or c == "_":
                # 函数名
                j = i
                while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                    j += 1
                tokens.append(expr[i:j])
                i = j
            elif c.isdigit() or c == ".":
                # 数字
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                    j += 1
                tokens.append(expr[i:j])
                i = j
            else:
                raise SyntaxError(f"无法识别的字符: {c} (位置 {i})")
        return tokens

    def _parse_expr(self) -> Expression:
        """解析表达式（处理二元运算，左结合）"""
        left = self._parse_term()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in self.BINARY_OPS:
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_term()
            left = self.BINARY_OPS[op](left, right)
        return left

    def _parse_term(self) -> Expression:
        """解析项（处理 * /）"""
        left = self._parse_factor()
        while self._pos < len(self._tokens) and self._tokens[self._pos] in ("*", "/"):
            op = self._tokens[self._pos]
            self._pos += 1
            right = self._parse_factor()
            left = self.BINARY_OPS[op](left, right)
        return left

    def _parse_factor(self) -> Expression:
        """解析因子（原子）"""
        if self._pos >= len(self._tokens):
            raise SyntaxError("意外的表达式结束")

        token = self._tokens[self._pos]

        if token == "(":
            self._pos += 1
            node = self._parse_expr()
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ")":
                raise SyntaxError("缺少右括号 )")
            self._pos += 1
            return node

        if token.startswith("$"):
            self._pos += 1
            return Feature(token)

        # 数字
        try:
            val = float(token)
            self._pos += 1
            return Constant(val)
        except ValueError:
            pass

        # 函数调用
        if token in self.registry.list_operators():
            self._pos += 1
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != "(":
                raise SyntaxError(f"函数 {token} 后必须跟 (")
            self._pos += 1
            args = []
            if self._tokens[self._pos] != ")":
                args.append(self._parse_expr())
                while self._pos < len(self._tokens) and self._tokens[self._pos] == ",":
                    self._pos += 1
                    args.append(self._parse_expr())
            if self._pos >= len(self._tokens) or self._tokens[self._pos] != ")":
                raise SyntaxError(f"函数 {token} 缺少右括号 )")
            self._pos += 1
            op_class = self.registry.get(token)
            return op_class(*args)

        raise SyntaxError(f"无法识别的 token: {token}")


# ── 因子族配置（借鉴 Qlib Alpha158 get_feature_config）─────
class FactorFamily:
    """
    因子族配置化定义（借鉴 Qlib Alpha158/Alpha360）

    每个因子族通过 get_feature_config() 返回 {group: [expr_list]}，
    因子可插拔、可序列化、可审计。
    """

    @staticmethod
    def alpha_a_share() -> Dict[str, List[str]]:
        """
        A 股专用因子族（对标 jingni-trader 现有 compute_a_share_factors）

        返回 {group: [expression_list]}，每个表达式可被 ExpressionParser 解析。
        """
        return {
            # 动量反转组（A 股反转效应，预期负 IC）
            "momentum": [
                "Ref($close, 20) / $close - 1",   # 20 日动量
                "Ref($close, 60) / $close - 1",   # 60 日动量
                "-(Ref($close, 5) / $close - 1)", # 5 日反转
            ],
            # 规模组
            "size": [
                "$lncap",
            ],
            # 交易组
            "turnover": [
                "Mean($turnover_rate, 20)",
                "Mean($turnover_rate, 5) / Mean($turnover_rate, 20) - 1",
            ],
            # 波动率组
            "volatility": [
                "Std($close / Ref($close, 1) - 1, 20)",
            ],
            # 量价组
            "volume": [
                "$volume / Mean($volume, 20)",
            ],
            # 资金流组
            "moneyflow": [
                "Sum($change_pct * $amount, 20)",
            ],
        }

    @staticmethod
    def alpha158_lite() -> Dict[str, List[str]]:
        """
        Alpha158 精简版（借鉴 Qlib Alpha158，6 大类）

        原始 Alpha158 含 158 个因子，这里精选代表性因子验证表达式引擎能力。
        """
        cfg = {}
        # 趋势类
        cfg["trend"] = [
            "Ref($close, 5) / $close - 1",
            "Ref($close, 10) / $close - 1",
            "Ref($close, 20) / $close - 1",
            "Ref($close, 30) / $close - 1",
            "Ref($close, 60) / $close - 1",
        ]
        # 均值回归类
        cfg["mean_revert"] = [
            "$close / Mean($close, 5) - 1",
            "$close / Mean($close, 10) - 1",
            "$close / Mean($close, 20) - 1",
            "$close / Mean($close, 30) - 1",
            "$close / Mean($close, 60) - 1",
        ]
        # 成交量类
        cfg["volume"] = [
            "$volume / Mean($volume, 5) - 1",
            "$volume / Mean($volume, 10) - 1",
            "$volume / Mean($volume, 20) - 1",
            "$volume / Mean($volume, 30) - 1",
            "$volume / Mean($volume, 60) - 1",
        ]
        # 波动率类
        cfg["volatility"] = [
            "Std($close / Ref($close, 1) - 1, 5)",
            "Std($close / Ref($close, 1) - 1, 10)",
            "Std($close / Ref($close, 1) - 1, 20)",
            "Std($close / Ref($close, 1) - 1, 30)",
            "Std($close / Ref($close, 1) - 1, 60)",
        ]
        # 资金流类
        cfg["money_flow"] = [
            "Sum($close / Ref($close, 1) - 1, 5)",
            "Sum($close / Ref($close, 1) - 1, 10)",
            "Sum($close / Ref($close, 1) - 1, 20)",
            "Sum($close / Ref($close, 1) - 1, 30)",
            "Sum($close / Ref($close, 1) - 1, 60)",
        ]
        # 复合类
        cfg["composite"] = [
            "Corr($close, $volume, 20)",
            "Corr($close, $volume, 60)",
            "Mean($close, 5) / Mean($close, 20) - 1",
            "Mean($close, 10) / Mean($close, 30) - 1",
            "Mean($close, 20) / Mean($close, 60) - 1",
        ]
        return cfg


# ── 因子引擎 ───────────────────────────────────────────────
class FactorExpressionEngine:
    """
    因子表达式引擎

    将字符串表达式批量求值为因子 DataFrame，支持：
      - 表达式解析与缓存（同表达式不重复解析）
      - 因子族批量计算
      - 因子元信息（名称、表达式、所需回看窗口）导出
    """

    def __init__(self, registry: type = OperatorRegistry):
        self.parser = ExpressionParser(registry)
        self._ast_cache: Dict[str, Expression] = {}

    def _get_ast(self, expr: str) -> Expression:
        """解析表达式（带缓存）"""
        if expr not in self._ast_cache:
            self._ast_cache[expr] = self.parser.parse(expr)
        return self._ast_cache[expr]

    def compute_factor(
        self, data: pd.DataFrame, expr: str, name: Optional[str] = None
    ) -> pd.Series:
        """
        计算单个因子

        参数:
            data: 含 code, date, OHLCV 列的 DataFrame
            expr: 因子表达式字符串
            name: 因子列名（默认用表达式字符串）
        """
        ast = self._get_ast(expr)
        values = ast.load(data)
        values.name = name or expr
        return values

    def compute_family(
        self, data: pd.DataFrame, family: Dict[str, List[str]]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        批量计算因子族

        返回:
            factor_df: 含 code, date, [各因子列] 的 DataFrame
            metadata: 每个因子的元信息（名称、表达式、分组、回看窗口）
        """
        result = data[["code", "date"]].copy()
        metadata = []

        for group, exprs in family.items():
            for expr in exprs:
                factor_name = f"{group}_{self._short_name(expr)}"
                try:
                    values = self.compute_factor(data, expr, factor_name)
                    result[factor_name] = values.values
                    ast = self._get_ast(expr)
                    left_ext, _ = ast.get_extended_window_size()
                    metadata.append({
                        "name": factor_name,
                        "expression": expr,
                        "group": group,
                        "left_extend": left_ext,
                        "ast_str": str(ast),
                    })
                except Exception as e:
                    logger.warning(f"因子 {factor_name} ({expr}) 计算失败: {e}")
                    result[factor_name] = np.nan
                    metadata.append({
                        "name": factor_name,
                        "expression": expr,
                        "group": group,
                        "error": str(e),
                    })

        return result, metadata

    @staticmethod
    def _short_name(expr: str) -> str:
        """把表达式转为简短列名"""
        # 提取关键信息：算子名 + 窗口
        import re
        ops = re.findall(r"([A-Za-z]+)\(", expr)
        nums = re.findall(r"(\d+)", expr)
        parts = ops[:2] + nums[:2]
        return "_".join(parts) if parts else "f"

    def get_factor_info(self, expr: str) -> Dict[str, Any]:
        """获取因子元信息（不计算）"""
        ast = self._get_ast(expr)
        left, right = ast.get_extended_window_size()
        return {
            "expression": expr,
            "ast_str": str(ast),
            "left_extend": left,
            "right_extend": right,
        }
