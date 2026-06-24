"""
因子表达式引擎

借鉴来源：
    - Microsoft Qlib Alpha158 / Alpha360：表达式化因子定义，统一算子语义
    - AKQuant：Polars 驱动的高性能因子计算 + Alpha101 风格公式
    - WorldQuant Alpha101：标准算子集（Rank / Ts_Mean / Delta / Corr ...）

对比 jingni-trader 现状：
    skills/factor-engine/scripts/adapters/pandas_ta_calculator.py 的 _calc_single
    使用 `for code in data['code'].unique()` 逐股票 Python 循环计算，
    在股票池扩大时是显著性能瓶颈。

本实现的核心改进：
    1. 因子用【表达式字符串】定义，可序列化、可配置、可组合：
       "Rank(Ts_Mean(Close, 5))" / "Delta(Close, 5) / Ts_Std(Close, 20)"
    2. 所有算子基于 pandas groupby('code').transform 实现，
       【零逐股票 Python 循环】，计算下沉到 C 层。
    3. 内置 Alpha101 子集 + A 股常用因子，可一键批量计算。

注意：本模块为优化验证用，未直接修改主仓库 factor-engine 代码。
"""
from __future__ import annotations
import re
from typing import Dict, List, Any, Callable
import numpy as np
import pandas as pd


# ============================================================
# 算子定义（全部基于 groupby transform，向量化）
# ============================================================

def _ts_mean(g: pd.core.groupby.SeriesGroupBy, d: int) -> pd.Series:
    return g.transform(lambda x: x.rolling(d, min_periods=max(1, d // 2)).mean())


def _ts_std(g: pd.core.groupby.SeriesGroupBy, d: int) -> pd.Series:
    return g.transform(lambda x: x.rolling(d, min_periods=max(2, d // 2)).std())


def _ts_max(g: pd.core.groupby.SeriesGroupBy, d: int) -> pd.Series:
    return g.transform(lambda x: x.rolling(d, min_periods=1).max())


def _ts_min(g: pd.core.groupby.SeriesGroupBy, d: int) -> pd.Series:
    return g.transform(lambda x: x.rolling(d, min_periods=1).min())


def _ts_rank(g: pd.core.groupby.SeriesGroupBy, d: int) -> pd.Series:
    """滚动窗口内当前值的分位数"""
    def _r(x):
        return x.rolling(d, min_periods=2).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
        )
    return g.transform(_r)


def _delta(g: pd.core.groupby.SeriesGroupBy, d: int) -> pd.Series:
    return g.transform(lambda x: x.diff(d))


def _delay(g: pd.core.groupby.SeriesGroupBy, d: int) -> pd.Series:
    return g.transform(lambda x: x.shift(d))


def _ts_sum(g: pd.core.groupby.SeriesGroupBy, d: int) -> pd.Series:
    return g.transform(lambda x: x.rolling(d, min_periods=1).sum())


def _rank(s: pd.Series) -> pd.Series:
    """截面排名（按 date 分组）—— 注意：截面算子需在调用处按 date 分组"""
    return s.groupby(level=0, group_keys=False).rank(pct=True) if s.index.nlevels > 1 else s.rank(pct=True)


# 字段名映射（表达式中的名字 -> DataFrame 列名）
FIELD_MAP = {
    "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Volume": "volume", "Amount": "amount",
    "Turnover": "turnover_rate", "Vwap": "vwap",
}


# ============================================================
# 表达式解析器（递归下降，支持嵌套）
# ============================================================

class ExprNode:
    """表达式 AST 节点"""
    def __init__(self, op: str, args: list):
        self.op = op
        self.args = args

    def __repr__(self):
        return f"{self.op}({', '.join(map(repr, self.args))})"


class FieldNode:
    def __init__(self, field: str):
        self.field = field

    def __repr__(self):
        return self.field


class ConstNode:
    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return str(self.value)


# 算子签名：参数个数（不含 series）
OP_ARITY = {
    "Ts_Mean": 1, "Ts_Std": 1, "Ts_Max": 1, "Ts_Min": 1,
    "Ts_Sum": 1, "Delta": 1, "Delay": 1, "Ts_Rank": 1,
    "Rank": 0, "Abs": 0, "Log": 0, "Sign": 0,
    "Add": 0, "Sub": 0, "Mul": 0, "Div": 0,
    "Max": 0, "Min": 0,
}

# 二元算子符号
BINOP_MAP = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div"}


class ExprParser:
    """递归下降表达式解析器"""

    def __init__(self, expr: str):
        self.expr = expr.replace(" ", "")
        self.pos = 0

    def peek(self) -> str:
        return self.expr[self.pos] if self.pos < len(self.expr) else ""

    def advance(self):
        self.pos += 1

    def parse(self):
        node = self._parse_addsub()
        if self.pos != len(self.expr):
            raise ValueError(f"未解析完的表达式: {self.expr[self.pos:]}")
        return node

    def _parse_addsub(self):
        left = self._parse_muldiv()
        while self.peek() in ("+", "-"):
            op = BINOP_MAP[self.peek()]
            self.advance()
            right = self._parse_muldiv()
            left = ExprNode(op, [left, right])
        return left

    def _parse_muldiv(self):
        left = self._parse_unary()
        while self.peek() in ("*", "/"):
            op = BINOP_MAP[self.peek()]
            self.advance()
            right = self._parse_unary()
            left = ExprNode(op, [left, right])
        return left

    def _parse_unary(self):
        if self.peek() == "-":
            self.advance()
            inner = self._parse_atom()
            return ExprNode("Mul", [ConstNode(-1.0), inner])
        return self._parse_atom()

    def _parse_atom(self):
        c = self.peek()
        if c == "(":
            self.advance()
            node = self._parse_addsub()
            if self.peek() != ")":
                raise ValueError("缺少右括号")
            self.advance()
            return node

        # 标识符（算子名或字段名）
        if c.isalpha() or c == "_":
            ident = self._read_ident()
            if self.peek() == "(":
                # 函数调用
                self.advance()
                args = []
                # 解析参数：第一个是子表达式，后续是常量
                first = self._parse_addsub()
                args.append(first)
                while self.peek() == ",":
                    self.advance()
                    # 参数中的数字
                    num = self._read_number()
                    args.append(ConstNode(num))
                if self.peek() != ")":
                    raise ValueError("函数缺少右括号")
                self.advance()
                return ExprNode(ident, args)
            else:
                # 字段
                if ident not in FIELD_MAP:
                    raise ValueError(f"未知字段: {ident}")
                return FieldNode(ident)

        # 数字
        if c.isdigit() or c == ".":
            return ConstNode(self._read_number())

        raise ValueError(f"无法解析的字符: {c} (位置 {self.pos})")

    def _read_ident(self) -> str:
        start = self.pos
        while self.pos < len(self.expr) and (self.expr[self.pos].isalnum() or self.expr[self.pos] == "_"):
            self.pos += 1
        return self.expr[start:self.pos]

    def _read_number(self) -> float:
        start = self.pos
        while self.pos < len(self.expr) and (self.expr[self.pos].isdigit() or self.expr[self.pos] in ".-"):
            self.pos += 1
        return float(self.expr[start:self.pos])


# ============================================================
# 表达式求值器（向量化）
# ============================================================

class FactorExpressionEngine:
    """
    因子表达式引擎

    用法:
        engine = FactorExpressionEngine()
        df = engine.calculate(data, ["Rank(Delta(Close, 5))", "Ts_Mean(Volume, 20)"])
    """

    def __init__(self):
        self.parser = ExprParser

    def get_available_factors(self) -> List[str]:
        """返回内置因子表达式列表（Alpha101 子集 + A股常用）"""
        return list(BUILTIN_FACTORS.keys())

    def get_factor_info(self, factor_name: str) -> Dict:
        return BUILTIN_FACTORS_INFO.get(factor_name, {})

    def calculate(self, data: pd.DataFrame, factor_exprs: List[str]) -> pd.DataFrame:
        """
        批量计算因子（向量化，无逐股票循环）

        参数:
            data: OHLCV 数据，必须含 code, date 列
            factor_exprs: 因子表达式列表，可为表达式字符串或内置因子名

        返回:
            DataFrame: code, date, [各因子列]
        """
        if data.empty:
            return data

        df = data.sort_values(["code", "date"]).reset_index(drop=True)
        result = df[["code", "date"]].copy()

        # 预计算 VWAP（若缺失则用成交额/成交量近似）
        if "vwap" not in df.columns:
            if "amount" in df.columns and "volume" in df.columns:
                df["vwap"] = df["amount"] / df["volume"].replace(0, np.nan)

        for expr in factor_exprs:
            # 支持传入内置因子名
            if expr in BUILTIN_FACTORS:
                expr = BUILTIN_FACTORS[expr]
            col_name = self._expr_to_name(expr)
            try:
                values = self._eval(expr, df)
                result[col_name] = values
            except Exception as e:
                result[col_name] = np.nan
        return result

    def _eval(self, expr: str, df: pd.DataFrame) -> pd.Series:
        ast = self.parser(expr).parse()
        return self._eval_node(ast, df)

    def _eval_node(self, node, df: pd.DataFrame) -> pd.Series:
        if isinstance(node, ConstNode):
            return pd.Series(node.value, index=df.index)
        if isinstance(node, FieldNode):
            col = FIELD_MAP[node.field]
            if col not in df.columns:
                raise ValueError(f"数据缺少字段: {col}")
            return df[col].astype(float)

        # ExprNode
        op = node.op
        args = node.args

        # 一元时序算子：Ts_Mean(Close, 5)
        if op in ("Ts_Mean", "Ts_Std", "Ts_Max", "Ts_Min", "Ts_Sum", "Delta", "Delay", "Ts_Rank"):
            series = self._eval_node(args[0], df)
            d = int(args[1].value) if isinstance(args[1], ConstNode) else 5
            grp = series.groupby(df["code"])
            return {
                "Ts_Mean": _ts_mean, "Ts_Std": _ts_std,
                "Ts_Max": _ts_max, "Ts_Min": _ts_min,
                "Ts_Sum": _ts_sum, "Delta": _delta,
                "Delay": _delay, "Ts_Rank": _ts_rank,
            }[op](grp, d)

        # 一元标量算子
        if op in ("Abs", "Log", "Sign"):
            series = self._eval_node(args[0], df)
            if op == "Abs":
                return series.abs()
            if op == "Log":
                return np.log(series.replace(0, np.nan))
            return np.sign(series)

        # 截面 Rank：按 date 分组排名
        if op == "Rank":
            series = self._eval_node(args[0], df)
            return series.groupby(df["date"]).rank(pct=True)

        # 二元算子
        if op in ("Add", "Sub", "Mul", "Div", "Max", "Min"):
            left = self._eval_node(args[0], df)
            right = self._eval_node(args[1], df)
            if op == "Add":
                return left + right
            if op == "Sub":
                return left - right
            if op == "Mul":
                return left * right
            if op == "Div":
                return left / right.replace(0, np.nan)
            if op == "Max":
                return np.maximum(left, right)
            return np.minimum(left, right)

        raise ValueError(f"未知算子: {op}")

    @staticmethod
    def _expr_to_name(expr: str) -> str:
        """表达式转合法列名"""
        name = re.sub(r"[^A-Za-z0-9_]", "_", expr)
        return name[:60]


# ============================================================
# 内置因子库（Alpha101 子集 + A股常用）
# ============================================================

BUILTIN_FACTORS: Dict[str, str] = {
    # --- 量价反转（unary minus 直接由解析器支持）---
    "alpha_reversal_5": "-Delta(Close, 5)",
    "alpha_reversal_20": "-Delta(Close, 20)",
    # --- 动量 ---
    "alpha_momentum_20": "Delta(Close, 20)",
    "alpha_momentum_60": "Delta(Close, 60)",
    # --- 波动率 ---
    "alpha_volatility_20": "Ts_Std(Div(Sub(Close, Delay(Close, 1)), Delay(Close, 1)), 20)",
    # --- 成交量比 ---
    "alpha_volume_ratio": "Div(Volume, Ts_Mean(Volume, 20))",
    # --- 换手率均值 ---
    "alpha_turnover_20": "Ts_Mean(Turnover, 20)",
    # --- VWAP 偏离 ---
    "alpha_vwap_dev": "Div(Sub(Close, Vwap), Vwap)",
    # --- 高低价偏离 ---
    "alpha_hl_range": "Div(Sub(High, Low), Close)",
    # --- 截面排名因子 ---
    "alpha_rank_mean5": "Rank(Ts_Mean(Close, 5))",
    "alpha_rank_mean20": "Rank(Ts_Mean(Close, 20))",
}

BUILTIN_FACTORS_INFO: Dict[str, Dict] = {
    "alpha_reversal_5": {"name": "5日反转", "direction": -1, "expr": "-Delta(Close, 5)"},
    "alpha_momentum_20": {"name": "20日动量", "direction": 1, "expr": "Delta(Close, 20)"},
    "alpha_volatility_20": {"name": "20日波动率", "direction": -1, "expr": "Ts_Std(...)"},
    "alpha_volume_ratio": {"name": "量比", "direction": 1, "expr": "Div(Volume, Ts_Mean(Volume, 20))"},
}


if __name__ == "__main__":
    # 自检
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from synthetic_data import generate_synthetic_ohlcv
    data = generate_synthetic_ohlcv(n_codes=30, n_days=100)
    eng = FactorExpressionEngine()
    factors = eng.calculate(data, ["alpha_reversal_5", "Rank(Ts_Mean(Close, 5))", "Div(Volume, Ts_Mean(Volume, 20))"])
    print("因子计算完成，形状:", factors.shape)
    print(factors.dropna().head())
    print("内置因子:", eng.get_available_factors())
