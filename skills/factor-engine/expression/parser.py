"""
因子表达式解析器
借鉴: quant-stream pyparsing-based expression language + Qlib 表达式语法
支持: 变量引用($close)、函数调用(RANK/TS_MEAN/RSI等)、嵌套表达式
"""
import re
import pandas as pd
import numpy as np
from typing import List, Optional

from .operators import OperatorRegistry


class FactorExpressionParser:
    """因子表达式解析器，将文本表达式解析为可计算的因子值"""

    VARIABLE_MAP = {
        "$close": "close",
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$volume": "volume",
        "$amount": "amount",
        "$turnover": "turnover_rate",
        "$change_pct": "change_pct",
    }

    def __init__(self, registry: Optional[OperatorRegistry] = None):
        self.registry = registry or OperatorRegistry()

    def parse_and_compute(
        self,
        expression: str,
        data: pd.DataFrame,
        date_col: str = "date",
        code_col: str = "code"
    ) -> pd.Series:
        expr = expression.strip()

        # 替换变量引用
        if "$" in expr:
            for var, col in self.VARIABLE_MAP.items():
                if col in data.columns:
                    expr = expr.replace(var, f"__{col}__")

        return self._evaluate(expr, data, date_col, code_col)

    def _evaluate(self, expr: str, data: pd.DataFrame, date_col: str, code_col: str) -> pd.Series:
        expr = expr.strip()
        outer = self._strip_outer_parens(expr)

        # 函数调用: FUNC(args)
        match = re.match(r'^([A-Z_][A-Z0-9_]*)\((.*)\)$', expr, re.IGNORECASE)
        if match:
            func_name = match.group(1)
            args = self._parse_args(match.group(2))
            evaluated = []
            for arg in args:
                arg = arg.strip()
                try:
                    val = float(arg)
                    evaluated.append(int(val) if val == int(val) else val)
                except ValueError:
                    evaluated.append(self._evaluate(arg, data, date_col, code_col))
            return self.registry.get(func_name)(*evaluated)

        # 算术运算（仅在外层处理）
        if not expr.startswith("__"):
            for op in ["+", "-", "*", "/"]:
                if op in outer:
                    parts = self._split_by_operator(expr, op)
                    if len(parts) > 1:
                        result = self._evaluate(parts[0], data, date_col, code_col)
                        for part in parts[1:]:
                            other = self._evaluate(part, data, date_col, code_col)
                            if op == "+":
                                result = result + other
                            elif op == "-":
                                result = result - other
                            elif op == "*":
                                result = result * other
                            elif op == "/":
                                result = result / other.replace(0, np.nan)
                        return result

        # 列引用
        if expr.startswith("__") and expr.endswith("__"):
            col_name = expr[2:-2]
            if col_name in data.columns:
                return data[col_name]
            raise ValueError(f"列 {col_name} 不在数据中")

        try:
            return float(expr)
        except ValueError:
            raise ValueError(f"无法解析表达式: {expr}")

    def _strip_outer_parens(self, expr: str) -> str:
        expr = expr.strip()
        if expr.startswith("(") and expr.endswith(")"):
            depth = 0
            for ch in expr[:-1]:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if depth == 0:
                    return expr
            return expr[1:-1]
        return expr

    def _split_by_operator(self, expr: str, operator: str) -> List[str]:
        parts = []
        depth = 0
        current = ""
        for ch in expr:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == operator and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += ch
        if current:
            parts.append(current)
        return [p.strip() for p in parts]

    def _parse_args(self, args_str: str) -> List[str]:
        args = []
        depth = 0
        current = ""
        for ch in args_str:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current.strip())
        return args