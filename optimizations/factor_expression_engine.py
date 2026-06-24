"""
因子表达式引擎（验证用）
========================
OPTIMIZATION 2: 用表达式字符串声明式定义因子，替代硬编码。

借鉴来源：
- Qlib 的表达式 DSL（算子 + 横截面/时序语义分离）
- WorldQuant Alpha 101 的算子集（MA/STD/TS_MAX/CORR/COV/RANK/DELTA/REF ...）
- AKQuant 的轻量解析思路

设计：
- Parser: 用 Python 标准库 ast 解析表达式字符串 -> AST（白名单校验，安全）
- Executor: 递归遍历 AST -> 计算后的 pandas.Series（与 data 索引对齐）

支持：
- 数据字段: Close/Open/High/Low/Volume/Amount/Turnover
  (映射列 close/open/high/low/volume/amount/turnover_rate)
- 算术: + - * / 与一元负号
- 函数:
  时序(组内 groupby code): MA(x,n) STD(x,n) SUM(x,n) REF(x,n) DELTA(x,n)
                           TS_MAX(x,n) TS_MIN(x,n) CORR(x,y,n) COV(x,y,n)
  横截面(groupby date):    RANK(x)
  逐元素:                  ABS(x) LOG(x)

示例: "RANK(-MA(Close, 5))" == 对 -MA(Close,5) 做横截面百分位排名
"""
from __future__ import annotations
import ast
import numpy as np
import pandas as pd


# 字段名 -> 数据列名
_FIELD_MAP = {
    "Close": "close",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Volume": "volume",
    "Amount": "amount",
    "Turnover": "turnover_rate",
}

# 函数名 -> 所需参数个数
_FUNC_ARITY = {
    "MA": 2, "STD": 2, "SUM": 2, "REF": 2, "DELTA": 2,
    "TS_MAX": 2, "TS_MIN": 2, "CORR": 3, "COV": 3,
    "RANK": 1, "ABS": 1, "LOG": 1,
}

# 允许的二元/一元算子
_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}
_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class FactorExpressionEngine:
    """
    因子表达式引擎。

    参数:
        data: 含 code/date 及 OHLCV 等列的 DataFrame；内部按 ['code','date'] 排序并重置索引
    """

    def __init__(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError("data 不能为空")
        self._data = data.sort_values(["code", "date"]).reset_index(drop=True)
        # 组内时序需要 code/date 对齐 Series
        self._code = self._data["code"]
        self._date = self._data["date"]
        # 各 code 的整数位置索引（用于双序列滚动算子）
        self._code_indices = self._data.groupby("code", sort=False).indices

    # ---------------- 对外接口 ----------------
    def evaluate(self, expr: str) -> pd.Series:
        """解析并计算表达式，返回与 data 索引对齐的 pandas.Series"""
        tree = self._parse(expr)
        result = self._eval(tree)
        # 标量结果广播为 Series
        if np.isscalar(result) or isinstance(result, (int, float, np.number)):
            result = pd.Series(float(result), index=self._data.index)
        result = pd.Series(result, index=self._data.index)
        return result

    # ---------------- 解析（白名单校验） ----------------
    def _parse(self, expr: str) -> ast.AST:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"表达式语法错误: {expr!r} ({e})")
        self._validate(tree)
        return tree

    def _validate(self, node):
        """递归白名单校验，仅允许 Name/Call/BinOp/UnaryOp/Constant"""
        if isinstance(node, ast.Expression):
            self._validate(node.body)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN_OPS:
                raise ValueError(f"不支持的二元算子: {type(node.op).__name__}")
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY_OPS:
                raise ValueError(f"不支持的一元算子: {type(node.op).__name__}")
            self._validate(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("仅支持简单函数调用")
            if node.keywords:
                raise ValueError("不支持关键字参数")
            fname = node.func.id
            if fname not in _FUNC_ARITY:
                raise ValueError(f"未知函数: {fname}")
            if len(node.args) != _FUNC_ARITY[fname]:
                raise ValueError(
                    f"函数 {fname} 参数个数应为 {_FUNC_ARITY[fname]}，实际 {len(node.args)}"
                )
            for a in node.args:
                self._validate(a)
        elif isinstance(node, ast.Name):
            if node.id not in _FIELD_MAP:
                raise ValueError(f"未知字段: {node.id}")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"仅支持数值常量: {node.value!r}")
        else:
            raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    # ---------------- 求值 ----------------
    def _eval(self, node):
        if isinstance(node, ast.Expression):
            return self._eval(node.body)

        if isinstance(node, ast.Constant):
            return node.value  # 标量

        if isinstance(node, ast.Name):
            col = _FIELD_MAP[node.id]
            if col not in self._data.columns:
                raise ValueError(f"数据缺少列: {col}")
            return self._data[col].astype(float)

        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            return _BIN_OPS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            return _UNARY_OPS[type(node.op)](operand)

        if isinstance(node, ast.Call):
            fname = node.func.id
            args = [self._eval(a) for a in node.args]
            return self._apply_func(fname, args)

        raise ValueError(f"无法求值的节点: {type(node).__name__}")

    # ---------------- 算子实现 ----------------
    def _to_series(self, x):
        """标量 -> 与 data 对齐的 Series；Series 原样返回"""
        if isinstance(x, pd.Series):
            return x
        return pd.Series(float(x), index=self._data.index)

    def _ts_rolling(self, x, n, func):
        """组内(groupby code)滚动：func 接收子 Series 返回等长子 Series"""
        x = self._to_series(x)
        return x.groupby(self._code).transform(func)

    def _ts_rolling2(self, x, y, n, how):
        """双序列组内滚动（CORR/COV），按 code 位置分块计算"""
        x = self._to_series(x)
        y = self._to_series(y)
        result = pd.Series(np.nan, index=self._data.index, dtype=float)
        xv = np.asarray(x.values, dtype=float)
        yv = np.asarray(y.values, dtype=float)
        for _code, idx in self._code_indices.items():
            xs = pd.Series(xv[idx])
            ys = pd.Series(yv[idx])
            if how == "corr":
                r = xs.rolling(n, min_periods=1).corr(ys).values
            else:  # cov
                r = xs.rolling(n, min_periods=1).cov(ys).values
            result.iloc[idx] = r
        return result

    def _apply_func(self, name, args):
        if name == "MA":
            x, n = args[0], int(args[1])
            return self._ts_rolling(x, n, lambda s: s.rolling(n, min_periods=1).mean())

        if name == "STD":
            x, n = args[0], int(args[1])
            return self._ts_rolling(x, n, lambda s: s.rolling(n, min_periods=1).std())

        if name == "SUM":
            x, n = args[0], int(args[1])
            return self._ts_rolling(x, n, lambda s: s.rolling(n, min_periods=1).sum())

        if name == "REF":
            x, n = args[0], int(args[1])
            x = self._to_series(x)
            return x.groupby(self._code).shift(n)

        if name == "DELTA":
            x, n = args[0], int(args[1])
            x = self._to_series(x)
            return x - x.groupby(self._code).shift(n)

        if name == "TS_MAX":
            x, n = args[0], int(args[1])
            return self._ts_rolling(x, n, lambda s: s.rolling(n, min_periods=1).max())

        if name == "TS_MIN":
            x, n = args[0], int(args[1])
            return self._ts_rolling(x, n, lambda s: s.rolling(n, min_periods=1).min())

        if name == "CORR":
            x, y, n = args[0], args[1], int(args[2])
            return self._ts_rolling2(x, y, n, "corr")

        if name == "COV":
            x, y, n = args[0], args[1], int(args[2])
            return self._ts_rolling2(x, y, n, "cov")

        if name == "RANK":
            # 横截面：按 date 分组百分位排名
            x = self._to_series(args[0])
            return x.groupby(self._date).rank(pct=True)

        if name == "ABS":
            x = self._to_series(args[0])
            return x.abs()

        if name == "LOG":
            x = self._to_series(args[0])
            return np.log(x)

        raise ValueError(f"未知函数: {name}")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import generate_test_data

    data, _ = generate_test_data(n_stocks=10, n_days=60, seed=1)
    eng = FactorExpressionEngine(data)

    ma5 = eng.evaluate("MA(Close, 5)")
    expect = data.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    print("MA(Close,5) max diff:", (ma5 - expect).abs().max())

    comp = eng.evaluate("RANK(-MA(Close, 5))")
    print("RANK(-MA(Close,5)) head:\n", comp.head())
