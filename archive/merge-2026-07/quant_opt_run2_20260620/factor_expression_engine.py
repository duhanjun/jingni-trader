"""
因子表达式引擎（声明式因子定义）

借鉴来源：Microsoft Qlib Expression Engine
  - https://qlib.readthedocs.io/en/latest/component/data.html
  - https://arxiv.org/abs/2009.11189 (Qlib 论文)

设计目标：
  1. 用声明式表达式字符串定义因子，替代 jingni-trader 现有
     pandas_ta_calculator.py 中 if/elif 硬编码链
  2. 通过 groupby+transform 向量化计算，替代逐股票 for 循环
  3. 支持算子嵌套，便于组合复杂 Alpha 因子

与现有实现对比：
  - 现有: _calc_single() 对每只股票循环 data[data['code']==code]
  - 本引擎: 解析表达式为 AST -> 一次性 groupby 向量化计算

注意：本文件位于 feat/quant-opt-20260620 分支，不修改 main 分支代码。
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd


# ============================================================
# 算子定义
# ============================================================

class Operator:
    """算子基类：每个算子知道如何在 groupby 分组上计算自身"""

    def compute(self, grp: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class Field(Operator):
    """原始字段引用，如 Close / Volume"""

    def __init__(self, name: str):
        self.name = name

    def compute(self, grp: pd.DataFrame) -> pd.Series:
        col = _resolve_field(grp, self.name)
        return col.astype(float)


class Constant(Operator):
    """常量"""

    def __init__(self, value: float):
        self.value = float(value)

    def compute(self, grp: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=grp.index)


class BinOp(Operator):
    """二元算术算子"""

    _OPS: Dict[str, Callable] = {
        "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide,
        "**": np.power,
    }

    def __init__(self, op: str, left: Operator, right: Operator):
        self.op = op
        self.func = self._OPS[op]
        self.left = left
        self.right = right

    def compute(self, grp: pd.DataFrame) -> pd.Series:
        l = self.left.compute(grp)
        r = self.right.compute(grp)
        return pd.Series(self.func(l.values, r.values), index=grp.index)


class UnaryOp(Operator):
    """一元算子"""

    _OPS: Dict[str, Callable] = {
        "Abs": np.abs, "Log": np.log, "Sign": np.sign,
        "Neg": np.negative,
    }

    def __init__(self, op: str, operand: Operator):
        self.op = op
        self.func = self._OPS[op]
        self.operand = operand

    def compute(self, grp: pd.DataFrame) -> pd.Series:
        v = self.operand.compute(grp)
        return pd.Series(self.func(v.values), index=grp.index)


class Rolling(Operator):
    """滚动窗口算子：MA / STD / MAX / MIN / SUM / RANK(截面) 等

    滚动类算子按时间序列在每只股票内部计算。
    """

    def __init__(self, op: str, operand: Operator, window: int):
        self.op = op
        self.operand = operand
        self.window = int(window)

    def compute(self, grp: pd.DataFrame) -> pd.Series:
        v = self.operand.compute(grp)
        w = self.window
        if self.op == "MA":
            return v.rolling(w, min_periods=max(1, w // 2)).mean()
        if self.op == "STD":
            return v.rolling(w, min_periods=max(2, w // 2)).std()
        if self.op == "MAX":
            return v.rolling(w, min_periods=1).max()
        if self.op == "MIN":
            return v.rolling(w, min_periods=1).min()
        if self.op == "SUM":
            return v.rolling(w, min_periods=1).sum()
        if self.op == "Ref":  # Ref(x, n) = x.shift(n)
            return v.shift(w)
        if self.op == "Delta":  # Delta(x, n) = x - x.shift(n)
            return v - v.shift(w)
        if self.op == "WMA":  # 线性加权移动平均
            weights = np.arange(1, w + 1, dtype=float)
            weights /= weights.sum()
            return v.rolling(w).apply(lambda x: np.dot(x, weights), raw=True)
        raise ValueError(f"未知滚动算子: {self.op}")


class CrossSection(Operator):
    """截面算子：在同一日期的全体股票上计算

    Rank: 横截面百分位排名 (0~1)
    ZScore: 横截面标准化
    """

    def __init__(self, op: str, operand: Operator):
        self.op = op
        self.operand = operand

    def compute(self, grp: pd.DataFrame) -> pd.Series:
        # 注意：截面算子需要全市场数据，不能只在单只股票 group 上算
        # 这里返回单只股票的值，由引擎在外层做截面聚合
        return self.operand.compute(grp)


# ============================================================
# 字段名解析（兼容大小写 / 别名）
# ============================================================

_FIELD_ALIASES = {
    "open": "open", "high": "high", "low": "low", "close": "close",
    "volume": "volume", "vol": "volume", "amount": "amount",
    "vwap": "vwap", "turnover": "turnover_rate",
}


def _resolve_field(df: pd.DataFrame, name: str) -> pd.Series:
    """从 DataFrame 中解析字段，支持大小写与别名"""
    key = _FIELD_ALIASES.get(name.lower(), name.lower())
    for col in df.columns:
        if col.lower() == key:
            return df[col]
    raise KeyError(f"字段 {name} 不存在于数据列: {list(df.columns)}")


# ============================================================
# 表达式解析器（递归下降）
# ============================================================

_TOKEN_RE = re.compile(r"""
    \s*(?:
        (?P<NUMBER>\d+\.?\d*([eE][+-]?\d+)?)  |
        (?P<NAME>[A-Za-z_][A-Za-z0-9_]*)   |
        (?P<OP>\*\*|[+\-*/(),])             |
        (?P<FIELD>\$[A-Za-z_][A-Za-z0-9_]*)
    )
""", re.VERBOSE)

_ROLLING_OPS = {"MA", "STD", "MAX", "MIN", "SUM", "Ref", "Delta", "WMA"}
_UNARY_OPS = {"Abs", "Log", "Sign", "Neg"}
_CROSS_OPS = {"Rank", "ZScore"}


class Parser:
    """递归下降解析器：表达式字符串 -> Operator AST"""

    def __init__(self, expr: str):
        self.expr = expr
        self.tokens = self._tokenize(expr)
        self.pos = 0

    @staticmethod
    def _tokenize(expr: str) -> List[str]:
        tokens = []
        pos = 0
        while pos < len(expr):
            m = _TOKEN_RE.match(expr, pos)
            if not m or m.end() == pos:
                if expr[pos].isspace():
                    pos += 1
                    continue
                raise SyntaxError(f"无法解析的字符: {expr[pos]} (位置 {pos})")
            for kind in ("NUMBER", "NAME", "OP", "FIELD"):
                val = m.group(kind)
                if val is not None:
                    tokens.append(val)
                    break
            pos = m.end()
        return tokens

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> Operator:
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            raise SyntaxError(f"未消费的 token: {self.tokens[self.pos:]}")
        return node

    def _parse_expr(self) -> Operator:
        node = self._parse_term()
        while self._peek() in ("+", "-"):
            op = self._next()
            right = self._parse_term()
            node = BinOp(op, node, right)
        return node

    def _parse_term(self) -> Operator:
        node = self._parse_power()
        while self._peek() in ("*", "/"):
            op = self._next()
            right = self._parse_power()
            node = BinOp(op, node, right)
        return node

    def _parse_power(self) -> Operator:
        node = self._parse_unary()
        if self._peek() == "**":
            self._next()
            right = self._parse_power()  # 右结合
            return BinOp("**", node, right)
        return node

    def _parse_unary(self) -> Operator:
        if self._peek() == "-":
            self._next()
            return UnaryOp("Neg", self._parse_unary())
        if self._peek() == "+":
            self._next()
            return self._parse_unary()
        return self._parse_atom()

    def _parse_atom(self) -> Operator:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("表达式意外结束")
        if tok == "(":
            self._next()
            node = self._parse_expr()
            if self._peek() != ")":
                raise SyntaxError("缺少右括号 )")
            self._next()
            return node
        # $field 形式：tokenizer 把 $close 作为整体 token
        if tok.startswith("$"):
            self._next()
            return Field(tok[1:])
        # 数字
        try:
            val = float(tok)
            self._next()
            return Constant(val)
        except ValueError:
            pass
        # 名称：可能是函数调用或字段
        name = self._next()
        if self._peek() == "(":
            return self._parse_call(name)
        # 裸名称当作字段
        return Field(name)

    def _parse_call(self, name: str) -> Operator:
        self._next()  # 消费 (
        args: List[Operator] = []
        if self._peek() != ")":
            args.append(self._parse_expr())
            while self._peek() == ",":
                self._next()
                args.append(self._parse_expr())
        if self._peek() != ")":
            raise SyntaxError("函数调用缺少右括号 )")
        self._next()  # 消费 )

        if name in _ROLLING_OPS:
            if len(args) != 2:
                raise SyntaxError(f"{name} 需要 2 个参数 (字段, 窗口)")
            # 第二个参数应为常量窗口
            if not isinstance(args[1], Constant):
                raise SyntaxError(f"{name} 的窗口参数必须是常量")
            return Rolling(name, args[0], int(args[1].value))
        if name in _UNARY_OPS:
            if len(args) != 1:
                raise SyntaxError(f"{name} 需要 1 个参数")
            return UnaryOp(name, args[0])
        if name in _CROSS_OPS:
            if len(args) != 1:
                raise SyntaxError(f"{name} 需要 1 个参数")
            return CrossSection(name, args[0])
        raise SyntaxError(f"未知函数: {name}")


def parse(expr: str) -> Operator:
    """解析因子表达式，返回 AST 根节点"""
    return Parser(expr).parse()


# ============================================================
# 引擎：执行 AST，支持截面算子
# ============================================================

class FactorExpressionEngine:
    """
    因子表达式引擎

    用法:
        engine = FactorExpressionEngine()
        result = engine.calculate(data, ["MA($close, 20) - MA($close, 5)",
                                         "Rank($close / Ref($close, 5))"])
    """

    def calculate(self, data: pd.DataFrame, expressions: List[str]) -> pd.DataFrame:
        """
        批量计算因子表达式

        参数:
            data: 必须含 code, date, 以及表达式引用的字段
            expressions: 因子表达式字符串列表

        返回:
            DataFrame: code, date, [因子列]
        """
        if data.empty:
            return data[["code", "date"]].copy() if {"code", "date"}.issubset(data.columns) else data.copy()

        work = data.sort_values(["code", "date"]).reset_index(drop=True)
        out = work[["code", "date"]].copy()

        for i, expr in enumerate(expressions):
            col_name = f"factor_{i}"
            ast = parse(expr)
            out[col_name] = self._eval_with_cross_section(work, ast)

        return out

    def _eval_with_cross_section(self, data: pd.DataFrame, ast: Operator) -> pd.Series:
        """评估 AST；若含截面算子则按日期分组做截面计算"""
        if not self._has_cross_section(ast):
            # 纯时序：按股票分组计算，手动拼接保证长度/索引对齐
            # （groupby.apply 在单组时行为不稳定，改用显式循环）
            parts = []
            for _, g in data.groupby("code", sort=False):
                parts.append(ast.compute(g))
            if parts:
                result = pd.concat(parts)
                # 对齐回 data 原始顺序
                result = result.reindex(data.index)
            else:
                result = pd.Series(np.nan, index=data.index)
            return result

        # 含截面算子：先按股票算时序部分，再按日期算截面部分
        return self._eval_mixed(data, ast)

    def _has_cross_section(self, ast: Operator) -> bool:
        if isinstance(ast, CrossSection):
            return True
        if isinstance(ast, (BinOp,)):
            return self._has_cross_section(ast.left) or self._has_cross_section(ast.right)
        if isinstance(ast, UnaryOp):
            return self._has_cross_section(ast.operand)
        if isinstance(ast, Rolling):
            return self._has_cross_section(ast.operand)
        return False

    def _eval_mixed(self, data: pd.DataFrame, ast: Operator) -> pd.Series:
        """处理含截面算子的表达式（单层截面，满足常见 Alpha 因子需求）"""
        if isinstance(ast, CrossSection):
            inner = self._eval_mixed(data, ast.operand)
            tmp = data[["code", "date"]].copy()
            tmp["_v"] = inner.values
            if ast.op == "Rank":
                tmp["_out"] = tmp.groupby("date")["_v"].rank(pct=True)
            elif ast.op == "ZScore":
                g = tmp.groupby("date")["_v"]
                tmp["_out"] = (tmp["_v"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
            else:
                raise ValueError(f"未知截面算子: {ast.op}")
            return tmp["_out"].reset_index(drop=True)

        if isinstance(ast, BinOp):
            l = self._eval_mixed(data, ast.left)
            r = self._eval_mixed(data, ast.right)
            return pd.Series(ast.func(l.values, r.values), index=data.index)

        if isinstance(ast, UnaryOp):
            v = self._eval_mixed(data, ast.operand)
            return pd.Series(ast.func(v.values), index=data.index)

        if isinstance(ast, Rolling):
            inner = self._eval_mixed(data, ast.operand)
            tmp = data[["code", "date"]].copy()
            tmp["_v"] = inner.values
            res = tmp.groupby("code", group_keys=False)["_v"].apply(
                lambda s: self._rolling_apply(s, ast)
            )
            return res.reset_index(drop=True)

        # 叶子节点
        return ast.compute(data).reset_index(drop=True)

    @staticmethod
    def _rolling_apply(series: pd.Series, ast: Rolling) -> pd.Series:
        w = ast.window
        v = series
        if ast.op == "MA":
            return v.rolling(w, min_periods=max(1, w // 2)).mean()
        if ast.op == "STD":
            return v.rolling(w, min_periods=max(2, w // 2)).std()
        if ast.op == "MAX":
            return v.rolling(w, min_periods=1).max()
        if ast.op == "MIN":
            return v.rolling(w, min_periods=1).min()
        if ast.op == "SUM":
            return v.rolling(w, min_periods=1).sum()
        if ast.op == "Ref":
            return v.shift(w)
        if ast.op == "Delta":
            return v - v.shift(w)
        if ast.op == "WMA":
            weights = np.arange(1, w + 1, dtype=float)
            weights /= weights.sum()
            return v.rolling(w).apply(lambda x: np.dot(x, weights), raw=True)
        raise ValueError(f"未知滚动算子: {ast.op}")


# ============================================================
# 预置因子表达式库（Alpha101 风格子集）
# ============================================================

PRESET_FACTORS: Dict[str, str] = {
    # 动量类
    "mom_5": "Ref($close, 5) / $close - 1",          # 5日收益率
    "mom_20": "Ref($close, 20) / $close - 1",        # 20日收益率
    "reversal_5": "-($close / Ref($close, 5) - 1)",  # 5日反转
    # 均线偏离
    "ma_bias_20": "$close / MA($close, 20) - 1",     # 收盘价相对20日均线偏离
    "ma_cross": "MA($close, 5) - MA($close, 20)",    # 双均线差
    # 波动率
    "vol_20": "STD($close / Ref($close, 1) - 1, 20)",  # 20日收益率波动
    # 量价
    "vp_ratio": "$volume / MA($volume, 20)",         # 量比
    # 截面排名
    "rank_mom_20": "Rank(Ref($close, 20) / $close - 1)",
    "rank_vol_20": "Rank(STD($close / Ref($close, 1) - 1, 20))",
    # 复合
    "bb_position": "($close - MA($close, 20)) / (STD($close, 20) + 1e-8)",  # 布林带位置
}


def list_preset_factors() -> List[str]:
    """返回预置因子名列表"""
    return list(PRESET_FACTORS.keys())


def get_preset_expression(name: str) -> str:
    """获取预置因子的表达式字符串"""
    if name not in PRESET_FACTORS:
        raise KeyError(f"未知预置因子: {name}，可用: {list(PRESET_FACTORS.keys())}")
    return PRESET_FACTORS[name]
