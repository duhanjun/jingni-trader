"""
因子表达式引擎（验证原型）

借鉴来源：
- AKQuant: 内置 Polars 驱动的因子表达式引擎，支持
  `Rank(Ts_Mean(Close, 5))` 等 Alpha101 风格公式。
- Qlib (Microsoft): 表达式 DSL，将因子定义为可组合的算子树。
- WorldQuant Alpha101: 101 个公式化因子的标准算子集。

优化目标：
jingni-trader 现有 skills/factor-engine/scripts/adapters/pandas_ta_calculator.py
通过 if/elif 硬编码每个因子，新增因子需修改源码，可扩展性差；
且 _calc_single 用 `for code in unique(): data[mask]` 循环，O(n²)。

本模块实现一个轻量表达式引擎：
1. 解析 `Rank(Ts_Mean(Close, 5))` 风格字符串为 AST
2. 算子全部基于 groupby('code') 的向量化实现（消除 per-code 循环）
3. 横截面算子（Rank、Zscore）按 date 分组
4. 时间序列算子（Ts_Mean、Ts_Std、Delta、Delay）按 code 分组
5. 支持任意嵌套组合，无需改源码即可定义新因子

支持算子：
- 字段: Close, Open, High, Low, Volume, Amount, Returns
- 时序: Ts_Mean(x, n), Ts_Std(x, n), Ts_Max(x, n), Ts_Min(x, n),
        Ts_Rank(x, n), Delta(x, n), Delay(x, n), Ts_Sum(x, n),
        WMA(x, n), EMA(x, n)
- 横截面: Rank(x), Zscore(x), Scale(x)
- 二元: Add(x, y), Sub(x, y), Mul(x, y), Div(x, y),
        Corr(x, y, n), Cov(x, y, n)
- 常量: 数字直接作为常量因子
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Callable
import numpy as np
import pandas as pd


# ---------------- 表达式 AST 节点 ----------------

class Node:
    """AST 基类"""
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class FieldNode(Node):
    """字段引用，如 Close / Volume"""
    FIELD_MAP = {
        'Close': 'close', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Volume': 'volume', 'Amount': 'amount',
    }

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        col = self.FIELD_MAP.get(self.name, self.name.lower())
        if col not in df.columns:
            raise KeyError(f"字段 {self.name}({col}) 不存在")
        return df[col].copy()

    def __repr__(self):
        return f"Field({self.name})"


class ConstNode(Node):
    """常量"""
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index)

    def __repr__(self):
        return f"Const({self.value})"


class ReturnsNode(Node):
    """收益率 Returns(Close, n)"""
    def __init__(self, field: str, n: int):
        self.field = field
        self.n = n

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        col = FieldNode.FIELD_MAP.get(self.field, self.field.lower())
        return df.groupby('code')[col].pct_change(self.n)

    def __repr__(self):
        return f"Returns({self.field},{self.n})"


class TsOpNode(Node):
    """时间序列算子（按 code 分组）"""
    TS_OPS = {
        'Ts_Mean': lambda x, n: x.rolling(n, min_periods=max(1, n // 2)).mean(),
        'Ts_Sum': lambda x, n: x.rolling(n, min_periods=max(1, n // 2)).sum(),
        'Ts_Std': lambda x, n: x.rolling(n, min_periods=max(2, n // 2)).std(),
        'Ts_Max': lambda x, n: x.rolling(n, min_periods=1).max(),
        'Ts_Min': lambda x, n: x.rolling(n, min_periods=1).min(),
        'Ts_Rank': lambda x, n: x.rolling(n, min_periods=2).rank(pct=True),
        'Delta': lambda x, n: x.diff(n),
        'Delay': lambda x, n: x.shift(n),
        'WMA': None,   # 单独实现
        'EMA': None,   # 单独实现
    }

    def __init__(self, op: str, child: Node, n: int):
        self.op = op
        self.child = child
        self.n = n

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        x = self.child.evaluate(df)
        n = self.n
        if self.op == 'WMA':
            # 加权移动平均：权重 1..n
            def _wma(s):
                weights = np.arange(1, n + 1, dtype=float)
                return s.rolling(n, min_periods=n).apply(
                    lambda v: np.dot(v, weights) / weights.sum(), raw=True
                )
            return x.groupby(df['code']).transform(_wma)
        if self.op == 'EMA':
            span = max(2, n)
            return x.groupby(df['code']).transform(lambda s: s.ewm(span=span, adjust=False).mean())

        fn = self.TS_OPS[self.op]
        return x.groupby(df['code']).transform(fn, n)

    def __repr__(self):
        return f"{self.op}({self.child},{self.n})"


class CrossOpNode(Node):
    """横截面算子（按 date 分组）"""
    CROSS_OPS = {
        'Rank': lambda x: x.rank(pct=True),
        'Zscore': lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1),
        'Scale': lambda x: x / (x.abs().sum() if x.abs().sum() > 0 else 1),
    }

    def __init__(self, op: str, child: Node):
        self.op = op
        self.child = child

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        x = self.child.evaluate(df)
        fn = self.CROSS_OPS[self.op]
        return x.groupby(df['date']).transform(fn)

    def __repr__(self):
        return f"{self.op}({self.child})"


class BinaryOpNode(Node):
    """二元算子"""
    BIN_OPS = {
        'Add': np.add, 'Sub': np.subtract,
        'Mul': np.multiply, 'Div': np.divide,
        'Max': np.maximum, 'Min': np.minimum,
    }

    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.evaluate(df)
        r = self.right.evaluate(df)
        fn = self.BIN_OPS[self.op]
        return pd.Series(fn(l.values, r.values), index=df.index)

    def __repr__(self):
        return f"{self.op}({self.left},{self.right})"


class PairTsOpNode(Node):
    """二元时序算子 Corr/Cov"""
    def __init__(self, op: str, left: Node, right: Node, n: int):
        self.op = op
        self.left = left
        self.right = right
        self.n = n

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.evaluate(df)
        r = self.right.evaluate(df)
        tmp = pd.DataFrame({'l': l, 'r': r, 'code': df['code']})
        if self.op == 'Corr':
            return tmp.groupby('code').apply(
                lambda g: g['l'].rolling(self.n, min_periods=self.n).corr(g['r'])
            ).reset_index(level=0, drop=True)
        else:  # Cov
            return tmp.groupby('code').apply(
                lambda g: g['l'].rolling(self.n, min_periods=self.n).cov(g['r'])
            ).reset_index(level=0, drop=True)

    def __repr__(self):
        return f"{self.op}({self.left},{self.right},{self.n})"


# ---------------- 表达式解析器（递归下降） ----------------

class ExpressionParser:
    """
    解析形如 Rank(Ts_Mean(Close, 5)) 的表达式
    文法：
      expr   := term (('+'|'-') term)*
      term   := factor (('*'|'/') factor)*
      factor := IDENT '(' args ')' | IDENT | NUMBER | '(' expr ')'
      args   := expr (',' expr)* | expr ',' NUMBER
    """

    TOKEN_RE = re.compile(r'\s*(?:(\d+\.?\d*)|([A-Za-z_]\w*)|([(),+\-*/]))')

    def __init__(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        pos = 0
        while pos < len(text):
            m = self.TOKEN_RE.match(text, pos)
            if not m:
                if text[pos].isspace():
                    pos += 1
                    continue
                raise ValueError(f"无法解析字符: {text[pos]} @ {pos}")
            for g in m.groups():
                if g is not None:
                    tokens.append(g)
                    break
            pos = m.end()
        return tokens

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError(f"未消费的 token: {self.tokens[self.pos:]}")
        return node

    def _parse_expr(self) -> Node:
        node = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._next()
            right = self._parse_term()
            node = BinaryOpNode('Add' if op == '+' else 'Sub', node, right)
        return node

    def _parse_term(self) -> Node:
        node = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._next()
            right = self._parse_factor()
            node = BinaryOpNode('Mul' if op == '*' else 'Div', node, right)
        return node

    def _parse_factor(self) -> Node:
        tok = self._peek()
        # 一元正负号
        if tok in ('+', '-'):
            self._next()
            child = self._parse_factor()
            return BinaryOpNode('Mul', ConstNode(-1.0 if tok == '-' else 1.0), child)
        if tok == '(':
            self._next()
            node = self._parse_expr()
            if self._next() != ')':
                raise ValueError("缺少右括号")
            return node
        if tok is None:
            raise ValueError("意外的表达式结束")

        # 数字
        if re.fullmatch(r'\d+\.?\d*', tok):
            self._next()
            return ConstNode(float(tok))

        # 标识符：可能是函数调用或字段
        self._next()
        if self._peek() == '(':
            return self._parse_call(tok)
        # 字段
        if tok == 'Returns':
            # Returns(Close, 5) 特殊处理
            raise ValueError("Returns 需以函数形式调用: Returns(Close, 5)")
        return FieldNode(tok)

    def _parse_call(self, name: str) -> Node:
        self._next()  # 消费 '('
        args = []
        while self._peek() != ')':
            args.append(self._parse_expr())
            if self._peek() == ',':
                self._next()
        self._next()  # 消费 ')'

        # 根据算子名构造节点
        ts_ops = set(TsOpNode.TS_OPS.keys()) | {'WMA', 'EMA'}
        cross_ops = set(CrossOpNode.CROSS_OPS.keys())
        bin_ops = set(BinaryOpNode.BIN_OPS.keys())
        pair_ts_ops = {'Corr', 'Cov'}

        if name == 'Returns':
            field_node, n_node = args
            if not isinstance(field_node, FieldNode) or not isinstance(n_node, ConstNode):
                raise ValueError("Returns(field, n) 参数错误")
            return ReturnsNode(field_node.name, int(n_node.value))

        if name in ts_ops:
            if len(args) != 2:
                raise ValueError(f"{name} 需要 2 个参数 (x, n)")
            child, n_node = args
            if not isinstance(n_node, ConstNode):
                raise ValueError(f"{name} 的第二个参数必须是数字")
            return TsOpNode(name, child, int(n_node.value))

        if name in cross_ops:
            if len(args) != 1:
                raise ValueError(f"{name} 需要 1 个参数 (x)")
            return CrossOpNode(name, args[0])

        if name in bin_ops:
            if len(args) != 2:
                raise ValueError(f"{name} 需要 2 个参数")
            return BinaryOpNode(name, args[0], args[1])

        if name in pair_ts_ops:
            if len(args) != 3:
                raise ValueError(f"{name} 需要 3 个参数 (x, y, n)")
            l, r, n_node = args
            if not isinstance(n_node, ConstNode):
                raise ValueError(f"{name} 的第三个参数必须是数字")
            return PairTsOpNode(name, l, r, int(n_node.value))

        raise ValueError(f"未知算子: {name}")


# ---------------- 引擎入口 ----------------

class FactorExpressionEngine:
    """因子表达式引擎"""

    # 预置 Alpha101 风格因子库
    PRESET_FACTORS: Dict[str, str] = {
        # 反转
        'rev_5': '-Returns(Close, 5)',
        'rev_20': '-Returns(Close, 20)',
        # 动量
        'mom_20': 'Returns(Close, 20)',
        'mom_60': 'Returns(Close, 60)',
        # 波动率
        'vol_20': 'Ts_Std(Returns(Close, 1), 20)',
        # 量价
        'vol_ratio': 'Div(Volume, Ts_Mean(Volume, 20))',
        # 横截面排名反转
        'rank_rev_5': 'Rank(-Returns(Close, 5))',
        'rank_rev_20': 'Rank(-Returns(Close, 20))',
        # 均线偏离
        'ma_bias_20': 'Div(Sub(Close, Ts_Mean(Close, 20)), Ts_Mean(Close, 20))',
        # 换手排名（若有 turnover_rate 字段）
        'rank_turnover': 'Rank(Ts_Mean(turnover_rate, 20))',
        # Alpha101#2 风格: -1 * correlation(delta(log(volume), 2), ((close-open)/open), 6) 简化版
        'alpha2_like': 'Rank(Div(Sub(Close, Open), Open))',
        # 标准化动量
        'zscore_mom_20': 'Zscore(Returns(Close, 20))',
        # WMA 反转
        'wma_rev_10': 'Div(Sub(Close, WMA(Close, 10)), WMA(Close, 10))',
    }

    def __init__(self):
        self.parser_cls = ExpressionParser

    def get_available_factors(self) -> List[str]:
        return list(self.PRESET_FACTORS.keys())

    def get_factor_info(self, name: str) -> Dict[str, Any]:
        expr = self.PRESET_FACTORS.get(name)
        if not expr:
            return {}
        return {"name": name, "expression": expr, "type": "expression"}

    def calculate(self, data: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """批量计算表达式因子"""
        if data.empty:
            return data

        df = data.sort_values(['code', 'date']).reset_index(drop=True)
        result = df[['code', 'date']].copy()

        for name in factor_names:
            expr = self.PRESET_FACTORS.get(name)
            if expr is None:
                # 允许直接传入表达式字符串
                if '(' in name and ')' in name:
                    expr = name
                else:
                    raise ValueError(f"未知因子: {name}")
            try:
                ast = ExpressionParser(expr).parse()
                result[name] = ast.evaluate(df).values
            except Exception as e:
                raise RuntimeError(f"计算因子 {name} (expr={expr}) 失败: {e}") from e

        return result

    def calculate_expression(self, data: pd.DataFrame, expression: str,
                             name: str = 'custom_factor') -> pd.DataFrame:
        """直接计算任意表达式"""
        df = data.sort_values(['code', 'date']).reset_index(drop=True)
        ast = ExpressionParser(expression).parse()
        out = df[['code', 'date']].copy()
        out[name] = ast.evaluate(df).values
        return out
