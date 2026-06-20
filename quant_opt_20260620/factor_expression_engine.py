"""
因子表达式引擎 (Factor Expression Engine)
借鉴来源: akquant 因子表达式引擎 + WorldQuant Alpha101 算子集

设计目标:
    将 jingni-trader 中硬编码的因子计算 (factor-engine.compute_a_share_factors)
    改造为声明式的字符串公式驱动, 使得新增因子无需修改引擎代码, 仅需写公式.

核心思想:
    - 用户用类 Alpha101 公式描述因子, 例如 "Rank(Ts_Mean(Close, 20))"
    - 引擎解析公式为算子树, 在 (code, date) 面板数据上向量化求值
    - 横截面算子 (Rank) 按 date 分组, 时序算子 (Ts_Mean) 按 code 分组

与现有 factor-engine 的对比:
    现有: 每个因子都是一段 pandas 代码, 写死在 compute_a_share_factors 中
    优化后: 因子 = 公式字符串, 可由配置/LLM 生成, 引擎统一求值
"""
from __future__ import annotations

import re
import math
import time
import operator as op
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 算子注册表
# ============================================================

class Operators:
    """
    算子集合. 分为三类:
      - 时序算子 (Ts_*): 沿 code 分组, 在时间轴上滚动
      - 横截面算子 (Rank / Scale): 沿 date 分组, 在股票截面求值
      - 逐元素算子 (Abs / Sign / Log / +,-,*,/): 逐元素运算
    """

    # ---------- 时序算子 ----------
    @staticmethod
    def ts_mean(df: pd.DataFrame, col: str, d: int) -> pd.Series:
        return df.groupby('code')[col].transform(
            lambda x: x.rolling(d, min_periods=max(1, d // 2)).mean()
        )

    @staticmethod
    def ts_sum(df: pd.DataFrame, col: str, d: int) -> pd.Series:
        return df.groupby('code')[col].transform(
            lambda x: x.rolling(d, min_periods=max(1, d // 2)).sum()
        )

    @staticmethod
    def ts_std(df: pd.DataFrame, col: str, d: int) -> pd.Series:
        return df.groupby('code')[col].transform(
            lambda x: x.rolling(d, min_periods=max(1, d // 2)).std()
        )

    @staticmethod
    def ts_max(df: pd.DataFrame, col: str, d: int) -> pd.Series:
        return df.groupby('code')[col].transform(
            lambda x: x.rolling(d, min_periods=max(1, d // 2)).max()
        )

    @staticmethod
    def ts_min(df: pd.DataFrame, col: str, d: int) -> pd.Series:
        return df.groupby('code')[col].transform(
            lambda x: x.rolling(d, min_periods=max(1, d // 2)).min()
        )

    @staticmethod
    def ts_rank(df: pd.DataFrame, col: str, d: int) -> pd.Series:
        """时序排名: 当前值在过去 d 期中的分位数"""
        def _rank(x):
            return x.rolling(d, min_periods=max(2, d // 2)).apply(
                lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
            )
        return df.groupby('code')[col].transform(_rank)

    @staticmethod
    def ts_argmax(df: pd.DataFrame, col: str, d: int) -> pd.Series:
        """过去 d 期最大值出现的位置 (归一化到 [0,1])"""
        def _argmax(x):
            return x.rolling(d, min_periods=max(2, d // 2)).apply(
                lambda w: float(np.argmax(w)) / max(1, (len(w) - 1)), raw=True
            )
        return df.groupby('code')[col].transform(_argmax)

    @staticmethod
    def delay(df: pd.DataFrame, col: str, d: int) -> pd.Series:
        return df.groupby('code')[col].transform(lambda x: x.shift(d))

    @staticmethod
    def delta(df: pd.DataFrame, col: str, d: int) -> pd.Series:
        return df.groupby('code')[col].transform(lambda x: x.diff(d))

    @staticmethod
    def correlation(df: pd.DataFrame, a: str, b: str, d: int) -> pd.Series:
        """滚动相关系数"""
        def _corr(g):
            return g[a].rolling(d, min_periods=max(2, d // 2)).corr(g[b])
        return df.groupby('code', group_keys=False).apply(_corr)

    @staticmethod
    def covariance(df: pd.DataFrame, a: str, b: str, d: int) -> pd.Series:
        def _cov(g):
            return g[a].rolling(d, min_periods=max(2, d // 2)).cov(g[b])
        return df.groupby('code', group_keys=False).apply(_cov)

    # ---------- 横截面算子 ----------
    @staticmethod
    def rank(df: pd.DataFrame, col: str) -> pd.Series:
        """横截面百分位排名"""
        return df.groupby('date')[col].transform(lambda x: x.rank(pct=True))

    @staticmethod
    def scale(df: pd.DataFrame, col: str) -> pd.Series:
        """横截面标准化: x / sum(|x|)"""
        def _scale(x):
            s = x.abs().sum()
            return x / s if s != 0 else x * 0
        return df.groupby('date')[col].transform(_scale)

    # ---------- 逐元素算子 ----------
    @staticmethod
    def abs_(s: pd.Series) -> pd.Series:
        return s.abs()

    @staticmethod
    def sign(s: pd.Series) -> pd.Series:
        return np.sign(s)

    @staticmethod
    def log(s: pd.Series) -> pd.Series:
        return np.log(s.where(s > 0, np.nan))

    @staticmethod
    def signed_power(s: pd.Series, p: float) -> pd.Series:
        return np.sign(s) * (np.abs(s) ** p)


# ============================================================
# 公式解析器 (递归下降)
# ============================================================

class FormulaParser:
    """
    将公式字符串解析为 AST.

    文法:
        expr   := term (('+' | '-') term)*
        term   := factor (('*' | '/') factor)*
        factor := number | ident | func '(' args ')' | '(' expr ')' | '-' factor
        args   := expr (',' expr)*

    示例:
        "Rank(Ts_Mean(Close, 20))"  ->  Func('Rank', [Func('Ts_Mean', [Ident('Close'), Num(20)])])
        "-1 * Correlation(Open, Volume, 10)" -> BinOp('*', Num(-1), Func(...))
    """

    TOKEN_RE = re.compile(r"""
        \s*(?:
            (?P<NUM>\d+\.?\d*)        |
            (?P<ID>[A-Za-z_][A-Za-z0-9_]*)  |
            (?P<OP>[+\-*/(),])
        )
    """, re.VERBOSE)

    def __init__(self, formula: str):
        self.formula = formula
        self.tokens = self._tokenize(formula)
        self.pos = 0

    def _tokenize(self, s: str) -> List[Tuple[str, str]]:
        tokens = []
        for m in self.TOKEN_RE.finditer(s):
            kind = m.lastgroup
            if kind is None:
                continue
            # 用 m.group(kind) 取命名组内容, 排除前导 \s*
            tokens.append((kind, m.group(kind)))
        return tokens

    def _peek(self) -> Optional[Tuple[str, str]]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> Tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> Tuple[str, Any]:
        node = self._expr()
        if self.pos != len(self.tokens):
            raise SyntaxError(f"未消费的 token: {self.tokens[self.pos:]}")
        return node

    def _expr(self) -> Tuple[str, Any]:
        node = self._term()
        while self._peek() and self._peek()[0] == 'OP' and self._peek()[1] in '+-':
            o = self._next()[1]
            rhs = self._term()
            node = ('binop', o, node, rhs)
        return node

    def _term(self) -> Tuple[str, Any]:
        node = self._factor()
        while self._peek() and self._peek()[0] == 'OP' and self._peek()[1] in '*/':
            o = self._next()[1]
            rhs = self._factor()
            node = ('binop', o, node, rhs)
        return node

    def _factor(self) -> Tuple[str, Any]:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("意外的公式结尾")
        kind, val = tok
        if kind == 'OP' and val == '(':
            self._next()
            node = self._expr()
            if not (self._peek() and self._peek()[1] == ')'):
                raise SyntaxError("缺少右括号")
            self._next()
            return node
        if kind == 'OP' and val == '-':
            self._next()
            node = self._factor()
            return ('neg', node)
        if kind == 'NUM':
            self._next()
            return ('num', float(val))
        if kind == 'ID':
            self._next()
            # 函数调用?
            if self._peek() and self._peek()[1] == '(':
                self._next()
                args = [self._expr()]
                while self._peek() and self._peek()[1] == ',':
                    self._next()
                    args.append(self._expr())
                if not (self._peek() and self._peek()[1] == ')'):
                    raise SyntaxError("函数缺少右括号")
                self._next()
                return ('func', val, args)
            return ('ident', val)
        raise SyntaxError(f"意外的 token: {tok}")


# ============================================================
# 表达式引擎
# ============================================================

class FactorExpressionEngine:
    """
    因子表达式引擎

    用法:
        engine = FactorExpressionEngine()
        result = engine.evaluate("Rank(Ts_Mean(Close, 20))", panel_df)
        # result: pd.Series, 与 panel_df 行对齐
    """

    # 函数名 -> (是否时序/截面算子, 实现函数)
    # 时序/截面算子需要整个 df; 逐元素算子只接收 Series
    FUNC_TABLE: Dict[str, Tuple[str, Callable]] = {
        'Ts_Mean':       ('panel', Operators.ts_mean),
        'Ts_Sum':        ('panel', Operators.ts_sum),
        'Ts_StdDev':     ('panel', Operators.ts_std),
        'Ts_Max':        ('panel', Operators.ts_max),
        'Ts_Min':        ('panel', Operators.ts_min),
        'Ts_Rank':       ('panel', Operators.ts_rank),
        'Ts_ArgMax':     ('panel', Operators.ts_argmax),
        'Delay':         ('panel', Operators.delay),
        'Delta':         ('panel', Operators.delta),
        'Correlation':   ('panel', Operators.correlation),
        'Covariance':    ('panel', Operators.covariance),
        'Rank':          ('panel', Operators.rank),
        'Scale':         ('panel', Operators.scale),
        'Abs':           ('elem',  Operators.abs_),
        'Sign':          ('elem',  Operators.sign),
        'Log':           ('elem',  Operators.log),
        'SignedPower':   ('elem',  Operators.signed_power),
    }

    BINOP_TABLE = {
        '+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv,
    }

    def __init__(self):
        self._cache: Dict[str, Tuple] = {}

    def _parse(self, formula: str) -> Tuple:
        if formula not in self._cache:
            self._cache[formula] = FormulaParser(formula).parse()
        return self._cache[formula]

    def evaluate(self, formula: str, df: pd.DataFrame) -> pd.Series:
        """
        在面板数据 df 上求值公式.

        df 需包含列: code, date, 以及公式中引用的字段 (如 Close, Open, Volume).
        返回与 df 行对齐的 pd.Series.
        """
        ast = self._parse(formula)
        result = self._eval_node(ast, df)
        if isinstance(result, pd.Series):
            return result
        # 常量情况
        return pd.Series(float(result), index=df.index)

    def _eval_node(self, node: Tuple, df: pd.DataFrame):
        kind = node[0]
        if kind == 'num':
            return node[1]
        if kind == 'ident':
            name = node[1]
            # 大小写不敏感匹配列名
            col = self._match_column(df, name)
            if col is None:
                raise KeyError(f"公式引用了不存在的字段: {name}")
            return df[col]
        if kind == 'neg':
            v = self._eval_node(node[1], df)
            return -v
        if kind == 'binop':
            _, o, lhs, rhs = node
            lv = self._eval_node(lhs, df)
            rv = self._eval_node(rhs, df)
            return self.BINOP_TABLE[o](lv, rv)
        if kind == 'func':
            _, fname, args = node
            return self._eval_func(fname, args, df)
        raise ValueError(f"未知节点类型: {kind}")

    def _eval_func(self, fname: str, args: List, df: pd.DataFrame):
        key = fname  # 大小写不敏感
        entry = None
        for k, v in self.FUNC_TABLE.items():
            if k.lower() == key.lower():
                entry = v
                break
        if entry is None:
            raise KeyError(f"未知算子: {fname}")
        kind, impl = entry
        evaluated = [self._eval_node(a, df) for a in args]
        if kind == 'panel':
            # panel 算子: 第一个参数是列名(字符串), 后续可能是数字
            col_name = evaluated[0]
            if isinstance(col_name, pd.Series):
                # 动态列: 先写入 df
                tmp_col = f"__tmp_{fname}_{id(col_name)}"
                df = df.copy()
                df[tmp_col] = col_name
                col_name = tmp_col
            else:
                col_name = self._match_column(df, str(args[0][1])) if args[0][0] == 'ident' else col_name
            if fname.lower().startswith('correlation') or fname.lower().startswith('covariance'):
                a_col = self._resolve_col(evaluated[0], df, args[0])
                b_col = self._resolve_col(evaluated[1], df, args[1])
                d = int(evaluated[2])
                return impl(df, a_col, b_col, d)
            d = int(evaluated[1]) if len(evaluated) > 1 else None
            if d is None:
                return impl(df, col_name)
            return impl(df, col_name, d)
        else:
            # 逐元素算子
            s = evaluated[0]
            if fname.lower() == 'signedpower':
                p = float(evaluated[1])
                return impl(s, p)
            return impl(s)

    def _resolve_col(self, val, df: pd.DataFrame, arg_node):
        if isinstance(val, str):
            return self._match_column(df, val) or val
        if isinstance(val, pd.Series):
            tmp = f"__tmp_{id(val)}"
            df[tmp] = val
            return tmp
        return val

    @staticmethod
    def _match_column(df: pd.DataFrame, name: str) -> Optional[str]:
        """大小写不敏感匹配列名"""
        cols = {c.lower(): c for c in df.columns}
        return cols.get(name.lower())


# ============================================================
# Alpha101 因子公式集 (节选, 用于验证)
# ============================================================

ALPHA101_FORMULAS: Dict[str, str] = {
    # Alpha#3: -1 * correlation(rank(open), rank(volume), 10)
    "alpha003": "-1 * Correlation(Rank(Open), Rank(Volume), 10)",
    # Alpha#4: -1 * Ts_Rank(rank(low), 9)
    "alpha004": "-1 * Ts_Rank(Rank(Low), 9)",
    # Alpha#6: -1 * correlation(open, volume, 10)
    "alpha006": "-1 * Correlation(Open, Volume, 10)",
    # Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    "alpha012": "Sign(Delta(Volume, 1)) * (-1 * Delta(Close, 1))",
    # Alpha#23: ((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0  (简化为非条件版)
    "alpha023_simple": "-1 * Delta(High, 2)",
    # Alpha#33: rank(-1 * (1 - (open / close)))
    "alpha033": "Rank(-1 * (1 - (Open / Close)))",
    # Alpha#52: 简化版 -1 * Ts_Rank(Rank(Close), 5)
    "alpha052_simple": "-1 * Ts_Rank(Rank(Close), 5)",
    # 自定义: 20日反转因子 = -Rank(Ts_Mean(Return, 20))
    "reversal_20d": "-1 * Rank(Ts_Mean(Close, 20))",
    # 自定义: 量价背离 = Rank(Correlation(Close, Volume, 20))
    "price_volume_divergence": "Rank(Correlation(Close, Volume, 20))",
}
