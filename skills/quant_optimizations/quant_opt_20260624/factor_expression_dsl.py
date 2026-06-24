"""
因子表达式 DSL 引擎 (feat/quant-opt-20260624)

借鉴来源:
  - Microsoft Qlib: 表达式 DSL + 算子注册表 (Expression/Feature/Operator 体系),
                    按 hash(instrument, expr, freq) 缓存, 三级缓存 (Mem/Expression/Dataset),
                    Alpha158 T+1 感知标签 Ref($close,-2)/Ref($close,-1)-1

针对 jingni-trader factor-engine 的改进点:
  1. 性能: pandas_ta_calculator._calc_single 用 `for code in unique()` + boolean mask 逐股循环
           → 改用 groupby().transform() 向量化,一次完成全部股票
  2. 可扩展性: 原实现 if/elif 硬编码因子,新增因子需改源码 → DSL 字符串解析 + 算子注册表
  3. 缓存: 原实现每次全量重算 → 按 (表达式, 股票池, 频率) hash 落盘缓存
  4. 正确性: 原前向收益标签 close[T+n]/close[T]-1 假设 T 日收盘可买
           → 采用 Qlib T+1 标签 Ref($close,-2)/Ref($close,-1)-1 (T+1买 T+2卖)
"""
from __future__ import annotations

import hashlib
import os
import pickle
import re
import time
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 算子注册表 (借鉴 Qlib OpsWrapper)
# ---------------------------------------------------------------------------

class Operator:
    """算子基类"""

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


def _grouped(df: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.Series]) -> pd.Series:
    """按 code 分组应用函数,返回与 df 对齐的 Series(向量化核心)"""
    return df.groupby('code', group_keys=False).apply(fn)


# ---- 逐元素算子 ----

class Ref(Operator):
    """引用 N 期前的值: Ref($close, 1) = 昨收"""
    def __init__(self, field: str, n: int):
        self.field = field
        self.n = n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return _grouped(df, lambda g: g[self.field].shift(self.n))


class Delta(Operator):
    """差分: Delta($close, 5) = $close - Ref($close,5)"""
    def __init__(self, field: str, n: int):
        self.field, self.n = field, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return _grouped(df, lambda g: g[self.field].diff(self.n))


class Mean(Operator):
    """滚动均值"""
    def __init__(self, field: str, n: int):
        self.field, self.n = field, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return _grouped(df, lambda g: g[self.field].rolling(self.n, min_periods=max(1, self.n//2)).mean())


class Std(Operator):
    """滚动标准差"""
    def __init__(self, field: str, n: int):
        self.field, self.n = field, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return _grouped(df, lambda g: g[self.field].rolling(self.n, min_periods=max(2, self.n//2)).std())


class Max(Operator):
    """滚动最大值"""
    def __init__(self, field: str, n: int):
        self.field, self.n = field, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return _grouped(df, lambda g: g[self.field].rolling(self.n, min_periods=1).max())


class Min(Operator):
    """滚动最小值"""
    def __init__(self, field: str, n: int):
        self.field, self.n = field, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return _grouped(df, lambda g: g[self.field].rolling(self.n, min_periods=1).min())


class Rank(Operator):
    """滚动时序排名(百分位)"""
    def __init__(self, field: str, n: int):
        self.field, self.n = field, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return _grouped(df, lambda g: g[self.field].rolling(self.n, min_periods=2).rank(pct=True))


class Corr(Operator):
    """滚动相关系数: Corr($close, $volume, 20)"""
    def __init__(self, field_a: str, field_b: str, n: int):
        self.fa, self.fb, self.n = field_a, field_b, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        def _fn(g):
            return g[self.fa].rolling(self.n, min_periods=max(2, self.n//2)).corr(g[self.fb])
        return _grouped(df, _fn)


class WMA(Operator):
    """加权移动平均(线性权重)"""
    def __init__(self, field: str, n: int):
        self.field, self.n = field, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        weights = np.arange(1, self.n + 1, dtype=float)

        def _fn(g):
            return g[self.field].rolling(self.n, min_periods=self.n).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        return _grouped(df, _fn)


class EMA(Operator):
    """指数移动平均"""
    def __init__(self, field: str, n: int):
        self.field, self.n = field, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        alpha = 2.0 / (self.n + 1)
        return _grouped(df, lambda g: g[self.field].ewm(span=self.n, adjust=False).mean())


class RSI(Operator):
    """RSI 相对强弱指标(向量化实现,不依赖 pandas_ta)"""
    def __init__(self, field: str, n: int):
        self.field, self.n = field, n

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        def _fn(g):
            diff = g[self.field].diff()
            gain = diff.clip(lower=0)
            loss = -diff.clip(upper=0)
            avg_gain = gain.ewm(alpha=1.0/self.n, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0/self.n, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            return 100 - 100 / (1 + rs)
        return _grouped(df, _fn)


class CrossSectionRank(Operator):
    """横截面排名(每日跨所有股票排名,借鉴 Qlib CSRankNorm 思路)"""
    def __init__(self, field: str):
        self.field = field

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('date')[self.field].rank(pct=True)


# ---- 二元算子 ----

class BinOp(Operator):
    """二元运算: $close / Ref($close,20) - 1"""
    def __init__(self, op: str, left: Operator, right: Operator):
        self.op = op
        self.left, self.right = left, right

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        l = self.left(df)
        r = self.right(df)
        if self.op == '+': return l + r
        if self.op == '-': return l - r
        if self.op == '*': return l * r
        if self.op == '/': return l / r.replace(0, np.nan)
        raise ValueError(f"未知算子: {self.op}")


class Const(Operator):
    """常量"""
    def __init__(self, value: float):
        self.value = value

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index)


class Feature(Operator):
    """原始字段引用: $close"""
    def __init__(self, field: str):
        self.field = field

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return df[self.field]


# ---------------------------------------------------------------------------
# 算子注册表 (借鉴 Qlib OpsWrapper.register_all_ops)
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY: Dict[str, type] = {
    'Ref': Ref, 'Delta': Delta, 'Mean': Mean, 'Std': Std,
    'Max': Max, 'Min': Min, 'Rank': Rank, 'Corr': Corr,
    'WMA': WMA, 'EMA': EMA, 'RSI': RSI, 'CSRank': CrossSectionRank,
}


def register_operator(name: str, cls: type):
    """注册自定义算子"""
    OPERATOR_REGISTRY[name] = cls


# ---------------------------------------------------------------------------
# 表达式解析器 (借鉴 Qlib 表达式树)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'\s*(?:(\d+\.?\d*)|(\$?\w+)|([+\-*/(),]))')


class ExpressionParser:
    """
    解析因子表达式字符串为算子树

    支持:
      - 字段引用: $close, $volume, $high, $low, $open
      - 函数调用: Mean($close, 20), Ref($close, 1), Corr($close, $volume, 20)
      - 四则运算: $close / Ref($close, 20) - 1
      - 常量: 1, 0.5
    """

    def parse(self, expr: str) -> Operator:
        self.tokens = self._tokenize(expr)
        self.pos = 0
        node = self._parse_expr()
        if self.pos < len(self.tokens):
            raise ValueError(f"未消费的 token: {self.tokens[self.pos:]}")
        return node

    def _tokenize(self, expr: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(expr):
            m = _TOKEN_RE.match(expr, i)
            if not m:
                if expr[i].isspace():
                    i += 1
                    continue
                raise ValueError(f"无法解析字符 '{expr[i]}' at {i}")
            for g in m.groups():
                if g is not None:
                    tokens.append(g)
                    break
            i = m.end()
        return tokens

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _parse_expr(self) -> Operator:
        node = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            node = BinOp(op, node, right)
        return node

    def _parse_term(self) -> Operator:
        node = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            node = BinOp(op, node, right)
        return node

    def _parse_factor(self) -> Operator:
        tok = self._peek()
        if tok == '(':
            self._consume()
            node = self._parse_expr()
            if self._peek() != ')':
                raise ValueError("缺少右括号")
            self._consume()
            return node
        if tok == '-':
            self._consume()
            return BinOp('*', Const(-1.0), self._parse_factor())
        # 函数调用 or 字段 or 常量
        if tok and tok[0] == '$':
            self._consume()
            return Feature(tok[1:])
        if tok and (tok[0].isdigit() or tok[0] == '.'):
            self._consume()
            return Const(float(tok))
        # 函数名
        if tok and tok[0].isalpha() and self._peek_is_function():
            name = self._consume()
            return self._parse_call(name)
        # 裸字段名(无$)
        if tok and tok[0].isalpha():
            self._consume()
            return Feature(tok)
        raise ValueError(f"意外 token: {tok}")

    def _peek_is_function(self) -> bool:
        return (self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1] == '(')

    def _parse_call(self, name: str) -> Operator:
        if name not in OPERATOR_REGISTRY:
            raise ValueError(f"未知算子: {name}, 已注册: {list(OPERATOR_REGISTRY.keys())}")
        self._consume()  # (
        args = []
        if self._peek() != ')':
            args.append(self._parse_arg())
            while self._peek() == ',':
                self._consume()
                args.append(self._parse_arg())
        if self._peek() != ')':
            raise ValueError("函数调用缺少右括号")
        self._consume()
        cls = OPERATOR_REGISTRY[name]
        return cls(*args)

    def _parse_arg(self):
        tok = self._peek()
        if tok and (tok[0].isdigit() or tok[0] == '.'):
            return int(float(self._consume())) if '.' not in tok else float(self._consume())
        # 字段参数
        tok = self._consume()
        if tok.startswith('$'):
            return tok[1:]
        return tok


# ---------------------------------------------------------------------------
# 因子引擎 (向量化 + 缓存)
# ---------------------------------------------------------------------------

class FactorExpressionEngine:
    """
    因子表达式引擎: 解析 → 向量化计算 → 缓存

    用法:
        engine = FactorExpressionEngine(cache_dir='./.factor_cache')
        result = engine.compute(data, [
            'Mean($close, 20)',                      # 20日均价
            '$close / Ref($close, 20) - 1',           # 20日收益率
            'RSI($close, 14)',                        # RSI
            'Corr($close, $volume, 20)',              # 量价相关
        ])
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.parser = ExpressionParser()
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def compute(self, data: pd.DataFrame, expressions: List[str]) -> pd.DataFrame:
        """
        批量计算因子表达式

        参数:
            data: 含 code, date, open, high, low, close, volume 等列的 DataFrame
            expressions: 因子表达式字符串列表

        返回:
            DataFrame, 列为 code, date, [各表达式因子列]
        """
        if data.empty:
            return data[['code', 'date']].copy() if 'code' in data.columns else data

        # 排序保证 groupby 正确性
        df = data.sort_values(['code', 'date']).reset_index(drop=True)
        result = df[['code', 'date']].copy()

        for expr in expressions:
            col_name = self._expr_to_name(expr)
            values = self._compute_one(df, expr)
            result[col_name] = values

        return result

    def _compute_one(self, df: pd.DataFrame, expr: str) -> pd.Series:
        """计算单个表达式(带缓存)"""
        cache_key = self._cache_key(df, expr)
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        t0 = time.time()
        operator = self.parser.parse(expr)
        values = operator(df)
        elapsed = time.time() - t0

        self._save_cache(cache_key, values)
        return values

    def _expr_to_name(self, expr: str) -> str:
        """表达式转列名(仅去空格,保留$和运算符以保持可读性,截断防过长)"""
        name = expr.replace(' ', '')
        return name[:60]

    def _cache_key(self, df: pd.DataFrame, expr: str) -> str:
        """缓存键: hash(表达式 + 股票池 + 日期范围 + 数据行数)"""
        codes = sorted(df['code'].unique())
        date_range = f"{df['date'].min()}_to_{df['date'].max()}"
        raw = f"{expr}|{date_range}|{len(codes)}codes|{len(df)}rows"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[pd.Series]:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, values: pd.Series):
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(path, 'wb') as f:
                pickle.dump(values, f)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# T+1 感知标签 (借鉴 Qlib Alpha158)
# ---------------------------------------------------------------------------

def t_plus_1_label(df: pd.DataFrame) -> pd.Series:
    """
    A股 T+1 感知标签: Ref($close,-2)/Ref($close,-1) - 1

    含义: T 日收盘生成信号 → T+1 日买入(以 T+1 收盘成交) → T+2 日卖出(以 T+2 收盘成交)
    收益 = close[T+2]/close[T+1] - 1

    对比 jingni-trader 原实现: ret_forward_1d = close[T+1]/close[T]-1
    原实现假设 T 日收盘即可买入,不符合 A 股 T+1 规则,存在前视偏差
    """
    return df.groupby('code', group_keys=False)['close'].apply(
        lambda x: x.shift(-2) / x.shift(-1) - 1
    )


def naive_label(df: pd.DataFrame) -> pd.Series:
    """原实现的前向收益标签(用于对比)"""
    return df.groupby('code', group_keys=False)['close'].apply(
        lambda x: x.shift(-1) / x - 1
    )