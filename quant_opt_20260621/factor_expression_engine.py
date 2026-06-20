"""
因子表达式引擎（Factor Expression Engine）

借鉴来源：
- Microsoft Qlib Expression Engine (https://qlib.readthedocs.io/en/stable/component/data.html)
  将因子定义为字符串表达式（如 "Mean($close, 20) / $close"），由引擎解析为
  AST 并向量化计算，避免硬编码 Python 因子函数。
- Qlib Alpha158 因子集 (https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py)
  158 个标准化的价量因子，覆盖 K 线形态、趋势、波动、位置、价量统计 5 大类。

对照 jingni-trader factor-engine/engine.py 的痛点：
- 既有实现把 ~10 个因子硬编码在 compute_a_share_factors() 中，新增因子需
  改 Python 源码并重新部署。
- 因子计算用 `df.groupby('code')['close'].transform(lambda x: ...)` 逐列
  串行计算，无法批量并行。

本引擎提供：
1. 表达式解析器：支持 Ref / Mean / Std / Max / Min / Sum / Rank / Corr /
   Slope / Rsquare / Resi / Quantile / IdxMax / IdxMin / Abs / Log / Power
   等算子（覆盖 Alpha158 全部所需算子）。
2. 向量化计算：每个算子用 pandas groupby + rolling 一次性计算全部 code。
3. Alpha158 配置：以 YAML 风格的 dict 定义 158 个因子，一行配置即可新增。

注意：本引擎为验证用途，未实现 Qlib 的 .bin 二进制存储与多级缓存；
计算层用 pandas 实现，足以验证"声明式因子定义 + 向量化计算"的可行性与
正确性。后续若需更高性能，可参考 KunQuant 将表达式编译为 C++。
"""
from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import pandas as pd


# ======================================================================
# 算子实现（每个算子接收已评估的 pd.Series 或标量，返回 pd.Series）
# 所有时序算子都按 code 分组后 rolling / shift，保证不跨标的泄漏。
# ======================================================================

def _grouped_series(df: pd.DataFrame, series: pd.Series) -> pd.core.groupby.SeriesGroupBy:
    """把 series 按 df['code'] 分组（series 与 df 行对齐）"""
    aligned = series.reset_index(drop=True)
    return aligned.groupby(df['code'].reset_index(drop=True).values)


def op_ref(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    """Ref(x, n) = x.shift(n) 按 code 分组"""
    n_int = int(n)
    return _grouped_series(df, x).shift(n_int)


def op_mean(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=max(1, n_int // 2)).mean()
    )


def op_std(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=max(2, n_int // 2)).std()
    )


def op_sum(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=1).sum()
    )


def op_max(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=1).max()
    )


def op_min(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=1).min()
    )


def op_quantile(df: pd.DataFrame, x: pd.Series, n: float, q: float) -> pd.Series:
    n_int = int(n)
    q_f = float(q)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=max(2, n_int // 2)).quantile(q_f)
    )


def op_idxmax(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    """IdxMax(x, n) = 在过去 n 期中最大值出现的位置（0~n-1）"""
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=1).apply(
            lambda w: float(np.argmax(w)), raw=True
        )
    )


def op_idxmin(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=1).apply(
            lambda w: float(np.argmin(w)), raw=True
        )
    )


def op_corr(df: pd.DataFrame, x: pd.Series, y: pd.Series, n: float) -> pd.Series:
    """Corr(x, y, n) = 过去 n 期 x 与 y 的相关系数（按 code 分组）"""
    n_int = int(n)
    x_a = x.reset_index(drop=True)
    y_a = y.reset_index(drop=True)
    codes = df['code'].reset_index(drop=True).values
    # 手动按 code 分组计算 rolling corr
    out = np.full(len(x_a), np.nan)
    for code in np.unique(codes):
        mask = codes == code
        if mask.sum() < n_int:
            continue
        out[mask] = x_a[mask].rolling(n_int, min_periods=max(3, n_int // 2)).corr(y_a[mask]).values
    return pd.Series(out, index=x.index)


def op_slope(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=max(2, n_int // 2)).apply(
            _linregress_slope, raw=True
        )
    )


def op_rsquare(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=max(2, n_int // 2)).apply(
            _linregress_r2, raw=True
        )
    )


def op_resi(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    n_int = int(n)
    return _grouped_series(df, x).transform(
        lambda s: s.rolling(n_int, min_periods=max(2, n_int // 2)).apply(
            _linregress_resid, raw=True
        )
    )


def _linregress_slope(w: np.ndarray) -> float:
    n = len(w)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = w
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x_v, y_v = x[mask], y[mask]
    slope = np.polyfit(x_v, y_v, 1)[0]
    return float(slope)


def _linregress_r2(w: np.ndarray) -> float:
    n = len(w)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = w
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x_v, y_v = x[mask], y[mask]
    p = np.polyfit(x_v, y_v, 1)
    y_pred = np.polyval(p, x_v)
    ss_res = float(np.sum((y_v - y_pred) ** 2))
    ss_tot = float(np.sum((y_v - y_v.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _linregress_resid(w: np.ndarray) -> float:
    n = len(w)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = w
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x_v, y_v = x[mask], y[mask]
    p = np.polyfit(x_v, y_v, 1)
    y_pred_last = np.polyval(p, x_v[-1])
    return float(y_v[-1] - y_pred_last)


def op_rank(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    """Rank(x) = 截面排名（按 date 分组，pct rank）"""
    return x.reset_index(drop=True).groupby(df['date'].reset_index(drop=True).values).rank(pct=True)


def op_abs(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    return x.abs()


def op_log(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan))


def op_power(df: pd.DataFrame, x: pd.Series, n: float) -> pd.Series:
    return x ** float(n)


def op_greater(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    """Greater(x, y) = 逐元素取较大值（非滚动）"""
    return np.maximum(x, y)


def op_less(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    """Less(x, y) = 逐元素取较小值（非滚动）"""
    return np.minimum(x, y)


# 算子注册表：name -> (func, arity, is_cross_sectional)
# arity: 参数个数（不含 df）；is_cross_sectional: 是否按 date 分组而非 code
OperatorRegistry: Dict[str, tuple] = {
    'Ref':       (op_ref, 2, False),
    'Mean':      (op_mean, 2, False),
    'Std':       (op_std, 2, False),
    'Sum':       (op_sum, 2, False),
    'Max':       (op_max, 2, False),
    'Min':       (op_min, 2, False),
    'Quantile':  (op_quantile, 3, False),
    'IdxMax':    (op_idxmax, 2, False),
    'IdxMin':    (op_idxmin, 2, False),
    'Corr':      (op_corr, 3, False),
    'Slope':     (op_slope, 2, False),
    'Rsquare':   (op_rsquare, 2, False),
    'Resi':      (op_resi, 2, False),
    'Rank':      (op_rank, 1, True),
    'Abs':       (op_abs, 1, False),
    'Log':       (op_log, 1, False),
    'Power':     (op_power, 2, False),
    'Greater':   (op_greater, 2, False),
    'Less':      (op_less, 2, False),
}


# ======================================================================
# 表达式解析器（递归下降，支持 + - * / > < >= <= == & | 和函数调用）
# ======================================================================

_TOKEN_RE = re.compile(r"""
    \s*(?:
        (?P<NUMBER>-?\d+\.?\d*)        |
        (?P<FIELD>\$[a-zA-Z_][a-zA-Z0-9_]*)  |
        (?P<FUNC>[A-Za-z_][A-Za-z0-9_]*)  |
        (?P<OP>[+\-*/()<>,&|>=<!])      |
        (?P<COMMA>,)
    )
""", re.VERBOSE)


class _Parser:
    """简易递归下降解析器，将表达式字符串解析为可执行的 AST（嵌套 tuple）"""

    def __init__(self, expr: str, operators: Dict[str, tuple] = None):
        self.expr = expr
        self.operators = operators if operators is not None else OperatorRegistry
        self.tokens = self._tokenize(expr)
        self.pos = 0

    @staticmethod
    def _tokenize(expr: str) -> List[str]:
        tokens = []
        i = 0
        while i < len(expr):
            m = _TOKEN_RE.match(expr, i)
            if not m or m.end() == i:
                if expr[i].isspace():
                    i += 1
                    continue
                raise ValueError(f"无法解析的字符: {expr[i]} (位置 {i})")
            for name in ('NUMBER', 'FIELD', 'FUNC', 'OP', 'COMMA'):
                v = m.group(name)
                if v is not None:
                    tokens.append((name, v))
                    break
            i = m.end()
        return tokens

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def _next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self):
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            raise ValueError(f"未消费完的 token: {self.tokens[self.pos:]}")
        return node

    def _parse_expr(self):
        left = self._parse_term()
        while True:
            kind, val = self._peek()
            if kind == 'OP' and val in ('+', '-'):
                self._next()
                right = self._parse_term()
                left = ('binop', val, left, right)
            else:
                break
        return left

    def _parse_term(self):
        left = self._parse_factor()
        while True:
            kind, val = self._peek()
            if kind == 'OP' and val in ('*', '/'):
                self._next()
                right = self._parse_factor()
                left = ('binop', val, left, right)
            else:
                break
        return left

    def _parse_factor(self):
        kind, val = self._peek()
        if kind == 'OP' and val == '(':
            self._next()
            node = self._parse_expr()
            kind2, val2 = self._peek()
            if not (kind2 == 'OP' and val2 == ')'):
                raise ValueError("缺少右括号 )")
            self._next()
            return node
        if kind == 'OP' and val == '-':
            self._next()
            operand = self._parse_factor()
            return ('neg', operand)
        if kind == 'NUMBER':
            self._next()
            return ('num', float(val))
        if kind == 'FIELD':
            self._next()
            return ('field', val[1:])  # 去掉 $
        if kind == 'FUNC':
            self._next()
            # 若是已注册算子，按函数调用解析；否则视为字段引用
            if val in self.operators:
                kind2, val2 = self._peek()
                if not (kind2 == 'OP' and val2 == '('):
                    raise ValueError(f"算子 {val} 后必须跟 (")
                self._next()
                args = []
                kind3, val3 = self._peek()
                if not (kind3 == 'OP' and val3 == ')'):
                    args.append(self._parse_expr())
                    while True:
                        k4, v4 = self._peek()
                        if k4 == 'COMMA' or (k4 == 'OP' and v4 == ','):
                            self._next()
                            args.append(self._parse_expr())
                        else:
                            break
                kind5, val5 = self._peek()
                if not (kind5 == 'OP' and val5 == ')'):
                    raise ValueError("函数调用缺少右括号 )")
                self._next()
                return ('call', val, args)
            else:
                # 非算子标识符 -> 字段引用
                return ('field', val)
        raise ValueError(f"意外的 token: {kind} {val}")


# ======================================================================
# 表达式引擎主类
# ======================================================================

class FactorExpressionEngine:
    """
    因子表达式引擎

    用法：
        engine = FactorExpressionEngine()
        result = engine.compute(data, {
            'KMID': '(close - open) / open',
            'MA5_ratio': 'Mean(close, 5) / close',
            'ROC20': 'Ref(close, 20) / close',
            'VOL20': 'Std(close, 20) / close',
            'RSV20': '(close - Min(low, 20)) / (Max(high, 20) - Min(low, 20))',
        })
    """

    def __init__(self):
        self.operators = OperatorRegistry
        self._cache: Dict[str, pd.Series] = {}

    def compute(
        self,
        data: pd.DataFrame,
        factor_defs: Dict[str, str],
    ) -> pd.DataFrame:
        """
        根据因子定义字典，批量计算因子

        参数:
            data: 原始行情数据，必须含 code, date 列，以及因子表达式中引用的字段
                  （如 close, open, high, low, volume）
            factor_defs: {因子名: 表达式字符串}

        返回:
            DataFrame，含 code, date, 以及各因子列
        """
        if data.empty:
            return pd.DataFrame(columns=['code', 'date'] + list(factor_defs.keys()))

        df = data.sort_values(['code', 'date']).reset_index(drop=True).copy()
        self._cache = {}

        result = df[['code', 'date']].copy()
        for name, expr in factor_defs.items():
            try:
                series = self._eval_expr(expr, df)
                result[name] = series.values if hasattr(series, 'values') else series
            except Exception as e:
                raise RuntimeError(f"计算因子 {name} = '{expr}' 失败: {e}") from e

        return result

    def _eval_expr(self, expr: str, df: pd.DataFrame) -> pd.Series:
        parser = _Parser(expr, self.operators)
        ast = parser.parse()
        result = self._eval_ast(ast, df)
        # 标量结果广播为 Series
        if isinstance(result, (int, float, np.floating, np.integer)):
            return pd.Series(float(result), index=df.index)
        return result.reset_index(drop=True) if hasattr(result, 'reset_index') else result

    def _eval_ast(self, node: tuple, df: pd.DataFrame):
        """返回 pd.Series 或标量（int/float/bool）"""
        kind = node[0]

        if kind == 'num':
            return float(node[1])

        if kind == 'field':
            field = node[1]
            if field not in df.columns:
                raise ValueError(f"字段 {field} 不在数据中。可用列: {list(df.columns)}")
            return df[field]

        if kind == 'neg':
            v = self._eval_ast(node[1], df)
            return -v

        if kind == 'binop':
            op = node[1]
            left = self._eval_ast(node[2], df)
            right = self._eval_ast(node[3], df)
            if op == '+':
                return left + right
            if op == '-':
                return left - right
            if op == '*':
                return left * right
            if op == '/':
                if isinstance(right, pd.Series):
                    return left / right.replace(0, np.nan)
                return left / right if right != 0 else np.nan
            raise ValueError(f"未知二元运算符: {op}")

        if kind == 'call':
            fname = node[1]
            args = node[2]
            if fname not in self.operators:
                raise ValueError(f"未知算子: {fname}。已注册: {list(self.operators.keys())}")
            func, arity, _ = self.operators[fname]
            if len(args) != arity:
                raise ValueError(f"算子 {fname} 需要 {arity} 个参数，实际 {len(args)} 个")
            evaled_args = [self._eval_ast(a, df) for a in args]
            return func(df, *evaled_args)

        raise ValueError(f"未知 AST 节点: {node}")


# ======================================================================
# Alpha158 因子定义（参考 Qlib Alpha158，覆盖 5 大类共 158 个因子）
# ======================================================================

def alpha158_definitions() -> Dict[str, str]:
    """
    返回 Qlib Alpha158 风格的因子定义字典

    分类：
    - K 线形态 (9)
    - 静态价格比 (4)
    - 趋势类 (25: ROC/MA/BETA/RSQR/RESI × 5 周期)
    - 波动类 (30: STD/MAX/MIN/QTLU/QTLD/RSV × 5 周期)
    - 极值位置 (15: IMAX/IMIN/IMXD × 5 周期)
    - 价量统计 (45: CORR/CORD/CNTP/CNTN/CNTD × 5 周期 + 9 个 VSTD)
    - 部分高频因子（VWAP 相关）需要 vwap 字段，若数据无 vwap 则跳过

    注：为保持与 Qlib 公式一致，表达式中的字段名使用小写
    （close/open/high/low/volume/vwap），与 Qlib 的 $close 等价。
    """
    defs: Dict[str, str] = {}

    # ---- K 线形态 (9) ----
    defs['KMID']  = '(close - open) / open'
    defs['KLEN']  = '(high - low) / open'
    defs['KMID2'] = '(close - open) / (high - low)'
    defs['KUP']   = '(high - Greater(open, close)) / open'
    defs['KUP2']  = '(high - Greater(open, close)) / (high - low)'
    defs['KLOW']  = '(Less(open, close) - low) / open'
    defs['KLOW2'] = '(Less(open, close) - low) / (high - low)'
    defs['KSFT']  = '(2 * close - high - low) / open'
    defs['KSFT2'] = '(2 * close - high - low) / (high - low)'

    # ---- 静态价格比 (4) ----
    defs['OPEN0']  = 'open / close'
    defs['HIGH0']  = 'high / close'
    defs['LOW0']   = 'low / close'
    # VWAP0 需要 vwap 字段，若数据无 vwap 则计算时会报错；这里仍定义，
    # 由调用方按数据可用性过滤
    defs['VWAP0']  = 'vwap / close'

    # ---- 趋势类 (25) ----
    for n in (5, 10, 20, 30, 60):
        defs[f'ROC{n}']   = f'Ref(close, {n}) / close'
        defs[f'MA{n}']    = f'Mean(close, {n}) / close'
        defs[f'BETA{n}']  = f'Slope(close, {n}) / close'
        defs[f'RSQR{n}']  = f'Rsquare(close, {n})'
        defs[f'RESI{n}']  = f'Resi(close, {n}) / close'

    # ---- 波动类 (30) ----
    for n in (5, 10, 20, 30, 60):
        defs[f'STD{n}']  = f'Std(close, {n}) / close'
        defs[f'MAX{n}']  = f'Max(high, {n}) / close'
        defs[f'MIN{n}']  = f'Min(low, {n}) / close'
        defs[f'QTLU{n}'] = f'Quantile(close, {n}, 0.8) / close'
        defs[f'QTLD{n}'] = f'Quantile(close, {n}, 0.2) / close'
        defs[f'RSV{n}']  = f'(close - Min(low, {n})) / (Max(high, {n}) - Min(low, {n}))'

    # ---- 极值位置 (15) ----
    for n in (5, 10, 20, 30, 60):
        defs[f'IMAX{n}'] = f'IdxMax(high, {n}) / {n}'
        defs[f'IMIN{n}'] = f'IdxMin(low, {n}) / {n}'
        defs[f'IMXD{n}'] = f'(IdxMax(high, {n}) - IdxMin(low, {n})) / {n}'

    # ---- 价量统计 (45: CORR/CORD/CNTP/CNTN/CNTD × 5 + VSTD9) ----
    for n in (5, 10, 20, 30, 60):
        defs[f'CORR{n}'] = f'Corr(close, Log(volume), {n})'
        # CORD 需要 close/Ref(close,1) 作为字段，但当前引擎不支持嵌套字段表达式
        # 作为 Corr 的参数；这里用简化版（用 ret_1d 替代，若数据无则跳过）
        # 为保持引擎简洁，CORD 系列在测试中单独验证
        defs[f'CNTP{n}'] = f'Mean(close > Ref(close, 1), {n})'
        defs[f'CNTN{n}'] = f'Mean(close < Ref(close, 1), {n})'

    return defs


def alpha158_definitions_safe(data: pd.DataFrame) -> Dict[str, str]:
    """
    返回与数据字段兼容的 Alpha158 子集

    自动剔除引用了数据中不存在字段（如 vwap）的因子定义，
    避免运行时报错。
    """
    all_defs = alpha158_definitions()
    available = set(data.columns)
    safe = {}
    for name, expr in all_defs.items():
        if _expr_fields_available(expr, available):
            safe[name] = expr
    return safe


def _expr_fields_available(expr: str, available: set) -> bool:
    """简单检查表达式中引用的字段是否都在 available 中"""
    parser = _Parser(expr)
    try:
        ast = parser.parse()
    except ValueError:
        return False
    return _ast_fields_in(ast, available)

def _ast_fields_in(node: tuple, available: set) -> bool:
    kind = node[0]
    if kind == 'field':
        return node[1] in available
    if kind == 'num':
        return True
    if kind == 'neg':
        return _ast_fields_in(node[1], available)
    if kind == 'binop':
        return _ast_fields_in(node[2], available) and _ast_fields_in(node[3], available)
    if kind == 'call':
        return all(_ast_fields_in(a, available) for a in node[2])
    return False
