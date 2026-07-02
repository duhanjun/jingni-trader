"""
因子表达式引擎（借鉴 Microsoft Qlib 的表达式语法设计）

核心思想：
    将因子从硬编码的 Python 函数解耦为"声明式表达式字符串"，
    例如 "Mean(Ref($close, 1), 20)" 表示「过去20日的昨日收盘价均值」。
    这样可以：
      1. 通过配置/字符串动态注册因子，无需修改引擎代码
      2. 因子可序列化、可持久化、可被 LLM 自动生成（对接 FactorEngine 论文思路）
      3. 因子带元信息（方向、类别、说明），便于因子库管理与中性化

借鉴来源：
    - Microsoft Qlib: qlib/data/ops.py 的表达式算子体系（Ref/Mean/Std/Corr 等）
    - FactorEngine (arXiv:2603.16365): 将因子视为可执行、可审计的程序

与现有 factor-engine 的关系：
    本模块不修改 main 分支代码，作为独立优化验证模块存在。
    现有 FactorEngine.compute_a_share_factors() 是硬编码实现，
    本引擎提供等价能力但通过表达式驱动，可平滑替换。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("factor-expression-engine")


# ---------------------------------------------------------------------------
# 1. 因子元信息（借鉴 Qlib 因子注册表思路）
# ---------------------------------------------------------------------------

@dataclass
class FactorMeta:
    """因子元信息：方向、类别、说明，便于因子库管理与中性化"""
    name: str
    expression: str
    direction: int = 1          # 1=正向（值越大越看多），-1=反向
    category: str = "custom"    # price/volume/volatility/reversal/momentum/...
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. 表达式算子（借鉴 Qlib qlib/data/ops.py）
#    所有算子接收按 code 分组后的 Series，返回同长度 Series
# ---------------------------------------------------------------------------

def _op_ref(s: pd.Series, n: int) -> pd.Series:
    """Ref($x, n): 取 n 日前的值（前视防护：shift 用正数，绝不使用未来数据）"""
    return s.groupby(level='code').shift(n)


def _op_mean(s: pd.Series, n: int) -> pd.Series:
    """Mean($x, n): 过去 n 日均值"""
    return s.groupby(level='code').transform(lambda x: x.rolling(n, min_periods=max(1, n // 2)).mean())


def _op_std(s: pd.Series, n: int) -> pd.Series:
    """Std($x, n): 过去 n 日标准差"""
    return s.groupby(level='code').transform(lambda x: x.rolling(n, min_periods=max(2, n // 2)).std())


def _op_sum(s: pd.Series, n: int) -> pd.Series:
    """Sum($x, n): 过去 n 日求和"""
    return s.groupby(level='code').transform(lambda x: x.rolling(n, min_periods=1).sum())


def _op_max(s: pd.Series, n: int) -> pd.Series:
    return s.groupby(level='code').transform(lambda x: x.rolling(n, min_periods=1).max())


def _op_min(s: pd.Series, n: int) -> pd.Series:
    return s.groupby(level='code').transform(lambda x: x.rolling(n, min_periods=1).min())


def _op_rank(s: pd.Series, n: int) -> pd.Series:
    """Rank($x, n): 当日截面分位数（0~1），n 参数保留以兼容签名但截面 rank 不滚动"""
    return s.groupby(level='date').rank(pct=True)


def _op_delta(s: pd.Series, n: int) -> pd.Series:
    """Delta($x, n): $x - Ref($x, n)"""
    return s - _op_ref(s, n)


def _op_ret(s: pd.Series, n: int) -> pd.Series:
    """Ret($x, n): 过去 n 日收益率 = Ref($x,n) / $x - 1 的相反，即 pct_change(n)"""
    return s.groupby(level='code').pct_change(n)


def _op_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Corr($x, $y, n): 过去 n 日滚动相关系数"""
    def _rolling_corr(g: pd.DataFrame) -> pd.Series:
        return g.iloc[:, 0].rolling(n, min_periods=max(2, n // 2)).corr(g.iloc[:, 1])
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1)
    return df.groupby(level='code').apply(_rolling_corr).reset_index(level=0, drop=True)


def _op_scale(s: pd.Series, n: int = 1) -> pd.Series:
    """Scale($x): 截面标准化（去均值除以标准差）"""
    mean = s.groupby(level='date').transform('mean')
    std = s.groupby(level='date').transform('std')
    return (s - mean) / std.replace(0, np.nan)


# 算子注册表：名称 -> (函数, 参数数量, 是否接收两个 Series)
_OPERATOR_REGISTRY: Dict[str, Tuple[Callable, int, bool]] = {
    "Ref": (_op_ref, 1, False),
    "Mean": (_op_mean, 1, False),
    "Std": (_op_std, 1, False),
    "Sum": (_op_sum, 1, False),
    "Max": (_op_max, 1, False),
    "Min": (_op_min, 1, False),
    "Rank": (_op_rank, 1, False),
    "Delta": (_op_delta, 1, False),
    "Ret": (_op_ret, 1, False),
    "Corr": (_op_corr, 1, True),
    "Scale": (_op_scale, 0, False),
}

# 字段映射：$close -> data['close']
_FIELD_MAP: Dict[str, str] = {
    "$open": "open",
    "$high": "high",
    "$low": "low",
    "$close": "close",
    "$volume": "volume",
    "$amount": "amount",
    "$turnover_rate": "turnover_rate",
    "$vwap": "vwap",
}


# ---------------------------------------------------------------------------
# 3. 表达式解析器（轻量递归下降，借鉴 Qlib 的 Expr DSL）
# ---------------------------------------------------------------------------

class _Tokenizer:
    """极简分词器：识别标识符、数字、$field、运算符、括号、逗号"""

    def __init__(self, expr: str):
        self.expr = expr
        self.pos = 0
        self.tokens: List[Tuple[str, Any]] = []
        self._tokenize()

    def _tokenize(self):
        s = self.expr
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c.isspace():
                i += 1
                continue
            if c == '$':
                j = i + 1
                while j < n and (s[j].isalnum() or s[j] == '_'):
                    j += 1
                self.tokens.append(("FIELD", s[i:j]))
                i = j
                continue
            if c.isalpha() or c == '_':
                j = i
                while j < n and (s[j].isalnum() or s[j] == '_'):
                    j += 1
                self.tokens.append(("IDENT", s[i:j]))
                i = j
                continue
            if c.isdigit() or (c == '-' and i + 1 < n and s[i + 1].isdigit()):
                j = i + 1
                while j < n and (s[j].isdigit() or s[j] == '.'):
                    j += 1
                self.tokens.append(("NUM", float(s[i:j])))
                i = j
                continue
            if c in '+-*/(),':
                self.tokens.append((c, c))
                i += 1
                continue
            raise ValueError(f"无法识别的字符 {c!r} 于位置 {i}，表达式: {self.expr}")

    def peek(self) -> Optional[Tuple[str, Any]]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> Tuple[str, Any]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok


class ExpressionParser:
    """
    递归下降解析器，将表达式字符串解析为 AST（嵌套 tuple）。

    文法（简化）:
        expr    := term (('+' | '-') term)*
        term    := factor (('*' | '/') factor)*
        factor  := NUM | FIELD | IDENT '(' args ')' | '(' expr ')' | '-' factor
        args    := expr (',' expr)*

    AST 节点形式:
        ('num', 5.0)
        ('field', '$close')
        ('binop', '+', left_ast, right_ast)
        ('neg', ast)
        ('call', 'Mean', [ast, ...])
    """

    def __init__(self, expr: str):
        self.expr = expr
        self.tok = _Tokenizer(expr)

    def parse(self):
        node = self._parse_expr()
        if self.tok.peek() is not None:
            raise ValueError(f"表达式解析未完成，剩余 token: {self.tok.peek()}，表达式: {self.expr}")
        return node

    def _parse_expr(self):
        node = self._parse_term()
        while True:
            t = self.tok.peek()
            if t and t[0] in ('+', '-'):
                self.tok.next()
                right = self._parse_term()
                node = ('binop', t[0], node, right)
            else:
                break
        return node

    def _parse_term(self):
        node = self._parse_factor()
        while True:
            t = self.tok.peek()
            if t and t[0] in ('*', '/'):
                self.tok.next()
                right = self._parse_factor()
                node = ('binop', t[0], node, right)
            else:
                break
        return node

    def _parse_factor(self):
        t = self.tok.peek()
        if t is None:
            raise ValueError(f"意外的表达式结尾: {self.expr}")
        if t[0] == 'NUM':
            self.tok.next()
            return ('num', t[1])
        if t[0] == 'FIELD':
            self.tok.next()
            return ('field', t[1])
        if t[0] == '-':
            self.tok.next()
            return ('neg', self._parse_factor())
        if t[0] == '(':
            self.tok.next()
            node = self._parse_expr()
            close = self.tok.next()
            if close[0] != ')':
                raise ValueError(f"期望 ')' 但得到 {close}，表达式: {self.expr}")
            return node
        if t[0] == 'IDENT':
            name = t[1]
            self.tok.next()
            nxt = self.tok.peek()
            if nxt and nxt[0] == '(':
                self.tok.next()  # consume '('
                args = []
                if self.tok.peek() and self.tok.peek()[0] != ')':
                    args.append(self._parse_expr())
                    while self.tok.peek() and self.tok.peek()[0] == ',':
                        self.tok.next()
                        args.append(self._parse_expr())
                close = self.tok.next()
                if close[0] != ')':
                    raise ValueError(f"函数 {name} 缺少右括号，表达式: {self.expr}")
                return ('call', name, args)
            # 裸标识符不支持
            raise ValueError(f"未定义的标识符 {name}，表达式: {self.expr}")
        raise ValueError(f"意外的 token {t}，表达式: {self.expr}")


# ---------------------------------------------------------------------------
# 4. 表达式求值器：AST -> pd.Series
# ---------------------------------------------------------------------------

class ExpressionEvaluator:
    """在 OHLCV DataFrame 上求值 AST，返回 pd.Series（MultiIndex: code, date）"""

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data.index, pd.MultiIndex):
            # 期望 data 含 code/date 列，构造 MultiIndex
            data = data.set_index(['code', 'date']).sort_index()
        self.data = data

    def eval(self, ast: Tuple) -> pd.Series:
        kind = ast[0]
        if kind == 'num':
            return pd.Series(ast[1], index=self.data.index)
        if kind == 'field':
            col = _FIELD_MAP.get(ast[1])
            if col is None or col not in self.data.columns:
                raise KeyError(f"未知字段 {ast[1]} 或列 {col} 不存在")
            return self.data[col]
        if kind == 'neg':
            return -self.eval(ast[1])
        if kind == 'binop':
            op, left, right = ast[1], ast[2], ast[3]
            l = self.eval(left)
            r = self.eval(right)
            if op == '+':
                return l + r
            if op == '-':
                return l - r
            if op == '*':
                return l * r
            if op == '/':
                return l / r.replace(0, np.nan)
        if kind == 'call':
            name, args = ast[1], ast[2]
            if name not in _OPERATOR_REGISTRY:
                raise KeyError(f"未知算子 {name}")
            func, n_params, two_series = _OPERATOR_REGISTRY[name]
            evaluated = [self.eval(a) for a in args]
            # 数值参数（如 Mean($close, 20) 中的 20）
            num_args = [a[1] for a in args if a[0] == 'num']
            series_args = [a for a in evaluated if isinstance(a, pd.Series)]
            if two_series:
                if len(series_args) < 2:
                    raise ValueError(f"算子 {name} 需要两个 Series 参数")
                n = int(num_args[0]) if num_args else 5
                return func(series_args[0], series_args[1], n)
            else:
                s = series_args[0]
                n = int(num_args[0]) if num_args else 1
                return func(s, n)
        raise ValueError(f"未知 AST 节点: {ast}")


# ---------------------------------------------------------------------------
# 5. 因子表达式引擎：注册 + 批量计算
# ---------------------------------------------------------------------------

class FactorExpressionEngine:
    """
    因子表达式引擎主入口。

    用法:
        engine = FactorExpressionEngine()
        engine.register(FactorMeta(
            name="ma20_reversal",
            expression="-Ret($close, 20)",   # 20日反转
            direction=1,
            category="reversal",
            description="20日反转因子",
        ))
        factor_df = engine.compute(ohlc_data)
    """

    def __init__(self):
        self.registry: Dict[str, FactorMeta] = {}

    def register(self, meta: FactorMeta) -> None:
        """注册一个因子（含元信息）"""
        # 预解析表达式以尽早发现语法错误
        ast = ExpressionParser(meta.expression).parse()
        # 校验所有算子是否已注册（语义校验）
        self._validate_operators(ast, meta.expression)
        self.registry[meta.name] = meta
        logger.debug(f"已注册因子 {meta.name}: {meta.expression}")

    @staticmethod
    def _validate_operators(ast: Tuple, expr: str) -> None:
        """递归校验 AST 中所有算子是否已注册"""
        if not isinstance(ast, tuple):
            return
        if ast[0] == 'call':
            op_name = ast[1]
            if op_name not in _OPERATOR_REGISTRY:
                raise KeyError(f"未知算子 {op_name!r}，表达式: {expr}")
            for arg in ast[2]:
                FactorExpressionEngine._validate_operators(arg, expr)
        elif ast[0] == 'binop':
            FactorExpressionEngine._validate_operators(ast[2], expr)
            FactorExpressionEngine._validate_operators(ast[3], expr)
        elif ast[0] == 'neg':
            FactorExpressionEngine._validate_operators(ast[1], expr)

    def register_many(self, metas: List[FactorMeta]) -> None:
        for m in metas:
            self.register(m)

    def list_factors(self) -> List[str]:
        return list(self.registry.keys())

    def compute(
        self,
        data: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        批量计算已注册因子。

        参数:
            data: OHLCV DataFrame，需含 code/date 列
            factor_names: 指定计算的因子子集；None 表示全部

        返回:
            DataFrame，列为 code, date, [各因子列]
        """
        if data.empty:
            return data[['code', 'date']] if 'code' in data.columns else data

        names = factor_names or list(self.registry.keys())
        if not names:
            return data[['code', 'date']].copy() if 'code' in data.columns else data

        evaluator = ExpressionEvaluator(data)
        # 输出按 (code, date) 排序，与 evaluator 的 MultiIndex 对齐
        out = data[['code', 'date']].copy() if 'code' in data.columns else data.reset_index()[['code', 'date']].copy()
        out = out.sort_values(['code', 'date']).reset_index(drop=True)

        for name in names:
            meta = self.registry.get(name)
            if meta is None:
                logger.warning(f"因子 {name} 未注册，跳过")
                continue
            ast = ExpressionParser(meta.expression).parse()
            series = evaluator.eval(ast)
            # 应用方向
            if meta.direction == -1:
                series = -series
            # series 的 MultiIndex 为 (code, date) 且已排序，与 out 对齐
            if isinstance(series.index, pd.MultiIndex):
                aligned = series.reset_index(name=name)
                out = out.merge(aligned[['code', 'date', name]], on=['code', 'date'], how='left')
            else:
                out[name] = series.values

        logger.info(f"因子表达式引擎计算完成，共 {len(names)} 个因子")
        return out


# ---------------------------------------------------------------------------
# 6. 预置因子库（对标 Qlib Alpha158 的子集，便于对比验证）
# ---------------------------------------------------------------------------

def build_default_factor_library() -> List[FactorMeta]:
    """
    构建默认因子库，对标 jingni-trader 现有 compute_a_share_factors 的能力，
    但以声明式表达式定义，便于扩展和 LLM 自动生成。
    """
    return [
        FactorMeta("ret_1d", "Ret($close, 1)", 1, "momentum", "1日收益率"),
        FactorMeta("ret_5d", "Ret($close, 5)", 1, "momentum", "5日收益率"),
        FactorMeta("ret_20d", "Ret($close, 20)", 1, "momentum", "20日收益率"),
        FactorMeta("reversal_5d", "-Ret($close, 5)", 1, "reversal", "5日反转"),
        FactorMeta("reversal_20d", "-Ret($close, 20)", 1, "reversal", "20日反转"),
        FactorMeta("volatility_20d", "Std(Ret($close,1), 20)", 1, "volatility", "20日波动率"),
        FactorMeta("volume_mean_20d", "Mean($volume, 20)", 1, "volume", "20日均量"),
        FactorMeta("volume_ratio", "$volume / Mean($volume, 20)", 1, "volume", "量比"),
        FactorMeta("turnover_mean_20d", "Mean($turnover_rate, 20)", 1, "volume", "20日均换手"),
        FactorMeta("turnover_mean_5d", "Mean($turnover_rate, 5)", 1, "volume", "5日均换手"),
        FactorMeta("price_to_ma20", "$close / Mean($close, 20)", -1, "mean_reversion", "价格相对20日均线"),
        FactorMeta("high_low_range_20d", "(Max($high, 20) - Min($low, 20)) / $close", 1, "volatility", "20日振幅"),
    ]
