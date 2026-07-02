"""
因子表达式引擎 (Factor Expression Engine)

借鉴来源:
  - Qlib Alpha158/Alpha360: 因子表达式 DSL，用算子组合生成因子
  - akquant: Polars 驱动的因子表达式引擎，支持 Rank(Ts_Mean(Close, 5)) 等 Alpha101 风格公式

优化点:
  原 factor-engine 的因子硬编码在 compute_a_share_factors() 中，新增因子需改源码。
  本引擎提供表达式 DSL，用户可用字符串定义因子，无需写代码：
    "Rank(Ts_Mean(Close, 5))"          # 5日均量排名
    "Div(Sub(High, Low), Close)"        # 振幅
    "Mul(-1, Ts_Mean(Return, 20))"      # 20日反转

  支持的算子:
    - 截面算子: Rank, Zscore, Scale
    - 时序算子: Ts_Mean, Ts_Std, Ts_Max, Ts_Min, Ts_Sum, Ts_Rank, Ts_Delta, Ts_Delay
    - 算术算子: Add, Sub, Mul, Div, Abs, Log, Sign
    - 字段: Open, High, Low, Close, Volume, Amount, Turnover, Return

该模块为独立实现，不修改 main 分支代码。
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Callable


# ── 字段映射 ──
FIELD_MAP = {
    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
    'Volume': 'volume', 'Amount': 'amount', 'Turnover': 'turnover_rate',
    'Return': 'ret_1d',
}


class FactorExpressionEngine:
    """因子表达式解析与计算引擎"""

    # 算子注册表: name -> (func, is_timeseries, n_args)
    # is_timeseries=True 的算子需要按 code 分组做时序运算
    OPERATORS: Dict[str, tuple] = {}

    def __init__(self):
        self._register_defaults()

    # ── 算子注册 ──
    def _register_defaults(self):
        # 截面算子（按 date 分组）
        self.register('Rank', self._op_rank, is_timeseries=False)
        self.register('Zscore', self._op_zscore, is_timeseries=False)
        self.register('Scale', self._op_scale, is_timeseries=False)

        # 时序算子（按 code 分组）
        self.register('Ts_Mean', self._op_ts_mean, is_timeseries=True)
        self.register('Ts_Std', self._op_ts_std, is_timeseries=True)
        self.register('Ts_Max', self._op_ts_max, is_timeseries=True)
        self.register('Ts_Min', self._op_ts_min, is_timeseries=True)
        self.register('Ts_Sum', self._op_ts_sum, is_timeseries=True)
        self.register('Ts_Rank', self._op_ts_rank, is_timeseries=True)
        self.register('Ts_Delta', self._op_ts_delta, is_timeseries=True)
        self.register('Ts_Delay', self._op_ts_delay, is_timeseries=True)

        # 算术算子（逐元素）
        self.register('Add', self._op_add, is_timeseries=False)
        self.register('Sub', self._op_sub, is_timeseries=False)
        self.register('Mul', self._op_mul, is_timeseries=False)
        self.register('Div', self._op_div, is_timeseries=False)
        self.register('Abs', self._op_abs, is_timeseries=False)
        self.register('Log', self._op_log, is_timeseries=False)
        self.register('Sign', self._op_sign, is_timeseries=False)

    def register(self, name: str, func: Callable, is_timeseries: bool):
        """注册自定义算子"""
        self.OPERATORS[name] = (func, is_timeseries)

    # ── 截面算子实现 ──
    @staticmethod
    def _op_rank(*args) -> pd.Series:
        x = args[0]
        return x.groupby(level='date').rank(pct=True)

    @staticmethod
    def _op_zscore(*args) -> pd.Series:
        x = args[0]
        g = x.groupby(level='date')
        return (x - g.transform('mean')) / g.transform('std').replace(0, np.nan)

    @staticmethod
    def _op_scale(*args) -> pd.Series:
        x = args[0]
        s = x.groupby(level='date').transform(lambda v: v / v.abs().sum())
        return s

    # ── 时序算子实现 ──
    @staticmethod
    def _to_scalar(v):
        """将常量 Series 提取为标量"""
        if isinstance(v, pd.Series):
            return v.iloc[0]
        return v

    @staticmethod
    def _ts_window(x: pd.Series, window: int, func: str, min_periods: int = 1) -> pd.Series:
        """通用时序滚动窗口"""
        g = x.groupby(level='code')
        if func == 'mean':
            return g.transform(lambda v: v.rolling(window, min_periods=min_periods).mean())
        elif func == 'std':
            return g.transform(lambda v: v.rolling(window, min_periods=min_periods).std())
        elif func == 'max':
            return g.transform(lambda v: v.rolling(window, min_periods=min_periods).max())
        elif func == 'min':
            return g.transform(lambda v: v.rolling(window, min_periods=min_periods).min())
        elif func == 'sum':
            return g.transform(lambda v: v.rolling(window, min_periods=min_periods).sum())
        raise ValueError(f"未知窗口函数: {func}")

    @staticmethod
    def _op_ts_mean(*args) -> pd.Series:
        x = args[0]
        w = int(FactorExpressionEngine._to_scalar(args[1]))
        return FactorExpressionEngine._ts_window(x, w, 'mean')

    @staticmethod
    def _op_ts_std(*args) -> pd.Series:
        x = args[0]
        w = int(FactorExpressionEngine._to_scalar(args[1]))
        return FactorExpressionEngine._ts_window(x, w, 'std', min_periods=max(2, w // 2))

    @staticmethod
    def _op_ts_max(*args) -> pd.Series:
        x = args[0]
        w = int(FactorExpressionEngine._to_scalar(args[1]))
        return FactorExpressionEngine._ts_window(x, w, 'max')

    @staticmethod
    def _op_ts_min(*args) -> pd.Series:
        x = args[0]
        w = int(FactorExpressionEngine._to_scalar(args[1]))
        return FactorExpressionEngine._ts_window(x, w, 'min')

    @staticmethod
    def _op_ts_sum(*args) -> pd.Series:
        x = args[0]
        w = int(FactorExpressionEngine._to_scalar(args[1]))
        return FactorExpressionEngine._ts_window(x, w, 'sum')

    @staticmethod
    def _op_ts_rank(*args) -> pd.Series:
        x = args[0]
        w = int(FactorExpressionEngine._to_scalar(args[1]))
        return x.groupby(level='code').transform(
            lambda v: v.rolling(w, min_periods=1).rank(pct=True)
        )

    @staticmethod
    def _op_ts_delta(*args) -> pd.Series:
        x = args[0]
        w = int(FactorExpressionEngine._to_scalar(args[1]))
        return x.groupby(level='code').transform(lambda v: v.diff(w))

    @staticmethod
    def _op_ts_delay(*args) -> pd.Series:
        x = args[0]
        w = int(FactorExpressionEngine._to_scalar(args[1]))
        return x.groupby(level='code').transform(lambda v: v.shift(w))

    # ── 算术算子实现 ──
    @staticmethod
    def _op_add(*args) -> pd.Series:
        return args[0] + args[1]
    @staticmethod
    def _op_sub(*args) -> pd.Series:
        return args[0] - args[1]
    @staticmethod
    def _op_mul(*args) -> pd.Series:
        return args[0] * args[1]
    @staticmethod
    def _op_div(*args) -> pd.Series:
        return args[0] / args[1].replace(0, np.nan)
    @staticmethod
    def _op_abs(*args) -> pd.Series:
        return args[0].abs()
    @staticmethod
    def _op_log(*args) -> pd.Series:
        return np.log(args[0].clip(lower=1e-12))
    @staticmethod
    def _op_sign(*args) -> pd.Series:
        return np.sign(args[0])

    # ── 表达式解析（递归下降）──
    TOKEN_RE = re.compile(r'\s*(?:(?P<func>[A-Za-z_][A-Za-z0-9_]*)\()|(?P<num>-?\d+\.?\d*)|(?P<field>[A-Za-z_][A-Za-z0-9_]*)|(?P<comma>,)|(?P<rpar>\))')

    def parse(self, expr: str):
        """解析表达式为 AST（嵌套 tuple）"""
        pos = [0]
        tokens = []
        for m in self.TOKEN_RE.finditer(expr):
            if m.group('func'):
                tokens.append(('func', m.group('func')))
            elif m.group('num') is not None:
                tokens.append(('num', m.group('num')))
            elif m.group('field'):
                tokens.append(('field', m.group('field')))
            elif m.group('comma'):
                tokens.append(('comma', ','))
            elif m.group('rpar'):
                tokens.append(('rpar', ')'))

        def parse_expr(i):
            tok = tokens[i]
            if tok[0] == 'func':
                name = tok[1]
                i += 1  # skip func name, expect '('
                # next token should be '(' which is part of func token
                args = []
                arg, i = parse_expr(i)
                args.append(arg)
                while i < len(tokens) and tokens[i][0] == 'comma':
                    i += 1
                    arg, i = parse_expr(i)
                    args.append(arg)
                # expect rpar
                if i < len(tokens) and tokens[i][0] == 'rpar':
                    i += 1
                return (name, args), i
            elif tok[0] == 'num':
                return ('num', float(tok[1])), i + 1
            elif tok[0] == 'field':
                return ('field', tok[1]), i + 1
            else:
                raise SyntaxError(f"无法解析 token: {tok} at {i}")

        ast, _ = parse_expr(0)
        return ast

    def evaluate(self, expr: str, data: pd.DataFrame) -> pd.Series:
        """
        计算表达式，返回因子值 Series

        参数:
            expr: 因子表达式，如 "Rank(Ts_Mean(Close, 5))"
            data: 含 code, date, open, high, low, close, volume 等列的 DataFrame

        返回:
            pd.Series, index 为 MultiIndex(code, date)
        """
        # 准备数据：建立 MultiIndex
        df = data.set_index(['code', 'date']).sort_index()
        ast = self.parse(expr)
        result = self._eval_node(ast, df)
        return result

    def _eval_node(self, node, df: pd.DataFrame) -> pd.Series:
        kind, val = node
        if kind == 'num':
            # 常量：广播为与 df 等长的 Series
            return pd.Series(val, index=df.index)
        elif kind == 'field':
            col = FIELD_MAP.get(val, val.lower())
            if col not in df.columns:
                raise KeyError(f"字段 {val} -> {col} 不在数据列中: {list(df.columns)}")
            return df[col]
        elif kind in self.OPERATORS:
            # 函数节点: node = (op_name, [arg_nodes])
            args_nodes = val
            args = [self._eval_node(a, df) for a in args_nodes]
            func, _ = self.OPERATORS[kind]
            return func(*args)
        else:
            # 大小写不敏感查找（如 Ts_std -> Ts_Std）
            op_lower = kind.lower()
            for op_name in self.OPERATORS:
                if op_name.lower() == op_lower:
                    args_nodes = val
                    args = [self._eval_node(a, df) for a in args_nodes]
                    func, _ = self.OPERATORS[op_name]
                    return func(*args)
            raise KeyError(f"未知算子: {kind}")

    def batch_evaluate(self, expressions: Dict[str, str], data: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算多个表达式因子

        参数:
            expressions: {因子名: 表达式}
            data: 原始行情数据

        返回:
            DataFrame, 含 code, date, [各因子列]
        """
        results = {}
        for name, expr in expressions.items():
            try:
                s = self.evaluate(expr, data)
                results[name] = s
            except Exception as e:
                raise RuntimeError(f"计算因子 {name}='{expr}' 失败: {e}")
        out = pd.DataFrame(results).reset_index()
        return out


# ── 预定义因子库（借鉴 Qlib Alpha158 风格）──
ALPHA158_EXPRESSIONS = {
    # 价格动量/反转
    'km_5': 'Mul(-1, Ts_Mean(Return, 5))',
    'km_20': 'Mul(-1, Ts_Mean(Return, 20))',
    # 波动率
    'vol_20': 'Ts_Std(Return, 20)',
    'vol_60': 'Ts_std(Return, 60)',
    # 振幅
    'amplitude': 'Div(Sub(High, Low), Close)',
    # 成交量比率
    'vol_ratio': 'Div(Volume, Ts_Mean(Volume, 20))',
    # 价格相对位置
    'price_pos': 'Div(Sub(Close, Ts_Min(Close, 20)), Sub(Ts_Max(Close, 20), Ts_Min(Close, 20)))',
    # 换手率动量
    'turnover_mom': 'Ts_Delta(Turnover, 5)',
}
