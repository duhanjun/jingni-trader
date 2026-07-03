"""
因子表达式引擎 (Factor Expression Engine)

借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
核心思想: 将因子定义为表达式字符串 (如 "Ref($close, 20) / $close")，
         通过 AST 解析为算子树，递归计算，并自动处理回看窗口对齐。

与 jingni-trader 现有 factor-engine/engine.py 的对比:
  - 原实现: 因子逻辑硬编码在 compute_a_share_factors() 方法中，
            每个因子是一段手写 pandas 代码，新增因子需修改源码
  - 本实现: 因子声明式定义，表达式即因子，支持组合嵌套，
            算子自报告回看窗口，自动扩展数据窗口避免边界 NaN

Qlib 表达式引擎的三个关键设计:
  1. Parse: 表达式字符串 -> AST
  2. Convert: AST -> 算子树 (ExpressionOps)
  3. Execute: 递归计算，每个算子报告自身回看窗口

本文件为验证性实现，不修改 main 分支的任何代码。
"""
from __future__ import annotations

import re
import ast
import logging
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("factor-expression-engine")


# ============================================================
# 算子基类 (借鉴 Qlib ExpressionOps 层次结构)
# ============================================================

class Operator:
    """算子基类，所有算子必须实现 load() 和 get_longest_back_rolling()"""

    def load(self, df: pd.DataFrame, code: str) -> pd.Series:
        """在单只股票的 DataFrame 上计算该算子，返回时间序列"""
        raise NotImplementedError

    def get_longest_back_rolling(self) -> int:
        """返回该算子需要的最大回看窗口 (用于自动扩展数据窗口)"""
        return 0

    def get_feature_names(self) -> List[str]:
        """返回该算子依赖的原始字段名 (如 ['close', 'volume'])"""
        return []


class Feature(Operator):
    """原始字段引用，如 $close, $volume, $open"""

    def __init__(self, name: str):
        self.name = name

    def load(self, df: pd.DataFrame, code: str) -> pd.Series:
        col = self.name
        if col not in df.columns:
            raise KeyError(f"字段 '{col}' 不存在于数据中，可用字段: {list(df.columns)}")
        return df[col].astype(float)

    def get_longest_back_rolling(self) -> int:
        return 0

    def get_feature_names(self) -> List[str]:
        return [self.name]

    def __repr__(self):
        return f"${self.name}"


class ElemOperator(Operator):
    """一元算子: Log, Abs, Neg, Sign, Rank"""

    def __init__(self, operand: Operator):
        self.operand = operand

    def get_longest_back_rolling(self) -> int:
        return self.operand.get_longest_back_rolling()

    def get_feature_names(self) -> List[str]:
        return self.operand.get_feature_names()


class PairOperator(Operator):
    """二元算子: Add, Sub, Mul, Div"""

    def __init__(self, left: Operator, right: Operator):
        self.left = left
        self.right = right

    def get_longest_back_rolling(self) -> int:
        return max(self.left.get_longest_back_rolling(), self.right.get_longest_back_rolling())

    def get_feature_names(self) -> List[str]:
        return list(set(self.left.get_feature_names() + self.right.get_feature_names()))


class Rolling(Operator):
    """滚动窗口算子: Ref, Mean, Std, Max, Min, Sum, Quantile"""

    def __init__(self, operand: Operator, window: int):
        self.operand = operand
        self.window = int(window)

    def get_longest_back_rolling(self) -> int:
        return self.operand.get_longest_back_rolling() + self.window

    def get_feature_names(self) -> List[str]:
        return self.operand.get_feature_names()


# ---- 一元算子实现 ----

class Log(ElemOperator):
    def load(self, df, code):
        v = self.operand.load(df, code)
        return np.log(v.where(v > 0, np.nan))

class Abs(ElemOperator):
    def load(self, df, code):
        return self.operand.load(df, code).abs()

class Neg(ElemOperator):
    def load(self, df, code):
        return -self.operand.load(df, code)

class Sign(ElemOperator):
    def load(self, df, code):
        return np.sign(self.operand.load(df, code))

class Rank(ElemOperator):
    """截面排名 (按日期分组，pct rank)"""
    def load(self, df, code):
        return self.operand.load(df, code)  # 单只股票内返回原值，截面排名在引擎层处理


# ---- 二元算子实现 ----

class Add(PairOperator):
    def load(self, df, code):
        return self.left.load(df, code) + self.right.load(df, code)

class Sub(PairOperator):
    def load(self, df, code):
        return self.left.load(df, code) - self.right.load(df, code)

class Mul(PairOperator):
    def load(self, df, code):
        return self.left.load(df, code) * self.right.load(df, code)

class Div(PairOperator):
    def load(self, df, code):
        l = self.left.load(df, code)
        r = self.right.load(df, code)
        return l / r.replace(0, np.nan)

class Greater(PairOperator):
    def load(self, df, code):
        return np.maximum(self.left.load(df, code), self.right.load(df, code))

class Less(PairOperator):
    def load(self, df, code):
        return np.minimum(self.left.load(df, code), self.right.load(df, code))


# ---- 滚动窗口算子实现 ----

class Ref(Rolling):
    """引用 N 天前的值: Ref($close, 5) = 5天前的收盘价"""
    def load(self, df, code):
        return self.operand.load(df, code).shift(self.window)

class Mean(Rolling):
    """滚动均值"""
    def load(self, df, code):
        return self.operand.load(df, code).rolling(self.window, min_periods=max(1, self.window // 2)).mean()

class Std(Rolling):
    """滚动标准差"""
    def load(self, df, code):
        return self.operand.load(df, code).rolling(self.window, min_periods=max(1, self.window // 2)).std()

class Max(Rolling):
    def load(self, df, code):
        return self.operand.load(df, code).rolling(self.window, min_periods=max(1, self.window // 2)).max()

class Min(Rolling):
    def load(self, df, code):
        return self.operand.load(df, code).rolling(self.window, min_periods=max(1, self.window // 2)).min()

class Sum(Rolling):
    def load(self, df, code):
        return self.operand.load(df, code).rolling(self.window, min_periods=max(1, self.window // 2)).sum()

class Quantile(Rolling):
    def __init__(self, operand, window, qscore=0.5):
        super().__init__(operand, window)
        self.qscore = qscore
    def load(self, df, code):
        return self.operand.load(df, code).rolling(self.window, min_periods=max(1, self.window // 2)).quantile(self.qscore)

class Delta(Rolling):
    """差分: Delta($close, 5) = $close - Ref($close, 5)"""
    def load(self, df, code):
        v = self.operand.load(df, code)
        return v - v.shift(self.window)

class Slope(Rolling):
    """滚动线性回归斜率"""
    def load(self, df, code):
        v = self.operand.load(df, code)
        x = np.arange(self.window)
        def _slope(arr):
            if len(arr) < self.window or np.isnan(arr).any():
                return np.nan
            y = arr - arr.mean()
            return np.dot(x - x.mean(), y) / np.dot(x - x.mean(), x - x.mean())
        return v.rolling(self.window).apply(_slope, raw=True)

class WMA(Rolling):
    """加权移动平均 (近期权重更高)"""
    def load(self, df, code):
        v = self.operand.load(df, code)
        weights = np.arange(1, self.window + 1, dtype=float)
        weights /= weights.sum()
        def _wma(arr):
            if len(arr) < self.window:
                return np.nan
            return np.dot(arr, weights)
        return v.rolling(self.window).apply(_wma, raw=True)

class Corr(Rolling):
    """滚动相关系数 (需要两个操作数)"""
    def __init__(self, left: Operator, right: Operator, window: int):
        super().__init__(left, window)
        self.right_op = right
    def get_longest_back_rolling(self) -> int:
        return max(self.operand.get_longest_back_rolling(), self.right_op.get_longest_back_rolling()) + self.window
    def get_feature_names(self) -> List[str]:
        return list(set(self.operand.get_feature_names() + self.right_op.get_feature_names()))
    def load(self, df, code):
        l = self.operand.load(df, code)
        r = self.right_op.load(df, code)
        return l.rolling(self.window, min_periods=max(2, self.window // 2)).corr(r)


# ============================================================
# 算子注册表 (借鉴 Qlib register_all_ops)
# ============================================================

OPERATOR_REGISTRY: Dict[str, Type[Operator]] = {
    # 一元
    'Log': Log, 'Abs': Abs, 'Neg': Neg, 'Sign': Sign, 'Rank': Rank,
    # 二元
    'Add': Add, 'Sub': Sub, 'Mul': Mul, 'Div': Div,
    'Greater': Greater, 'Less': Less,
    # 滚动
    'Ref': Ref, 'Mean': Mean, 'Std': Std, 'Max': Max, 'Min': Min,
    'Sum': Sum, 'Quantile': Quantile, 'Delta': Delta, 'Slope': Slope,
    'WMA': WMA, 'Corr': Corr,
}


def register_operator(name: str, op_class: Type[Operator]):
    """注册自定义算子"""
    OPERATOR_REGISTRY[name] = op_class


# ============================================================
# 表达式解析器 (Parse: string -> AST -> Operator tree)
# ============================================================

class ExpressionParser:
    """
    解析因子表达式字符串为算子树

    支持语法:
      $close, $volume          - 原始字段
      Ref($close, 20)          - 函数调用
      Ref($close, 20) / $close - 二元运算
      $close + $open           - 简单运算
      (a + b) * c              - 括号
      0.5                      - 常数
    """

    # $field 模式
    FEATURE_PATTERN = re.compile(r'\$(\w+)')

    def parse(self, expr: str) -> Operator:
        """解析表达式字符串，返回算子树根节点"""
        expr = expr.strip()
        return self._parse_expression(expr)

    def _parse_expression(self, expr: str) -> Operator:
        """递归下降解析器"""
        expr = expr.strip()

        # 一元负号: 仅否定紧随其后的原子项 (函数调用/$字段/常数/括号)
        # 例如 -Ref($close,5)/$close => (Neg(Ref(...))) / $close
        if expr.startswith('-'):
            atom_str, rest = self._extract_atom(expr[1:].strip())
            inner = self._parse_expression(atom_str)
            negated = Neg(inner)
            rest = rest.strip()
            if not rest:
                return negated
            # 剩余部分以二元运算符开头，组合
            return self._combine_binary(negated, rest)

        # 尝试二元运算 (按优先级从低到高)
        for op_symbol, op_class in [('+', Add), ('-', Sub)]:
            result = self._try_binary_op(expr, op_symbol, op_class)
            if result is not None:
                return result

        for op_symbol, op_class in [('*', Mul), ('/', Div)]:
            result = self._try_binary_op(expr, op_symbol, op_class)
            if result is not None:
                return result

        # 括号
        if expr.startswith('(') and expr.endswith(')'):
            return self._parse_expression(expr[1:-1])

        # 函数调用: Name(args)
        func_match = re.match(r'^(\w+)\((.+)\)$', expr)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            return self._parse_function_call(func_name, args_str)

        # 字段引用: $close
        feat_match = re.match(r'^\$(\w+)$', expr)
        if feat_match:
            return Feature(feat_match.group(1))

        # 常数
        try:
            val = float(expr)
            return Constant(val)
        except ValueError:
            pass

        raise ValueError(f"无法解析表达式: {expr}")

    def _extract_atom(self, expr: str) -> tuple:
        """
        提取表达式开头的原子项，返回 (atom_str, rest_str)
        原子项: 函数调用 / $字段 / 常数 / 括号表达式
        """
        expr = expr.strip()
        if not expr:
            raise ValueError("空表达式")

        # 括号表达式
        if expr[0] == '(':
            depth = 0
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        return expr[:i+1], expr[i+1:]
            raise ValueError(f"括号不匹配: {expr}")

        # 函数调用: Name(...)
        func_match = re.match(r'^(\w+)\(', expr)
        if func_match:
            depth = 0
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        return expr[:i+1], expr[i+1:]
            raise ValueError(f"函数括号不匹配: {expr}")

        # $字段
        feat_match = re.match(r'^\$\w+', expr)
        if feat_match:
            end = feat_match.end()
            return expr[:end], expr[end:]

        # 常数 (含小数)
        num_match = re.match(r'^\d+\.?\d*', expr)
        if num_match:
            end = num_match.end()
            return expr[:end], expr[end:]

        raise ValueError(f"无法提取原子项: {expr}")

    def _combine_binary(self, left_op: Operator, rest: str) -> Operator:
        """将已解析的左操作数与剩余的二元运算表达式组合"""
        rest = rest.strip()
        if not rest:
            return left_op
        # 剩余以二元运算符开头
        if rest[0] in '+-*/':
            symbol = rest[0]
            right = self._parse_expression(rest[1:].strip())
            op_map = {'+': Add, '-': Sub, '*': Mul, '/': Div}
            return op_map[symbol](left_op, right)
        raise ValueError(f"剩余表达式无法解析为二元运算: {rest}")

    def _try_binary_op(self, expr: str, symbol: str, op_class: Type[PairOperator]) -> Optional[Operator]:
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            c = expr[i]
            if c == ')':
                depth += 1
            elif c == '(':
                depth -= 1
            elif depth == 0 and c == symbol and i > 0:
                # 确保不是负号 (前面是运算符或开头)
                prev = expr[i - 1]
                if symbol == '-' and prev in '+-*/(':
                    continue
                left = self._parse_expression(expr[:i])
                right = self._parse_expression(expr[i + 1:])
                return op_class(left, right)
        return None

    def _parse_function_call(self, func_name: str, args_str: str) -> Operator:
        """解析函数调用"""
        if func_name not in OPERATOR_REGISTRY:
            raise ValueError(f"未知算子: {func_name}，已注册: {list(OPERATOR_REGISTRY.keys())}")

        op_class = OPERATOR_REGISTRY[func_name]
        args = self._split_args(args_str)

        # 滚动窗口算子: 第一个参数是操作数，第二个是窗口大小
        if issubclass(op_class, Rolling) and op_class != Corr:
            if len(args) < 2:
                raise ValueError(f"{func_name} 需要至少 2 个参数: (操作数, 窗口)")
            operand = self._parse_expression(args[0])
            window = int(float(args[1].strip()))
            if func_name == 'Quantile' and len(args) >= 3:
                return op_class(operand, window, float(args[2].strip()))
            return op_class(operand, window)

        # Corr 需要两个操作数 + 窗口
        if op_class == Corr:
            if len(args) < 3:
                raise ValueError("Corr 需要 3 个参数: (left, right, window)")
            left = self._parse_expression(args[0])
            right = self._parse_expression(args[1])
            window = int(float(args[2].strip()))
            return op_class(left, right, window)

        # 一元算子
        if issubclass(op_class, ElemOperator):
            operand = self._parse_expression(args[0])
            return op_class(operand)

        raise ValueError(f"无法处理算子 {func_name} 的参数")

    def _split_args(self, args_str: str) -> List[str]:
        """按逗号分割参数 (考虑括号嵌套)"""
        args = []
        depth = 0
        current = []
        for c in args_str:
            if c == '(':
                depth += 1
                current.append(c)
            elif c == ')':
                depth -= 1
                current.append(c)
            elif c == ',' and depth == 0:
                args.append(''.join(current))
                current = []
            else:
                current.append(c)
        if current:
            args.append(''.join(current))
        return args


class Constant(Operator):
    """常数算子"""
    def __init__(self, value: float):
        self.value = float(value)
    def load(self, df, code):
        s = pd.Series(self.value, index=df.index, dtype=float)
        return s
    def get_feature_names(self):
        return []


# ============================================================
# 因子表达式引擎
# ============================================================

class FactorExpressionEngine:
    """
    因子表达式引擎

    借鉴 Qlib 的设计:
    1. 解析表达式字符串为算子树
    2. 自动检测回看窗口，扩展数据窗口避免边界 NaN
    3. 逐股票计算 (向量化 pandas 操作)
    4. 支持截面操作 (Rank 在截面层处理)

    用法:
        engine = FactorExpressionEngine()
        factor = engine.compute("Ref($close, 20) / $close - 1", data)
        # data 是包含 code, date, open, high, low, close, volume 等列的 DataFrame
    """

    def __init__(self):
        self.parser = ExpressionParser()

    def compute(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """
        计算因子表达式

        参数:
            expression: 因子表达式字符串
            data: 原始数据 DataFrame，需包含 code, date 列及行情字段

        返回:
            pd.Series, 索引与 data 对齐，值为因子值
        """
        logger.info(f"计算因子表达式: {expression}")
        op_tree = self.parser.parse(expression)
        lookback = op_tree.get_longest_back_rolling()
        features = op_tree.get_feature_names()

        missing = [f for f in features if f not in data.columns]
        if missing:
            raise ValueError(f"数据缺少字段: {missing}，数据现有字段: {list(data.columns)}")

        logger.info(f"算子树回看窗口: {lookback}, 依赖字段: {features}")

        results = []
        for code, group in data.groupby('code'):
            # 保存原始索引，排序计算后恢复，确保与原 data 对齐
            orig_index = group.index
            sorted_group = group.sort_values('date').reset_index(drop=True)
            try:
                values = op_tree.load(sorted_group, code)
                values.name = expression
                values.index = orig_index  # 恢复原始索引
                results.append(values)
            except Exception as e:
                logger.warning(f"计算股票 {code} 因子失败: {e}")
                values = pd.Series(np.nan, index=orig_index, name=expression)
                results.append(values)

        result = pd.concat(results)
        return result

    def compute_batch(self, expressions: Dict[str, str], data: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算多个因子表达式

        参数:
            expressions: {因子名: 表达式字符串}
            data: 原始数据

        返回:
            DataFrame, 列为因子名，索引与 data 对齐
        """
        results = {}
        for name, expr in expressions.items():
            try:
                results[name] = self.compute(expr, data)
            except Exception as e:
                logger.error(f"因子 {name} ({expr}) 计算失败: {e}")
                results[name] = pd.Series(np.nan, index=data.index, name=name)
        return pd.DataFrame(results)

    def cross_section_rank(self, factor: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        截面排名 (借鉴 Qlib Rank 算子)
        按日期分组，对因子值做百分位排名
        """
        df = data[['code', 'date']].copy()
        df['factor'] = factor.values
        df['rank'] = df.groupby('date')['factor'].rank(pct=True)
        return df['rank']


# ============================================================
# 预定义因子库 (Alpha101 风格, 借鉴 Qlib Alpha158)
# ============================================================

PREDEFINED_FACTORS: Dict[str, str] = {
    # 反转因子
    'reversal_5d':  '-Ref($close, 5) / $close',
    'reversal_20d': '-Ref($close, 20) / $close',
    'reversal_60d': '-Ref($close, 60) / $close',

    # 动量因子
    'momentum_20d': 'Ref($close, 20) / $close - 1',

    # 波动率因子
    'volatility_20d': 'Std($close / Ref($close, 1) - 1, 20)',
    'volatility_60d': 'Std($close / Ref($close, 1) - 1, 60)',

    # 均线偏离因子
    'ma5_bias':  '$close / Mean($close, 5) - 1',
    'ma20_bias': '$close / Mean($close, 20) - 1',
    'ma60_bias': '$close / Mean($close, 60) - 1',

    # 成交量因子
    'volume_ratio_5_20': 'Mean($volume, 5) / Mean($volume, 20)',
    'volume_ratio_1_20': '$volume / Mean($volume, 20)',

    # 振幅因子
    'amplitude_20d': 'Mean(($high - $low) / $close, 20)',

    # 换手率因子 (需数据含 turnover_rate 字段)
    'turnover_20d': 'Mean($turnover_rate, 20)',

    # BBI 因子 (4条均线合成)
    'bbi_bias': '$close / (Mean($close, 3) + Mean($close, 6) + Mean($close, 12) + Mean($close, 24)) * 4 - 1',

    # 价格加速度
    'price_accel': 'Delta($close, 5) / Delta(Ref($close, 5), 5)',

    # 量价相关
    'vp_corr_20d': 'Corr($close, $volume, 20)',

    # 加权移动平均偏离
    'wma20_bias': '$close / WMA($close, 20) - 1',
}
