"""
验证测试：因子表达式引擎
借鉴来源：Microsoft Qlib (github.com/microsoft/qlib) - Expression Engine
优化方向：factor-engine - 声明式因子定义替代硬编码因子计算

Qlib 的核心创新之一是因子表达式引擎，允许用 DSL 语法声明式定义因子：
  $close, Ref($close, 5), Mean($close, 20), Corr($close, $volume, 20)

当前 jingni-trader 的 factor-engine 在 compute_a_share_factors() 中硬编码了
约 15 个因子，扩展性差。本测试验证一个简化版表达式引擎的可行性。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, List, Optional, Union
import time
import re


# ============================================================================
# 1. 表达式引擎核心实现
# ============================================================================

class FactorExpressionEngine:
    """
    简化版因子表达式引擎，支持类 Qlib DSL 语法。

    支持的运算符：
      - $field            : 引用原始数据字段
      - Ref($field, N)    : N 期前值
      - Mean($field, N)   : N 期滚动均值
      - Std($field, N)    : N 期滚动标准差
      - Max($field, N)    : N 期滚动最大值
      - Min($field, N)    : N 期滚动最小值
      - Sum($field, N)    : N 期滚动求和
      - Corr($f1, $f2, N) : $f1 与 $f2 的 N 期滚动相关系数
      - Rank($field)       : 截面排名（百分位）
      - Slope($field, N)   : N 期线性回归斜率
      - $f1 / $f2           : 除法
      - $f1 - $f2           : 减法
      - $f1 + $f2           : 加法
      - $f1 * $f2           : 乘法
      - -$field             : 取负
      - Log($field)         : 自然对数
      - Abs($field)         : 绝对值
      - Delay($field, N)    : 延迟 N 期（同 Ref）
    """

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._cache: Dict[str, pd.Series] = {}

    def set_data(self, df: pd.DataFrame):
        """设置数据源，数据需包含 code, date 列"""
        self._data = df.sort_values(['code', 'date']).copy()
        self._cache = {}

    def _get_field(self, name: str, group_col: str = 'code') -> pd.Series:
        """获取原始字段的 Series"""
        if name in self._data.columns:
            return self._data[name]
        raise KeyError(f"字段 '{name}' 不存在于数据中")

    def _rolling_apply(self, series: pd.Series, window: int, func: Callable,
                       group_col: str = 'code') -> pd.Series:
        """按分组执行滚动计算"""
        return self._data.groupby(group_col)[series.name].transform(
            lambda x: x.rolling(window, min_periods=max(3, window // 2)).apply(func, raw=True)
        )

    def evaluate(self, expr: str) -> pd.Series:
        """
        解析并计算因子表达式。

        示例:
          engine.evaluate("Mean($close, 20) / $close - 1")
          engine.evaluate("-Ref($close, 5) / $close + 1")
          engine.evaluate("Corr($close, $volume, 20)")
        """
        if expr in self._cache:
            return self._cache[expr]

        result = self._parse(expr)
        self._cache[expr] = result
        return result

    def _parse(self, expr: str) -> pd.Series:
        """递归下降解析表达式"""
        expr = expr.strip()

        # 处理括号包裹的表达式
        if expr.startswith('(') and expr.endswith(')'):
            # 验证括号是配对的同一对
            depth = 0
            for i, ch in enumerate(expr):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0 and i == len(expr) - 1:
                        return self._parse(expr[1:-1])
                    elif depth == 0:
                        break

        # 取负运算符
        if expr.startswith('-'):
            # 跳过括号包裹的嵌套表达式前的减号
            inner = expr[1:].strip()
            return -self._parse(inner)

        # 四则运算（从低优先级到高优先级）
        for op in ['+', '-', '*', '/']:
            # 找到最外层（不在括号内的）运算符
            depth = 0
            for i, ch in enumerate(expr):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == op and depth == 0 and i > 0:
                    left = expr[:i].strip()
                    right = expr[i + 1:].strip()
                    if not left or not right:
                        continue
                    left_val = self._parse(left)
                    right_val = self._parse(right)
                    if op == '+':
                        return left_val + right_val
                    elif op == '-':
                        return left_val - right_val
                    elif op == '*':
                        return left_val * right_val
                    elif op == '/':
                        return left_val / right_val.replace(0, np.nan)

        # 函数调用
        func_match = re.match(r'^(\w+)\((.*)\)$', expr)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            args = self._split_args(args_str)
            return self._eval_function(func_name, args)

        # 字段引用
        if expr.startswith('$'):
            field_name = expr[1:]
            return self._get_field(field_name)

        # 数值字面量
        try:
            val = float(expr)
            return pd.Series(val, index=self._data.index)
        except ValueError:
            pass

        raise ValueError(f"无法解析表达式: {expr}")

    def _split_args(self, args_str: str) -> List[str]:
        """按逗号分割参数，处理嵌套括号"""
        args = []
        depth = 0
        current = []
        for ch in args_str:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if ch == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            args.append(''.join(current).strip())
        return args

    def _eval_function(self, func_name: str, args: List[str]) -> pd.Series:
        """执行内置函数"""
        code_col = self._data['code']

        if func_name == 'Ref':
            field, n = self._parse(args[0]), int(args[1])
            result = field.groupby(code_col).shift(n)
            result.name = f"Ref({args[0]},{n})"
            return result

        elif func_name == 'Delay':
            return self._eval_function('Ref', args)

        elif func_name == 'Mean':
            field_series = self._parse(args[0])
            n = int(args[1])
            result = field_series.groupby(code_col).transform(
                lambda x: x.rolling(n, min_periods=max(3, n // 2)).mean()
            )
            result.name = f"Mean({args[0]},{n})"
            return result

        elif func_name == 'Std':
            field_series = self._parse(args[0])
            n = int(args[1])
            result = field_series.groupby(code_col).transform(
                lambda x: x.rolling(n, min_periods=max(5, n // 2)).std()
            )
            result.name = f"Std({args[0]},{n})"
            return result

        elif func_name == 'Max':
            field_series = self._parse(args[0])
            n = int(args[1])
            result = field_series.groupby(code_col).transform(
                lambda x: x.rolling(n, min_periods=max(3, n // 2)).max()
            )
            result.name = f"Max({args[0]},{n})"
            return result

        elif func_name == 'Min':
            field_series = self._parse(args[0])
            n = int(args[1])
            result = field_series.groupby(code_col).transform(
                lambda x: x.rolling(n, min_periods=max(3, n // 2)).min()
            )
            result.name = f"Min({args[0]},{n})"
            return result

        elif func_name == 'Sum':
            field_series = self._parse(args[0])
            n = int(args[1])
            result = field_series.groupby(code_col).transform(
                lambda x: x.rolling(n, min_periods=max(3, n // 2)).sum()
            )
            result.name = f"Sum({args[0]},{n})"
            return result

        elif func_name == 'Corr':
            f1 = self._parse(args[0])
            f2 = self._parse(args[1])
            n = int(args[2])
            combined = pd.DataFrame({'f1': f1, 'f2': f2, 'code': code_col})
            result = combined.groupby('code').apply(
                lambda g: g['f1'].rolling(n, min_periods=max(5, n // 2)).corr(g['f2'])
            ).reset_index(level=0, drop=True)
            result.name = f"Corr({args[0]},{args[1]},{n})"
            return result

        elif func_name == 'Rank':
            field_series = self._parse(args[0])
            result = field_series.groupby(self._data['date']).rank(pct=True)
            result.name = f"Rank({args[0]})"
            return result

        elif func_name == 'Slope':
            field_series = self._parse(args[0])
            n = int(args[1])
            result = field_series.groupby(code_col).transform(
                lambda x: x.rolling(n, min_periods=max(5, n // 2)).apply(
                    lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= 3 else np.nan
                )
            )
            result.name = f"Slope({args[0]},{n})"
            return result

        elif func_name == 'Log':
            field_series = self._parse(args[0])
            result = np.log(field_series.replace(0, np.nan))
            result.name = f"Log({args[0]})"
            return result

        elif func_name == 'Abs':
            field_series = self._parse(args[0])
            result = abs(field_series)
            result.name = f"Abs({args[0]})"
            return result

        else:
            raise ValueError(f"未知函数: {func_name}")

    def batch_evaluate(self, expressions: Dict[str, str]) -> pd.DataFrame:
        """
        批量计算因子表达式，返回 DataFrame。

        参数:
          expressions: {因子名: 表达式} 字典
        返回:
          DataFrame，包含 code, date 及所有因子列
        """
        result = self._data[['code', 'date']].copy()
        for name, expr in expressions.items():
            result[name] = self.evaluate(expr)
        return result

    def to_alpha158_subset(self) -> Dict[str, str]:
        """
        生成 Alpha158 因子集的核心表达式子集（约 30 个因子）。
        参考 Qlib Alpha158 的因子分类体系。
        """
        return {
            # === K线基础因子 ===
            "KMID": "($close - $open) / $open",
            "KLEN": "($high - $low) / $open",
            "KMID2": "($close - $open) / ($high - $low)",
            "KSFT": "(2 * $close - $high - $low) / $open",

            # === 价格趋势因子 ===
            "ROC5": "Ref($close, 5) / $close",
            "ROC10": "Ref($close, 10) / $close",
            "ROC20": "Ref($close, 20) / $close",
            "MA5": "Mean($close, 5) / $close",
            "MA10": "Mean($close, 10) / $close",
            "MA20": "Mean($close, 20) / $close",
            "BETA5": "Slope($close, 5) / $close",
            "BETA10": "Slope($close, 10) / $close",
            "BETA20": "Slope($close, 20) / $close",

            # === 波动率因子 ===
            "STD5": "Std($close, 5) / $close",
            "STD10": "Std($close, 10) / $close",
            "STD20": "Std($close, 20) / $close",
            "MAX5": "Max($high, 5) / $close",
            "MAX10": "Max($high, 10) / $close",
            "MIN5": "Min($low, 5) / $close",
            "MIN10": "Min($low, 10) / $close",

            # === 价量相关性 ===
            "CORR10": "Corr($close, Log($volume + 1), 10)",
            "CORR20": "Corr($close, Log($volume + 1), 20)",
            "CORD10": "Corr($close / Ref($close, 1), Log($volume / Ref($volume, 1) + 1), 10)",
            "CORD20": "Corr($close / Ref($close, 1), Log($volume / Ref($volume, 1) + 1), 20)",

            # === 反转因子 ===
            "REVERSAL_5": "-(Ref($close, 5) / $close - 1)",
            "REVERSAL_10": "-(Ref($close, 10) / $close - 1)",
            "REVERSAL_20": "-(Ref($close, 20) / $close - 1)",

            # === 换手率因子 ===
            "TURN_MA5": "Mean($turnover_rate, 5)",
            "TURN_MA20": "Mean($turnover_rate, 20)",
            "TURN_CHG": "Mean($turnover_rate, 5) / Mean($turnover_rate, 20) - 1",
        }


# ============================================================================
# 2. 测试代码
# ============================================================================

def generate_test_data(n_stocks: int = 10, n_days: int = 252) -> pd.DataFrame:
    """生成模拟的 A 股日线数据"""
    np.random.seed(42)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

    rows = []
    for code in codes:
        start_price = np.random.uniform(8, 50)
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = start_price * np.cumprod(1 + returns)

        df_one = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'close': prices,
            'volume': np.random.lognormal(12, 0.5, n_days),
            'amount': np.random.lognormal(16, 0.5, n_days),
            'turnover_rate': np.random.uniform(0.01, 0.08, n_days),
        })
        rows.append(df_one)

    df = pd.concat(rows, ignore_index=True)
    return df.sort_values(['code', 'date']).reset_index(drop=True)


def test_basic_expression():
    """测试基础表达式解析"""
    print("=" * 60)
    print("测试 1: 基础表达式解析")
    print("=" * 60)

    df = generate_test_data(n_stocks=3, n_days=100)
    engine = FactorExpressionEngine()
    engine.set_data(df)

    # 测试字段引用
    result = engine.evaluate("$close")
    assert len(result) == len(df), f"$close 长度错误: {len(result)} vs {len(df)}"
    assert np.allclose(result.values, df['close'].values), "$close 值不匹配"
    print("  PASS: $close 字段引用")

    # 测试 Ref
    result = engine.evaluate("Ref($close, 5)")
    expected = df.groupby('code')['close'].shift(5)
    assert np.allclose(result.values, expected.values, equal_nan=True), "Ref($close, 5) 不匹配"
    print("  PASS: Ref($close, 5)")

    # 测试 Mean
    result = engine.evaluate("Mean($close, 20)")
    expected = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    assert np.allclose(result.values, expected.values, equal_nan=True), "Mean($close, 20) 不匹配"
    print("  PASS: Mean($close, 20)")

    # 测试复合表达式
    result = engine.evaluate("Mean($close, 20) / $close - 1")
    ma20 = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    expected = ma20 / df['close'] - 1
    assert np.allclose(result.values, expected.values, equal_nan=True), "复合表达式不匹配"
    print("  PASS: Mean($close, 20) / $close - 1")

    print()


def test_all_operators():
    """测试所有内置运算符"""
    print("=" * 60)
    print("测试 2: 所有内置运算符")
    print("=" * 60)

    df = generate_test_data(n_stocks=3, n_days=100)
    engine = FactorExpressionEngine()
    engine.set_data(df)

    expressions = {
        "Std": "Std($close, 20)",
        "Max": "Max($high, 10)",
        "Min": "Min($low, 10)",
        "Sum": "Sum($volume, 5)",
        "Corr": "Corr($close, $volume, 20)",
        "Rank": "Rank($close)",
        "Slope": "Slope($close, 20)",
        "Log": "Log($volume + 1)",
        "Abs": "Abs($close - Ref($close, 1))",
        "Neg": "-Ref($close, 5) / $close + 1",
        "Add": "$high + $low",
        "Sub": "$high - $low",
        "Mul": "$close * $volume",
        "Div": "$close / $open",
    }

    for name, expr in expressions.items():
        try:
            result = engine.evaluate(expr)
            n_valid = result.notna().sum()
            print(f"  PASS: {name:6s} | {expr:40s} | 有效值: {n_valid}/{len(result)}")
        except Exception as e:
            print(f"  FAIL: {name:6s} | {expr:40s} | 错误: {e}")

    print()


def test_alpha158_subset():
    """测试 Alpha158 因子子集批量计算"""
    print("=" * 60)
    print("测试 3: Alpha158 因子子集批量计算")
    print("=" * 60)

    df = generate_test_data(n_stocks=10, n_days=252)
    engine = FactorExpressionEngine()
    engine.set_data(df)

    alpha_factors = engine.to_alpha158_subset()
    print(f"  因子数量: {len(alpha_factors)}")

    start = time.time()
    result_df = engine.batch_evaluate(alpha_factors)
    elapsed = time.time() - start
    print(f"  计算耗时: {elapsed:.3f}s")
    print(f"  结果形状: {result_df.shape}")

    for name in alpha_factors:
        valid_pct = result_df[name].notna().mean() * 100
        print(f"  {name:20s}: 有效率 {valid_pct:5.1f}% | 均值 {result_df[name].mean():.6f}")

    print()


def test_extensibility():
    """测试扩展性：添加自定义因子"""
    print("=" * 60)
    print("测试 4: 扩展性 - 自定义因子定义")
    print("=" * 60)

    df = generate_test_data(n_stocks=5, n_days=100)
    engine = FactorExpressionEngine()
    engine.set_data(df)

    # 模拟用户自定义因子：类似于 Qlib 中通过表达式定义新因子
    custom_factors = {
        "BB_UPPER": "Mean($close, 20) + 2 * Std($close, 20)",
        "BB_LOWER": "Mean($close, 20) - 2 * Std($close, 20)",
        "BB_WIDTH": "4 * Std($close, 20) / Mean($close, 20)",
        "MOM_5": "$close - Ref($close, 5)",
        "MOM_RANK": "Rank($close - Ref($close, 20))",
        "VWAP_DEV": "$close / (Sum($amount, 5) / Sum($volume, 5)) - 1",
        "RET_BIAS_20": "Mean($close, 20) / $close - 1",
        "TURN_IMPACT": "Abs($close - Ref($close, 1)) / $close * Log($volume + 1)",
    }

    print("  自定义因子定义:")
    for name, expr in custom_factors.items():
        print(f"    {name}: {expr[:80]}...")

    result_df = engine.batch_evaluate(custom_factors)
    print(f"\n  结果形状: {result_df.shape}")

    for name in custom_factors:
        valid_pct = result_df[name].notna().mean() * 100
        print(f"  {name:15s}: 有效率 {valid_pct:5.1f}%")

    print()


def test_performance_comparison():
    """性能对比：表达式引擎 vs 硬编码"""
    print("=" * 60)
    print("测试 5: 性能对比 - 表达式引擎 vs 硬编码")
    print("=" * 60)

    df = generate_test_data(n_stocks=50, n_days=252)
    engine = FactorExpressionEngine()
    engine.set_data(df)

    expressions = {
        "ret_5d": "-Ref($close, 5) / $close + 1",
        "ret_20d": "-Ref($close, 20) / $close + 1",
        "reversal_5d": "Ref($close, 5) / $close - 1",
        "reversal_20d": "Ref($close, 20) / $close - 1",
        "volatility_20d": "Std($close, 20)",
        "turnover_20d": "Mean($turnover_rate, 20)",
        "volume_ratio": "$volume / Mean($volume, 20)",
        "ma_20_dev": "Mean($close, 20) / $close - 1",
    }

    # 表达式引擎计算
    start = time.time()
    expr_result = engine.batch_evaluate(expressions)
    expr_time = time.time() - start

    # 硬编码等效计算（模拟 jingni-trader 当前方式）
    start = time.time()
    hardcoded = df[['code', 'date']].copy()
    hardcoded['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    hardcoded['ret_20d'] = df.groupby('code')['close'].pct_change(20)
    hardcoded['reversal_5d'] = -hardcoded['ret_5d']
    hardcoded['reversal_20d'] = -hardcoded['ret_20d']
    hardcoded['volatility_20d'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )
    hardcoded['turnover_20d'] = df.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    hardcoded['volume_ratio'] = df['volume'] / df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    hardcoded['ma_20_dev'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    ) / df['close'] - 1
    hardcoded_time = time.time() - start

    print(f"  表达式引擎耗时: {expr_time:.4f}s")
    print(f"  硬编码耗时:     {hardcoded_time:.4f}s")
    print(f"  速度比:         {hardcoded_time / expr_time:.2f}x")
    print(f"  (注: 表达式引擎首次运行较慢，但优势在于可扩展性和可维护性)")

    # 验证结果一致性
    for col in expressions:
        if col in expr_result.columns and col in hardcoded.columns:
            corr = expr_result[col].corr(hardcoded[col])
            print(f"  {col:20s}: 相关系数 = {corr:.6f}")

    print()


def test_cache_mechanism():
    """测试缓存机制"""
    print("=" * 60)
    print("测试 6: 缓存机制")
    print("=" * 60)

    df = generate_test_data(n_stocks=10, n_days=100)
    engine = FactorExpressionEngine()
    engine.set_data(df)

    expr = "Mean($close, 20) / $close - 1"

    # 首次计算
    start = time.time()
    _ = engine.evaluate(expr)
    first_time = time.time() - start

    # 缓存命中
    start = time.time()
    _ = engine.evaluate(expr)
    cached_time = time.time() - start

    print(f"  首次计算耗时: {first_time:.6f}s")
    print(f"  缓存命中耗时: {cached_time:.6f}s")
    print(f"  加速比:       {first_time / cached_time:.0f}x" if cached_time > 0 else "  加速比: 极大")
    print()

    cache_size = len(engine._cache)
    print(f"  缓存条目数: {cache_size}")
    print()


# ============================================================================
# 主测试入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("因子表达式引擎验证测试")
    print("借鉴来源: Microsoft Qlib Expression Engine")
    print("优化方向: factor-engine - 声明式因子定义")
    print("=" * 60 + "\n")

    test_basic_expression()
    test_all_operators()
    test_alpha158_subset()
    test_extensibility()
    test_performance_comparison()
    test_cache_mechanism()

    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)