"""
优化方向：因子表达式引擎 (Factor Expression Engine)
借鉴来源：microsoft/qlib (https://github.com/microsoft/qlib)
         Qlib 的表达式引擎允许用 DSL 声明因子，如 $close, Ref($close, 1), Mean($close, 3)

核心亮点：
  - Qlib 将因子声明为函数而非数据，使得 LLM 可生成因子表达式
  - 表达式引擎自动处理缓存和列式计算
  - 支持嵌套表达式，如 Mean($high - $low, 5)

本测试验证：
  1. 表达式解析器的正确性
  2. 与现有硬编码因子计算的输出一致性
  3. 表达式引擎的性能表现
  4. 嵌套表达式的处理能力
"""

import sys
import os
import re
import time
import json
from typing import Dict, List, Callable, Any, Union, Optional
from functools import lru_cache

sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd


# ============================================================
# 1. 因子表达式引擎原型实现
# ============================================================

# 表达式语法定义:
#   $open, $high, $low, $close, $volume, $amount       -- 原始字段
#   Ref(expr, N)                                        -- 前 N 期值
#   Mean(expr, N)                                       -- N 期滚动均值
#   Std(expr, N)                                        -- N 期滚动标准差
#   Sum(expr, N)                                        -- N 期滚动求和
#   Max(expr, N)                                        -- N 期滚动最大值
#   Min(expr, N)                                        -- N 期滚动最小值
#   PctChange(expr, N)                                  -- N 期变化率
#   Corr(expr1, expr2, N)                               -- 两表达式 N 期滚动相关系数
#   Rank(expr)                                          -- 截面排名
#   二元运算: expr + expr, expr - expr, expr * expr, expr / expr
#   一元运算: -expr, Abs(expr), Log(expr), Sign(expr)
#   条件: If(cond_expr, true_expr, false_expr)          -- 条件选择


class FactorExpressionParser:
    """因子表达式解析器 - 将 DSL 字符串解析为可执行的操作树"""

    # Token 模式
    FIELD_PATTERN = re.compile(r'\$(open|high|low|close|volume|amount|vwap|turnover|pre_close)')

    def __init__(self):
        self._field_map = {
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
            'volume': 'volume', 'amount': 'amount', 'vwap': 'vwap',
            'turnover': 'turnover', 'pre_close': 'pre_close'
        }

    def parse(self, expr: str) -> Dict[str, Any]:
        """
        解析表达式字符串为 AST

        返回: {'type': 'op'|'field'|'func', ...}
        """
        expr = expr.strip()
        return self._parse_expr(expr)

    def _parse_expr(self, expr: str) -> Dict[str, Any]:
        """递归下降解析"""
        expr = expr.strip()

        # 处理括号包裹
        if expr.startswith('(') and expr.endswith(')'):
            depth = 0
            for i, ch in enumerate(expr):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                if depth == 0 and i == len(expr) - 1:
                    return self._parse_expr(expr[1:-1])
                if depth == 0:
                    break

        # 处理一元负号 (表达式开头为 -)
        if expr.startswith('-'):
            inner = self._parse_expr(expr[1:].strip())
            return {'type': 'op', 'op': 'neg', 'args': [inner]}

        # 处理函数调用: FuncName(args)
        func_match = re.match(r'^(\w+)\s*\((.*)\)\s*$', expr)
        if func_match and self._is_balanced(expr):
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            args = self._split_args(args_str)
            parsed_args = [self._parse_expr(a.strip()) for a in args]
            return {'type': 'func', 'func': func_name, 'args': parsed_args}

        # 处理字段: $open, $close, etc.
        field_match = self.FIELD_PATTERN.fullmatch(expr)
        if field_match:
            return {'type': 'field', 'field': self._field_map.get(field_match.group(1), field_match.group(1))}

        # 处理二元运算符 (按优先级: +, - 最低)
        for op in ['+', '-']:
            parts = self._split_by_op(expr, op)
            if len(parts) > 1:
                left = self._parse_expr(parts[0].strip())
                right = self._parse_expr(op.join(parts[1:]).strip())
                return {'type': 'op', 'op': op, 'args': [left, right]}

        for op in ['*', '/']:
            parts = self._split_by_op(expr, op)
            if len(parts) > 1:
                left = self._parse_expr(parts[0].strip())
                right = self._parse_expr(op.join(parts[1:]).strip())
                return {'type': 'op', 'op': op, 'args': [left, right]}

        # 处理数字常量
        try:
            val = float(expr)
            return {'type': 'const', 'value': val}
        except ValueError:
            pass

        raise ValueError(f"无法解析表达式: {expr}")

    def _is_balanced(self, expr: str) -> bool:
        """检查是否是完整的函数调用：FuncName(balanced_args)"""
        # Find the first '('
        try:
            paren_start = expr.index('(')
        except ValueError:
            return False

        # Everything before '(' should be a function name (word chars only)
        prefix = expr[:paren_start]
        if not re.match(r'^\w+$', prefix):
            return False

        # Must end with ')'
        if not expr.endswith(')'):
            return False

        # Check that parentheses close precisely at the end
        depth = 0
        opened = False
        for i, ch in enumerate(expr):
            if ch == '(':
                depth += 1
                opened = True
            elif ch == ')':
                depth -= 1
            if opened and depth == 0 and i < len(expr) - 1:
                # Closed before the end -> not a single function call
                return False
        return depth == 0

    def _split_args(self, args_str: str) -> List[str]:
        """按逗号分隔函数参数，正确处理嵌套"""
        parts = []
        depth = 0
        current = []
        for ch in args_str:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts

    def _split_by_op(self, expr: str, op: str) -> List[str]:
        """按运算符分割，正确处理嵌套"""
        parts = []
        depth = 0
        last = 0
        for i, ch in enumerate(expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == op and depth == 0 and i > 0:
                # 对于 +, *, /，只要不在括号内就允许分割
                # 对于 -，需要区分负号和减号
                if op == '-':
                    if expr[i-1] not in '(,+-*/':
                        parts.append(expr[last:i])
                        last = i + 1
                else:
                    parts.append(expr[last:i])
                    last = i + 1
        parts.append(expr[last:])
        return parts


class FactorExpressionEvaluator:
    """因子表达式求值器 - 将 AST 转换为 pandas 操作"""

    def __init__(self, parser: FactorExpressionParser = None):
        self.parser = parser or FactorExpressionParser()
        self._cache = {}

    def evaluate(self, expr: str, data: pd.DataFrame, group_col: str = 'code') -> pd.Series:
        """
        对数据执行因子表达式求值

        参数:
            expr: 因子表达式字符串
            data: 行情数据 DataFrame，须包含 date, code 及 OHLCV 列
            group_col: 分组列名

        返回:
            pd.Series, 与 data 同 index
        """
        cache_key = (expr, id(data))
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        ast = self.parser.parse(expr) if isinstance(expr, str) else expr
        result = self._eval_node(ast, data, group_col)

        # 缓存（仅限非空结果）
        if result is not None and len(result) > 0:
            self._cache[cache_key] = result.copy()

        return result

    def _eval_node(self, node: Dict, data: pd.DataFrame, group_col: str) -> pd.Series:
        """递归求值 AST 节点"""
        node_type = node['type']

        if node_type == 'field':
            field = node['field']
            if field in data.columns:
                return data[field].copy()
            raise KeyError(f"数据中不包含字段: {field}")

        elif node_type == 'const':
            val = node['value']
            return pd.Series(val, index=data.index)

        elif node_type == 'op':
            op = node['op']
            args = [self._eval_node(a, data, group_col) for a in node['args']]

            if op == 'neg':
                return -args[0]
            elif op == '+':
                return args[0] + args[1]
            elif op == '-':
                return args[0] - args[1]
            elif op == '*':
                return args[0] * args[1]
            elif op == '/':
                return args[0] / args[1].replace(0, np.nan)
            else:
                raise ValueError(f"未知运算符: {op}")

        elif node_type == 'func':
            func_name = node['func']
            args = [self._eval_node(a, data, group_col) for a in node['args']]

            return self._apply_func(func_name, args, data, group_col)

        raise ValueError(f"未知节点类型: {node_type}")

    def _apply_func(self, func_name: str, args: List[pd.Series],
                    data: pd.DataFrame, group_col: str) -> pd.Series:
        """应用函数到参数"""
        lookup = {
            'Ref': self._func_ref,
            'Mean': self._func_rolling,
            'Std': self._func_rolling,
            'Sum': self._func_rolling,
            'Max': self._func_rolling,
            'Min': self._func_rolling,
            'PctChange': self._func_pct_change,
            'Corr': self._func_corr,
            'Abs': self._func_abs,
            'Log': self._func_log,
            'Sign': self._func_sign,
            'Rank': self._func_rank,
            'If': self._func_if,
        }
        if func_name not in lookup:
            raise ValueError(f"未知函数: {func_name}")
        return lookup[func_name](args, data, group_col, func_name)

    def _func_ref(self, args, data, group_col, func_name):
        """Ref(expr, N) -- 前 N 期值"""
        expr, n = args[0], int(self._extract_const(args[1]))
        return expr.groupby(data[group_col]).shift(n)

    def _func_rolling(self, args, data, group_col, func_name):
        """Mean/Std/Sum/Max/Min(expr, N) -- N 期滚动"""
        expr, n = args[0], int(self._extract_const(args[1]))
        roll = expr.groupby(data[group_col]).rolling(n, min_periods=max(3, n // 2))
        method_map = {
            'Mean': 'mean', 'Std': 'std', 'Sum': 'sum',
            'Max': 'max', 'Min': 'min'
        }
        method = method_map[func_name]
        result = getattr(roll, method)()
        # reset multi-index back to original index
        result = result.reset_index(level=0, drop=True)
        result = result.reindex(data.index)
        return result

    def _func_pct_change(self, args, data, group_col, func_name):
        """PctChange(expr, N) -- N 期变化率"""
        expr, n = args[0], int(self._extract_const(args[1]))
        return expr.groupby(data[group_col]).pct_change(n)

    def _func_corr(self, args, data, group_col, func_name):
        """Corr(expr1, expr2, N) -- 两表达式 N 期滚动相关系数"""
        e1, e2, n = args[0], args[1], int(self._extract_const(args[2]))
        result = pd.Series(index=data.index, dtype=float)
        for code, grp in data.groupby(group_col):
            idx = grp.index
            r1 = e1.loc[idx]
            r2 = e2.loc[idx]
            corr = r1.rolling(n, min_periods=max(5, n // 2)).corr(r2)
            result.loc[idx] = corr.values
        return result

    def _func_abs(self, args, data, group_col, func_name):
        return args[0].abs()

    def _func_log(self, args, data, group_col, func_name):
        return np.log(args[0].clip(lower=1e-10))

    def _func_sign(self, args, data, group_col, func_name):
        return np.sign(args[0])

    def _func_rank(self, args, data, group_col, func_name):
        """Rank(expr) -- 截面排名 (percentile)"""
        expr = args[0]
        df = pd.DataFrame({'date': data['date'], 'val': expr})
        result = df.groupby('date')['val'].rank(pct=True)
        return result.reindex(data.index)

    def _func_if(self, args, data, group_col, func_name):
        """If(cond, true_expr, false_expr)"""
        cond, true_val, false_val = args[0], args[1], args[2]
        return pd.Series(np.where(cond > 0, true_val, false_val), index=data.index)

    def _extract_const(self, node_or_series) -> float:
        """从 AST 节点提取常量"""
        if isinstance(node_or_series, pd.Series):
            return node_or_series.iloc[0] if len(node_or_series) > 0 else 0
        if isinstance(node_or_series, dict) and node_or_series.get('type') == 'const':
            return node_or_series['value']
        return float(node_or_series)


# ============================================================
# 2. 测试用例
# ============================================================

def make_test_data(n_codes: int = 10, n_days: int = 500) -> pd.DataFrame:
    """生成测试用行情数据"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    codes = [f'{600000 + i:06d}.SH' for i in range(n_codes)]

    records = []
    for code in codes:
        price = 10 + np.random.randn() * 5
        for date in dates:
            ret = np.random.randn() * 0.02
            price = price * (1 + ret)
            open_p = price * (1 + np.random.randn() * 0.005)
            high_p = max(open_p, price) * (1 + abs(np.random.randn() * 0.01))
            low_p = min(open_p, price) * (1 - abs(np.random.randn() * 0.01))
            close_p = price
            volume = np.random.lognormal(10, 0.5)
            records.append({
                'date': date, 'code': code,
                'open': open_p, 'high': high_p, 'low': low_p,
                'close': close_p, 'volume': volume,
                'amount': close_p * volume,
            })

    df = pd.DataFrame(records).sort_values(['code', 'date']).reset_index(drop=True)
    return df


def test_parser_correctness():
    """测试1: 表达式解析器正确性"""
    print("\n" + "=" * 60)
    print("测试1: 表达式解析器正确性")
    print("=" * 60)

    parser = FactorExpressionParser()
    test_cases = [
        ("$close", "field"),
        ("Ref($close, 1)", "func"),
        ("Mean($close, 20)", "func"),
        ("$high - $low", "op"),
        ("$close / Ref($close, 1) - 1", "op"),
        ("Mean($high - $low, 5)", "func"),
        ("Std($close, 20)", "func"),
        ("Corr($close, $volume, 20)", "func"),
        ("-($close - Ref($close, 1))", "op"),
        ("Rank(Mean($close, 20))", "func"),
        ("Abs($close - Ref($close, 1))", "func"),
        ("If($close - Ref($close, 5), $close, Ref($close, 1))", "func"),
    ]

    all_passed = True
    for expr, expected_type in test_cases:
        try:
            ast = parser.parse(expr)
            actual_type = ast['type']
            status = "PASS" if actual_type == expected_type else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  {status}: '{expr}' -> type={actual_type} (expected {expected_type})")
        except Exception as e:
            all_passed = False
            print(f"  FAIL: '{expr}' -> 解析异常: {e}")

    return all_passed


def test_evaluator_correctness():
    """测试2: 表达式求值器输出一致性 (与硬编码因子对比)"""
    print("\n" + "=" * 60)
    print("测试2: 表达式求值器 vs 硬编码因子输出一致性")
    print("=" * 60)

    df = make_test_data(n_codes=5, n_days=200)
    evaluator = FactorExpressionEvaluator()

    # 重新实现项目的硬编码因子逻辑
    hard_coded = df.sort_values(['code', 'date']).copy()
    hard_coded['ret_1d'] = hard_coded.groupby('code')['close'].pct_change()
    hard_coded['ma20'] = hard_coded.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    hard_coded['volatility_20d'] = hard_coded.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )
    hard_coded['reversal_5d'] = -hard_coded.groupby('code')['close'].pct_change(5)
    hard_coded['hl_spread'] = hard_coded['high'] - hard_coded['low']
    hard_coded['hl_spread_ma5'] = hard_coded.groupby('code')['hl_spread'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )

    comparisons = [
        ("ret_1d", "$close / Ref($close, 1) - 1"),
        ("ma20", "Mean($close, 20)"),
        ("reversal_5d", "-($close / Ref($close, 5) - 1)"),
        ("hl_spread", "$high - $low"),
        ("hl_spread_ma5", "Mean($high - $low, 5)"),
    ]

    all_passed = True
    for hard_name, expr in comparisons:
        hard_val = hard_coded[hard_name].values
        try:
            expr_val = evaluator.evaluate(expr, df, group_col='code').values
            # 比较非 NaN 部分
            mask = ~(np.isnan(hard_val) | np.isnan(expr_val))
            if mask.sum() == 0:
                print(f"  SKIP: {hard_name}: 无有效比较数据")
                continue
            diff = np.abs(hard_val[mask] - expr_val[mask])
            max_diff = diff.max()
            mean_diff = diff.mean()
            status = "PASS" if max_diff < 1e-6 else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  {status}: {hard_name} vs '{expr}' -- max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
        except Exception as e:
            all_passed = False
            print(f"  FAIL: {hard_name} vs '{expr}' -- 异常: {e}")

    return all_passed


def test_expression_cache_performance():
    """测试3: 表达式缓存性能"""
    print("\n" + "=" * 60)
    print("测试3: 表达式缓存与重复求值性能")
    print("=" * 60)

    df = make_test_data(n_codes=10, n_days=500)

    evaluator = FactorExpressionEvaluator()
    expressions = [
        "Mean($close, 20)",
        "Std($close, 20)",
        "$high - $low",
        "Mean($high - $low, 5)",
        "Corr($close, $volume, 20)",
        "$close / Ref($close, 1) - 1",
        "Rank(Mean($close, 20))",
    ]

    # 首次求值 (无缓存)
    start = time.perf_counter()
    for expr in expressions:
        evaluator._cache.clear()
        evaluator.evaluate(expr, df)
    no_cache_time = time.perf_counter() - start

    # 有缓存重复求值
    evaluator._cache.clear()
    for expr in expressions:
        evaluator.evaluate(expr, df)  # 首次（建立缓存）
    start = time.perf_counter()
    for expr in expressions:
        evaluator.evaluate(expr, df)  # 二次（命中缓存）
    cache_time = time.perf_counter() - start

    speedup = no_cache_time / cache_time if cache_time > 0 else float('inf')
    print(f"  无缓存耗时: {no_cache_time:.4f}s")
    print(f"  有缓存耗时: {cache_time:.4f}s")
    print(f"  加速比: {speedup:.1f}x")
    print(f"  结论: {'PASS - 缓存有效' if speedup > 1.5 else 'FAIL - 缓存无显著效果'}")

    return speedup > 1.5


def test_nested_expression():
    """测试4: 嵌套表达式处理"""
    print("\n" + "=" * 60)
    print("测试4: 嵌套表达式处理能力")
    print("=" * 60)

    df = make_test_data(n_codes=3, n_days=100)
    evaluator = FactorExpressionEvaluator()

    # 复杂嵌套表达式
    nested_exprs = [
        "Mean($close, 20) / Std($close, 20)",           # 收益风险比
        "($close - Mean($close, 20)) / Std($close, 20)", # 标准化偏离
        "Rank(Mean($close, 20) / Mean($close, 60))",     # 排名动量比
        "If($close - Ref($close, 5), $close, Ref($close, 1))", # 条件选择
    ]

    all_passed = True
    for expr in nested_exprs:
        try:
            result = evaluator.evaluate(expr, df)
            non_null = result.notna().sum()
            print(f"  PASS: '{expr}' -> 结果行数={len(result)}, 非空={non_null}")
        except Exception as e:
            all_passed = False
            print(f"  FAIL: '{expr}' -> 异常: {e}")

    return all_passed


def test_expression_vs_hardcoded_performance():
    """测试5: 表达式引擎 vs 硬编码计算性能对比"""
    print("\n" + "=" * 60)
    print("测试5: 表达式引擎 vs 硬编码性能对比")
    print("=" * 60)

    sizes = [(10, 252), (50, 252), (100, 252), (10, 1008)]

    for n_codes, n_days in sizes:
        df = make_test_data(n_codes=n_codes, n_days=n_days)

        # 硬编码方式
        start = time.perf_counter()
        hc = df.sort_values(['code', 'date']).copy()
        hc['ret_1d'] = hc.groupby('code')['close'].pct_change()
        hc['ma20'] = hc.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
        hc['vol_20'] = hc.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=10).std())
        hc['hl'] = hc['high'] - hc['low']
        hc_time = time.perf_counter() - start

        # 表达式引擎方式
        evaluator = FactorExpressionEvaluator()
        evaluator._cache.clear()
        start = time.perf_counter()
        evaluator.evaluate("$close / Ref($close, 1) - 1", df)
        evaluator.evaluate("Mean($close, 20)", df)
        evaluator.evaluate("Std($close, 20)", df)
        evaluator.evaluate("$high - $low", df)
        expr_time = time.perf_counter() - start

        print(f"  {n_codes}只 x {n_days}天: 硬编码={hc_time:.4f}s, 表达式={expr_time:.4f}s, "
              f"比率={expr_time/hc_time:.1f}x")

        # 释放缓存
        evaluator._cache.clear()

    return True


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("因子表达式引擎验证测试")
    print("借鉴来源: microsoft/qlib Expression Engine")
    print("=" * 60)

    results = {}
    results['解析器正确性'] = test_parser_correctness()
    results['求值器一致性'] = test_evaluator_correctness()
    results['缓存性能'] = test_expression_cache_performance()
    results['嵌套表达式'] = test_nested_expression()
    results['性能对比'] = test_expression_vs_hardcoded_performance()

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")
    print(f"\n总体结果: {'全部通过' if all_pass else '存在失败项'}")

    sys.exit(0 if all_pass else 1)