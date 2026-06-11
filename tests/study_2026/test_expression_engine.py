"""
验证文件: 表达式驱动的因子计算引擎

借鉴来源:
  - Microsoft Qlib (github.com/microsoft/qlib) — 基于表达式的因子计算引擎，
    使用 $close, Ref($close, 5), Mean($close, 20) 等 DSL 定义因子。
    因子以函数形式声明而非数据，LLM agent 可直接生成因子表达式。
  - factor-expr (PyPI) — 基于 S-Expression 的超高性能因子表达式计算库，
    48 个因子在 24.5M 行数据集上仅需 150 秒。
  - RD-Agent (Microsoft) — 自动化因子挖掘，LLM 生成表达式 → Qlib 自动回测。

优化方向:
  当前 jingni-trader 的因子全部硬编码为 Python 函数，定义新因子需要编写代码。
  借鉴 Qlib 的表达式引擎和 Factor Engine 的装饰器模式，用字符串表达式替代硬编码，
  使因子定义更简洁、更易序列化、更易与 LLM agent 集成。

验证目标:
  1. 表达式解析正确性：字符串表达式计算结果与硬编码等价
  2. 表达能力：支持 ref, mean, std, rank, delay 等常见因子操作
  3. 性能对比：表达式引擎 vs 硬编码
  4. LLM 集成可行性：表达式可被 LLM 直接生成

创建日期: 2026-06-11
分支: feature/quant-stream-inspired (建议)
"""

import unittest
import timeit
import re
import sys
import os
from typing import Dict, Any, Callable, Union, List
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── 表达式引擎核心 ────────────────────────────────────────────

@dataclass
class Token:
    """表达式 Token"""
    type: str  # 'LITERAL', 'FUNC', 'COLREF', 'OP', 'NUMBER'
    value: str


class ExpressionParser:
    """因子表达式解析器"""

    # 支持的函数表
    FUNCTIONS = {
        'Ref': {'args': 2, 'desc': '前值引用 Ref($close, N)'},
        'Mean': {'args': 2, 'desc': 'N期均值 Mean($close, N)'},
        'Std': {'args': 2, 'desc': 'N期标准差 Std($close, N)'},
        'Rank': {'args': 1, 'desc': '截面排名 Rank($volume)'},
        'Delay': {'args': 2, 'desc': '延迟N期 Delay($close, N)'},
        'Log': {'args': 1, 'desc': '对数 Log($close)'},
        'Delta': {'args': 2, 'desc': '差分 Delta($close, N)'},
        'Sum': {'args': 2, 'desc': 'N期求和 Sum($close, N)'},
        'TSRank': {'args': 2, 'desc': '时序排名 TSRank($close, N)'},
    }

    def parse(self, expr: str) -> Dict[str, Any]:
        """解析表达式字符串为语法树"""
        expr = expr.strip()

        # 匹配函数调用: FuncName($col, N) 或 FuncName($col)
        func_match = re.match(r'(\w+)\((.+)\)$', expr)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)

            # 智能分割参数（考虑嵌套括号）
            args = self._split_args(args_str)
            return {
                'type': 'call',
                'function': func_name,
                'args': args,
            }

        # 匹配列引用: $close
        col_match = re.match(r'\$(\w+)', expr)
        if col_match:
            return {
                'type': 'column',
                'name': col_match.group(1),
            }

        # 匹配数字
        num_match = re.match(r'(-?[\d.]+)', expr)
        if num_match:
            return {
                'type': 'literal',
                'value': float(num_match.group(1)),
            }

        raise ValueError(f"无法解析表达式: {expr}")

    def _split_args(self, args_str: str) -> List[str]:
        """智能分割函数参数，处理嵌套括号"""
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


class ExpressionEvaluator:
    """因子表达式求值器"""

    def __init__(self, parser: ExpressionParser = None):
        self.parser = parser or ExpressionParser()

    def evaluate(self, expr_str: str, data: pd.DataFrame) -> pd.Series:
        """
        在给定的 DataFrame 上求值因子表达式

        参数:
            expr_str: 如 "Ref($close, 5)", "Mean($close, 20)", "(($high - $low) / $close)"
            data: 包含 code, date 列和行情列的面板数据
        """
        tree = self.parser.parse(expr_str)

        if tree['type'] == 'column':
            return data[tree['name']]

        elif tree['type'] == 'literal':
            return pd.Series(tree['value'], index=data.index)

        elif tree['type'] == 'call':
            func_name = tree['function']
            args = tree['args']

            # 递归求值参数
            evaluated_args = []
            for arg in args:
                if arg.startswith('$'):
                    # 简单列引用
                    col_name = arg[1:]
                    evaluated_args.append(data[col_name])
                elif re.match(r'^-?[\d.]+$', arg):
                    # 数字
                    evaluated_args.append(float(arg))
                else:
                    # 嵌套表达式
                    evaluated_args.append(self.evaluate(arg, data))

            return self._apply_function(func_name, *evaluated_args, data=data)

        else:
            raise ValueError(f"未知表达式类型: {tree['type']}")

    def _apply_function(self, name: str, *args, data: pd.DataFrame) -> pd.Series:
        """应用函数操作"""
        if name == 'Ref':
            col, n = args[0], int(args[1])
            return col.groupby(data['code']).shift(n)

        elif name == 'Mean':
            col, n = args[0], int(args[1])
            return col.groupby(data['code']).transform(
                lambda x: x.rolling(int(n), min_periods=max(1, int(n)//2)).mean()
            )

        elif name == 'Std':
            col, n = args[0], int(args[1])
            return col.groupby(data['code']).transform(
                lambda x: x.rolling(int(n), min_periods=max(1, int(n)//2)).std()
            )

        elif name == 'Rank':
            col = args[0]
            return col.groupby(data['date']).rank(pct=True)

        elif name == 'Delay':
            col, n = args[0], int(args[1])
            return col.groupby(data['code']).shift(n)

        elif name == 'Log':
            col = args[0]
            return np.log(col.replace(0, np.nan))

        elif name == 'Delta':
            col, n = args[0], int(args[1])
            return col.groupby(data['code']).diff(n)

        elif name == 'Sum':
            col, n = args[0], int(args[1])
            return col.groupby(data['code']).transform(
                lambda x: x.rolling(int(n), min_periods=max(1, int(n)//2)).sum()
            )

        elif name == 'TSRank':
            col, n = args[0], int(args[1])
            return col.groupby(data['code']).transform(
                lambda x: x.rolling(int(n), min_periods=max(1, int(n)//2)).rank(pct=True)
            )

        else:
            raise ValueError(f"不支持的函数: {name}")


class FactorExpressionEngine:
    """因子表达式引擎（完整封装）"""

    def __init__(self):
        self.evaluator = ExpressionEvaluator()
        self._factor_definitions: Dict[str, str] = {}
        self._register_builtin_factors()

    def _register_builtin_factors(self):
        """注册内置因子表达式"""
        factors = {
            'ret_1d': 'Delta(Log($close), 1)',     # (log)日收益率
            'ret_5d': 'Ref($close, 5)',            # 5日前价格（用于比较）
            'volatility_20d': 'Std(Delta(Log($close), 1), 20)',  # 20日波动率
            'volume_mean_20d': 'Mean($volume, 20)',  # 20日均量
            'turnover_20d': 'Mean($turnover_rate, 20)',  # 20日均换手率
        }
        self._factor_definitions.update(factors)

    def register_factor(self, name: str, expression: str):
        """注册自定义因子表达式"""
        self._factor_definitions[name] = expression

    def compute(self, data: pd.DataFrame, factor_names: List[str] = None) -> pd.DataFrame:
        """批量计算因子"""
        if factor_names is None:
            factor_names = list(self._factor_definitions.keys())

        df = data.sort_values(['code', 'date']).copy()
        result = df[['code', 'date']].copy()

        for name in factor_names:
            expr = self._factor_definitions.get(name)
            if expr is None:
                raise ValueError(f"未定义的因子: {name}")
            result[name] = self.evaluator.evaluate(expr, df)

        return result

    def list_factors(self) -> Dict[str, str]:
        return self._factor_definitions.copy()

    @classmethod
    def from_config(cls, factor_config: Dict[str, str]) -> 'FactorExpressionEngine':
        """从配置字典创建引擎（支持 LLM 生成的因子定义 JSON）"""
        engine = cls()
        for name, expr in factor_config.items():
            engine.register_factor(name, expr)
        return engine


# ── 对照实现 ──────────────────────────────────────────────────

def hardcoded_compute(data: pd.DataFrame) -> pd.DataFrame:
    """硬编码计算（对照）"""
    df = data.sort_values(['code', 'date']).copy()
    result = df[['code', 'date']].copy()

    result['ret_1d'] = np.log(df.groupby('code')['close'].pct_change() + 1)
    result['ret_5d'] = df.groupby('code')['close'].shift(5)
    log_ret = np.log(df.groupby('code')['close'].pct_change() + 1)
    result['volatility_20d'] = log_ret.groupby(df['code']).transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    result['volume_mean_20d'] = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    result['turnover_20d'] = df.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )

    return result


# ── 测试数据 ──────────────────────────────────────────────────

def generate_test_data(n_stocks: int = 10, n_days: int = 252) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for code in [f"SH600{i:03d}" for i in range(n_stocks)]:
        price = np.random.uniform(10, 50)
        for d in range(n_days):
            trend = np.random.normal(0.0002, 0.02)
            price = price * (1 + trend)
            if price < 3:
                price = 3
            rows.append({
                'code': code,
                'date': pd.Timestamp('2025-01-02') + pd.Timedelta(days=d),
                'open': price * (1 + np.random.normal(0, 0.005)),
                'close': price,
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'volume': np.random.uniform(1e5, 1e7),
                'amount': np.random.uniform(5e5, 5e8),
                'turnover_rate': np.random.uniform(0.005, 0.05),
            })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ── 测试类 ────────────────────────────────────────────────────

class TestExpressionEngine(unittest.TestCase):
    """测试表达式驱动的因子计算引擎"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_test_data(n_stocks=10, n_days=252)

    def test_01_parser_parses_simple_column(self):
        """解析器：列引用"""
        parser = ExpressionParser()
        tree = parser.parse('$close')
        self.assertEqual(tree['type'], 'column')
        self.assertEqual(tree['name'], 'close')

    def test_02_parser_parses_function_call(self):
        """解析器：函数调用"""
        parser = ExpressionParser()
        tree = parser.parse('Ref($close, 5)')
        self.assertEqual(tree['type'], 'call')
        self.assertEqual(tree['function'], 'Ref')
        self.assertEqual(tree['args'], ['$close', '5'])

    def test_03_parser_parses_rank_function(self):
        """解析器：单参数函数"""
        parser = ExpressionParser()
        tree = parser.parse('Rank($volume)')
        self.assertEqual(tree['function'], 'Rank')
        self.assertEqual(tree['args'], ['$volume'])

    def test_04_evaluator_ref(self):
        """求值器：Ref 操作"""
        evaluator = ExpressionEvaluator()
        result = evaluator.evaluate('Ref($close, 5)', self.data)
        # Ref($close, 5) 应等于 close.shift(5)
        expected = self.data.groupby('code')['close'].shift(5)
        pd.testing.assert_series_equal(
            result.fillna(-99999).reset_index(drop=True),
            expected.fillna(-99999).reset_index(drop=True),
            check_names=False,
        )
        print(f"  [PASS] Ref($close, 5) 正确")

    def test_05_evaluator_mean(self):
        """求值器：Mean 操作"""
        evaluator = ExpressionEvaluator()
        result = evaluator.evaluate('Mean($close, 20)', self.data)
        expected = self.data.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        pd.testing.assert_series_equal(
            result.fillna(-99999).reset_index(drop=True),
            expected.fillna(-99999).reset_index(drop=True),
            check_names=False,
        )
        print(f"  [PASS] Mean($close, 20) 正确")

    def test_06_evaluator_std(self):
        """求值器：Std 操作"""
        evaluator = ExpressionEvaluator()
        result = evaluator.evaluate('Std($close, 20)', self.data)
        expected = self.data.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        pd.testing.assert_series_equal(
            result.fillna(-99999).reset_index(drop=True),
            expected.fillna(-99999).reset_index(drop=True),
            check_names=False,
        )
        print(f"  [PASS] Std($close, 20) 正确")

    def test_07_evaluator_rank(self):
        """求值器：Rank 操作（截面排名）"""
        evaluator = ExpressionEvaluator()
        result = evaluator.evaluate('Rank($volume)', self.data)
        # 截面排名应为各日期内 volume 的百分比排名
        self.assertGreater(result.max(), 0.9)
        self.assertLessEqual(result.min(), 0.1)
        print(f"  [PASS] Rank($volume) 正确（截面百分比排名）")

    def test_08_full_engine_correctness(self):
        """完整引擎：表达式 vs 硬编码"""
        engine = FactorExpressionEngine()
        result_expr = engine.compute(self.data)
        result_hard = hardcoded_compute(self.data)

        factors = ['ret_1d', 'ret_5d', 'volatility_20d', 'volume_mean_20d', 'turnover_20d']

        for name in factors:
            a = result_expr[name].fillna(-99999).values
            b = result_hard[name].fillna(-99999).values
            np.testing.assert_array_almost_equal(
                a, b, decimal=5,
                err_msg=f"因子 {name} 计算结果不一致"
            )
        print(f"  [PASS] 所有 {len(factors)} 个因子表达式计算结果与硬编码一致")

    def test_09_custom_factor_registration(self):
        """自定义因子注册"""
        engine = FactorExpressionEngine()

        # 注册自定义因子：20日收益 = Ref($close, -20) / $close - 1
        engine.register_factor('custom_return_20d',
                               'Delta(Log($close), 20)')

        result = engine.compute(self.data, ['custom_return_20d'])
        self.assertIn('custom_return_20d', result.columns)
        # 应有足够的非 NaN 值
        valid = result['custom_return_20d'].dropna()
        self.assertGreater(len(valid), 100)
        print(f"  [PASS] 自定义因子注册成功，{len(valid)} 个有效值")

    def test_10_llm_friendly_config(self):
        """LLM 集成可行性：从字典/JSON 配置创建引擎"""
        llm_generated_config = {
            "alpha_001": "Rank(Delta(Log($close), 1))",
            "alpha_002": "Std($volume, 20)",
            "alpha_003": "Delta($close, 5)",
        }

        engine = FactorExpressionEngine.from_config(llm_generated_config)
        result = engine.compute(self.data, list(llm_generated_config.keys()))

        for name in llm_generated_config:
            self.assertIn(name, result.columns)
            valid = result[name].dropna()
            self.assertGreater(len(valid), 0, f"因子 {name} 无有效值")

        print(f"  [PASS] LLM 生成配置可正常创建引擎，"
              f"注册 {len(llm_generated_config)} 个因子")

    def test_11_performance_comparison(self):
        """性能对比：表达式引擎 vs 硬编码"""
        engine = FactorExpressionEngine()
        factors = ['ret_1d', 'ret_5d', 'volatility_20d', 'volume_mean_20d', 'turnover_20d']

        n_runs = 10

        t_expr = timeit.timeit(
            lambda: engine.compute(self.data, factors),
            number=n_runs
        )
        t_hard = timeit.timeit(
            lambda: hardcoded_compute(self.data),
            number=n_runs
        )

        avg_expr = t_expr / n_runs * 1000
        avg_hard = t_hard / n_runs * 1000
        ratio = avg_expr / avg_hard

        print(f"\n  性能对比 ({n_runs} 次运行, {len(self.data)} 行):")
        print(f"    表达式引擎: {avg_expr:.2f} ms/次")
        print(f"    硬编码:     {avg_hard:.2f} ms/次")
        print(f"    比率:       {ratio:.1f}x")

        # 表达式引擎允许有一定性能开销（<5x），因为每次需要解析
        self.assertLess(ratio, 5.0, f"表达式引擎性能过差 ({ratio:.1f}x)")
        print(f"  [PASS] 表达式引擎性能在可接受范围 ({ratio:.1f}x)")

    def test_12_error_handling(self):
        """错误处理：无效表达式"""
        engine = FactorExpressionEngine()

        with self.assertRaises(ValueError):
            engine.compute(self.data, ['nonexistent_factor'])

        evaluator = ExpressionEvaluator()
        with self.assertRaises(ValueError):
            evaluator.evaluate('InvalidFunc($close, 5)', self.data)

        print(f"  [PASS] 无效表达式和未定义因子的错误处理正常")


if __name__ == '__main__':
    print("=" * 60)
    print("验证：表达式驱动的因子计算引擎")
    print("借鉴来源：Microsoft Qlib / factor-expr / RD-Agent")
    print("=" * 60)
    unittest.main(verbosity=2)