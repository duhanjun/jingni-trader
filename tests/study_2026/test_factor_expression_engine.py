"""
=============================================================================
优化方向: 因子表达式引擎 (Factor Expression Engine)
借鉴来源: Microsoft Qlib (表达式引擎 + 公式化Alpha), quant-stream (DSL表达式语言)
日期: 2026-06-13
=============================================================================

核心思路:
  Qlib 提供了强大的表达式引擎，支持如 "Ref($close, 60) / $close" 这样的公式化因子定义。
  quant-stream 提供了类似的 DSL: "RANK(DELTA($close, 5))"。
  jingni-trader 当前在 factor-engine 中硬编码因子计算逻辑，每次新增因子都需修改源码。
  引入表达式引擎后，用户可通过字符串表达式定义因子，大幅提升因子库的可扩展性和易用性。

验证目标:
  1. 实现一个轻量级因子表达式解析器，支持变量引用($close, $volume, etc.)
  2. 支持常用操作: RANK, DELTA, DELAY, TS_MEAN, TS_STD, ZSCORE, 算术运算
  3. 验证解析结果与硬编码计算结果一致
  4. 对比新老方式在新增因子时的开发效率
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional
import re
from datetime import datetime, timedelta


# =============================================================================
# 因子表达式引擎核心实现
# =============================================================================

class FactorExpressionEngine:
    """
    轻量级因子表达式引擎

    支持操作:
      - 变量引用: $close, $open, $high, $low, $volume, $amount, $turnover
      - 时序操作: DELTA(x, d), DELAY(x, d), TS_MEAN(x, d), TS_STD(x, d),
                  TS_MAX(x, d), TS_MIN(x, d), TS_CORR(x, y, d)
      - 截面操作: RANK(x), ZSCORE(x), SCALE(x)
      - 算术运算: +, -, *, /, (, )
      - 数学函数: ABS(x), LOG(x), SIGN(x), POW(x, n)
    """

    # 变量映射: 表达式中的变量名 -> DataFrame中的列名
    VARIABLE_MAP = {
        "$close": "close",
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$volume": "volume",
        "$amount": "amount",
        "$turnover": "turnover_rate",
        "$vwap": "vwap",
    }

    def __init__(self, data: pd.DataFrame):
        """
        参数:
            data: 包含OHLCV等字段的DataFrame，需有 code, date 列
        """
        self.data = data.copy()
        self.data = self.data.sort_values(['code', 'date']).reset_index(drop=True)

    def evaluate(self, expression: str, name: str = None) -> pd.Series:
        """
        解析并计算因子表达式

        参数:
            expression: 因子表达式字符串
            name: 因子名称

        返回:
            计算结果Series，与输入data对齐
        """
        result = self._parse_expression(expression)
        if name:
            result = pd.Series(result, name=name)
        return result

    def _parse_expression(self, expr: str) -> pd.Series:
        """递归下降解析表达式"""
        expr = expr.strip()

        # 处理括号
        if expr.startswith('(') and expr.endswith(')'):
            # 检查括号是否匹配
            depth = 0
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    break
            if depth == 0 and i == len(expr) - 1:
                return self._parse_expression(expr[1:-1])

        # 尝试函数调用: 手动匹配括号，避免贪婪匹配
        word_match = re.match(r'^(\w+)\(', expr)
        if word_match:
            func_name = word_match.group(1).upper()
            paren_start = word_match.end() - 1  # 左括号位置
            # 找到匹配的右括号
            depth = 1
            paren_end = -1
            for i in range(paren_start + 1, len(expr)):
                if expr[i] == '(':
                    depth += 1
                elif expr[i] == ')':
                    depth -= 1
                    if depth == 0:
                        paren_end = i
                        break
            if paren_end == len(expr) - 1:
                args_str = expr[paren_start + 1:paren_end]
                args = self._split_args(args_str)
                return self._call_function(func_name, args)

        # 变量引用
        if expr in self.VARIABLE_MAP:
            col_name = self.VARIABLE_MAP[expr]
            if col_name in self.data.columns:
                return self.data[col_name].values
            else:
                raise ValueError(f"变量 {expr} 对应的列 {col_name} 不存在于数据中")

        # 数字常量
        try:
            val = float(expr)
            return np.full(len(self.data), val)
        except ValueError:
            pass

        # 算术运算 (按优先级: +, - 最低)
        # 从右向左扫描，找到最外层运算符
        for op_pair in [('+', self._add), ('-', self._sub)]:
            op, _ = op_pair
            idx = self._find_outer_operator(expr, op)
            if idx > 0:
                left = self._parse_expression(expr[:idx])
                right = self._parse_expression(expr[idx + 1:])
                return self._apply_binary_op(left, right, op)

        for op_pair in [('*', self._mul), ('/', self._div)]:
            op, _ = op_pair
            idx = self._find_outer_operator(expr, op)
            if idx > 0:
                left = self._parse_expression(expr[:idx])
                right = self._parse_expression(expr[idx + 1:])
                return self._apply_binary_op(left, right, op)

        raise ValueError(f"无法解析表达式: {expr}")

    def _find_outer_operator(self, expr: str, op: str) -> int:
        """找到最外层未被括号包裹的运算符位置"""
        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            if expr[i] == ')':
                depth += 1
            elif expr[i] == '(':
                depth -= 1
            elif depth == 0 and expr[i] == op:
                # 对于 - 和 +，排除一元运算符
                # 一元运算符: 位于开头，或前面紧邻运算符/左括号
                # 也检查前面是空格的情况（如 " * -1"）
                if op in ('-', '+'):
                    if i == 0:
                        continue  # 一元运算符
                    # 向前跳过空格，找前一个非空字符
                    j = i - 1
                    while j >= 0 and expr[j] == ' ':
                        j -= 1
                    if j >= 0 and expr[j] in '+-*/(':
                        continue  # 一元运算符
                return i
        return -1

    def _split_args(self, args_str: str) -> List[str]:
        """按逗号分割函数参数，考虑括号嵌套"""
        args = []
        depth = 0
        current = ""
        for c in args_str:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if c == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += c
        if current.strip():
            args.append(current.strip())
        return args

    def _apply_binary_op(self, left, right, op: str):
        """应用二元运算"""
        l = left if isinstance(left, np.ndarray) else np.array(left)
        r = right if isinstance(right, np.ndarray) else np.array(right)
        if op == '+':
            return l + r
        elif op == '-':
            return l - r
        elif op == '*':
            return l * r
        elif op == '/':
            with np.errstate(divide='ignore', invalid='ignore'):
                result = l / r
                result[np.isinf(result)] = np.nan
            return result
        raise ValueError(f"不支持的运算符: {op}")

    # ---- 内置函数 ----

    def _call_function(self, name: str, args: List[str]) -> np.ndarray:
        """调用内置函数"""
        parsed_args = [self._parse_expression(a) for a in args]

        if name == 'DELTA':
            return self._func_delta(parsed_args[0], int(args[1]) if len(args) > 1 else 1)
        elif name == 'DELAY':
            return self._func_delay(parsed_args[0], int(args[1]) if len(args) > 1 else 1)
        elif name == 'TS_MEAN':
            return self._func_ts_mean(parsed_args[0], int(args[1]) if len(args) > 1 else 20)
        elif name == 'TS_STD':
            return self._func_ts_std(parsed_args[0], int(args[1]) if len(args) > 1 else 20)
        elif name == 'TS_MAX':
            return self._func_ts_max(parsed_args[0], int(args[1]) if len(args) > 1 else 20)
        elif name == 'TS_MIN':
            return self._func_ts_min(parsed_args[0], int(args[1]) if len(args) > 1 else 20)
        elif name == 'TS_CORR':
            return self._func_ts_corr(parsed_args[0], parsed_args[1], int(args[2]) if len(args) > 2 else 20)
        elif name == 'RANK':
            return self._func_rank(parsed_args[0])
        elif name == 'ZSCORE':
            return self._func_zscore(parsed_args[0])
        elif name == 'SCALE':
            return self._func_scale(parsed_args[0])
        elif name == 'ABS':
            return np.abs(parsed_args[0])
        elif name == 'LOG':
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.log(np.maximum(parsed_args[0], 1e-10))
        elif name == 'SIGN':
            return np.sign(parsed_args[0])
        elif name == 'POW':
            return np.power(parsed_args[0], int(args[1]) if len(args) > 1 else 2)
        elif name == 'SQRT':
            return np.sqrt(np.maximum(parsed_args[0], 0))
        else:
            raise ValueError(f"未知函数: {name}")

    def _func_delta(self, x: np.ndarray, d: int) -> np.ndarray:
        return self._group_apply(x, lambda s: s.diff(d))

    def _func_delay(self, x: np.ndarray, d: int) -> np.ndarray:
        return self._group_apply(x, lambda s: s.shift(d))

    def _func_ts_mean(self, x: np.ndarray, d: int) -> np.ndarray:
        return self._group_apply(x, lambda s: s.rolling(d, min_periods=max(3, d // 3)).mean())

    def _func_ts_std(self, x: np.ndarray, d: int) -> np.ndarray:
        return self._group_apply(x, lambda s: s.rolling(d, min_periods=max(3, d // 3)).std())

    def _func_ts_max(self, x: np.ndarray, d: int) -> np.ndarray:
        return self._group_apply(x, lambda s: s.rolling(d, min_periods=max(3, d // 3)).max())

    def _func_ts_min(self, x: np.ndarray, d: int) -> np.ndarray:
        return self._group_apply(x, lambda s: s.rolling(d, min_periods=max(3, d // 3)).min())

    def _func_ts_corr(self, x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
        result = np.full(len(x), np.nan)
        codes = self.data['code'].values
        dates = self.data['date'].values
        for code in np.unique(codes):
            mask = codes == code
            s_x = pd.Series(x[mask])
            s_y = pd.Series(y[mask])
            corr = s_x.rolling(d, min_periods=max(3, d // 3)).corr(s_y)
            result[mask] = corr.values
        return result

    def _func_rank(self, x: np.ndarray) -> np.ndarray:
        result = np.full(len(x), np.nan)
        dates = self.data['date'].values
        for dt in np.unique(dates):
            mask = dates == dt
            valid = mask & ~np.isnan(x)
            result[valid] = pd.Series(x[valid]).rank(pct=True).values
        return result

    def _func_zscore(self, x: np.ndarray) -> np.ndarray:
        result = np.full(len(x), np.nan)
        dates = self.data['date'].values
        for dt in np.unique(dates):
            mask = dates == dt
            valid = mask & ~np.isnan(x)
            vals = x[valid]
            if len(vals) > 1:
                mean = np.mean(vals)
                std = np.std(vals)
                if std > 0:
                    result[valid] = (vals - mean) / std
        return result

    def _func_scale(self, x: np.ndarray) -> np.ndarray:
        result = np.full(len(x), np.nan)
        dates = self.data['date'].values
        for dt in np.unique(dates):
            mask = dates == dt
            valid = mask & ~np.isnan(x)
            vals = x[valid]
            if len(vals) > 1:
                min_v = np.min(vals)
                max_v = np.max(vals)
                if max_v > min_v:
                    result[valid] = (vals - min_v) / (max_v - min_v)
        return result

    def _group_apply(self, x: np.ndarray, func: Callable) -> np.ndarray:
        """按code分组应用函数"""
        result = np.full(len(x), np.nan)
        codes = self.data['code'].values
        for code in np.unique(codes):
            mask = codes == code
            s = pd.Series(x[mask])
            result[mask] = func(s).values
        return result

    @staticmethod
    def _add(a, b):
        return a + b

    @staticmethod
    def _sub(a, b):
        return a - b

    @staticmethod
    def _mul(a, b):
        return a * b

    @staticmethod
    def _div(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = a / b
            r[np.isinf(r)] = np.nan
        return r


# =============================================================================
# 测试用例
# =============================================================================

class TestFactorExpressionEngine(unittest.TestCase):
    """因子表达式引擎测试"""

    @classmethod
    def setUpClass(cls):
        """生成测试数据: 模拟3只股票60个交易日的行情"""
        np.random.seed(42)
        n_stocks = 3
        n_days = 60
        codes = ['000001.SZ', '600000.SH', '000002.SZ']
        dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

        rows = []
        for code in codes:
            price = 10 + np.cumsum(np.random.randn(n_days) * 0.3)
            for i, dt in enumerate(dates):
                rows.append({
                    'code': code,
                    'date': dt,
                    'close': price[i],
                    'open': price[i] * (1 + np.random.randn() * 0.01),
                    'high': price[i] * (1 + abs(np.random.randn() * 0.015)),
                    'low': price[i] * (1 - abs(np.random.randn() * 0.015)),
                    'volume': np.random.randint(10000, 100000),
                    'amount': np.random.randint(100000, 1000000),
                    'turnover_rate': np.random.rand() * 5,
                })

        cls.test_data = pd.DataFrame(rows)
        # 排序以匹配引擎内部的数据顺序
        cls.test_data = cls.test_data.sort_values(['code', 'date']).reset_index(drop=True)
        cls.engine = FactorExpressionEngine(cls.test_data)

    def test_variable_reference(self):
        """测试变量引用: $close, $volume 等"""
        result = self.engine.evaluate("$close")
        np.testing.assert_array_almost_equal(result, self.test_data['close'].values)

    def test_simple_arithmetic(self):
        """测试简单算术运算"""
        result = self.engine.evaluate("$close + $open")
        expected = self.test_data['close'].values + self.test_data['open'].values
        np.testing.assert_array_almost_equal(result, expected)

    def test_nested_arithmetic(self):
        """测试嵌套算术运算"""
        result = self.engine.evaluate("($close - $open) / $close")
        expected = (self.test_data['close'].values - self.test_data['open'].values) / self.test_data['close'].values
        np.testing.assert_array_almost_equal(result, expected)

    def test_delta_operator(self):
        """测试 DELTA 操作符 (等同于 diff)"""
        result = self.engine.evaluate("DELTA($close, 1)")
        # 手动计算: 按code分组后的diff
        expected = np.full(len(self.test_data), np.nan)
        for code in self.test_data['code'].unique():
            mask = self.test_data['code'] == code
            expected[mask] = self.test_data.loc[mask, 'close'].diff().values
        np.testing.assert_array_almost_equal(
            result[~np.isnan(result)],
            expected[~np.isnan(expected)]
        )

    def test_delay_operator(self):
        """测试 DELAY 操作符 (等同于 shift)"""
        result = self.engine.evaluate("DELAY($close, 2)")
        expected = np.full(len(self.test_data), np.nan)
        for code in self.test_data['code'].unique():
            mask = self.test_data['code'] == code
            expected[mask] = self.test_data.loc[mask, 'close'].shift(2).values
        np.testing.assert_array_almost_equal(
            result[~np.isnan(result)],
            expected[~np.isnan(expected)]
        )

    def test_ts_mean_operator(self):
        """测试 TS_MEAN 操作符 (滚动均值)"""
        result = self.engine.evaluate("TS_MEAN($close, 5)")
        expected = np.full(len(self.test_data), np.nan)
        for code in self.test_data['code'].unique():
            mask = self.test_data['code'] == code
            expected[mask] = self.test_data.loc[mask, 'close'].rolling(5, min_periods=3).mean().values
        np.testing.assert_array_almost_equal(
            result[~np.isnan(result)],
            expected[~np.isnan(expected)]
        )

    def test_rank_operator(self):
        """测试 RANK 操作符 (截面排名)"""
        result = self.engine.evaluate("RANK($close)")
        # 验证每个日期的排名值在 0-1 之间
        self.assertTrue(np.all((result[~np.isnan(result)] >= 0) & (result[~np.isnan(result)] <= 1)))
        # 验证每个日期排名存在
        for dt in self.test_data['date'].unique():
            mask = self.test_data['date'] == dt
            vals = result[mask]
            self.assertTrue(len(vals[~np.isnan(vals)]) > 0)

    def test_zscore_operator(self):
        """测试 ZSCORE 操作符"""
        result = self.engine.evaluate("ZSCORE($close)")
        # 验证每个日期截面标准化后均值为0（近似）
        for dt in self.test_data['date'].unique():
            mask = self.test_data['date'] == dt
            vals = result[mask]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 1:
                self.assertAlmostEqual(np.mean(valid), 0, delta=0.1)

    def test_complex_reversal_factor(self):
        """测试复合因子: 20日反转因子 (Qlib: Ref($close, 20) / $close - 1)"""
        # 表达式: DELTA($close, 20) / DELAY($close, 20)
        # 等价于: (close_t - close_{t-20}) / close_{t-20}
        result = self.engine.evaluate("DELTA($close, 20) / DELAY($close, 20)")

        expected = np.full(len(self.test_data), np.nan)
        for code in self.test_data['code'].unique():
            mask = self.test_data['code'] == code
            close_s = self.test_data.loc[mask, 'close']
            expected[mask] = (close_s.diff(20) / close_s.shift(20)).values

        np.testing.assert_array_almost_equal(
            result[~np.isnan(result)],
            expected[~np.isnan(expected)]
        )

    def test_volume_price_factor(self):
        """测试量价复合因子: (close - open) / close * volume_ratio"""
        expr = "($close - $open) / $close * (DELTA($volume, 1) / DELAY($volume, 1))"
        result = self.engine.evaluate(expr)

        close = self.test_data['close'].values
        open_ = self.test_data['open'].values
        vol = self.test_data['volume'].values

        expected = (close - open_) / close
        for code in self.test_data['code'].unique():
            mask = self.test_data['code'] == code
            expected[mask] = expected[mask] * (pd.Series(vol[mask]).diff() / pd.Series(vol[mask]).shift(1)).values

        np.testing.assert_array_almost_equal(
            result[~np.isnan(result)],
            expected[~np.isnan(expected)],
            decimal=5
        )

    def test_expression_vs_hardcoded(self):
        """测试表达式引擎 vs 硬编码计算: 确保结果一致"""
        # 硬编码方式计算 20日动量因子
        close = self.test_data['close'].values
        codes = self.test_data['code'].values
        hardcoded = np.full(len(self.test_data), np.nan)
        for code in np.unique(codes):
            mask = codes == code
            hardcoded[mask] = pd.Series(close[mask]).pct_change(20).values

        # 表达式方式
        expr_result = self.engine.evaluate("DELTA($close, 20) / DELAY($close, 20)")

        # 比较
        common = ~np.isnan(hardcoded) & ~np.isnan(expr_result)
        np.testing.assert_array_almost_equal(hardcoded[common], expr_result[common], decimal=8)

    def test_factor_registry_extensibility(self):
        """测试因子注册表: 模拟用户通过表达式注册新因子，无需修改源码"""
        # 模拟因子注册表
        factor_registry = {
            "momentum_20d": "DELTA($close, 20) / DELAY($close, 20)",
            "reversal_5d": "DELTA($close, 5) / DELAY($close, 5) * -1",
            "volatility_20d": "TS_STD(DELTA($close, 1) / DELAY($close, 1), 20)",
            "volume_ratio": "($volume - TS_MEAN($volume, 20)) / TS_STD($volume, 20)",
            "amplitude": "($high - $low) / $open",
            "rsi_14": "TS_MEAN(DELTA($close, 1), 14) / TS_STD(DELTA($close, 1), 14)",
        }

        results = {}
        for name, expr in factor_registry.items():
            results[name] = self.engine.evaluate(expr, name)
            self.assertIsNotNone(results[name])
            self.assertEqual(len(results[name]), len(self.test_data))

        # 验证所有因子计算成功且无全NaN
        for name, result in results.items():
            valid_ratio = (~np.isnan(result)).mean()
            self.assertGreater(valid_ratio, 0.1, f"因子 {name} 有效值比例过低: {valid_ratio:.2%}")


class TestExpressionEnginePerformance(unittest.TestCase):
    """性能对比测试"""

    @classmethod
    def setUpClass(cls):
        """生成较大规模测试数据"""
        np.random.seed(42)
        n_stocks = 50
        n_days = 252  # 一年的交易日
        codes = [f'{i:06d}.SH' for i in range(600000, 600000 + n_stocks)]
        dates = pd.date_range('2023-01-01', periods=n_days, freq='B')

        rows = []
        for code in codes:
            price = 10 + np.cumsum(np.random.randn(n_days) * 0.3)
            for i, dt in enumerate(dates):
                rows.append({
                    'code': code,
                    'date': dt,
                    'close': price[i],
                    'open': price[i] * (1 + np.random.randn() * 0.01),
                    'high': price[i] * (1 + abs(np.random.randn() * 0.015)),
                    'low': price[i] * (1 - abs(np.random.randn() * 0.015)),
                    'volume': np.random.randint(10000, 100000),
                    'amount': np.random.randint(100000, 1000000),
                    'turnover_rate': np.random.rand() * 5,
                })

        cls.large_data = pd.DataFrame(rows)
        cls.large_data = cls.large_data.sort_values(['code', 'date']).reset_index(drop=True)
        cls.engine = FactorExpressionEngine(cls.large_data)

    def test_expression_engine_speed(self):
        """测试表达式引擎计算速度 (50只股票, 252天, 6个因子)"""
        import time

        factor_expressions = {
            "momentum_20d": "DELTA($close, 20) / DELAY($close, 20)",
            "reversal_5d": "DELTA($close, 5) / DELAY($close, 5) * -1",
            "volatility_20d": "TS_STD(DELTA($close, 1) / DELAY($close, 1), 20)",
            "volume_ratio": "($volume - TS_MEAN($volume, 20)) / TS_STD($volume, 20)",
            "amplitude": "($high - $low) / $open",
            "price_position": "($close - TS_MIN($low, 20)) / (TS_MAX($high, 20) - TS_MIN($low, 20))",
        }

        start = time.time()
        for name, expr in factor_expressions.items():
            self.engine.evaluate(expr, name)
        elapsed = time.time() - start

        print(f"\n  表达式引擎: 计算 {len(factor_expressions)} 个因子, "
              f"{len(self.large_data)} 行数据, 耗时 {elapsed:.3f}s")

        self.assertLess(elapsed, 5.0, "表达式引擎计算超时 (>5s)")


if __name__ == '__main__':
    unittest.main(verbosity=2)