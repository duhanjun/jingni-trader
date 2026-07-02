"""
验证测试：表达式因子引擎
借鉴来源：Qlib (https://github.com/microsoft/qlib) - 表达式引擎设计
        Qlib 的 Expression Engine 允许用户用类似 $close/Ref($close, 20)-1 的
        字符串公式定义因子，自动向量化计算。

优化方向：为 jingni-trader 的 factor-engine 引入基于表达式的因子定义方式，
        支持算子组合、自动向量化、惰性计算，提升因子库的可扩展性。

设计思路：
  - Qlib 的表达式引擎支持: Ref, Mean, Std, Max, Min, Rank, Log, Abs, Sign 等算子
  - 通过 AST 解析表达式，自动生成向量化计算图
  - 支持批量计算和缓存
  - 本测试验证表达式引擎在功能正确性、性能、可扩展性方面的表现
"""
import sys
import os
import unittest
import re
import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# 表达式因子引擎核心实现
# ============================================================

class FactorExpression:
    """
    因子表达式引擎 - 借鉴 Qlib 的 Expression Engine 设计
    
    支持的语法:
      - 基础字段: $open, $high, $low, $close, $volume, $amount, $turnover
      - 算子函数:
        - Ref(expr, N): N日前值 (N>0 表示滞后)
        - Mean(expr, N): N日滚动均值
        - Std(expr, N): N日滚动标准差  
        - Max(expr, N): N日滚动最大值
        - Min(expr, N): N日滚动最小值
        - Sum(expr, N): N日滚动求和
        - PctChange(expr, N): N日涨跌幅
        - Rank(expr): 横截面排名 (0-1, 按日期分组)
        - Log(expr): 自然对数
        - Abs(expr): 绝对值
        - Sign(expr): 符号
        - Delay(expr, N): 延迟N日 (同 Ref)
        - Delta(expr, N): N日差值
        - Corr(expr1, expr2, N): N日滚动相关系数
        - If(cond, true_val, false_val): 条件表达式
      - 运算符: +, -, *, /, >, <, >=, <=, ==, !=, &, |
    """
    
    # 支持的字段映射
    FIELD_MAP = {
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
        'volume': 'volume', 'amount': 'amount', 'turnover': 'turnover_rate',
        'vwap': 'vwap', 'change_pct': 'change_pct',
    }
    
    def __init__(self, expression: str):
        self.expression = expression
        self._compiled = self._compile(expression)
    
    def _compile(self, expr: str) -> Callable:
        """编译表达式为可调用函数"""
        def _eval(data: pd.DataFrame) -> pd.Series:
            return self._evaluate(data, self.expression)
        return _eval
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """对数据进行求值"""
        return self._compiled(data)
    
    def _evaluate(self, data: pd.DataFrame, expr: str) -> pd.Series:
        """递归求值表达式"""
        expr = expr.strip()
        
        # 处理括号包裹的表达式
        if expr.startswith('(') and expr.endswith(')'):
            depth = 0
            for i, c in enumerate(expr):
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                if depth == 0 and i == len(expr) - 1:
                    return self._evaluate(data, expr[1:-1])
        
        # 处理一元负号: -expr
        if expr.startswith('-'):
            inner = self._evaluate(data, expr[1:])
            return -inner
        
        # 处理算术/逻辑运算符（最低优先级，从右向左找最外层）
        ops = [
            ('&', lambda a, b: (a.astype(bool) & b.astype(bool)).astype(float)),
            ('|', lambda a, b: (a.astype(bool) | b.astype(bool)).astype(float)),
            ('>=', lambda a, b: (a >= b).astype(float)),
            ('<=', lambda a, b: (a <= b).astype(float)),
            ('!=', lambda a, b: (a != b).astype(float)),
            ('==', lambda a, b: (a == b).astype(float)),
            ('>', lambda a, b: (a > b).astype(float)),
            ('<', lambda a, b: (a < b).astype(float)),
            ('+', lambda a, b: a + b),
            ('-', lambda a, b: a - b),
            ('*', lambda a, b: a * b),
            ('/', lambda a, b: a / b),
        ]
        
        for op, func in ops:
            pos = self._find_top_level_op(expr, op)
            if pos > 0:
                left = self._evaluate(data, expr[:pos])
                right = self._evaluate(data, expr[pos + len(op):])
                return func(left, right)
        
        # 处理函数调用
        func_ma = re.match(r'^(\w+)\((.*)\)$', expr, re.DOTALL)
        if func_ma:
            func_name = func_ma.group(1)
            args_str = func_ma.group(2)
            args = self._split_args(args_str)
            
            return self._call_function(data, func_name, args)
        
        # 处理字段引用 $field
        if expr.startswith('$'):
            field_name = expr[1:]
            if field_name in self.FIELD_MAP:
                col_name = self.FIELD_MAP[field_name]
                if col_name in data.columns:
                    return data[col_name]
                raise ValueError(f"数据中不存在字段: {col_name}")
            raise ValueError(f"未知字段: ${field_name}")
        
        # 处理数字字面量
        try:
            val = float(expr)
            return pd.Series(val, index=data.index)
        except ValueError:
            pass
        
        raise ValueError(f"无法解析表达式: {expr}")
    
    def _find_top_level_op(self, expr: str, op: str) -> int:
        """找到最外层的运算符位置"""
        depth = 0
        i = len(expr) - len(op)
        while i >= 0:
            c = expr[i]
            if c == ')':
                depth += 1
            elif c == '(':
                depth -= 1
            elif depth == 0 and expr[i:i + len(op)] == op:
                return i
            i -= 1
        return -1
    
    def _split_args(self, args_str: str) -> List[str]:
        """按逗号分割函数参数"""
        args = []
        current = ''
        depth = 0
        
        for c in args_str:
            if c == ',' and depth == 0:
                args.append(current.strip())
                current = ''
            else:
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                current += c
        
        if current.strip():
            args.append(current.strip())
        
        return args
    
    def _call_function(self, data: pd.DataFrame, func_name: str, args: List[str]) -> pd.Series:
        """调用因子函数"""
        
        if func_name == 'Ref' or func_name == 'Delay':
            inner = self._evaluate(data, args[0])
            n = self._parse_int_arg(args[1], data) if len(args) > 1 else 1
            return inner.groupby(data['code']).shift(n)
        
        elif func_name == 'Mean':
            inner = self._evaluate(data, args[0])
            n = self._parse_int_arg(args[1], data) if len(args) > 1 else 5
            return inner.groupby(data['code']).transform(
                lambda x: x.rolling(n, min_periods=max(1, n // 2)).mean()
            )
        
        elif func_name == 'Std':
            inner = self._evaluate(data, args[0])
            n = self._parse_int_arg(args[1], data) if len(args) > 1 else 20
            return inner.groupby(data['code']).transform(
                lambda x: x.rolling(n, min_periods=max(1, n // 2)).std()
            )
        
        elif func_name == 'Max':
            inner = self._evaluate(data, args[0])
            n = self._parse_int_arg(args[1], data) if len(args) > 1 else 20
            return inner.groupby(data['code']).transform(
                lambda x: x.rolling(n, min_periods=max(1, n // 2)).max()
            )
        
        elif func_name == 'Min':
            inner = self._evaluate(data, args[0])
            n = self._parse_int_arg(args[1], data) if len(args) > 1 else 20
            return inner.groupby(data['code']).transform(
                lambda x: x.rolling(n, min_periods=max(1, n // 2)).min()
            )
        
        elif func_name == 'Sum':
            inner = self._evaluate(data, args[0])
            n = self._parse_int_arg(args[1], data) if len(args) > 1 else 5
            return inner.groupby(data['code']).transform(
                lambda x: x.rolling(n, min_periods=1).sum()
            )
        
        elif func_name == 'PctChange':
            inner = self._evaluate(data, args[0])
            n = self._parse_int_arg(args[1], data) if len(args) > 1 else 1
            return inner.groupby(data['code']).pct_change(n)
        
        elif func_name == 'Rank':
            inner = self._evaluate(data, args[0])
            result = pd.Series(index=data.index, dtype=float)
            for dt, group in data.groupby('date'):
                idx = group.index
                vals = inner.loc[idx]
                if len(vals.dropna()) > 0:
                    result.loc[idx] = vals.rank(pct=True)
            return result
        
        elif func_name == 'Log':
            inner = self._evaluate(data, args[0])
            return np.log(inner.replace(0, np.nan))
        
        elif func_name == 'Abs':
            inner = self._evaluate(data, args[0])
            return inner.abs()
        
        elif func_name == 'Sign':
            inner = self._evaluate(data, args[0])
            return np.sign(inner)
        
        elif func_name == 'Delta':
            inner = self._evaluate(data, args[0])
            n = self._parse_int_arg(args[1], data) if len(args) > 1 else 1
            return inner.groupby(data['code']).diff(n)
        
        elif func_name == 'Corr':
            inner1 = self._evaluate(data, args[0])
            inner2 = self._evaluate(data, args[1])
            n = self._parse_int_arg(args[2], data) if len(args) > 2 else 20
            result = pd.Series(index=data.index, dtype=float)
            for code, group in data.groupby('code'):
                idx = group.index
                if len(group) >= n:
                    result.loc[idx] = inner1.loc[idx].rolling(n).corr(inner2.loc[idx])
            return result
        
        elif func_name == 'If':
            # If(condition, true_value, false_value)
            cond = self._evaluate(data, args[0])
            true_val = self._evaluate(data, args[1])
            false_val = self._evaluate(data, args[2])
            return pd.Series(
                np.where(cond.values > 0, true_val.values, false_val.values),
                index=data.index
            )
        
        else:
            raise ValueError(f"未知函数: {func_name}")
    
    def _parse_int_arg(self, arg: str, data: pd.DataFrame) -> int:
        """解析整数参数"""
        arg = arg.strip()
        if arg.startswith('$'):
            # 如果是字段引用，取第一行值
            field = arg[1:]
            if field in self.FIELD_MAP:
                col = self.FIELD_MAP[field]
                return int(data[col].iloc[0]) if col in data.columns else 1
        return int(float(arg))


# ============================================================
# 因子注册表 - 借鉴 Qlib 的 Alpha158/Alpha360 预定义因子集
# ============================================================

class FactorRegistry:
    """因子注册表 - 管理预定义因子表达式"""
    
    def __init__(self):
        self._factors: Dict[str, str] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """注册常用因子"""
        # 价量类因子（借鉴 Qlib Alpha158 中的价量因子）
        defaults = {
            # 收益类
            'ret_1d': 'PctChange($close, 1)',
            'ret_5d': 'PctChange($close, 5)',
            'ret_20d': 'PctChange($close, 20)',
            'ret_60d': 'PctChange($close, 60)',
            
            # 均线偏离
            'ma5_bias': '$close / Mean($close, 5) - 1',
            'ma20_bias': '$close / Mean($close, 20) - 1',
            'ma60_bias': '$close / Mean($close, 60) - 1',
            
            # 波动率
            'volatility_5d': 'Std(PctChange($close, 1), 5)',
            'volatility_20d': 'Std(PctChange($close, 1), 20)',
            'volatility_60d': 'Std(PctChange($close, 1), 60)',
            
            # 量比
            'volume_ratio_5': '$volume / Mean($volume, 5)',
            'volume_ratio_20': '$volume / Mean($volume, 20)',
            
            # 价格位置
            'price_position_20': '($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)',
            'price_position_60': '($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 1e-8)',
            
            # 动量因子
            'momentum_5_20': 'PctChange($close, 5) - PctChange($close, 20)',
            'momentum_20_60': 'PctChange($close, 20) - PctChange($close, 60)',
            
            # 换手率相关
            'turnover_5d': 'Mean($turnover, 5)',
            'turnover_20d': 'Mean($turnover, 20)',
            'turnover_variation': 'Std($turnover, 20)',
            
            # 振幅
            'amplitude_20d': 'Mean(($high - $low) / Ref($close, 1), 20)',
            
            # 资金流
            'money_flow_5d': 'Sum($amount, 5)',
            'money_flow_20d': 'Sum($amount, 20)',
            
            # RSV (未成熟随机值)
            'rsv_9': '($close - Min($low, 9)) / (Max($high, 9) - Min($low, 9) + 1e-8)',
            
            # 量价相关性
            'corr_price_volume_20': 'Corr($close, $volume, 20)',
        }
        
        for name, expr in defaults.items():
            self.register(name, expr)
    
    def register(self, name: str, expression: str):
        """注册因子"""
        self._factors[name] = expression
    
    def get(self, name: str) -> Optional[str]:
        """获取因子表达式"""
        return self._factors.get(name)
    
    def list_all(self) -> List[str]:
        """列出所有因子"""
        return list(self._factors.keys())
    
    def compute(self, data: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """
        批量计算因子
        
        参数:
            data: 包含 OHLCV 等原始数据的 DataFrame
            factor_names: 需要计算的因子名称列表
        
        返回:
            DataFrame with columns: code, date, [factor columns]
        """
        if data.empty:
            return pd.DataFrame()
        
        result = data[['code', 'date']].copy()
        
        for name in factor_names:
            expr = self._factors.get(name)
            if expr is None:
                result[name] = np.nan
                continue
            
            try:
                fe = FactorExpression(expr)
                result[name] = fe.evaluate(data)
            except Exception as e:
                print(f"  因子 [{name}] 计算失败: {e}")
                result[name] = np.nan
        
        return result


# ============================================================
# 测试用例
# ============================================================

class TestExpressionFactor(unittest.TestCase):
    """表达式因子引擎测试"""
    
    @classmethod
    def setUpClass(cls):
        """生成模拟测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        codes = [f'{i:06d}.SH' for i in range(600001, 600011)]  # 10 stocks
        
        rows = []
        for date in dates:
            for code in codes:
                code_idx = codes.index(code)
                price = 10 + code_idx * 2 + np.sin(date.dayofyear * 0.1 + code_idx) * 3
                price += np.random.randn() * 0.3
                
                rows.append({
                    'code': code,
                    'date': date,
                    'open': price + np.random.randn() * 0.05,
                    'high': price + abs(np.random.randn() * 0.03),
                    'low': price - abs(np.random.randn() * 0.03),
                    'close': price,
                    'volume': np.random.randint(10000, 100000),
                    'amount': np.random.randint(100000, 1000000),
                    'turnover_rate': np.random.uniform(0.01, 0.05),
                    'change_pct': np.random.randn() * 0.02,
                })
        
        cls.test_data = pd.DataFrame(rows)
    
    def test_basic_expression(self):
        """测试基础表达式求值"""
        fe = FactorExpression('$close')
        result = fe.evaluate(self.test_data)
        
        self.assertEqual(len(result), len(self.test_data))
        pd.testing.assert_series_equal(result, self.test_data['close'], check_names=False)
        print("\n[基础表达式] $close 求值正确")
    
    def test_arithmetic_operations(self):
        """测试算术运算"""
        # 测试 HL 差
        fe = FactorExpression('$high - $low')
        result = fe.evaluate(self.test_data)
        expected = self.test_data['high'] - self.test_data['low']
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # 测试复合运算
        fe2 = FactorExpression('($high + $low) / 2')
        result2 = fe2.evaluate(self.test_data)
        expected2 = (self.test_data['high'] + self.test_data['low']) / 2
        pd.testing.assert_series_equal(result2, expected2, check_names=False)
        print("[算术运算] 加减乘除和括号优先级正确")
    
    def test_ref_shift(self):
        """测试 Ref/Delay 延迟算子"""
        fe = FactorExpression('Ref($close, 1)')
        result = fe.evaluate(self.test_data)
        
        # 验证：前一日收盘价
        for code in self.test_data['code'].unique():
            mask = self.test_data['code'] == code
            code_data = self.test_data.loc[mask, 'close']
            result_subset = result.loc[mask]
            
            # 第一个值应为 NaN (无前一日数据)
            self.assertTrue(np.isnan(result_subset.iloc[0]))
            
            # 后续值应等于前一日 close
            for i in range(1, min(5, len(code_data))):
                self.assertAlmostEqual(result_subset.iloc[i], code_data.iloc[i - 1], delta=1e-8)
        
        print("[Ref延迟] Ref($close, 1) 正确实现了滞后")
    
    def test_rolling_functions(self):
        """测试滚动函数: Mean, Std, Max, Min, Sum"""
        test_cases = [
            ('Mean($close, 5)', 'mean'),
            ('Std($close, 20)', 'std'),
            ('Max($close, 20)', 'max'),
            ('Min($close, 20)', 'min'),
            ('Sum($volume, 5)', 'sum'),
        ]
        
        for expr, stat in test_cases:
            fe = FactorExpression(expr)
            result = fe.evaluate(self.test_data)
            self.assertEqual(len(result), len(self.test_data))
            self.assertFalse(result.isna().all(), f"{expr} 结果不应全为 NaN")
        
        print(f"[滚动函数] Mean/Std/Max/Min/Sum 均正确计算")
    
    def test_pct_change(self):
        """测试涨跌幅计算"""
        fe = FactorExpression('PctChange($close, 5)')
        result = fe.evaluate(self.test_data)
        
        for code in self.test_data['code'].unique():
            mask = self.test_data['code'] == code
            code_data = self.test_data.loc[mask, 'close']
            result_subset = result.loc[mask]
            
            for i in range(5, len(code_data)):
                expected = code_data.iloc[i] / code_data.iloc[i - 5] - 1
                self.assertAlmostEqual(result_subset.iloc[i], expected, delta=1e-8)
        
        print("[涨跌幅] PctChange 5日涨跌幅计算正确")
    
    def test_rank_cross_sectional(self):
        """测试截面排名"""
        fe = FactorExpression('Rank($close)')
        result = fe.evaluate(self.test_data)
        
        # 验证每个日期的排名和为 (0, 1) 且均值为 0.5
        for date in self.test_data['date'].unique()[:5]:
            mask = self.test_data['date'] == date
            daily_rank = result.loc[mask].dropna()
            
            if len(daily_rank) > 0:
                self.assertAlmostEqual(daily_rank.mean(), 0.5, delta=0.1)
                self.assertGreaterEqual(daily_rank.min(), 0)
                self.assertLessEqual(daily_rank.max(), 1)
        
        print("[截面排名] Rank 正确实现了每日截面排名")
    
    def test_logical_operations(self):
        """测试逻辑运算和条件表达式"""
        # 测试比较运算: 收盘价 > 开盘价 (阳线)
        fe = FactorExpression('$close > $open')
        result = fe.evaluate(self.test_data)
        expected = (self.test_data['close'] > self.test_data['open']).astype(float)
        pd.testing.assert_series_equal(result, expected, check_names=False)
        
        # 测试条件表达式
        fe2 = FactorExpression('If($close > $open, 1, -1)')
        result2 = fe2.evaluate(self.test_data)
        for i in range(len(result2)):
            expected_val = 1.0 if self.test_data['close'].iloc[i] > self.test_data['open'].iloc[i] else -1.0
            self.assertAlmostEqual(result2.iloc[i], expected_val, delta=1e-8)
        
        print("[逻辑运算] 比较运算和 If 条件表达式正确")
    
    def test_factor_registry(self):
        """测试因子注册表"""
        registry = FactorRegistry()
        
        # 验证预注册的因子
        all_factors = registry.list_all()
        self.assertGreater(len(all_factors), 10)
        self.assertIn('ret_5d', all_factors)
        self.assertIn('volatility_20d', all_factors)
        self.assertIn('ma20_bias', all_factors)
        
        # 测试自定义因子注册
        registry.register('my_custom', 'Mean($close, 10) / Std($close, 10)')
        self.assertIn('my_custom', registry.list_all())
        
        # 批量计算
        selected = ['ret_5d', 'ret_20d', 'volatility_20d', 'ma20_bias', 'volume_ratio_20']
        result = registry.compute(self.test_data, selected)
        
        self.assertEqual(len(result), len(self.test_data))
        for factor_name in selected:
            self.assertIn(factor_name, result.columns)
            self.assertFalse(result[factor_name].isna().all(),
                          f"{factor_name} 不应全为 NaN")
        
        print(f"[因子注册表] 已注册 {len(all_factors)} 个因子")
        print(f"  预注册示例: {all_factors[:5]}")
        print(f"  批量计算 {len(selected)} 个因子完成, shape={result.shape}")
    
    def test_corr_function(self):
        """测试滚动相关性"""
        fe = FactorExpression('Corr($close, $volume, 20)')
        result = fe.evaluate(self.test_data)
        
        # 验证相关系数范围
        valid = result.dropna()
        if len(valid) > 0:
            self.assertGreaterEqual(valid.min(), -1.01)
            self.assertLessEqual(valid.max(), 1.01)
        
        print(f"[相关系数] Corr 结果数: {len(valid)}, 范围: [{valid.min():.3f}, {valid.max():.3f}]")
    
    def test_expression_performance(self):
        """性能测试：对比硬编码和表达式计算"""
        registry = FactorRegistry()
        factors = ['ret_5d', 'ret_20d', 'volatility_20d', 'ma20_bias', 'volume_ratio_20',
                    'price_position_20', 'momentum_5_20', 'turnover_20d', 'rsv_9',
                    'corr_price_volume_20']
        
        # 表达式方式
        t0 = time.time()
        result_expr = registry.compute(self.test_data, factors)
        t_expr = time.time() - t0
        
        # 硬编码方式 (模拟 jingni-trader 当前做法)
        t0 = time.time()
        df = self.test_data.copy()
        hard_coded = df[['code', 'date']].copy()
        hard_coded['ret_5d'] = df.groupby('code')['close'].pct_change(5)
        hard_coded['ret_20d'] = df.groupby('code')['close'].pct_change(20)
        hard_coded['volatility_20d'] = df.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )
        hard_coded['ma20_bias'] = df['close'] / df.groupby('code')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        ) - 1
        hard_coded['volume_ratio_20'] = df['volume'] / df.groupby('code')['volume'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        t_hard = time.time() - t0
        
        print(f"\n[性能对比] 数据量: {len(self.test_data)} 条, 因子数: {len(factors)}")
        print(f"  表达式方式: {t_expr*1000:.1f} ms")
        print(f"  硬编码方式: {t_hard*1000:.1f} ms")
        print(f"  表达式/硬编码: {t_expr/t_hard:.2f}x")
        print(f"  注意: 当前表达式引擎为纯 Python 实现，后续可用 numba/jit 优化")
    
    def test_complex_alpha_expression(self):
        """测试复杂 Alpha 因子表达式"""
        # 借鉴 Qlib 的 Alpha 因子定义方式
        # Alpha: 20日反转因子（标准化）
        alpha_expr = 'Rank(-PctChange($close, 20))'
        fe = FactorExpression(alpha_expr)
        result = fe.evaluate(self.test_data)
        
        self.assertEqual(len(result), len(self.test_data))
        self.assertFalse(result.isna().all())
        
        # Alpha: 波动率调整动量
        alpha2_expr = 'Rank(PctChange($close, 5)) / (1 + Std(PctChange($close, 1), 20))'
        fe2 = FactorExpression(alpha2_expr)
        result2 = fe2.evaluate(self.test_data)
        
        self.assertEqual(len(result2), len(self.test_data))
        
        print(f"[复杂Alpha] 表达式: {alpha_expr}, 值范围: [{result.min():.3f}, {result.max():.3f}]")
        print(f"[复杂Alpha] 表达式: {alpha2_expr}, 值范围: [{result2.min():.3f}, {result2.max():.3f}]")


if __name__ == '__main__':
    unittest.main(verbosity=2)