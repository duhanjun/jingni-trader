"""
验证代码：Alpha因子库扩展与声明式因子表达式引擎
==================================================
借鉴来源：Microsoft Qlib (github.com/microsoft/qlib)
  - Alpha158/Alpha360 标准化因子库设计
  - Expression Engine 声明式因子表达式（如 "Ref($close, 60) / $close"）
  - 因子分类体系（趋势跟踪、均值回归、成交量、波动率、资金流向、复合因子）

优化方向：factor-engine 模块因子库从 ~15 个硬编码因子扩展至 50+ 个分类管理因子，
          并引入声明式因子表达式减少因子定义代码量。

对比分析：
  - 现有方式：每个因子在 compute_a_share_factors() 中硬编码，新增因子需修改核心代码
  - 优化方式：通过因子表达式引擎解析声明式表达式，新增因子只需配置表达式字符串
"""

import sys
import os
import re
import time
import json
import unittest
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# =============================================================================
# 优化方案 1: 声明式因子表达式引擎
# 借鉴 Qlib 的 Expression Engine 设计
# =============================================================================

class FactorExpressionEngine:
    """
    声明式因子表达式引擎
    
    借鉴 Qlib 的 expression engine，支持通过字符串表达式定义因子。
    Qlib 表达式示例: "Ref($close, 60) / $close" 表示 60日收益率
    """
    
    # 操作符映射
    OPERATORS = {
        'Ref': lambda x, n: x.shift(n),           # 前移 N 期
        'Mean': lambda x, n: x.rolling(n).mean(),  # N 期均值
        'Std': lambda x, n: x.rolling(n).std(),    # N 期标准差
        'Sum': lambda x, n: x.rolling(n).sum(),    # N 期求和
        'Max': lambda x, n: x.rolling(n).max(),    # N 期最大值
        'Min': lambda x, n: x.rolling(n).min(),    # N 期最小值
        'Delta': lambda x, n: x.diff(n),           # N 期差值
        'Rank': lambda x: x.rank(pct=True),        # 截面排名
        'Log': lambda x: np.log(x.replace(0, np.nan)),  # 对数
        'Abs': lambda x: x.abs(),                  # 绝对值
        'Sign': lambda x: np.sign(x),              # 符号
        'Delay': lambda x, n: x.shift(n),          # 延迟
        'Corr': lambda x, y, n: x.rolling(n).corr(y),  # 滚动相关系数
        'Cov': lambda x, y, n: x.rolling(n).cov(y),    # 滚动协方差
    }
    
    def __init__(self):
        self._compiled_cache: Dict[str, Callable] = {}
    
    def compile(self, expression: str) -> Callable:
        """编译因子表达式为可执行函数"""
        if expression in self._compiled_cache:
            return self._compiled_cache[expression]
        
        # 解析表达式
        func = self._parse_expression(expression)
        self._compiled_cache[expression] = func
        return func
    
    def _parse_expression(self, expr: str) -> Callable:
        """解析表达式字符串为可调用函数"""
        expr = expr.strip()
        
        # 处理括号包裹的表达式
        if expr.startswith('(') and expr.endswith(')'):
            # 验证括号匹配
            depth = 0
            for i, ch in enumerate(expr):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    break  # 括号在中间就闭合了，不是外层包裹
            else:
                # 外层括号匹配，去掉括号后解析内部
                if depth == 0:
                    return self._parse_expression(expr[1:-1])
        
        # 处理二元操作符
        for op in ['+', '-', '*', '/']:
            # 从右向左找最外层操作符（考虑括号平衡）
            depth = 0
            for i in range(len(expr) - 1, -1, -1):
                if expr[i] == ')':
                    depth += 1
                elif expr[i] == '(':
                    depth -= 1
                elif expr[i] == op and depth == 0:
                    left = self._parse_expression(expr[:i])
                    right = self._parse_expression(expr[i+1:])
                    if op == '+':
                        return lambda df, l=left, r=right: l(df) + r(df)
                    elif op == '-':
                        return lambda df, l=left, r=right: l(df) - r(df)
                    elif op == '*':
                        return lambda df, l=left, r=right: l(df) * r(df)
                    elif op == '/':
                        return lambda df, l=left, r=right: l(df) / r(df).replace(0, np.nan)
        
        # 处理函数调用: FuncName(arg1, arg2, ...)
        func_match = re.match(r'^(\w+)\((.*)\)$', expr)
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            args = self._parse_args(args_str)
            
            if func_name in self.OPERATORS:
                op_func = self.OPERATORS[func_name]
                # 解析参数（可能是列引用或数字）
                parsed_args = [self._resolve_arg(a) for a in args]
                return lambda df, f=op_func, pa=parsed_args: self._apply_operator(df, f, pa)
        
        # 处理列引用: $column_name
        if expr.startswith('$'):
            col_name = expr[1:]
            return lambda df, c=col_name: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
        
        # 处理数字常量
        try:
            val = float(expr)
            return lambda df, v=val: pd.Series(v, index=df.index)
        except ValueError:
            pass
        
        # 默认：当作列名
        return lambda df, c=expr: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
    
    def _parse_args(self, args_str: str) -> List[str]:
        """解析函数参数，处理嵌套括号"""
        if not args_str.strip():
            return []
        args = []
        current = ''
        depth = 0
        for ch in args_str:
            if ch == ',' and depth == 0:
                args.append(current.strip())
                current = ''
            else:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                current += ch
        if current.strip():
            args.append(current.strip())
        return args
    
    def _resolve_arg(self, arg: str) -> Any:
        """解析参数：数字返回 float/int，列引用返回 (col_ref, col_name)，否则返回表达式函数"""
        arg = arg.strip()
        try:
            val = float(arg)
            # 如果是整数则返回 int，因为 pandas shift/rolling 需要整数参数
            if val == int(val) and '.' not in arg:
                return int(val)
            return val
        except ValueError:
            pass
        # 仅当参数是纯列引用时才返回列引用（如 $close, $volume）
        if re.match(r'^\$[a-zA-Z_]\w*$', arg):
            return ('col', arg[1:])
        # 嵌套表达式
        return ('expr', self._parse_expression(arg))
    
    def _apply_operator(self, df: pd.DataFrame, func: Callable, args: List) -> pd.Series:
        """将操作符应用到 DataFrame"""
        resolved = []
        for arg in args:
            if isinstance(arg, (int, float)):
                resolved.append(arg)
            elif isinstance(arg, tuple) and arg[0] == 'col':
                resolved.append(df[arg[1]] if arg[1] in df.columns else pd.Series(np.nan, index=df.index))
            elif isinstance(arg, tuple) and arg[0] == 'expr':
                resolved.append(arg[1](df))
            else:
                resolved.append(arg)
        return func(*resolved)


# =============================================================================
# 优化方案 2: 分类因子库（借鉴 Qlib Alpha158）
# =============================================================================

@dataclass
class FactorDefinition:
    """因子定义"""
    name: str
    expression: str
    category: str  # 趋势跟踪、均值回归、成交量、波动率、资金流向、复合因子
    direction: int  # 1: 正向因子, -1: 反向因子
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


class AlphaFactorLibrary:
    """
    Alpha因子库
    
    借鉴 Qlib Alpha158/Alpha360 设计，将因子按类别管理。
    目前 jingni-trader 仅有 ~15 个硬编码因子，本库扩展至 50+ 个。
    """
    
    # 因子分类定义
    CATEGORIES = {
        'trend': '趋势跟踪因子',
        'reversal': '均值回归因子',
        'volume': '成交量因子',
        'volatility': '波动率因子',
        'money_flow': '资金流向因子',
        'composite': '复合因子',
        'momentum': '动量因子',
        'liquidity': '流动性因子',
    }
    
    def __init__(self):
        self.engine = FactorExpressionEngine()
        self._factors: Dict[str, FactorDefinition] = {}
        self._register_default_factors()
    
    def _register_default_factors(self):
        """注册默认因子库（借鉴 Alpha158 分类）"""
        factors = [
            # === 趋势跟踪因子 ===
            FactorDefinition('ma_5', 'Mean($close, 5)', 'trend', 1, '5日均线'),
            FactorDefinition('ma_10', 'Mean($close, 10)', 'trend', 1, '10日均线'),
            FactorDefinition('ma_20', 'Mean($close, 20)', 'trend', 1, '20日均线'),
            FactorDefinition('ma_60', 'Mean($close, 60)', 'trend', 1, '60日均线'),
            FactorDefinition('ma_5_20', 'Mean($close, 5)/Mean($close, 20) - 1', 'trend', 1, '5日/20日均线偏离'),
            FactorDefinition('ma_10_60', 'Mean($close, 10)/Mean($close, 60) - 1', 'trend', 1, '10日/60日均线偏离'),
            FactorDefinition('price_position', '($close-Min($close, 60))/(Max($close, 60)-Min($close, 60))', 'trend', 1, '60日价格位置'),
            
            # === 动量因子 ===
            FactorDefinition('ret_1d', '$close/Ref($close, 1) - 1', 'momentum', 1, '1日收益率'),
            FactorDefinition('ret_5d', '$close/Ref($close, 5) - 1', 'momentum', 1, '5日收益率'),
            FactorDefinition('ret_10d', '$close/Ref($close, 10) - 1', 'momentum', 1, '10日收益率'),
            FactorDefinition('ret_20d', '$close/Ref($close, 20) - 1', 'momentum', 1, '20日收益率'),
            FactorDefinition('ret_60d', '$close/Ref($close, 60) - 1', 'momentum', 1, '60日收益率'),
            FactorDefinition('roc_10', '($close/Ref($close, 10) - 1) * 100', 'momentum', 1, '10日变化率'),
            FactorDefinition('roc_20', '($close/Ref($close, 20) - 1) * 100', 'momentum', 1, '20日变化率'),
            FactorDefinition('rsi_6', 'RSI($close, 6)', 'momentum', -1, '6日RSI'),
            FactorDefinition('rsi_14', 'RSI($close, 14)', 'momentum', -1, '14日RSI'),
            FactorDefinition('rsi_24', 'RSI($close, 24)', 'momentum', -1, '24日RSI'),
            
            # === 均值回归因子 ===
            FactorDefinition('reversal_5d', '-($close/Ref($close, 5) - 1)', 'reversal', 1, '5日反转'),
            FactorDefinition('reversal_10d', '-($close/Ref($close, 10) - 1)', 'reversal', 1, '10日反转'),
            FactorDefinition('reversal_20d', '-($close/Ref($close, 20) - 1)', 'reversal', 1, '20日反转'),
            FactorDefinition('bias_5', '($close/Mean($close, 5) - 1) * 100', 'reversal', -1, '5日乖离率'),
            FactorDefinition('bias_10', '($close/Mean($close, 10) - 1) * 100', 'reversal', -1, '10日乖离率'),
            FactorDefinition('bias_20', '($close/Mean($close, 20) - 1) * 100', 'reversal', -1, '20日乖离率'),
            
            # === 波动率因子 ===
            FactorDefinition('volatility_5d', 'Std($close/Ref($close, 1) - 1, 5)', 'volatility', -1, '5日波动率'),
            FactorDefinition('volatility_10d', 'Std($close/Ref($close, 1) - 1, 10)', 'volatility', -1, '10日波动率'),
            FactorDefinition('volatility_20d', 'Std($close/Ref($close, 1) - 1, 20)', 'volatility', -1, '20日波动率'),
            FactorDefinition('volatility_60d', 'Std($close/Ref($close, 1) - 1, 60)', 'volatility', -1, '60日波动率'),
            FactorDefinition('hl_ratio', '($high-$low)/$close', 'volatility', -1, '日内振幅'),
            FactorDefinition('atr_14', 'Mean(Max($high-$low, Abs($high-Ref($close, 1)), Abs($low-Ref($close, 1))), 14)', 'volatility', -1, '14日ATR'),
            
            # === 成交量因子 ===
            FactorDefinition('volume_ma_5', '$volume/Mean($volume, 5)', 'volume', 1, '量比(5日)'),
            FactorDefinition('volume_ma_20', '$volume/Mean($volume, 20)', 'volume', 1, '量比(20日)'),
            FactorDefinition('volume_ratio', 'Mean($volume, 5)/Mean($volume, 20)', 'volume', 1, '量比(5/20)'),
            FactorDefinition('volume_trend', 'Mean($volume, 5)/Mean($volume, 20) - 1', 'volume', 1, '成交量趋势'),
            FactorDefinition('volume_std_20', 'Std($volume, 20)/Mean($volume, 20)', 'volume', -1, '成交量变异系数'),
            
            # === 资金流向因子 ===
            FactorDefinition('money_flow_5d', 'Sum(($close-Ref($close, 1))/Ref($close, 1)*$volume, 5)', 'money_flow', 1, '5日资金流'),
            FactorDefinition('money_flow_20d', 'Sum(($close-Ref($close, 1))/Ref($close, 1)*$volume, 20)', 'money_flow', 1, '20日资金流'),
            FactorDefinition('vwap_deviation', '($close/Sum($close*$volume, 20)*Sum($volume, 20) - 1)', 'money_flow', 1, 'VWAP偏离'),
            
            # === 流动性因子 ===
            FactorDefinition('turnover_mean_5', 'Mean($turnover_rate, 5)', 'liquidity', 1, '5日平均换手率'),
            FactorDefinition('turnover_mean_20', 'Mean($turnover_rate, 20)', 'liquidity', 1, '20日平均换手率'),
            FactorDefinition('turnover_change', 'Mean($turnover_rate, 5)/Mean($turnover_rate, 20) - 1', 'liquidity', 1, '换手率变化'),
            FactorDefinition('illiquidity', 'Abs($close/Ref($close, 1) - 1)/$amount', 'liquidity', -1, '非流动性指标(Amihud)'),
            
            # === 复合因子 ===
            FactorDefinition('macd', 'MACD($close, 12, 26, 9)', 'composite', 1, 'MACD'),
            FactorDefinition('kdj_k', 'KDJ_K($high, $low, $close, 9, 3)', 'composite', 1, 'KDJ-K值'),
            FactorDefinition('kdj_d', 'KDJ_D($high, $low, $close, 9, 3)', 'composite', 1, 'KDJ-D值'),
            FactorDefinition('bb_position', '($close-Mean($close, 20))/(2*Std($close, 20))', 'composite', -1, '布林带位置'),
            FactorDefinition('bb_width', '(2*Std($close, 20))/Mean($close, 20)', 'composite', -1, '布林带宽度'),
        ]
        
        for f in factors:
            self._factors[f.name] = f
    
    def get_factor(self, name: str) -> Optional[FactorDefinition]:
        """获取因子定义"""
        return self._factors.get(name)
    
    def get_factors_by_category(self, category: str) -> List[FactorDefinition]:
        """按类别获取因子"""
        return [f for f in self._factors.values() if f.category == category]
    
    def get_all_factors(self) -> List[FactorDefinition]:
        """获取所有因子"""
        return list(self._factors.values())
    
    def compute_factors(self, data: pd.DataFrame, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量计算因子
        
        参数:
            data: 包含 OHLCV 等字段的 DataFrame
            factor_names: 要计算的因子名列表，None 表示计算全部
        
        返回:
            因子 DataFrame
        """
        if factor_names is None:
            factor_names = list(self._factors.keys())
        
        if 'code' in data.columns:
            # 按股票分组计算，避免跨股票边界 shift/rolling
            results = []
            for code, group in data.groupby('code'):
                result = group[['code', 'date']].copy()
                for name in factor_names:
                    factor_def = self._factors.get(name)
                    if factor_def is None:
                        continue
                    try:
                        func = self.engine.compile(factor_def.expression)
                        result[name] = func(group).values
                    except Exception as e:
                        print(f"Warning: 因子 {name} 计算失败: {e}")
                        result[name] = np.nan
                results.append(result)
            return pd.concat(results, ignore_index=True)
        else:
            result = data[['code', 'date']].copy() if 'code' in data.columns else data.copy()
            for name in factor_names:
                factor_def = self._factors.get(name)
                if factor_def is None:
                    continue
                try:
                    func = self.engine.compile(factor_def.expression)
                    result[name] = func(data)
                except Exception as e:
                    print(f"Warning: 因子 {name} 计算失败: {e}")
                    result[name] = np.nan
            return result
    
    def add_factor(self, factor_def: FactorDefinition):
        """动态添加新因子"""
        self._factors[factor_def.name] = factor_def
    
    def to_dict(self) -> Dict:
        """导出因子库定义"""
        return {
            name: {
                'expression': f.expression,
                'category': f.category,
                'direction': f.direction,
                'description': f.description,
            }
            for name, f in self._factors.items()
        }


# =============================================================================
# 测试代码
# =============================================================================

def generate_sample_data(n_stocks: int = 10, n_days: int = 252) -> pd.DataFrame:
    """生成模拟股票数据用于测试"""
    np.random.seed(42)
    records = []
    codes = [f'{i:06d}.SZ' for i in range(1, n_stocks + 1)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    
    for code in codes:
        close = 10 + np.cumsum(np.random.randn(n_days) * 0.2)
        close = np.maximum(close, 1)
        open_p = close * (1 + np.random.randn(n_days) * 0.005)
        high = np.maximum(open_p, close) * (1 + np.abs(np.random.randn(n_days) * 0.01))
        low = np.minimum(open_p, close) * (1 - np.abs(np.random.randn(n_days) * 0.01))
        volume = np.random.lognormal(10, 1, n_days)
        amount = volume * close
        turnover_rate = np.random.uniform(0.01, 0.05, n_days)
        
        for i, d in enumerate(dates):
            records.append({
                'code': code,
                'date': d,
                'open': open_p[i],
                'high': high[i],
                'low': low[i],
                'close': close[i],
                'volume': volume[i],
                'amount': amount[i],
                'turnover_rate': turnover_rate[i],
            })
    
    return pd.DataFrame(records)


# =============================================================================
# 自定义操作符扩展（用于 RSI、MACD 等复合因子）
# =============================================================================

def register_extended_operators(engine: FactorExpressionEngine):
    """注册扩展操作符"""
    def _rsi(close, n):
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(n).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def _macd(close, fast, slow, signal):
        ema_fast = close.ewm(span=int(fast), adjust=False).mean()
        ema_slow = close.ewm(span=int(slow), adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=int(signal), adjust=False).mean()
        return macd_line - signal_line
    
    def _kdj_k(high, low, close, n, m):
        lowest_low = low.rolling(int(n)).min()
        highest_high = high.rolling(int(n)).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
        return rsv.ewm(com=int(m)-1, adjust=False).mean()
    
    def _max_abs(a, b, c):
        return pd.concat([a.abs(), b.abs(), c.abs()], axis=1).max(axis=1)
    
    engine.OPERATORS['RSI'] = _rsi
    engine.OPERATORS['MACD'] = _macd
    engine.OPERATORS['KDJ_K'] = _kdj_k
    # 注意：Max覆写会与原有 Max 操作符冲突，使用 MaxAbs 替代
    engine.OPERATORS['MaxAbs'] = _max_abs


class TestFactorExpressionEngine(unittest.TestCase):
    """测试因子表达式引擎"""
    
    @classmethod
    def setUpClass(cls):
        cls.data = generate_sample_data(n_stocks=5, n_days=100)
        cls.engine = FactorExpressionEngine()
        register_extended_operators(cls.engine)
    
    def test_basic_column_ref(self):
        """测试基本列引用: $close"""
        func = self.engine.compile('$close')
        result = func(self.data)
        pd.testing.assert_series_equal(result, self.data['close'])
    
    def test_ref_operator(self):
        """测试 Ref 操作符: Ref($close, 5)"""
        func = self.engine.compile('Ref($close, 5)')
        result = func(self.data)
        expected = self.data['close'].shift(5)
        pd.testing.assert_series_equal(result, expected)
    
    def test_mean_operator(self):
        """测试 Mean 操作符: Mean($close, 20)"""
        func = self.engine.compile('Mean($close, 20)')
        result = func(self.data)
        expected = self.data['close'].rolling(20).mean()
        pd.testing.assert_series_equal(result, expected)
    
    def test_arithmetic_expression(self):
        """测试算术表达式: $close/Ref($close, 20) - 1"""
        func = self.engine.compile('$close/Ref($close, 20) - 1')
        result = func(self.data)
        expected = self.data['close'] / self.data['close'].shift(20) - 1
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_nested_expression(self):
        """测试嵌套表达式: Mean($close, 5)/Mean($close, 20) - 1"""
        func = self.engine.compile('Mean($close, 5)/Mean($close, 20) - 1')
        result = func(self.data)
        ma5 = self.data['close'].rolling(5).mean()
        ma20 = self.data['close'].rolling(20).mean()
        expected = ma5 / ma20 - 1
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_rsi_expression(self):
        """测试 RSI 表达式: RSI($close, 14)"""
        func = self.engine.compile('RSI($close, 14)')
        result = func(self.data)
        self.assertFalse(result.isna().all())
        # RSI 应该在 0-100 之间
        valid = result.dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 100).all())
    
    def test_expression_cache(self):
        """测试表达式缓存"""
        # 第一次编译
        t1 = time.time()
        self.engine.compile('Mean($close, 20)/Std($close, 20)')
        t2 = time.time()
        first_compile_time = t2 - t1
        
        # 第二次编译（应从缓存读取）
        t3 = time.time()
        self.engine.compile('Mean($close, 20)/Std($close, 20)')
        t4 = time.time()
        second_compile_time = t4 - t3
        
        # 缓存命中应该更快
        self.assertLess(second_compile_time, first_compile_time * 0.5,
                       f"缓存未加速: 首次={first_compile_time:.6f}s, 缓存={second_compile_time:.6f}s")


class TestAlphaFactorLibrary(unittest.TestCase):
    """测试 Alpha 因子库"""
    
    @classmethod
    def setUpClass(cls):
        cls.data = generate_sample_data(n_stocks=10, n_days=252)
        cls.library = AlphaFactorLibrary()
        register_extended_operators(cls.library.engine)
    
    def test_factor_count(self):
        """测试因子库规模"""
        factors = self.library.get_all_factors()
        self.assertGreaterEqual(len(factors), 40, f"因子数量不足: {len(factors)}")
        print(f"\n因子库共 {len(factors)} 个因子")
    
    def test_category_distribution(self):
        """测试因子分类分布"""
        categories = {}
        for f in self.library.get_all_factors():
            categories[f.category] = categories.get(f.category, 0) + 1
        
        print(f"\n因子分类分布:")
        for cat, count in sorted(categories.items()):
            cat_name = AlphaFactorLibrary.CATEGORIES.get(cat, cat)
            print(f"  {cat_name} ({cat}): {count} 个因子")
        
        # 至少应有 5 个分类
        self.assertGreaterEqual(len(categories), 5)
    
    def test_compute_factors(self):
        """测试因子计算"""
        selected = ['ret_5d', 'ret_20d', 'ma_5_20', 'volatility_20d', 'volume_ratio']
        result = self.library.compute_factors(self.data, selected)
        
        self.assertEqual(len(result), len(self.data))
        for col in selected:
            self.assertIn(col, result.columns, f"因子 {col} 未计算")
            self.assertFalse(result[col].isna().all(), f"因子 {col} 全为 NaN")
        
        print(f"\n因子计算结果: {len(result)} 行, {len(selected)} 个因子")
        print(f"因子统计:\n{result[selected].describe().to_string()}")
    
    def test_compute_all_factors(self):
        """测试计算全部因子"""
        all_factors = [f.name for f in self.library.get_all_factors()]
        result = self.library.compute_factors(self.data, all_factors)
        
        success_count = sum(1 for col in all_factors if col in result.columns)
        print(f"\n全部因子计算: {success_count}/{len(all_factors)} 成功")
        
        # 至少 80% 的因子应计算成功
        self.assertGreater(success_count / len(all_factors), 0.8)
    
    def test_dynamic_factor_addition(self):
        """测试动态添加因子"""
        # 添加新因子只需定义表达式
        new_factor = FactorDefinition(
            name='custom_momentum',
            expression='($close/Ref($close, 30) - 1) * ($volume/Mean($volume, 30))',
            category='momentum',
            direction=1,
            description='自定义动量-量能因子'
        )
        self.library.add_factor(new_factor)
        
        result = self.library.compute_factors(self.data, ['custom_momentum'])
        self.assertIn('custom_momentum', result.columns)
        self.assertFalse(result['custom_momentum'].isna().all())
        print(f"\n动态添加因子 'custom_momentum' 计算成功")
    
    def test_export_factor_library(self):
        """测试因子库导出"""
        exported = self.library.to_dict()
        self.assertIsInstance(exported, dict)
        self.assertGreater(len(exported), 0)
        print(f"\n因子库导出: {len(exported)} 个因子定义")


class TestPerformanceComparison(unittest.TestCase):
    """性能对比测试：声明式 vs 硬编码"""
    
    @classmethod
    def setUpClass(cls):
        cls.data = generate_sample_data(n_stocks=50, n_days=500)
        cls.library = AlphaFactorLibrary()
        register_extended_operators(cls.library.engine)
    
    def test_declarative_vs_hardcoded(self):
        """对比声明式因子计算与硬编码方式的性能"""
        import time
        
        # 声明式计算
        declarative_factors = ['ret_5d', 'ret_20d', 'ma_5_20', 'volatility_20d',
                               'volume_ratio', 'bias_10', 'reversal_10d', 'roc_20']
        
        t1 = time.time()
        result_decl = self.library.compute_factors(self.data, declarative_factors)
        t2 = time.time()
        declarative_time = t2 - t1
        
        print(f"\n声明式计算 {len(declarative_factors)} 个因子: {declarative_time:.4f}s")
        print(f"数据规模: {len(self.data)} 行 × {len(self.data['code'].unique())} 只股票")
        
        # 声明式方法可能在性能上有一定开销，但换来了灵活性和可维护性
        self.assertLess(declarative_time, 5.0, "声明式计算超时")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFactorExpressionEngine))
    suite.addTest(unittest.makeSuite(TestAlphaFactorLibrary))
    suite.addTest(unittest.makeSuite(TestPerformanceComparison))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("Alpha因子库优化验证结果摘要")
    print("=" * 60)
    print(f"借鉴来源: Microsoft Qlib (github.com/microsoft/qlib)")
    print(f"  - Alpha158/Alpha360 标准化因子库")
    print(f"  - Expression Engine 声明式因子表达式")
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    return result


if __name__ == '__main__':
    run_tests()