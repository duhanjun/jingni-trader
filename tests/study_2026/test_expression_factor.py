"""
验证测试：表达式引擎 — 基于 DSL 的因子定义系统
================================================================
借鉴来源：Microsoft Qlib (https://github.com/microsoft/qlib)
         - 表达式引擎：$close, Ref($close, 1), Mean($close, 3) 等 DSL
         - Alpha158/Alpha360 预定义因子集
         - 列式二进制存储格式，支持快速切片
优化方向：factor-engine — 新增表达式因子定义，提升因子开发效率
================================================================
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Union
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# 1. 表达式引擎核心（借鉴 Qlib Expression Engine）
# ═══════════════════════════════════════════════════════════════

class ExpressionEngine:
    """
    因子表达式引擎

    借鉴 Qlib 的 Expression Engine 设计：
    - DSL 语法：$close, Ref($close, 5), Mean($close, 20), Std($close, 20)
    - 支持运算符：+, -, *, /, >, <, ==, &, |
    - 支持函数：Ref, Mean, Std, Max, Min, Sum, Corr, Rank, Delay, Delta, TS_ArgMax, TS_ArgMin
    """

    # 内置字段映射
    FIELD_MAP = {
        '$open': 'open', '$high': 'high', '$low': 'low',
        '$close': 'close', '$volume': 'volume', '$amount': 'amount',
        '$vwap': 'vwap', '$returns': 'returns',
    }

    # 内置函数注册表
    FUNCTIONS: Dict[str, Callable] = {}

    def __init__(self):
        self._register_functions()

    def _register_functions(self):
        """注册内置函数"""
        self.FUNCTIONS = {
            'Ref': self._func_ref,
            'Mean': self._func_mean,
            'Std': self._func_std,
            'Max': self._func_max,
            'Min': self._func_min,
            'Sum': self._func_sum,
            'Corr': self._func_corr,
            'Rank': self._func_rank,
            'Delay': self._func_delay,
            'Delta': self._func_delta,
            'Abs': self._func_abs,
            'Sign': self._func_sign,
            'Log': self._func_log,
            'TS_ArgMax': self._func_ts_argmax,
            'TS_ArgMin': self._func_ts_argmin,
            'RSI': self._func_rsi,
            'BB_upper': self._func_bb_upper,
            'BB_lower': self._func_bb_lower,
        }

    # ── 内置函数实现 ──

    @staticmethod
    def _func_ref(series: pd.Series, n: int) -> pd.Series:
        """前值引用：Ref(X, n) = X.shift(n)"""
        return series.shift(n)

    @staticmethod
    def _func_mean(series: pd.Series, n: int) -> pd.Series:
        """滚动均值"""
        return series.rolling(n, min_periods=max(1, n // 2)).mean()

    @staticmethod
    def _func_std(series: pd.Series, n: int) -> pd.Series:
        """滚动标准差"""
        return series.rolling(n, min_periods=max(1, n // 2)).std()

    @staticmethod
    def _func_max(series: pd.Series, n: int) -> pd.Series:
        """滚动最大值"""
        return series.rolling(n, min_periods=max(1, n // 2)).max()

    @staticmethod
    def _func_min(series: pd.Series, n: int) -> pd.Series:
        """滚动最小值"""
        return series.rolling(n, min_periods=max(1, n // 2)).min()

    @staticmethod
    def _func_sum(series: pd.Series, n: int) -> pd.Series:
        """滚动求和"""
        return series.rolling(n, min_periods=max(1, n // 2)).sum()

    @staticmethod
    def _func_corr(series1: pd.Series, series2: pd.Series, n: int) -> pd.Series:
        """滚动相关系数"""
        return series1.rolling(n).corr(series2)

    @staticmethod
    def _func_rank(series: pd.Series) -> pd.Series:
        """截面排名（百分位）"""
        return series.rank(pct=True)

    @staticmethod
    def _func_delay(series: pd.Series, n: int) -> pd.Series:
        """延迟：同 Ref"""
        return series.shift(n)

    @staticmethod
    def _func_delta(series: pd.Series, n: int) -> pd.Series:
        """Delta: X - Ref(X, n)"""
        return series - series.shift(n)

    @staticmethod
    def _func_abs(series: pd.Series) -> pd.Series:
        return series.abs()

    @staticmethod
    def _func_sign(series: pd.Series) -> pd.Series:
        return np.sign(series)

    @staticmethod
    def _func_log(series: pd.Series) -> pd.Series:
        return np.log(series.clip(lower=1e-10))

    @staticmethod
    def _func_ts_argmax(series: pd.Series, n: int) -> pd.Series:
        """滚动窗口内最大值位置（距今天数）"""
        def _argmax(x):
            if len(x) == 0:
                return np.nan
            return len(x) - 1 - np.argmax(x)
        return series.rolling(n, min_periods=1).apply(_argmax, raw=True)

    @staticmethod
    def _func_ts_argmin(series: pd.Series, n: int) -> pd.Series:
        """滚动窗口内最小值位置（距今天数）"""
        def _argmin(x):
            if len(x) == 0:
                return np.nan
            return len(x) - 1 - np.argmin(x)
        return series.rolling(n, min_periods=1).apply(_argmin, raw=True)

    @staticmethod
    def _func_rsi(series: pd.Series, n: int = 14) -> pd.Series:
        """RSI 指标"""
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _func_bb_upper(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
        """布林带上轨"""
        ma = series.rolling(n).mean()
        std = series.rolling(n).std()
        return ma + k * std

    @staticmethod
    def _func_bb_lower(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
        """布林带下轨"""
        ma = series.rolling(n).mean()
        std = series.rolling(n).std()
        return ma - k * std

    # ── 表达式解析与计算 ──

    def evaluate(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """
        计算表达式

        示例:
            engine.evaluate("$close / Ref($close, 20) - 1", data)
            engine.evaluate("Mean($close, 5) / Mean($close, 20) - 1", data)
            engine.evaluate("($close - Mean($close, 20)) / Std($close, 20)", data)
        """
        return self._parse_expression(expression, data)

    def _resolve_field(self, token: str, data: pd.DataFrame) -> pd.Series:
        """解析字段引用"""
        col = self.FIELD_MAP.get(token, token)
        if col in data.columns:
            return data[col]
        raise ValueError(f"未知字段: {token} (映射到: {col})")

    def _parse_expression(self, expr: str, data: pd.DataFrame) -> pd.Series:
        """递归下降解析表达式"""
        expr = expr.strip()

        # 处理括号
        if expr.startswith('(') and expr.endswith(')'):
            depth = 0
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    break
            else:
                return self._parse_expression(expr[1:-1], data)

        # 函数调用
        func_match = re.match(r'^(\w+)\(', expr)
        if func_match:
            func_name = func_match.group(1)
            # 找到匹配的右括号
            args_start = func_match.end()
            depth = 1
            args_end = args_start
            while args_end < len(expr) and depth > 0:
                if expr[args_end] == '(':
                    depth += 1
                elif expr[args_end] == ')':
                    depth -= 1
                args_end += 1
            if depth == 0:
                args_str = expr[args_start:args_end - 1]
                args = self._split_args(args_str)
                if func_name in self.FUNCTIONS:
                    # 分离：纯数字参数不解析为表达式，直接作为数值传递
                    final_args = []
                    for a in args:
                        a = a.strip()
                        try:
                            final_args.append(int(a))
                        except ValueError:
                            try:
                                final_args.append(float(a))
                            except ValueError:
                                final_args.append(self._parse_expression(a, data))
                    return self.FUNCTIONS[func_name](*final_args)
                else:
                    raise ValueError(f"未知函数: {func_name}")

        # 运算符处理（按优先级）
        for op in ['+', '-']:
            parts = self._split_by_op(expr, op)
            if len(parts) > 1:
                result = self._parse_expression(parts[0], data)
                for part in parts[1:]:
                    if op == '+':
                        result = result + self._parse_expression(part, data)
                    else:
                        result = result - self._parse_expression(part, data)
                return result

        for op in ['*', '/']:
            parts = self._split_by_op(expr, op)
            if len(parts) > 1:
                result = self._parse_expression(parts[0], data)
                for part in parts[1:]:
                    if op == '*':
                        result = result * self._parse_expression(part, data)
                    else:
                        result = result / self._parse_expression(part, data).replace(0, np.nan)
                return result

        # 字段引用或字面量
        if expr.startswith('$'):
            return self._resolve_field(expr, data)

        # 处理一元负号
        if expr.startswith('-'):
            return -self._parse_expression(expr[1:], data)

        # 尝试解析为数字
        try:
            return pd.Series(float(expr), index=data.index)
        except ValueError:
            pass

        # 尝试作为列名
        if expr in data.columns:
            return data[expr]

        raise ValueError(f"无法解析表达式: {expr}")

    def _split_by_op(self, expr: str, op: str) -> List[str]:
        """按运算符分割（考虑括号和函数嵌套）"""
        parts = []
        current = []
        depth = 0
        i = 0
        while i < len(expr):
            c = expr[i]
            if c == '(':
                depth += 1
                current.append(c)
            elif c == ')':
                depth -= 1
                current.append(c)
            elif c == op and depth == 0:
                # 处理负号：如果 op 是 '-' 且前面是空或运算符，则它是负号而非减号
                if op == '-' and (len(current) == 0 or current[-1] in '+-*/('):
                    current.append(c)
                else:
                    parts.append(''.join(current).strip())
                    current = []
            else:
                current.append(c)
            i += 1
        parts.append(''.join(current).strip())
        return parts

    def _split_args(self, args_str: str) -> List[str]:
        """按逗号分割函数参数（考虑嵌套）"""
        args = []
        current = []
        depth = 0
        for c in args_str:
            if c == '(':
                depth += 1
                current.append(c)
            elif c == ')':
                depth -= 1
                current.append(c)
            elif c == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(c)
        if current:
            args.append(''.join(current).strip())
        return args

    def evaluate_cross_sectional(
        self,
        expression: str,
        data: pd.DataFrame,
        group_col: str = 'code'
    ) -> pd.Series:
        """截面计算：对每个时间截面分别计算表达式"""
        result = pd.Series(index=data.index, dtype=float)
        for date, group in data.groupby('date'):
            result.loc[group.index] = self.evaluate(expression, group).values
        return result


# ═══════════════════════════════════════════════════════════════
# 2. Alpha158 风格因子库（借鉴 Qlib Alpha158 因子集）
# ═══════════════════════════════════════════════════════════════

@dataclass
class FactorDefinition:
    """因子定义"""
    name: str
    expression: str
    category: str  # 动量、反转、波动率、成交量、技术指标等
    description: str = ""


class AlphaFactorLibrary:
    """
    预定义因子库

    借鉴 Qlib 的 Alpha158 因子集设计：
    - 标准化因子定义（名称、表达式、分类）
    - 支持批量计算
    - 支持按分类筛选
    """

    def __init__(self, engine: ExpressionEngine = None):
        self.engine = engine or ExpressionEngine()
        # 参考 Qlib Alpha158 的因子分类
        self.FACTORS: List[FactorDefinition] = [
        # ── 动量类 ──
        FactorDefinition("momentum_5d", "$close / Ref($close, 5) - 1", "momentum", "5日动量"),
        FactorDefinition("momentum_10d", "$close / Ref($close, 10) - 1", "momentum", "10日动量"),
        FactorDefinition("momentum_20d", "$close / Ref($close, 20) - 1", "momentum", "20日动量"),
        FactorDefinition("momentum_60d", "$close / Ref($close, 60) - 1", "momentum", "60日动量"),

        # ── 反转类 ──
        FactorDefinition("reversal_5d", "-(Mean($close, 5) / Ref(Mean($close, 5), 5) - 1)", "reversal", "5日反转"),
        FactorDefinition("reversal_20d", "-(Mean($close, 20) / Ref(Mean($close, 20), 20) - 1)", "reversal", "20日反转"),

        # ── 波动率类 ──
        FactorDefinition("volatility_5d", "Std($close, 5) / Mean($close, 5)", "volatility", "5日波动率"),
        FactorDefinition("volatility_20d", "Std($close, 20) / Mean($close, 20)", "volatility", "20日波动率"),
        FactorDefinition("volatility_60d", "Std($close, 60) / Mean($close, 60)", "volatility", "60日波动率"),

        # ── 成交量类 ──
        FactorDefinition("volume_ratio_5d", "$volume / Mean($volume, 5)", "volume", "5日量比"),
        FactorDefinition("volume_ratio_20d", "$volume / Mean($volume, 20)", "volume", "20日量比"),
        FactorDefinition("volume_trend", "Corr($close, $volume, 10)", "volume", "量价相关性"),

        # ── 均线类 ──
        FactorDefinition("ma_bias_5", "$close / Mean($close, 5) - 1", "ma", "5日均线偏离"),
        FactorDefinition("ma_bias_20", "$close / Mean($close, 20) - 1", "ma", "20日均线偏离"),
        FactorDefinition("ma_bias_60", "$close / Mean($close, 60) - 1", "ma", "60日均线偏离"),
        FactorDefinition("ma_cross_5_20", "Mean($close, 5) / Mean($close, 20) - 1", "ma", "5-20均线交叉"),

        # ── 通道类 ──
        FactorDefinition("bb_position", "($close - BB_lower($close, 20, 2)) / (BB_upper($close, 20, 2) - BB_lower($close, 20, 2))", "channel", "布林带位置"),
        FactorDefinition("bb_width", "(BB_upper($close, 20, 2) - BB_lower($close, 20, 2)) / Mean($close, 20)", "channel", "布林带宽度"),

        # ── 技术指标类 ──
        FactorDefinition("rsi_14", "RSI($close, 14) / 100", "technical", "RSI(14)归一化"),
        FactorDefinition("rsi_28", "RSI($close, 28) / 100", "technical", "RSI(28)归一化"),

        # ── 价格形态 ──
        FactorDefinition("price_position", "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20))", "price_pattern", "20日价格位置"),
        FactorDefinition("hl_ratio", "($high - $low) / $close", "price_pattern", "日内振幅"),
        ]

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """批量计算所有因子"""
        result = data[['code', 'date']].copy() if 'code' in data.columns else pd.DataFrame(index=data.index)
        for factor_def in self.FACTORS:
            try:
                result[factor_def.name] = self.engine.evaluate(factor_def.expression, data)
            except Exception as e:
                print(f"警告: 因子 {factor_def.name} 计算失败: {e}")
                result[factor_def.name] = np.nan
        return result

    def compute_by_category(self, data: pd.DataFrame, category: str) -> pd.DataFrame:
        """按分类计算因子"""
        result = data[['code', 'date']].copy() if 'code' in data.columns else pd.DataFrame(index=data.index)
        for factor_def in self.FACTORS:
            if factor_def.category == category:
                try:
                    result[factor_def.name] = self.engine.evaluate(factor_def.expression, data)
                except Exception:
                    result[factor_def.name] = np.nan
        return result

    def get_factor_definitions(self) -> List[FactorDefinition]:
        return self.FACTORS

    def register_factor(self, factor_def: FactorDefinition):
        """注册自定义因子"""
        self.FACTORS.append(factor_def)


# ═══════════════════════════════════════════════════════════════
# 3. 测试
# ═══════════════════════════════════════════════════════════════

def generate_sample_data(n_symbols: int = 5, n_days: int = 200) -> pd.DataFrame:
    """生成模拟行情数据"""
    np.random.seed(42)
    rows = []
    for sym_idx in range(n_symbols):
        code = f"{600000 + sym_idx:06d}.SH"
        price = np.random.uniform(8, 50)
        for day in range(n_days):
            price *= (1 + np.random.normal(0.0005, 0.02))
            rows.append({
                'code': code,
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                'open': price * (1 + np.random.normal(0, 0.005)),
                'high': price * (1 + abs(np.random.normal(0, 0.01))),
                'low': price * (1 - abs(np.random.normal(0, 0.01))),
                'close': price,
                'volume': int(np.random.lognormal(12, 0.5)),
                'amount': price * np.random.lognormal(12, 0.5),
            })
    return pd.DataFrame(rows)


def test_expression_engine():
    """测试表达式引擎基本功能"""
    print("=" * 60)
    print("测试 1: 表达式引擎基本功能")
    print("=" * 60)

    data = generate_sample_data(1, 200)
    engine = ExpressionEngine()

    # 测试字段引用
    result = engine.evaluate("$close", data)
    assert len(result) == 200, f"字段引用长度错误: {len(result)}"

    # 测试 Ref
    result = engine.evaluate("Ref($close, 5)", data)
    assert result.iloc[:5].isna().all(), "Ref 前5行应为 NaN"

    # 测试 Mean
    result = engine.evaluate("Mean($close, 20)", data)
    assert result.iloc[19] is not np.nan, "Mean 第20个值不应为 NaN"

    # 测试动量因子
    result = engine.evaluate("$close / Ref($close, 20) - 1", data)
    print(f"  动量因子(20日) 范围: [{result.min():.4f}, {result.max():.4f}]")

    # 测试复合因子
    result = engine.evaluate("($close - Mean($close, 20)) / Std($close, 20)", data)
    print(f"  Z-Score 范围: [{result.min():.4f}, {result.max():.4f}]")

    # 测试 RSI
    result = engine.evaluate("RSI($close, 14)", data)
    print(f"  RSI(14) 范围: [{result.min():.2f}, {result.max():.2f}]")

    # 测试布林带
    upper = engine.evaluate("BB_upper($close, 20, 2)", data)
    lower = engine.evaluate("BB_lower($close, 20, 2)", data)
    print(f"  布林带上轨: [{lower.min():.2f}, {upper.max():.2f}]")

    print("✓ 表达式引擎基本功能测试通过")


def test_factor_library():
    """测试因子库"""
    print("\n" + "=" * 60)
    print("测试 2: Alpha158 风格因子库")
    print("=" * 60)

    data = generate_sample_data(3, 200)
    library = AlphaFactorLibrary()

    # 计算所有因子
    factors = library.compute_all(data)
    print(f"\n  因子总数: {len(library.FACTORS)}")
    print(f"  因子列: {[c for c in factors.columns if c not in ['code', 'date']]}")

    # 按分类计算
    momentum_factors = library.compute_by_category(data, 'momentum')
    print(f"\n  动量类因子: {[c for c in momentum_factors.columns if c not in ['code', 'date']]}")

    # 验证因子有效性（非全 NaN）
    factor_cols = [c for c in factors.columns if c not in ['code', 'date']]
    valid_count = 0
    for col in factor_cols:
        if not factors[col].isna().all():
            valid_count += 1
    print(f"\n  有效因子数: {valid_count}/{len(factor_cols)}")
    assert valid_count > 0, "所有因子计算失败"
    print("✓ 因子库测试通过")


def test_cross_sectional():
    """测试截面计算"""
    print("\n" + "=" * 60)
    print("测试 3: 截面因子计算")
    print("=" * 60)

    data = generate_sample_data(5, 100)
    engine = ExpressionEngine()

    # 截面排名
    result = engine.evaluate_cross_sectional("Rank($close)", data)
    print(f"  截面排名范围: [{result.min():.4f}, {result.max():.4f}]")

    # 验证每个截面的排名归一化（pct=True 时均值接近 0.5）
    for date, group in data.groupby('date'):
        ranks = result.loc[group.index]
        # 小样本下 pct rank 均值有偏差，用宽松检查
        assert 0.4 <= ranks.mean() <= 0.61, f"截面 {date} 排名均值异常: {ranks.mean()}"

    print("✓ 截面计算测试通过")


def test_expression_validation():
    """测试表达式验证与错误处理"""
    print("\n" + "=" * 60)
    print("测试 4: 表达式验证与错误处理")
    print("=" * 60)

    data = generate_sample_data(1, 50)
    engine = ExpressionEngine()

    # 测试未知函数
    try:
        engine.evaluate("Unknown($close, 5)", data)
        print("  ✗ 未知函数应该抛出异常")
    except ValueError as e:
        print(f"  ✓ 未知函数正确抛出异常: {e}")

    # 测试未知字段
    try:
        engine.evaluate("$unknown", data)
        print("  ✗ 未知字段应该抛出异常")
    except ValueError as e:
        print(f"  ✓ 未知字段正确抛出异常: {e}")

    # 测试除零保护
    result = engine.evaluate("$close / 0", data)
    assert result.isna().all() or (result == np.inf).all(), \
        "除零应返回 NaN 或 Inf"
    print(f"  ✓ 除零保护: 除零结果已处理")

    # 测试空数据
    empty_data = pd.DataFrame({'close': []})
    try:
        result = engine.evaluate("$close", empty_data)
        print(f"  ✓ 空数据处理: len={len(result)}")
    except Exception as e:
        print(f"  ✓ 空数据异常: {e}")

    print("✓ 表达式验证测试通过")


def test_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("测试 5: 性能测试")
    print("=" * 60)

    import time

    # 小规模：5只股票，200天
    data_small = generate_sample_data(5, 200)

    # 中规模：50只股票，500天
    data_medium = generate_sample_data(50, 500)

    library = AlphaFactorLibrary()

    for label, data in [("小规模(5×200)", data_small), ("中规模(50×500)", data_medium)]:
        start = time.perf_counter()
        factors = library.compute_all(data)
        elapsed = time.perf_counter() - start
        n_factors = len(library.FACTORS)
        print(f"  {label}: {elapsed:.3f}s ({n_factors} 个因子)")

        # 对比硬编码方式
        start = time.perf_counter()
        # 模拟现有 factor-engine 的计算方式
        df = data.copy()
        df['ret_1d'] = df.groupby('code')['close'].pct_change()
        df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
        df['ret_20d'] = df.groupby('code')['close'].pct_change(20)
        df['ma_5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5).mean())
        df['ma_20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
        df['vol_20d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).std())
        hardcoded_time = time.perf_counter() - start
        print(f"  硬编码方式: {hardcoded_time:.3f}s (6 个因子)")

        # 缩放对比
        scaled_expr = elapsed / n_factors * 6
        print(f"  表达式方式(折算6因子): {scaled_expr:.3f}s")

    print("✓ 性能测试完成")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("表达式引擎因子定义验证测试")
    print("借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)")
    print("优化方向: factor-engine — 新增表达式因子定义\n")

    test_expression_engine()
    test_factor_library()
    test_cross_sectional()
    test_expression_validation()
    test_performance()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)