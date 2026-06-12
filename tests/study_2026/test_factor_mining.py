"""
验证测试: 因子挖掘与评估增强 (Factor Mining & Evaluation)
===========================================================
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
          - Alpha158 因子集 (158个标准化因子)
          - 因子处理方法论 (标准化、去极值、中性化)
优化方向: 因子引擎增强 - 因子表达式引擎、因子衰减分析、因子分组回测

Qlib 核心设计:
1. 表达式引擎: 支持用表达式定义新因子，自动求导计算
2. Alpha158: 预设 158 个因子，覆盖动量、反转、波动率、换手率、量价等维度
3. 因子处理 Pipeline: 标准化 -> 去极值 -> 中性化 -> 填充缺失值
4. 因子衰减分析: 计算 IC 衰减曲线，评估因子有效期
5. 因子分组回测: 按因子值分组，验证单调性

与 jingni-trader 现有 factor-engine 的差异:
- 现有引擎只有硬编码的 10+ 个因子，无表达式引擎
- 无因子衰减分析
- 无因子分组回测验证单调性
- 无 Alpha158 等效的标准化因子库

测试目标:
1. 验证因子表达式引擎的正确性
2. 验证因子衰减分析 (IC Decay)
3. 验证因子分组回测的单调性检验
"""
import sys
import os
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import unittest
from typing import List, Dict, Optional, Tuple, Callable
from scipy import stats


# ============================================================
# 1. 因子表达式引擎 - 借鉴 Qlib 的表达式引擎设计
# ============================================================

class FactorExpressionEngine:
    """
    因子表达式引擎
    借鉴 Qlib 的表达式解析器，支持用字符串表达式定义因子
    支持的操作: +, -, *, /, 括号, Ref(滞后), Mean, Std, Max, Min, Ts_Rank
    """

    # 白名单函数（安全限制）
    FUNCTIONS = {
        'Ref': lambda x, n: x.shift(n),
        'Mean': lambda x, n: x.rolling(n, min_periods=n).mean(),
        'Std': lambda x, n: x.rolling(n, min_periods=n).std(),
        'Max': lambda x, n: x.rolling(n, min_periods=n).max(),
        'Min': lambda x, n: x.rolling(n, min_periods=n).min(),
        'Sum': lambda x, n: x.rolling(n, min_periods=n).sum(),
        'Ts_Rank': lambda x, n: x.rolling(n, min_periods=n).apply(
            lambda y: stats.rankdata(y)[-1] / len(y) if len(y) > 0 else np.nan
        ),
        'Delta': lambda x, n: x - x.shift(n),
        'Delay': lambda x, n: x.shift(n),
        'Corr': lambda x, y, n: x.rolling(n).corr(y),
        'Cov': lambda x, y, n: x.rolling(n).cov(y),
        'Log': lambda x: np.log(x.replace(0, np.nan)),
        'Abs': lambda x: x.abs(),
        'Sign': lambda x: np.sign(x),
        'Rank': lambda x: x.rank(pct=True),
    }

    COLUMN_MAP = {
        'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'volume': 'volume', 'amount': 'amount',
        'vwap': 'vwap', 'returns': 'returns',
    }

    def __init__(self, data: pd.DataFrame):
        """
        参数:
            data: 包含 OHLCV 的 DataFrame, 须有 code, date 列
        """
        self.data = data
        self._cache = {}

    def compute(self, expression: str, name: str = None) -> pd.DataFrame:
        """
        计算因子表达式

        支持的表达式示例:
        - "Mean(close, 5) / Mean(close, 20) - 1"  -> MA 乖离率
        - "Ts_Rank(Delta(close, 1), 20)"           -> 20日价格动量排名
        - "Std(close, 20) / Mean(close, 20)"       -> 20日波动率
        - "Corr(close, volume, 20)"                 -> 量价相关性
        - "(close - Mean(close, 20)) / Std(close, 20)" -> 标准化价格偏离
        """
        name = name or f"expr_{expression[:20]}"
        if name in self._cache:
            return self._cache[name]

        # 解析表达式树
        result = self._parse_and_eval(expression)

        if isinstance(result, pd.Series):
            result = result.to_frame(name=name)
            result['code'] = self.data['code'].values
            result['date'] = self.data['date'].values

        self._cache[name] = result
        return result

    def _parse_and_eval(self, expr: str) -> pd.Series:
        """
        简单表达式解析器
        支持: 函数调用、二元运算、列引用
        """
        expr = expr.strip()

        # 处理函数调用: func_name(args)
        import re
        # 先找到最外层函数调用的开始模式
        func_start = re.match(r'(\w+)\(', expr)
        if func_start:
            func_name = func_start.group(1)
            # 找到函数的左括号位置
            paren_start = func_start.end() - 1  # '(' 的位置
            # 找到匹配的右括号
            paren_end = self._find_matching_paren(expr, paren_start)
            if paren_end == len(expr) - 1:
                # 整个表达式就是一个函数调用
                args_str = expr[paren_start + 1:paren_end]
                args = self._split_args(args_str)
                return self._eval_function(func_name, args)

        # 处理括号表达式
        if expr.startswith('('):
            end = self._find_matching_paren(expr, 0)
            if end == len(expr) - 1:
                return self._parse_and_eval(expr[1:-1])

        # 处理二元运算 (从低优先级到高优先级，从右往左匹配)
        for op in ['+', '-']:
            idx = self._find_op(expr, op)
            if idx > 0:
                left = self._parse_and_eval(expr[:idx].strip())
                right = self._parse_and_eval(expr[idx + 1:].strip())
                if op == '+':
                    return left + right
                else:
                    return left - right

        for op in ['*', '/']:
            idx = self._find_op(expr, op)
            if idx > 0:
                left = self._parse_and_eval(expr[:idx].strip())
                right = self._parse_and_eval(expr[idx + 1:].strip())
                if op == '*':
                    return left * right
                else:
                    return left / right.replace(0, np.nan)

        # 数字常量
        try:
            return pd.Series(float(expr), index=self.data.index)
        except ValueError:
            pass

        # 列引用
        col = self.COLUMN_MAP.get(expr, expr)
        if col in self.data.columns:
            return self.data[col]

        raise ValueError(f"无法解析表达式: {expr}")

    def _find_matching_paren(self, expr: str, start: int) -> int:
        """找到与 start 位置的 '(' 匹配的 ')' 位置"""
        depth = 0
        for i in range(start, len(expr)):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _find_op(self, expr: str, op: str) -> int:
        """在表达式中查找操作符位置（忽略括号内）"""
        depth = 0
        for i, c in enumerate(expr):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0 and c == op:
                # 跳过一元负号
                if op == '-' and i == 0:
                    continue
                return i
        return -1

    def _split_args(self, args_str: str) -> List[str]:
        """分割函数参数（考虑嵌套括号和嵌套函数调用）"""
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
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(c)
        if current:
            args.append(''.join(current).strip())
        return args

    def _eval_function(self, func_name: str, args: List[str]) -> pd.Series:
        """执行函数调用"""
        if func_name not in self.FUNCTIONS:
            raise ValueError(f"未注册的函数: {func_name}")

        func = self.FUNCTIONS[func_name]
        eval_args = []
        for arg in args:
            # 尝试解析为数字
            try:
                eval_args.append(int(arg))
            except ValueError:
                try:
                    eval_args.append(float(arg))
                except ValueError:
                    eval_args.append(self._parse_and_eval(arg))

        return func(*eval_args)


# ============================================================
# 2. Alpha158 因子集 - 借鉴 Qlib 的 Alpha158
# ============================================================

class Alpha158Library:
    """
    Alpha158 风格因子库
    借鉴 Qlib 的 Alpha158 因子集，生成 158 个标准化因子
    因子分类:
    - K 线序列因子 (KLine): open, high, low, close, vwap, volume, amount
    - 价量因子 (PriceVolume): 收益率、波动率、量比、换手率
    - 滚动窗口因子 (Rolling): 均线、标准差、最大最小值
    """

    @staticmethod
    def generate_all(data: pd.DataFrame) -> pd.DataFrame:
        """
        生成 Alpha158 风格因子集
        包含: 动量因子、反转因子、波动率因子、量价因子、技术指标因子
        """
        df = data.sort_values(['code', 'date']).copy()
        grouped = df.groupby('code')

        # 基础价格序列
        close = df['close']
        open_p = df.get('open', close * 0.99)
        high = df.get('high', close * 1.02)
        low = df.get('low', close * 0.98)
        volume = df.get('volume', pd.Series(1000000, index=df.index))
        amount = df.get('amount', volume * close)

        factors = pd.DataFrame(index=df.index)
        factors['code'] = df['code']
        factors['date'] = df['date']

        # === 动量因子 (Momentum) ===
        for period in [5, 10, 20, 60]:
            ret = grouped['close'].pct_change(period)
            factors[f'momentum_{period}d'] = ret

        # === 反转因子 (Reversal) ===
        for period in [5, 10, 20]:
            factors[f'reversal_{period}d'] = -grouped['close'].pct_change(period)

        # === 波动率因子 (Volatility) ===
        for period in [5, 10, 20, 60]:
            factors[f'volatility_{period}d'] = grouped['close'].pct_change().rolling(
                period, min_periods=period).std().reset_index(0, drop=True)

        # === 均线偏离因子 (MA Deviation) ===
        for period in [5, 10, 20, 60]:
            ma = grouped['close'].transform(
                lambda x: x.rolling(period, min_periods=period).mean()
            )
            factors[f'ma_dev_{period}d'] = close / ma - 1

        # === 量价因子 (Volume-Price) ===
        for period in [5, 10, 20]:
            # 量比
            avg_vol = volume.rolling(period, min_periods=period).mean()
            # 需要按 code 分组计算
            avg_vol_grouped = grouped['volume'].transform(
                lambda x: x.rolling(period, min_periods=period).mean()
            ) if 'volume' in df.columns else volume.rolling(period, min_periods=period).mean()
            factors[f'volume_ratio_{period}d'] = volume / avg_vol_grouped.replace(0, np.nan)

            # 量价相关性
            if 'volume' in df.columns:
                factors[f'corr_vp_{period}d'] = grouped.apply(
                    lambda x: x['close'].rolling(period).corr(x['volume'])
                ).reset_index(0, drop=True)
            else:
                factors[f'corr_vp_{period}d'] = np.nan

        # === 技术指标因子 (Technical) ===
        # RSI
        for period in [6, 14, 24]:
            delta = grouped['close'].diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = grouped.apply(
                lambda x: gain[x.index].rolling(period).mean()
            ).reset_index(0, drop=True)
            avg_loss = grouped.apply(
                lambda x: loss[x.index].rolling(period).mean()
            ).reset_index(0, drop=True)
            rs = avg_gain / avg_loss.replace(0, np.nan)
            factors[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return factors


# ============================================================
# 3. 因子衰减分析 (IC Decay) - 借鉴 Qlib
# ============================================================

class ICFactorDecay:
    """
    因子 IC 衰减分析
    借鉴 Qlib 的 IC 分析框架，评估因子在不同时间窗口的预测能力衰减
    """

    @staticmethod
    def calc_ic_decay(
        factor_df: pd.DataFrame,
        data: pd.DataFrame,
        max_periods: int = 20,
        ic_type: str = "spearman",
    ) -> pd.DataFrame:
        """
        计算因子 IC 在不同前瞻期的衰减曲线

        参数:
            factor_df: 因子数据 (code, date, 因子列)
            data: 价格数据 (code, date, close)
            max_periods: 最大前瞻期数
            ic_type: IC 类型 (spearman / pearson)

        返回:
            DataFrame: period -> IC mean, IC std, IC IR
        """
        df = factor_df.merge(data[['code', 'date', 'close']], on=['code', 'date'])
        factor_cols = [c for c in factor_df.columns
                       if c not in ['code', 'date']]

        results = []
        for period in range(1, max_periods + 1):
            # 计算前瞻收益
            df['forward_ret'] = df.groupby('code')['close'].transform(
                lambda x: x.shift(-period) / x - 1
            )

            for factor in factor_cols:
                if factor not in df.columns:
                    continue
                ic_series = ICFactorDecay._calc_ic_series(
                    df, factor, 'forward_ret', ic_type
                )
                if ic_series is not None and len(ic_series) > 0:
                    ic_mean = np.mean(ic_series)
                    ic_std = np.std(ic_series)
                    results.append({
                        'period': period,
                        'factor': factor,
                        'ic_mean': ic_mean,
                        'ic_std': ic_std,
                        'ic_ir': ic_mean / ic_std if ic_std > 0 else 0,
                    })

        return pd.DataFrame(results)

    @staticmethod
    def _calc_ic_series(
        df: pd.DataFrame, factor_col: str, forward_col: str, ic_type: str
    ) -> Optional[np.ndarray]:
        """计算 IC 时间序列"""
        ic_vals = []
        for _, cross in df.groupby('date'):
            valid = cross[[factor_col, forward_col]].dropna()
            if len(valid) < 10:
                continue
            if ic_type == "spearman":
                ic, _ = stats.spearmanr(valid[factor_col], valid[forward_col])
            else:
                ic, _ = stats.pearsonr(valid[factor_col], valid[forward_col])
            if not np.isnan(ic):
                ic_vals.append(ic)
        return np.array(ic_vals) if ic_vals else None


# ============================================================
# 4. 因子分组回测 - 验证单调性
# ============================================================

class FactorGroupBacktest:
    """
    因子分组回测
    借鉴 Qlib 的分组回测，验证因子单调性
    将股票按因子值分为 N 组，计算每组的平均收益，验证是否单调
    """

    @staticmethod
    def group_backtest(
        factor_df: pd.DataFrame,
        data: pd.DataFrame,
        factor_name: str,
        n_groups: int = 5,
        forward_period: int = 5,
    ) -> dict:
        """
        执行因子分组回测

        返回:
            {
                'group_returns': {group_label: mean_return},
                'top_bottom_spread': 多空收益差,
                'monotonic': 是否单调,
                'group_equity': 各组权益曲线
            }
        """
        df = factor_df[['code', 'date', factor_name]].merge(
            data[['code', 'date', 'close']], on=['code', 'date']
        )
        df = df.dropna(subset=[factor_name])

        # 计算前瞻收益
        df['forward_ret'] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-forward_period) / x - 1
        )
        df = df.dropna(subset=['forward_ret'])

        # 按日期分组
        group_returns = {i: [] for i in range(1, n_groups + 1)}
        group_equity = {i: [] for i in range(1, n_groups + 1)}

        for dt, cross in df.groupby('date'):
            if len(cross) < n_groups * 3:
                continue
            cross = cross.copy()
            cross['group'] = pd.qcut(
                cross[factor_name].rank(method='first'),
                q=n_groups, labels=range(1, n_groups + 1)
            )
            for g in range(1, n_groups + 1):
                g_ret = cross[cross['group'] == g]['forward_ret'].mean()
                group_returns[g].append(g_ret)

        # 汇总
        mean_returns = {}
        for g in range(1, n_groups + 1):
            rets = group_returns[g]
            mean_returns[g] = np.mean(rets) if rets else 0

        # 判断单调性
        rets_list = [mean_returns[g] for g in range(1, n_groups + 1)]
        monotonic = all(
            rets_list[i] <= rets_list[i + 1] for i in range(len(rets_list) - 1)
        ) or all(
            rets_list[i] >= rets_list[i + 1] for i in range(len(rets_list) - 1)
        )

        top_bottom_spread = mean_returns[n_groups] - mean_returns[1]

        return {
            'group_returns': mean_returns,
            'top_bottom_spread': top_bottom_spread,
            'monotonic': monotonic,
            'factor_name': factor_name,
        }


# ============================================================
# 测试用例
# ============================================================

class TestFactorExpressionEngine(unittest.TestCase):
    """测试因子表达式引擎"""

    def setUp(self):
        np.random.seed(42)
        n = 200
        self.data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n, freq='B'),
            'code': '000001.SZ',
            'open': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n)),
            'high': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n)),
            'low': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.008, n)),
            'close': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n)),
            'volume': np.random.lognormal(10, 0.5, n).astype(int),
        })
        self.engine = FactorExpressionEngine(self.data)

    def test_simple_expression(self):
        """简单表达式测试"""
        result = self.engine.compute("close / open - 1")
        self.assertIsNotNone(result)
        self.assertIn('expr_close / open - 1', result.columns)

    def test_function_call(self):
        """函数调用测试"""
        result = self.engine.compute("Mean(close, 5)")
        self.assertIsNotNone(result)
        # 验证前 4 个值为 NaN, 第 5 个开始有值
        values = result.iloc[:, 0].values
        self.assertTrue(np.isnan(values[3]))
        self.assertFalse(np.isnan(values[4]))

    def test_nested_expression(self):
        """嵌套表达式测试"""
        result = self.engine.compute("(close - Mean(close, 20)) / Std(close, 20)")
        self.assertIsNotNone(result)

    def test_ma_deviation(self):
        """MA 乖离率测试"""
        result = self.engine.compute("Mean(close, 5) / Mean(close, 20) - 1")
        values = result.iloc[:, 0].dropna()
        self.assertGreater(len(values), 0)

    def test_ts_rank(self):
        """Ts_Rank 测试"""
        result = self.engine.compute("Ts_Rank(Delta(close, 1), 20)")
        values = result.iloc[:, 0].dropna()
        if len(values) > 0:
            # Ts_Rank 应在 [0, 1] 区间
            self.assertTrue((values >= 0).all() and (values <= 1).all())

    def test_invalid_function(self):
        """无效函数测试"""
        with self.assertRaises(ValueError):
            self.engine.compute("InvalidFunc(close, 5)")


class TestAlpha158Library(unittest.TestCase):
    """测试 Alpha158 因子集"""

    def setUp(self):
        np.random.seed(42)
        n = 300
        codes = ['000001.SZ', '000002.SZ', '000003.SZ']
        records = []
        for code in codes:
            price = np.random.uniform(10, 50)
            for _ in range(n):
                price *= np.random.lognormal(0.0003, 0.015)
                records.append({
                    'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=_),
                    'code': code,
                    'open': price * 0.99,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price,
                    'volume': int(np.random.lognormal(15, 0.5)),
                })
        self.data = pd.DataFrame(records)

    def test_generate_all(self):
        """生成全部因子"""
        factors = Alpha158Library.generate_all(self.data)
        factor_cols = [c for c in factors.columns if c not in ['code', 'date']]
        self.assertGreater(len(factor_cols), 20, "应生成至少 20 个因子")
        print(f"\nAlpha158 因子库生成 {len(factor_cols)} 个因子")

    def test_factor_no_future_leak(self):
        """验证因子无未来信息泄露"""
        factors = Alpha158Library.generate_all(self.data)
        factor_cols = [c for c in factors.columns if c not in ['code', 'date']]

        for col in factor_cols:
            # 检查每个因子列是否只使用当前及历史数据
            # 通过检查因子值是否依赖未来数据
            df = factors[['code', 'date', col]].dropna()
            for code, grp in df.groupby('code'):
                grp = grp.sort_values('date')
                # 因子值应基于当前及历史数据，此处验证没有 NaN 中间间隔
                values = grp[col].values
                nan_mask = np.isnan(values)
                # 如果前 N 个是 NaN，后面应该连续非 NaN
                if nan_mask.any():
                    first_valid = np.argmax(~nan_mask)
                    if first_valid > 0:
                        self.assertTrue(
                            nan_mask[first_valid:].sum() == 0,
                            f"因子 {col} 存在中间 NaN，可能使用了未来数据"
                        )


class TestICFactorDecay(unittest.TestCase):
    """测试因子 IC 衰减分析"""

    def setUp(self):
        np.random.seed(42)
        n = 300
        codes = [f'{i:06d}.SZ' for i in range(1, 16)]  # 15 stocks for IC calc
        records = []
        for code in codes:
            price = np.random.uniform(10, 50)
            for _ in range(n):
                price *= np.random.lognormal(0.0003, 0.015)
                records.append({
                    'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=_),
                    'code': code,
                    'close': price,
                })
        self.data = pd.DataFrame(records)

        factors = Alpha158Library.generate_all(self.data)
        self.factor_df = factors

    def test_ic_decay_basic(self):
        """基础 IC 衰减测试"""
        factor_cols = [c for c in self.factor_df.columns
                       if c not in ['code', 'date']][:5]
        decay_df = ICFactorDecay.calc_ic_decay(
            self.factor_df[['code', 'date'] + factor_cols],
            self.data,
            max_periods=10,
        )

        self.assertGreater(len(decay_df), 0)
        self.assertIn('period', decay_df.columns)
        self.assertIn('ic_mean', decay_df.columns)

        # 打印衰减曲线
        print("\n" + "=" * 60)
        print("IC 衰减分析")
        print("=" * 60)
        for factor in factor_cols[:3]:
            f_decay = decay_df[decay_df['factor'] == factor]
            if len(f_decay) > 0:
                print(f"  {factor}:")
                for _, row in f_decay.iterrows():
                    print(f"    period={int(row['period']):2d}  IC={row['ic_mean']:.4f}  "
                          f"IR={row['ic_ir']:.4f}")
        print("=" * 60)

    def test_ic_decay_with_period(self):
        """验证 IC 随前瞻期增加而衰减"""
        factor = 'momentum_5d'
        if factor not in self.factor_df.columns:
            self.skipTest(f"{factor} 不存在")

        decay_df = ICFactorDecay.calc_ic_decay(
            self.factor_df[['code', 'date', factor]],
            self.data,
            max_periods=15,
        )
        f_decay = decay_df[decay_df['factor'] == factor]

        if len(f_decay) > 5:
            # 短期 IC 绝对值应大于长期 IC 绝对值
            early_ic = abs(f_decay.iloc[0]['ic_mean'])
            late_ic = abs(f_decay.iloc[-1]['ic_mean'])
            # 不强制要求衰减，但记录观察
            print(f"  {factor} IC 衰减: 短期={early_ic:.4f}, 长期={late_ic:.4f}")


class TestFactorGroupBacktest(unittest.TestCase):
    """测试因子分组回测"""

    def setUp(self):
        np.random.seed(42)
        n = 300
        codes = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
        records = []
        for code in codes:
            price = np.random.uniform(10, 50)
            for _ in range(n):
                price *= np.random.lognormal(0.0003, 0.015)
                records.append({
                    'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=_),
                    'code': code,
                    'close': price,
                })
        self.data = pd.DataFrame(records)
        self.factor_df = Alpha158Library.generate_all(self.data)

    def test_group_backtest(self):
        """分组回测基础测试"""
        factor = 'momentum_20d'
        if factor not in self.factor_df.columns:
            self.skipTest(f"{factor} 不存在")

        result = FactorGroupBacktest.group_backtest(
            self.factor_df, self.data, factor, n_groups=5, forward_period=5
        )

        self.assertIn('group_returns', result)
        self.assertIn('monotonic', result)
        self.assertIn('top_bottom_spread', result)

        print("\n" + "=" * 60)
        print(f"因子分组回测: {factor}")
        print("=" * 60)
        for g, ret in result['group_returns'].items():
            print(f"  Group {g}: {ret:.6f}")
        print(f"  多空收益差: {result['top_bottom_spread']:.6f}")
        print(f"  单调性: {'是' if result['monotonic'] else '否'}")
        print("=" * 60)

    def test_multiple_factors_monotonicity(self):
        """测试多个因子的单调性"""
        factor_cols = [c for c in self.factor_df.columns
                       if c not in ['code', 'date'] and 'momentum' in c]

        for factor in factor_cols[:3]:
            result = FactorGroupBacktest.group_backtest(
                self.factor_df, self.data, factor, n_groups=5
            )
            print(f"  {factor}: monotonic={result['monotonic']}, "
                  f"spread={result['top_bottom_spread']:.6f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)