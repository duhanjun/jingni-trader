"""
================================================================================
优化方向: 因子注册系统 (Factor Registry System)
借鉴来源: Microsoft Qlib (Alpha158 factor set), FactorHub (180+ factor catalog)
日期: 2026-06-12

核心思想:
- Qlib 的 Alpha158 提供了标准化的因子分类与注册机制，每个因子有明确的元信息
  (名称、方向、分类、参数等)，便于因子的系统化管理与自动化分析。
- FactorHub 提供了 180+ 因子的完整评估体系 (IC/IR、单调性检验、换手率分析)。
- 当前 jingni-trader 的 factor-engine 采用硬编码方式计算因子，缺乏统一的
  因子注册与元信息管理，新增因子需要修改核心引擎代码。

验证目标:
1. 验证因子注册表 (FactorRegistry) 的设计可行性
2. 验证基于注册表的因子批量计算、IC分析、单调性检验流程
3. 对比硬编码方式与注册表方式的扩展性差异
================================================================================
"""

import sys
import os
import json
import time
import unittest
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum

import numpy as np
import pandas as pd


# ============================================================================
# 因子注册系统核心设计 (借鉴 Qlib Alpha158 + FactorHub)
# ============================================================================

class FactorCategory(Enum):
    """因子分类体系 (借鉴 FactorHub 的 17 大分类)"""
    MOMENTUM = "momentum"           # 动量类
    REVERSAL = "reversal"            # 反转类
    VOLUME = "volume"                # 成交量类
    VOLATILITY = "volatility"        # 波动率类
    FUND_FLOW = "fund_flow"          # 资金流向类
    VALUATION = "valuation"          # 估值类
    QUALITY = "quality"              # 质量类
    GROWTH = "growth"                # 成长类
    TECHNICAL = "technical"          # 技术指标类
    CUSTOM = "custom"                # 自定义类


@dataclass
class FactorMeta:
    """
    因子元信息 (借鉴 Qlib 中每个因子的结构化描述)

    与 jingni-trader 现状对照:
    - 当前: 因子直接在 FactorEngine.compute_a_share_factors() 中硬编码计算
    - 改进: 每个因子有独立的元信息，支持自动化注册、发现、分析
    """
    name: str                              # 因子名称，如 "reversal_20d"
    display_name: str = ""                 # 显示名称，如 "20日反转"
    category: FactorCategory = FactorCategory.CUSTOM
    description: str = ""                  # 因子描述
    direction: int = 1                     # 因子方向: 1=正向, -1=反向, 0=未知
    params: Dict[str, Any] = field(default_factory=dict)  # 因子参数
    required_columns: List[str] = field(default_factory=list)  # 计算所需列
    version: str = "1.0"                   # 因子版本
    author: str = ""                       # 因子作者
    tags: List[str] = field(default_factory=list)  # 标签


@dataclass
class FactorICResult:
    """因子IC分析结果"""
    factor_name: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_positive_ratio: float
    ic_t_stat: float
    rank_ic_mean: float = 0.0
    monotonicity_score: float = 0.0        # 单调性得分 (借鉴 FactorHub)
    turnover_rate: float = 0.0             # 平均换手率 (借鉴 FactorHub)


class FactorRegistry:
    """
    因子注册表 (核心新增组件)

    借鉴 Qlib 的因子管理方式:
    - Qlib 通过 Expression Engine 支持 $close, Ref(), Mean() 等 DSL 表达式
    - 本设计采用 Python 函数式注册，更灵活且兼容现有代码
    - 提供 register() / unregister() / list_factors() / get_factor() API
    """

    def __init__(self):
        self._registry: Dict[str, FactorMeta] = {}
        self._calculators: Dict[str, Callable] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, meta: FactorMeta, calculator: Callable):
        """注册因子"""
        self._registry[meta.name] = meta
        self._calculators[meta.name] = calculator
        return self

    def unregister(self, name: str):
        """注销因子"""
        self._registry.pop(name, None)
        self._calculators.pop(name, None)

    def get_factor(self, name: str) -> Optional[FactorMeta]:
        return self._registry.get(name)

    def get_calculator(self, name: str) -> Optional[Callable]:
        return self._calculators.get(name)

    def list_factors(self, category: Optional[FactorCategory] = None) -> List[FactorMeta]:
        if category:
            return [m for m in self._registry.values() if m.category == category]
        return list(self._registry.values())

    def list_names(self, category: Optional[FactorCategory] = None) -> List[str]:
        return [m.name for m in self.list_factors(category)]

    def calculate(self, data: pd.DataFrame, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """批量计算因子 (与现有 FactorEngine 接口兼容)"""
        if factor_names is None:
            factor_names = list(self._calculators.keys())

        result = data[['code', 'date']].copy()
        for name in factor_names:
            calc = self._calculators.get(name)
            if calc is None:
                continue
            try:
                meta = self._registry[name]
                missing = [c for c in meta.required_columns if c not in data.columns]
                if missing:
                    print(f"  [WARN] 因子 {name} 缺少列: {missing}")
                    continue
                result[name] = calc(data)
            except Exception as e:
                print(f"  [ERROR] 因子 {name} 计算失败: {e}")
        return result

    def __len__(self):
        return len(self._registry)


# ============================================================================
# 预置因子计算函数 (对应 jingni-trader 现有因子，重构为可注册形式)
# ============================================================================

def _calc_ret_1d(data):
    return data.groupby('code')['close'].pct_change()

def _calc_ret_5d(data):
    return data.groupby('code')['close'].pct_change(5)

def _calc_ret_20d(data):
    return data.groupby('code')['close'].pct_change(20)

def _calc_ret_60d(data):
    return data.groupby('code')['close'].pct_change(60)

def _calc_reversal_5d(data):
    return -data.groupby('code')['close'].pct_change(5)

def _calc_reversal_20d(data):
    return -data.groupby('code')['close'].pct_change(20)

def _calc_lncap(data):
    if 'amount' in data.columns and 'turnover_rate' in data.columns:
        mv = data['amount'] / data['turnover_rate'].replace(0, np.nan) * 100
        return mv.replace(0, np.nan).apply(lambda x: np.log(x) if x > 0 else np.nan)
    return pd.Series(np.nan, index=data.index)

def _calc_turnover_20d(data):
    if 'turnover_rate' not in data.columns:
        return pd.Series(np.nan, index=data.index)
    return data.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

def _calc_turnover_5d(data):
    if 'turnover_rate' not in data.columns:
        return pd.Series(np.nan, index=data.index)
    return data.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )

def _calc_turnover_change(data):
    t5 = _calc_turnover_5d(data)
    t20 = _calc_turnover_20d(data)
    return t5 / t20.replace(0, np.nan) - 1

def _calc_volatility_20d(data):
    return data.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )

def _calc_volume_ratio(data):
    vol_20d = data.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    return data['volume'] / vol_20d.replace(0, np.nan)

def _calc_money_flow_20d(data):
    if 'change_pct' in data.columns:
        mf = data['change_pct'] * data.get('amount', data['volume'])
    else:
        mf = data.groupby('code')['close'].pct_change() * data.get('amount', data['volume'])
    return mf.groupby(data['code']).transform(lambda x: x.rolling(20, min_periods=5).sum())


def build_default_registry() -> FactorRegistry:
    """构建默认因子注册表 (映射 jingni-trader 现有因子)"""
    registry = FactorRegistry()

    factors = [
        (FactorMeta("ret_1d", "1日收益", FactorCategory.MOMENTUM,
                    "1日价格变化率", 1, {}, ['close']),
         _calc_ret_1d),
        (FactorMeta("ret_5d", "5日收益", FactorCategory.MOMENTUM,
                    "5日价格变化率", 1, {}, ['close']),
         _calc_ret_5d),
        (FactorMeta("ret_20d", "20日收益", FactorCategory.MOMENTUM,
                    "20日价格变化率", 1, {}, ['close']),
         _calc_ret_20d),
        (FactorMeta("ret_60d", "60日收益", FactorCategory.MOMENTUM,
                    "60日价格变化率", 1, {}, ['close']),
         _calc_ret_60d),
        (FactorMeta("reversal_5d", "5日反转", FactorCategory.REVERSAL,
                    "5日短期反转因子", -1, {}, ['close']),
         _calc_reversal_5d),
        (FactorMeta("reversal_20d", "20日反转", FactorCategory.REVERSAL,
                    "20日中期反转因子", -1, {}, ['close']),
         _calc_reversal_20d),
        (FactorMeta("lncap", "对数市值", FactorCategory.VALUATION,
                    "对数总市值", 0, {}, ['amount', 'turnover_rate']),
         _calc_lncap),
        (FactorMeta("turnover_20d", "20日均换手率", FactorCategory.VOLUME,
                    "20日平均换手率", 0, {}, ['turnover_rate']),
         _calc_turnover_20d),
        (FactorMeta("turnover_change", "换手率变化", FactorCategory.VOLUME,
                    "5日/20日换手率变化", 1, {}, ['turnover_rate']),
         _calc_turnover_change),
        (FactorMeta("volatility_20d", "20日波动率", FactorCategory.VOLATILITY,
                    "20日收益率标准差", -1, {}, ['close']),
         _calc_volatility_20d),
        (FactorMeta("volume_ratio", "量比", FactorCategory.VOLUME,
                    "当日成交量/20日均量", 1, {}, ['volume']),
         _calc_volume_ratio),
        (FactorMeta("money_flow_20d", "20日资金流", FactorCategory.FUND_FLOW,
                    "20日累计资金流", 1, {}, ['close', 'amount', 'volume']),
         _calc_money_flow_20d),
    ]

    for meta, calc in factors:
        registry.register(meta, calc)

    return registry


# ============================================================================
# IC分析 & 单调性检验 (借鉴 FactorHub 完整评估体系)
# ============================================================================

def calc_ic_series(factor_df: pd.DataFrame, forward_returns: pd.DataFrame,
                   factor_name: str, forward_col: str = 'ret_forward_5d',
                   method: str = 'spearman') -> pd.Series:
    """计算因子IC时间序列"""
    merged = factor_df[['code', 'date', factor_name]].merge(
        forward_returns[['code', 'date', forward_col]], on=['code', 'date'], how='inner'
    )
    ic_list = []
    for dt in sorted(merged['date'].unique()):
        cross = merged[merged['date'] == dt].dropna(subset=[factor_name, forward_col])
        if len(cross) < 10:
            continue
        if method == 'spearman':
            from scipy import stats
            ic, _ = stats.spearmanr(cross[factor_name], cross[forward_col], nan_policy='omit')
        else:
            ic = cross[factor_name].corr(cross[forward_col], method='pearson')
        if not np.isnan(ic):
            ic_list.append({'date': dt, 'ic': ic})

    if not ic_list:
        return pd.Series(dtype=float)
    ic_df = pd.DataFrame(ic_list)
    ic_df['date'] = pd.to_datetime(ic_df['date'])
    return ic_df.set_index('date')['ic']


def calc_monotonicity(factor_df: pd.DataFrame, forward_returns: pd.DataFrame,
                      factor_name: str, forward_col: str = 'ret_forward_5d',
                      n_groups: int = 5) -> float:
    """
    单调性检验 (借鉴 FactorHub 的分层单调性验证)

    将因子按截面排名分为 n_groups 组，计算每组的平均未来收益，
    若因子有效，则各组收益应呈单调变化。
    返回: 单调性得分 (0~1，越高越好)
    """
    merged = factor_df[['code', 'date', factor_name]].merge(
        forward_returns[['code', 'date', forward_col]], on=['code', 'date'], how='inner'
    )
    merged = merged.dropna(subset=[factor_name, forward_col])

    all_group_returns = []
    for dt in sorted(merged['date'].unique()):
        cross = merged[merged['date'] == dt]
        if len(cross) < n_groups * 5:
            continue
        cross['group'] = pd.qcut(cross[factor_name].rank(pct=True), n_groups,
                                  labels=False, duplicates='drop')
        group_ret = cross.groupby('group')[forward_col].mean()
        all_group_returns.append(group_ret)

    if not all_group_returns:
        return 0.0

    avg_group_returns = pd.concat(all_group_returns, axis=1).mean(axis=1)
    # 单调性: 相邻组收益差的方向一致性
    diffs = avg_group_returns.diff().dropna()
    if len(diffs) == 0:
        return 0.0
    consistency = (np.sign(diffs) == np.sign(diffs.iloc[0])).mean()
    # 结合首末组收益差 (top-bottom spread) 归一化
    spread = abs(avg_group_returns.iloc[-1] - avg_group_returns.iloc[0])
    return float(consistency * min(spread * 10, 1.0))


def calc_turnover(signals: pd.DataFrame, signal_col: str = 'signal') -> float:
    """
    换手率分析 (借鉴 FactorHub)

    计算相邻调仓期的持仓变化率，衡量策略换手频率。
    返回: 平均单边换手率
    """
    if signal_col not in signals.columns:
        return 0.0

    dates = sorted(signals['date'].unique())
    turnovers = []
    prev_long = set()

    for dt in dates:
        day = signals[signals['date'] == dt]
        long_codes = set(day[day[signal_col] > 0]['code'].tolist())
        if prev_long:
            turnover = len(long_codes.symmetric_difference(prev_long)) / \
                       max(len(prev_long.union(long_codes)), 1)
            turnovers.append(turnover / 2)  # 单边换手
        prev_long = long_codes

    return float(np.mean(turnovers)) if turnovers else 0.0


# ============================================================================
# 单元测试
# ============================================================================

class TestFactorRegistry(unittest.TestCase):
    """因子注册系统测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟A股数据 (含多只股票、多日)"""
        np.random.seed(42)
        codes = [f"{c:06d}.SZ" for c in range(1, 21)]  # 000001.SZ ~ 000020.SZ
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')
        rows = []
        for code in codes:
            n = len(dates)
            start_price = np.random.uniform(5, 50)
            # 几何布朗运动
            daily_returns = np.random.normal(0.0003, 0.018, n)
            prices = start_price * np.cumprod(1 + daily_returns)
            volumes = np.random.lognormal(10, 0.6, n).astype(int)
            turnover_rates = np.random.uniform(0.005, 0.05, n)

            df = pd.DataFrame({
                'date': dates,
                'code': code,
                'open': prices * (1 + np.random.normal(0, 0.003, n)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
                'close': prices,
                'volume': volumes,
                'amount': volumes * prices * np.random.uniform(0.9, 1.1),
                'turnover_rate': turnover_rates,
                'change_pct': np.insert(daily_returns[1:] * 100, 0, 0),
            })
            rows.append(df)

        cls.test_data = pd.concat(rows, ignore_index=True)

        # 生成前向收益
        cls.forward_returns = cls.test_data[['code', 'date']].copy()
        for period in [1, 5, 20]:
            cls.forward_returns[f'ret_forward_{period}d'] = \
                cls.test_data.groupby('code')['close'].transform(
                    lambda x: x.shift(-period) / x - 1
                )

        cls.registry = build_default_registry()

    def test_01_registry_initialization(self):
        """测试因子注册表初始化"""
        self.assertGreater(len(self.registry), 0)
        print(f"\n  [PASS] 注册表初始化成功，共 {len(self.registry)} 个因子")

    def test_02_list_by_category(self):
        """测试按分类列出因子"""
        momentum = self.registry.list_names(FactorCategory.MOMENTUM)
        reversal = self.registry.list_names(FactorCategory.REVERSAL)
        volume = self.registry.list_names(FactorCategory.VOLUME)

        print(f"  [INFO] 动量类因子: {momentum}")
        print(f"  [INFO] 反转类因子: {reversal}")
        print(f"  [INFO] 成交量类因子: {volume}")

        self.assertIn('ret_20d', momentum)
        self.assertIn('reversal_20d', reversal)
        self.assertIn('turnover_20d', volume)

    def test_03_factor_calculation(self):
        """测试因子批量计算"""
        factor_df = self.registry.calculate(self.test_data)
        factor_names = [c for c in factor_df.columns if c not in ['code', 'date']]
        self.assertGreater(len(factor_names), 0)
        print(f"\n  [PASS] 因子计算成功，共 {len(factor_names)} 个因子列")
        print(f"  [INFO] 因子列: {factor_names}")

    def test_04_ic_analysis(self):
        """测试IC分析"""
        factor_df = self.registry.calculate(self.test_data)
        for fname in ['reversal_20d', 'volatility_20d']:
            ic_series = calc_ic_series(factor_df, self.forward_returns, fname,
                                        forward_col='ret_forward_5d')
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            print(f"  [IC] {fname}: IC_mean={ic_mean:.4f}, IC_std={ic_std:.4f}, IC_IR={ic_ir:.4f}")
            self.assertFalse(np.isnan(ic_mean), f"{fname} IC 计算异常")

    def test_05_monotonicity(self):
        """测试单调性检验"""
        factor_df = self.registry.calculate(self.test_data)
        for fname in ['reversal_20d', 'volatility_20d']:
            score = calc_monotonicity(factor_df, self.forward_returns, fname)
            print(f"  [MONO] {fname}: monotonicity_score={score:.4f}")
            self.assertTrue(0 <= score <= 1, f"单调性得分应在 [0,1] 之间，实际={score}")

    def test_06_factor_meta(self):
        """测试因子元信息完整性"""
        for name in self.registry.list_names():
            meta = self.registry.get_factor(name)
            self.assertIsNotNone(meta, f"因子 {name} 元信息为空")
            self.assertIsNotNone(meta.category, f"因子 {name} 分类为空")
            self.assertIsNot(meta.description, "", f"因子 {name} 描述为空")
        print(f"\n  [PASS] 所有因子元信息完整")

    def test_07_registry_extensibility(self):
        """测试注册表扩展性: 新增因子无需修改核心代码"""
        # 新增一个自定义因子: 20日振幅
        def _calc_amplitude_20d(data):
            high_max = data.groupby('code')['high'].transform(
                lambda x: x.rolling(20, min_periods=5).max()
            )
            low_min = data.groupby('code')['low'].transform(
                lambda x: x.rolling(20, min_periods=5).min()
            )
            return (high_max - low_min) / data['close']

        meta = FactorMeta(
            name="amplitude_20d", display_name="20日振幅因子",
            category=FactorCategory.VOLATILITY,
            description="20日最高最低价振幅",
            required_columns=['high', 'low', 'close']
        )
        self.registry.register(meta, _calc_amplitude_20d)

        # 验证新因子可计算
        factor_df = self.registry.calculate(self.test_data, ['amplitude_20d'])
        self.assertIn('amplitude_20d', factor_df.columns)
        self.assertFalse(factor_df['amplitude_20d'].isna().all())

        # 验证可查询
        self.assertIn('amplitude_20d', self.registry.list_names(FactorCategory.VOLATILITY))
        print(f"\n  [PASS] 新因子 amplitude_20d 注册成功，无需修改核心引擎")

    def test_08_comparison_with_hardcoded(self):
        """对比: 注册表方式 vs 硬编码方式"""
        # 模拟硬编码方式 (当前 jingni-trader 模式)
        t0 = time.time()
        df = self.test_data.sort_values(['code', 'date']).copy()
        result_hc = df[['code', 'date']].copy()
        result_hc['ret_20d'] = df.groupby('code')['close'].pct_change(20)
        result_hc['reversal_20d'] = -result_hc['ret_20d']
        result_hc['volatility_20d'] = df.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )
        t_hardcoded = time.time() - t0

        # 注册表方式
        t0 = time.time()
        result_registry = self.registry.calculate(self.test_data,
                                                   ['ret_20d', 'reversal_20d', 'volatility_20d'])
        t_registry = time.time() - t0

        print(f"\n  [BENCH] 硬编码方式: {t_hardcoded*1000:.2f}ms")
        print(f"  [BENCH] 注册表方式: {t_registry*1000:.2f}ms")
        print(f"  [BENCH] 差异: {(t_registry - t_hardcoded)*1000:.2f}ms")

        # 注册表方式有额外开销(元信息查找 + 错误检查)，但换来了可扩展性
        # 实际使用中注册表开销在接受范围内 (主要是数据量大时的 pandas 计算本身)
        overhead_pct = (t_registry / t_hardcoded - 1) * 100 if t_hardcoded > 0 else 0
        print(f"  [BENCH] 注册表额外开销: {overhead_pct:.1f}%")
        # 注: 本次测试中硬编码仅计算了 3 个因子，注册表还进行了元信息检查
        # 在真实场景中因子数量多时，额外开销占比会显著降低
        self.assertLess(overhead_pct, 200.0, "注册表方式额外开销应在 200% 以内")


# ============================================================================
# 主运行入口
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("因子注册系统 (Factor Registry) 验证测试")
    print("借鉴来源: Microsoft Qlib Alpha158 + FactorHub")
    print("=" * 70)

    # 运行测试
    unittest.main(verbosity=2, argv=[''], exit=False)

    # 汇总
    registry = build_default_registry()
    print("\n" + "=" * 70)
    print("【验证结论】")
    print("-" * 70)
    print(f"  注册因子总数: {len(registry)}")
    print(f"  分类覆盖: {[c.value for c in set(m.category for m in registry.list_factors())]}")
    print(f"  新增因子: 无需修改核心引擎代码, 只需 register() 注册")
    print(f"  IC分析接口: 已支持 Spearman IC 和 Pearson IC")
    print(f"  单调性检验: 已实现分层单调性验证 (借鉴 FactorHub)")
    print(f"  换手率分析: 已实现 (借鉴 FactorHub)")
    print(f"  接口兼容: 与现有 FactorEngine 接口保持一致")
    print("=" * 70)