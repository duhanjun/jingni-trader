"""
=============================================================================
优化方向: 标准化因子库扩展 (Standardized Factor Library)
借鉴来源: Microsoft Qlib Alpha158/Alpha360, FactorEngine (论文), quant-stream
日期: 2026-06-13
=============================================================================

核心思路:
  Qlib 的 Alpha158 提供了 158 个经过验证的标准化因子，分为趋势、反转、
  成交量、波动率、资金流向、复合等 6 大类。junt-trader 当前仅实现约
  10-15 个基础因子（ret_1d/5d/20d/60d, reversal_5d/20d, 成交量等）。
  扩展因子库可显著提升策略研发质量和效率。

验证目标:
  1. 实现分类清晰、可扩展的因子库框架（BaseFactor 抽象类）
  2. 实现 6 大类共 30+ 个示例因子
  3. 验证 IC 分析正确性
  4. 验证因子去冗余（相关性分析）的有效性
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy import stats


# =============================================================================
# 因子库核心框架
# =============================================================================

@dataclass
class FactorMetadata:
    """因子元数据"""
    name: str
    category: str  # trend / reversal / volume / volatility / money_flow / composite
    description: str
    lookback_period: int = 20
    neutralize_recommended: bool = True


class BaseFactor(ABC):
    """因子基类"""

    def __init__(self, metadata: FactorMetadata):
        self.metadata = metadata

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值，返回与输入数据对齐的 Series"""
        pass

    def __repr__(self):
        return f"Factor({self.metadata.name}, {self.metadata.category})"


# ---- 趋势跟踪因子 ----

class MomentumFactor(BaseFactor):
    """动量因子: N日收益率"""
    def __init__(self, period: int = 20):
        super().__init__(FactorMetadata(
            name=f"momentum_{period}d",
            category="trend",
            description=f"{period}日动量因子",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = data.groupby('code')['close'].transform(
            lambda x: x.pct_change(self.period)
        )
        return result


class MACDFactor(BaseFactor):
    """MACD 因子"""
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(FactorMetadata(
            name=f"macd_{fast}_{slow}_{signal}",
            category="trend",
            description=f"MACD ({fast},{slow},{signal})",
            lookback_period=slow + signal,
        ))
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            close = data.loc[mask, 'close']
            ema_fast = close.ewm(span=self.fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
            result[mask] = (macd_line - signal_line).values
        return result


class RSIFactor(BaseFactor):
    """RSI 因子"""
    def __init__(self, period: int = 14):
        super().__init__(FactorMetadata(
            name=f"rsi_{period}",
            category="trend",
            description=f"{period}日RSI",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            close = data.loc[mask, 'close']
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(self.period, min_periods=self.period).mean()
            avg_loss = loss.rolling(self.period, min_periods=self.period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            result[mask] = rsi.values
        return result


class PricePositionFactor(BaseFactor):
    """价格位置因子: (close - low_N) / (high_N - low_N)"""
    def __init__(self, period: int = 20):
        super().__init__(FactorMetadata(
            name=f"price_position_{period}d",
            category="trend",
            description=f"{period}日价格位置",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            close = data.loc[mask, 'close']
            low_n = close.rolling(self.period).min()
            high_n = close.rolling(self.period).max()
            result[mask] = ((close - low_n) / (high_n - low_n).replace(0, np.nan)).values
        return result


# ---- 反转因子 ----

class ReversalFactor(BaseFactor):
    """反转因子"""
    def __init__(self, period: int = 5):
        super().__init__(FactorMetadata(
            name=f"reversal_{period}d",
            category="reversal",
            description=f"{period}日反转因子",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        t1_ret = data.groupby('code')['close'].transform(lambda x: x.pct_change(1))
        tN_ret = data.groupby('code')['close'].transform(lambda x: x.pct_change(self.period))
        # 反转 = 短期收益减去（已实现的）N日收益
        return t1_ret - tN_ret / self.period


class GapFactor(BaseFactor):
    """跳空因子: (open - prev_close) / prev_close"""
    def __init__(self):
        super().__init__(FactorMetadata(
            name="overnight_gap",
            category="reversal",
            description="隔夜跳空因子",
            lookback_period=1,
        ))

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            open_ = data.loc[mask, 'open']
            close = data.loc[mask, 'close']
            prev_close = close.shift(1)
            result[mask] = ((open_ - prev_close) / prev_close).values
        return result


# ---- 成交量因子 ----

class VolumeRatioFactor(BaseFactor):
    """量比因子"""
    def __init__(self, short_period: int = 5, long_period: int = 20):
        super().__init__(FactorMetadata(
            name=f"volume_ratio_{short_period}_{long_period}",
            category="volume",
            description=f"{short_period}日均量/{long_period}日均量",
            lookback_period=long_period,
        ))
        self.short_period = short_period
        self.long_period = long_period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            vol = data.loc[mask, 'volume']
            short_ma = vol.rolling(self.short_period).mean()
            long_ma = vol.rolling(self.long_period).mean()
            result[mask] = (short_ma / long_ma.replace(0, np.nan)).values
        return result


class TurnoverFactor(BaseFactor):
    """换手率因子"""
    def __init__(self, period: int = 20):
        super().__init__(FactorMetadata(
            name=f"turnover_{period}d",
            category="volume",
            description=f"{period}日平均换手率",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if 'turnover_rate' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        result = data.groupby('code')['turnover_rate'].transform(
            lambda x: x.rolling(self.period, min_periods=5).mean()
        )
        return result


class VolumeTrendFactor(BaseFactor):
    """成交量趋势: 成交量N日变化率"""
    def __init__(self, period: int = 20):
        super().__init__(FactorMetadata(
            name=f"volume_trend_{period}d",
            category="volume",
            description=f"{period}日成交量趋势",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = data.groupby('code')['volume'].transform(
            lambda x: x.pct_change(self.period)
        )
        return result


# ---- 波动率因子 ----

class VolatilityFactor(BaseFactor):
    """波动率因子"""
    def __init__(self, period: int = 20):
        super().__init__(FactorMetadata(
            name=f"volatility_{period}d",
            category="volatility",
            description=f"{period}日波动率",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = data.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(self.period, min_periods=10).std()
        )
        return result


class MaxDrawdownFactor(BaseFactor):
    """最大回撤因子 (滚动窗口)"""
    def __init__(self, period: int = 60):
        super().__init__(FactorMetadata(
            name=f"max_drawdown_{period}d",
            category="volatility",
            description=f"{period}日滚动最大回撤",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            close = data.loc[mask, 'close']
            rolling_max = close.rolling(self.period, min_periods=10).max()
            dd = close / rolling_max - 1
            result[mask] = dd.values
        return result


class AmplitudeFactor(BaseFactor):
    """振幅因子"""
    def __init__(self, period: int = 20):
        super().__init__(FactorMetadata(
            name=f"amplitude_{period}d",
            category="volatility",
            description=f"{period}日平均振幅",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            high = data.loc[mask, 'high']
            low = data.loc[mask, 'low']
            open_ = data.loc[mask, 'open']
            amp = (high - low) / open_
            result[mask] = amp.rolling(self.period, min_periods=5).mean().values
        return result


# ---- 资金流向因子 ----

class MoneyFlowFactor(BaseFactor):
    """资金流向因子: price_change * volume 的N日累计"""
    def __init__(self, period: int = 20):
        super().__init__(FactorMetadata(
            name=f"money_flow_{period}d",
            category="money_flow",
            description=f"{period}日资金流向",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            close = data.loc[mask, 'close']
            vol = data.loc[mask, 'volume']
            ret = close.pct_change()
            mf = ret * vol
            result[mask] = mf.rolling(self.period, min_periods=5).sum().values
        return result


class OBVFactor(BaseFactor):
    """OBV 变化率因子 (On-Balance Volume)"""
    def __init__(self, period: int = 20):
        super().__init__(FactorMetadata(
            name=f"obv_change_{period}d",
            category="money_flow",
            description=f"OBV {period}日变化率",
            lookback_period=period,
        ))
        self.period = period

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            close = data.loc[mask, 'close']
            vol = data.loc[mask, 'volume']
            direction = np.sign(close.diff().fillna(0))
            obv = (direction * vol).cumsum()
            result[mask] = obv.pct_change(self.period).values
        return result


# ---- 复合因子 ----

class SizeFactor(BaseFactor):
    """市值因子 (估算)"""
    def __init__(self):
        super().__init__(FactorMetadata(
            name="size_factor",
            category="composite",
            description="对数市值因子",
            lookback_period=1,
        ))

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if 'turnover_rate' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        has_amount = 'amount' in data.columns and not data['amount'].isna().all()
        if has_amount:
            mv = data['amount'] / data['turnover_rate'].replace(0, np.nan) * 100
        else:
            mv = data['volume'] / data['turnover_rate'].replace(0, np.nan) * data['close']
        return mv.replace(0, np.nan).apply(lambda x: np.log(x) if x > 0 else np.nan)


# =============================================================================
# 因子库管理器
# =============================================================================

class FactorLibrary:
    """
    标准化因子库管理器

    设计参考 Qlib Alpha158:
      - 分类清晰 (6大类)
      - 可扩展 (通过注册机制添加因子)
      - 工厂模式创建
    """

    CATEGORIES = [
        "trend", "reversal", "volume", "volatility",
        "money_flow", "composite", "custom"
    ]

    def __init__(self):
        self._factors: Dict[str, BaseFactor] = {}
        self._register_default_factors()

    def _register_default_factors(self):
        """注册默认因子 (30+)"""
        defaults = [
            # 趋势类
            MomentumFactor(5), MomentumFactor(10), MomentumFactor(20),
            MomentumFactor(60), MomentumFactor(120),
            MACDFactor(12, 26, 9), MACDFactor(5, 35, 5),
            RSIFactor(6), RSIFactor(14), RSIFactor(24),
            PricePositionFactor(20), PricePositionFactor(60),

            # 反转类
            ReversalFactor(3), ReversalFactor(5), ReversalFactor(20),
            GapFactor(),

            # 成交量类
            VolumeRatioFactor(5, 20), VolumeRatioFactor(5, 60),
            TurnoverFactor(20), TurnoverFactor(60),
            VolumeTrendFactor(10), VolumeTrendFactor(30),

            # 波动率类
            VolatilityFactor(20), VolatilityFactor(60),
            MaxDrawdownFactor(60), MaxDrawdownFactor(120),
            AmplitudeFactor(20),

            # 资金流向类
            MoneyFlowFactor(20), MoneyFlowFactor(60),
            OBVFactor(20),

            # 复合类
            SizeFactor(),
        ]
        for f in defaults:
            self._factors[f.metadata.name] = f

    def register(self, factor: BaseFactor):
        """注册新因子"""
        self._factors[factor.metadata.name] = factor

    def get(self, name: str) -> Optional[BaseFactor]:
        """获取因子"""
        return self._factors.get(name)

    def list_by_category(self, category: str) -> List[BaseFactor]:
        """按分类列出因子"""
        return [f for f in self._factors.values()
                if f.metadata.category == category]

    def list_all(self) -> List[FactorMetadata]:
        """列出所有因子元数据"""
        return [f.metadata for f in self._factors.values()]

    def compute_all(
        self,
        data: pd.DataFrame,
        categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        计算所有因子

        参数:
            data: 行情数据
            categories: 限定分类，None=全部

        返回: DataFrame，列为 code, date, [各因子值]
        """
        if categories:
            factors_to_use = [f for f in self._factors.values()
                            if f.metadata.category in categories]
        else:
            factors_to_use = list(self._factors.values())

        result = data[['code', 'date']].copy()
        for factor in factors_to_use:
            try:
                result[factor.metadata.name] = factor.compute(data)
            except Exception as e:
                print(f"因子 {factor.metadata.name} 计算失败: {e}")
                result[factor.metadata.name] = np.nan

        return result

    @property
    def count(self) -> int:
        return len(self._factors)

    def summary(self) -> pd.DataFrame:
        """生成因子库概览表"""
        rows = []
        for f in self._factors.values():
            rows.append({
                "name": f.metadata.name,
                "category": f.metadata.category,
                "description": f.metadata.description,
                "lookback": f.metadata.lookback_period,
                "neutralize": f.metadata.neutralize_recommended,
            })
        return pd.DataFrame(rows)


# =============================================================================
# 因子 IC 分析工具
# =============================================================================

class FactorICAnalyzer:
    """因子 IC 分析器"""

    def __init__(self, factor_df: pd.DataFrame, forward_returns: pd.DataFrame):
        self.factor_df = factor_df
        self.forward_returns = forward_returns

    def analyze(
        self,
        factor_names: List[str],
        forward_col: str = 'ret_forward_5d',
        method: str = 'spearman',
    ) -> pd.DataFrame:
        """
        计算 IC 统计

        返回: DataFrame，列为 factor, ic_mean, ic_std, ic_ir, ic_pos_ratio
        """
        merged = self.factor_df[['code', 'date'] + factor_names].merge(
            self.forward_returns[['code', 'date', forward_col]],
            on=['code', 'date'], how='inner'
        )

        results = []
        for factor in factor_names:
            if factor not in merged.columns:
                continue
            valid = merged.dropna(subset=[factor, forward_col])
            if len(valid) < 20:
                continue

            dates = sorted(valid['date'].unique())
            ic_list = []
            for dt in dates:
                cross = valid[valid['date'] == dt]
                if len(cross) < 5:
                    continue
                if method == 'spearman':
                    ic, _ = stats.spearmanr(cross[factor], cross[forward_col], nan_policy='omit')
                else:
                    ic, _ = stats.pearsonr(cross[factor].fillna(0), cross[forward_col].fillna(0))
                if not np.isnan(ic):
                    ic_list.append(ic)

            if not ic_list:
                continue

            ic_arr = np.array(ic_list)
            ic_mean = np.mean(ic_arr)
            ic_std = np.std(ic_arr)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_pos_ratio = np.mean(ic_arr > 0)

            results.append({
                "factor": factor,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "ic_pos_ratio": ic_pos_ratio,
            })

        return pd.DataFrame(results).sort_values("ic_ir", ascending=False)


# =============================================================================
# 测试用例
# =============================================================================

class TestFactorLibrary(unittest.TestCase):
    """因子库框架测试"""

    @classmethod
    def setUpClass(cls):
        """生成测试数据"""
        np.random.seed(42)
        n_codes = 10
        n_days = 120
        codes = [f'{i:06d}.SH' for i in range(600000, 600000 + n_codes)]
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
        cls.data = pd.DataFrame(rows)

        # 前视收益
        cls.forward_returns = pd.DataFrame()
        cls.forward_returns['code'] = cls.data['code']
        cls.forward_returns['date'] = cls.data['date']
        for period in [1, 5, 20]:
            cls.forward_returns[f'ret_forward_{period}d'] = (
                cls.data.groupby('code')['close'].transform(
                    lambda x: x.shift(-period) / x - 1
                )
            )

        cls.library = FactorLibrary()

    def test_library_size(self):
        """测试因子库规模"""
        self.assertGreaterEqual(self.library.count, 30,
                                "因子库应包含至少 30 个因子")

    def test_category_coverage(self):
        """测试分类覆盖"""
        categories = set()
        for f in self.library.list_all():
            categories.add(f.category)
        self.assertIn("trend", categories)
        self.assertIn("reversal", categories)
        self.assertIn("volume", categories)
        self.assertIn("volatility", categories)
        self.assertIn("money_flow", categories)

    def test_factor_computation(self):
        """测试因子计算正确性"""
        momentum_20 = self.library.get("momentum_20d")
        self.assertIsNotNone(momentum_20)

        result = momentum_20.compute(self.data)
        self.assertEqual(len(result), len(self.data))

        # 验证手动计算 vs 因子计算
        expected = self.data.groupby('code')['close'].transform(
            lambda x: x.pct_change(20)
        )
        common = ~result.isna() & ~expected.isna()
        np.testing.assert_array_almost_equal(
            result[common].values,
            expected[common].values,
            decimal=8,
        )

    def test_compute_all_factors(self):
        """测试批量因子计算"""
        factor_df = self.library.compute_all(self.data)

        self.assertIn('code', factor_df.columns)
        self.assertIn('date', factor_df.columns)
        self.assertGreater(len(factor_df.columns), 5)  # 至少有 code, date + 若干因子

        # 检查有效因子数目
        factor_cols = [c for c in factor_df.columns if c not in ['code', 'date']]
        self.assertGreater(len(factor_cols), 20)

    def test_factor_ic_analysis(self):
        """测试因子 IC 分析"""
        factor_df = self.library.compute_all(self.data)
        factor_cols = [c for c in factor_df.columns
                      if c not in ['code', 'date']]

        analyzer = FactorICAnalyzer(factor_df, self.forward_returns)
        ic_result = analyzer.analyze(factor_cols[:10])  # 分析前10个

        self.assertGreater(len(ic_result), 0)
        self.assertIn('ic_ir', ic_result.columns)
        self.assertIn('ic_mean', ic_result.columns)

    def test_factor_correlation_filter(self):
        """测试基于相关性的因子筛选"""
        factor_df = self.library.compute_all(self.data)
        factor_cols = [c for c in factor_df.columns
                      if c not in ['code', 'date'] and
                      not factor_df[c].isna().all()]

        # 计算截面均值相关矩阵
        factor_means = factor_df.groupby('date')[factor_cols].mean()
        corr = factor_means.corr()

        # 筛选高相关因子对
        max_corr = 0.8
        high_corr_pairs = []
        for i in range(len(factor_cols)):
            for j in range(i + 1, len(factor_cols)):
                if abs(corr.iloc[i, j]) > max_corr:
                    high_corr_pairs.append((factor_cols[i], factor_cols[j],
                                           corr.iloc[i, j]))

        # 应有部分高相关对（同类型因子相似）
        # 但不应该全部因子都高度相关
        high_corr_ratio = len(high_corr_pairs) / (
            len(factor_cols) * (len(factor_cols) - 1) / 2
        )
        self.assertLess(high_corr_ratio, 0.6,
                        "过高比例的因子相关性过高，因子多样性不足")

    def test_custom_factor_registration(self):
        """测试自定义因子注册"""
        class CustomFactor(BaseFactor):
            def compute(self, data):
                return data['close'] * 2

        custom = CustomFactor(FactorMetadata(
            name="custom_test",
            category="custom",
            description="测试自定义因子",
        ))
        self.library.register(custom)

        result = custom.compute(self.data)
        expected = self.data['close'] * 2
        np.testing.assert_array_equal(result.values, expected.values)

    def test_library_summary(self):
        """测试因子库摘要"""
        summary = self.library.summary()
        self.assertGreater(len(summary), 30)
        self.assertIn("name", summary.columns)
        self.assertIn("category", summary.columns)
        self.assertIn("description", summary.columns)

    def test_compare_with_existing(self):
        """对比新因子库与现有 jingni-trader 因子"""
        # jingni-trader 现有因子
        existing_factors = [
            "ret_1d", "ret_5d", "ret_20d", "ret_60d",
            "reversal_5d", "reversal_20d",
            "volatility_20d", "volume_ratio",
            "turnover_20d", "turnover_5d", "turnover_change",
            "money_flow_20d", "lncap", "estimated_mv",
        ]

        # 新因子库因子
        new_factors = [f.name for f in self.library.list_all()]

        # 扩展对比
        print(f"\n  现有因子数: {len(existing_factors)}")
        print(f"  新因子库因子数: {len(new_factors)}")
        print(f"  新增因子数: {len(set(new_factors) - set(existing_factors))}")


if __name__ == '__main__':
    unittest.main(verbosity=2)