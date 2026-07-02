"""
验证测试：装饰器式因子注册 API

借鉴来源：
  - Factor Engine (arXiv:2602.14138, Ata Keskin, TUM, 2026-02):
    "Built around a modular and extensible API that leverages Python decorators,
     Factor Engine enables users to define custom factors with ease and integrates
     seamlessly with the modern data science ecosystem."

优化方向：
  - 将 jingni-trader 中 factor-engine 的硬编码因子计算（compute_a_share_factors）
    重构为装饰器式注册机制，提升因子库的可扩展性和自定义能力。

测试内容：
  1. 装饰器注册 API 正确性
  2. 因子自动发现与组合
  3. 行业中性化集成
  4. 对比当前硬编码方式的因子输出一致性
"""

import sys
import os
import time
import logging
import unittest
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test-decorator-factor")


# ============================================================================
# 提案代码：装饰器式因子注册系统
# ============================================================================

class FactorRegistry:
    """因子注册中心 —— 管理所有已注册的因子定义"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factors = {}
            cls._instance._categories = {}
        return cls._instance

    def register(self, name: str = None, category: str = "alpha",
                 requires: list = None, neutralize: bool = True,
                 description: str = ""):
        """装饰器：将函数注册为因子计算器"""
        def decorator(func):
            factor_name = name or func.__name__
            self._factors[factor_name] = {
                "func": func,
                "category": category,
                "requires": requires or [],
                "neutralize": neutralize,
                "description": description,
            }
            self._categories.setdefault(category, []).append(factor_name)
            logger.debug(f"注册因子: {factor_name} (类别: {category})")
            return func
        return decorator

    def list_factors(self, category: str = None) -> list:
        """列出已注册的因子"""
        if category:
            return self._categories.get(category, [])
        return list(self._factors.keys())

    def get_factor(self, name: str) -> dict:
        """获取因子元信息"""
        return self._factors.get(name)

    def clear(self):
        """清空注册表（用于测试）"""
        self._factors = {}
        self._categories = {}


# 全局注册中心
registry = FactorRegistry()


# ============================================================================
# 使用装饰器定义因子（替代硬编码方式）
# ============================================================================

@registry.register(
    name="ret_1d",
    category="momentum",
    requires=["close"],
    neutralize=False,
    description="1日收益率"
)
def compute_ret_1d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].pct_change().rename("ret_1d")


@registry.register(
    name="ret_5d",
    category="momentum",
    requires=["close"],
    neutralize=False,
    description="5日收益率"
)
def compute_ret_5d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].pct_change(5).rename("ret_5d")


@registry.register(
    name="ret_20d",
    category="momentum",
    requires=["close"],
    neutralize=False,
    description="20日收益率"
)
def compute_ret_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].pct_change(20).rename("ret_20d")


@registry.register(
    name="reversal_5d",
    category="reversal",
    requires=["close"],
    neutralize=True,
    description="5日反转因子（负5日收益）"
)
def compute_reversal_5d(df: pd.DataFrame) -> pd.Series:
    ret_5d = df.groupby('code')['close'].pct_change(5)
    return (-ret_5d).rename("reversal_5d")


@registry.register(
    name="reversal_20d",
    category="reversal",
    requires=["close"],
    neutralize=True,
    description="20日反转因子（负20日收益）"
)
def compute_reversal_20d(df: pd.DataFrame) -> pd.Series:
    ret_20d = df.groupby('code')['close'].pct_change(20)
    return (-ret_20d).rename("reversal_20d")


@registry.register(
    name="volatility_20d",
    category="risk",
    requires=["close"],
    neutralize=True,
    description="20日波动率"
)
def compute_volatility_20d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    ).rename("volatility_20d")


@registry.register(
    name="turnover_20d",
    category="liquidity",
    requires=["turnover_rate"],
    neutralize=True,
    description="20日均换手率"
)
def compute_turnover_20d(df: pd.DataFrame) -> pd.Series:
    if 'turnover_rate' not in df.columns:
        return pd.Series(np.nan, index=df.index, name="turnover_20d")
    return df.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    ).rename("turnover_20d")


@registry.register(
    name="volume_ratio",
    category="liquidity",
    requires=["volume"],
    neutralize=True,
    description="量比（当日成交量 / 20日均量）"
)
def compute_volume_ratio(df: pd.DataFrame) -> pd.Series:
    vol_20 = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    return (df['volume'] / vol_20.replace(0, np.nan)).rename("volume_ratio")


@registry.register(
    name="money_flow_20d",
    category="sentiment",
    requires=["close", "volume"],
    neutralize=True,
    description="20日资金流向"
)
def compute_money_flow_20d(df: pd.DataFrame) -> pd.Series:
    ret_1d = df.groupby('code')['close'].pct_change()
    mf_raw = ret_1d * df.get('amount', df['volume'])
    return mf_raw.groupby(df['code']).transform(
        lambda x: x.rolling(20, min_periods=5).sum()
    ).rename("money_flow_20d")


@registry.register(
    name="lncap",
    category="size",
    requires=["amount", "turnover_rate"],
    neutralize=False,
    description="对数市值"
)
def compute_lncap(df: pd.DataFrame) -> pd.Series:
    if 'amount' not in df.columns or 'turnover_rate' not in df.columns:
        return pd.Series(np.nan, index=df.index, name="lncap")
    mv = df['amount'] / df['turnover_rate'].replace(0, np.nan) * 100
    return mv.replace(0, np.nan).apply(lambda x: np.log(x) if x > 0 else np.nan).rename("lncap")


# ============================================================================
# 用户自定义因子示例（演示可扩展性）
# ============================================================================

@registry.register(
    name="custom_momentum_divergence",
    category="alpha",
    requires=["close"],
    neutralize=True,
    description="自定义动量背离因子：短期收益与长期收益之差"
)
def compute_custom_momentum_divergence(df: pd.DataFrame) -> pd.Series:
    short = df.groupby('code')['close'].pct_change(5)
    long = df.groupby('code')['close'].pct_change(60)
    return (short - long).rename("custom_momentum_divergence")


# ============================================================================
# 因子编排器（替代原 FactorEngine.compute_a_share_factors）
# ============================================================================

class DecoratorFactorEngine:
    """基于装饰器注册的因子计算编排器"""

    def __init__(self, registry: FactorRegistry = None):
        self.registry = registry or FactorRegistry()

    def compute_all(self, data: pd.DataFrame, factor_names: list = None,
                    category: str = None) -> pd.DataFrame:
        """批量计算已注册的因子"""
        if factor_names is None:
            factor_names = self.registry.list_factors(category)

        df = data.sort_values(['code', 'date']).copy()
        result = df[['code', 'date']].copy()

        for name in factor_names:
            factor_info = self.registry.get_factor(name)
            if factor_info is None:
                logger.warning(f"因子 {name} 未注册，跳过")
                continue

            try:
                factor_series = factor_info["func"](df)
                result[name] = factor_series.values
            except Exception as e:
                logger.warning(f"计算因子 {name} 失败: {e}")
                result[name] = np.nan

        return result

    def compute_with_neutralize(
        self, data: pd.DataFrame, industry_df: pd.DataFrame,
        factor_names: list = None
    ) -> pd.DataFrame:
        """计算因子并执行行业中性化"""
        from sklearn.linear_model import LinearRegression

        result = self.compute_all(data, factor_names)
        factor_cols = [f for f in factor_names if f in result.columns]

        result = result.merge(industry_df[['code', 'industry']], on='code', how='left')

        for factor in factor_cols:
            factor_info = self.registry.get_factor(factor)
            if factor_info is None or not factor_info.get("neutralize", True):
                continue

            neutralized = pd.Series(index=result.index, dtype=float)
            for dt in result['date'].unique():
                cross = result[result['date'] == dt]
                if len(cross) < 30:
                    neutralized.loc[cross.index] = cross[factor]
                    continue

                dummies = pd.get_dummies(cross['industry'], prefix='ind')
                X = dummies.values
                y = cross[factor].fillna(0).values

                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    residual = y - model.predict(X)
                    neutralized.loc[cross.index] = residual
                except Exception:
                    neutralized.loc[cross.index] = cross[factor]

            result[f"{factor}_neutral"] = neutralized

        return result


# ============================================================================
# 单元测试
# ============================================================================

class TestFactorRegistry(unittest.TestCase):
    """因子注册中心测试"""

    def setUp(self):
        self.reg = FactorRegistry()
        self.reg.clear()

    def test_register_and_list(self):
        """测试注册与列表功能"""
        @self.reg.register(name="test_factor", category="test")
        def test_fn(df):
            return df['close']

        factors = self.reg.list_factors()
        self.assertIn("test_factor", factors)
        self.assertEqual(self.reg.get_factor("test_factor")["category"], "test")
        self.assertIn("test_factor", self.reg.list_factors("test"))

    def test_get_nonexistent(self):
        """测试获取不存在因子返回 None"""
        self.assertIsNone(self.reg.get_factor("nonexistent"))

    def test_default_name(self):
        """测试默认使用函数名"""
        @self.reg.register()
        def my_custom_factor(df):
            return df['close']

        self.assertIn("my_custom_factor", self.reg.list_factors())

    def test_multiple_categories(self):
        """测试多分类因子"""
        @self.reg.register(name="f1", category="momentum")
        def f1(df):
            return df['close']

        @self.reg.register(name="f2", category="reversal")
        def f2(df):
            return df['close']

        @self.reg.register(name="f3", category="momentum")
        def f3(df):
            return df['close']

        self.assertEqual(len(self.reg.list_factors("momentum")), 2)
        self.assertEqual(len(self.reg.list_factors("reversal")), 1)
        self.assertEqual(len(self.reg.list_factors()), 3)


class TestDecoratorFactorEngine(unittest.TestCase):
    """装饰器因子引擎测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟数据"""
        np.random.seed(42)
        n_codes = 10
        n_days = 252
        codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

        rows = []
        for code in codes:
            price = np.random.uniform(10, 50)
            for date in dates:
                ret = np.random.normal(0.0005, 0.02)
                price = price * (1 + ret)
                rows.append({
                    'code': code,
                    'date': date,
                    'open': price * (1 + np.random.normal(0, 0.003)),
                    'high': price * (1 + abs(np.random.normal(0, 0.01))),
                    'low': price * (1 - abs(np.random.normal(0, 0.01))),
                    'close': price,
                    'volume': int(np.random.lognormal(12, 0.5)),
                    'amount': price * np.random.lognormal(12, 0.5),
                    'turnover_rate': abs(np.random.normal(0.02, 0.01)),
                })

        cls.test_data = pd.DataFrame(rows)
        cls.engine = DecoratorFactorEngine()

    def test_compute_all_factors(self):
        """测试计算所有已注册因子"""
        result = self.engine.compute_all(self.test_data)
        factor_names = [c for c in result.columns if c not in ['code', 'date']]
        self.assertIn("reversal_20d", factor_names)
        self.assertIn("volatility_20d", factor_names)
        self.assertIn("custom_momentum_divergence", factor_names)

    def test_compute_by_category(self):
        """测试按类别计算因子"""
        result = self.engine.compute_all(self.test_data, category="momentum")
        factor_names = [c for c in result.columns if c not in ['code', 'date']]
        self.assertIn("ret_5d", factor_names)
        self.assertIn("ret_20d", factor_names)
        self.assertNotIn("volatility_20d", factor_names)  # 属于 risk 类别

    def test_output_shape(self):
        """测试输出形状正确"""
        result = self.engine.compute_all(self.test_data)
        self.assertEqual(len(result), len(self.test_data))
        self.assertIn('code', result.columns)
        self.assertIn('date', result.columns)

    def test_custom_factor_extensibility(self):
        """测试自定义因子的可扩展性"""
        result = self.engine.compute_all(self.test_data)
        self.assertIn("custom_momentum_divergence", result.columns)
        # 验证正确性：短期收益与长期收益之差
        short_ret = self.test_data.groupby('code')['close'].pct_change(5)
        long_ret = self.test_data.groupby('code')['close'].pct_change(60)
        expected = (short_ret - long_ret).values
        actual = result['custom_momentum_divergence'].values
        # 忽略 NaN 位置
        mask = ~(np.isnan(expected) & np.isnan(actual))
        self.assertTrue(np.allclose(expected[mask], actual[mask], equal_nan=True),
                       "自定义因子计算值不正确")


class TestConsistencyWithOriginal(unittest.TestCase):
    """与原有 FactorEngine 的一致性对比测试"""

    @classmethod
    def setUpClass(cls):
        """生成与原始计算逻辑相同的输入数据"""
        np.random.seed(2024)
        n_codes = 5
        n_days = 120
        codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

        rows = []
        for code in codes:
            price = np.random.uniform(10, 50)
            for date in dates:
                ret = np.random.normal(0.0005, 0.02)
                price = price * (1 + ret)
                turnover = abs(np.random.normal(0.02, 0.01))
                vol = int(np.random.lognormal(12, 0.5))
                rows.append({
                    'code': code,
                    'date': date,
                    'close': price,
                    'volume': vol,
                    'amount': price * vol,
                    'turnover_rate': turnover,
                    'change_pct': ret * 100,
                })

        cls.test_data = pd.DataFrame(rows)
        cls.engine = DecoratorFactorEngine()

    def test_reversal_20d_consistency(self):
        """验证反转因子与原始实现一致"""
        # 原始方式
        df = self.test_data.sort_values(['code', 'date']).copy()
        original_ret_20d = df.groupby('code')['close'].pct_change(20)
        original_reversal_20d = -original_ret_20d

        # 新方式
        result = self.engine.compute_all(self.test_data, factor_names=["reversal_20d"])

        actual = result['reversal_20d'].values
        expected = original_reversal_20d.values

        mask = ~(np.isnan(expected) & np.isnan(actual))
        self.assertTrue(np.allclose(expected[mask], actual[mask], equal_nan=True),
                       "reversal_20d 因子与原始实现不一致")

    def test_volatility_20d_consistency(self):
        """验证波动率因子与原始实现一致"""
        df = self.test_data.sort_values(['code', 'date']).copy()
        original_vol = df.groupby('code')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )

        result = self.engine.compute_all(self.test_data, factor_names=["volatility_20d"])
        actual = result['volatility_20d'].values
        expected = original_vol.values

        mask = ~(np.isnan(expected) & np.isnan(actual))
        self.assertTrue(np.allclose(expected[mask], actual[mask], equal_nan=True),
                       "volatility_20d 因子与原始实现不一致")


# ============================================================================
# 扩展性演示：运行时动态注册新因子
# ============================================================================

def demo_dynamic_registration():
    """演示运行时动态注册新因子 —— 无需修改引擎代码"""
    logger.info("=== 演示：运行时动态注册新因子 ===")

    @registry.register(
        name="rsi_14",
        category="technical",
        requires=["close"],
        neutralize=True,
        description="14日RSI指标"
    )
    def compute_rsi_14(df: pd.DataFrame) -> pd.Series:
        """计算14日RSI"""
        delta = df.groupby('code')['close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.groupby(df['code']).transform(
            lambda x: x.rolling(14, min_periods=5).mean()
        )
        avg_loss = loss.groupby(df['code']).transform(
            lambda x: x.rolling(14, min_periods=5).mean()
        )
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return rsi.rename("rsi_14")

    logger.info(f"当前已注册因子: {registry.list_factors()}")
    logger.info(f"动量类因子: {registry.list_factors('momentum')}")
    logger.info(f"技术类因子: {registry.list_factors('technical')}")


if __name__ == "__main__":
    demo_dynamic_registration()
    unittest.main(verbosity=2)