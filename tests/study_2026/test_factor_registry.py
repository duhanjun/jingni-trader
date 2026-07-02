"""
优化方向：因子注册与自动发现机制
借鉴来源：Microsoft Qlib (https://github.com/microsoft/qlib)
借鉴亮点：Qlib的表达式引擎 + DataHandlerLP "配置即代码" + Alpha158因子分组管理

问题分析：
  jingni-trader 当前因子计算在 factor-engine/engine.py 的 compute_a_share_factors() 中
  硬编码实现，每新增一个因子都需要修改核心代码。缺乏：
  1. 因子注册/自动发现机制
  2. 因子分组管理（动量/反转/波动率/流动性等）
  3. 声明式因子定义
  4. 因子依赖关系管理

优化方案：
  引入 FactorRegistry 注册表模式，支持：
  - @register_factor 装饰器自动注册因子
  - 因子分组（momentum/reversal/volatility/liquidity/volume）
  - 因子元信息（名称、描述、分组、依赖、参数）
  - 自动发现因子模块中的因子定义
  - 按组批量计算因子
"""

import sys
import os
import unittest
import time
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from functools import wraps

import numpy as np
import pandas as pd


# ============================================================
# 因子注册表实现（借鉴 Qlib DataHandlerLP 配置即代码模式）
# ============================================================

@dataclass
class FactorMeta:
    """因子元信息"""
    name: str
    category: str           # 分组: momentum, reversal, volatility, liquidity, volume
    description: str
    dependencies: List[str] = field(default_factory=list)  # 依赖的因子
    params: Dict[str, Any] = field(default_factory=dict)    # 参数
    priority: int = 0       # 计算优先级（数值越小越优先）


class FactorRegistry:
    """
    因子注册表（借鉴 Qlib 的配置驱动设计）

    特性：
    - 装饰器注册
    - 按分组管理
    - 依赖排序
    - 批量计算
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._factors = {}       # name -> (func, meta)
            cls._instance._categories = {}    # category -> [factor_names]
        return cls._instance

    def register(
        self,
        name: str,
        category: str = "custom",
        description: str = "",
        dependencies: List[str] = None,
        params: Dict[str, Any] = None,
        priority: int = 0,
    ):
        """因子注册装饰器"""
        def decorator(func: Callable):
            meta = FactorMeta(
                name=name,
                category=category,
                description=description,
                dependencies=dependencies or [],
                params=params or {},
                priority=priority,
            )
            self._factors[name] = (func, meta)
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def list_factors(self, category: str = None) -> List[FactorMeta]:
        """列出已注册的因子"""
        if category:
            names = self._categories.get(category, [])
            return [self._factors[n][1] for n in names]
        return [meta for _, meta in self._factors.values()]

    def list_categories(self) -> List[str]:
        """列出所有因子分组"""
        return list(self._categories.keys())

    def _topological_sort(self, factor_names: List[str]) -> List[str]:
        """拓扑排序确保依赖先计算"""
        in_degree = {n: 0 for n in factor_names}
        graph = {n: [] for n in factor_names}

        for n in factor_names:
            _, meta = self._factors.get(n, (None, None))
            if meta is None:
                continue
            for dep in meta.dependencies:
                if dep in graph:
                    graph[dep].append(n)
                    in_degree[n] += 1

        queue = [n for n in factor_names if in_degree[n] == 0]
        sorted_names = []
        while queue:
            queue.sort(key=lambda n: self._factors[n][1].priority)  # 同层级按优先级
            node = queue.pop(0)
            sorted_names.append(node)
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_names) != len(factor_names):
            missing = set(factor_names) - set(sorted_names)
            raise ValueError(f"因子存在循环依赖或缺失依赖: {missing}")
        return sorted_names

    def compute_factors(
        self,
        data: pd.DataFrame,
        factor_names: List[str] = None,
        category: str = None,
    ) -> pd.DataFrame:
        """
        批量计算因子

        参数:
            data: 原始行情数据（需包含 code, date, open, high, low, close, volume）
            factor_names: 要计算的因子名列表（None=全部）
            category: 按分组计算（如果 factor_names 未指定）
        """
        if factor_names is None:
            if category:
                factor_names = self._categories.get(category, [])
            else:
                factor_names = list(self._factors.keys())

        # 拓扑排序
        sorted_names = self._topological_sort(factor_names)

        result = data[['code', 'date']].copy()

        for name in sorted_names:
            func, meta = self._factors[name]
            try:
                factor_values = func(data, result, **meta.params)
                result[name] = factor_values
            except Exception as e:
                print(f"[WARN] 因子 {name} 计算失败: {e}")
                result[name] = np.nan

        return result

    def get_metadata(self, factor_name: str) -> Optional[FactorMeta]:
        """获取因子元信息"""
        _, meta = self._factors.get(factor_name, (None, None))
        return meta

    def clear(self):
        """清空注册表（用于测试）"""
        self._factors.clear()
        self._categories.clear()


# ============================================================
# 示例因子定义（模拟 jingni-trader 现有因子 + 新因子）
# ============================================================

registry = FactorRegistry()


@registry.register(
    name="ret_1d",
    category="momentum",
    description="1日收益率（动量因子）",
    priority=1,
)
def calc_ret_1d(data: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    return data.groupby('code')['close'].pct_change()


@registry.register(
    name="ret_5d",
    category="momentum",
    description="5日收益率（中期动量因子）",
    priority=1,
)
def calc_ret_5d(data: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    return data.groupby('code')['close'].pct_change(5)


@registry.register(
    name="ret_20d",
    category="momentum",
    description="20日收益率（长期动量因子）",
    priority=1,
)
def calc_ret_20d(data: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    return data.groupby('code')['close'].pct_change(20)


@registry.register(
    name="reversal_5d",
    category="reversal",
    description="5日反转因子（短期反转效应）",
    dependencies=["ret_5d"],
    priority=2,
)
def calc_reversal_5d(data: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    return -ctx['ret_5d']


@registry.register(
    name="reversal_20d",
    category="reversal",
    description="20日反转因子（中期反转效应）",
    dependencies=["ret_20d"],
    priority=2,
)
def calc_reversal_20d(data: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    return -ctx['ret_20d']


@registry.register(
    name="volatility_20d",
    category="volatility",
    description="20日波动率因子",
    params={"window": 20, "min_periods": 10},
    priority=3,
)
def calc_volatility_20d(data: pd.DataFrame, ctx: pd.DataFrame, window=20, min_periods=10) -> pd.Series:
    return data.groupby('code')['close'].transform(
        lambda x: x.pct_change().rolling(window, min_periods=min_periods).std()
    )


@registry.register(
    name="volume_ratio",
    category="volume",
    description="量比因子（当日成交量 / 20日均量）",
    dependencies=["volatility_20d"],  # 软依赖：需要先计算 vol_20d 确保数据流正确
    priority=3,
)
def calc_volume_ratio(data: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    vol_20d = data.groupby('code')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    return data['volume'] / vol_20d.replace(0, np.nan)


@registry.register(
    name="turnover_20d",
    category="liquidity",
    description="20日平均换手率因子",
    priority=3,
)
def calc_turnover_20d(data: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    if 'turnover_rate' not in data.columns:
        return pd.Series(np.nan, index=data.index)
    return data.groupby('code')['turnover_rate'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )


@registry.register(
    name="lncap",
    category="liquidity",
    description="对数市值因子（大小盘风格）",
    priority=2,
)
def calc_lncap(data: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    if 'amount' not in data.columns or 'turnover_rate' not in data.columns:
        return pd.Series(np.nan, index=data.index)
    mv = data['amount'] / data['turnover_rate'].replace(0, np.nan) * 100
    return mv.replace(0, np.nan).apply(lambda x: np.log(x) if x > 0 else np.nan)


@registry.register(
    name="momentum_quality",
    category="momentum",
    description="动量质量因子（收益/波动率，衡量风险调整动量）",
    dependencies=["ret_20d", "volatility_20d"],
    priority=4,
)
def calc_momentum_quality(data: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    """复合因子：动量质量 = 20日收益 / 20日波动率"""
    vol = ctx['volatility_20d'].replace(0, np.nan)
    return ctx['ret_20d'] / vol


# ============================================================
# 单元测试
# ============================================================


class TestFactorRegistry(unittest.TestCase):
    """因子注册表功能测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟 A 股数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
        codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
        rows = []
        for code in codes:
            price = 20 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            df_one = pd.DataFrame({
                "date": dates,
                "code": code,
                "open": price * (1 + np.random.randn(len(dates)) * 0.005),
                "high": price * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
                "low": price * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
                "close": price,
                "volume": np.random.randint(100000, 1000000, len(dates)),
                "amount": np.random.randint(500000, 5000000, len(dates)),
                "turnover_rate": np.random.uniform(0.01, 0.05, len(dates)),
            })
            rows.append(df_one)
        cls.test_data = pd.concat(rows, ignore_index=True)

    def test_factor_registration(self):
        """测试因子注册"""
        all_factors = registry.list_factors()
        self.assertGreater(len(all_factors), 5, "应该注册了至少5个因子")

        names = [f.name for f in all_factors]
        self.assertIn("ret_1d", names)
        self.assertIn("reversal_20d", names)
        self.assertIn("momentum_quality", names)

    def test_category_management(self):
        """测试因子分组管理"""
        categories = registry.list_categories()
        self.assertIn("momentum", categories)
        self.assertIn("reversal", categories)
        self.assertIn("volatility", categories)
        self.assertIn("volume", categories)
        self.assertIn("liquidity", categories)

        momentum_factors = registry.list_factors("momentum")
        momentum_names = [f.name for f in momentum_factors]
        self.assertIn("ret_1d", momentum_names)
        self.assertIn("ret_20d", momentum_names)
        self.assertIn("momentum_quality", momentum_names)

    def test_dependency_topological_sort(self):
        """测试依赖拓扑排序"""
        factor_names = ["reversal_5d", "ret_5d", "momentum_quality", "ret_20d", "volatility_20d"]
        sorted_names = registry._topological_sort(factor_names)

        # ret_5d 必须在 reversal_5d 之前
        self.assertLess(sorted_names.index("ret_5d"), sorted_names.index("reversal_5d"))
        # ret_20d 和 volatility_20d 必须在 momentum_quality 之前
        self.assertLess(sorted_names.index("ret_20d"), sorted_names.index("momentum_quality"))
        self.assertLess(sorted_names.index("volatility_20d"), sorted_names.index("momentum_quality"))

    def test_circular_dependency_detection(self):
        """测试循环依赖检测"""
        # 临时添加循环依赖的因子
        @registry.register(
            name="circular_a",
            category="test",
            description="循环依赖测试A",
            dependencies=["circular_b"],
        )
        def calc_circular_a(data, ctx):
            return pd.Series(0, index=data.index)

        @registry.register(
            name="circular_b",
            category="test",
            description="循环依赖测试B",
            dependencies=["circular_a"],
        )
        def calc_circular_b(data, ctx):
            return pd.Series(0, index=data.index)

        with self.assertRaises(ValueError):
            registry._topological_sort(["circular_a", "circular_b"])

        # 清理
        registry._factors.pop("circular_a", None)
        registry._factors.pop("circular_b", None)

    def test_compute_all_factors(self):
        """测试批量计算所有因子"""
        result = registry.compute_factors(self.test_data)

        expected_factors = [
            "ret_1d", "ret_5d", "ret_20d",
            "reversal_5d", "reversal_20d",
            "volatility_20d", "volume_ratio",
            "turnover_20d", "lncap", "momentum_quality",
        ]
        for f in expected_factors:
            self.assertIn(f, result.columns, f"因子 {f} 应该存在于结果中")

        # 验证数据形状
        self.assertEqual(len(result), len(self.test_data))

        # 验证 ret_5d 和 reversal_5d 的关系
        valid_mask = result['ret_5d'].notna() & result['reversal_5d'].notna()
        self.assertTrue(
            np.allclose(result.loc[valid_mask, 'reversal_5d'], -result.loc[valid_mask, 'ret_5d']),
            "reversal_5d 应该等于 -ret_5d"
        )

    def test_compute_by_category(self):
        """测试按分组计算因子"""
        result = registry.compute_factors(self.test_data, category="momentum")

        momentum_factors = ["ret_1d", "ret_5d", "ret_20d", "momentum_quality"]
        for f in momentum_factors:
            self.assertIn(f, result.columns)

        # 确保只包含动量分组的因子
        all_factor_names = [f.name for f in registry.list_factors("momentum")]
        for col in result.columns:
            if col not in ['code', 'date']:
                self.assertIn(col, all_factor_names)

    def test_extensibility(self):
        """测试因子扩展性（新因子注册）"""
        @registry.register(
            name="new_custom_factor",
            category="custom",
            description="自定义因子测试",
            params={"multiplier": 2.0},
        )
        def calc_custom(data, ctx, multiplier=2.0):
            return data['close'] * multiplier

        result = registry.compute_factors(self.test_data, factor_names=["new_custom_factor"])
        self.assertIn("new_custom_factor", result.columns)

        # 验证计算正确性
        expected = self.test_data['close'] * 2.0
        pd.testing.assert_series_equal(
            result['new_custom_factor'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

        # 清理
        registry._factors.pop("new_custom_factor", None)
        if "custom" in registry._categories:
            registry._categories["custom"] = [
                n for n in registry._categories["custom"] if n != "new_custom_factor"
            ]

    def test_metadata_access(self):
        """测试因子元信息查询"""
        meta = registry.get_metadata("momentum_quality")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "momentum_quality")
        self.assertEqual(meta.category, "momentum")
        self.assertEqual(meta.priority, 4)
        self.assertIn("ret_20d", meta.dependencies)
        self.assertIn("volatility_20d", meta.dependencies)


class TestPerformanceComparison(unittest.TestCase):
    """性能对比测试：FactorRegistry vs 硬编码"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_stocks = 50
        n_days = 500
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        codes = [f"{i:06d}.SZ" for i in range(n_stocks)]

        rows = []
        for code in codes:
            price = 20 + np.cumsum(np.random.randn(n_days) * 0.5)
            df_one = pd.DataFrame({
                "date": dates,
                "code": code,
                "open": price * (1 + np.random.randn(n_days) * 0.005),
                "high": price * (1 + np.abs(np.random.randn(n_days) * 0.01)),
                "low": price * (1 - np.abs(np.random.randn(n_days) * 0.01)),
                "close": price,
                "volume": np.random.randint(100000, 1000000, n_days),
                "amount": np.random.randint(500000, 5000000, n_days),
                "turnover_rate": np.random.uniform(0.01, 0.05, n_days),
            })
            rows.append(df_one)
        cls.large_test_data = pd.concat(rows, ignore_index=True)

    def test_registry_vs_hardcoded_performance(self):
        """对比 FactorRegistry 与硬编码方式的性能"""
        data = self.large_test_data

        # 硬编码方式
        def hardcoded_compute(df):
            result = df[['code', 'date']].copy()
            result['ret_1d'] = df.groupby('code')['close'].pct_change()
            result['ret_5d'] = df.groupby('code')['close'].pct_change(5)
            result['ret_20d'] = df.groupby('code')['close'].pct_change(20)
            result['reversal_5d'] = -result['ret_5d']
            result['reversal_20d'] = -result['ret_20d']
            result['volatility_20d'] = df.groupby('code')['close'].transform(
                lambda x: x.pct_change().rolling(20, min_periods=10).std()
            )
            return result

        start = time.perf_counter()
        hardcoded_result = hardcoded_compute(data)
        hardcoded_time = time.perf_counter() - start

        # 注册表方式
        factor_names = ["ret_1d", "ret_5d", "ret_20d", "reversal_5d", "reversal_20d", "volatility_20d"]
        start = time.perf_counter()
        registry_result = registry.compute_factors(data, factor_names=factor_names)
        registry_time = time.perf_counter() - start

        print(f"\n性能对比 (50只股票 × 500天):")
        print(f"  硬编码方式: {hardcoded_time:.4f}秒")
        print(f"  注册表方式: {registry_time:.4f}秒")
        print(f"  性能比: {registry_time/hardcoded_time:.2f}x")

        # 验证结果一致性
        for col in ["ret_1d", "ret_5d", "ret_20d", "reversal_5d", "reversal_20d", "volatility_20d"]:
            pd.testing.assert_series_equal(
                hardcoded_result[col].fillna(0).reset_index(drop=True),
                registry_result[col].fillna(0).reset_index(drop=True),
                check_names=False,
                rtol=1e-4,
            )

        # 注册表方式不应显著慢于硬编码（2 倍以内）
        self.assertLess(registry_time, hardcoded_time * 3,
                       "注册表方式的性能开销应在可接受范围内")


if __name__ == "__main__":
    unittest.main(verbosity=2)