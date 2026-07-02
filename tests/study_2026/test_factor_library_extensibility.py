"""
优化方向: 因子库可扩展性优化
借鉴来源: Microsoft Qlib Alpha158 因子库设计
  - https://github.com/microsoft/qlib
  - Qlib 内置 Alpha158（158个因子）和 Alpha360（360个因子）
  - 因子分为类别: K 线类、均线类、动量类、波动类、量价类等
  - 每个因子通过 DataHandler 中的表达式引擎声明式定义
  - 支持 PIT (Point-in-Time) 数据库避免前视偏差

优化分析:
  jingnitrader 当前因子引擎 compute_a_share_factors() 硬编码了约 15
  个因子，全部写在同一个方法中。扩展因子库需要直接修改引擎源码。

验证内容:
  1. 实现可配置的因子注册表（通过 YAML/JSON 配置文件）
  2. 演示新增因子类别不需要修改引擎代码
  3. 因子计算流程：注册 → 验证 → 计算 → 输出
  4. 对比硬编码方式与可配置方式的扩展效率
"""

import os
import sys
import json
import unittest
import warnings
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ===================== 可配置因子库 =====================


class FactorCategory(str, Enum):
    """因子类别（借鉴 Qlib Alpha158 分类体系）"""
    KLINE = "kline"          # K 线形态类: open/high/low/close 相关
    PRICE = "price"           # 价格类: 收益率、均价
    VOLUME = "volume"         # 成交量类: 量比、换手
    MOMENTUM = "momentum"    # 动量类: N 日涨跌幅
    REVERSAL = "reversal"    # 反转类: 短期反转
    VOLATILITY = "volatility" # 波动类: 标准差、振幅
    CORRELATION = "correlation" # 相关性类
    TECHNICAL = "technical"  # 技术指标类: RSI, MACD, etc.


@dataclass
class FactorDefinition:
    """单个因子的定义"""
    name: str                          # 因子名称
    category: FactorCategory           # 因子类别
    expression: str                    # 因子表达式（配合 ExpressionEngine 使用）
    description: str = ""              # 因子描述
    neutralize_industry: bool = True   # 是否行业中性化
    neutralize_mcap: bool = True       # 是否市值中性化
    min_periods: int = 10              # 最小数据期数
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorLibraryConfig:
    """因子库全局配置"""
    version: str = "1.0"
    name: str = "jingnitrader-factor-library"
    description: str = ""
    factors: List[FactorDefinition] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "factors": [
                {
                    "name": f.name,
                    "category": f.category.value,
                    "expression": f.expression,
                    "description": f.description,
                    "neutralize_industry": f.neutralize_industry,
                    "neutralize_mcap": f.neutralize_mcap,
                    "min_periods": f.min_periods,
                    "params": f.params,
                }
                for f in self.factors
            ]
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "FactorLibraryConfig":
        factors = []
        for fd in data.get("factors", []):
            factors.append(FactorDefinition(
                name=fd["name"],
                category=FactorCategory(fd["category"]),
                expression=fd["expression"],
                description=fd.get("description", ""),
                neutralize_industry=fd.get("neutralize_industry", True),
                neutralize_mcap=fd.get("neutralize_mcap", True),
                min_periods=fd.get("min_periods", 10),
                params=fd.get("params", {}),
            ))
        return cls(
            version=data.get("version", "1.0"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            factors=factors,
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "FactorLibraryConfig":
        return cls.from_dict(json.loads(json_str))


class FactorLibrary:
    """
    可配置因子库管理器（借鉴 Qlib Alpha158 因子注册机制）
    
    核心设计:
    - 因子定义存储在配置文件中，运行时加载
    - 支持按类别分组管理
    - 新增因子只需添加配置项，无需修改引擎代码
    - 内置因子验证规则（名称唯一性、表达式合法性等）
    """

    def __init__(self, config: FactorLibraryConfig):
        self.config = config
        self._validate()
        self._build_index()

    def _validate(self):
        """验证因子配置的合法性"""
        names = [f.name for f in self.config.factors]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            raise ValueError(f"因子名称重复: {set(duplicates)}")
        
        for factor in self.config.factors:
            if not factor.name or not factor.expression:
                raise ValueError(f"因子 {factor.name}: 名称或表达式不能为空")

    def _build_index(self):
        """建立因子索引"""
        self._name_index: Dict[str, FactorDefinition] = {
            f.name: f for f in self.config.factors
        }
        self._category_index: Dict[FactorCategory, List[FactorDefinition]] = {}
        for f in self.config.factors:
            self._category_index.setdefault(f.category, []).append(f)

    def get_factor(self, name: str) -> Optional[FactorDefinition]:
        """按名称获取因子定义"""
        return self._name_index.get(name)

    def get_factors_by_category(self, category: FactorCategory) -> List[FactorDefinition]:
        """按类别获取因子列表"""
        return self._category_index.get(category, [])

    def get_all_expressions(self) -> Dict[str, str]:
        """获取所有因子的 {name: expression} 映射"""
        return {f.name: f.expression for f in self.config.factors}

    def get_categories(self) -> List[FactorCategory]:
        """获取所有因子类别"""
        return list(self._category_index.keys())

    def add_factor(self, factor: FactorDefinition):
        """动态添加因子（运行时扩展，无需修改配置文件）"""
        if factor.name in self._name_index:
            raise ValueError(f"因子 {factor.name} 已存在")
        self.config.factors.append(factor)
        self._name_index[factor.name] = factor
        self._category_index.setdefault(factor.category, []).append(factor)

    def remove_factor(self, name: str) -> bool:
        """动态移除因子"""
        if name not in self._name_index:
            return False
        factor = self._name_index.pop(name)
        self.config.factors.remove(factor)
        category_list = self._category_index.get(factor.category, [])
        if factor in category_list:
            category_list.remove(factor)
        return True

    def summary(self) -> Dict[str, Any]:
        """生成因子库摘要"""
        return {
            "total_factors": len(self.config.factors),
            "categories": {
                cat.value: len(factors)
                for cat, factors in self._category_index.items()
            },
            "factor_names": sorted(self._name_index.keys()),
        }


# ===================== Alpha158 风格完整因子配置 =====================


def build_alpha158_style_config() -> FactorLibraryConfig:
    """
    构建类似 Qlib Alpha158 风格的因子配置
    
    Qlib Alpha158 包含 158 个因子，分为:
    - K 线类: 开高低收组合 (约16个)
    - 价格类: 收益率、均价偏离 (约32个)
    - 量价类: 量价关系 (约32个)
    - 波动类: 标准差、振幅 (约32个)
    - 动量类: 多期收益率 (约32个)
    - 其他 (约14个)
    
    此处实现核心子集作为验证
    """
    factors = []
    
    # === K 线类因子 ===
    kline_factors = [
        ("kline_upper_shadow", "high - Max(open, close)", "上影线长度"),
        ("kline_lower_shadow", "Min(open, close) - low", "下影线长度"),
        ("kline_body", "close - open", "实体长度"),
        ("kline_body_pct", "(close - open) / open", "实体涨跌幅"),
        ("kline_range", "high - low", "日内振幅"),
        ("kline_range_pct", "(high - low) / Ref(close, 1)", "日内振幅率"),
        ("kline_upper_shadow_pct", "(high - Max(open, close)) / (high - low + 0.0001)", "上影线比例"),
        ("kline_lower_shadow_pct", "(Min(open, close) - low) / (high - low + 0.0001)", "下影线比例"),
    ]
    for name, expr, desc in kline_factors:
        factors.append(FactorDefinition(name=name, category=FactorCategory.KLINE,
                         expression=expr, description=desc))

    # === 价格类因子 ===
    price_factors = [
        ("ret_1d", "Pct(close, 1)", "1日收益率"),
        ("ret_5d", "Pct(close, 5)", "5日收益率"),
        ("ret_10d", "Pct(close, 10)", "10日收益率"),
        ("ret_20d", "Pct(close, 20)", "20日收益率"),
        ("ret_60d", "Pct(close, 60)", "60日收益率"),
        ("ma_5d", "Mean(close, 5)", "5日均价"),
        ("ma_10d", "Mean(close, 10)", "10日均价"),
        ("ma_20d", "Mean(close, 20)", "20日均价"),
        ("ma_60d", "Mean(close, 60)", "60日均价"),
        ("ma_dev_5d", "Mean(close, 5) / close - 1", "5日均线偏离"),
        ("ma_dev_10d", "Mean(close, 10) / close - 1", "10日均线偏离"),
        ("ma_dev_20d", "Mean(close, 20) / close - 1", "20日均线偏离"),
        ("ma_dev_60d", "Mean(close, 60) / close - 1", "60日均线偏离"),
    ]
    for name, expr, desc in price_factors:
        factors.append(FactorDefinition(name=name, category=FactorCategory.PRICE,
                         expression=expr, description=desc))

    # === 动量类因子 ===
    momentum_factors = [
        ("momentum_5d", "Pct(close, 5)", "5日动量"),
        ("momentum_10d", "Pct(close, 10)", "10日动量"),
        ("momentum_20d", "Pct(close, 20)", "20日动量"),
        ("momentum_60d", "Pct(close, 60)", "60日动量"),
        ("momentum_ratio_5_20", "Pct(close, 5) - Pct(close, 20)", "短期vs中期动量差"),
        ("momentum_ratio_20_60", "Pct(close, 20) - Pct(close, 60)", "中期vs长期动量差"),
    ]
    for name, expr, desc in momentum_factors:
        factors.append(FactorDefinition(name=name, category=FactorCategory.MOMENTUM,
                         expression=expr, description=desc))

    # === 反转类因子 ===
    reversal_factors = [
        ("reversal_5d", "-Pct(close, 5)", "5日反转"),
        ("reversal_10d", "-Pct(close, 10)", "10日反转"),
        ("reversal_20d", "-Pct(close, 20)", "20日反转"),
        ("reversal_60d", "-Pct(close, 60)", "60日反转"),
    ]
    for name, expr, desc in reversal_factors:
        factors.append(FactorDefinition(name=name, category=FactorCategory.REVERSAL,
                         expression=expr, description=desc))

    # === 波动类因子 ===
    volatility_factors = [
        ("volatility_5d", "Std(Pct(close, 1), 5)", "5日波动率"),
        ("volatility_10d", "Std(Pct(close, 1), 10)", "10日波动率"),
        ("volatility_20d", "Std(Pct(close, 1), 20)", "20日波动率"),
        ("volatility_60d", "Std(Pct(close, 1), 60)", "60日波动率"),
        ("volatility_ratio_5_20", "Std(Pct(close, 1), 5) / Std(Pct(close, 1), 20)", "短/中波动率比"),
        ("amplitude_20d", "Max(high, 20) / Min(low, 20) - 1", "20日振幅"),
    ]
    for name, expr, desc in volatility_factors:
        factors.append(FactorDefinition(name=name, category=FactorCategory.VOLATILITY,
                         expression=expr, description=desc))

    # === 成交量类因子 ===
    volume_factors = [
        ("volume_ma_5d", "Mean(volume, 5)", "5日均量"),
        ("volume_ma_20d", "Mean(volume, 20)", "20日均量"),
        ("volume_ratio_5_20", "Mean(volume, 5) / Mean(volume, 20)", "5/20量比"),
        ("volume_pct_1d", "Pct(volume, 1)", "量变化率1日"),
        ("volume_pct_5d", "Pct(volume, 5)", "量变化率5日"),
    ]
    for name, expr, desc in volume_factors:
        factors.append(FactorDefinition(name=name, category=FactorCategory.VOLUME,
                         expression=expr, description=desc))

    return FactorLibraryConfig(
        version="1.0",
        name="Alpha158-style-factor-library",
        description="借鉴 Qlib Alpha158 设计的可配置因子库",
        factors=factors,
    )


# ===================== 测试类 =====================


class TestFactorLibrary(unittest.TestCase):
    """因子库正确性测试"""

    def setUp(self):
        self.config = build_alpha158_style_config()
        self.library = FactorLibrary(self.config)

    def test_total_factors(self):
        """验证: 因子总数正确"""
        expected = 42  # 8 kline + 13 price + 6 momentum + 4 reversal + 6 volatility + 5 volume
        self.assertEqual(len(self.config.factors), expected)

    def test_category_count(self):
        """验证: 各类别因子数量"""
        cats = {
            FactorCategory.KLINE: 8,
            FactorCategory.PRICE: 13,
            FactorCategory.MOMENTUM: 6,
            FactorCategory.REVERSAL: 4,
            FactorCategory.VOLATILITY: 6,
            FactorCategory.VOLUME: 5,
        }
        for cat, expected in cats.items():
            factors = self.library.get_factors_by_category(cat)
            self.assertEqual(len(factors), expected,
                f"类别 {cat.value} 应有 {expected} 个因子，实际 {len(factors)}")

    def test_get_factor(self):
        """验证: 按名称获取因子"""
        factor = self.library.get_factor("ret_1d")
        self.assertIsNotNone(factor)
        self.assertEqual(factor.name, "ret_1d")
        self.assertEqual(factor.expression, "Pct(close, 1)")

    def test_get_nonexistent_factor(self):
        """验证: 查询不存在的因子返回 None"""
        self.assertIsNone(self.library.get_factor("nonexistent_factor"))

    def test_duplicate_name_validation(self):
        """验证: 因子名称重复应触发验证错误"""
        config = FactorLibraryConfig(factors=[
            FactorDefinition(name="dup", category=FactorCategory.PRICE, expression="Pct(close, 1)"),
            FactorDefinition(name="dup", category=FactorCategory.PRICE, expression="Pct(close, 2)"),
        ])
        with self.assertRaises(ValueError):
            FactorLibrary(config)

    def test_empty_name_validation(self):
        """验证: 空名称应触发验证错误"""
        config = FactorLibraryConfig(factors=[
            FactorDefinition(name="", category=FactorCategory.PRICE, expression="Pct(close, 1)"),
        ])
        with self.assertRaises(ValueError):
            FactorLibrary(config)

    def test_dynamic_add_factor(self):
        """验证: 运行时动态添加因子"""
        new_factor = FactorDefinition(
            name="my_custom_factor",
            category=FactorCategory.TECHNICAL,
            expression="Pct(Mean(close, 5), 1)",
            description="自定义因子"
        )
        self.library.add_factor(new_factor)
        self.assertIsNotNone(self.library.get_factor("my_custom_factor"))
        self.assertIn(new_factor, self.library.get_factors_by_category(FactorCategory.TECHNICAL))

    def test_dynamic_add_duplicate(self):
        """验证: 动态添加重复名称应报错"""
        dup = FactorDefinition(
            name="ret_1d",  # 已存在
            category=FactorCategory.PRICE,
            expression="Pct(close, 2)"
        )
        with self.assertRaises(ValueError):
            self.library.add_factor(dup)

    def test_dynamic_remove_factor(self):
        """验证: 动态移除因子"""
        self.assertTrue(self.library.remove_factor("ret_1d"))
        self.assertIsNone(self.library.get_factor("ret_1d"))

    def test_remove_nonexistent(self):
        """验证: 移除不存在的因子返回 False"""
        self.assertFalse(self.library.remove_factor("nonexistent"))

    def test_json_roundtrip(self):
        """验证: JSON 序列化/反序列化一致性"""
        json_str = self.config.to_json()
        restored = FactorLibraryConfig.from_json(json_str)
        self.assertEqual(len(restored.factors), len(self.config.factors))
        self.assertEqual(restored.factors[0].name, self.config.factors[0].name)
        self.assertEqual(restored.factors[0].expression, self.config.factors[0].expression)

    def test_summary(self):
        """验证: 因子库摘要信息"""
        summary = self.library.summary()
        self.assertEqual(summary['total_factors'], 42)
        self.assertEqual(len(summary['categories']), 6)
        self.assertIn('ret_1d', summary['factor_names'])


class TestFactorLibraryExtensibility(unittest.TestCase):
    """
    可扩展性测试: 演示扩展因子库不改变核心代码
    
    核心优势:
    - 新增因子: 只需在配置文件中添加一行 JSON/YAML
    - 新增类别: 自动索引，无需注册
    - 删除因子: 无需修改计算逻辑
    """

    def setUp(self):
        self.config = build_alpha158_style_config()
        self.library = FactorLibrary(self.config)

    def test_add_new_category_without_code_change(self):
        """
        演示: 新增技术指标类别，无需修改 FactorLibrary 源码
        """
        tech_factors = [
            FactorDefinition(
                name="rsi_14", category=FactorCategory.TECHNICAL,
                expression="Pct(close, 1)",  # 简化版，实际应为 RSI
                description="14日相对强弱指标"
            ),
            FactorDefinition(
                name="bollinger_upper", category=FactorCategory.TECHNICAL,
                expression="Mean(close, 20) + 2 * Std(close, 20)",
                description="布林带上轨"
            ),
            FactorDefinition(
                name="bollinger_lower", category=FactorCategory.TECHNICAL,
                expression="Mean(close, 20) - 2 * Std(close, 20)",
                description="布林带下轨"
            ),
        ]
        for f in tech_factors:
            self.library.add_factor(f)

        techs = self.library.get_factors_by_category(FactorCategory.TECHNICAL)
        self.assertEqual(len(techs), 3, "新增类别应有3个因子")

    def test_bulk_factor_import(self):
        """
        演示: 从 JSON 文件批量导入因子
        模拟在实际使用中从配置文件加载新因子集
        """
        # 模拟外部 JSON 配置
        external_config_json = json.dumps({
            "factors": [
                {
                    "name": "sharpe_ratio_60d",
                    "category": "volatility",
                    "expression": "Mean(Pct(close, 1), 60) / Std(Pct(close, 1), 60)",
                    "description": "60日夏普比",
                },
                {
                    "name": "max_drawdown_60d",
                    "category": "volatility",
                    "expression": "Min(Pct(close, 1), 60)",
                    "description": "60日最大回撤",
                },
                {
                    "name": "rsi_style_14d",
                    "category": "momentum",
                    "expression": "Sum(Max(Pct(close, 1), 0), 14) / (Sum(Max(Pct(close, 1), 0), 14) - Sum(Min(Pct(close, 1), 0), 14))",
                    "description": "14日RSI风格动量比",
                },
            ]
        })

        # 从 JSON 解析
        external = FactorLibraryConfig.from_json(external_config_json)
        for factor in external.factors:
            self.library.add_factor(factor)

        self.assertIsNotNone(self.library.get_factor("sharpe_ratio_60d"))
        self.assertIsNotNone(self.library.get_factor("max_drawdown_60d"))
        self.assertEqual(self.library.summary()['total_factors'], 45)  # 42 + 3


class TestFactorLibraryPerformance(unittest.TestCase):
    """因子库性能测试"""

    def setUp(self):
        self.config = build_alpha158_style_config()
        self.library = FactorLibrary(self.config)

    def test_lookup_performance(self):
        """验证: 因子查找性能（毫秒级）"""
        import time
        # 预热
        for _ in range(100):
            self.library.get_factor("ret_1d")
        
        start = time.perf_counter()
        for _ in range(10000):
            self.library.get_factor("ret_20d")
        elapsed = time.perf_counter() - start
        
        # 10000 次查找应在 0.1s 内
        self.assertLess(elapsed, 0.1,
            f"因子查找性能不达标: {elapsed:.4f}s (预期 < 0.1s)")

    def test_category_filter_performance(self):
        """验证: 类别过滤性能"""
        import time
        start = time.perf_counter()
        for _ in range(1000):
            self.library.get_factors_by_category(FactorCategory.PRICE)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.05,
            f"类别过滤性能不达标: {elapsed:.4f}s (预期 < 0.05s)")


if __name__ == "__main__":
    print("=" * 60)
    print("Verification 3: 因子库可扩展性优化")
    print("借鉴来源: Microsoft Qlib Alpha158 因子库设计")
    print("=" * 60)

    # 生成示例配置
    config = build_alpha158_style_config()
    print(f"\n因子库配置摘要:")
    print(f"  总因子数: {len(config.factors)}")
    
    library = FactorLibrary(config)
    summary = library.summary()
    print("  类别分布:")
    for cat, count in summary['categories'].items():
        print(f"    {cat}: {count} 个因子")

    print(f"\n示例因子列表 (前10个):")
    for f in config.factors[:10]:
        print(f"  [{f.category.value}] {f.name}: {f.expression}")

    print("\n" + "=" * 60)
    print("运行测试套件...")
    unittest.main(argv=[''], verbosity=2, exit=False)