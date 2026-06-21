"""
因子注册表与元数据体系

借鉴来源:
- Microsoft Qlib Alpha158: 结构化因子库，每个因子带方向/类别/描述元数据
- QuantaAlpha (arXiv:2602.07085): 因子池维护需 Rank IC、低冗余、容量三重门槛

设计目标:
- 替代 jingni-trader factor-engine 中硬编码的因子列表
- 提供因子方向(direction)、类别(category)、计算参数等元信息
- 支持因子自动发现与注册，便于扩展
- 为多因子融合提供方向调整依据(正向/反向因子)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import logging

logger = logging.getLogger("factor-registry")


class FactorDirection(int):
    """因子方向: +1 正向(值越大收益越高), -1 反向(值越大收益越低)"""
    POSITIVE = 1
    NEGATIVE = -1


class FactorCategory(str):
    """因子类别"""
    MOMENTUM = "momentum"          # 动量/反转
    SIZE = "size"                  # 规模
    LIQUIDITY = "liquidity"        # 流动性/换手
    VOLATILITY = "volatility"      # 波动率
    MONEY_FLOW = "money_flow"      # 资金流
    QUALITY = "quality"            # 质量
    VALUE = "value"                # 价值
    GROWTH = "growth"              # 成长
    TECHNICAL = "technical"        # 技术指标
    CUSTOM = "custom"              # 自定义


@dataclass
class FactorMeta:
    """因子元数据"""
    name: str                                   # 因子名
    direction: int = FactorDirection.POSITIVE   # 方向 +1/-1
    category: str = FactorCategory.CUSTOM       # 类别
    description: str = ""                       # 文字说明
    params: Dict[str, Any] = field(default_factory=dict)  # 计算参数
    expected_ic_sign: Optional[int] = None      # 预期IC符号(用于校验)
    min_obs: int = 30                           # 最小样本数
    decay_horizon: int = 5                      # 预测衰减周期(天)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FactorRegistry:
    """
    因子注册表

    用法:
        registry = FactorRegistry()
        registry.register(FactorMeta(
            name="reversal_20d",
            direction=FactorDirection.NEGATIVE,
            category=FactorCategory.MOMENTUM,
            description="20日反转因子",
            expected_ic_sign=-1,
        ))
        meta = registry.get("reversal_20d")
        all_momentum = registry.list_by_category(FactorCategory.MOMENTUM)
    """

    def __init__(self):
        self._factors: Dict[str, FactorMeta] = {}

    def register(self, meta: FactorMeta) -> None:
        if meta.name in self._factors:
            logger.warning(f"因子 {meta.name} 已存在，覆盖注册")
        self._factors[meta.name] = meta
        logger.debug(f"注册因子: {meta.name} ({meta.category}, dir={meta.direction})")

    def get(self, name: str) -> Optional[FactorMeta]:
        return self._factors.get(name)

    def list_all(self) -> List[FactorMeta]:
        return list(self._factors.values())

    def list_names(self) -> List[str]:
        return list(self._factors.keys())

    def list_by_category(self, category: str) -> List[FactorMeta]:
        return [m for m in self._factors.values() if m.category == category]

    def list_by_direction(self, direction: int) -> List[FactorMeta]:
        return [m for m in self._factors.values() if m.direction == direction]

    def remove(self, name: str) -> bool:
        if name in self._factors:
            del self._factors[name]
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {name: meta.to_dict() for name, meta in self._factors.items()}

    def adjust_direction(self, factor_values: Dict[str, "pd.Series"]) -> Dict[str, "pd.Series"]:
        """
        根据注册的方向元数据调整因子值符号
        反向因子取负，使所有因子统一为"值越大预期收益越高"

        参数:
            factor_values: {因子名: 值序列}

        返回:
            调整方向后的因子值字典
        """
        adjusted = {}
        for name, series in factor_values.items():
            meta = self.get(name)
            if meta is None:
                adjusted[name] = series
                continue
            if meta.direction == FactorDirection.NEGATIVE:
                adjusted[name] = -series
            else:
                adjusted[name] = series
        return adjusted

    def validate_ic_sign(self, factor_name: str, ic_mean: float, tolerance: float = 0.0) -> bool:
        """
        校验实际IC符号是否与预期一致

        参数:
            factor_name: 因子名
            ic_mean: 实际IC均值
            tolerance: 容差(0表示严格同号)

        返回:
            True 表示符号一致或无预期
        """
        meta = self.get(factor_name)
        if meta is None or meta.expected_ic_sign is None:
            return True
        if tolerance > 0 and abs(ic_mean) < tolerance:
            return True
        return (ic_mean > 0) == (meta.expected_ic_sign > 0)


def build_default_a_share_registry() -> FactorRegistry:
    """
    构建 A 股默认因子注册表

    覆盖 jingni-trader factor-engine.compute_a_share_factors 中的全部因子，
    并补充方向/类别/预期IC符号等元数据。
    """
    reg = FactorRegistry()

    # 动量/反转类 (A股短期反转效应显著)
    reg.register(FactorMeta(
        name="reversal_5d", direction=FactorDirection.POSITIVE,
        category=FactorCategory.MOMENTUM,
        description="5日反转因子(已取负，正向IC预期)",
        params={"window": 5}, expected_ic_sign=1, decay_horizon=5,
    ))
    reg.register(FactorMeta(
        name="reversal_20d", direction=FactorDirection.POSITIVE,
        category=FactorCategory.MOMENTUM,
        description="20日反转因子(已取负，正向IC预期)",
        params={"window": 20}, expected_ic_sign=1, decay_horizon=10,
    ))
    reg.register(FactorMeta(
        name="ret_60d", direction=FactorDirection.NEGATIVE,
        category=FactorCategory.MOMENTUM,
        description="60日动量(A股长期动量反转，预期负IC)",
        params={"window": 60}, expected_ic_sign=-1, decay_horizon=20,
    ))

    # 规模类
    reg.register(FactorMeta(
        name="lncap", direction=FactorDirection.NEGATIVE,
        category=FactorCategory.SIZE,
        description="对数市值(A股小盘溢价，预期负IC)",
        expected_ic_sign=-1, decay_horizon=20,
    ))

    # 流动性/换手类
    reg.register(FactorMeta(
        name="turnover_20d", direction=FactorDirection.NEGATIVE,
        category=FactorCategory.LIQUIDITY,
        description="20日平均换手率(高换手往往低收益)",
        params={"window": 20}, expected_ic_sign=-1, decay_horizon=5,
    ))
    reg.register(FactorMeta(
        name="turnover_change", direction=FactorDirection.NEGATIVE,
        category=FactorCategory.LIQUIDITY,
        description="短期换手率相对变化",
        expected_ic_sign=-1, decay_horizon=5,
    ))
    reg.register(FactorMeta(
        name="volume_ratio", direction=FactorDirection.NEGATIVE,
        category=FactorCategory.LIQUIDITY,
        description="量比(放量往往见顶)",
        expected_ic_sign=-1, decay_horizon=1,
    ))

    # 波动率类
    reg.register(FactorMeta(
        name="volatility_20d", direction=FactorDirection.NEGATIVE,
        category=FactorCategory.VOLATILITY,
        description="20日波动率(低波动异象)",
        params={"window": 20}, expected_ic_sign=-1, decay_horizon=10,
    ))

    # 资金流类
    reg.register(FactorMeta(
        name="money_flow_20d", direction=FactorDirection.POSITIVE,
        category=FactorCategory.MONEY_FLOW,
        description="20日累计资金流",
        params={"window": 20}, expected_ic_sign=1, decay_horizon=5,
    ))

    return reg
