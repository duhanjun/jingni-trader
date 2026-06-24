"""
模块化成本模型（借鉴 LEAN / QuantConnect 的可插拔架构）

jingni-trader 现有回测引擎将佣金/滑点/税费硬编码为标量参数，
难以灵活测试不同成本假设。本模块参考 LEAN 的 FillModel / SlippageModel /
FeeModel 分离设计，提供可插拔的成本模型，便于：
  - 测试不同滑点假设对策略收益的影响
  - 模拟成交量相关的市场冲击成本
  - 统一成本计算逻辑，避免回测与实盘不一致

借鉴来源: QuantConnect LEAN - SlippageModel / FeeModel / BrokerageModel
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TradeContext:
    """单笔交易的上下文信息"""
    price: float          # 原始价格
    shares: float         # 成交股数
    side: str  # 'buy' or 'sell'
    amount: Optional[float] = None  # 成交额（= price * shares）
    volume: Optional[float] = None  # 当日成交量（用于量价滑点）


class SlippageModel(ABC):
    """滑点模型基类（借鉴 LEAN ISlippageModel）"""

    @abstractmethod
    def apply(self, ctx: TradeContext) -> float:
        """返回考虑滑点后的实际成交价"""
        ...


class ConstantSlippage(SlippageModel):
    """固定比例滑点（jingni-trader 现有行为，向后兼容）"""

    def __init__(self, slippage_rate: float = 0.001):
        self.slippage_rate = slippage_rate

    def apply(self, ctx: TradeContext) -> float:
        if ctx.side == 'buy':
            return ctx.price * (1 + self.slippage_rate)
        else:
            return ctx.price * (1 - self.slippage_rate)


class VolumeShareSlippage(SlippageModel):
    """
    成交量占比滑点模型（借鉴 LEAN VolumeShareSlippageModel）

    当订单占当日成交量比例越高，市场冲击越大：
        impact = volume_share ** 2 * price_impact
        实际价格 = 原价 * (1 +/- impact)

    这更贴近真实市场：大单吃流动性的成本远高于小单。
    """

    def __init__(self, price_impact: float = 0.1, max_volume_share: float = 0.25):
        self.price_impact = price_impact
        self.max_volume_share = max_volume_share

    def apply(self, ctx: TradeContext) -> float:
        if ctx.volume is None or ctx.volume <= 0:
            # 无成交量信息时退化为微小固定滑点
            return ctx.price * (1 + 0.0005) if ctx.side == 'buy' else ctx.price * (1 - 0.0005)

        volume_share = min(ctx.shares / ctx.volume, self.max_volume_share)
        impact = volume_share ** 2 * self.price_impact
        if ctx.side == 'buy':
            return ctx.price * (1 + impact)
        else:
            return ctx.price * (1 - impact)


class FeeModel(ABC):
    """手续费模型基类（借鉴 LEAN IFeeModel）"""

    @abstractmethod
    def calculate(self, ctx: TradeContext) -> dict:
        """返回 {'commission': float, 'tax': float}"""
        ...


class AShareFeeModel(FeeModel):
    """
    A股标准费用模型

    - 佣金: max(amount * commission_rate, min_commission)
    - 印花税: 仅卖出收取（2023.08 起减半至 0.05%，这里参数化）
    - 过户费: 沪市双向 0.001%（简化为统一收取）
    """

    def __init__(
        self,
        commission_rate: float = 0.00025,
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.0005,
        transfer_fee_rate: float = 0.00001,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate

    def calculate(self, ctx: TradeContext) -> dict:
        amount = ctx.amount if ctx.amount is not None else ctx.price * ctx.shares
        commission = max(amount * self.commission_rate, self.min_commission)
        tax = amount * self.stamp_tax_rate if ctx.side == 'sell' else 0.0
        transfer = amount * self.transfer_fee_rate
        return {"commission": commission, "tax": tax, "transfer_fee": transfer}


class CostCalculator:
    """
    统一成本计算器，组合滑点模型与费用模型

    用法:
        calc = CostCalculator(
            slippage_model=VolumeShareSlippage(price_impact=0.1),
            fee_model=AShareFeeModel(stamp_tax_rate=0.0005),
        )
        fill_price, fees = calc.compute(TradeContext(...))
    """

    def __init__(self, slippage_model: SlippageModel, fee_model: FeeModel):
        self.slippage_model = slippage_model
        self.fee_model = fee_model

    def compute(self, ctx: TradeContext) -> tuple:
        """
        返回 (fill_price, fees_dict)

        fill_price: 考虑滑点后的实际成交价
        fees_dict: {'commission', 'tax', 'transfer_fee'}
        """
        fill_price = self.slippage_model.apply(ctx)
        # 费用基于滑点后的成交额计算
        ctx_with_slippage = TradeContext(
            price=fill_price,
            shares=ctx.shares,
            side=ctx.side,
            amount=fill_price * ctx.shares,
            volume=ctx.volume,
        )
        fees = self.fee_model.calculate(ctx_with_slippage)
        return fill_price, fees
