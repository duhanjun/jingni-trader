"""
扩展因子库
借鉴来源: Qlib Alpha158 标准化因子体系
包含 6 大类 47 个因子，按动量、反转、波动率、成交量、技术指标、资金流向分类
"""
from .alphafactors import Alpha158FactorEngine

__all__ = ["Alpha158FactorEngine"]