"""
LLM Agent 因子挖掘模块
借鉴来源: RD-Agent-Quant (Microsoft, NeurIPS 2025)
Research → Develop → Feedback 闭环自动因子发现
"""
from .miner import FactorDiscoveryAgent

__all__ = ["FactorDiscoveryAgent"]