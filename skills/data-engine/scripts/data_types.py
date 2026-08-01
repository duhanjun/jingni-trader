"""
数据类型注册表

定义所有数据类型及其元数据，用于按数据类型粒度的独立降级。
每种数据类型独立遍历优先级链，互不影响。
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class DataTypeMeta:
    """数据类型元数据"""
    name: str                    # 数据类型标识：daily/financial/capital_flow/...
    display_name: str            # 中文名
    method_name: str             # 适配器方法名：get_daily/get_financial/get_capital_flow/...
    artifact_filename: str       # 落盘文件名
    artifact_key: str            # ctx.artifacts 中的键名
    allow_synthetic: bool        # 是否允许模拟数据兜底
    required: bool               # 是否为管线必需（缺失则报错）
    description: str             # 描述


# 所有数据类型注册
DATA_TYPES: Dict[str, DataTypeMeta] = {
    "daily": DataTypeMeta(
        name="daily",
        display_name="日线行情",
        method_name="get_daily",
        artifact_filename="cleaned_data.parquet",
        artifact_key="DATA",
        allow_synthetic=True,
        required=True,
        description="OHLCV 日线行情，管线必需数据",
    ),
    "financial": DataTypeMeta(
        name="financial",
        display_name="财务数据",
        method_name="get_financial",
        artifact_filename="financial_data.parquet",
        artifact_key="FINANCIAL",
        allow_synthetic=False,
        required=False,
        description="PE/PB/ROE/毛利率等财务指标",
    ),
    "capital_flow": DataTypeMeta(
        name="capital_flow",
        display_name="资金面数据",
        method_name="get_capital_flow",
        artifact_filename="capital_flow.parquet",
        artifact_key="CAPITAL_FLOW",
        allow_synthetic=False,
        required=False,
        description="主力资金流向、北向资金",
    ),
    "dragon_tiger": DataTypeMeta(
        name="dragon_tiger",
        display_name="龙虎榜数据",
        method_name="get_dragon_tiger",
        artifact_filename="dragon_tiger.parquet",
        artifact_key="DRAGON_TIGER",
        allow_synthetic=False,
        required=False,
        description="龙虎榜上榜明细",
    ),
    "shareholder": DataTypeMeta(
        name="shareholder",
        display_name="股东结构数据",
        method_name="get_shareholder",
        artifact_filename="shareholder_data.parquet",
        artifact_key="SHAREHOLDER",
        allow_synthetic=False,
        required=False,
        description="十大股东、持股变动",
    ),
}
