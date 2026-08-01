"""
财务因子计算模块

从 financial_data.parquet 提取财务因子，映射到标准因子列。
"""
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict

logger = logging.getLogger("financial_factors")


def compute(price_data: pd.DataFrame, ctx=None) -> pd.DataFrame:
    """
    计算财务因子

    参数:
        price_data: OHLCV 日线数据（用于对齐 code/date）
        ctx: Context 对象，需包含 artifacts['FINANCIAL']

    返回:
        DataFrame，列为 code, date, [各财务因子]
    """
    if ctx is None:
        return pd.DataFrame()

    financial_path = ctx.get_artifact("FINANCIAL") if hasattr(ctx, 'get_artifact') else None
    if not financial_path or not os.path.exists(financial_path):
        logger.warning("财务数据产物不存在，跳过财务因子计算")
        return pd.DataFrame()

    logger.info("开始计算财务因子...")
    fin_df = pd.read_parquet(financial_path)
    if fin_df.empty:
        return pd.DataFrame()

    # 取每只股票最新一行的财务数据
    if 'code' in fin_df.columns:
        latest_fin = fin_df.sort_values('report_date', ascending=False).drop_duplicates(subset='code', keep='first')
    else:
        latest_fin = fin_df.iloc[[-1]]

    # 对齐到 price_data 的 code/date
    result = price_data[['code', 'date']].copy()

    # 字段映射：financial_data 标准字段 → 因子列名
    field_map = {
        'roe': 'roe_ttm',
        'roa': 'roa_ttm',
        'gross_margin': 'gross_margin_ttm',
        'net_margin': 'net_margin_ttm',
        'revenue_growth': 'revenue_growth_yoy',
        'profit_growth': 'profit_growth_yoy',
        'debt_ratio': 'debt_ratio',
        'current_ratio': 'current_ratio',
        'ocf': 'ocf',
    }

    for src_col, dst_col in field_map.items():
        if src_col in latest_fin.columns:
            # 将财务数据 merge 到每个交易日（前向填充）
            mapping = latest_fin[['code', src_col]].set_index('code')[src_col].to_dict()
            result[dst_col] = result['code'].map(mapping)
        else:
            result[dst_col] = np.nan

    logger.info(f"财务因子计算完成，共 {len(field_map)} 个因子")
    return result
