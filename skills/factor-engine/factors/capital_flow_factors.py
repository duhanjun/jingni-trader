"""
资金面因子计算模块

从 capital_flow.parquet 提取资金面因子。
"""
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict

logger = logging.getLogger("capital_flow_factors")


def compute(price_data: pd.DataFrame, ctx=None) -> pd.DataFrame:
    """
    计算资金面因子

    参数:
        price_data: OHLCV 日线数据（用于对齐 code/date）
        ctx: Context 对象，需包含 artifacts['CAPITAL_FLOW']

    返回:
        DataFrame，列为 code, date, [各资金面因子]
    """
    if ctx is None:
        return pd.DataFrame()

    cf_path = ctx.get_artifact("CAPITAL_FLOW") if hasattr(ctx, 'get_artifact') else None
    if not cf_path or not os.path.exists(cf_path):
        logger.warning("资金面数据产物不存在，跳过资金面因子计算")
        return pd.DataFrame()

    logger.info("开始计算资金面因子...")
    cf_df = pd.read_parquet(cf_path)
    if cf_df.empty:
        return pd.DataFrame()

    result = price_data[['code', 'date']].copy()

    # 确保有 date 列用于对齐
    if 'date' not in cf_df.columns:
        logger.warning("资金面数据无 date 列，无法对齐")
        return pd.DataFrame()

    cf_df['date'] = pd.to_datetime(cf_df['date']).dt.strftime('%Y-%m-%d')
    result['date_str'] = result['date'].astype(str).str[:10]

    # 直接映射字段
    field_map = {
        'main_net_inflow': 'main_net_inflow',
        'main_net_inflow_5d': 'main_net_inflow_5d',
        'north_net_inflow': 'north_net_inflow',
    }

    for src_col, dst_col in field_map.items():
        if src_col in cf_df.columns:
            merge_df = cf_df[['code', 'date', src_col]].copy()
            merge_df.rename(columns={src_col: dst_col, 'date': 'date_str'}, inplace=True)
            result = result.merge(merge_df, on=['code', 'date_str'], how='left')
            result[dst_col] = result[dst_col].astype(float)
        else:
            result[dst_col] = np.nan

    result = result.drop(columns=['date_str'], errors='ignore')

    logger.info(f"资金面因子计算完成，共 {len(field_map)} 个因子")
    return result
