"""
股东结构因子计算模块

从 shareholder_data.parquet 提取股东结构因子。
"""
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict

logger = logging.getLogger("shareholder_factors")


def compute(price_data: pd.DataFrame, ctx=None) -> pd.DataFrame:
    """
    计算股东结构因子

    参数:
        price_data: OHLCV 日线数据（用于对齐 code/date）
        ctx: Context 对象，需包含 artifacts['SHAREHOLDER']

    返回:
        DataFrame，列为 code, date, [各股东结构因子]
    """
    if ctx is None:
        return pd.DataFrame()

    sh_path = ctx.get_artifact("SHAREHOLDER") if hasattr(ctx, 'get_artifact') else None
    if not sh_path or not os.path.exists(sh_path):
        logger.warning("股东结构数据产物不存在，跳过股东结构因子计算")
        return pd.DataFrame()

    logger.info("开始计算股东结构因子...")
    sh_df = pd.read_parquet(sh_path)
    if sh_df.empty:
        return pd.DataFrame()

    result = price_data[['code', 'date']].copy()

    if 'code' in sh_df.columns:
        # 十大股东变动：统计增减方向
        if 'change_type' in sh_df.columns:
            # 将变动方向映射为数值：增持=1, 减持=-1, 不变=0
            def _parse_change(val):
                val_str = str(val) if pd.notna(val) else ""
                if "增" in val_str:
                    return 1
                elif "减" in val_str:
                    return -1
                return 0
            sh_df['change_numeric'] = sh_df['change_type'].apply(_parse_change)
            holder_change = sh_df.groupby('code')['change_numeric'].sum().to_dict()
            result['top_holder_change'] = result['code'].map(holder_change).fillna(0).astype(int)
        else:
            result['top_holder_change'] = 0

        # 持股集中度：十大股东持股比例之和
        if 'hold_ratio' in sh_df.columns:
            concentration = sh_df.groupby('code')['hold_ratio'].sum().to_dict()
            result['holder_concentration'] = result['code'].map(concentration).fillna(0.0).astype(float)
        else:
            result['holder_concentration'] = 0.0
    else:
        result['top_holder_change'] = 0
        result['holder_concentration'] = 0.0

    logger.info("股东结构因子计算完成，共 2 个因子")
    return result
