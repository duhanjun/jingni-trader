"""
龙虎榜因子计算模块

从 dragon_tiger.parquet 提取龙虎榜因子。
"""
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict

logger = logging.getLogger("dragon_tiger_factors")


def compute(price_data: pd.DataFrame, ctx=None) -> pd.DataFrame:
    """
    计算龙虎榜因子

    参数:
        price_data: OHLCV 日线数据（用于对齐 code/date）
        ctx: Context 对象，需包含 artifacts['DRAGON_TIGER']

    返回:
        DataFrame，列为 code, date, [各龙虎榜因子]
    """
    if ctx is None:
        return pd.DataFrame()

    dt_path = ctx.get_artifact("DRAGON_TIGER") if hasattr(ctx, 'get_artifact') else None
    if not dt_path or not os.path.exists(dt_path):
        logger.warning("龙虎榜数据产物不存在，跳过龙虎榜因子计算")
        return pd.DataFrame()

    logger.info("开始计算龙虎榜因子...")
    dt_df = pd.read_parquet(dt_path)
    if dt_df.empty:
        return pd.DataFrame()

    result = price_data[['code', 'date']].copy()

    # 龙虎榜数据通常是近5日汇总，取最新值映射到所有交易日
    if 'code' in dt_df.columns:
        # 上榜次数
        if 'has_data' in dt_df.columns:
            lhb_count = dt_df[dt_df['has_data'] == True].groupby('code').size().to_dict()
            result['lhb_count_5d'] = result['code'].map(lhb_count).fillna(0).astype(int)
        else:
            result['lhb_count_5d'] = 0

        # 净买入额
        if 'net_buy' in dt_df.columns:
            net_buy = dt_df.groupby('code')['net_buy'].sum().to_dict()
            result['lhb_net_buy'] = result['code'].map(net_buy).fillna(0.0).astype(float)
        else:
            result['lhb_net_buy'] = 0.0
    else:
        result['lhb_count_5d'] = 0
        result['lhb_net_buy'] = 0.0

    logger.info("龙虎榜因子计算完成，共 2 个因子")
    return result
