"""
估值因子计算模块

从 financial_data.parquet 提取估值因子，并计算历史分位。

P0-1 PIT 契约：读取财务数据后强制走 pit_filter，防止 look-ahead bias。
"""
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict

logger = logging.getLogger("valuation_factors")


def compute(price_data: pd.DataFrame, ctx=None) -> pd.DataFrame:
    """
    计算估值因子

    参数:
        price_data: OHLCV 日线数据（用于对齐 code/date 和计算历史分位）
        ctx: Context 对象，需包含 artifacts['FINANCIAL']

    返回:
        DataFrame，列为 code, date, [各估值因子]
    """
    if ctx is None:
        return pd.DataFrame()

    financial_path = ctx.get_artifact("FINANCIAL") if hasattr(ctx, 'get_artifact') else None
    if not financial_path or not os.path.exists(financial_path):
        logger.warning("财务数据产物不存在，跳过估值因子计算")
        return pd.DataFrame()

    logger.info("开始计算估值因子...")
    fin_df = pd.read_parquet(financial_path)
    if fin_df.empty:
        return pd.DataFrame()

    # P0-1.4 PIT 强制契约：使用财务数据前必须经过 pit_filter
    # asof 取 price_data 的最新日期（即"当前交易日"，防止用未来披露的财报）
    if 'date' in price_data.columns and not price_data.empty:
        latest_date = price_data['date'].max()
        if hasattr(latest_date, 'strftime'):
            pit_asof = latest_date.strftime('%Y%m%d')
        else:
            pit_asof = str(latest_date).replace('-', '')[:8]
    else:
        pit_asof = ""

    if pit_asof:
        try:
            from scripts.pit_guard import pit_filter
            fin_df = pit_filter(fin_df, pit_asof, caller="valuation_factors.compute")
            if fin_df is None or fin_df.empty:
                logger.warning(f"PIT 过滤后财务数据为空（asof={pit_asof}），跳过估值因子计算")
                return pd.DataFrame()
        except ImportError:
            logger.warning("PIT 守卫不可用（scripts.pit_guard 导入失败），跳过 PIT 过滤")
        except ValueError as ve:
            logger.warning(f"PIT 过滤失败: {ve}，跳过估值因子计算")
            return pd.DataFrame()

    latest_fin = fin_df.sort_values('report_date', ascending=False).drop_duplicates(subset='code', keep='first') \
        if 'code' in fin_df.columns else fin_df.iloc[[-1]]

    result = price_data[['code', 'date']].copy()

    # 估值字段直接映射
    valuation_fields = ['pe_ttm', 'pb', 'ps_ttm', 'dv_ratio']
    for col in valuation_fields:
        if col in latest_fin.columns:
            mapping = latest_fin[['code', col]].set_index('code')[col].to_dict()
            result[col] = result['code'].map(mapping)
        else:
            result[col] = np.nan

    # 计算历史分位（PE/PB 在当前价格序列中的分位）
    for code in result['code'].unique():
        mask = result['code'] == code
        for col, percentile_col in [('pe_ttm', 'pe_percentile'), ('pb', 'pb_percentile')]:
            values = result.loc[mask, col].dropna()
            if len(values) > 0:
                result.loc[mask, percentile_col] = values.rank(pct=True).values
            else:
                result.loc[mask, percentile_col] = np.nan

    logger.info(f"估值因子计算完成，共 {len(result.columns) - 2} 个因子")
    return result
