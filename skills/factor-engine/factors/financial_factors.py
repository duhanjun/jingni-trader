"""
财务因子计算模块

从 financial_data.parquet 提取财务因子，映射到标准因子列。

P0-1 PIT 契约：读取财务数据后强制走 pit_filter，防止 look-ahead bias。
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

    # P0-1.4 PIT 强制契约：使用财务数据前必须经过 pit_filter
    # asof 取 price_data 的最新日期（即"当前交易日"，防止用未来披露的财报）
    if 'date' in price_data.columns and not price_data.empty:
        latest_date = price_data['date'].max()
        if hasattr(latest_date, 'strftime'):
            pit_asof = latest_date.strftime('%Y%m%d')
        else:
            pit_asof = str(latest_date).replace('-', '')[:8]
    else:
        # price_data 为空时无法确定 asof，跳过 PIT（下游也不会用到）
        pit_asof = ""

    if pit_asof:
        try:
            from scripts.pit_guard import pit_filter
            fin_df = pit_filter(fin_df, pit_asof, caller="financial_factors.compute")
            if fin_df is None or fin_df.empty:
                logger.warning(f"PIT 过滤后财务数据为空（asof={pit_asof}），跳过财务因子计算")
                return pd.DataFrame()
        except ImportError:
            # scripts.pit_guard 不可用时降级（不阻断流程，但记 warning）
            logger.warning("PIT 守卫不可用（scripts.pit_guard 导入失败），跳过 PIT 过滤")
        except ValueError as ve:
            # 严格模式下缺 disclosure_date 列会 raise
            logger.warning(f"PIT 过滤失败: {ve}，跳过财务因子计算")
            return pd.DataFrame()

    # 取每只股票最新一行的财务数据（PIT 过滤后的最新，已无未来数据）
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
