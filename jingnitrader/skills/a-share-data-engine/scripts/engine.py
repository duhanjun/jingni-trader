"""
A股数据引擎主逻辑
负责调度适配器、数据清洗、本地存储
"""
import os
import logging
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .config import DATA_BACKEND, DATA_FORMAT, ADJUST_MODE, CACHE_DIR, MAX_MISSING_RATIO
from .base.base_data_provider import BaseDataProvider

# 动态加载适配器
def _load_adapter() -> BaseDataProvider:
    if DATA_BACKEND == "tushare":
        from .adapters.tushare_adapter import TushareAdapter
        return TushareAdapter()
    elif DATA_BACKEND == "baostock":
        from .adapters.baostock_adapter import BaostockAdapter
        return BaostockAdapter()
    elif DATA_BACKEND == "akshare":
        from .adapters.akshare_adapter import AkshareAdapter
        return AkshareAdapter()
    else:
        raise ValueError(f"不支持的数据源: {DATA_BACKEND}")

logger = logging.getLogger("a-share-data-engine")


class DataEngine:
    """A股数据引擎"""

    def __init__(self, provider: Optional[BaseDataProvider] = None):
        self.provider = provider or _load_adapter()

    def fetch_and_clean(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = ADJUST_MODE,
        exclude_st: bool = True,
        exclude_new: bool = True,
        min_listed_days: int = 60,
        fill_suspend: bool = False
    ) -> pd.DataFrame:
        """
        获取并清洗日线数据

        参数:
            symbols: 股票代码列表，为空则获取全部A股
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期
            adjust: 复权方式
            exclude_st: 剔除ST
            exclude_new: 剔除新股
            min_listed_days: 最少上市天数
            fill_suspend: 停牌是否前向填充

        返回:
            清洗后的 DataFrame，包含统一列
        """
        # 如果未指定股票池，获取全市场列表
        if not symbols:
            stock_df = self.provider.get_stock_list()
            if stock_df.empty:
                logger.error("无法获取股票列表")
                return pd.DataFrame()
            symbols = stock_df['code'].tolist()

        logger.info(f"开始获取 {len(symbols)} 只股票的日线数据，时间 {start_date} 至 {end_date}")
        df = self.provider.get_daily(symbols, start_date, end_date, adjust=adjust)
        if df.empty:
            logger.error("未获取到任何数据")
            return df

        # ── 数据清洗 ──────────────────────
        logger.info("开始数据清洗...")
        initial_rows = len(df)

        # 1. 剔除上市不足 min_listed_days 交易日的新股
        if exclude_new:
            stock_info = self.provider.get_stock_list()
            if not stock_info.empty and 'list_date' in stock_info.columns:
                # 合并上市日期
                stock_info['list_date'] = pd.to_datetime(stock_info['list_date'], format='%Y%m%d', errors='coerce')
                df = df.merge(stock_info[['code', 'list_date']], on='code', how='left')
                # 计算上市天数
                df['listed_days'] = (df['date'] - df['list_date']).dt.days
                df = df[df['listed_days'] >= min_listed_days]
                logger.info(f"剔除新股后剩余 {len(df)} 行 (剔除 {initial_rows - len(df)} 行)")

        # 2. 停牌处理
        if not fill_suspend:
            # 剔除停牌日（volume == 0 或 close 为 NaN 视为停牌）
            df = df[df['volume'] > 0]
        else:
            # 前向填充停牌期间的 OHLC（需按股票分组）
            df = df.sort_values(['code', 'date'])
            # 创建连续日期索引
            df = df.set_index(['code', 'date'])
            # 重新索引为完整日期范围（较复杂，此处简化：仅对已有数据填充 NaN）
            df = df.groupby('code').apply(
                lambda x: x.ffill()
            ).reset_index(level=0, drop=True)
            df = df.reset_index()

        # 3. 涨跌停标记（若数据源未提供）
        if 'is_limit_up' not in df.columns or df['is_limit_up'].isna().all():
            df = self._mark_price_limits(df)
        if 'is_st' not in df.columns or df['is_st'].isna().all():
            df = self._mark_st(df)

        # 4. 剔除 ST 股票（若需）
        if exclude_st:
            st_mask = df['is_st'] == True
            df = df[~st_mask]

        # 5. 缺失值处理
        df = df.dropna(subset=['close'])

        # 最终检查
        logger.info(f"清洗完成，最终 {len(df)} 行数据")
        return df.sort_values(['date', 'code']).reset_index(drop=True)

    def _mark_price_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据涨跌幅标记涨跌停"""
        if 'change_pct' not in df.columns:
            return df
        # A股正常涨跌停 ±10%（ST为±5%，但此处简单处理）
        limit_up = df['change_pct'] >= 9.9  # 考虑四舍五入
        limit_down = df['change_pct'] <= -9.9
        df['is_limit_up'] = limit_up
        df['is_limit_down'] = limit_down
        return df

    def _mark_st(self, df: pd.DataFrame) -> pd.DataFrame:
        """标记ST股票"""
        # 若已有 is_st 列则跳过
        if 'is_st' in df.columns and not df['is_st'].isna().all():
            return df
        # 简单通过代码名称判断，实际应通过 stock_basic
        stock_list = self.provider.get_stock_list()
        if not stock_list.empty and 'is_st' in stock_list.columns:
            st_codes = stock_list[stock_list['is_st'] == True]['code'].tolist()
            df['is_st'] = df['code'].isin(st_codes)
        else:
            df['is_st'] = False
        return df

    def save_data(self, df: pd.DataFrame, path: str):
        """保存数据到文件"""
        if DATA_FORMAT == 'parquet':
            df.to_parquet(path, index=False)
        elif DATA_FORMAT == 'csv':
            df.to_csv(path, index=False)
        elif DATA_FORMAT == 'sql':
            # 可扩展 SQL 存储
            from sqlalchemy import create_engine
            engine = create_engine(os.environ.get("QUANT_DB_URL", "sqlite:///quant.db"))
            df.to_sql('daily', engine, if_exists='append', index=False)
        else:
            raise ValueError(f"不支持的存储格式: {DATA_FORMAT}")
        logger.info(f"数据已保存至 {path}")


# ── Skill 统一入口 ────────────────────────
def run(ctx) -> Dict[str, Any]:
    """
    a-share-data-engine 的 run 函数
    由 quant-trading-master 调度

    参数:
        ctx: Context 对象，需包含:
            - stock_pool: list
            - start_date: str
            - end_date: str
            - 可选: artifacts 已有产物路径可跳过

    返回:
        {
            "success": bool,
            "artifact_path": str,
            "metadata": dict,
            "error": str
        }
    """
    try:
        # 如果上下文已有数据产物，跳过
        existing = ctx.get_artifact("DATA")
        if existing and os.path.exists(existing):
            return {
                "success": True,
                "artifact_path": existing,
                "metadata": {"source": "cache"},
                "error": ""
            }

        engine = DataEngine()
        df = engine.fetch_and_clean(
            symbols=ctx.stock_pool,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
            adjust=ADJUST_MODE
        )
        if df.empty:
            return {
                "success": False,
                "artifact_path": "",
                "metadata": {},
                "error": "未获取到任何有效数据"
            }

        # 产物路径
        output_dir = os.environ.get("QUANT_DATA_DIR", "./workspace/data")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "cleaned_data.parquet")
        engine.save_data(df, path)

        return {
            "success": True,
            "artifact_path": path,
            "metadata": {
                "rows": len(df),
                "symbols_count": df['code'].nunique(),
                "date_range": f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}"
            },
            "error": ""
        }
    except Exception as e:
        logger.exception("数据引擎执行失败")
        return {
            "success": False,
            "artifact_path": "",
            "metadata": {},
            "error": str(e)
        }


# ── CLI 入口 ──────────────────────────────
if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jingnitrader.scripts.context import Context

    # 简单模拟 context 输入
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            ctx_dict = json.load(f)
        ctx = Context.from_dict(ctx_dict)
    else:
        ctx = Context(
            task_id="test",
            stock_pool=["000001.SZ", "600000.SH"],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
    result = run(ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False))