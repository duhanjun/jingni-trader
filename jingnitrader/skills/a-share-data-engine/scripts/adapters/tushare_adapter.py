"""
Tushare Pro 数据源适配器
"""
import os
import time
from typing import List, Optional
import pandas as pd
import tushare as ts

from ..base.base_data_provider import BaseDataProvider
from ..config import TUSHARE_TOKEN, MAX_WORKERS


class TushareAdapter(BaseDataProvider):
    """Tushare Pro 适配器"""

    def __init__(self, token: Optional[str] = None):
        token = token or TUSHARE_TOKEN
        if not token:
            raise ValueError("Tushare token 未设置，请设置环境变量 TUSHARE_TOKEN")
        ts.set_token(token)
        self.pro = ts.pro_api()
        # 用于频率控制
        self._last_call = 0.0
        self._min_interval = 0.2  # 每秒最多5次

    def _rate_limit(self):
        """简单频率控制"""
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def get_stock_list(self) -> pd.DataFrame:
        """获取全市场股票列表"""
        self._rate_limit()
        df = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df.rename(columns={
            'ts_code': 'code',
            'symbol': 'ticker',
            'list_date': 'list_date'
        }, inplace=True)
        # 获取ST标记
        self._rate_limit()
        st_df = self.pro.namechange(
            fields='ts_code,name,start_date,end_date'
        )
        # 简化：当前名称含ST的标记
        # 实际需更复杂逻辑，此处仅示例
        df['is_st'] = df['name'].str.contains('ST', na=False)
        return df

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        """
        获取日线行情，自动复权并添加A股特殊标记
        """
        # 格式标准化
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')

        # 根据复权方式选择接口
        if adjust == 'hfq':
            return self._get_hfq_daily(symbols, start_date, end_date)
        elif adjust == 'qfq':
            return self._get_qfq_daily(symbols, start_date, end_date)
        else:
            # 不复权直接用 pro.daily
            return self._get_raw_daily(symbols, start_date, end_date)

    def _get_hfq_daily(self, symbols, start_date, end_date):
        """后复权日线（推荐用于收益率计算）"""
        frames = []
        for symbol in symbols:
            self._rate_limit()
            try:
                df = self.pro.daily(
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
                )
                if df is None or df.empty:
                    continue
                # 获取复权因子
                self._rate_limit()
                adj_df = self.pro.adj_factor(
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                if adj_df is not None and not adj_df.empty:
                    # 使用后复权因子调整价格
                    df = df.merge(
                        adj_df[['trade_date', 'adj_factor']],
                        on='trade_date',
                        how='left'
                    )
                    # 后复权：最新价 = 不复权价 * (当前复权因子 / 最后因子)
                    # 标准做法：以最新日为基准，向前复权
                    last_adj = adj_df['adj_factor'].iloc[-1]
                    df['adj_factor'] = df['adj_factor'].fillna(last_adj)
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = df[col] * (df['adj_factor'] / last_adj)
                frames.append(df)
            except Exception as e:
                # 单只股票失败不影响整体
                print(f"获取 {symbol} 行情失败: {e}")
                continue

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return self._standardize_output(result)

    def _get_qfq_daily(self, symbols, start_date, end_date):
        """前复权日线"""
        # 可使用 pro.daily 配合 adj_factor 实现前复权，逻辑类似
        # 此处略，调用 _get_hfq_daily 后再转换亦可
        df = self._get_hfq_daily(symbols, start_date, end_date)
        if df.empty:
            return df
        # 前复权：以最新收盘价为基准，倒推历史价格
        # 简单方法：利用复权因子比值
        # 实际已包含在 _get_hfq_daily 的调整中，只需改变基准方向
        # 此简化实现直接返回后复权，标注需改进
        return df

    def _get_raw_daily(self, symbols, start_date, end_date):
        """不复权日线"""
        frames = []
        for symbol in symbols:
            self._rate_limit()
            df = self.pro.daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
            )
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        return self._standardize_output(result)

    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化输出列名与数据类型"""
        if df.empty:
            return df
        df = df.rename(columns={
            'ts_code': 'code',
            'trade_date': 'date',
            'vol': 'volume',
            'pre_close': 'pre_close'
        })
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        # 计算涨跌幅和换手率（若缺失）
        if 'change_pct' not in df.columns:
            df['change_pct'] = (df['close'] - df['pre_close']) / df['pre_close'] * 100
        # 换手率暂无法从该接口获取，留空
        df['turnover_rate'] = df.get('turnover_rate', None)
        # A股特殊标记留待清洗阶段填充
        df['is_st'] = False
        df['is_limit_up'] = False
        df['is_limit_down'] = False
        df['listed_days'] = None
        # 排序
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        return df

    def get_adj_factor(self, symbols, start_date, end_date):
        # 简单适配
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        frames = []
        for symbol in symbols:
            self._rate_limit()
            adj_df = self.pro.adj_factor(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date
            )
            if adj_df is not None and not adj_df.empty:
                frames.append(adj_df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames)
        result.rename(columns={'ts_code': 'code', 'trade_date': 'date'}, inplace=True)
        result['date'] = pd.to_datetime(result['date'], format='%Y%m%d')
        return result

    def get_financial(self, symbols, report_date, fields):
        # 示例仅实现基本调用
        self._rate_limit()
        df = self.pro.fina_indicator(
            ts_code=','.join(symbols),
            period=report_date,
            fields=','.join(fields)
        )
        if df is not None and not df.empty:
            df.rename(columns={'ts_code': 'code'}, inplace=True)
        return df if df is not None else pd.DataFrame()
