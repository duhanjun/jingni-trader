"""
BaoStock 数据源适配器
免费、无需注册，适合快速验证
"""
import pandas as pd
import baostock as bs
from typing import List, Optional
from ..base.base_data_provider import BaseDataProvider


class BaostockAdapter(BaseDataProvider):
    """BaoStock 适配器"""

    def __init__(self):
        self._logged_in = False

    def _ensure_login(self):
        if not self._logged_in:
            bs.login()
            self._logged_in = True

    def _format_code(self, code: str) -> str:
        """将 000001.SZ 转为 sh.600000 或 sz.000001"""
        if '.' in code:
            ticker, exchange = code.split('.')
            if exchange.upper() == 'SH':
                return f"sh.{ticker}"
            else:
                return f"sz.{ticker}"
        return code

    def get_stock_list(self) -> pd.DataFrame:
        self._ensure_login()
        # 获取全量股票数据
        rs = bs.query_stock_basic()
        if rs.error_code != '0':
            return pd.DataFrame()
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=rs.fields)
        # 映射为统一格式
        df['code'] = df['code'].apply(lambda x: x.replace('sh.', 'SH.').replace('sz.', 'SZ.'))
        df['name'] = df['code_name']
        df['is_st'] = df['code_name'].str.contains('ST', na=False)
        return df

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        self._ensure_login()
        frames = []
        for code in symbols:
            bs_code = self._format_code(code)
            # 根据复权方式选择字段
            if adjust == 'hfq':
                fields = "date,open,high,low,close,preclose,volume,amount,turn,adjustflag,tradestatus,isST"
            else:
                fields = "date,open,high,low,close,preclose,volume,amount,turn,adjustflag,tradestatus,isST"
            # 修改日期格式为 YYYY-MM-DD
            start = start_date.replace('-', '')
            end = end_date.replace('-', '')
            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="2" if adjust == "qfq" else "1"  # 1:后复权, 2:前复权
            )
            if rs.error_code != '0':
                continue
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            if not data:
                continue
            df = pd.DataFrame(data, columns=rs.fields)
            # 类型转换
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['code'] = code
            df['date'] = pd.to_datetime(df['date'])
            df['is_st'] = df['isST'].astype(bool) if 'isST' in df.columns else False
            # 涨跌停标记：BaoStock未直接提供，需自行计算，此处留空
            df['is_limit_up'] = False
            df['is_limit_down'] = False
            df['listed_days'] = None
            # 映射列名
            df = df.rename(columns={
                'preclose': 'pre_close',
                'turn': 'turnover_rate'
            })
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(['code', 'date']).reset_index(drop=True)

    def get_adj_factor(self, symbols, start_date, end_date):
        # BaoStock 已内置复权，无需额外因子
        return pd.DataFrame()

    def get_financial(self, symbols, report_date, fields):
        self._ensure_login()
        # 暂不实现详细财务，返回空
        return pd.DataFrame()
