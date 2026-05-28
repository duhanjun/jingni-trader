"""
AkShare 数据源适配器
免费另类数据源，支持龙虎榜、大宗交易、北向资金等特色数据
"""
import logging
from typing import List, Optional
import pandas as pd
import akshare as ak

from ..base.base_data_provider import BaseDataProvider


class AkshareAdapter(BaseDataProvider):
    """AkShare 适配器，专注于另类数据和补充数据源"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        """
        获取日线行情（通过 stock_zh_a_hist 接口）

        注意: AkShare 每日批量获取全市场数据效率较低，建议仅用于少量股票
              或作为 Tushare/BaoStock 的补充源。
        """
        self._logger.info(f"AkShare: 获取 {len(symbols)} 只股票日线数据")
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        all_data = []
        for code in symbols:
            try:
                ticker = self._extract_ticker(code)
                period = "daily"
                df = ak.stock_zh_a_hist(
                    symbol=ticker,
                    period=period,
                    start_date=start,
                    end_date=end,
                    adjust=adjust
                )
                if df is None or df.empty:
                    continue
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "change_pct",
                    "涨跌额": "change",
                    "换手率": "turnover_rate",
                })
                df["code"] = code
                all_data.append(df)
            except Exception as e:
                self._logger.warning(f"获取 {code} 数据失败: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
        result["open"] = pd.to_numeric(result["open"], errors="coerce")
        result["high"] = pd.to_numeric(result["high"], errors="coerce")
        result["low"] = pd.to_numeric(result["low"], errors="coerce")
        result["close"] = pd.to_numeric(result["close"], errors="coerce")
        result["volume"] = pd.to_numeric(result["volume"], errors="coerce")
        result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
        result["change_pct"] = pd.to_numeric(result["change_pct"], errors="coerce")
        result["turnover_rate"] = pd.to_numeric(result["turnover_rate"], errors="coerce")
        return result

    def get_stock_list(self) -> pd.DataFrame:
        """获取全市场股票列表"""
        try:
            df = ak.stock_info_a_code_name()
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"code": "code", "name": "name"})
            df["is_st"] = df["name"].str.contains("ST", na=False)
            return df
        except Exception as e:
            self._logger.warning(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_adj_factor(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取复权因子

        AkShare 无直接复权因子接口，此处返回空 DataFrame。
        建议使用 Tushare 或 BaoStock 获取复权因子。
        """
        self._logger.warning("AkShare 不直接提供复权因子，请使用后复权价格")
        return pd.DataFrame()

    def get_financial(
        self,
        symbols: List[str],
        report_date: str,
        fields: List[str]
    ) -> pd.DataFrame:
        """获取财务数据"""
        self._logger.warning("AkShare 财务数据接口暂不支持批量获取")
        return pd.DataFrame()

    def get_lhb_data(self, trade_date: str) -> pd.DataFrame:
        """
        获取龙虎榜数据（AkShare 特色功能）

        参数:
            trade_date: 交易日期 YYYYMMDD 或 YYYY-MM-DD

        返回:
            DataFrame 包含龙虎榜上榜股票明细
        """
        trade_date = trade_date.replace("-", "")
        try:
            df = ak.stock_sina_lhb_detail_daily(trade_date=trade_date)
            if df is not None and not df.empty:
                df["trade_date"] = trade_date
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            self._logger.warning(f"获取龙虎榜数据失败: {e}")
            return pd.DataFrame()

    def get_block_trade(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取大宗交易数据（AkShare 特色功能）

        参数:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD

        返回:
            DataFrame 大宗交易明细
        """
        try:
            df = ak.stock_dzjy_mrmx(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                symbol="沪深A股"
            )
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            self._logger.warning(f"获取大宗交易数据失败: {e}")
            return pd.DataFrame()

    def get_north_flow(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取北向资金流向（AkShare 特色功能）

        参数:
            start_date: 开始日期
            end_date: 结束日期

        返回:
            DataFrame 北向资金每日净买入（按个股）
        """
        try:
            df = ak.stock_hsgt_individual_em(
                symbol="沪股通",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", "")
            )
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            self._logger.warning(f"获取北向资金数据失败: {e}")
            return pd.DataFrame()

    def _extract_ticker(self, code: str) -> str:
        """从 000001.SZ 格式中提取纯数字代码"""
        if "." in code:
            return code.split(".")[0]
        return code