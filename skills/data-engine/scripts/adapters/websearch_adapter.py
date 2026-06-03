"""
WebSearch 适配器（终极数据源）

当 baostock 和 akshare 都没有覆盖某只标的数据时，调用搜索引擎查询。

设计要点：
- 搜索引擎由调用方注入（web_search_fn: Callable[[str], str]）
- 实际使用 jingni-trader 时，agent 通过 Context 注入 WebSearch 工具函数
- 本地测试可注入 mock 函数

⚠️ 限制：
- 搜索引擎对历史数据的覆盖有限（多数只到最近几个月）
- 单次只能查询一个标的一个数据点（不适合批量）
- 解析依赖正则/LLM，复杂数据点可能失败
- 适合"补缺"而非"主源"
"""
import logging
import re
from typing import List, Callable, Optional
import pandas as pd

from ..base.base_data_provider import BaseDataProvider
from ..errors import DataSourceError, DataNotFoundError


logger = logging.getLogger("websearch-adapter")


# WebSearch 注入函数的类型签名
WebSearchFn = Callable[[str], str]


class WebSearchAdapter(BaseDataProvider):
    """
    WebSearch 适配器：通过搜索引擎获取具体数据点

    使用方式（jingni-trader agent 注入）：
        from websearch_tool import web_search  # 实际工程中由 agent 注入
        adapter = WebSearchAdapter(web_search_fn=web_search)
    """

    def __init__(self, web_search_fn: Optional[WebSearchFn] = None):
        """
        参数:
            web_search_fn: 注入的搜索函数，签名 (query: str) -> str
                          实际工程中由 jingni-trader agent 通过 Context 注入
        """
        self.web_search_fn = web_search_fn
        if not web_search_fn:
            logger.warning(
                "WebSearchAdapter 未注入 web_search_fn，"
                "如需启用请通过 Context.external_data.web_search_fn 注入"
            )

    def _check_available(self):
        if not self.web_search_fn:
            raise DataSourceError(
                "websearch",
                "web_search_fn 未注入（jingni-trader 应通过 Context 注入）"
            )

    def _build_query(self, symbol: str, date: str) -> str:
        """构造搜索查询语句"""
        return f"{symbol} {date} 收盘价 开盘价 最高价 最低价 成交量"

    def _parse_search_result(self, symbol: str, date: str, raw_text: str) -> dict:
        """
        从搜索结果文本中解析出 OHLCV 数据

        期望从搜索结果中提取类似：
        "511090 在 2024-01-15 的收盘价为 105.23 元，开盘 104.80 元，最高 105.50 元，最低 104.50 元，成交量 12345 手"

        解析策略：先尝试多个正则模式
        """
        text = raw_text.replace(",", "").replace("，", "")
        result = {"code": symbol, "date": date}

        # 收盘价
        for pattern in [
            r"收盘价?[:\s为]?\s*(\d+\.?\d*)",
            r"收盘[:\s为]?\s*(\d+\.?\d*)",
            r"收\s*[:\s为]?\s*(\d+\.?\d*)",
        ]:
            m = re.search(pattern, text)
            if m:
                result["close"] = float(m.group(1))
                break

        # 开盘价
        for pattern in [
            r"开盘价?[:\s为]?\s*(\d+\.?\d*)",
            r"开盘[:\s为]?\s*(\d+\.?\d*)",
            r"开\s*[:\s为]?\s*(\d+\.?\d*)",
        ]:
            m = re.search(pattern, text)
            if m:
                result["open"] = float(m.group(1))
                break

        # 最高价
        for pattern in [
            r"最高价?[:\s为]?\s*(\d+\.?\d*)",
            r"最高[:\s为]?\s*(\d+\.?\d*)",
            r"高\s*[:\s为]?\s*(\d+\.?\d*)",
        ]:
            m = re.search(pattern, text)
            if m:
                result["high"] = float(m.group(1))
                break

        # 最低价
        for pattern in [
            r"最低价?[:\s为]?\s*(\d+\.?\d*)",
            r"最低[:\s为]?\s*(\d+\.?\d*)",
            r"低\s*[:\s为]?\s*(\d+\.?\d*)",
        ]:
            m = re.search(pattern, text)
            if m:
                result["low"] = float(m.group(1))
                break

        # 成交量
        for pattern in [
            r"成交量[:\s为]?\s*(\d+\.?\d*)\s*[万]?手",
            r"成交[:\s为]?\s*(\d+\.?\d*)\s*[万]?手",
            r"量[:\s为]?\s*(\d+\.?\d*)",
        ]:
            m = re.search(pattern, text)
            if m:
                result["vol"] = float(m.group(1))
                break

        # 如果连收盘价都没解析出来，标记为 DataNotFoundError
        if "close" not in result:
            raise DataNotFoundError(
                "websearch",
                f"无法从搜索结果中解析出 {symbol} @ {date} 的数据（搜索引擎无相关数据或结果无法解析）"
            )
        # 默认值
        for k in ["open", "high", "low", "vol"]:
            if k not in result:
                result[k] = pd.NA
        return result

    def get_daily(self, symbols: List[str], start_date: str, end_date: str, adjust: str = 'hfq') -> pd.DataFrame:
        """
        通过 WebSearch 逐个数据点查询

        ⚠️ 注意：搜索引擎对历史数据覆盖有限，本方法适合"补缺"
        对于大段时间范围，会调用大量搜索请求

        抛出的异常：
        - DataNotFoundError: 搜索引擎无相关数据
        - DataSourceError: web_search_fn 未注入
        """
        self._check_available()
        from ..config import MAX_WORKERS
        # 简化实现：逐个标的关键点查询
        # 实际工程中可并发，但搜索请求本身有节流
        rows = []
        failed_symbols = []
        for symbol in symbols:
            # 取首尾两点作为代表（避免太多搜索请求）
            sample_dates = [start_date, end_date]
            for d in sample_dates:
                try:
                    query = self._build_query(symbol, d)
                    logger.info(f"WebSearch 查询: {query}")
                    raw = self.web_search_fn(query)
                    # 检查搜索结果是否包含"未找到"等无数据提示
                    if any(kw in raw for kw in ["未找到", "无相关", "无搜索结果", "没有找到", "no results", "未搜到"]):
                        logger.warning(f"WebSearch 未找到 {symbol} @ {d}")
                        continue
                    parsed = self._parse_search_result(symbol, d, raw)
                    rows.append(parsed)
                except DataNotFoundError as e:
                    logger.warning(f"WebSearch 未拿到 {symbol} @ {d}: {e.message}")
                    continue
        if not rows:
            # 所有查询都失败
            raise DataNotFoundError(
                "websearch",
                f"WebSearch 未能找到 {len(symbols)} 个标的（{', '.join(symbols[:3])}...）的有效数据"
            )
        return pd.DataFrame(rows)

    def get_stock_list(self) -> pd.DataFrame:
        """
        WebSearch 不适合拉全市场列表，应避免调用
        """
        raise DataSourceError(
            "websearch",
            "WebSearchAdapter 不支持 get_stock_list（请使用 baostock 或 akshare）"
        )

    def get_adj_factor(self, symbols, start_date, end_date):
        return pd.DataFrame()

    def get_financial(self, symbols, report_date, fields):
        return pd.DataFrame()
