"""
迅投 xtquant 适配器（占位实现）

xtquant 是迅投 QMT/xtp 量化交易平台的 Python SDK：
- 提供实时行情（Level-2/Tick）
- 提供历史 K 线（需 xtdata 模块）
- 提供交易接口（需 xttrader 模块）

依赖：
- 本地安装 xtquant 包（pip install xtquant 仅在迅投提供源时可用）
- 启动了 QMT 客户端（提供本地数据服务）

注：xtquant 的安装包和文档非公开，本项目仅预留占位实现。
如果你的环境能拿到 xtquant，请按 BaseDataProvider 接口实现以下方法。
"""
import logging
from typing import List
import pandas as pd
from ..base.base_data_provider import BaseDataProvider


logger = logging.getLogger("xtquant-adapter")


class XtQuantAdapter(BaseDataProvider):
    """
    迅投 xtquant 适配器

    ⚠️ 默认未启用。要启用需：
    1. 部署迅投 QMT 客户端并启动
    2. pip install xtquant（来自迅投内部源）
    3. 设置环境变量启用：
       export DATA_BACKEND_FALLBACK_CHAIN="tushare,xtquant,akshare"
    """

    def __init__(self):
        try:
            from xtquant import xtdata  # noqa
            self.xtdata = xtdata
            self.available = True
        except ImportError as e:
            logger.warning(
                f"xtquant 包未安装，XtQuantAdapter 不可用。\n"
                f"  原因: {e}\n"
                f"  请先 pip install xtquant（需迅投内部源）并启动 QMT 客户端"
            )
            self.available = False

    def _check_available(self):
        if not self.available:
            from ..errors import DataSourceError
            raise DataSourceError(
                "xtquant",
                "xtquant 包未安装或 QMT 客户端未启动，请先部署环境"
            )

    def get_daily(self, symbols: List[str], start_date: str, end_date: str, adjust: str = 'hfq') -> pd.DataFrame:
        self._check_available()
        # 实际实现需要调用 xtdata.get_market_data_ex()，本项目仅占位
        return pd.DataFrame(columns=['date', 'code', 'open', 'high', 'low', 'close', 'vol'])

    def get_stock_list(self) -> pd.DataFrame:
        self._check_available()
        return pd.DataFrame(columns=['code', 'name', 'list_date'])

    def get_adj_factor(self, symbols, start_date, end_date):
        return pd.DataFrame()

    def get_financial(self, symbols, report_date, fields):
        return pd.DataFrame()
