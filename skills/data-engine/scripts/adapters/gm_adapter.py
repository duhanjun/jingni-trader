"""
掘金 gm 适配器（占位实现）

掘金量化（https://www.myquant.cn/）是商业量化平台：
- 提供统一 REST/WebSocket API
- 覆盖 A股/港股/美股/期货/期权/外汇/基金
- 需付费 + 单独 token（GM_TOKEN）

依赖：
- pip install gm
- 设置环境变量 GM_TOKEN=<你的 token>

注：gm 包的官方源在 https://pypi.org/project/gm/（已停止维护）
活跃 fork 在 https://github.com/whencome/gmstest 等
"""
import logging
import os
from typing import List
import pandas as pd
from ..base.base_data_provider import BaseDataProvider
from ..config import GM_TOKEN


logger = logging.getLogger("gm-adapter")


class GmAdapter(BaseDataProvider):
    """
    掘金量化 gm 适配器

    ⚠️ 默认未启用。要启用需：
    1. pip install gm（来自商业平台）
    2. export GM_TOKEN=<你的掘金 token>
    3. export DATA_BACKEND_FALLBACK_CHAIN="tushare,gm,akshare"
    """

    def __init__(self):
        self.token = GM_TOKEN or os.environ.get("GM_TOKEN")
        if not self.token:
            logger.warning(
                "GM_TOKEN 未设置，GmAdapter 不可用。\n"
                "  请到 https://www.myquant.cn/ 申请 token 并设置：export GM_TOKEN=xxx"
            )
            self.available = False
        else:
            try:
                import gm  # noqa
                self.gm = gm
                self.available = True
            except ImportError as e:
                logger.warning(
                    f"gm 包未安装，GmAdapter 不可用。\n"
                    f"  原因: {e}\n"
                    f"  请 pip install gm（来自商业平台源）"
                )
                self.available = False

    def _check_available(self):
        if not self.available:
            from ..errors import DataSourceError
            raise DataSourceError(
                "gm",
                "gm 包未安装或 GM_TOKEN 未设置，请先部署环境"
            )

    def get_daily(self, symbols: List[str], start_date: str, end_date: str, adjust: str = 'hfq') -> pd.DataFrame:
        self._check_available()
        # 实际实现需要调用 gm.api(...) 或 gm.history(...)
        return pd.DataFrame(columns=['date', 'code', 'open', 'high', 'low', 'close', 'vol'])

    def get_stock_list(self) -> pd.DataFrame:
        self._check_available()
        return pd.DataFrame(columns=['code', 'name', 'list_date'])

    def get_adj_factor(self, symbols, start_date, end_date):
        return pd.DataFrame()

    def get_financial(self, symbols, report_date, fields):
        return pd.DataFrame()
