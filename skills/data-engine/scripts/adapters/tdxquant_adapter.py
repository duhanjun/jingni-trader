"""
通达信 TdxQuant 适配器（占位实现）

TdxQuant 是通达信（深圳市财富趋势科技）的 Python 量化接口：
- 官方文档: https://help.tdx.com.cn/quant/
- 数据范围: 实时/历史 K线、Tick、财务数据、新股、分类等
- 行情模块: tqcenter（运行在通达信金融终端本地进程）

依赖：
- 启动支持 TQ 策略的通达信金融终端/专业研究版/量化模拟版/期货通
- 官方 Python SDK（通常通过 tqcenter 服务访问）
- 支持 Python 3.7 ~ 3.14

注：通达信 TdxQuant 的安装包和文档非公开，本项目仅预留占位实现。
如果你的环境能拿到 TdxQuant，请按 BaseDataProvider 接口实现以下方法。
"""
import logging
import os
from typing import List
import pandas as pd
from ..base.base_data_provider import BaseDataProvider
from ..errors import DataSourceError


logger = logging.getLogger("tdxquant-adapter")


class TdxQuantAdapter(BaseDataProvider):
    """
    通达信 TdxQuant 适配器

    ⚠️ 默认未启用。要启用需：
    1. 安装通达信金融终端（专业研究版/量化模拟版/期货通均可，需支持 TQ 策略）
    2. 启动终端并启用 tqcenter
    3. 设置环境变量启用：
       export DATA_BACKENDS="tushare,tdxquant,baostock,akshare,websearch"

    行情数据范围（官方文档摘录）:
    - 行情数据：实时与历史的快照、K线、分笔（Tick）数据
    - 基本面数据：除权除息、基本财务、专业财务、股票交易数据
    - 标的基础信息、新股申购、可转债
    - 分类数据：市场类型、行业分类、自定义板块
    """

    def __init__(self):
        self.tqcenter_url = os.environ.get("TDX_TQCENTER_URL", "tcp://127.0.0.1:8181")
        self._available = False
        try:
            # 尝试导入 tdxquant（实际包名待官方确认）
            import tdxquant  # noqa
            self.tdxquant = tdxquant
            self._available = True
        except ImportError:
            try:
                # 备选：通达信老牌 pytdx 包（与 tdxquant 不完全等价）
                import pytdx  # noqa
                logger.info(
                    "检测到 pytdx，但 TdxQuant 官方接口未安装。"
                    "请到 https://help.tdx.com.cn/quant/ 获取官方 TdxQuant 包"
                )
            except ImportError:
                logger.warning(
                    "TdxQuant 包未安装，TdxQuantAdapter 不可用。\n"
                    "  请到 https://help.tdx.com.cn/quant/ 获取官方 TdxQuant 包\n"
                    "  并启动通达信金融终端（需支持 TQ 策略）"
                )

    def _check_available(self):
        if not self._available:
            raise DataSourceError(
                "tdxquant",
                "TdxQuant 包未安装或通达信金融终端未启动（需支持 TQ 策略）"
            )

    def get_daily(self, symbols: List[str], start_date: str, end_date: str, adjust: str = 'hfq') -> pd.DataFrame:
        self._check_available()
        # 实际实现需要通过 tqcenter 调用 get_kline 等接口
        return pd.DataFrame(columns=['date', 'code', 'open', 'high', 'low', 'close', 'vol'])

    def get_stock_list(self) -> pd.DataFrame:
        self._check_available()
        return pd.DataFrame(columns=['code', 'name', 'list_date'])

    def get_adj_factor(self, symbols, start_date, end_date):
        return pd.DataFrame()

    def get_financial(self, symbols, report_date, fields):
        return pd.DataFrame()
