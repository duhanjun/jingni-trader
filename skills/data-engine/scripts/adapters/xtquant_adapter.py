"""
迅投 xtquant (QMT/xtp) 适配器 —— 真实实现

通过本地运行的 QMT 客户端数据服务（xtdata）获取 A股行情。
前置条件：
- 已安装 xtquant 包（来自 QMT 客户端目录的 bin.x64/Lib/site-packages/xtquant）
- 已启动并登录 QMT 客户端（极速交易/投研模式），本地数据服务在监听
- xtdata.connect() 会自动连接本地服务（默认 127.0.0.1，端口由客户端分配）

复权：xtdata 的复权参数为 dividend_type，映射关系：
    hfq -> 'back'（后复权）
    qfq -> 'front'（前复权）
    none-> 'none'（不复权）
"""
import logging
from typing import List, Optional
import pandas as pd

from ..base.base_data_provider import BaseDataProvider
from ..errors import DataSourceError


logger = logging.getLogger("xtquant-adapter")

_DIVIDEND_MAP = {"hfq": "back", "qfq": "front", "none": "none", "": "none"}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """把各适配器返回的裸行情统一成下游期望的标准 schema。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "code", "date", "open", "high", "low", "close", "volume", "amount",
            "pre_close", "change_pct", "turnover_rate",
            "is_st", "is_limit_up", "is_limit_down", "listed_days"
        ])
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    if "pre_close" not in df.columns:
        df["pre_close"] = df.groupby("code", group_keys=False)["close"].shift(1)
    if "change_pct" not in df.columns:
        df["change_pct"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100.0
    for c in ["turnover_rate", "is_st", "is_limit_up", "is_limit_down", "listed_days"]:
        if c not in df.columns:
            df[c] = None
    df["is_st"] = df["is_st"].fillna(False)
    df["is_limit_up"] = df["is_limit_up"].fillna(False)
    df["is_limit_down"] = df["is_limit_down"].fillna(False)
    cols = ["code", "date", "open", "high", "low", "close", "volume", "amount",
            "pre_close", "change_pct", "turnover_rate",
            "is_st", "is_limit_up", "is_limit_down", "listed_days"]
    return df.sort_values(["code", "date"]).reset_index(drop=True)[cols]


class XtQuantAdapter(BaseDataProvider):
    """迅投 xtquant (QMT) 适配器：连接本地 QMT 数据服务拉取真实行情。"""

    SUPPORTED_DATA_TYPES = {"daily", "financial"}

    def __init__(self):
        try:
            from xtquant import xtdata
            self.xtdata = xtdata
            self.available = True
        except ImportError as e:
            logger.warning(
                "xtquant 包未安装或 QMT 客户端未启动，XtQuantAdapter 不可用: %s", e
            )
            self.available = False

    def _check_available(self):
        if not self.available:
            raise DataSourceError(
                "xtquant",
                "xtquant 包未安装或 QMT 客户端未启动，请先部署环境并登录 QMT"
            )

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        self._check_available()
        xtdata = self.xtdata
        xtdata.connect()
        dividend_type = _DIVIDEND_MAP.get(adjust, "none")
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")
        field_list = ["open", "high", "low", "close", "volume", "amount"]

        frames = []
        for code in symbols:
            # 先确保本地已缓存该区间历史数据
            try:
                xtdata.download_history_data(code, "1d", sd, ed)
            except Exception as e:
                logger.warning("xtdata 下载 %s 历史数据失败（继续尝试读取）: %s", code, e)
            try:
                raw = xtdata.get_market_data_ex(
                    field_list, [code], period="1d",
                    start_time=sd, end_time=ed,
                    dividend_type=dividend_type
                )
            except Exception as e:
                raise DataSourceError("xtquant", f"获取 {code} 行情失败: {e}") from e
            df = raw.get(code) if isinstance(raw, dict) else raw
            if df is None or len(df) == 0:
                logger.warning("xtdata 未返回 %s 的数据", code)
                continue
            d = df.copy()
            d.index.name = "date"
            d = d.reset_index()
            d["date"] = pd.to_datetime(d["date"])
            d["code"] = code
            frames.append(d)

        if not frames:
            return _finalize(pd.DataFrame())
        return _finalize(pd.concat(frames, ignore_index=True))

    def get_stock_list(self) -> pd.DataFrame:
        self._check_available()
        xtdata = self.xtdata
        xtdata.connect()
        try:
            codes = xtdata.get_stock_list_in_sector("沪深A股") or []
        except Exception as e:
            logger.warning("xtdata 获取股票列表失败: %s", e)
            return pd.DataFrame(columns=["code", "name", "industry", "list_date", "is_st"])
        rows = []
        for c in codes:
            name = c
            try:
                det = xtdata.get_instrument_detail(c)
                if det and isinstance(det, dict):
                    name = det.get("InstrumentName", c) or c
            except Exception:
                pass
            rows.append({"code": c, "name": name, "is_st": "ST" in str(name)})
        df = pd.DataFrame(rows, columns=["code", "name", "is_st"])
        df["industry"] = None
        df["list_date"] = None
        return df

    def get_adj_factor(self, symbols, start_date, end_date):
        # xtdata 复权已内置于 dividend_type，这里不单独提供因子
        return pd.DataFrame(columns=["code", "date", "adj_factor"])

    def get_financial(self, symbols, report_date, fields):
        """获取财务数据，返回统一标准 schema。

        xtquant 的 get_financial_data 主要提供股本结构(Capital)、十大股东
        (Top10Holder)、股东户数(HolderNum)等股权类数据，不直接提供
        PE/PB/ROE/营收等基本面指标。此处通过 get_instrument_detail 获取
        股票名称，其余财务字段留空(None)，由上层降级链切换到其他源补充。

        返回 DataFrame 包含标准字段:
            code, report_date, pe_ttm, pb, ps_ttm, dv_ratio,
            roe, roa, gross_margin, net_margin,
            revenue_growth, profit_growth,
            debt_ratio, current_ratio, quick_ratio, ocf,
            industry, name
        """
        self._check_available()
        xtdata = self.xtdata

        standard_cols = [
            'code', 'report_date', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio',
            'roe', 'roa', 'gross_margin', 'net_margin',
            'revenue_growth', 'profit_growth',
            'debt_ratio', 'current_ratio', 'quick_ratio', 'ocf',
            'industry', 'name',
        ]

        # 标准化报告期: '2024-09-30' -> '20240930'
        period = report_date.replace('-', '')

        rows = []
        for code in symbols:
            row = {col: None for col in standard_cols}
            row['code'] = code
            row['report_date'] = period

            # 股票名称: get_instrument_detail 返回 InstrumentName 字段
            try:
                det = xtdata.get_instrument_detail(code)
                if det and isinstance(det, dict):
                    row['name'] = det.get('InstrumentName') or None
            except Exception as e:
                logger.debug("xtdata 获取 %s 名称失败: %s", code, e)

            rows.append(row)

        out = pd.DataFrame(rows, columns=standard_cols)

        # 如果调用方指定了 fields，按需过滤列（code/report_date 始终保留）
        if fields:
            keep = ['code', 'report_date'] + [f for f in fields if f in standard_cols]
            keep = list(dict.fromkeys(keep))
            out = out[keep]

        return out.reset_index(drop=True)
