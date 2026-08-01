"""
万得 WindPy 适配器

接入方法：
- 通过 WindPy 的 w.start() 初始化连接（依赖本地 Wind 金融终端）
- 日线数据用 w.wsd(codes, fields, beginTime, endTime, options, usedf=True)
  返回 (error_code, DataFrame)
- 字段: open, high, low, close, volume, amt, oi
- 代码格式: {symbol}.{exchange}（如 600000.SH / 000001.SZ），与 jingni-trader 约定一致

前置条件：
- 已购买 Wind 金融终端并安装好 WindPy（随终端分发，不在 PyPI 公开发布）
- Wind 终端已启动并登录（WindPy 通过本地服务通信）

复权方式（Wind wsd 的 PriceAdj 参数）：
    hfq -> "B"（后复权，Back）
    qfq -> "F"（前复权，Front）
    none-> "U"（不复权，Unadjusted）
    说明：Wind 的 PriceAdj 命名与 jingni-trader 的 adjust 参数语义相反，
         此处做显式映射，避免混淆。
"""
import logging
from typing import List, Optional
import pandas as pd

from ..base.base_data_provider import BaseDataProvider
from ..errors import DataSourceError, NetworkError, DataNotFoundError, InvalidParameterError


logger = logging.getLogger("wind-adapter")


# jingni-trader adjust 语义 -> Wind PriceAdj 取值
# 注意：Wind 文档中 "F"=Front=前复权, "B"=Back=后复权
#       jingni-trader 中 "hfq"=后复权, "qfq"=前复权
# 映射时必须显式翻转，防止误用
_ADJUST_MAP = {
    "hfq": "B",   # 后复权
    "qfq": "F",   # 前复权
    "none": "U",  # 不复权
    "": "U",
}


# jingni-trader 代码后缀 -> Wind 交易所后缀
# 代码本身已是 {6位数字}.{SH/SZ} 格式，Wind 直接接受
# 仅做校验，无需转换
_SUPPORTED_SUFFIX = {".SH", ".SZ", ".CFE", ".SHF", ".CZC", ".DCE", ".INE"}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """把 Wind 返回的裸行情统一成下游期望的标准 schema。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "code", "date", "open", "high", "low", "close", "volume", "amount",
            "pre_close", "change_pct", "turnover_rate",
            "is_st", "is_limit_up", "is_limit_down", "listed_days"
        ])
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "amt" in df.columns and "amount" not in df.columns:
        df = df.rename(columns={"amt": "amount"})
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


class WindAdapter(BaseDataProvider):
    """万得 WindPy 适配器：通过本地 Wind 终端拉取 A股行情。

    依赖 WindPy 包（由 Wind 金融终端分发），需先启动并登录 Wind 终端。
    """

    SUPPORTED_DATA_TYPES = {"daily", "financial"}

    def __init__(self):
        try:
            from WindPy import w
            self.w = w
            self._connected = False
            self._connect()
        except ImportError as e:
            raise DataSourceError(
                "wind",
                f"WindPy 包未安装或 Wind 终端未启动: {e}。"
                f"请先安装 Wind 金融终端并配置 WindPy。"
            ) from e

    def _connect(self):
        """初始化 WindPy 连接。"""
        if self._connected:
            return
        try:
            if self.w.isconnected():
                self._connected = True
                return
            data = self.w.start()
            if data.ErrorCode:
                raise NetworkError(
                    "wind",
                    f"WindPy 连接失败，错误码: {data.ErrorCode}"
                )
            self._connected = True
            logger.info("WindPy 连接成功")
        except DataSourceError:
            raise
        except Exception as e:
            raise NetworkError("wind", f"WindPy 启动异常: {e}") from e

    def _ensure_connected(self):
        """每次查询前确保连接可用，断线自动重连。"""
        if not self._connected or not self.w.isconnected():
            self._connect()

    @staticmethod
    def _format_date(date_str: str) -> str:
        """把 YYYY-MM-DD 或 YYYYMMDD 统一为 Wind 期望的 YYYY-MM-DD。"""
        s = date_str.replace("-", "").replace("/", "")
        if len(s) == 8:
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return date_str

    @staticmethod
    def _validate_symbol(symbol: str):
        """校验代码格式，Wind 接受 {6位数字}.{SH/SZ/...}。"""
        if not any(symbol.endswith(suf) for suf in _SUPPORTED_SUFFIX):
            raise InvalidParameterError(
                "wind",
                f"不支持的代码格式: {symbol}，应为 XXXXXX.SH/.SZ 等"
            )

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        """获取日线行情。

        使用 w.wsd 批量查询，fields 取 open/high/low/close/volume/amt/oi。
        """
        self._ensure_connected()
        price_adj = _ADJUST_MAP.get(adjust, "U")
        begin = self._format_date(start_date)
        end = self._format_date(end_date)
        fields = "open,high,low,close,volume,amt,oi"
        # PriceAdj 设置复权；usedf=True 让 wsd 返回 (ErrorCode, DataFrame)
        options = f"PriceAdj={price_adj};Fill=Previous"

        frames = []
        for symbol in symbols:
            self._validate_symbol(symbol)
            try:
                error, df = self.w.wsd(
                    codes=symbol,
                    fields=fields,
                    beginTime=begin,
                    endTime=end,
                    options=options,
                    usedf=True,
                )
            except Exception as e:
                raise NetworkError("wind", f"wsd 查询 {symbol} 异常: {e}") from e

            if error:
                # ErrorCode 非 0 视为失败
                raise DataSourceError(
                    "wind",
                    f"wsd 查询 {symbol} 失败，错误码: {error}"
                )
            if df is None or len(df) == 0:
                logger.warning("Wind 未返回 %s 的数据", symbol)
                continue

            d = df.copy()
            # wsd 返回的 DataFrame index 是日期，列名是大写的 OHLC
            d.index.name = "date"
            d = d.reset_index()
            # Wind wsd 返回的列名可能是 OPEN/HIGH/LOW/CLOSE/VOLUME/AMT/OI
            d.columns = [c.lower() for c in d.columns]
            if "oi" in d.columns:
                d = d.drop(columns=["oi"])
            d["date"] = pd.to_datetime(d["date"])
            d["code"] = symbol
            frames.append(d)

        if not frames:
            return _finalize(pd.DataFrame())
        return _finalize(pd.concat(frames, ignore_index=True))

    def get_stock_list(self) -> pd.DataFrame:
        """获取全市场 A 股列表。

        使用 w.wset("sectorconstituent", "sectorid=a001010100000000") 拉取
        沪深A股成分股（借鉴 Wind 常用 sectorid 约定）。
        """
        self._ensure_connected()
        try:
            error, df = self.w.wset(
                "sectorconstituent",
                "sectorid=a001010100000000;field=wind_code,sec_name,list_date",
                usedf=True,
            )
        except Exception as e:
            raise NetworkError("wind", f"wset 查询股票列表异常: {e}") from e

        if error or df is None or len(df) == 0:
            logger.warning("Wind 获取股票列表失败，返回空")
            return pd.DataFrame(columns=["code", "name", "industry", "list_date", "is_st"])

        out = pd.DataFrame()
        out["code"] = df.get("wind_code", df.iloc[:, 0])
        out["name"] = df.get("sec_name", df.iloc[:, 1] if df.shape[1] > 1 else None)
        out["list_date"] = df.get("list_date", None)
        if out["list_date"] is not None:
            out["list_date"] = pd.to_datetime(
                out["list_date"].astype(str), format="%Y%m%d", errors="coerce"
            )
        out["industry"] = None  # Wind 行业需另行调用 wss，此处留空
        out["is_st"] = out["name"].astype(str).str.contains("ST", na=False)
        return out

    def get_adj_factor(self, symbols, start_date, end_date):
        """复权因子：Wind wsd 已通过 PriceAdj 直接返回复权后价格，
        无需单独提供复权因子，返回空 DataFrame 保持接口一致。
        """
        return pd.DataFrame(columns=["code", "date", "adj_factor"])

    def get_financial(self, symbols, report_date, fields):
        """获取财务数据（可选实现）。

        Wind 财务数据通过 wss 单点查询，字段众多。此处提供基础估值字段
        （pe_ttm/pb/ps_ttm/dv_ratio）的查询，其他字段留空。
        """
        self._ensure_connected()
        standard_cols = [
            "code", "report_date", "pe_ttm", "pb", "ps_ttm", "dv_ratio",
            "roe", "roa", "gross_margin", "net_margin",
            "revenue_growth", "profit_growth",
            "debt_ratio", "current_ratio", "quick_ratio", "ocf",
            "industry", "name",
        ]
        if not symbols:
            return pd.DataFrame(columns=standard_cols)

        # Wind 财务指标字段名（部分常用）
        wind_fields = "sec_name,industry,pe_ttm,pb,ps_ttm,dv_ratio,roe,roa"
        codes = ",".join(symbols)
        trade_date = self._format_date(report_date)

        try:
            error, df = self.w.wss(
                codes=codes,
                fields=wind_fields,
                fdate=trade_date,
                usedf=True,
            )
        except Exception as e:
            logger.warning("Wind wss 财务数据查询异常: %s", e)
            return pd.DataFrame(columns=standard_cols)

        if error or df is None or len(df) == 0:
            return pd.DataFrame(columns=standard_cols)

        df = df.copy()
        # wss 返回的 DataFrame index 是 codes，reset_index 后作为 code 列
        df.index.name = "code"
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        out = pd.DataFrame()
        out["code"] = df.get("code", symbols)
        out["report_date"] = trade_date
        out["pe_ttm"] = df.get("pe_ttm")
        out["pb"] = df.get("pb")
        out["ps_ttm"] = df.get("ps_ttm")
        out["dv_ratio"] = df.get("dv_ratio")
        out["roe"] = df.get("roe")
        out["roa"] = df.get("roa")
        for c in ["gross_margin", "net_margin", "revenue_growth", "profit_growth",
                  "debt_ratio", "current_ratio", "quick_ratio", "ocf"]:
            out[c] = None
        out["industry"] = df.get("industry")
        out["name"] = df.get("sec_name")
        return out[standard_cols].reset_index(drop=True)
