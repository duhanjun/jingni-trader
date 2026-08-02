"""
同花顺 iFinD 适配器

接入方法：
- 通过 THS_iFinDLogin(username, password) 登录（依赖 iFinDPy 包）
- 日线数据用 THS_HQ(symbol, indicators, params, start, end)
  返回对象含 .errorcode 和 .data (DataFrame)
- indicators 用分号分隔: "open;high;low;close;volume;amount;openInterest"
- params: "Fill:Original"（股票加 ",CPS:2" 表示前复权）
- 代码格式: {symbol}.{exchange}（如 600000.SH / 000001.SZ），与 jingni-trader 约定一致

前置条件：
- 已购买同花顺 iFinD 数据服务权限并获得 iFinDPy 包
- 已配置 IFIND_USERNAME / IFIND_PASSWORD 环境变量

复权方式（iFinD 的 CPS 参数）：
    qfq -> CPS:2（前复权，分红再投）
    hfq -> CPS:1（后复权）
    none-> 不设置 CPS（原始价）
"""
import logging
from typing import List, Optional
import pandas as pd

from ..base.base_data_provider import BaseDataProvider
from ..config import IFIND_USERNAME, IFIND_PASSWORD
from ..errors import DataSourceError, NetworkError, InvalidParameterError


logger = logging.getLogger("ifind-adapter")


# jingni-trader adjust 语义 -> iFinD CPS 取值
_ADJUST_MAP = {
    "hfq": "1",   # 后复权
    "qfq": "2",   # 前复权（分红再投）
    "none": None, # 不复权
    "": None,
}


_SUPPORTED_SUFFIX = {".SH", ".SZ", ".CFE", ".SHF", ".CZC", ".DCE"}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """把 iFinD 返回的裸行情统一成下游期望的标准 schema。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "code", "date", "open", "high", "low", "close", "volume", "amount",
            "pre_close", "change_pct", "turnover_rate",
            "is_st", "is_limit_up", "is_limit_down", "listed_days"
        ])
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "openinterest" in df.columns:
        df = df.drop(columns=["openinterest"])
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


class IfindAdapter(BaseDataProvider):
    """同花顺 iFinD 适配器：通过 iFinDPy 拉取 A股行情。

    依赖 iFinDPy 包（由同花顺分发，不在 PyPI 公开发布），
    需配置 IFIND_USERNAME / IFIND_PASSWORD 环境变量。
    """

    SUPPORTED_DATA_TYPES = {"daily", "financial"}

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        username = username or IFIND_USERNAME
        password = password or IFIND_PASSWORD
        if not username or not password:
            raise DataSourceError(
                "ifind",
                "iFinD 账号或密码未设置，请配置环境变量 IFIND_USERNAME / IFIND_PASSWORD"
            )
        try:
            from iFinDPy import THS_iFinDLogin, THS_HQ, THSData, THS_BD
            self._THS_iFinDLogin = THS_iFinDLogin
            self._THS_HQ = THS_HQ
            self._THSData = THSData
            self._THS_BD = THS_BD
        except ImportError as e:
            raise DataSourceError(
                "ifind",
                f"iFinDPy 包未安装: {e}。"
                f"请购买同花顺 iFinD 数据服务并安装 iFinDPy。"
            ) from e

        self.username = username
        self.password = password
        self._inited = False
        self._login()

    def _login(self):
        """登录 iFinD。

        错误码约定(同花顺官方):
        - 0: 登录成功
        - -201: 重复登录(视为成功,官方示例 `if thsLogin in {0, -201}` 判定为成功)
        - -2: 用户名或密码错误
        - 其他: 登录失败
        """
        if self._inited:
            return
        try:
            code = self._THS_iFinDLogin(self.username, self.password)
            # 0=成功, -201=重复登录(同花顺官方视为成功)
            if code not in (0, -201):
                raise NetworkError(
                    "ifind",
                    f"iFinD 登录失败，错误码: {code}"
                )
            self._inited = True
            logger.info("iFinD 登录成功, 错误码: %s", code)
        except DataSourceError:
            raise
        except Exception as e:
            raise NetworkError("ifind", f"iFinD 登录异常: {e}") from e

    def _ensure_inited(self):
        if not self._inited:
            self._login()

    @staticmethod
    def _format_datetime(date_str: str) -> str:
        """把 YYYY-MM-DD 或 YYYYMMDD 统一为 iFinD 期望的 YYYY-MM-DD HH:MM:SS。"""
        s = date_str.replace("-", "").replace("/", "")
        if len(s) == 8:
            return f"{s[:4]}-{s[4:6]}-{s[6:8]} 00:00:00"
        return date_str

    @staticmethod
    def _validate_symbol(symbol: str):
        if not any(symbol.endswith(suf) for suf in _SUPPORTED_SUFFIX):
            raise InvalidParameterError(
                "ifind",
                f"不支持的代码格式: {symbol}，应为 XXXXXX.SH/.SZ 等"
            )

    @staticmethod
    def _is_stock(symbol: str) -> bool:
        """判断是否为股票（用于决定是否加前复权参数）。"""
        return symbol.endswith(".SH") or symbol.endswith(".SZ")

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        """获取日线行情。

        使用 THS_HQ 查询，indicators 用分号分隔。
        股票自动加 CPS 参数实现复权。
        """
        self._ensure_inited()
        cps = _ADJUST_MAP.get(adjust, None)
        begin = self._format_datetime(start_date)
        end = self._format_datetime(end_date)
        indicators = "open;high;low;close;volume;amount;openInterest"

        frames = []
        for symbol in symbols:
            ifind_symbol = symbol.upper()
            self._validate_symbol(ifind_symbol)

            # 构造 params
            params = "Fill:Original"
            if self._is_stock(ifind_symbol) and cps:
                params += f",CPS:{cps}"

            try:
                result = self._THS_HQ(
                    ifind_symbol,
                    indicators,
                    params,
                    begin,
                    end,
                )
            except Exception as e:
                raise NetworkError("ifind", f"THS_HQ 查询 {symbol} 异常: {e}") from e

            if result.errorcode:
                raise DataSourceError(
                    "ifind",
                    f"THS_HQ 查询 {symbol} 失败，错误码: {result.errorcode}"
                )
            if result.data is None or len(result.data) == 0:
                logger.warning("iFinD 未返回 %s 的数据", symbol)
                continue

            d = result.data.copy()
            # iFinD 返回的 time 列可能是 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM"
            if "time" in d.columns:
                d["date"] = pd.to_datetime(d["time"], errors="coerce")
                d = d.drop(columns=["time"])
            d["code"] = symbol
            # 列名标准化为小写
            d.columns = [c.lower() for c in d.columns]
            if "openinterest" in d.columns:
                d = d.drop(columns=["openinterest"])
            frames.append(d)

        if not frames:
            return _finalize(pd.DataFrame())
        return _finalize(pd.concat(frames, ignore_index=True))

    def get_stock_list(self) -> pd.DataFrame:
        """获取全市场 A 股列表。

        iFinD 提供 THS_BasicCenter / THS_DateSequence 等接口拉取成分股，
        但字段映射较复杂。此处采用简化实现：通过 THS_HQ 拉取沪深300
        作为占位，实际生产环境建议调用 THS_BasicCenter。
        """
        self._ensure_inited()
        logger.warning(
            "iFinD get_stock_list 采用简化实现，仅返回沪深300成分股。"
            "如需全市场列表，请使用 tushare/baostock/akshare 适配器。"
        )
        try:
            from iFinDPy import THS_BasicCenter
            df = THS_BasicCenter("沪深A股")
        except Exception as e:
            logger.warning("iFinD 获取股票列表失败: %s", e)
            return pd.DataFrame(columns=["code", "name", "industry", "list_date", "is_st"])

        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["code", "name", "industry", "list_date", "is_st"])

        out = pd.DataFrame()
        out["code"] = df.iloc[:, 0] if df.shape[1] > 0 else None
        out["name"] = df.iloc[:, 1] if df.shape[1] > 1 else None
        out["industry"] = None
        out["list_date"] = None
        out["is_st"] = out["name"].astype(str).str.contains("ST", na=False)
        return out

    def get_adj_factor(self, symbols, start_date, end_date):
        """复权因子：iFinD 已通过 CPS 参数直接返回复权后价格，
        无需单独提供复权因子，返回空 DataFrame 保持接口一致。
        """
        return pd.DataFrame(columns=["code", "date", "adj_factor"])

    def get_financial(self, symbols, report_date, fields):
        """获取财务数据，返回统一标准 schema。

        使用 iFinD 的 THS_BD 基础数据接口逐指标查询单期财务数据。
        THS_BD 不支持分号批量多指标(会返回 -209)，故逐个指标调用。

        已验证可用的指标(均返回 errorcode=0):
        - ths_pe_ttm_stock(PE TTM)、ths_pb_stock(PB)、ths_ps_ttm_stock(PS TTM)
        - ths_roe_stock(ROE)、ths_roa_stock(ROA)、ths_dividend_ratio_stock(股息率)
        - ths_or_yoy_stock(营收同比)、ths_np_yoy_stock(净利润同比)
        - ths_current_ratio_stock(流动比率)、ths_quick_ratio_stock(速动比率)
        - ths_stock_short_name_stock(股票简称)

        暂无可用指标的: gross_margin / net_margin / debt_ratio / ocf / industry(留空)

        返回 DataFrame 包含标准字段:
            code, report_date, pe_ttm, pb, ps_ttm, dv_ratio,
            roe, roa, gross_margin, net_margin,
            revenue_growth, profit_growth,
            debt_ratio, current_ratio, quick_ratio, ocf,
            industry, name, disclosure_date
        """
        self._ensure_inited()

        # P0-1 PIT 契约：末尾追加 disclosure_date
        # ifind 无原生披露日接口，出口回填为 report_date（保守降级）
        standard_cols = [
            "code", "report_date", "pe_ttm", "pb", "ps_ttm", "dv_ratio",
            "roe", "roa", "gross_margin", "net_margin",
            "revenue_growth", "profit_growth",
            "debt_ratio", "current_ratio", "quick_ratio", "ocf",
            "industry", "name", "disclosure_date",
        ]

        # 标准化报告期: '20240930' -> '2024-09-30'(iFinD 期望的日期格式)
        period = report_date.replace("-", "")
        period_std = f"{period[:4]}-{period[4:6]}-{period[6:8]}" if len(period) == 8 else report_date

        # 指标 -> 标准 schema 字段映射(逐个 THS_BD 调用)
        indicator_map = {
            "pe_ttm": "ths_pe_ttm_stock",
            "pb": "ths_pb_stock",
            "ps_ttm": "ths_ps_ttm_stock",
            "roe": "ths_roe_stock",
            "roa": "ths_roa_stock",
            "dv_ratio": "ths_dividend_ratio_stock",
            "revenue_growth": "ths_or_yoy_stock",
            "profit_growth": "ths_np_yoy_stock",
            "current_ratio": "ths_current_ratio_stock",
            "quick_ratio": "ths_quick_ratio_stock",
            "name": "ths_stock_short_name_stock",
        }

        rows = []
        for symbol in symbols:
            ifind_symbol = symbol.upper()
            self._validate_symbol(ifind_symbol)
            row = {col: None for col in standard_cols}
            row["code"] = symbol
            row["report_date"] = period
            # P0-1 PIT 契约：ifind 无原生披露日，回填为 report_date（保守降级）
            row["disclosure_date"] = period

            for dst_col, indicator in indicator_map.items():
                try:
                    result = self._THS_BD(ifind_symbol, indicator, period_std)
                except Exception as e:
                    raise NetworkError("ifind", f"THS_BD 查询 {symbol} {indicator} 异常: {e}") from e

                # errorcode 非 0 视为该指标无数据(如 -209)，跳过留空
                if getattr(result, "errorcode", None):
                    logger.debug("iFinD %s %s 无数据(errorcode=%s)",
                                 symbol, indicator, result.errorcode)
                    continue

                data = getattr(result, "data", None)
                if data is not None and not data.empty and indicator in data.columns:
                    raw_val = data.iloc[0][indicator]
                    # name 为字符串字段，其余为数值字段
                    row[dst_col] = str(raw_val) if dst_col == "name" else self._to_num(raw_val)

            rows.append(row)

        out = pd.DataFrame(rows, columns=standard_cols)

        # 如果调用方指定了 fields，按需过滤列
        # P0-1 PIT 契约：code/report_date/disclosure_date 始终保留
        if fields:
            keep = ["code", "report_date", "disclosure_date"] + [f for f in fields if f in standard_cols]
            keep = list(dict.fromkeys(keep))
            out = out[keep]

        return out.reset_index(drop=True)

    @staticmethod
    def _to_num(val):
        """安全转换为 float"""
        if val is None:
            return None
        try:
            import math
            f = float(val)
            return None if (isinstance(f, float) and math.isnan(f)) else f
        except (TypeError, ValueError):
            return None
