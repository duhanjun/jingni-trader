"""
掘金量化 (GoldMiner / gm) 适配器 —— 真实实现

通过 gm.api 的 history 接口拉取 A股历史日线。
前置条件：
- 已安装 gm 包（pip install gm）
- 已设置环境变量 GM_TOKEN（掘金量化 token，https://www.myquant.cn/）
- 调用 history 需要有效 token（gm 历史数据走掘金云端 API）

代码格式：项目统一使用 '600000.SH' / '000001.SZ'，
gm.api 使用 'SHSE.600000' / 'SZSE.000001'，适配器内部完成转换。

复权：gm 的 adjust 参数为整数枚举（gm.enum）：ADJUST_NONE(0)=不复权 / ADJUST_PREV(1)=前复权 / ADJUST_POST(2)=后复权。

返回格式：新版 gm.api.history() 返回 list[dict]，eob 字段为带时区 datetime。
volume 单位为股，适配器内部自动转为手。
"""
import logging
import os
from typing import List, Optional
import pandas as pd

from ..base.base_data_provider import BaseDataProvider
from ..config import GM_TOKEN
from ..errors import DataSourceError


logger = logging.getLogger("gm-adapter")

from gm.enum import ADJUST_NONE, ADJUST_PREV, ADJUST_POST
_ADJ_MAP = {"hfq": ADJUST_POST, "qfq": ADJUST_PREV, "none": ADJUST_NONE, "": ADJUST_NONE}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
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


class GmAdapter(BaseDataProvider):
    """掘金量化 gm 适配器：基于 gm.api.history 拉取真实行情。"""

    def __init__(self):
        self.token = GM_TOKEN or os.environ.get("GM_TOKEN")
        if not self.token:
            logger.warning(
                "GM_TOKEN 未设置，GmAdapter 不可用。"
                "请到 https://www.myquant.cn/ 申请 token 并设置环境变量 GM_TOKEN"
            )
            self.available = False
            return
        try:
            import gm  # noqa
            self.available = True
        except ImportError:
            logger.warning("gm 包未安装，GmAdapter 不可用，请 pip install gm")
            self.available = False

    def _check_available(self):
        if not self.available:
            raise DataSourceError(
                "gm",
                "gm 包未安装或 GM_TOKEN 未设置，请先 pip install gm 并配置 GM_TOKEN"
            )

    @staticmethod
    def _to_gm(code: str) -> str:
        """'600000.SH' -> 'SHSE.600000'；'000001.SZ' -> 'SZSE.000001'。"""
        c = code.split(".")[0]
        if code.endswith(".SZ"):
            return "SZSE." + c
        return "SHSE." + c

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        self._check_available()
        from gm.api import set_token, history
        set_token(self.token)
        sd = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        ed = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        adj = _ADJ_MAP.get(adjust, ADJUST_NONE)
        fields = "eob,open,high,low,close,volume,amount"

        all_rows = []
        for code in symbols:
            sym = self._to_gm(code)
            try:
                raw = history(
                    symbol=sym, frequency="1d",
                    start_time=sd, end_time=ed,
                    fields=fields, adjust=adj
                )
            except Exception as e:
                raise DataSourceError("gm", f"获取 {code} 行情失败: {e}") from e
            if not raw:
                logger.warning("gm 未返回 %s 的数据", code)
                continue
            for record in raw:
                row = {
                    "code": code,
                    "date": record["eob"],
                    "open": record["open"],
                    "high": record["high"],
                    "low": record["low"],
                    "close": record["close"],
                    "volume": record["volume"] / 100.0,  # 股 -> 手
                    "amount": record["amount"],
                }
                all_rows.append(row)
        if not all_rows:
            return _finalize(pd.DataFrame())
        d = pd.DataFrame(all_rows)
        d["date"] = pd.to_datetime(d["date"])
        return _finalize(d)

    def get_stock_list(self) -> pd.DataFrame:
        self._check_available()
        from gm.api import set_token, get_instruments
        set_token(self.token)
        rows = []
        try:
            df = get_instruments(
                exchanges="SHSE,SZSE", sec_types=1,
                fields="symbol,sec_name,listed_date",
                skip_suspended=False, skip_st=False,
                df=True
            )
            if df is not None and len(df) > 0:
                for _, r in df.iterrows():
                    sym = r.get("symbol", "")
                    name = r.get("sec_name", "")
                    market = "SH" if str(sym).startswith("SHSE") else "SZ"
                    code = sym.split(".")[-1] + "." + market
                    rows.append({"code": code, "name": name,
                                 "is_st": "ST" in str(name)})
        except Exception as e:
            logger.warning("gm 获取股票列表失败（返回空）: %s", e)
            return pd.DataFrame(columns=["code", "name", "industry", "list_date", "is_st"])
        df = pd.DataFrame(rows, columns=["code", "name", "is_st"])
        df["industry"] = None
        df["list_date"] = None
        return df

    def get_adj_factor(self, symbols, start_date, end_date):
        return pd.DataFrame(columns=["code", "date", "adj_factor"])

    def get_financial(self, symbols, report_date, fields):
        return pd.DataFrame()
