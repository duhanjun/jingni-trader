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

    SUPPORTED_DATA_TYPES = {"daily", "financial"}

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
        """获取财务数据，返回统一标准 schema。

        使用掘金量化新版基本面接口:
        - stk_get_daily_valuation: 每日估值指标(pe_ttm/pb_mrq/ps_ttm/dy_ttm)
        - stk_get_finance_prime: 财务主要指标(roe/总资产/总负债/净利润/营收/
          经营现金流/营收同比/归母净利润同比)
        - get_instruments: 股票名称

        返回 DataFrame 包含标准字段:
            code, report_date, pe_ttm, pb, ps_ttm, dv_ratio,
            roe, roa, gross_margin, net_margin,
            revenue_growth, profit_growth,
            debt_ratio, current_ratio, quick_ratio, ocf,
            industry, name, disclosure_date
        """
        self._check_available()
        from gm.api import set_token, stk_get_daily_valuation, stk_get_finance_prime, get_instruments
        set_token(self.token)

        # P0-1 PIT 契约：末尾追加 disclosure_date
        # gm 无原生披露日接口，出口回填为 report_date（保守降级）
        standard_cols = [
            'code', 'report_date', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio',
            'roe', 'roa', 'gross_margin', 'net_margin',
            'revenue_growth', 'profit_growth',
            'debt_ratio', 'current_ratio', 'quick_ratio', 'ocf',
            'industry', 'name', 'disclosure_date',
        ]

        # 标准化报告期: '20240930' <-> '2024-09-30'
        period = report_date.replace('-', '')
        period_std = f"{period[:4]}-{period[4:6]}-{period[6:8]}" if len(period) == 8 else report_date

        rows = []
        for code in symbols:
            sym = self._to_gm(code)
            row = {col: None for col in standard_cols}
            row['code'] = code
            row['report_date'] = period
            # P0-1 PIT 契约：gm 无原生披露日，回填为 report_date（保守降级）
            row['disclosure_date'] = period

            # 1) 每日估值指标(PE/PB/PS/股息率): 在报告期附近取最接近的一日
            try:
                val_df = stk_get_daily_valuation(
                    symbol=sym, fields='pe_ttm,pb_mrq,ps_ttm,dy_ttm',
                    start_date=period_std, end_date=period_std, df=True
                )
                if val_df is not None and not val_df.empty:
                    v = val_df.iloc[-1]
                    row['pe_ttm'] = self._to_num(v.get('pe_ttm'))
                    row['pb'] = self._to_num(v.get('pb_mrq'))
                    row['ps_ttm'] = self._to_num(v.get('ps_ttm'))
                    row['dv_ratio'] = self._to_num(v.get('dy_ttm'))
            except Exception as e:
                logger.debug("gm 获取 %s 估值失败: %s", code, e)

            # 2) 财务主要指标: 按 rpt_date 匹配报告期
            try:
                prime_df = stk_get_finance_prime(
                    symbol=sym,
                    fields='roe,ttl_ast,ttl_liab,net_cf_oper,inc_oper_yoy,'
                           'net_prof_pcom_yoy,inc_oper,net_prof_pcom',
                    rpt_type=None, data_type=None,
                    start_date=period_std, end_date=period_std, df=True
                )
                if prime_df is not None and not prime_df.empty:
                    # 优先精确匹配 rpt_date，其次取最接近且 <= 报告期的一行
                    if 'rpt_date' in prime_df.columns:
                        prime_df = prime_df.sort_values('rpt_date')
                        exact = prime_df[prime_df['rpt_date'].astype(str) == period_std]
                        m = exact.iloc[0] if not exact.empty else prime_df.iloc[-1]
                    else:
                        m = prime_df.iloc[-1]
                    roe = self._to_num(m.get('roe'))
                    ttl_ast = self._to_num(m.get('ttl_ast'))
                    ttl_liab = self._to_num(m.get('ttl_liab'))
                    net_cf_oper = self._to_num(m.get('net_cf_oper'))
                    inc_oper = self._to_num(m.get('inc_oper'))
                    net_prof = self._to_num(m.get('net_prof_pcom'))
                    row['roe'] = roe
                    row['ocf'] = net_cf_oper
                    row['revenue_growth'] = self._to_num(m.get('inc_oper_yoy'))
                    row['profit_growth'] = self._to_num(m.get('net_prof_pcom_yoy'))
                    # 衍生比率(数值均以"元"为单位，比率换算为百分比)
                    if ttl_ast and ttl_ast != 0 and ttl_liab is not None:
                        row['debt_ratio'] = ttl_liab / ttl_ast * 100.0
                    if net_prof is not None and ttl_ast and ttl_ast != 0:
                        row['roa'] = net_prof / ttl_ast * 100.0
                    if net_prof is not None and inc_oper and inc_oper != 0:
                        row['net_margin'] = net_prof / inc_oper * 100.0
            except Exception as e:
                logger.debug("gm 获取 %s 财务主要指标失败: %s", code, e)

            # 3) 股票名称
            try:
                gi = get_instruments(symbols=sym, fields='symbol,sec_name', df=True)
                if gi is not None and not gi.empty:
                    row['name'] = str(gi.iloc[0].get('sec_name', '') or '')
            except Exception as e:
                logger.debug("gm 获取 %s 名称失败: %s", code, e)

            rows.append(row)

        out = pd.DataFrame(rows, columns=standard_cols)

        # 如果调用方指定了 fields，按需过滤列
        # P0-1 PIT 契约：code/report_date/disclosure_date 始终保留
        if fields:
            keep = ['code', 'report_date', 'disclosure_date'] + [f for f in fields if f in standard_cols]
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
