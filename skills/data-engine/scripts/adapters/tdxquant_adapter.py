"""
通达信 TdxQuant 适配器 —— 真实实现（基于 pytdx）

实现说明：
- 通达信官方 TdxQuant SDK（tqcenter）非公开分发，pip 无法直接安装。
- 本适配器使用社区公开的 pytdx 连接通达信行情服务器，拉取真实行情数据。
- pytdx 连接的是通达信公共行情服务器（与用户打开的通达信客户端同源数据），
  因此返回的是真实行情；缺点：行情服务器仅提供【不复权】原始价，
  复权需交由其它源（tushare/xtquant/akshare）完成。

连接策略：
- 依次尝试 best_ip.stock_ip 列表中的服务器（优先 gtjas/cjis/daton 等域名），
  第一个连通的即复用该连接拉取全部标的数据。

代码格式：项目统一使用 '600000.SH' / '000001.SZ'，本适配器内部转换为
pytdx 的 (market, code)：market 0=深圳 1=上海，code 为纯数字。
"""
import logging
import os
from typing import List, Optional
import pandas as pd

from ..base.base_data_provider import BaseDataProvider
from ..errors import DataSourceError


logger = logging.getLogger("tdxquant-adapter")


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


class TdxQuantAdapter(BaseDataProvider):
    """通达信行情适配器（pytdx 实现）。"""

    SUPPORTED_DATA_TYPES = {"daily", "financial"}

    def __init__(self):
        try:
            import pytdx  # noqa
            self.available = True
        except ImportError:
            logger.warning(
                "pytdx 未安装，TdxQuantAdapter 不可用。"
                "请 pip install pytdx（通达信行情库）。"
            )
            self.available = False

    def _check_available(self):
        if not self.available:
            raise DataSourceError(
                "tdxquant",
                "pytdx 未安装或无法连接通达信行情服务器，请 pip install pytdx"
            )

    @staticmethod
    def _split_code(code: str):
        """'600000.SH' -> (market=1, '600000')；'000001.SZ' -> (market=0, '000001')。"""
        code = code.strip()
        pure = code.split(".")[0]
        if code.endswith(".SZ") or pure.startswith(("0", "3")):
            market = 0
        else:
            market = 1
        return market, pure

    def _connect(self):
        from pytdx.hq import TdxHq_API
        from pytdx.util import best_ip as bip
        # 优先尝试已知稳定的域名，再回退到内置全量列表
        candidates = [
            ("jstdx.gtjas.com", 7709),
            ("shtdx.gtjas.com", 7709),
            ("sztdx.gtjas.com", 7709),
            ("hq.cjis.cn", 7709),
            ("hq1.daton.com.cn", 7709),
        ]
        try:
            candidates += [(x["ip"], int(x["port"])) for x in bip.stock_ip]
        except Exception:
            pass
        last_err = None
        for ip, port in candidates:
            try:
                api = TdxHq_API()
                if api.connect(ip, port, time_out=2):
                    logger.info("TdxQuant 已连接行情服务器 %s:%s", ip, port)
                    return api
                try:
                    api.disconnect()
                except Exception:
                    pass
            except Exception as e:
                last_err = e
        raise DataSourceError("tdxquant", f"无法连接任何通达信行情服务器: {last_err}")

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        self._check_available()
        from pytdx.params import TDXParams
        if adjust not in ("none", ""):
            logger.warning(
                "通达信行情服务器仅提供不复权数据，忽略 adjust=%s，返回不复权价", adjust
            )
        sd_dt = pd.to_datetime(start_date).normalize()
        ed_dt = pd.to_datetime(end_date).normalize()
        # 单次取足够覆盖区间的交易日（pytdx 内部自动分片拼接）
        count = max(800, (ed_dt - sd_dt).days + 10)

        api = self._connect()
        try:
            frames = []
            for code in symbols:
                market, pure = self._split_code(code)
                try:
                    bars = api.get_security_bars(
                        TDXParams.KLINE_TYPE_DAILY, market, pure, 0, count
                    )
                except Exception as e:
                    logger.warning("TdxQuant 获取 %s 失败: %s", code, e)
                    continue
                if not bars:
                    continue
                d = pd.DataFrame(list(bars))
                d["date"] = pd.to_datetime(d["datetime"]).dt.normalize()
                d = d[(d["date"] >= sd_dt) & (d["date"] <= ed_dt)]
                if d.empty:
                    continue
                d["vol"] = d["vol"] / 100.0  # pytdx 成交量单位为股，转为与 tushare/xtdata 一致的"手"
                d = d.rename(columns={"vol": "volume"})
                d["code"] = code
                frames.append(d[["date", "code", "open", "high", "low", "close",
                                  "volume", "amount"]])
            if not frames:
                return _finalize(pd.DataFrame())
            return _finalize(pd.concat(frames, ignore_index=True))
        finally:
            try:
                api.disconnect()
            except Exception:
                pass

    def get_stock_list(self) -> pd.DataFrame:
        self._check_available()
        api = self._connect()
        try:
            rows = []
            for market in (0, 1):
                suffix = ".SZ" if market == 0 else ".SH"
                start = 0
                while True:
                    batch = api.get_security_list(market, start)
                    if not batch:
                        break
                    for it in batch:
                        c = it.get("code", "")
                        name = it.get("name", "")
                        if not c:
                            continue
                        rows.append({
                            "code": c + suffix,
                            "name": name,
                            "is_st": "ST" in str(name),
                        })
                    if len(batch) < 1000:
                        break
                    start += len(batch)
            df = pd.DataFrame(rows, columns=["code", "name", "is_st"])
            df["industry"] = None
            df["list_date"] = None
            return df
        finally:
            try:
                api.disconnect()
            except Exception:
                pass

    def get_adj_factor(self, symbols, start_date, end_date):
        # 行情服务器不提供复权因子
        return pd.DataFrame(columns=["code", "date", "adj_factor"])

    def get_financial(self, symbols, report_date, fields):
        """获取财务数据，返回统一标准 schema。

        使用 pytdx 的 get_finance_info 接口获取基础财务数据(最新一期)。
        通达信行情服务器仅提供最新一期的累计财务数据，无法按指定报告期
        精确查询；此处取最新数据，并按报告期字段填充。PE/PB/PS 等估值
        指标及同比增速留空(由其他源补充)。

        返回 DataFrame 包含标准字段:
            code, report_date, pe_ttm, pb, ps_ttm, dv_ratio,
            roe, roa, gross_margin, net_margin,
            revenue_growth, profit_growth,
            debt_ratio, current_ratio, quick_ratio, ocf,
            industry, name, disclosure_date
        """
        self._check_available()

        # P0-1 PIT 契约：末尾追加 disclosure_date
        # tdxquant 无原生披露日接口，出口回填为 report_date（保守降级）
        standard_cols = [
            'code', 'report_date', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio',
            'roe', 'roa', 'gross_margin', 'net_margin',
            'revenue_growth', 'profit_growth',
            'debt_ratio', 'current_ratio', 'quick_ratio', 'ocf',
            'industry', 'name', 'disclosure_date',
        ]

        # 标准化报告期: '2024-09-30' -> '20240930'
        period = report_date.replace('-', '')

        api = self._connect()
        try:
            # 预取各市场前若干批股票名称(用于 best-effort 名称填充)
            name_map = {}
            try:
                for market in (0, 1):
                    start = 0
                    for _ in range(3):  # 取前 3 批(约 3000 只)
                        batch = api.get_security_list(market, start) or []
                        if not batch:
                            break
                        for it in batch:
                            c = it.get("code", "")
                            if c:
                                suffix = ".SZ" if market == 0 else ".SH"
                                name_map[c + suffix] = it.get("name", "")
                        if len(batch) < 1000:
                            break
                        start += len(batch)
            except Exception as e:
                logger.debug("TdxQuant 获取股票名称失败: %s", e)

            rows = []
            for code in symbols:
                market, pure = self._split_code(code)
                row = {col: None for col in standard_cols}
                row['code'] = code
                row['report_date'] = period
                # P0-1 PIT 契约：tdxquant 无原生披露日，回填为 report_date（保守降级）
                row['disclosure_date'] = period
                row['name'] = name_map.get(code, '')

                try:
                    fin = api.get_finance_info(market, pure)
                except Exception as e:
                    logger.warning("TdxQuant 获取 %s 财务失败: %s", code, e)
                    rows.append(row)
                    continue

                if not fin:
                    rows.append(row)
                    continue

                zongzichan = self._to_num(fin.get('zongzichan'))
                jingzichan = self._to_num(fin.get('jingzichan'))
                zhuyingshouru = self._to_num(fin.get('zhuyingshouru'))
                jinglirun = self._to_num(fin.get('jinglirun'))
                zhuyinglirun = self._to_num(fin.get('zhuyinglirun'))
                ocf = self._to_num(fin.get('jingyingxianjinliu'))

                # 经营现金流净额
                row['ocf'] = ocf

                # 资产负债率 = 总负债 / 总资产; 总负债 ≈ 总资产 - 净资产
                if zongzichan and zongzichan != 0 and jingzichan is not None:
                    row['debt_ratio'] = (zongzichan - jingzichan) / zongzichan * 100.0
                # ROE(未年化) = 净利润 / 净资产
                if jinglirun is not None and jingzichan and jingzichan != 0:
                    row['roe'] = jinglirun / jingzichan * 100.0
                # ROA = 净利润 / 总资产
                if jinglirun is not None and zongzichan and zongzichan != 0:
                    row['roa'] = jinglirun / zongzichan * 100.0
                # 销售净利率 = 净利润 / 营业收入
                if jinglirun is not None and zhuyingshouru and zhuyingshouru != 0:
                    row['net_margin'] = jinglirun / zhuyingshouru * 100.0
                # 主营利润率(近似毛利率) = 主营利润 / 营业收入
                if zhuyinglirun is not None and zhuyingshouru and zhuyingshouru != 0:
                    row['gross_margin'] = zhuyinglirun / zhuyingshouru * 100.0

                rows.append(row)
        finally:
            try:
                api.disconnect()
            except Exception:
                pass

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
