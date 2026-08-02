"""
AkShare 数据源适配器
免费另类数据源，支持龙虎榜、大宗交易、北向资金等特色数据
"""
import logging
from typing import List, Optional
import pandas as pd
import akshare as ak

from ..base.base_data_provider import BaseDataProvider


class AkshareAdapter(BaseDataProvider):
    """AkShare 适配器，专注于另类数据和补充数据源"""

    SUPPORTED_DATA_TYPES = {"daily", "financial", "capital_flow",
                            "dragon_tiger", "shareholder"}

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        """
        获取日线行情（通过 stock_zh_a_hist 接口）

        注意: AkShare 每日批量获取全市场数据效率较低，建议仅用于少量股票
              或作为 Tushare/BaoStock 的补充源。
        """
        self._logger.info(f"AkShare: 获取 {len(symbols)} 只股票日线数据")
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        all_data = []
        for code in symbols:
            try:
                ticker = self._extract_ticker(code)
                period = "daily"
                df = ak.stock_zh_a_hist(
                    symbol=ticker,
                    period=period,
                    start_date=start,
                    end_date=end,
                    adjust=adjust
                )
                if df is None or df.empty:
                    continue
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "change_pct",
                    "涨跌额": "change",
                    "换手率": "turnover_rate",
                })
                df["code"] = code
                all_data.append(df)
            except Exception as e:
                self._logger.warning(f"获取 {code} 数据失败: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
        result["open"] = pd.to_numeric(result["open"], errors="coerce")
        result["high"] = pd.to_numeric(result["high"], errors="coerce")
        result["low"] = pd.to_numeric(result["low"], errors="coerce")
        result["close"] = pd.to_numeric(result["close"], errors="coerce")
        result["volume"] = pd.to_numeric(result["volume"], errors="coerce")
        result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
        result["change_pct"] = pd.to_numeric(result["change_pct"], errors="coerce")
        result["turnover_rate"] = pd.to_numeric(result["turnover_rate"], errors="coerce")
        return result

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取全市场股票列表，并附加申万行业分类。

        在原有 stock_info_a_code_name 基础上，通过
        ak.stock_board_industry_summary_ths() / ak.stock_individual_info_em()
        尝试补齐 industry 字段（取不到时留空）。
        """
        try:
            df = ak.stock_info_a_code_name()
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={"code": "code", "name": "name"})
            df["is_st"] = df["name"].str.contains("ST", na=False)
            # list_date / industry 默认留空，下方尝试补齐
            if "list_date" not in df.columns:
                df["list_date"] = ""
            if "industry" not in df.columns:
                df["industry"] = ""
        except Exception as e:
            self._logger.warning(f"获取股票列表失败: {e}")
            return pd.DataFrame()

        # 尝试获取行业分类（申万/同花顺概念板块）
        try:
            industry_df = ak.stock_board_industry_summary_ths()
            if industry_df is not None and not industry_df.empty:
                # 找到板块名/成分股代码列
                # 同花顺板块成分股：每行一只股票 + 板块名
                # 注意：stock_board_industry_summary_ths 返回板块聚合，需要展开
                # 此处简化：直接通过 stock_individual_info_em 按需补齐
                pass
        except Exception as e:
            self._logger.debug(f"stock_board_industry_summary_ths 获取失败: {e}")

        # 逐只补齐 industry（性能较弱，但作为补充源可接受）
        # 仅在 industry 列全为空时触发，避免覆盖已有数据
        industry_col = df.get("industry", pd.Series(dtype=str))
        industry_empty = industry_col.isna().all() or (industry_col.astype(str).str.strip() == "").all()
        if industry_empty:
            self._logger.info("AkShare: 尝试逐只获取行业信息")
            industries = []
            for code in df["code"].tolist():
                ind = ""
                try:
                    info = ak.stock_individual_info_em(symbol=code)
                    if info is not None and not info.empty:
                        # stock_individual_info_em 返回两列: item / value
                        row = info[info["item"].astype(str).str.contains("行业", na=False)]
                        if not row.empty:
                            ind = str(row.iloc[0]["value"]).strip()
                except Exception:
                    pass
                industries.append(ind)
            df["industry"] = industries

        return df

    def get_adj_factor(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取复权因子

        AkShare 无直接复权因子接口，此处返回空 DataFrame。
        建议使用 Tushare 或 BaoStock 获取复权因子。
        """
        self._logger.warning("AkShare 不直接提供复权因子，请使用后复权价格")
        return pd.DataFrame()

    @staticmethod
    def _parse_pct(val):
        """解析百分比字符串/数值：'5.23%' -> 5.23, '0.0523' -> 0.0523"""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if not s or s in ("False", "nan", "None", "--"):
            return None
        if s.endswith("%"):
            try:
                return float(s[:-1])
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None

    def get_financial(
        self,
        symbols: List[str],
        report_date: str,
        fields: List[str]
    ) -> pd.DataFrame:
        """
        获取财务数据，返回统一标准 schema。

        改进：优先使用 stock_financial_analysis_indicator（86 列，字段最丰富），
        回退到 stock_financial_abstract_ths（同花顺财务摘要，25 列）。

        使用:
        - ak.stock_financial_analysis_indicator(): 财务分析指标（新浪，字段最全）
        - ak.stock_financial_abstract_ths(): 财务摘要（同花顺，回退）
        - ak.stock_a_indicator_lg(): 估值指标（PE/PB/PS/股息率）
        - ak.stock_individual_info_em(): 行业、股票名称

        返回 DataFrame 包含标准字段:
            code, report_date, pe_ttm, pb, ps_ttm, dv_ratio,
            roe, roa, gross_margin, net_margin,
            revenue_growth, profit_growth,
            debt_ratio, current_ratio, quick_ratio, ocf,
            industry, name, disclosure_date
        """
        # P0-1 PIT 契约：末尾追加 disclosure_date
        # akshare 无原生披露日接口，出口回填为 report_date（保守降级）
        standard_cols = [
            'code', 'report_date', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio',
            'roe', 'roa', 'gross_margin', 'net_margin',
            'revenue_growth', 'profit_growth',
            'debt_ratio', 'current_ratio', 'quick_ratio', 'ocf',
            'industry', 'name', 'disclosure_date',
        ]

        # 标准化报告期格式: '2024-09-30' -> '20240930'
        period = report_date.replace('-', '')

        rows = []
        for code in symbols:
            ticker = self._extract_ticker(code)
            row = {col: None for col in standard_cols}
            row['code'] = code
            row['report_date'] = period
            # P0-1 PIT 契约：akshare 无原生披露日，回填为 report_date（保守降级）
            row['disclosure_date'] = period

            # 1a) 优先：stock_financial_analysis_indicator（新浪，字段最全）
            try:
                start_year = period[:4] if period else "2020"
                fina = ak.stock_financial_analysis_indicator(symbol=ticker, start_year=start_year)
                if fina is not None and not fina.empty:
                    # 按报告期匹配：日期列为 '日期'，格式 'YYYY-MM-DD'
                    fina['period_str'] = fina['日期'].astype(str).str.replace('-', '')
                    # 优先精确，其次取最接近且 <= period 的
                    matched = fina[fina['period_str'] == period]
                    if matched.empty:
                        before = fina[fina['period_str'] <= period]
                        matched = before.tail(1) if not before.empty else fina.tail(1)
                    if not matched.empty:
                        m = matched.iloc[0]
                        # 新浪字段映射（含 % 字符串需解析）
                        if row.get('roe') is None:
                            row['roe'] = self._parse_pct(m.get('净资产收益率(%)'))
                        if row.get('roa') is None:
                            # 新浪提供 "总资产利润率(%)" 与 "总资产净利润率(%)"，取后者更接近 ROA
                            row['roa'] = self._parse_pct(m.get('总资产净利润率(%)')) or self._parse_pct(m.get('总资产利润率(%)'))
                        if row.get('gross_margin') is None:
                            row['gross_margin'] = self._parse_pct(m.get('销售毛利率(%)'))
                        if row.get('net_margin') is None:
                            row['net_margin'] = self._parse_pct(m.get('销售净利率(%)'))
                        if row.get('revenue_growth') is None:
                            row['revenue_growth'] = self._parse_pct(m.get('主营业务收入增长率(%)'))
                        if row.get('profit_growth') is None:
                            row['profit_growth'] = self._parse_pct(m.get('净利润增长率(%)'))
                        if row.get('debt_ratio') is None:
                            row['debt_ratio'] = self._parse_pct(m.get('资产负债率(%)'))
                        if row.get('current_ratio') is None:
                            row['current_ratio'] = self._parse_pct(m.get('流动比率'))
                        if row.get('quick_ratio') is None:
                            row['quick_ratio'] = self._parse_pct(m.get('速动比率'))
                        if row.get('dv_ratio') is None:
                            row['dv_ratio'] = self._parse_pct(m.get('股息发放率(%)'))
            except Exception as e:
                self._logger.debug(f"获取 {code} 财务分析指标失败: {e}")

            # 1b) 回退：stock_financial_abstract_ths（同花顺）
            try:
                fina = ak.stock_financial_abstract_ths(symbol=ticker)
                if fina is not None and not fina.empty:
                    # 同花顺字段映射到标准 schema
                    col_map = {
                        '净资产收益率': 'roe',
                        '销售毛利率': 'gross_margin',
                        '销售净利率': 'net_margin',
                        '营业总收入同比增长率': 'revenue_growth',
                        '净利润同比增长率': 'profit_growth',
                        '资产负债率': 'debt_ratio',
                        '流动比率': 'current_ratio',
                        '速动比率': 'quick_ratio',
                        '每股经营现金流': 'ocf',
                    }
                    # 寻找报告期列
                    period_col = None
                    for c in fina.columns:
                        if '报告期' in str(c) or '日期' in str(c):
                            period_col = c
                            break

                    matched_row = None
                    if period_col is not None:
                        # 优先精确匹配，其次回退到最接近且 <= period 的报告期
                        best_row = None
                        best_period = ""
                        for _, r in fina.iterrows():
                            rp = str(r.get(period_col, '')).replace('-', '').replace('/', '')
                            if rp == period:
                                matched_row = r
                                break
                            if rp and rp <= period and rp > best_period:
                                best_period = rp
                                best_row = r
                        if matched_row is None and best_row is not None:
                            matched_row = best_row
                    elif len(fina) >= 1:
                        matched_row = fina.iloc[0]
                    # 兜底：仍无匹配时取第一行
                    if matched_row is None and len(fina) > 0:
                        matched_row = fina.iloc[0]

                    if matched_row is not None:
                        for src_col, dst_col in col_map.items():
                            if row.get(dst_col) is None and src_col in matched_row.index:
                                row[dst_col] = self._parse_pct(matched_row[src_col])
            except Exception as e:
                self._logger.debug(f"获取 {code} 财务摘要失败: {e}")

            # 2) 估值指标（PE/PB/PS/股息率）：取最近一条
            try:
                val_df = ak.stock_a_indicator_lg(symbol=ticker)
                if val_df is not None and not val_df.empty:
                    # 取最接近 period 的一行（trade_date 字段）
                    if 'trade_date' in val_df.columns:
                        val_df['trade_date_str'] = val_df['trade_date'].astype(str).str.replace('-', '')
                        # 优先精确匹配，否则取最后一条
                        exact = val_df[val_df['trade_date_str'] == period]
                        latest = exact.iloc[-1] if not exact.empty else val_df.iloc[-1]
                    else:
                        latest = val_df.iloc[-1]

                    for src, dst in [
                        ('pe_ttm', 'pe_ttm'),
                        ('pe', 'pe_ttm'),
                        ('pb', 'pb'),
                        ('ps_ttm', 'ps_ttm'),
                        ('ps', 'ps_ttm'),
                        ('dv_ratio', 'dv_ratio'),
                        ('dv_ttm', 'dv_ratio'),
                    ]:
                        if src in latest.index and row.get(dst) is None:
                            try:
                                row[dst] = pd.to_numeric(latest[src], errors='coerce')
                            except Exception:
                                pass
            except Exception as e:
                self._logger.debug(f"获取 {code} 估值数据失败: {e}")

            # 3) 行业 + 名称
            try:
                info = ak.stock_individual_info_em(symbol=ticker)
                if info is not None and not info.empty:
                    # 行业
                    ind_row = info[info["item"].astype(str).str.contains("行业", na=False)]
                    if not ind_row.empty:
                        row['industry'] = str(ind_row.iloc[0]["value"]).strip()
                    # 股票简称
                    name_row = info[info["item"].astype(str).str.contains("股票简称", na=False)]
                    if not name_row.empty:
                        row['name'] = str(name_row.iloc[0]["value"]).strip()
            except Exception as e:
                self._logger.debug(f"获取 {code} 个体信息失败: {e}")

            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=standard_cols)

        out = pd.DataFrame(rows, columns=standard_cols)

        # 如果调用方指定了 fields，按需过滤列
        # P0-1 PIT 契约：code/report_date/disclosure_date 始终保留
        if fields:
            keep = ['code', 'report_date', 'disclosure_date'] + [f for f in fields if f in standard_cols]
            keep = list(dict.fromkeys(keep))  # 去重保序
            out = out[keep]

        return out.reset_index(drop=True)

    def get_lhb_data(self, trade_date: str) -> pd.DataFrame:
        """
        获取龙虎榜数据（AkShare 特色功能）

        参数:
            trade_date: 交易日期 YYYYMMDD 或 YYYY-MM-DD

        返回:
            DataFrame 包含龙虎榜上榜股票明细

        注意: 旧接口 ak.stock_sina_lhb_detail_daily 已在新版 akshare 中移除，
              改用 ak.stock_lhb_detail_em（东方财富龙虎榜，按日期区间查询）。
        """
        trade_date = trade_date.replace("-", "")
        try:
            # 新接口：stock_lhb_detail_em 按 start_date/end_date 查询
            df = ak.stock_lhb_detail_em(start_date=trade_date, end_date=trade_date)
            if df is not None and not df.empty:
                df["trade_date"] = trade_date
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            self._logger.warning(f"获取龙虎榜数据失败: {e}")
            return pd.DataFrame()

    def get_block_trade(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取大宗交易数据（AkShare 特色功能）

        参数:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD

        返回:
            DataFrame 大宗交易明细
        """
        try:
            df = ak.stock_dzjy_mrmx(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                symbol="沪深A股"
            )
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            self._logger.warning(f"获取大宗交易数据失败: {e}")
            return pd.DataFrame()

    def get_north_flow(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取北向资金流向（AkShare 特色功能）

        参数:
            start_date: 开始日期
            end_date: 结束日期

        返回:
            DataFrame 北向资金每日净买入（按个股）
        """
        try:
            df = ak.stock_hsgt_individual_em(
                symbol="沪股通",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", "")
            )
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            self._logger.warning(f"获取北向资金数据失败: {e}")
            return pd.DataFrame()

    def _extract_ticker(self, code: str) -> str:
        """从 000001.SZ 格式中提取纯数字代码"""
        if "." in code:
            return code.split(".")[0]
        return code

    # ================================================================
    # 统一数据类型接口（供 fetch_by_type 调用）
    # ================================================================

    def get_capital_flow(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取资金面数据（主力资金流向 + 北向资金）

        改进：
        1. 加入指数退避重试（akshare 资金流接口易被远端断连）
        2. 单只失败不影响其他股票
        3. 北向资金合并逻辑保留为占位

        返回 DataFrame 标准字段:
            code, date, main_net_inflow, main_net_inflow_5d,
            super_large_net, large_net, medium_net, small_net,
            north_net_inflow
        """
        import time
        from datetime import datetime, timedelta
        import numpy as np

        all_rows = []
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        def _fetch_with_retry(func, max_retries=3, base_delay=1.0):
            """指数退避重试包装：针对 akshare 网络抖动"""
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func()
                except Exception as e:
                    last_exc = e
                    # 仅对网络类错误重试
                    if "Connection" in type(e).__name__ or "RemoteDisconnected" in str(e):
                        delay = base_delay * (2 ** attempt)
                        self._logger.debug(f"重试 {attempt+1}/{max_retries} (延迟 {delay}s): {e}")
                        time.sleep(delay)
                        continue
                    # 非网络错误直接抛出
                    raise
            raise last_exc

        for code in symbols:
            ticker = self._extract_ticker(code)
            market = "sh" if code.startswith("6") else "sz"
            try:
                # 主力资金流向（带重试）
                df = _fetch_with_retry(
                    lambda: ak.stock_individual_fund_flow(stock=ticker, market=market)
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "日期": "date", "收盘价": "close",
                        "涨跌幅": "change_pct",
                        "主力净流入-净额": "main_net_inflow",
                        "主力净流入-净占比": "main_net_ratio",
                        "超大单净流入-净额": "super_large_net",
                        "大单净流入-净额": "large_net",
                        "中单净流入-净额": "medium_net",
                        "小单净流入-净额": "small_net",
                    })
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    df["code"] = code
                    # 过滤日期范围
                    if "date" in df.columns:
                        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
                    if not df.empty:
                        all_rows.append(df)
            except Exception as e:
                self._logger.warning(f"获取 {code} 资金流向失败（已重试）: {e}")

        if not all_rows:
            return pd.DataFrame()

        result = pd.concat(all_rows, ignore_index=True)

        # 计算5日主力净流入均值
        result = result.sort_values(["code", "date"])
        if "main_net_inflow" in result.columns:
            result["main_net_inflow_5d"] = result.groupby("code")["main_net_inflow"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

        # 北向资金（个股层面，可选字段）
        try:
            for board in ["沪股通", "深股通"]:
                north_df = self.get_north_flow(start_date, end_date)
                if north_df is not None and not north_df.empty:
                    # 合并到 result（简化处理，北向资金为可选字段）
                    pass
        except Exception as e:
            self._logger.warning(f"获取北向资金失败: {e}")

        # 确保标准列存在
        for col in ["main_net_inflow", "main_net_inflow_5d", "super_large_net",
                     "large_net", "medium_net", "small_net", "north_net_inflow"]:
            if col not in result.columns:
                result[col] = np.nan

        return result

    def get_dragon_tiger(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取龙虎榜数据（近7日区间查询）

        改进：使用 stock_lhb_detail_em 批量区间查询，避免逐日逐股票调用。
        若股票在区间内未上榜，返回 has_data=False 的占位行。

        返回 DataFrame 标准字段:
            code, trade_date, has_data, reason, net_buy, seats
        """
        import numpy as np

        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        # 一次性获取区间内所有龙虎榜数据
        lhb_all = pd.DataFrame()
        try:
            lhb_all = ak.stock_lhb_detail_em(start_date=start, end_date=end)
        except Exception as e:
            self._logger.warning(f"获取龙虎榜区间数据失败: {e}")

        # 标准化列名匹配：stock_lhb_detail_em 返回中文列
        # 字段: 序号/代码/名称/上榜日/解读/收盘价/涨跌幅/龙虎榜净买额/...
        code_col = "代码" if "代码" in lhb_all.columns else None
        date_col = "上榜日" if "上榜日" in lhb_all.columns else None
        net_buy_col = "龙虎榜净买额" if "龙虎榜净买额" in lhb_all.columns else None
        reason_col = "解读" if "解读" in lhb_all.columns else None

        all_rows = []
        for code in symbols:
            ticker = self._extract_ticker(code)
            matched = pd.DataFrame()
            if not lhb_all.empty and code_col:
                matched = lhb_all[lhb_all[code_col].astype(str).str.contains(ticker, na=False)]

            if not matched.empty:
                # 取最近一次上榜记录
                row = matched.iloc[0]
                trade_date_val = str(row.get(date_col, "")) if date_col else ""
                # 统一日期格式为 YYYYMMDD
                if trade_date_val:
                    trade_date_val = trade_date_val.replace("-", "").replace("/", "")[:8]
                net_buy_val = 0.0
                if net_buy_col and pd.notna(row.get(net_buy_col)):
                    try:
                        net_buy_val = float(row.get(net_buy_col))
                    except (ValueError, TypeError):
                        net_buy_val = 0.0
                all_rows.append({
                    "code": code,
                    "trade_date": trade_date_val,
                    "has_data": True,
                    "reason": str(row.get(reason_col, "")) if reason_col else "",
                    "net_buy": net_buy_val,
                    "seats": "",
                })
            else:
                all_rows.append({
                    "code": code,
                    "trade_date": "",
                    "has_data": False,
                    "reason": "",
                    "net_buy": 0.0,
                    "seats": "",
                })

        return pd.DataFrame(all_rows)

    def get_shareholder(
        self,
        symbols: List[str],
        report_date: str = "",
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取十大流通股东数据

        改进：使用 stock_gdfx_free_top_10_em（按 symbol+date 查询，新版接口）。
        旧接口 stock_gdfx_free_holding_detail_em 只接受 date 参数（全市场批量），
        传 symbol 会报签名错误。

        参数:
            symbols: 股票代码列表 ['002594.SZ']
            report_date: 报告期 YYYYMMDD 或 YYYY-MM-DD（如 20240930）
                         为空时取最近一期

        返回 DataFrame 标准字段:
            code, holder_name, hold_amount, hold_ratio, change_type, holder_type
        """
        import numpy as np

        # 标准化报告期格式: '2024-09-30' -> '20240930'
        period = report_date.replace("-", "") if report_date else ""

        all_rows = []
        for code in symbols:
            ticker = self._extract_ticker(code)
            # stock_gdfx_free_top_10_em 需要 sh/sz 前缀格式
            if code.startswith("6"):
                em_symbol = f"sh{ticker}"
            elif code.startswith(("0", "3")):
                em_symbol = f"sz{ticker}"
            elif code.startswith(("8", "4")):
                em_symbol = f"bj{ticker}"
            else:
                em_symbol = ticker

            try:
                # 若未提供 report_date，先尝试最近几个季度
                candidate_dates = []
                if period:
                    candidate_dates = [period]
                else:
                    # 回退查找最近 4 个季度
                    from datetime import datetime
                    today = datetime.now()
                    quarters = []
                    y, m = today.year, today.month
                    for _ in range(8):
                        qm = ((m - 1) // 3) * 3  # 3/6/9/12
                        if qm == 0:
                            qm = 12
                            y -= 1
                        quarters.append(f"{y}{qm:02d}30" if qm != 12 else f"{y}1231")
                        m = qm - 1
                        if m <= 0:
                            m = 12
                            y -= 1
                    candidate_dates = quarters

                df = None
                for d in candidate_dates:
                    try:
                        df = ak.stock_gdfx_free_top_10_em(symbol=em_symbol, date=d)
                        if df is not None and not df.empty:
                            break
                    except Exception:
                        continue

                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "股东名称": "holder_name",
                        "持股数": "hold_amount",
                        "占总流通股本持股比例": "hold_ratio",
                        "增减": "change_type",
                        "变动比率": "change_ratio",
                    })
                    df["code"] = code
                    df["holder_type"] = "流通股东"
                    all_rows.append(df)
            except Exception as e:
                self._logger.warning(f"获取 {code} 股东数据失败: {e}")

        if not all_rows:
            return pd.DataFrame()

        result = pd.concat(all_rows, ignore_index=True)
        # 确保标准列存在
        for col in ["code", "holder_name", "hold_amount", "hold_ratio",
                     "change_type", "holder_type"]:
            if col not in result.columns:
                result[col] = None

        return result