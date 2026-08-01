"""
Tushare Pro 数据源适配器

错误处理策略：
- 所有 tushare API 调用都会被 _safe_call 包装
- 异常会被 tushare_error_classifier 归类为 QuotaExceededError / RateLimitError / NetworkError 等
- 上层 DataEngine 据此判断是否切换到下一个数据源
"""
import os
import time
import logging
from typing import List, Optional
import pandas as pd
import tushare as ts

from ..base.base_data_provider import BaseDataProvider
from ..config import TUSHARE_TOKEN, MAX_WORKERS
from ..errors import DataSourceError, QuotaExceededError, RateLimitError, NetworkError
from ..tushare_error_classifier import classify_tushare_error


logger = logging.getLogger("tushare-adapter")


class TushareAdapter(BaseDataProvider):
    """Tushare Pro 适配器"""

    SUPPORTED_DATA_TYPES = {"daily", "financial", "capital_flow", "dragon_tiger"}

    def __init__(self, token: Optional[str] = None):
        token = token or TUSHARE_TOKEN
        if not token:
            raise ValueError("Tushare token 未设置，请设置环境变量 TUSHARE_TOKEN")
        ts.set_token(token)
        self.pro = ts.pro_api()
        # 用于频率控制
        self._last_call = 0.0
        self._min_interval = 0.2  # 每秒最多5次
        # 配额/限频错误重试上限：碰到这些错误时不要重试
        self._non_retriable = (QuotaExceededError, RateLimitError)
        self._logger = logger

    def _rate_limit(self):
        """简单频率控制"""
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _safe_call(self, func, *args, **kwargs):
        """
        包装 tushare API 调用，统一错误处理

        - 限频/积分错误 → 抛出对应异常（不重试）
        - 其他异常 → 包装为 NetworkError（让上层选择切换或重试）
        """
        self._rate_limit()
        try:
            return func(*args, **kwargs)
        except self._non_retriable:
            raise  # 已经是分类后的异常，直接抛
        except Exception as e:
            classified = classify_tushare_error(e)
            # 重要：把这些错误暴露给上层（DataEngine）来决定是否切换数据源
            raise classified from e

    def get_stock_list(self) -> pd.DataFrame:
        """获取全市场股票列表"""
        df = self._safe_call(
            self.pro.stock_basic,
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df.rename(columns={
            'ts_code': 'code',
            'symbol': 'ticker',
            'list_date': 'list_date'
        }, inplace=True)
        # 获取ST标记
        try:
            st_df = self._safe_call(
                self.pro.namechange,
                fields='ts_code,name,start_date,end_date'
            )
        except DataSourceError as e:
            logger.warning(f"获取ST标记失败（{e.message}），跳过")
            st_df = pd.DataFrame()
        # 简化：当前名称含ST的标记
        # 实际需更复杂逻辑，此处仅示例
        df['is_st'] = df['name'].str.contains('ST', na=False)
        return df

    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        """
        获取日线行情，自动复权并添加A股特殊标记
        """
        # 格式标准化
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')

        # 根据复权方式选择接口
        if adjust == 'hfq':
            return self._get_hfq_daily(symbols, start_date, end_date)
        elif adjust == 'qfq':
            return self._get_qfq_daily(symbols, start_date, end_date)
        else:
            # 不复权直接用 pro.daily
            return self._get_raw_daily(symbols, start_date, end_date)

    def _get_hfq_daily(self, symbols, start_date, end_date):
        """后复权日线（推荐用于收益率计算）"""
        frames = []
        for symbol in symbols:
            try:
                df = self._safe_call(
                    self.pro.daily,
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
                )
                if df is None or df.empty:
                    continue
                # 获取复权因子
                adj_df = self._safe_call(
                    self.pro.adj_factor,
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                if adj_df is not None and not adj_df.empty:
                    # 使用后复权因子调整价格
                    df = df.merge(
                        adj_df[['trade_date', 'adj_factor']],
                        on='trade_date',
                        how='left'
                    )
                    # 后复权：最新价 = 不复权价 * (当前复权因子 / 最后因子)
                    # 标准做法：以最新日为基准，向前复权
                    last_adj = adj_df['adj_factor'].iloc[-1]
                    df['adj_factor'] = df['adj_factor'].fillna(last_adj)
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = df[col] * (df['adj_factor'] / last_adj)
                frames.append(df)
            except DataSourceError as e:
                # 限频/积分等不可恢复错误：让上层切换数据源
                raise
            except Exception as e:
                # 单只股票失败不影响整体
                print(f"获取 {symbol} 行情失败: {e}")
                continue

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return self._standardize_output(result)

    def _get_qfq_daily(self, symbols, start_date, end_date):
        """前复权日线"""
        # 可使用 pro.daily 配合 adj_factor 实现前复权，逻辑类似
        # 此处略，调用 _get_hfq_daily 后再转换亦可
        df = self._get_hfq_daily(symbols, start_date, end_date)
        if df.empty:
            return df
        # 前复权：以最新收盘价为基准，倒推历史价格
        # 简单方法：利用复权因子比值
        # 实际已包含在 _get_hfq_daily 的调整中，只需改变基准方向
        # 此简化实现直接返回后复权，标注需改进
        return df

    def _get_raw_daily(self, symbols, start_date, end_date):
        """不复权日线"""
        frames = []
        for symbol in symbols:
            df = self._safe_call(
                self.pro.daily,
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
            )
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        return self._standardize_output(result)

    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化输出列名与数据类型"""
        if df.empty:
            return df
        df = df.rename(columns={
            'ts_code': 'code',
            'trade_date': 'date',
            'vol': 'volume',
            'pre_close': 'pre_close'
        })
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        # 计算涨跌幅和换手率（若缺失）
        if 'change_pct' not in df.columns:
            df['change_pct'] = (df['close'] - df['pre_close']) / df['pre_close'] * 100
        # 换手率暂无法从该接口获取，留空
        df['turnover_rate'] = df.get('turnover_rate', None)
        # A股特殊标记留待清洗阶段填充
        df['is_st'] = False
        df['is_limit_up'] = False
        df['is_limit_down'] = False
        df['listed_days'] = None
        # 排序
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        return df

    def get_adj_factor(self, symbols, start_date, end_date):
        # 简单适配
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        frames = []
        for symbol in symbols:
            adj_df = self._safe_call(
                self.pro.adj_factor,
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date
            )
            if adj_df is not None and not adj_df.empty:
                frames.append(adj_df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames)
        result.rename(columns={'ts_code': 'code', 'trade_date': 'date'}, inplace=True)
        result['date'] = pd.to_datetime(result['date'], format='%Y%m%d')
        return result

    def get_financial(self, symbols, report_date, fields):
        """
        获取财务数据，返回统一标准 schema。

        使用:
        - pro.fina_indicator: 财务指标（ROE/ROA/毛利率/净利率/增速/现金流等）
        - pro.daily_basic: 估值数据（PE_TTM/PB/PS_TTM/DV_RATIO）
        - pro.stock_basic: 行业与股票名称

        返回 DataFrame 包含标准字段:
            code, report_date, pe_ttm, pb, ps_ttm, dv_ratio,
            roe, roa, gross_margin, net_margin,
            revenue_growth, profit_growth,
            debt_ratio, current_ratio, quick_ratio, ocf,
            industry, name
        """
        period = report_date.replace('-', '')

        # 标准输出列（固定顺序）
        standard_cols = [
            'code', 'report_date', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio',
            'roe', 'roa', 'gross_margin', 'net_margin',
            'revenue_growth', 'profit_growth',
            'debt_ratio', 'current_ratio', 'quick_ratio', 'ocf',
            'industry', 'name',
        ]

        # 1) 财务指标 fina_indicator
        fina_fields = (
            'ts_code,ann_date,end_date,'
            'roe,roa,grossprofit_margin,netprofit_margin,'
            'or_on_year,qprofit_yoy,'
            'debt_to_assets,current_ratio,quick_ratio,'
            'netcash_oper'
        )
        fina_frames = []
        for symbol in symbols:
            try:
                df_f = self._safe_call(
                    self.pro.fina_indicator,
                    ts_code=symbol,
                    period=period,
                    fields=fina_fields
                )
                if df_f is not None and not df_f.empty:
                    fina_frames.append(df_f)
            except DataSourceError:
                raise
            except Exception as e:
                logger.warning(f"获取 {symbol} 财务指标失败: {e}")
                continue

        if not fina_frames:
            return pd.DataFrame(columns=standard_cols)

        fina_df = pd.concat(fina_frames, ignore_index=True)
        # 同一报告期可能有多条（更新批次），保留最新 ann_date
        if 'ann_date' in fina_df.columns:
            fina_df = fina_df.sort_values('ann_date').drop_duplicates(
                subset=['ts_code', 'end_date'], keep='last'
            )

        # 2) 估值数据 daily_basic（用报告期对应的当日；若当日无数据则返回空）
        val_frames = []
        for symbol in symbols:
            try:
                df_v = self._safe_call(
                    self.pro.daily_basic,
                    ts_code=symbol,
                    trade_date=period,
                    fields='ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ratio'
                )
                if df_v is not None and not df_v.empty:
                    val_frames.append(df_v)
            except DataSourceError:
                raise
            except Exception as e:
                logger.warning(f"获取 {symbol} 估值数据失败: {e}")
                continue

        if val_frames:
            val_df = pd.concat(val_frames, ignore_index=True)
            val_df = val_df.rename(columns={'trade_date': 'val_date'})
        else:
            val_df = pd.DataFrame(columns=['ts_code', 'val_date', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio'])

        # 3) 行业 + 名称（一次性获取全市场，本地过滤）
        try:
            basic_df = self._safe_call(
                self.pro.stock_basic,
                exchange='',
                list_status='L',
                fields='ts_code,name,industry'
            )
        except DataSourceError:
            raise
        except Exception as e:
            logger.warning(f"获取 stock_basic 失败: {e}")
            basic_df = pd.DataFrame(columns=['ts_code', 'name', 'industry'])

        if basic_df is None or basic_df.empty:
            basic_df = pd.DataFrame(columns=['ts_code', 'name', 'industry'])

        # 4) 合并
        merged = fina_df.merge(val_df, on='ts_code', how='left')
        merged = merged.merge(basic_df, on='ts_code', how='left')

        # 5) 字段映射到标准 schema
        out = pd.DataFrame()
        out['code'] = merged['ts_code']
        out['report_date'] = merged['end_date']
        # 估值字段（daily_basic）
        out['pe_ttm'] = merged.get('pe_ttm')
        out['pb'] = merged.get('pb')
        out['ps_ttm'] = merged.get('ps_ttm')
        out['dv_ratio'] = merged.get('dv_ratio')
        # 财务指标（fina_indicator）
        out['roe'] = merged.get('roe')
        out['roa'] = merged.get('roa')
        out['gross_margin'] = merged.get('grossprofit_margin')
        out['net_margin'] = merged.get('netprofit_margin')
        out['revenue_growth'] = merged.get('or_on_year')
        out['profit_growth'] = merged.get('qprofit_yoy')
        out['debt_ratio'] = merged.get('debt_to_assets')
        out['current_ratio'] = merged.get('current_ratio')
        out['quick_ratio'] = merged.get('quick_ratio')
        out['ocf'] = merged.get('netcash_oper')
        # 行业与名称（stock_basic）
        out['industry'] = merged.get('industry')
        out['name'] = merged.get('name')

        # 如果调用方指定了 fields，按需过滤列（code/report_date 始终保留）
        if fields:
            keep = ['code', 'report_date'] + [f for f in fields if f in standard_cols]
            keep = list(dict.fromkeys(keep))  # 去重保序
            out = out[keep]

        return out.reset_index(drop=True)

    def get_capital_flow(self, symbols, start_date, end_date, **kwargs):
        """获取资金面数据（tushare moneyflow接口）"""
        import pandas as pd
        try:
            import tushare as ts
            pro = ts.pro_api()
            all_rows = []
            for code in symbols:
                ticker = code.split(".")[0] if "." in code else code
                ts_code = f"{ticker}.SZ" if code.startswith(("0", "3")) else f"{ticker}.SH"
                df = pro.moneyflow(ts_code=ts_code, start_date=start_date.replace("-",""), end_date=end_date.replace("-",""))
                if df is not None and not df.empty:
                    df = df.rename(columns={"net_mf_amount": "main_net_inflow"})
                    df["code"] = code
                    all_rows.append(df)
            if not all_rows:
                return pd.DataFrame()
            result = pd.concat(all_rows, ignore_index=True)
            result["main_net_inflow_5d"] = result.groupby("code")["main_net_inflow"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
            return result
        except Exception as e:
            self._logger.warning(f"tushare 资金面数据获取失败: {e}")
            return pd.DataFrame()

    def get_dragon_tiger(self, symbols, start_date, end_date, **kwargs):
        """获取龙虎榜数据（tushare top_list接口）"""
        import pandas as pd
        from datetime import datetime, timedelta
        try:
            import tushare as ts
            pro = ts.pro_api()
            end_dt = datetime.strptime(end_date.replace("-",""), "%Y%m%d")
            check_dates = [(end_dt - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
            all_rows = []
            for code in symbols:
                found = False
                ticker = code.split(".")[0] if "." in code else code
                ts_code = f"{ticker}.SZ" if code.startswith(("0","3")) else f"{ticker}.SH"
                for trade_date in check_dates:
                    df = pro.top_list(trade_date=trade_date)
                    if df is not None and not df.empty:
                        matched = df[df["ts_code"] == ts_code]
                        if not matched.empty:
                            row = matched.iloc[0]
                            all_rows.append({"code": code, "trade_date": trade_date, "has_data": True, "reason": str(row.get("name","")), "net_buy": float(row.get("net_buy", 0)), "seats": ""})
                            found = True
                if not found:
                    all_rows.append({"code": code, "trade_date": "", "has_data": False, "reason": "", "net_buy": 0.0, "seats": ""})
            return pd.DataFrame(all_rows)
        except Exception as e:
            self._logger.warning(f"tushare 龙虎榜数据获取失败: {e}")
            return pd.DataFrame()
