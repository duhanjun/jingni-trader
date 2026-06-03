"""
A股数据引擎主逻辑
负责调度适配器、数据清洗、本地存储
优先使用 Agent 系统内置工具提供的外部数据
"""
import os
import sys
import logging
from typing import List, Optional, Dict, Any

# 注意：不要在这里 sys.path.insert，会破坏 from scripts.xxx 的包导入
# 由调用方负责正确设置 sys.path

import pandas as pd
import numpy as np

from scripts.config import (
    DATA_BACKEND, DATA_FORMAT, ADJUST_MODE, CACHE_DIR, MAX_MISSING_RATIO,
    DATA_BACKEND_FALLBACK_CHAIN,
)
from scripts.base.base_data_provider import BaseDataProvider
from scripts.errors import (
    DataSourceError, QuotaExceededError, RateLimitError, NetworkError,
    InvalidParameterError,
)


_ADAPTER_REGISTRY = {
    "tushare":   ("scripts.adapters.tushare_adapter",   "TushareAdapter"),
    "baostock":  ("scripts.adapters.baostock_adapter",  "BaostockAdapter"),
    "akshare":   ("scripts.adapters.akshare_adapter",   "AkshareAdapter"),
    "xtquant":   ("scripts.adapters.xtquant_adapter",   "XtQuantAdapter"),
    "gm":        ("scripts.adapters.gm_adapter",        "GmAdapter"),
}


def _load_adapter(backend: str) -> BaseDataProvider:
    """动态加载指定数据源的适配器"""
    if backend not in _ADAPTER_REGISTRY:
        raise ValueError(f"不支持的数据源: {backend}")
    module_path, class_name = _ADAPTER_REGISTRY[backend]
    import importlib
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise DataSourceError(backend, f"导入适配器模块失败: {e}") from e
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise DataSourceError(backend, f"适配器 {class_name} 不存在")
    return cls()


logger = logging.getLogger("data-engine")


class DataEngine:
    """A股数据引擎"""

    def __init__(
        self,
        provider: Optional[BaseDataProvider] = None,
        backend: Optional[str] = None,
        fallback_chain: Optional[List[str]] = None,
    ):
        """
        参数:
            provider: 自定义适配器（覆盖默认 DATA_BACKEND）
            backend: 主数据源名（默认取 config.DATA_BACKEND）
            fallback_chain: 降级链（默认取 config.DATA_BACKEND_FALLBACK_CHAIN）
        """
        self.backend = backend or DATA_BACKEND
        self.fallback_chain = fallback_chain if fallback_chain is not None else list(DATA_BACKEND_FALLBACK_CHAIN)
        # 确保降级链以主源开头，且不重复
        if not self.fallback_chain or self.fallback_chain[0] != self.backend:
            self.fallback_chain = [self.backend] + [
                b for b in self.fallback_chain if b != self.backend
            ]
        self.provider = provider or self._init_provider_with_fallback(self.backend)

    # ------------------------------------------------------------------
    # 多源降级（核心）
    # ------------------------------------------------------------------

    def _init_provider_with_fallback(self, primary: str) -> BaseDataProvider:
        """
        按降级链顺序尝试初始化 provider；
        如果某源初始化失败（例如 tushare token 缺失），自动切换到下一个。
        """
        last_err: Optional[Exception] = None
        for backend in self.fallback_chain:
            try:
                logger.info(f"尝试加载数据源适配器: {backend}")
                provider = _load_adapter(backend)
                logger.info(f"数据源 {backend} 加载成功")
                if backend != primary:
                    logger.warning(
                        f"主数据源 {primary} 初始化失败，已降级到 {backend}（{last_err}）"
                    )
                self.backend = backend
                return provider
            except Exception as e:
                last_err = e
                logger.warning(f"数据源 {backend} 初始化失败: {e}，尝试下一个")
                continue
        raise RuntimeError(
            f"所有数据源都初始化失败。降级链: {self.fallback_chain}，最后错误: {last_err}"
        )

    def _try_fetch_with_fallback(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str,
        exclude_st: bool,
        exclude_new: bool,
        min_listed_days: int,
        fill_suspend: bool,
    ) -> pd.DataFrame:
        """
        尝试拉取数据，按降级链切换：
        - 限频 (RateLimitError)
        - 积分 (QuotaExceededError)
        - 网络 (NetworkError)
        → 自动切换到下一个数据源
        - InvalidParameterError → 不切换，直接抛
        """
        # 从当前 backend 开始，按链向下找
        current_idx = self.fallback_chain.index(self.backend) if self.backend in self.fallback_chain else 0
        tried_backends: List[str] = []

        for idx in range(current_idx, len(self.fallback_chain)):
            backend = self.fallback_chain[idx]
            tried_backends.append(backend)
            # 切换 provider
            if idx > current_idx:
                logger.warning(f"降级到数据源: {backend}")
                try:
                    self.provider = _load_adapter(backend)
                    self.backend = backend
                except Exception as e:
                    logger.warning(f"切换到 {backend} 失败: {e}，尝试下一个")
                    continue

            try:
                logger.info(f"使用数据源 {backend} 获取 {len(symbols)} 只股票的日线数据")
                df = self.provider.get_daily(symbols, start_date, end_date, adjust=adjust)
                if idx > 0 and not df.empty:
                    logger.info(
                        f"数据源 {backend} 拉取成功 {len(df)} 行（已降级 {idx} 次，链路: {' → '.join(tried_backends)}）"
                    )
                return df
            except (QuotaExceededError, RateLimitError, NetworkError) as e:
                # 触发降级
                err_type = type(e).__name__
                reason = e.message if hasattr(e, "message") else str(e)
                retry_after = getattr(e, "retry_after", None)
                if isinstance(e, QuotaExceededError):
                    logger.warning(
                        f"数据源 {backend} 触发【积分/权限不足】（{reason}），自动降级"
                    )
                elif isinstance(e, RateLimitError):
                    logger.warning(
                        f"数据源 {backend} 触发【频率限制】（{reason}），自动降级"
                        + (f"，建议等待 {retry_after}s" if retry_after else "")
                    )
                else:
                    logger.warning(f"数据源 {backend} 触发【网络错误】（{reason}），自动降级")
                continue
            except InvalidParameterError as e:
                # token/auth 相关参数错误：仍应降级（下一家可能不需要 token）
                # 其他参数错误（如股票代码格式）：不切换
                msg = e.message.lower() if hasattr(e, "message") else str(e).lower()
                is_token_error = any(
                    kw in msg for kw in (
                        "token", "认证", "authorization", "登录",
                    )
                )
                if is_token_error:
                    logger.warning(
                        f"数据源 {backend} 触发【认证错误】（{e.message}），自动降级"
                    )
                    continue
                logger.error(f"数据源 {backend} 参数错误（{e.message}），不切换")
                raise
            except DataSourceError as e:
                logger.warning(f"数据源 {backend} 错误（{e.message}），尝试下一个")
                continue
            except Exception as e:
                # 未识别的异常，保守切换
                logger.warning(f"数据源 {backend} 未知异常: {e}，尝试下一个")
                continue

        raise RuntimeError(
            f"降级链 {self.fallback_chain} 中所有数据源都失败。"
            f"已尝试: {' → '.join(tried_backends)}"
        )

    def try_external_data(self, external_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        尝试从外部数据源获取数据（系统内置工具提供）

        返回清洗后的 DataFrame 或 None
        """
        daily = external_data.get("daily")
        if daily is None:
            return None

        if not isinstance(daily, pd.DataFrame):
            logger.info("external_data.daily 不是 DataFrame，跳过外部数据")
            return None

        if daily.empty:
            logger.info("external_data.daily 为空，跳过外部数据")
            return None

        source = external_data.get("source", "external")
        logger.info(f"使用系统内置工具提供的数据 (来源: {source})，共 {len(daily)} 行")

        required_cols = {"code", "date", "open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(daily.columns)
        if missing_cols:
            logger.error(f"外部数据缺少必需列: {missing_cols}")

        df = daily.copy()
        existing_cols = set(df.columns)

        if "code" not in existing_cols and "ts_code" in existing_cols:
            df["code"] = df["ts_code"]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "change_pct" not in existing_cols and "close" in existing_cols and "pre_close" in existing_cols:
            df["change_pct"] = (df["close"] - df["pre_close"]) / df["pre_close"] * 100
        if "is_st" not in existing_cols:
            df["is_st"] = False
        if "is_limit_up" not in existing_cols:
            if "change_pct" in existing_cols:
                df["is_limit_up"] = df["change_pct"] >= 9.9
                df["is_limit_down"] = df["change_pct"] <= -9.9
            else:
                df["is_limit_up"] = False
                df["is_limit_down"] = False

        return df

    def fetch_and_clean(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = ADJUST_MODE,
        exclude_st: bool = True,
        exclude_new: bool = True,
        min_listed_days: int = 60,
        fill_suspend: bool = False,
        external_data: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        获取并清洗日线数据

        数据源优先级:
        1. external_data (系统内置工具提供)
        2. Tushare/GM (环境变量配置)
        3. BaoStock/AkShare (免费离线)

        参数:
            symbols: 股票代码列表，为空则获取全部A股
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期
            adjust: 复权方式
            exclude_st: 剔除ST
            exclude_new: 剔除新股
            min_listed_days: 最少上市天数
            fill_suspend: 停牌是否前向填充
            external_data: 外部数据源提供的数据

        返回:
            清洗后的 DataFrame，包含统一列
        """
        if external_data:
            df = self.try_external_data(external_data)
            if df is not None and not df.empty:
                logger.info(f"外部数据清洗后 {len(df)} 行")
                return df.sort_values(['date', 'code']).reset_index(drop=True)
            logger.info("外部数据不可用，回退到内置适配器")

        if not symbols:
            stock_df = self.provider.get_stock_list()
            if stock_df.empty:
                logger.error("无法获取股票列表")
                return pd.DataFrame()
            symbols = stock_df['code'].tolist()

        logger.info(f"开始获取 {len(symbols)} 只股票的日线数据，时间 {start_date} 至 {end_date}")
        # 使用降级链拉取数据
        df = self._try_fetch_with_fallback(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            exclude_st=exclude_st,
            exclude_new=exclude_new,
            min_listed_days=min_listed_days,
            fill_suspend=fill_suspend,
        )
        if df.empty:
            logger.error("未获取到任何数据")
            return df

        logger.info("开始数据清洗...")
        initial_rows = len(df)

        if exclude_new:
            try:
                stock_info = self.provider.get_stock_list()
                if not stock_info.empty and 'list_date' in stock_info.columns:
                    stock_info['list_date'] = pd.to_datetime(stock_info['list_date'], format='%Y%m%d', errors='coerce')
                    df = df.merge(stock_info[['code', 'list_date']], on='code', how='left')
                    df['listed_days'] = (df['date'] - df['list_date']).dt.days
                    df = df[df['listed_days'] >= min_listed_days]
                    logger.info(f"剔除新股后剩余 {len(df)} 行 (剔除 {initial_rows - len(df)} 行)")
            except Exception as e:
                logger.warning(f"获取股票列表失败，跳过新股剔除: {e}")

        if not fill_suspend:
            df = df[df['volume'] > 0]
        else:
            df = df.sort_values(['code', 'date'])
            df = df.set_index(['code', 'date'])
            df = df.groupby('code').apply(
                lambda x: x.ffill()
            ).reset_index(level=0, drop=True)
            df = df.reset_index()

        if 'is_limit_up' not in df.columns or df['is_limit_up'].isna().all():
            df = self._mark_price_limits(df)
        if 'is_st' not in df.columns or df['is_st'].isna().all():
            df = self._mark_st(df)

        if exclude_st:
            st_mask = df['is_st'] == True
            df = df[~st_mask]

        df = df.dropna(subset=['close'])

        logger.info(f"清洗完成，最终 {len(df)} 行数据")
        return df.sort_values(['date', 'code']).reset_index(drop=True)

    def _mark_price_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据涨跌幅标记涨跌停"""
        if 'change_pct' not in df.columns:
            return df
        limit_up = df['change_pct'] >= 9.9
        limit_down = df['change_pct'] <= -9.9
        df['is_limit_up'] = limit_up
        df['is_limit_down'] = limit_down
        return df

    def _mark_st(self, df: pd.DataFrame) -> pd.DataFrame:
        """标记ST股票"""
        if 'is_st' in df.columns and not df['is_st'].isna().all():
            return df
        try:
            stock_list = self.provider.get_stock_list()
            if not stock_list.empty and 'is_st' in stock_list.columns:
                st_codes = stock_list[stock_list['is_st'] == True]['code'].tolist()
                df['is_st'] = df['code'].isin(st_codes)
            else:
                df['is_st'] = False
        except Exception:
            df['is_st'] = False
        return df

    def save_data(self, df: pd.DataFrame, path: str):
        """保存数据到文件"""
        if DATA_FORMAT == 'parquet':
            df.to_parquet(path, index=False)
        elif DATA_FORMAT == 'csv':
            df.to_csv(path, index=False)
        elif DATA_FORMAT == 'sql':
            from sqlalchemy import create_engine
            engine = create_engine(os.environ.get("QUANT_DB_URL", "sqlite:///quant.db"))
            df.to_sql('daily', engine, if_exists='append', index=False)
        else:
            raise ValueError(f"不支持的存储格式: {DATA_FORMAT}")
        logger.info(f"数据已保存至 {path}")


def run(ctx) -> Dict[str, Any]:
    """
    data-engine 的 run 函数
    由 jingnitrader 调度

    参数:
        ctx: Context 对象，需包含:
            - stock_pool: list
            - start_date: str
            - end_date: str
            - external_data: dict (可选，系统内置工具提供的数据)
            - 可选: artifacts 已有产物路径可跳过

    返回:
        {
            "success": bool,
            "artifact_path": str,
            "metadata": dict,
            "error": str
        }
    """
    try:
        existing = ctx.get_artifact("DATA")
        if existing and os.path.exists(existing):
            return {
                "success": True,
                "artifact_path": existing,
                "metadata": {"source": "cache"},
                "error": ""
            }

        external = ctx.external_data if ctx.external_data else None
        if external and external.get("daily") is not None:
            logger.info(f"检测到外部数据源: {external.get('source', 'unknown')}")

        engine = DataEngine()
        df = engine.fetch_and_clean(
            symbols=ctx.stock_pool,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
            adjust=ADJUST_MODE,
            external_data=external
        )
        if df.empty:
            return {
                "success": False,
                "artifact_path": "",
                "metadata": {},
                "error": "未获取到任何有效数据"
            }

        output_dir = os.environ.get("QUANT_DATA_DIR", "./workspace/data")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "cleaned_data.parquet")
        engine.save_data(df, path)

        data_source = "external" if external and external.get("daily") is not None else "native"
        return {
            "success": True,
            "artifact_path": path,
            "metadata": {
                "rows": len(df),
                "symbols_count": df['code'].nunique(),
                "date_range": f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}",
                "data_source": data_source
            },
            "error": ""
        }
    except Exception as e:
        logger.exception("数据引擎执行失败")
        return {
            "success": False,
            "artifact_path": "",
            "metadata": {},
            "error": str(e)
        }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            ctx_dict = json.load(f)
        from scripts.context import Context
        ctx = Context.from_dict(ctx_dict)
    else:
        from scripts.context import Context
        ctx = Context(
            task_id="test",
            stock_pool=["000001.SZ", "600000.SH"],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
    result = run(ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False))