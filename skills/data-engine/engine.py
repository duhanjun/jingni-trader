"""
A股数据引擎主逻辑
负责调度适配器、数据清洗、本地存储
优先使用 Agent 系统内置工具提供的外部数据

数据源架构（v3）：
- 默认优先级: tushare → baostock → akshare → websearch
- 显式 opt-in 源: xtquant / gm / tdxquant（需用户通过 DATA_BACKENDS 启用）
- 每个源都有明确的降级条件（见 config.DATA_FALLBACK_RULES）：
    * tushare 积分/权限/限频受限 → baostock
    * baostock 黑名单/未覆盖 → akshare
    * akshare 爬虫被限制 → websearch
    * websearch 搜索不到 → 模拟数据（告知用户）
- 触发降级的异常类型集中在 errors.FALLBACK_TRIGGERING_ERRORS
"""
import os
import sys
import logging
from typing import List, Optional, Dict, Any, Callable

# 注意：不要在这里 sys.path.insert，会破坏 from scripts.xxx 的包导入
# 由调用方负责正确设置 sys.path

import pandas as pd
import numpy as np

from scripts.config import (
    DATA_FORMAT, ADJUST_MODE, CACHE_DIR, MAX_MISSING_RATIO,
    DEFAULT_DATA_SOURCES, SUPPORTED_BACKENDS, PAID_OR_SPECIAL_BACKENDS,
    DATA_FALLBACK_RULES, ALLOW_SYNTHETIC_FALLBACK,
    notify_supported_backends,
)
from scripts.base.base_data_provider import BaseDataProvider
from scripts.errors import (
    DataSourceError, QuotaExceededError, RateLimitError, NetworkError,
    InvalidParameterError, BlacklistedError, DataNotFoundError,
    FALLBACK_TRIGGERING_ERRORS,
)


# 适配器注册表
# 注意：websearch 适配器需要 web_search_fn 注入，特殊处理
# tdxquant 是新增的 opt-in 源（通达信量化）
_ADAPTER_REGISTRY = {
    "tushare":   ("scripts.adapters.tushare_adapter",   "TushareAdapter",   {}),
    "baostock":  ("scripts.adapters.baostock_adapter",  "BaostockAdapter",  {}),
    "akshare":   ("scripts.adapters.akshare_adapter",   "AkshareAdapter",   {}),
    "websearch": ("scripts.adapters.websearch_adapter", "WebSearchAdapter", {"web_search_fn": None}),
    "xtquant":   ("scripts.adapters.xtquant_adapter",   "XtQuantAdapter",   {}),
    "gm":        ("scripts.adapters.gm_adapter",        "GmAdapter",        {}),
    "tdxquant":  ("scripts.adapters.tdxquant_adapter",  "TdxQuantAdapter",  {}),
}


# 触发降级的异常类型集合（从 errors 导入并复述，方便在引擎内引用）
_TRIGGERING_ERRORS = FALLBACK_TRIGGERING_ERRORS


def _load_adapter(backend: str, **extra_kwargs) -> BaseDataProvider:
    """动态加载指定数据源的适配器"""
    if backend not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"不支持的数据源: {backend}。"
            f"系统支持的数据源: {', '.join(SUPPORTED_BACKENDS)}"
        )
    module_path, class_name, default_kwargs = _ADAPTER_REGISTRY[backend]
    # 合并默认参数和传入参数
    merged_kwargs = {**default_kwargs, **extra_kwargs}
    import importlib
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise DataSourceError(backend, f"导入适配器模块失败: {e}") from e
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise DataSourceError(backend, f"适配器 {class_name} 不存在")
    return cls(**merged_kwargs)


logger = logging.getLogger("data-engine")


# 已通知标记（避免重复打印）
_NOTIFIED = False


class DataEngine:
    """A股数据引擎（v3：精准降级 + 模拟数据兜底）"""

    def __init__(
        self,
        provider: Optional[BaseDataProvider] = None,
        data_sources: Optional[List[str]] = None,
        web_search_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        参数:
            provider: 自定义适配器（覆盖整个降级链）
            data_sources: 数据源优先级链（默认取 config.DEFAULT_DATA_SOURCES）
                         默认: ["tushare", "baostock", "akshare", "websearch"]
            web_search_fn: 注入给 websearch 适配器的搜索函数
                          在 jingni-trader 中由 agent 的 WebSearch 工具提供
        """
        global _NOTIFIED
        self.web_search_fn = web_search_fn
        self.data_sources = data_sources if data_sources is not None else list(DEFAULT_DATA_SOURCES)

        # 校验 data_sources 中所有项都是支持的源
        unknown = [b for b in self.data_sources if b not in SUPPORTED_BACKENDS]
        if unknown:
            raise ValueError(
                f"data_sources 包含未知源 {unknown}。"
                f"系统支持: {', '.join(SUPPORTED_BACKENDS)}"
            )

        # 更新 websearch 默认注入函数
        if "websearch" in self.data_sources and web_search_fn:
            self._update_websearch_kwargs(web_search_fn)

        self.provider = provider or self._init_provider_with_fallback()
        # 用于在 fetch_and_clean 中告知调用方本次是否走了模拟数据
        self.is_synthetic = False

        # 首次初始化时打印支持的数据源全景
        if not _NOTIFIED:
            notify_supported_backends()
            _NOTIFIED = True

    def _update_websearch_kwargs(self, web_search_fn: Callable):
        """更新 websearch 适配器默认注入函数"""
        _ADAPTER_REGISTRY["websearch"] = (
            _ADAPTER_REGISTRY["websearch"][0],
            _ADAPTER_REGISTRY["websearch"][1],
            {"web_search_fn": web_search_fn},
        )

    # ------------------------------------------------------------------
    # 多源降级（核心 v3）
    # ------------------------------------------------------------------

    def _init_provider_with_fallback(self) -> BaseDataProvider:
        """
        按 data_sources 顺序尝试初始化 provider；
        第一个初始化成功的源被选中（不实际拉数据）
        """
        last_err: Optional[Exception] = None
        for backend in self.data_sources:
            try:
                logger.info(f"尝试加载数据源适配器: {backend}")
                provider = _load_adapter(backend)
                logger.info(f"数据源 {backend} 加载成功")
                self.backend = backend
                return provider
            except Exception as e:
                last_err = e
                logger.warning(f"数据源 {backend} 初始化失败: {e}，尝试下一个")
                continue
        raise RuntimeError(
            f"所有数据源都初始化失败。降级链: {self.data_sources}，最后错误: {last_err}"
        )

    def _should_fallback(self, backend: str, exc: Exception) -> bool:
        """
        判断某个异常是否应触发【降级到下一源】

        决策依据：
        1. 异常类型在 FALLBACK_TRIGGERING_ERRORS 中
        2. 进一步核对 config.DATA_FALLBACK_RULES 中该源对应的 trigger_errors
           （防止 tushare 的限频被误判为 baostock 的限频）
        3. 对 InvalidParameterError 做特殊判断：仅 token/认证类错误才降级
        """
        # 1) 类型白名单
        if not isinstance(exc, _TRIGGERING_ERRORS):
            return False

        # 2) 对应源的降级规则
        rule = DATA_FALLBACK_RULES.get(backend, {})
        trigger_errs_str = rule.get("trigger_errors", "")
        trigger_err_names = {
            name.strip() for name in trigger_errs_str.split(",") if name.strip()
        }
        exc_type_name = type(exc).__name__
        if trigger_err_names and exc_type_name not in trigger_err_names:
            # 该异常类型不在此源允许的降级错误列表里
            # 例外：InvalidParameterError 永远允许（因为引擎已判定为 token/auth）
            if exc_type_name != "InvalidParameterError":
                return False

        # 3) InvalidParameterError 仅在 token/认证场景下才降级
        if isinstance(exc, InvalidParameterError):
            msg = exc.message.lower() if hasattr(exc, "message") else str(exc).lower()
            is_token_error = any(
                kw in msg for kw in (
                    "token", "认证", "authorization", "登录", "登录失败",
                    "access key", "api key", "您的token",
                )
            )
            return is_token_error

        return True

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
        尝试拉取数据，按 data_sources 链切换（v3 精准降级）：
        每个源失败时，依据 DATA_FALLBACK_RULES 中该源对应的 trigger_errors
        来决定是否降级。
        走完整个降级链后仍未拿到数据 → 走模拟数据兜底。
        """
        tried_backends: List[str] = []
        last_errors: Dict[str, str] = {}

        for idx, backend in enumerate(self.data_sources):
            tried_backends.append(backend)
            # 切换 provider
            if idx > 0 or self.backend != backend:
                logger.info(f"切换到数据源: {backend}（链路: {' → '.join(tried_backends)}）")
                try:
                    self.provider = _load_adapter(backend)
                    self.backend = backend
                except Exception as e:
                    last_errors[backend] = f"初始化失败: {e}"
                    logger.warning(f"切换到 {backend} 失败: {e}，尝试下一个")
                    continue

            try:
                logger.info(f"使用数据源 {backend} 获取 {len(symbols)} 只股票的日线数据")
                df = self.provider.get_daily(symbols, start_date, end_date, adjust=adjust)
                if not df.empty:
                    if idx > 0:
                        logger.info(
                            f"✓ 数据源 {backend} 拉取成功 {len(df)} 行"
                            f"（链路: {' → '.join(tried_backends)}）"
                        )
                    self.is_synthetic = False
                    return df
                else:
                    last_errors[backend] = "返回空数据"
                    logger.warning(f"数据源 {backend} 返回空数据，尝试下一个源")
                    continue
            except Exception as e:
                # 统一转字符串、记录
                msg = getattr(e, "message", str(e))
                last_errors[backend] = f"{type(e).__name__}: {msg}"

                if self._should_fallback(backend, e):
                    reason = DATA_FALLBACK_RULES.get(backend, {}).get("downgrade_reason", "")
                    logger.warning(
                        f"数据源 {backend} 触发【{reason or type(e).__name__}】（{msg}），自动降级"
                    )
                    continue

                # 不可降级：直接抛
                if isinstance(e, DataSourceError):
                    logger.error(f"数据源 {backend} 错误（{msg}），不切换")
                else:
                    logger.error(f"数据源 {backend} 未知异常（{msg}），不切换")
                raise

        # 走完整个降级链仍失败
        logger.error(
            f"data_sources {self.data_sources} 中所有数据源都失败或返回空。"
            f"已尝试: {' → '.join(tried_backends)}"
        )
        for k, v in last_errors.items():
            logger.error(f"  • {k}: {v}")

        # ---- 模拟数据兜底 ----
        if not ALLOW_SYNTHETIC_FALLBACK:
            raise RuntimeError(
                f"所有数据源都失败，且 ALLOW_SYNTHETIC_FALLBACK=False。"
                f"已尝试: {' → '.join(tried_backends)}，错误: {last_errors}"
            )

        logger.warning("=" * 60)
        logger.warning("⚠️  所有外部数据源均不可用，启用【模拟数据 fallback】")
        logger.warning(
            "  这意味着回测结果仅供流程验证，"
            "不能用于真实交易决策。"
        )
        logger.warning(
            f"  降级链: {' → '.join(tried_backends)}"
        )
        for k, v in last_errors.items():
            logger.warning(f"    • {k}: {v}")
        logger.warning("=" * 60)

        self.is_synthetic = True
        self.backend = "synthetic"
        return self._generate_synthetic_data(symbols, start_date, end_date)

    # ------------------------------------------------------------------
    # 模拟数据生成（v3 兜底）
    # ------------------------------------------------------------------

    def _generate_synthetic_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        生成与真实 A 股日线统计特征近似的模拟数据。

        设计原则：
        - 仅供【流程跑通】和【集成测试】使用，不可用于真实交易
        - 满足 schema：date, code, open, high, low, close, vol, is_st, is_limit_up/down
        - 几何布朗运动 + 微小趋势，让后续回测引擎能正常推进
        - 每个标的独立一个价格序列
        """
        np.random.seed(20240101)
        s_date = pd.to_datetime(start_date)
        e_date = pd.to_datetime(end_date)
        all_dates = pd.bdate_range(start=s_date, end=e_date)
        n = len(all_dates)

        if n < 5:
            return pd.DataFrame(columns=[
                'date', 'code', 'open', 'high', 'low', 'close', 'vol',
                'is_st', 'is_limit_up', 'is_limit_down', 'change_pct',
            ])

        rows = []
        for sym in symbols:
            # 起始价：股票 10~50 元随机，债券 ETF 100~120 元
            is_etf = sym.startswith(("5", "1", "15")) and (sym.endswith(".SH") or sym.endswith(".SZ"))
            if is_etf:
                start_price = np.random.uniform(95.0, 115.0)
            else:
                start_price = np.random.uniform(8.0, 50.0)

            daily_drift = np.random.uniform(-0.0005, 0.0015)
            daily_vol = np.random.uniform(0.008, 0.020)

            returns = np.random.normal(daily_drift, daily_vol, n)
            for i in range(1, n):
                returns[i] += 0.15 * returns[i - 1]

            prices = [start_price]
            for r in returns[1:]:
                prices.append(prices[-1] * (1 + r))
            prices = np.array(prices)

            df_one = pd.DataFrame({
                'date': all_dates,
                'code': sym,
                'close': prices,
            })
            df_one['open'] = (
                df_one['close'].shift(1).fillna(df_one['close'].iloc[0]) *
                (1 + np.random.normal(0, 0.003, len(df_one)))
            )
            intraday_range = np.abs(np.random.normal(0, 0.005, len(df_one)))
            df_one['high'] = np.maximum(df_one['open'], df_one['close']) * (1 + intraday_range)
            df_one['low'] = np.minimum(df_one['open'], df_one['close']) * (1 - intraday_range)
            df_one['vol'] = np.random.lognormal(10, 0.5, len(df_one)).astype(int)

            df_one['pre_close'] = df_one['close'].shift(1).fillna(df_one['close'].iloc[0])
            df_one['change_pct'] = (df_one['close'] - df_one['pre_close']) / df_one['pre_close'] * 100
            df_one['is_st'] = False
            df_one['is_limit_up'] = df_one['change_pct'] >= 9.9
            df_one['is_limit_down'] = df_one['change_pct'] <= -9.9

            for c in ['open', 'high', 'low', 'close']:
                df_one[c] = df_one[c].round(4)

            rows.append(df_one)

        if not rows:
            return pd.DataFrame(columns=[
                'date', 'code', 'open', 'high', 'low', 'close', 'vol',
                'is_st', 'is_limit_up', 'is_limit_down', 'change_pct',
            ])

        df = pd.concat(rows, ignore_index=True)
        return df[['date', 'code', 'open', 'high', 'low', 'close', 'vol',
                   'pre_close', 'change_pct', 'is_st', 'is_limit_up', 'is_limit_down']]

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

        数据源优先级（默认 v3）:
        1. external_data (系统内置工具提供)
        2. tushare (默认主源)
        3. baostock (tushare 失败/缺数据)
        4. akshare (baostock 失败/缺数据)
        5. websearch (akshare 失败/缺数据)
        6. synthetic (所有源都失败时用模拟数据兜底)

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
            try:
                stock_df = self.provider.get_stock_list()
                if stock_df.empty:
                    logger.error("无法获取股票列表")
                    return pd.DataFrame()
                symbols = stock_df['code'].tolist()
            except Exception as e:
                logger.error(f"获取股票列表失败: {e}，无法获取全市场数据")
                return pd.DataFrame()

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

        # 模拟数据时跳过【新股剔除】和【停牌剔除】等需要真实 list_date 的清洗
        # 否则会因为没有 list_date 字段把全部数据剔光
        if not self.is_synthetic and exclude_new:
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

        if not self.is_synthetic and not fill_suspend:
            df = df[df['volume'] > 0] if 'volume' in df.columns else df[df['vol'] > 0]
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

        if exclude_st and not self.is_synthetic:
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
            - 可选: data_sources: List[str] (用户自定义降级链)
            - 可选: web_search_fn: Callable (agent 注入的搜索函数)

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

        # 从 context 获取可选参数
        data_sources = getattr(ctx, 'data_sources', None)
        web_search_fn = None
        if hasattr(ctx, 'external_data') and isinstance(ctx.external_data, dict):
            web_search_fn = ctx.external_data.get('web_search_fn')

        engine = DataEngine(
            data_sources=data_sources,
            web_search_fn=web_search_fn,
        )
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
                "data_source": data_source,
                "active_backend": engine.backend,
                "is_synthetic": engine.is_synthetic,
                "fallback_chain": engine.data_sources,
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


# ---------------------------------------------------------------------------
# 优化模块入口（已整合到 scripts/optimizations/）
# 使用方式: from engine import optimizations
# ---------------------------------------------------------------------------
from scripts.optimizations.pit_adapter import (
    PITDataAdapter as _PITDataAdapter,
    PITField as _PITField,
)


class optimizations:
    """数据引擎优化模块集合"""
    PITDataAdapter = _PITDataAdapter
    PITField = _PITField

    @staticmethod
    def get_pit_modules():
        """返回所有可用的 PIT 模块"""
        return {
            "pit_data_adapter": _PITDataAdapter,
            "pit_field": _PITField,
        }
