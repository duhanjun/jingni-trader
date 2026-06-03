"""
数据引擎专属配置
大部分全局配置从 master 继承，此处仅保留数据引擎特有设置
"""
import os
import logging
from typing import List, Optional, Callable


logger = logging.getLogger("data-engine.config")


# ── 系统支持的全部数据源 ────────────────────────────
# 任何时候都可以被启用，只是默认只启用通用免费源
SUPPORTED_BACKENDS: List[str] = [
    # 通用免费源（默认启用）
    "baostock",   # 老虎量化开源项目，无频次限制，复权规范
    "akshare",    # 聚合库，覆盖 A股/港股/美股/基金/期货等
    "websearch",  # 终极回退：通过 WebSearch 工具查询具体数据点
    # 显式 opt-in 源（需用户主动指定）
    "tushare",    # Tushare Pro（需 TUSHARE_TOKEN，积分/限频）
    "xtquant",    # 迅投 QMT/xtp（需本地客户端）
    "gm",         # 掘金量化（需 GM_TOKEN + 付费 SDK）
]

# ── 默认数据源（按优先级排序）────────────────────────
# 当用户未指定 DATA_BACKENDS 时，按此顺序自动降级：
#   1. baostock  - 免费、跨域可用、复权规范
#   2. akshare   - 覆盖更广，能补 baostock 缺少的字段
#   3. websearch - 终极回退：特定数据点（如某只票某天收盘价）
#
# ⚠️ 设计原则：
#   - 默认不启用 tushare/xtquant/gm（这些源需要额外配置/付费）
#   - 用户显式指定时优先使用用户的配置
DEFAULT_DATA_SOURCES: List[str] = [
    s.strip() for s in os.environ.get(
        "DATA_BACKENDS",
        "baostock,akshare,websearch"
    ).split(",")
    if s.strip()
]


# ── 系统支持的付费/特定源（用于友好提示）─────────────
# 首次启动时告知用户：除了默认源，系统还支持这 3 个源
PAID_OR_SPECIAL_BACKENDS: List[str] = ["tushare", "xtquant", "gm"]
PAID_OR_SPECIAL_DESCRIPTIONS = {
    "tushare": "Tushare Pro（需 TUSHARE_TOKEN，https://tushare.pro）",
    "xtquant": "迅投 QMT/xtp（需本地券商客户端）",
    "gm":      "掘金量化（需 GM_TOKEN + 付费 SDK，https://www.myquant.cn）",
}


# ── 兼容性：保留 DATA_BACKEND 单源选择 ───────────────
# 如果用户只指定了 DATA_BACKEND（单数），则使用它作为唯一源，不降级
# 推荐使用 DATA_BACKENDS 复数形式
DATA_BACKEND: Optional[str] = os.environ.get("DATA_BACKEND")
if DATA_BACKEND and not os.environ.get("DATA_BACKENDS"):
    DEFAULT_DATA_SOURCES = [DATA_BACKEND]
    logger.info(f"使用 DATA_BACKEND={DATA_BACKEND}（单源模式，不降级）")


# ── 数据存储格式 ──────────────────────────
DATA_FORMAT = os.environ.get("DATA_FORMAT", "parquet")  # parquet / csv / sql

# ── 并行下载线程数 ─────────────────────────
MAX_WORKERS = int(os.environ.get("DATA_MAX_WORKERS", 4))

# ── 行情复权方式 ──────────────────────────
ADJUST_MODE = os.environ.get("ADJUST_MODE", "hfq")  # hfq:后复权, qfq:前复权, None:不复权

# ── 缓存目录 ──────────────────────────────
CACHE_DIR = os.environ.get("DATA_CACHE_DIR", "./workspace/data_cache")

# ── 股票池默认文件 ─────────────────────────
STOCK_LIST_FILE = os.environ.get("STOCK_LIST_FILE", "")

# ── 数据质量阈值 ──────────────────────────
MAX_MISSING_RATIO = 0.05  # 单只股票允许的最大缺失率

# ── API 令牌（仅从环境变量读取）────────────
TUSHARE_TOKEN: Optional[str] = os.environ.get("TUSHARE_TOKEN")
GM_TOKEN: Optional[str] = os.environ.get("GM_TOKEN")

# ── 自动创建目录 ──────────────────────────
os.makedirs(CACHE_DIR, exist_ok=True)


def notify_supported_backends() -> None:
    """
    首次使用时提示用户：系统支持的数据源全景

    设计意图：让用户知道除了默认的 baostock/akshare/websearch 之外，
    还有 tushare/xtquant/gm 这 3 个源可以显式启用。
    """
    logger.info("=" * 60)
    logger.info("数据引擎已就绪。当前默认数据源（按优先级降级）：")
    for i, s in enumerate(DEFAULT_DATA_SOURCES, 1):
        logger.info(f"  {i}. {s}")
    logger.info("")
    logger.info(f"系统还支持以下 {len(PAID_OR_SPECIAL_BACKENDS)} 个源（需显式启用）：")
    for name in PAID_OR_SPECIAL_BACKENDS:
        logger.info(f"  • {name}: {PAID_OR_SPECIAL_DESCRIPTIONS[name]}")
    logger.info("")
    logger.info("启用方式：export DATA_BACKENDS=tushare,akshare,baostock,websearch")
    logger.info("=" * 60)
