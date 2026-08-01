"""
数据引擎专属配置
大部分全局配置从 master 继承，此处仅保留数据引擎特有设置
"""
import os
import logging
from typing import List, Optional, Callable, Dict


logger = logging.getLogger("data-engine.config")


# ── 系统支持的全部数据源 ────────────────────────────
SUPPORTED_BACKENDS: List[str] = [
    # 真正免费源（默认按以下顺序启用，无需任何 token/账号）
    "baostock",   # 主源：老虎量化开源项目（无需 Token）
    "akshare",    # 次源：聚合库（爬虫，无需 Token）
    "websearch",  # 终极回退：通过 WebSearch 工具查询
    # 显式 opt-in 源（需用户主动指定：配 token/账号或安装本地终端 SDK）
    "tushare",    # Tushare Pro 商业 API（需 TUSHARE_TOKEN）
    "xtquant",    # 迅投 QMT/xtp（需本地券商客户端）
    "gm",         # 掘金量化（需 GM_TOKEN + 付费 SDK）
    "tdxquant",   # 通达信量化（需本地通达信金融终端 TQ 策略）
    "wind",       # 万得 WindPy（需 Wind 金融终端 + WindPy）
    "ifind",      # 同花顺 iFinD（需 iFinDPy + 账号密码）
]


# ── 默认数据源（按优先级排序）────────────────────────
# 仅包含真正免费、无需任何配置的源。
# opt-in 源（tushare/wind/ifind 等）需用户通过对话或 DATA_BACKENDS 环境变量显式启用。
DEFAULT_DATA_SOURCES: List[str] = [
    s.strip() for s in os.environ.get(
        "DATA_BACKENDS",
        "baostock,akshare,websearch"
    ).split(",")
    if s.strip()
]


# ── 数据源降级条件表（v3 用户显式指定版）─────────────
# 每个数据源失败时，只有当触发特定条件才降级到下一个源。
# 这是 v3 的核心改进：从"通用错误降级"升级为"按失败原因精准降级"
DATA_FALLBACK_RULES: Dict[str, Dict[str, str]] = {
    "tushare": {
        "trigger_errors":  "QuotaExceededError, RateLimitError",
        "trigger_messages": "积分不足 / 权限不足 / 访问频率超限（1次/小时、1次/分钟、5次/天）",
        "downgrade_to":    "baostock",
        "downgrade_reason": "Tushare 积分/权限/限频受限",
    },
    "baostock": {
        "trigger_errors":  "BlacklistedError, DataNotFoundError, NetworkError",
        "trigger_messages": "服务器黑名单 / 标的未覆盖 / 网络错误",
        "downgrade_to":    "akshare",
        "downgrade_reason": "Baostock 被服务器限制或未覆盖该标的",
    },
    "akshare": {
        "trigger_errors":  "NetworkError, BlacklistedError, DataNotFoundError",
        "trigger_messages": "爬虫被服务器限制 / HTTPSConnectionPool 失败 / 标的未覆盖",
        "downgrade_to":    "websearch",
        "downgrade_reason": "AkShare 爬虫被限制或未覆盖该标的",
    },
    "websearch": {
        "trigger_errors":  "DataNotFoundError",
        "trigger_messages": "搜索引擎无相关数据 / 搜索结果无法解析为 OHLCV",
        "downgrade_to":    "synthetic（沙箱内最后一道回退：用模拟数据跑完流程并如实告知用户）",
        "downgrade_reason": "所有外部源都不可用",
    },
}


# ── 系统支持的 opt-in 源（用于友好提示）─────────────
# opt-in 源需要前置条件（token/账号/本地终端 SDK），默认不进入降级链。
# 用户可通过对话（如"用 tushare 取数据"）或 DATA_BACKENDS 环境变量显式启用。
PAID_OR_SPECIAL_BACKENDS: List[str] = ["tushare", "xtquant", "gm", "tdxquant", "wind", "ifind"]
PAID_OR_SPECIAL_DESCRIPTIONS = {
    "tushare":   "Tushare Pro 商业 API（需 TUSHARE_TOKEN，https://tushare.pro/）",
    "xtquant":   "迅投 QMT/xtp（需本地券商客户端）",
    "gm":        "掘金量化（需 GM_TOKEN + 付费 SDK，https://www.myquant.cn）",
    "tdxquant":  "通达信量化（需本地通达信金融终端 TQ 策略，https://help.tdx.com.cn/quant/）",
    "wind":      "万得 WindPy（需 Wind 金融终端 + WindPy，https://www.wind.com.cn/）",
    "ifind":     "同花顺 iFinD（需 iFinDPy + 账号密码，http://ft.10jqka.com.cn/）",
}


# ── 兼容性：保留 DATA_BACKEND 单源选择 ───────────────
DATA_BACKEND: Optional[str] = os.environ.get("DATA_BACKEND")
if DATA_BACKEND and not os.environ.get("DATA_BACKENDS"):
    DEFAULT_DATA_SOURCES = [DATA_BACKEND]
    logger.info(f"使用 DATA_BACKEND={DATA_BACKEND}（单源模式，不降级）")


# ── 数据存储格式 ──────────────────────────
DATA_FORMAT = os.environ.get("DATA_FORMAT", "parquet")

# ── 并行下载线程数 ─────────────────────────
MAX_WORKERS = int(os.environ.get("DATA_MAX_WORKERS", 4))

# ── 行情复权方式 ──────────────────────────
ADJUST_MODE = os.environ.get("ADJUST_MODE", "hfq")

# ── 缓存目录 ──────────────────────────────
_WORK_DIR = os.environ.get("QUANT_WORK_DIR", "./workspace")
CACHE_DIR = os.environ.get("DATA_CACHE_DIR", os.path.join(_WORK_DIR, "data_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

# ── 股票池默认文件 ─────────────────────────
STOCK_LIST_FILE = os.environ.get("STOCK_LIST_FILE", "")

# ── 数据质量阈值 ──────────────────────────
MAX_MISSING_RATIO = 0.05

# ── API 令牌（仅从环境变量读取）────────────
TUSHARE_TOKEN: Optional[str] = os.environ.get("TUSHARE_TOKEN")
GM_TOKEN: Optional[str] = os.environ.get("GM_TOKEN")
IFIND_USERNAME: Optional[str] = os.environ.get("IFIND_USERNAME")
IFIND_PASSWORD: Optional[str] = os.environ.get("IFIND_PASSWORD")

# ── 是否允许模拟数据 fallback ──────────────────
# 当所有外部源都失败时，引擎是否生成合成数据继续跑流程
# True:  用沙箱合成数据跑完流程，并在日志/报告中明确告知用户
# False: 抛出异常，让用户决定
ALLOW_SYNTHETIC_FALLBACK: bool = os.environ.get("ALLOW_SYNTHETIC_FALLBACK", "true").lower() == "true"


# ── 数据源依赖自动安装 ─────────────────────────
# 当某数据源适配器所需的第三方库未安装时，DataEngine 是否先尝试
# 自动 pip install 再使用，而不是直接跳过该数据源（默认开启）。
# 设为 false 可关闭（保留旧行为：缺依赖即跳过）。
AUTO_INSTALL_BACKENDS: bool = os.environ.get("AUTO_INSTALL_BACKENDS", "true").lower() == "true"

# 后端名 -> 需要的 pip 包名（用于自动安装；空列表表示无第三方依赖）
# 注意：wind/ifind 的 SDK 不在 PyPI 公开发布，BACKEND_PIP_PACKAGES 中
#       留空表示自动安装无法处理（SDK 由对应终端安装目录提供），
#       缺失时直接判定为不可用，由降级链切换到下一源。
BACKEND_PIP_PACKAGES: Dict[str, List[str]] = {
    "tushare":  ["tushare"],
    "baostock": ["baostock"],
    "akshare":  ["akshare"],
    "websearch": [],
    "xtquant":  ["xtquant"],
    "gm":       ["gm"],
    "tdxquant": ["tdxquant", "pytdx"],
    "wind":     [],   # WindPy 由 Wind 终端安装，非 PyPI
    "ifind":    [],   # iFinDPy 由同花顺终端安装，非 PyPI
}


def notify_supported_backends() -> None:
    """
    首次使用时提示用户：系统支持的数据源全景 + 降级条件

    设计意图：让用户知道每个源的优先级、降级条件、opt-in 源。
    """
    logger.info("=" * 60)
    logger.info("数据引擎已就绪。默认数据源（按优先级降级）：")
    for i, s in enumerate(DEFAULT_DATA_SOURCES, 1):
        rule = DATA_FALLBACK_RULES.get(s, {})
        trigger = rule.get("trigger_messages", "")
        logger.info(f"  {i}. {s}")
        if trigger:
            logger.info(f"     降级条件: {trigger}")
            logger.info(f"     降级到:   {rule.get('downgrade_to', '?')}")
    logger.info("")
    logger.info(f"系统还支持以下 {len(PAID_OR_SPECIAL_BACKENDS)} 个 opt-in 源（需显式启用）：")
    for name in PAID_OR_SPECIAL_BACKENDS:
        logger.info(f"  • {name}: {PAID_OR_SPECIAL_DESCRIPTIONS[name]}")
    logger.info("")
    logger.info(f"合成数据 fallback: {'[OK] 启用' if ALLOW_SYNTHETIC_FALLBACK else '[X] 禁用（失败会抛异常）'}")
    logger.info("")
    logger.info("启用方式：export DATA_BACKENDS=tushare,xtquant,gm,tdxquant,baostock,akshare,websearch")
    logger.info("=" * 60)
