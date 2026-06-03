"""
数据引擎专属配置
大部分全局配置从 master 继承，此处仅保留数据引擎特有设置
"""
import os
from typing import List, Optional

# ── 数据源选择 ────────────────────────────
DATA_BACKEND = os.environ.get("DATA_BACKEND", "tushare")
# 可选: tushare, baostock, akshare, xtquant, gm

# ── 数据源降级链 ──────────────────────────
# 当主数据源（DATA_BACKEND）遇到以下情况时，自动切换到下一个数据源：
#   - 积分/权限不足 (QuotaExceededError)
#   - 访问频率受限 (RateLimitError)
#   - 网络错误 (NetworkError)
# 格式：逗号分隔的数据源名列表。留空则不降级
_default_fallback = "tushare,baostock,akshare" if DATA_BACKEND == "tushare" else \
                    "baostock,akshare" if DATA_BACKEND == "baostock" else \
                    "akshare"
DATA_BACKEND_FALLBACK_CHAIN: List[str] = [
    s.strip() for s in os.environ.get("DATA_BACKEND_FALLBACK_CHAIN", _default_fallback).split(",")
    if s.strip()
]

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
