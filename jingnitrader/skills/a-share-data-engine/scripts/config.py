"""
数据引擎专属配置
大部分全局配置从 master 继承，此处仅保留数据引擎特有设置
"""
import os
from typing import Optional

# ── 数据源选择 ────────────────────────────
DATA_BACKEND = os.environ.get("DATA_BACKEND", "tushare")
# 可选: tushare, baostock, akshare, xtquant, gm

# ── 数据存储格式 ──────────────────────────
DATA_FORMAT = os.environ.get("DATA_FORMAT", "parquet")  # parquet / csv / sql

# ── 并行下载线程数 ─────────────────────────
MAX_WORKERS = int(os.environ.get("DATA_MAX_WORKERS", 4))

# ── 行情复权方式 ──────────────────────────
ADJUST_MODE = os.environ.get("ADJUST_MODE", "hfq")  # hfq:后复权, qfq:前复权, None:不复权

# ── 缓存目录 ──────────────────────────────
CACHE_DIR = os.environ.get("DATA_CACHE_DIR", "./quant_workspace/data_cache")

# ── 股票池默认文件 ─────────────────────────
STOCK_LIST_FILE = os.environ.get("STOCK_LIST_FILE", "")

# ── 数据质量阈值 ──────────────────────────
MAX_MISSING_RATIO = 0.05  # 单只股票允许的最大缺失率

# ── API 令牌（仅从环境变量读取）────────────
TUSHARE_TOKEN: Optional[str] = os.environ.get("TUSHARE_TOKEN")
GM_TOKEN: Optional[str] = os.environ.get("GM_TOKEN")

# ── 自动创建目录 ──────────────────────────
os.makedirs(CACHE_DIR, exist_ok=True)
