"""
因子引擎专属配置
"""
import os

# ── 指标计算后端 ──────────────────────────
FACTOR_BACKEND = os.environ.get("FACTOR_BACKEND", "talib")
# 可选: talib, pandas_ta

# ── 因子存储目录 ──────────────────────────
FACTOR_DIR = os.environ.get("FACTOR_DIR", "./quant_workspace/factors")

# ── 因子分析参数 ──────────────────────────
IC_TYPE = os.environ.get("IC_TYPE", "spearman")  # spearman / pearson
NEUTRALIZE_INDUSTRY = os.environ.get("NEUTRALIZE_INDUSTRY", "true").lower() == "true"
NEUTRALIZE_MARKET_CAP = os.environ.get("NEUTRALIZE_MARKET_CAP", "true").lower() == "true"

# ── 分层回测分组数 ────────────────────────
QUANTILES = int(os.environ.get("FACTOR_QUANTILES", 5))

# ── 单因子过滤阈值 ────────────────────────
MIN_IC = float(os.environ.get("MIN_IC", 0.02))           # 最小|IC|均值
MIN_IC_IR = float(os.environ.get("MIN_IC_IR", 0.3))      # 最小IC_IR
MAX_CORRELATION = float(os.environ.get("MAX_CORRELATION", 0.7))  # 因子间最大相关性

# ── 行业分类标准 ──────────────────────────
INDUSTRY_TYPE = os.environ.get("INDUSTRY_TYPE", "sw")  # sw:申万, zj:证监会

# ── 自动创建目录 ──────────────────────────
os.makedirs(FACTOR_DIR, exist_ok=True)
