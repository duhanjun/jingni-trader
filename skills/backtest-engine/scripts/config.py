"""
回测引擎专属配置
"""
import os

# ── 回测结果存储目录（优先读 QUANT_WORK_DIR 与主调度器对齐）──────────
_WORK_DIR = os.environ.get("QUANT_WORK_DIR", "./workspace")
BACKTEST_DIR = os.environ.get("BACKTEST_DIR", os.path.join(_WORK_DIR, "backtest_results"))

# ── A股交易费用 ───────────────────────────
COMMISSION_RATE = float(os.environ.get("COMMISSION_RATE", 0.00025))  # 万2.5
MIN_COMMISSION = float(os.environ.get("MIN_COMMISSION", 5.0))
STAMP_TAX_RATE = float(os.environ.get("STAMP_TAX_RATE", 0.001))     # 千1卖出
TRANSFER_FEE_RATE = float(os.environ.get("TRANSFER_FEE_RATE", 0.00002))  # 0.02‰

# ── 回测参数 ──────────────────────────────
INIT_CAPITAL = float(os.environ.get("INIT_CAPITAL", 1000000))
SLIPPAGE = float(os.environ.get("SLIPPAGE", 0.001))     # 滑点 0.1%
BENCHMARK = os.environ.get("BENCHMARK", "000300.SH")

# ── 绩效计算参数 ──────────────────────────
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", 0.03))

os.makedirs(BACKTEST_DIR, exist_ok=True)
