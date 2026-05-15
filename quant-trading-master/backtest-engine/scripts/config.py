"""
回测引擎专属配置
"""
import os

# ── 回测后端选择 ──────────────────────────
BACKTEST_BACKEND = os.environ.get("BACKTEST_BACKEND", "rqalpha")
# 可选: rqalpha, backtrader, gm

# ── 回测结果存储目录 ──────────────────────
BACKTEST_DIR = os.environ.get("BACKTEST_DIR", "./quant_workspace/backtest_results")

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

# ── Walk-Forward 参数 ──────────────────────
WF_TRAIN_MONTHS = int(os.environ.get("WF_TRAIN_MONTHS", 36))
WF_TEST_MONTHS = int(os.environ.get("WF_TEST_MONTHS", 12))

os.makedirs(BACKTEST_DIR, exist_ok=True)
