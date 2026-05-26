"""
全局配置文件
所有路径、后端选择、风控阈值等集中管理
"""
import os

# ── 工作目录 ──────────────────────────────
WORK_DIR = os.environ.get("QUANT_WORK_DIR", "./quant_workspace")
DATA_DIR = os.path.join(WORK_DIR, "data")
FACTOR_DIR = os.path.join(WORK_DIR, "factors")
MODEL_DIR = os.path.join(WORK_DIR, "models")
BACKTEST_DIR = os.path.join(WORK_DIR, "backtest_results")
PORTFOLIO_DIR = os.path.join(WORK_DIR, "portfolio")
REPORT_DIR = os.path.join(WORK_DIR, "reports")
LOG_DIR = os.path.join(WORK_DIR, "logs")

# ── 多后端选择 ────────────────────────────
# 可选: tushare / baostock / akshare / xtquant / gm
DATA_BACKEND = os.environ.get("DATA_BACKEND", "tushare")

# 可选: rqalpha / backtrader / gm
BACKTEST_BACKEND = os.environ.get("BACKTEST_BACKEND", "rqalpha")

# 可选: xtquant / gm
TRADE_BACKEND = os.environ.get("TRADE_BACKEND", "xtquant")

# ── 因子计算后端 ──────────────────────────
# 可选: talib / pandas_ta
FACTOR_BACKEND = os.environ.get("FACTOR_BACKEND", "talib")

# ── A股市场配置 ────────────────────────────
A_SHARE_COMMISSION_RATE = 0.00025    # 佣金 万2.5
A_SHARE_STAMP_TAX = 0.001           # 印花税 千1（卖出）
A_SHARE_MIN_COMMISSION = 5.0        # 最低佣金 5元
A_SHARE_MIN_LOT = 100               # 最小交易单位
A_SHARE_T_PLUS_1 = True             # T+1 交易

# ── 风控阈值 ──────────────────────────────
MAX_DAILY_LOSS_RATIO = 0.03         # 单日最大亏损 3%
MAX_SINGLE_STOCK_WEIGHT = 0.10      # 单票最大持仓 10%
MAX_INDUSTRY_DEVIATION = 0.05       # 行业偏离基准 ±5%
NEW_STOCK_EXCLUDE_DAYS = 60         # 新股保护期

# ── 数据库配置 ────────────────────────────
DATABASE_URL = os.environ.get("QUANT_DB_URL", f"sqlite:///{WORK_DIR}/quant.db")

# ── API 密钥（只从环境变量读取）────────────
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
GM_TOKEN = os.environ.get("GM_TOKEN", "")
XTP_ACCOUNT = os.environ.get("XTP_ACCOUNT", "")

# ── 自动创建目录 ──────────────────────────
for _dir in [WORK_DIR, DATA_DIR, FACTOR_DIR, MODEL_DIR,
             BACKTEST_DIR, PORTFOLIO_DIR, REPORT_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)
