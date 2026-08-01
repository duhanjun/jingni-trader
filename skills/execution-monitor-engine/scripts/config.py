"""
执行引擎专属配置
"""
import os

_WORK_DIR = os.environ.get("QUANT_WORK_DIR", "./workspace")
EXECUTION_DIR = os.environ.get("EXECUTION_DIR", os.path.join(_WORK_DIR, "execution"))
TRADE_MODE = os.environ.get("TRADE_MODE", "paper")
TRADE_BACKEND = os.environ.get("TRADE_BACKEND", "paper")
INIT_CAPITAL = float(os.environ.get("INIT_CAPITAL", 1000000))
MAX_DAILY_LOSS_RATIO = float(os.environ.get("MAX_DAILY_LOSS_RATIO", 0.02))
MAX_SINGLE_ORDER_RATIO = float(os.environ.get("MAX_SINGLE_ORDER_RATIO", 0.10))
MAX_SINGLE_STOCK_WEIGHT = float(os.environ.get("MAX_SINGLE_STOCK_WEIGHT", 0.10))
MAX_ORDER_FREQUENCY = int(os.environ.get("MAX_ORDER_FREQUENCY", 2))
MIN_COMMISSION = float(os.environ.get("MIN_COMMISSION", 5.0))
COMMISSION_RATE = float(os.environ.get("COMMISSION_RATE", 0.00025))
STAMP_TAX_RATE = float(os.environ.get("STAMP_TAX_RATE", 0.001))
SLIPPAGE = float(os.environ.get("SLIPPAGE", 0.001))
AUDIT_LOG_PATH = os.path.join(EXECUTION_DIR, "trade_log.jsonl")
ACCOUNT_STATE_PATH = os.path.join(EXECUTION_DIR, "account_state.json")

# 实盘交易后端配置
XTQUANT_PATH = os.environ.get("XTQUANT_PATH", "")  # miniQMT userdata_mini 路径
XTQUANT_ACCOUNT = os.environ.get("XTQUANT_ACCOUNT", "")  # miniQMT 资金账号
GM_TOKEN = os.environ.get("GM_TOKEN", "")  # 掘金量化 token
GM_ACCOUNT_ID = os.environ.get("GM_ACCOUNT_ID", "")  # 掘金账户ID(终端获取)

os.makedirs(EXECUTION_DIR, exist_ok=True)