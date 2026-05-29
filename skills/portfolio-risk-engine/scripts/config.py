"""
组合优化专属配置
"""
import os

PORTFOLIO_DIR = os.environ.get("PORTFOLIO_DIR", "./workspace/portfolio")
OPTIMIZATION_METHOD = os.environ.get("OPTIMIZATION_METHOD", "max_sharpe")
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", 0.03))
ESTIMATION_PERIOD = int(os.environ.get("ESTIMATION_PERIOD", 252))
MAX_SINGLE_STOCK_WEIGHT = float(os.environ.get("MAX_SINGLE_STOCK_WEIGHT", 0.10))
MAX_INDUSTRY_DEVIATION = float(os.environ.get("MAX_INDUSTRY_DEVIATION", 0.05))
MAX_TURNOVER = float(os.environ.get("MAX_TURNOVER", 0.30))
COVARIANCE_METHOD = os.environ.get("COVARIANCE_METHOD", "ledoit_wolf")
EXPECTED_RETURNS_METHOD = os.environ.get("EXPECTED_RETURNS_METHOD", "ema_historical")
MAX_DAILY_LOSS_RATIO = float(os.environ.get("MAX_DAILY_LOSS_RATIO", 0.02))
INDIVIDUAL_STOP_LOSS = float(os.environ.get("INDIVIDUAL_STOP_LOSS", 0.08))
VAR_CONFIDENCE = float(os.environ.get("VAR_CONFIDENCE", 0.95))
CVAR_CONFIDENCE = float(os.environ.get("CVAR_CONFIDENCE", 0.95))
BARRA_FACTORS = os.environ.get("BARRA_FACTORS", "size,value,momentum,volatility,quality,leverage,growth")
MIN_WEIGHT = float(os.environ.get("MIN_WEIGHT", 0.0))
BACKEND = os.environ.get("PORTFOLIO_BACKEND", "pypfopt")

os.makedirs(PORTFOLIO_DIR, exist_ok=True)