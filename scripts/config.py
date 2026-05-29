"""
全局配置文件
所有路径、后端选择、风控阈值等集中管理
"""
import os

# ── 工作目录 ──────────────────────────────
WORK_DIR = os.environ.get("QUANT_WORK_DIR", "./workspace")
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

# ── 回测参数（兼容子 Skill）────────────────
INIT_CAPITAL = 1000000.0            # 初始资金 100万
COMMISSION_RATE = 0.00025            # 交易佣金（万分之2.5）
MIN_COMMISSION = 5.0                # 最低佣金
STAMP_TAX_RATE = 0.001              # 印花税（千分之1）
TRANSFER_FEE_RATE = 0.00002          # 过户费（万分之0.2）
SLIPPAGE = 0.0001                    # 滑点（万分之1）
BENCHMARK = "000300.SH"             # 基准指数（沪深300）
RISK_FREE_RATE = 0.03               # 无风险利率（3%）

# ── 风控阈值 ──────────────────────────────
MAX_DAILY_LOSS_RATIO = 0.03         # 单日最大亏损 3%
MAX_SINGLE_STOCK_WEIGHT = 0.10      # 单票最大持仓 10%
MAX_INDUSTRY_DEVIATION = 0.05       # 行业偏离基准 ±5%
NEW_STOCK_EXCLUDE_DAYS = 60         # 新股保护期

# ── 数据采集参数 ──────────────────────────
DATA_FORMAT = "parquet"             # 数据格式：parquet / csv / hdf5
ADJUST_MODE = "hfq"                 # 复权模式：hfq（前复权）/ qfq（后复权）/ None
CACHE_DIR = os.path.join(WORK_DIR, "cache")
MAX_MISSING_RATIO = 0.3             # 最大缺失比例

# ── 因子参数 ──────────────────────────────
IC_TYPE = "normal"                  # IC类型：normal / rank
NEUTRALIZE_INDUSTRY = True         # 是否行业中性化
NEUTRALIZE_MARKET_CAP = True       # 是否市值中性化
QUANTILES = 5                      # 分位数数量
MIN_IC = 0.02                      # 最小IC阈值
MIN_IC_IR = 0.5                    # 最小IC_IR阈值
MAX_CORRELATION = 0.7              # 最大因子相关性

# ── 模型训练参数 ──────────────────────────
MODEL_TYPE = "lightgbm"             # 模型类型：lightgbm / xgboost / catboost
OPTUNA_TRIALS = 50                  # Optuna超参搜索次数
OPTUNA_TIMEOUT = 600               # Optuna超时时间（秒）
TRAIN_WINDOW_MONTHS = 36           # 训练窗口（月）
VALIDATION_WINDOW_MONTHS = 12      # 验证窗口（月）
TEST_WINDOW_MONTHS = 12            # 测试窗口（月）
PURGE_GAP_DAYS = 5                 # 清洗期（天）
FORWARD_PERIOD = 1                 # 前视期（天）
LABEL_TYPE = "return"              # 标签类型：return / classification
MLFLOW_TRACKING_URI = ""           # MLflow跟踪URI（可选）
MLFLOW_EXPERIMENT_NAME = "jingnitrader"  # MLflow实验名

# ── 组合优化参数 ──────────────────────────
OPTIMIZATION_METHOD = "hierarchical_risk_parity"  # 优化方法
ESTIMATION_PERIOD = 252            # 协方差估计周期（交易日）
MAX_TURNOVER = 0.5                 # 最大换手率
COVARIANCE_METHOD = "shrink"       # 协方差估计方法：shrink / ledoit_wolf / oas
EXPECTED_RETURNS_METHOD = "mean"   # 预期收益估计方法：mean / ema / capm
INDIVIDUAL_STOP_LOSS = 0.05        # 个股止损阈值
VAR_CONFIDENCE = 0.95              # VaR置信度
CVAR_CONFIDENCE = 0.95             # CVaR置信度
BARRA_FACTORS = ["size", "value", "momentum", "quality"]  # Barra风格因子
MIN_WEIGHT = 0.001                 # 最小持仓权重
PORTFOLIO_BACKEND = "cvxpy"        # 组合优化后端

# ── 执行监控参数 ──────────────────────────
EXECUTION_DIR = os.path.join(WORK_DIR, "execution")
TRADE_MODE = "paper"              # 交易模式：paper / live
MAX_SINGLE_ORDER_RATIO = 0.02      # 单笔订单最大金额比例
MAX_ORDER_FREQUENCY = 5            # 最大下单频率（笔/分钟）
AUDIT_LOG_PATH = os.path.join(LOG_DIR, "audit.log")
ACCOUNT_STATE_PATH = os.path.join(WORK_DIR, "account_state.json")

# ── 报告生成参数 ──────────────────────────
REPORT_TITLE = "jingnitrader 回测报告"
REPORT_FORMAT = "html"             # 报告格式：html / pdf
INDUSTRY_STANDARD = "sw1"         # 行业分类标准：sw1（一级）/ csi（证监会）
INCLUDE_TEARSHEET = True          # 是否包含完整绩效单页
INCLUDE_HEATMAP = True             # 是否包含因子收益热力图
INCLUDE_ATTRIBUTION = True         # 是否包含收益归因
INCLUDE_TRADES = True             # 是否包含交易记录
CHART_THEME = "light"             # 图表主题：light / dark

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
