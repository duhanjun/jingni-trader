"""
模型引擎专属配置
"""
import os

# ── 模型选择 ──────────────────────────────
MODEL_TYPE = os.environ.get("MODEL_TYPE", "lightgbm")
# 可选: lightgbm, catboost, logistic_regression, random_forest

# ── 模型存储目录 ──────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "./workspace/models")

# ── 超参数优化 ────────────────────────────
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", 100))
OPTUNA_TIMEOUT = int(os.environ.get("OPTUNA_TIMEOUT", 3600))  # 秒
OPTUNA_N_JOBS = int(os.environ.get("OPTUNA_N_JOBS", 1))

# ── 训练参数 ──────────────────────────────
TRAIN_WINDOW_MONTHS = int(os.environ.get("TRAIN_WINDOW_MONTHS", 36))
VALIDATION_WINDOW_MONTHS = int(os.environ.get("VALIDATION_WINDOW_MONTHS", 12))
TEST_WINDOW_MONTHS = int(os.environ.get("TEST_WINDOW_MONTHS", 12))
PURGE_GAP_DAYS = int(os.environ.get("PURGE_GAP_DAYS", 2))

# ── 标签定义 ──────────────────────────────
FORWARD_PERIOD = int(os.environ.get("FORWARD_PERIOD", 20))  # 未来N日收益
LABEL_TYPE = os.environ.get("LABEL_TYPE", "regression")  # regression / classification

# ── MLflow 配置 ───────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MODEL_DIR}/mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "a_share_quant")

# ── 自动创建目录 ──────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
