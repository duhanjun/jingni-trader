# 配置指南

本文档说明 strategy-model-engine 的配置选项。

## 环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| MODEL_TYPE | 模型类型 | 否 | "lightgbm" |
| MODEL_DIR | 模型存储目录 | 否 | "./workspace/models" |
| LABEL_TYPE | 标签类型 | 否 | "regression" |

## 配置文件

配置文件位于 `scripts/config.py`。

### 模型配置

```python
MODEL_TYPE = os.getenv("MODEL_TYPE", "lightgbm")
MODEL_DIR = os.path.expanduser(os.getenv("MODEL_DIR", "./workspace/models"))
LABEL_TYPE = os.getenv("LABEL_TYPE", "regression")
```

### 超参优化配置

```python
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "50"))
OPTUNA_TIMEOUT = int(os.getenv("OPTUNA_TIMEOUT", "600"))
```

### 数据窗口配置

```python
TRAIN_WINDOW_MONTHS = 24
VALIDATION_WINDOW_MONTHS = 6
TEST_WINDOW_MONTHS = 6
PURGE_GAP_DAYS = 5
FORWARD_PERIOD = 5
```

### MLflow 配置

```python
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "a-share-strategy")
```

## 模型类型

| 模型 | 说明 |
|------|------|
| lightgbm | LightGBM 梯度提升 |
| catboost | CatBoost |
| logistic_regression | 逻辑回归 |
| random_forest | 随机森林 |

## 使用示例

```python
import os
os.environ['MODEL_TYPE'] = 'lightgbm'
os.environ['LABEL_TYPE'] = 'classification'

from engine import run, ModelEngine

result = run(ctx)
```
