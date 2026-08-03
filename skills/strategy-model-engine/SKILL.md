---
name: strategy-model-engine
version: 1.0.0
description: A股策略开发与模型训练引擎。支持截面多因子选股模型（LightGBM/CatBoost/逻辑回归/随机森林）、超参数优化（Optuna）以及实验管理（MLflow）。内置过拟合防范机制（分组时序交叉验证 Purged Group Time Series Split）与策略模板库。支持 alpha 表达式引擎和动态权重分配。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - model-engine
  - 机器学习
  - LightGBM
  - 策略开发
dependencies:
  - lightgbm>=4.0.0
  - catboost>=1.2.0
  - scikit-learn>=1.3.0
  - optuna>=3.0.0
  - mlflow>=2.0.0
  - pandas>=2.0.0
  - numpy>=1.24.0
environment_variables:
  - name: MODEL_TYPE
    description: 模型类型（lightgbm / catboost / logistic_regression / random_forest）
    required: false
    default: "lightgbm"
  - name: QUANT_WORK_DIR
    description: 工作目录根路径
    required: false
    default: "./workspace"
  - name: LABEL_TYPE
    description: 标签类型（regression / classification）
    required: false
    default: "regression"
  - name: OPTUNA_TRIALS
    description: Optuna 超参搜索次数
    required: false
    default: "100"
  - name: OPTUNA_TIMEOUT
    description: Optuna 超时时间（秒）
    required: false
    default: "3600"
  - name: MLFLOW_TRACKING_URI
    description: MLflow 跟踪服务 URI
    required: false
  - name: MLFLOW_EXPERIMENT_NAME
    description: MLflow 实验名
    required: false
    default: "jingnitrader"
language: python
python_version: "3.9+"
entry_point: engine.py
model_types:
  - lightgbm
  - catboost
  - logistic_regression
  - random_forest
trigger_keywords:
  - 模型训练
  - 策略开发
  - 截面选股
  - 择时
  - 机器学习
  - LightGBM
  - 超参数优化
  - 实验管理
---

# strategy-model-engine

## 概述

strategy-model-engine 是 A 股量化投研的**策略开发与模型训练引擎**，提供：

1. **多模型支持**：LightGBM、CatBoost、逻辑回归、随机森林
2. **超参数优化**：Optuna 自动调参（支持搜索次数和超时限制）
3. **实验追踪**：MLflow 模型版本管理
4. **防过拟合**：分组时序交叉验证（Purged Group Time Series Split），含清洗期（purge gap）

## 模型训练流程

1. **数据准备**：加载因子数据，构建特征矩阵 X 和标签 y
2. **样本划分**：Purged Group Time Series Split（训练窗口 36 月 / 验证窗口 12 月 / 测试窗口 12 月）
3. **超参数搜索**：Optuna 在验证集上优化
4. **模型训练**：使用最优参数训练
5. **样本外测试**：在最近时间段评估
6. **模型保存**：MLflow 记录实验，保存模型文件（joblib）

## 模型类型

| 模型 | 适用场景 | 说明 |
|------|---------|------|
| lightgbm | 截面多因子选股 | 默认模型，速度快、效果好 |
| catboost | 含类别特征的截面选股 | 自动处理类别特征 |
| logistic_regression | 涨跌二分类预测 | 简单可解释 |
| random_forest | 非线性特征交互 | 防过拟合能力较强 |

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TRAIN_WINDOW_MONTHS | 36 | 训练窗口（月） |
| VALIDATION_WINDOW_MONTHS | 12 | 验证窗口（月） |
| TEST_WINDOW_MONTHS | 12 | 测试窗口（月） |
| PURGE_GAP_DAYS | 2 | 清洗期（天），防止训练集和验证集重叠 |
| FORWARD_PERIOD | 20 | 前视期（天），标签为 T+20 日收益 |
| OPTUNA_TRIALS | 100 | Optuna 搜索次数 |
| OPTUNA_TIMEOUT | 3600 | Optuna 超时（秒） |

## 使用示例

### Python API

```python
from engine import run
from context import Context

ctx = Context(
    task_id="task_001",
    user_intent="训练模型",
    current_stage="IDLE"
)

result = run(ctx)
```

### CLI 运行

```bash
python engine.py -i "训练LightGBM模型"
```

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)