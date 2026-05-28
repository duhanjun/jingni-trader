---
name: strategy-model-engine
version: 1.0.0
description: A股策略开发与模型训练引擎。支持截面多因子选股模型（LightGBM/CatBoost）、择时模型（基于技术指标分类器）、超参数优化（Optuna）以及实验管理（MLflow）。内置过拟合防范机制（分组时序交叉验证）与策略模板库。
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
    description: 模型类型
    required: false
    default: "lightgbm"
  - name: MODEL_DIR
    description: 模型存储目录
    required: false
    default: "./quant_workspace/models"
  - name: LABEL_TYPE
    description: 标签类型
    required: false
    default: "regression"
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
2. **超参数优化**：Optuna 自动调参
3. **实验追踪**：MLflow 模型版本管理
4. **防过拟合**：分组时序交叉验证 (Purged Group Time Series Split)
5. **策略模板**：规则型策略（反转、趋势等）

## 模型训练流程

1. **数据准备**：加载因子数据，构建特征矩阵X和标签y
2. **样本划分**：Purged Group Time Series Split
3. **超参数搜索**：Optuna 在验证集上优化
4. **模型训练**：使用最优参数训练
5. **样本外测试**：在最近时间段评估
6. **模型保存**：MLflow 记录实验，保存模型文件

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
