---
name: strategy-model-engine
description: >
  A股策略开发与模型训练引擎。支持截面多因子选股模型（LightGBM/CatBoost）、
  择时模型（基于技术指标分类器）、超参数优化（Optuna）以及实验管理（MLflow）。
  内置过拟合防范机制（分组时序交叉验证）与策略模板库。
trigger_keywords:
  - 模型训练
  - 策略开发
  - 截面选股
  - 择时
  - 机器学习
  - LightGBM
  - 超参数优化
  - 实验管理
version: 1.0.0
author: quant-team
dependencies:
  - lightgbm
  - catboost
  - scikit-learn
  - optuna
  - mlflow
  - pandas
  - numpy
model_types:
  - lightgbm
  - catboost
  - logistic_regression
  - random_forest
---

# strategy-model-engine

## 职责

- 接收因子数据和未来收益标签，训练截面选股或择时模型
- 支持多种机器学习算法，通过配置切换
- 超参数优化（Optuna）
- 实验追踪与模型版本管理（MLflow）
- 严格的分组时序交叉验证防过拟合
- 输出训练好的模型及预测信号
- 策略模板库：基于规则的简单策略也在此处生成信号

## 模型训练流程

1. **数据准备**：加载因子数据，构建特征矩阵X和标签y（未来N日收益）
2. **样本划分**：Purged Group Time Series Split
3. **超参数搜索**：Optuna在验证集上优化
4. **模型训练**：使用最优参数在训练集上训练
5. **样本外测试**：在最近的时间段上评估
6. **模型保存**：使用MLflow记录实验，保存模型文件

## 使用方式

由 `quant-trading-master` 调度，调用 `run(ctx)` 函数。
