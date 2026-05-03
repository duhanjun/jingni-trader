---
name: strategy-model-engine
version: 1.0.0
description: 量化策略模型引擎，支持策略模板生成、截面选股、择时模型、参数优化、实验管理和过拟合防范
author: OpenClaw Team
license: MIT
tags:
  - quant-trading
  - strategy
  - model
  - machine-learning
  - backtesting
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scikit-learn>=1.3.0
  - lightgbm>=4.0.0
  - catboost>=1.2.0
  - optuna>=3.0.0
  - mlflow>=2.0.0
  - pandas-ta>=0.3.0
  - ta-lib>=0.4.28
  - pyarrow>=14.0.0
  - scipy>=1.10.0
environment_variables:
  - name: MLFLOW_TRACKING_URI
    description: MLflow跟踪服务器URI
    required: false
    default: "./mlruns"
  - name: EXPERIMENT_NAME
    description: 实验名称
    required: false
    default: "strategy_model_engine"
language: python
python_version: "3.9+"
entry_point: engine.py
---

# strategy-model-engine Skill

## 概述

strategy-model-engine 是一个完整的量化策略模型引擎，提供从策略模板生成到模型训练、参数优化、实验管理和过拟合防范的全流程支持。

## 核心功能

### 1. 策略模板生成
- **趋势跟踪策略**：基于双均线、布林带突破、唐奇安通道
- **均值回归策略**：基于 RSI、布林带均值回归、配对交易
- **配对交易策略**：基于协整检验的配对交易

### 2. 截面选股模型
- **LightGBM 多因子选股**
- **CatBoost 多因子选股**
- 支持多因子输入、行业中性化

### 3. 择时模型
- **基于分钟线技术指标的3日涨跌分类器**
- 支持多种技术指标组合

### 4. 参数优化
- **Optuna 超参搜索**
- 支持 TPE、CMA-ES 等多种优化算法

### 5. 实验管理
- **MLflow 记录**
- 记录因子组合、模型参数、性能指标

### 6. 过拟合防范
- **Purged Group Time Series Split**
- 滚动窗口：36训练、12验证、12测试、间隔2天、行业分组

## 使用示例

```python
from engine import StrategyModelEngine
from config import get_config

# 创建引擎
config = get_config()
engine = StrategyModelEngine(config)

# 生成策略模板
trend_template = engine.generate_strategy_template("trend_following")
mean_reversion_template = engine.generate_strategy_template("mean_reversion")
pair_trading_template = engine.generate_strategy_template("pair_trading")

# 截面选股模型
selection_model = engine.train_stock_selection_model(
    factors=factor_data,
    returns=forward_returns,
    model_type="lightgbm"
)

# 择时模型
timing_model = engine.train_timing_model(
    minute_data=minute_data,
    model_type="xgboost"
)

# 参数优化
best_params = engine.optimize_hyperparameters(
    model_type="lightgbm",
    X=X_train,
    y=y_train,
    n_trials=50
)

# 过拟合防范
cv_splits = engine.get_purged_group_splits(
    data=full_data,
    group_col="industry",
    time_col="datetime"
)

# 实验记录
with engine.start_experiment():
    engine.log_model(selection_model, "stock_selection")
    engine.log_metrics({"ic": 0.05, "icir": 2.0})
```

## 配置说明

详见 references/config_guide.md

## 策略模板说明

详见 references/strategy_templates.md

## 模型使用指南

详见 references/model_guide.md

## 最佳实践

详见 references/best_practices.md
