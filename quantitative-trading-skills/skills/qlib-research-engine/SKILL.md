---
name: qlib-research-engine
version: 1.0.0
description: Qlib因子研究引擎，支持AI量化因子研究、因子挖掘、模型训练和解释性分析
author: OpenClaw Team
license: MIT
tags:
  - quant-trading
  - A股
  - factor-research
  - Qlib
  - AI
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - qlib>=0.2.0
  - lightgbm>=4.0.0
  - scikit-learn>=1.3.0
  - shap>=0.42.0
  - matplotlib>=3.7.0
environment_variables:
  - name: QLIB_DATA_DIR
    description: Qlib数据目录
    required: false
    default: "./qlib_data"
  - name: QLIB_CACHE_DIR
    description: Qlib缓存目录
    required: false
    default: "./qlib_cache"
language: python
python_version: "3.9+"
entry_point: engine.py
---

# qlib-research-engine Skill

## 概述

qlib-research-engine 是基于微软Qlib的AI量化因子研究引擎，提供：

1. **因子库**：内置360+因子（Alpha158/Alpha360）
2. **模型训练**：LightGBM、LSTM、Transformer支持
3. **因子挖掘**：自动化因子生成与验证
4. **解释性分析**：SHAP值、特征重要性
5. **回测集成**：与Backtrader无缝对接

## 核心模块

| 模块 | 说明 |
|------|------|
| BaseResearchEngine | 研究引擎抽象基类 |
| FactorAnalyzer | 因子分析器 |
| ModelTrainer | 模型训练器 |
| DataPreprocessor | 数据预处理 |
| SHAPAnalyzer | SHAP解释性分析 |

## 使用示例

```python
from engine import QlibResearchEngine
from config import get_config

# 创建引擎
config = get_config()
engine = QlibResearchEngine(config)

# 初始化Qlib
engine.init_qlib()

# 使用Alpha360因子
handler = engine.get_alpha360_handler(
    start_time="2020-01-01",
    end_time="2024-12-31"
)

# 训练多因子模型
model = engine.train_model(handler, model_type="lightgbm")

# 生成选股信号
predictions = engine.generate_signals(model, handler)

# SHAP分析
shap_values = engine.analyze_with_shap(model, handler)
```

## 支持的因子库

| 因子库 | 说明 |
|--------|------|
| Alpha158 | 经典158因子 |
| Alpha360 | 扩展360因子 |
| Custom | 自定义因子 |

## 支持的模型

| 模型 | 说明 |
|------|------|
| LightGBM | 梯度提升树 |
| XGBoost | XGBoost |
| LSTM | 深度学习时序模型 |
| Transformer | 注意力机制模型 |

## 配置说明

详见 references/config_guide.md

## API 文档

详见 references/api_reference.md