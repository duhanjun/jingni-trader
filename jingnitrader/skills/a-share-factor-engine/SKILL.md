---
name: a-share-factor-engine
version: 1.0.0
description: A股阿尔法因子研究与构建引擎。支持定义和计算A股专属的Alpha因子（动量反转、市值、换手率、资金流、事件驱动等），提供因子IC分析（含行业中性化处理）、分层回测、相关性去冗余、多因子融合等功能。底层技术指标计算支持TA-Lib和pandas-ta双后端切换。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - factor-engine
  - alpha-factor
  - talib
  - pandas-ta
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - scikit-learn>=1.3.0
  - statsmodels>=0.14.0
  - alphalens>=0.4.0
  - ta-lib (可选)
  - pandas-ta (可选)
environment_variables:
  - name: FACTOR_BACKEND
    description: 因子计算后端
    required: false
    default: "talib"
  - name: FACTOR_DIR
    description: 因子数据存储目录
    required: false
    default: "./workspace/factors"
  - name: IC_TYPE
    description: IC计算方式
    required: false
    default: "spearman"
language: python
python_version: "3.9+"
entry_point: engine.py
trigger_keywords:
  - 因子
  - 因子研究
  - Alpha因子
  - IC分析
  - 因子挖掘
  - 因子中性化
  - 因子相关性
  - 多因子
---

# a-share-factor-engine

## 概述

a-share-factor-engine 是 A 股量化投研的**因子研究与构建引擎**，提供：

1. **A股专用Alpha因子**：动量反转、市值、换手率、资金流、波动率等
2. **行业中性化处理**：市值+行业中性回归
3. **因子IC分析**：Spearman Rank IC / Pearson IC
4. **因子相关性分析**：去冗余处理
5. **多因子融合**：等权/IC加权融合

## 因子定义体系

### 动量因子
- `momentum_20d`: 20日动量（A股反转效应，预期负IC）
- `momentum_60d`: 60日动量
- `reversal_5d`: 5日反转

### 规模因子
- `lncap`: 对数市值
- `size`: 市值分组

### 交易因子
- `turnover_20d`: 20日平均换手率
- `volume_ratio`: 量比

### 资金流因子
- `money_flow_20d`: 20日累计资金流

### 质量因子
- `roe`: ROE
- `gross_margin`: 毛利率

## 使用示例

### Python API

```python
from engine import run
from context import Context

ctx = Context(
    task_id="task_001",
    user_intent="计算因子",
    current_stage="IDLE"
)
ctx.stock_pool = ["000001.SZ"]
ctx.start_date = "2021-01-01"
ctx.end_date = "2024-01-01"

result = run(ctx)
```

### CLI 运行

```bash
python engine.py -i "计算反转因子"
```

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)
