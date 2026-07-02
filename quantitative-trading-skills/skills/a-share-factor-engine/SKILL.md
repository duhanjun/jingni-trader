---
name: a-share-factor-engine
version: 1.0.0
description: A股多因子选股引擎，支持因子构建、IC分析、相关性分析、因子融合、存储与复用
author: OpenClaw Team
license: MIT
tags:
  - quant-trading
  - A股
  - factor-engine
  - multi-factor
  - alpha
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - pandas-ta>=0.3.0
  - ta-lib>=0.4.28
  - pyarrow>=14.0.0
  - fastparquet>=2023.10.0
  - scipy>=1.10.0
  - scikit-learn>=1.3.0
environment_variables:
  - name: FACTOR_DATA_DIR
    description: 因子数据存储目录
    required: false
    default: "./factor_data"
  - name: TA_LIBRARY
    description: 技术分析库选择：talib 或 pandas-ta
    required: false
    default: "pandas-ta"
language: python
python_version: "3.9+"
entry_point: engine.py
---

# a-share-factor-engine Skill

## 概述

a-share-factor-engine 是 A 股量化投研的多因子选股引擎，提供：

1. **多因子库**：技术因子、动量、价值、质量、情绪、A股专用因子
2. **IC 分析**：IC 序列、ICIR、行业中性化
3. **相关性分析**：相关性矩阵、去冗余建议
4. **多 Alpha 融合**：等权、IC 加权、风险归因加权
5. **存储与复用**：Parquet 文件存储、增量更新
6. **对抗性验证**：训练/测试分布一致性检查

## 因子列表

### 经典技术因子
- MA、EMA、MACD、RSI、KDJ、布林带、威廉指标等

### 动量因子
- 1个月、3个月、6个月动量，相对强弱
- 动量衰减动量，量价共振

### 价值因子
- PE、PB、PS、PCF、EV/EBITDA

### 质量因子
- ROE、ROA、毛利率、净利率、资产负债率

### 情绪因子
- 波动率、换手率、资金流、异常换手率

### A 股专用因子
- 1个月反转、lncap 市值、换手率、资金流、事件因子

## 使用示例

```python
from engine import AShareFactorEngine
from config import get_config

# 创建引擎
config = get_config()
engine = AShareFactorEngine(config)

# 计算因子
factors = engine.compute_factors(
    price_data=df,
    factor_list=["momentum_1m", "lncap", "turnover"]
)

# IC 分析
ic_report = engine.analyze_ic(factors, forward_returns)

# 相关性分析
corr_matrix = engine.analyze_correlation(factors)

# 因子融合
combined_alpha = engine.combine_factors(factors, method="ic_weighted")

# 保存因子
engine.save_factors(factors, "my_factors.parquet")

# 加载因子
factors_loaded = engine.load_factors("my_factors.parquet")
```

## 配置说明

详见 references/config_guide.md

## 因子说明

详见 references/factor_description.md

## IC 分析指南

详见 references/ic_analysis_guide.md

## 多因子融合方法

详见 references/ensemble_methods.md
