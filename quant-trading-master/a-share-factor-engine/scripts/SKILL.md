---
name: a-share-factor-engine
description: >
  A股阿尔法因子研究与构建引擎。支持定义和计算A股专属的Alpha因子（动量反转、
  市值、换手率、资金流、事件驱动等），提供因子IC分析（含行业中性化处理）、
  分层回测、相关性去冗余、多因子融合等功能。底层技术指标计算支持TA-Lib和
  pandas-ta双后端切换。
trigger_keywords:
  - 因子
  - 因子研究
  - Alpha因子
  - IC分析
  - 因子挖掘
  - 因子中性化
  - 因子相关性
  - 多因子
version: 1.0.0
author: quant-team
dependencies:
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - statsmodels
  - alphalens
  - ta-lib (可选)
  - pandas-ta (可选)
backends:
  - talib
  - pandas_ta
---

# a-share-factor-engine

## 职责

- 根据Context中的数据和参数计算Alpha因子
- A股专用因子：1个月反转、市值因子、换手率因子、资金流因子、事件因子
- 行业中性化处理：对因子进行市值+行业中性回归
- 单因子IC分析（Spearman Rank IC / Pearson IC）
- 因子分层回测（按因子值分组，计算各组超额收益）
- 因子相关性分析与冗余剔除
- 多因子等权/IC加权融合

## 因子定义体系

### 动量因子
- `momentum_20d`: 20日动量（A股反转效应，预期负IC）
- `momentum_60d`: 60日动量
- `reversal_5d`: 5日反转

### 规模因子
- `lncap`: 对数市值
- `size`: 市值分组

### 交易因子
- `turnover_20d`: 20日平均换手率（低换手溢价）
- `volume_ratio`: 量比

### 资金流因子
- `money_flow_20d`: 20日累计资金流

### 质量因子
- `roe`: ROE
- `gross_margin`: 毛利率
- `net_profit_growth`: 净利润增速

## 使用方式

由 `quant-trading-master` 调度，调用 `run(ctx)` 函数。