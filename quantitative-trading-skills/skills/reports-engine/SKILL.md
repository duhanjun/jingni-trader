---
name: reports-engine
version: 1.0.0
description: A股量化策略报告引擎，支持quantstats报告、风格暴露分析、Brinson归因、行业暴露分析等
author: OpenClaw Team
license: MIT
tags:
  - quant-trading
  - A股
  - report
  - quantstats
  - Brinson
  - 风格暴露
  - plotly
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - plotly>=5.14.0
  - quantstats>=0.0.59
  - empyrical>=0.5.5
  - scikit-learn>=1.3.0
  - seaborn>=0.12.0
environment_variables:
  - name: REPORT_DIR
    description: 报告存储目录
    required: false
    default: "./reports"
  - name: BENCHMARK_CODE
    description: 基准指数代码
    required: false
    default: "000300.SH"
language: python
python_version: "3.9+"
entry_point: engine.py
---

# reports-engine Skill

## 概述

reports-engine 是 A 股量化投研的完整报告引擎，提供：

1. **quantstats 一键报告**：自动生成完整的量化策略绩效报告
2. **滚动夏普与月度热力图**：直观展示策略风险收益的时变特征
3. **A股风格暴露分析**：大盘/小盘/成长/价值暴露，沪深300基准
4. **行业暴露分析**：申万一级行业暴露可视化
5. **Brinson归因**：超额收益分解（配置、选择、交互）
6. **动态净值曲线**：plotly交互图表
7. **因子衰减走势**：matplotlib静态图表
8. **过拟合风险警示**：当OOS夏普<全样本夏普60%时发出警告

## 使用示例

```python
from engine import ReportsEngine
from config import get_config
import pandas as pd

# 创建引擎
config = get_config()
engine = ReportsEngine(config)

# 生成完整报告
portfolio_returns = pd.Series(...)  # 策略日收益率
benchmark_returns = pd.Series(...)  # 基准日收益率
factor_exposures = pd.DataFrame(...)  # 因子暴露
industry_weights = pd.DataFrame(...)  # 行业权重

report = engine.generate_full_report(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    factor_exposures=factor_exposures,
    industry_weights=industry_weights
)

# 保存报告
engine.save_report(report, "my_strategy_report")
```

## 核心功能模块

| 模块 | 说明 |
|------|------|
| quantstats_report.py | quantstats一键生成报告 |
| rolling_sharpe.py | 滚动夏普、月度热力图 |
| style_exposure.py | A股风格暴露分析 |
| industry_exposure.py | 申万一级行业暴露 |
| brinson_attribution.py | Brinson归因分析 |
| equity_curve.py | 动态净值曲线 |
| factor_decay.py | 因子衰减走势图 |
| overfitting_warning.py | 过拟合风险警示 |

## 配置说明

详见 references/config_guide.md

## API 文档

详见 references/api_reference.md
