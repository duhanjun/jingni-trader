---
name: portfolio-risk-engine
version: 1.0.0
description: 投资组合风险管理引擎，支持协方差估计、组合优化、风格归因、VaR计算、止损机制和实时风险监控
author: OpenClaw Team
license: MIT
tags:
  - quant-trading
  - risk-management
  - portfolio-optimization
  - pypfopt
  - var
  - cvar
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - pypfopt>=0.5.4
  - cvxpy>=1.3.0
language: python
python_version: "3.9+"
entry_point: engine.py
---

# portfolio-risk-engine Skill

## 概述

portfolio-risk-engine 是面向 A 股的量化投资组合风险管理引擎，提供完整的风险分析和管理能力。

## 核心功能

1. **协方差矩阵估计**：历史协方差、指数加权协方差（EWMA）
2. **组合权重优化**：风险平价、最小方差、最大夏普比率
3. **Barra 风格因子归因**：行业暴露、风格暴露分析（CNE5 模型）
4. **A股约束处理**：单票持仓≤10%、行业暴露偏离±5%
5. **止损机制**：单日回撤-3%硬止损、个股放量跌破20日均线
6. **VaR/CVaR 计算**：历史模拟法、参数法
7. **实时风险监控**：持仓限额、保证金、盈亏预警

## 目录结构

```
portfolio-risk-engine/
├── SKILL.md
├── README.md
├── __init__.py
├── config.py
├── engine.py
├── requirements.txt
├── covariance/
│   ├── __init__.py
│   └── covariance_estimator.py
├── optimization/
│   ├── __init__.py
│   └── optimizer.py
├── attribution/
│   ├── __init__.py
│   └── barra_attribution.py
├── constraints/
│   ├── __init__.py
│   └── constraint_handler.py
├── stop_loss/
│   ├── __init__.py
│   └── stop_loss_manager.py
├── var/
│   ├── __init__.py
│   └── var_calculator.py
├── monitoring/
│   ├── __init__.py
│   └── risk_monitor.py
├── references/
│   └── ...
└── tests/
    └── ...
```

## 使用示例

```python
from engine import PortfolioRiskEngine
from config import get_config
import pandas as pd

# 创建引擎
config = get_config()
engine = PortfolioRiskEngine(config)

# 示例：组合优化
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
weights = engine.optimize_portfolio(
    returns=returns,
    method="max_sharpe",
    max_single_weight=0.1
)
print(weights)

# 示例：计算 VaR
var_result = engine.calculate_var(returns, weights, method="historical", confidence_level=0.95)
print(f"VaR (95%): {var_result['var']:.4f}")
```

## 配置说明

详见 references/config_guide.md

## API 文档

详见 references/api_reference.md
