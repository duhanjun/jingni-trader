---
name: portfolio-risk-engine
version: 1.0.0
description: A股组合优化与风控引擎。基于 PyPortfolioOpt / Riskfolio-Lib 进行组合权重优化（均值-方差、风险平价、Black-Litterman、CVaR），支持A股特有约束（个股权重上限、行业偏离限制、换手率控制），内置 Barra CNE5 风格因子归因与 VaR/CVaR 风险度量，提供组合层面和个股层面的多层止损机制。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - portfolio-engine
  - 组合优化
  - 风控
  - pypfopt
dependencies:
  - PyPortfolioOpt>=1.5.0
  - riskfolio-lib (可选)
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - scikit-learn>=1.3.0
  - statsmodels>=0.14.0
  - matplotlib>=3.7.0
environment_variables:
  - name: OPTIMIZATION_METHOD
    description: 优化方法
    required: false
    default: "max_sharpe"
  - name: PORTFOLIO_DIR
    description: 组合数据目录
    required: false
    default: "./workspace/portfolio"
  - name: RISK_FREE_RATE
    description: 无风险利率
    required: false
    default: "0.03"
language: python
python_version: "3.9+"
entry_point: engine.py
trigger_keywords:
  - 组合优化
  - 风控
  - 仓位
  - 权重
  - 止损
  - VaR
  - CVaR
  - 风险平价
  - Barra
  - 归因
  - 优化
  - 投资组合
---

# portfolio-risk-engine

## 概述

portfolio-risk-engine 是 A 股量化投研的**组合优化与风控引擎**，提供：

1. **多优化方法**：最大夏普、最小方差、风险平价、Black-Litterman、CVaR
2. **A股特有约束**：个股权重上限、行业偏离、换手率控制
3. **风险模型**：协方差矩阵估计、Barra CNE5 风格归因、VaR/CVaR
4. **多层止损**：组合层面单日亏损止损、个股层面破位止损

## 优化方法

| 方法 | 描述 |
|------|------|
| max_sharpe | 最大夏普比率组合 |
| min_variance | 最小方差组合 |
| risk_parity | 分层风险平价 (HRP) |
| black_litterman | Black-Litterman 模型 |
| cvar | CVaR 最优化 |

## A股约束

- 单一股票持仓权重 ≤ 10%
- 行业偏离基准 ≤ ±5%
- 个股权重下限 0（不可做空）
- 权重和为 1

## 使用示例

### Python API

```python
from engine import run
from context import Context

ctx = Context(
    task_id="task_001",
    user_intent="组合优化",
    current_stage="IDLE"
)

result = run(ctx)
```

### CLI 运行

```bash
python engine.py -i "优化我的组合"
```

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)
