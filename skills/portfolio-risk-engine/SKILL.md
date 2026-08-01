---
name: portfolio-risk-engine
version: 1.0.0
description: A股组合优化与风控引擎。基于 PyPortfolioOpt / Riskfolio-Lib 进行组合权重优化（均值-方差、风险平价、Black-Litterman、CVaR），支持A股特有约束（个股权重上限、行业偏离限制、换手率控制），内置 Barra CNE5 风格因子归因与 VaR/CVaR 风险度量，提供组合层面和个股层面的多层止损机制。默认使用分层风险平价（HRP）优化方法。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - portfolio-engine
  - 组合优化
  - 风控
  - pypfopt
  - riskfolio
dependencies:
  - PyPortfolioOpt>=1.5.0
  - riskfolio-lib (可选)
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - scikit-learn>=1.3.0
  - statsmodels>=0.14.0
  - matplotlib>=3.7.0
  - cvxpy (可选，组合优化后端)
environment_variables:
  - name: OPTIMIZATION_METHOD
    description: 优化方法（max_sharpe / min_variance / hierarchical_risk_parity / black_litterman / cvar）
    required: false
    default: "hierarchical_risk_parity"
  - name: QUANT_WORK_DIR
    description: 工作目录根路径
    required: false
    default: "./workspace"
  - name: RISK_FREE_RATE
    description: 无风险利率
    required: false
    default: "0.03"
  - name: PORTFOLIO_BACKEND
    description: 组合优化后端（cvxpy）
    required: false
    default: "cvxpy"
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
  - HRP
  - Walk-Forward
---

# portfolio-risk-engine

## 概述

portfolio-risk-engine 是 A 股量化投研的**组合优化与风控引擎**，提供：

1. **多优化方法**：最大夏普、最小方差、分层风险平价（HRP，默认）、Black-Litterman、CVaR
2. **A股特有约束**：个股权重上限、行业偏离、换手率控制
3. **风险模型**：协方差矩阵估计（shrink / ledoit_wolf / oas）、Barra CNE5 风格归因、VaR/CVaR
4. **多层止损**：组合层面单日亏损止损、个股层面破位止损
5. **Walk-Forward 验证**：稳健性验证

## 优化方法

| 方法 | 描述 | 后端 |
|------|------|------|
| max_sharpe | 最大夏普比率组合 | PyPortfolioOpt |
| min_variance | 最小方差组合 | PyPortfolioOpt |
| hierarchical_risk_parity（默认） | 分层风险平价（HRP） | PyPortfolioOpt |
| black_litterman | Black-Litterman 模型 | PyPortfolioOpt |
| cvar | CVaR 最优化 | PyPortfolioOpt |

## A股约束

- 单一股票持仓权重 ≤ 10%
- 行业偏离基准 ≤ ±5%
- 个股权重下限 0.001（不可做空）
- 权重和为 1
- 最大换手率 ≤ 50%

## 风险参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| ESTIMATION_PERIOD | 252 | 协方差估计周期（交易日） |
| MAX_TURNOVER | 0.5 | 最大换手率 |
| COVARIANCE_METHOD | shrink | 协方差估计方法 |
| EXPECTED_RETURNS_METHOD | mean | 预期收益估计方法 |
| VAR_CONFIDENCE | 0.95 | VaR 置信度 |
| CVAR_CONFIDENCE | 0.95 | CVaR 置信度 |
| INDIVIDUAL_STOP_LOSS | 0.05 | 个股止损阈值 |
| MAX_DAILY_LOSS_RATIO | 0.03 | 单日最大亏损 3% |

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

## 优化模块

可通过 `from engine import optimizations` 访问以下优化模块：

- **组合优化器 v2**：`PortfolioOptimizerV2`
- **风控引擎**：`RiskEngine`
- **断路器 v2**：`CircuitBreakerV2`
- **Walk-Forward 验证**：`WalkForwardValidator`

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)