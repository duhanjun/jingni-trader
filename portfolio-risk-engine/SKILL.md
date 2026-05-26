---
name: portfolio-risk-engine
description: >
  A股组合优化与风控引擎。基于 PyPortfolioOpt / Riskfolio-Lib 进行组合
  权重优化（均值-方差、风险平价、Black-Litterman、CVaR），支持A股特有
  约束（个股权重上限、行业偏离限制、换手率控制），内置 Barra CNE5 风格
  因子归因与 VaR/CVaR 风险度量，提供组合层面和个股层面的多层止损机制。
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
version: 1.0.0
author: quant-team
dependencies:
  - PyPortfolioOpt
  - riskfolio-lib
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - statsmodels
  - matplotlib
backends:
  - pypfopt
  - riskfolio
---

# portfolio-risk-engine

## 职责

- 接收策略信号/Alpha 预测，生成最优投资组合权重
- 支持多种优化方法：最大夏普、最小方差、风险平价、Black-Litterman、CVaR
- A股特殊约束：单一股票持仓 ≤ 10%、行业偏离 ≤ ±5%、换手率限制
- 风险模型：协方差矩阵估计、Barra CNE5 风格归因、VaR/CVaR
- 多层止损：组合层面单日亏损止损、个股层面破位止损

## 优化方法

- `max_sharpe`: 最大夏普比率组合
- `min_variance`: 最小方差组合
- `risk_parity`: 分层风险平价 (HRP)
- `black_litterman`: Black-Litterman 模型
- `cvar`: CVaR 最优化

## A股约束

- 单一股票持仓权重 ≤ 10%（可配置）
- 行业偏离基准（申万一级）≤ ±5%
- 个股权重下限 0（不可做空）
- 权重和为 1
- 最大换手率限制

## 使用方式

由 `quant-trading-master` 调度，调用 `run(ctx)` 函数。