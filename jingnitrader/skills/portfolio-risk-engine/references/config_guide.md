# 配置指南

本文档说明 portfolio-risk-engine 的配置选项。

## 环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| OPTIMIZATION_METHOD | 优化方法 | 否 | "max_sharpe" |
| PORTFOLIO_DIR | 组合数据目录 | 否 | "./quant_workspace/portfolio" |
| RISK_FREE_RATE | 无风险利率 | 否 | 0.03 |
| COVARIANCE_METHOD | 协方差估计方法 | 否 | "ledoit_wolf" |

## 配置文件

配置文件位于 `scripts/config.py`。

### 优化配置

```python
OPTIMIZATION_METHOD = os.getenv("OPTIMIZATION_METHOD", "max_sharpe")
PORTFOLIO_DIR = os.path.expanduser(os.getenv("PORTFOLIO_DIR", "./quant_workspace/portfolio"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.03"))
```

### A股约束配置

```python
MAX_SINGLE_STOCK_WEIGHT = 0.10      # 单票最大持仓 10%
MAX_INDUSTRY_DEVIATION = 0.05      # 行业偏离 ±5%
MAX_TURNOVER = 0.50               # 最大换手率 50%
MIN_WEIGHT = 0.0                  # 最小权重（不可做空）
```

### 风险配置

```python
MAX_DAILY_LOSS_RATIO = 0.02       # 单日最大亏损 2%
INDIVIDUAL_STOP_LOSS = 0.07       # 个股止损 7%
VAR_CONFIDENCE = 0.95             # VaR 置信度
CVAR_CONFIDENCE = 0.95            # CVaR 置信度
BARRA_FACTORS = "SIZE,BP,EP,ROE,Growth"
```

## 优化方法

| 方法 | 说明 |
|------|------|
| max_sharpe | 最大夏普比率 |
| min_variance | 最小方差 |
| risk_parity | 分层风险平价 |
| black_litterman | Black-Litterman |
| cvar | CVaR 优化 |

## 使用示例

```python
import os
os.environ['OPTIMIZATION_METHOD'] = 'risk_parity'

from engine import run, PortfolioOptimizer, RiskManager

result = run(ctx)
```
