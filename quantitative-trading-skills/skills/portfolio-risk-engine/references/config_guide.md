# 配置指南

## 配置类 (Config)

### 协方差估计配置

- `COVARIANCE_METHOD`: str, 默认 'historical'
  - 'historical': 历史协方差
  - 'ewma': 指数加权移动平均

- `EWMA_SPAN`: int, 默认 60
  - EWMA 的时间跨度

### 组合优化配置

- `OPTIMIZATION_METHOD`: str, 默认 'max_sharpe'
  - 'min_variance': 最小方差
  - 'max_sharpe': 最大夏普比率
  - 'risk_parity': 风险平价

- `RISK_FREE_RATE`: float, 默认 0.03
  - 无风险利率

### A 股约束配置

- `MAX_SINGLE_WEIGHT`: float, 默认 0.10 (10%)
  - 单股票最大权重

- `MIN_SINGLE_WEIGHT`: float, 默认 0.0
  - 单股票最小权重

- `MAX_INDUSTRY_DEVIATION`: float, 默认 0.05 (5%)
  - 行业暴露最大偏离

### 止损配置

- `DAILY_DRAWDOWN_THRESHOLD`: float, 默认 -0.03 (-3%)
  - 单日回撤止损阈值

- `MA_PERIOD`: int, 默认 20
  - 均线周期

- `VOLUME_RATIO_THRESHOLD`: float, 默认 2.0
  - 放量倍数阈值

### VaR 配置

- `VAR_METHOD`: str, 默认 'historical'
  - 'historical': 历史模拟法
  - 'parametric': 参数法（正态分布）

- `CONFIDENCE_LEVEL`: float, 默认 0.95 (95%)
  - 置信水平

- `VAR_HORIZON`: int, 默认 1
  - VaR 时间跨度（天数）

### 实时监控配置

- `MAX_POSITION_LIMIT`: float, 默认 10000000.0
  - 持仓限额

- `MARGIN_RATIO`: float, 默认 0.2
  - 保证金比例

- `PROFIT_WARNING_LEVEL`: float, 默认 -0.05 (-5%)
  - 盈亏预警水平

## 使用示例

```python
from config import get_config

# 自定义配置
config = get_config(
    MAX_SINGLE_WEIGHT=0.15,
    CONFIDENCE_LEVEL=0.99
)

# 使用配置
engine = PortfolioRiskEngine(config)
```
