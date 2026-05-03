# API 参考文档

## PortfolioRiskEngine

主引擎类，整合所有风险管理功能。

### 初始化

```python
from engine import PortfolioRiskEngine
from config import get_config

config = get_config()
engine = PortfolioRiskEngine(config)
```

### 方法

#### estimate_covariance(returns, method=None)

估计协方差矩阵。

- **参数**:
  - `returns`: pd.DataFrame, 收益率数据
  - `method`: str, 可选，'historical' 或 'ewma'

- **返回**: pd.DataFrame, 协方差矩阵

#### optimize_portfolio(returns, method=None, cov_matrix=None, max_weight=None, min_weight=None, apply_constraints=True)

优化投资组合权重。

- **参数**:
  - `returns`: pd.DataFrame, 收益率数据
  - `method`: str, 可选，'min_variance', 'max_sharpe', 'risk_parity'
  - `cov_matrix`: pd.DataFrame, 可选，协方差矩阵
  - `max_weight`: float, 可选，单资产最大权重
  - `min_weight`: float, 可选，单资产最小权重
  - `apply_constraints`: bool, 是否应用约束

- **返回**: Dict, 包含 weights 和 performance

#### calculate_var(returns, weights=None, method=None, cov_matrix=None)

计算 VaR 和 CVaR。

- **参数**:
  - `returns`: pd.DataFrame, 收益率数据
  - `weights`: Dict, 可选，权重字典
  - `method`: str, 可选，'historical' 或 'parametric'
  - `cov_matrix`: pd.DataFrame, 可选，协方差矩阵（参数法）

- **返回**: Dict, 包含 var, cvar 等

#### analyze_risk(returns, weights=None, factor_data=None, industry_data=None, benchmark_weights=None)

全面风险分析。

- **参数**:
  - `returns`: pd.DataFrame, 收益率数据
  - `weights`: Dict, 可选，权重字典
  - `factor_data`: pd.DataFrame, 可选，因子暴露数据
  - `industry_data`: pd.Series, 可选，行业数据
  - `benchmark_weights`: Dict, 可选，基准权重

- **返回**: Dict, 综合风险分析结果

#### check_stop_loss(portfolio_returns, individual_prices=None, individual_volumes=None)

检查止损条件。

- **参数**:
  - `portfolio_returns`: pd.Series, 组合收益率
  - `individual_prices`: Dict, 可选，个股价格
  - `individual_volumes`: Dict, 可选，个股成交量

- **返回**: Dict, 止损检查结果

#### monitor_risk(positions, equity, returns, initial_investment=1.0)

实时风险监控。

- **参数**:
  - `positions`: Dict, 持仓市值
  - `equity`: float, 权益
  - `returns`: pd.Series, 收益率
  - `initial_investment`: float, 初始投资

- **返回**: Dict, 监控结果
