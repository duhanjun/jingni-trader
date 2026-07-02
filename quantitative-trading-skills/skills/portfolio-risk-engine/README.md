# 投资组合风险管理引擎 (Portfolio Risk Engine)

## 概述

这是一个面向 A 股市场的量化投资组合风险管理引擎，提供完整的风险分析和管理功能。

## 功能特性

1. **协方差矩阵估计**
   - 历史协方差
   - 指数加权移动平均 (EWMA) 协方差

2. **组合权重优化**
   - 最小方差组合
   - 最大夏普比率组合
   - 风险平价组合 (HRP)

3. **Barra 风格因子归因**
   - 行业暴露分析
   - 风格因子暴露分析 (CNE5 模型)

4. **A 股约束处理**
   - 单股票持仓限制
   - 行业暴露偏离限制

5. **止损机制**
   - 单日回撤硬止损
   - 放量跌破均线止损

6. **VaR/CVaR 计算**
   - 历史模拟法
   - 参数法 (正态分布)

7. **实时风险监控**
   - 持仓限额监控
   - 保证金监控
   - 盈亏预警

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from engine import PortfolioRiskEngine
from config import get_config
import pandas as pd
import numpy as np

# 创建引擎
config = get_config()
engine = PortfolioRiskEngine(config)

# 生成示例数据
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=252)
assets = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '600519.SH']
returns = pd.DataFrame(
    np.random.normal(0, 0.02, (252, 5)),
    index=dates,
    columns=assets
)

# 优化组合
result = engine.optimize_portfolio(returns, method='max_sharpe')
print("最优权重:", result['weights'])
print("组合表现:", result['performance'])

# 计算 VaR
var_result = engine.calculate_var(returns, result['weights'])
print(f"VaR (95%): {var_result['var']:.4f}")
print(f"CVaR (95%): {var_result['cvar']:.4f}")

# 全面风险分析
risk_analysis = engine.analyze_risk(returns, result['weights'])
print("风险分析:", risk_analysis['constraints'])
```

## 项目结构

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
│   ├── api_reference.md
│   └── config_guide.md
└── tests/
    └── test_engine.py
```

## 文档

- [API 参考](references/api_reference.md)
- [配置指南](references/config_guide.md)

## 测试

```bash
cd tests
python test_engine.py
```

## 依赖项

- pandas>=2.0.0
- numpy>=1.24.0
- scipy>=1.10.0
- pypfopt>=0.5.4
- cvxpy>=1.3.0

## 许可证

MIT License
