# API 参考文档

本文档提供 portfolio-risk-engine 的完整 API 参考。

## 核心类

### PortfolioOptimizer

组合优化器类。

#### 方法

##### `__init__()`

初始化优化器。

```python
optimizer = PortfolioOptimizer()
```

##### `estimate_expected_returns(returns, method) -> pd.Series`

估计预期收益。

##### `estimate_covariance(returns, method) -> pd.DataFrame`

估计协方差矩阵。

##### `optimize(expected_rets, cov_matrix, method, constraints, current_weights) -> Tuple[pd.Series, Dict]`

执行组合优化。

**返回：**
```python
(weights, metrics)
```

### AShareConstraints

A股组合约束管理类。

#### 方法

##### `validate_constraints(weights, constraints) -> Dict[str, bool]`

验证组合约束。

### RiskManager

风险计算与止损管理类。

#### 方法

##### `check_portfolio_stop(current_nav) -> Dict`

检查组合层面止损。

##### `check_individual_stop(current_prices, entry_prices) -> pd.Series`

检查个股止损信号。

##### `calc_var(returns, confidence) -> float`

计算 VaR。

##### `calc_cvar(returns, confidence) -> float`

计算 CVaR。

##### `calc_portfolio_var_cvar(returns_df, weights, confidence) -> Dict`

计算组合 VaR/CVaR。

## 标准入口函数

### `run(ctx) -> Dict[str, Any]`

Skill 标准入口函数。

**参数：**
- `ctx` (Context): 上下文对象

**返回：**
```python
{
    "success": bool,
    "artifact_path": str,      # 权重文件路径
    "metadata": {
        "weights": {...},
        "metrics": {...},
        "var_cvar": {...},
        "stop_signals": {...},
        "constraint_check": {...},
        "optimization_method": str,
        "num_assets": int,
    },
    "error": str
}
```

**示例：**

```python
from engine import run

result = run(ctx)
if result['success']:
    print(f"优化完成，权重文件: {result['artifact_path']}")
    print(f"绩效指标: {result['metadata']['metrics']}")
```

## CLI 使用

```bash
python engine.py context.json
```
