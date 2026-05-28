# API 参考文档

本文档提供 a-share-factor-engine 的完整 API 参考。

## 核心类

### FactorEngine

因子引擎类，负责因子计算和分析。

#### 方法

##### `__init__()`

初始化因子引擎，自动加载配置的因子计算器。

```python
engine = FactorEngine()
```

##### `compute_a_share_factors(data: pd.DataFrame) -> pd.DataFrame`

计算A股专用Alpha因子。

**参数：**
- `data` (pd.DataFrame): 清洗后的日线数据

**返回：**
- `pd.DataFrame`: 包含因子的DataFrame

**示例：**

```python
factors = engine.compute_a_share_factors(data)
```

##### `neutralize(factor_df, industry_df, neutralize_mcap, neutralize_industry) -> pd.DataFrame`

因子中性化处理。

##### `ic_analysis(factor_df, forward_returns, factor_names) -> Dict`

计算因子IC分析。

##### `correlation_analysis(factor_df, factor_names, max_correlation) -> Dict`

因子相关性分析。

##### `factor_fusion(factor_df, ic_results, selected_factors, fusion_method) -> pd.DataFrame`

多因子融合。

## 标准入口函数

### `run(ctx) -> Dict[str, Any]`

Skill 标准入口函数。

**参数：**
- `ctx` (Context): 上下文对象

**返回：**
```python
{
    "success": bool,
    "artifact_path": str,      # 因子数据文件路径
    "metadata": {
        "factor_names": [...],
        "selected_factors": [...],
        "ic_results": {...}
    },
    "error": str
}
```

**示例：**

```python
from engine import run

result = run(ctx)
if result['success']:
    print(f"因子数据已保存至: {result['artifact_path']}")
```

## 因子列表

| 因子名 | 描述 | 类型 |
|--------|------|------|
| reversal_5d | 5日反转 | 动量 |
| reversal_20d | 20日反转 | 动量 |
| lncap | 对数市值 | 规模 |
| turnover_20d | 20日换手率 | 交易 |
| turnover_change | 换手率变化 | 交易 |
| volatility_20d | 20日波动率 | 风险 |
| volume_ratio | 量比 | 量价 |
| money_flow_20d | 20日资金流 | 资金流 |

## CLI 使用

```bash
python engine.py context.json
```
