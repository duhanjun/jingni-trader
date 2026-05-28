# API 参考文档

本文档提供 strategy-model-engine 的完整 API 参考。

## 核心类

### ModelEngine

策略模型引擎类。

#### 方法

##### `__init__()`

初始化模型引擎。

```python
engine = ModelEngine()
```

##### `prepare_data(factor_df, price_df, feature_cols, label_col) -> Tuple[pd.DataFrame, pd.Series, pd.Series]`

准备训练数据。

**返回：**
```python
(X, y, dates)
```

##### `purged_group_ts_split(dates, n_splits) -> List[Tuple]`

Purged Group Time Series Split 交叉验证。

##### `create_model(trial) -> model`

创建模型实例。

##### `optimize_hyperparams(X, y, dates, n_trials) -> Dict`

超参数优化。

##### `train(X, y, best_params, test_dates) -> Tuple[model, Dict, np.ndarray]`

训练模型并评估。

**返回：**
```python
(model, metrics, predictions)
```

##### `generate_rule_based_signal(factor_df, strategy_type) -> pd.DataFrame`

生成规则型策略信号。

## 标准入口函数

### `run(ctx) -> Dict[str, Any]`

Skill 标准入口函数。

**参数：**
- `ctx` (Context): 上下文对象

**返回：**
```python
{
    "success": bool,
    "artifact_path": str,        # 模型或信号文件路径
    "predictions_path": str,     # 预测信号路径
    "metadata": {
        "model_type": str,
        "metrics": {...},
        "feature_cols": [...],
    },
    "error": str
}
```

**示例：**

```python
from engine import run

result = run(ctx)
if result['success']:
    print(f"模型已保存: {result['artifact_path']}")
    print(f"绩效指标: {result['metadata']['metrics']}")
```

## CLI 使用

```bash
python engine.py context.json
```
