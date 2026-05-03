# 最佳实践

## 概述

本文档介绍了使用 strategy-model-engine 进行量化策略开发和回测的最佳实践，帮助您避免常见陷阱，提高策略的稳健性和实盘表现。

---

## 1. 过拟合防范

### 使用 Purged Group Time Series Cross-Validation

**为什么重要**：
- 传统的随机划分会导致数据泄露
- 金融数据有时间顺序，未来数据不能用于训练
- 同一行业的股票可能有相关性

**如何使用**：

```python
from engine import StrategyModelEngine

# 创建引擎
engine = StrategyModelEngine()

# 获取交叉验证划分
splits = engine.get_purged_group_splits(
    X=factors,
    groups=industry,
    time_col='date',
    n_splits=5
)

# 在每个折上训练和验证
for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
    X_train, y_train = factors.iloc[train_idx], forward_returns.iloc[train_idx]
    X_val, y_val = factors.iloc[val_idx], forward_returns.iloc[val_idx]
    
    # 训练模型
    model = engine.train_stock_selection_model(X_train, y_train)
    
    # 验证
    val_pred = engine.predict_stock_selection(X_val)
    
    print(f"Fold {fold_idx} 完成")
```

### 参数说明

- **训练窗口 (train_window)**: 36个时间单位（如月）
- **验证窗口 (val_window)**: 12个时间单位
- **测试窗口 (test_window)**: 12个时间单位
- **Purge间隔 (purge_gap)**: 2个时间单位，避免训练和验证数据重叠
- **行业分组**: 确保同一行业股票不会同时出现在训练和验证集中

---

## 2. 实验管理

### 使用 MLflow 记录实验

**为什么重要**：
- 便于实验对比和复现
- 自动记录参数、指标和模型
- 可视化实验结果

**如何使用**：

```python
from engine import StrategyModelEngine

engine = StrategyModelEngine()

# 使用上下文管理器
with engine.start_experiment(run_name='lgbm_v1', tags={'model': 'lightgbm', 'version': '1.0'}):
    # 记录参数
    engine.log_params({
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6
    })
    
    # 训练模型
    model = engine.train_stock_selection_model(factors, forward_returns)
    
    # 记录模型
    engine.log_model(model, 'stock_selection_model')
    
    # 记录指标
    engine.log_metrics({
        'train_rmse': 0.05,
        'val_rmse': 0.06,
        'ic': 0.03,
        'icir': 1.5
    })
    
    # 记录因子组合
    engine.log_factor_combination(['momentum', 'value', 'quality'])
```

### 查看实验结果

```python
# 搜索实验
runs = engine.experiment_manager.search_runs(
    order_by=['metrics.ic DESC'],
    max_results=10
)

print("最佳实验:")
print(runs[['run_id', 'metrics.ic', 'metrics.icir', 'params.n_estimators']])

# 加载最佳模型
best_run_id = runs.iloc[0]['run_id']
best_model = engine.experiment_manager.load_model(best_run_id, 'stock_selection_model')
```

---

## 3. 因子开发和选择

### 因子预处理

**为什么重要**：
- 原始因子可能有异常值
- 不同因子量级不同
- 行业效应需要中性化

**最佳实践**：

```python
# 1. 处理异常值（缩尾）
def winsorize(series, lower=0.01, upper=0.99):
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower_bound, upper_bound)

# 2. 标准化（Z-score）
def standardize(series):
    return (series - series.mean()) / series.std()

# 3. 行业中性化
def neutralize_industry(factor_df, industry_series):
    neutralized = factor_df.copy()
    for col in factor_df.columns:
        industry_mean = factor_df.groupby(industry_series)[col].transform('mean')
        neutralized[col] = factor_df[col] - industry_mean
    return neutralized
```

### 因子选择方法

1. **IC分析**：选择IC均值高、ICIR稳定的因子
2. **相关性分析**：避免使用高度相关的因子
3. **特征重要性**：使用模型输出的特征重要性
4. **逐步回归**：逐步添加/删除因子

---

## 4. 策略开发流程

### 标准工作流

```
1. 数据获取和清洗
   ↓
2. 因子开发和预处理
   ↓
3. 因子分析和选择
   ↓
4. 模型训练（使用交叉验证）
   ↓
5. 超参数优化
   ↓
6. 策略回测
   ↓
7. 过拟合检验（样本外测试）
   ↓
8. 实盘模拟
```

### 详细步骤

```python
from engine import StrategyModelEngine

# 步骤1: 创建引擎
engine = StrategyModelEngine()

# 步骤2: 数据准备（自行实现）
# factors, forward_returns, industry = prepare_data()

# 步骤3: 因子分析（使用 a-share-factor-engine）

# 步骤4: 交叉验证训练
splits = engine.get_purged_group_splits(factors, industry, time_col='date')

# 步骤5: 超参数优化
best_params = engine.optimize_hyperparameters(
    factors, forward_returns, n_trials=50
)

# 步骤6: 完整工作流（包含MLflow记录）
results = engine.full_workflow(
    factors, forward_returns,
    industry=industry,
    run_name='final_model',
    optimize=False,
    use_cv=True
)
```

---

## 5. 回测注意事项

### 避免前瞻偏差

❌ 错误示例：
```python
# 使用未来数据计算因子
for i in range(len(data)):
    future_data = data[i:i+30]  # 未来数据！
    factor[i] = compute_factor(future_data)
```

✅ 正确示例：
```python
# 只使用历史数据
for i in range(30, len(data)):
    past_data = data[i-30:i]  # 只使用历史数据
    factor[i] = compute_factor(past_data)
```

### 考虑交易成本

- 手续费：通常为成交额的0.03%-0.1%
- 滑点：考虑流动性成本
- 冲击成本：大额交易对价格的影响

### 样本外测试

- 预留最后1-2年的数据作为样本外
- 不要在样本外数据上进行优化
- 样本外表现是策略真实质量的更好指标

---

## 6. 风险管理

### 仓位管理

```python
def calculate_position(predictions, max_position=0.1, top_n=50):
    """
    根据预测结果计算仓位
    """
    # 选择预测收益最高的股票
    selected = predictions.nlargest(top_n)
    
    # 等权分配
    position_size = max_position / top_n
    
    positions = pd.Series(position_size, index=selected.index)
    return positions
```

### 止损和止盈

```python
def check_stop_loss_take_profit(entry_price, current_price, stop_loss=0.02, take_profit=0.05):
    """
    检查是否触发止损或止盈
    """
    return_pct = (current_price - entry_price) / entry_price
    
    if return_pct <= -stop_loss:
        return 'stop_loss'
    elif return_pct >= take_profit:
        return 'take_profit'
    else:
        return 'hold'
```

---

## 7. 性能优化

### 数据处理

- 使用 parquet 格式存储数据
- 向量化操作代替循环
- 适当使用 dask 处理大数据

### 模型训练

- 先在小数据集上调试
- 使用 Optuna 进行高效超参数搜索
- 利用并行计算（LightGBM 支持）

### 实验管理

- 使用 MLflow 跟踪所有实验
- 定期清理无用的实验记录
- 使用 artifacts 存储中间结果

---

## 8. 常见陷阱

### ❌ 陷阱1：过度优化

**表现**：
- 样本内表现极好
- 样本外表现很差
- 参数对历史数据过度拟合

**避免方法**：
- 使用严格的交叉验证
- 限制优化的参数数量
- 使用简单模型作为基准

### ❌ 陷阱2：忽略交易成本

**表现**：
- 回测收益很高
- 换手率极高
- 实盘收益远低于回测

**避免方法**：
- 回测时加入真实交易成本
- 控制换手率
- 优先考虑低频策略

### ❌ 陷阱3：幸存者偏差

**表现**：
- 只使用当前仍在上市的股票
- 忽略退市股票
- 高估策略表现

**避免方法**：
- 使用包含退市股票的数据库
- 构建 survivorship-bias-free 的回测框架

### ❌ 陷阱4：前视偏差

**表现**：
- 使用未来数据训练
- 因子计算包含未来信息
- 回测结果不可重复

**避免方法**：
- 严格区分训练、验证、测试数据
- 使用时间戳验证
- 代码审查时重点检查

---

## 总结

1. **使用 Purged Group Time Series CV**：防止过拟合
2. **记录所有实验**：使用 MLflow 便于复现和对比
3. **仔细处理因子**：缩尾、标准化、行业中性化
4. **严格验证**：样本内外测试，考虑交易成本
5. **控制风险**：合理的仓位管理、止损止盈
6. **避免陷阱**：过度优化、前视偏差、幸存者偏差

遵循这些最佳实践将大大提高您的量化策略的成功率！
