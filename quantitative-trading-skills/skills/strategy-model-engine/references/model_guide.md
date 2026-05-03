# 模型使用指南

## 概述

strategy-model-engine 提供了两种核心模型：截面选股模型和择时模型，分别用于股票选择和交易时机判断。

---

## 1. 截面选股模型

### 概述

截面选股模型使用机器学习算法（LightGBM 或 CatBoost）根据多个因子预测股票的未来收益，从而选出预期表现好的股票。

### 支持的模型

#### LightGBM
- 基于梯度提升决策树
- 训练速度快，内存占用小
- 对异常值鲁棒
- 支持并行计算

#### CatBoost
- 自动处理类别特征
- 减少超参数调优工作
- 对噪声数据鲁棒
- 防止过拟合

### 输入数据

#### 特征数据 (X)
- DataFrame，包含多个因子列
- 每一行代表一个股票在某个时间点的因子值
- 列名：因子名称

#### 目标数据 (y)
- Series，包含未来收益
- 长度应与 X 相同
- 通常使用未来 N 天的收益率

#### 行业标签 (industry) [可选]
- Series，包含行业分类
- 用于行业中性化，消除行业风险暴露

### 使用示例

```python
from engine import StrategyModelEngine
import pandas as pd
import numpy as np

# 创建引擎
engine = StrategyModelEngine()

# 准备数据（示例）
np.random.seed(42)
n_stocks = 100
n_days = 252

# 因子数据
factors = pd.DataFrame({
    'momentum': np.random.randn(n_stocks * n_days),
    'value': np.random.randn(n_stocks * n_days),
    'quality': np.random.randn(n_stocks * n_days),
    'volatility': np.random.randn(n_stocks * n_days),
    'liquidity': np.random.randn(n_stocks * n_days)
})

# 未来收益
forward_returns = pd.Series(np.random.randn(n_stocks * n_days))

# 行业标签
industry = pd.Series(np.random.choice(['tech', 'finance', 'consumer', 'healthcare'], n_stocks * n_days))

# 训练 LightGBM 模型
model = engine.train_stock_selection_model(
    X=factors,
    y=forward_returns,
    model_type='lightgbm',
    industry=industry
)

# 预测
predictions = engine.predict_stock_selection(factors, industry)

# 获取特征重要性
feature_importance = engine.stock_selector.get_feature_importance()
print("特征重要性:")
print(feature_importance)
```

### 自定义参数

```python
# 自定义 LightGBM 参数
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8
}

model = engine.train_stock_selection_model(
    X=factors,
    y=forward_returns,
    model_type='lightgbm',
    params=lgb_params
)
```

---

## 2. 择时模型

### 概述

择时模型基于分钟线数据和技术指标，预测未来3天的涨跌方向，用于判断买入和卖出时机。

### 输入数据

#### 分钟线数据
- DataFrame，包含以下列：
  - open: 开盘价
  - high: 最高价
  - low: 最低价
  - close: 收盘价
  - volume: 成交量

### 自动计算的技术指标

择时模型会自动计算以下技术指标作为特征：

**移动平均线**
- MA5, MA10, MA20, MA60

**动量指标**
- RSI (14日)
- MACD, MACD Signal, MACD Histogram
- 动量1日、5日、10日、20日

**波动率指标**
- 20日滚动波动率

**成交量指标**
- 成交量MA5, MA20
- 成交量比率

**布林带**
- BB Mid, BB Upper, BB Lower

### 使用示例

```python
from engine import StrategyModelEngine
import pandas as pd
import numpy as np

# 创建引擎
engine = StrategyModelEngine()

# 准备分钟线数据（示例）
np.random.seed(42)
n_minutes = 1000

# 生成模拟价格数据
base_price = 100
prices = base_price + np.cumsum(np.random.randn(n_minutes) * 0.1)

minute_data = pd.DataFrame({
    'open': prices + np.random.randn(n_minutes) * 0.05,
    'high': prices + np.random.rand(n_minutes) * 0.1,
    'low': prices - np.random.rand(n_minutes) * 0.1,
    'close': prices,
    'volume': np.random.randint(1000, 10000, n_minutes)
})

# 训练择时模型
model = engine.train_timing_model(minute_data)

# 预测
predictions = engine.predict_timing(minute_data)

# 预测概率（上涨/下跌的概率）
X, _ = engine.timing_model.prepare_features_and_target(minute_data)
probabilities = engine.timing_model.predict_proba(X)

print("预测概率:")
print(probabilities.head())
```

### 评估模型

```python
# 评估模型性能
X, y = engine.timing_model.prepare_features_and_target(minute_data)
metrics = engine.timing_model.evaluate(X, y)

print(f"准确率: {metrics['accuracy']:.4f}")
print(f"F1分数: {metrics['f1_score']:.4f}")
```

---

## 3. 超参数优化

### 概述

使用 Optuna 进行超参数优化，自动搜索最佳参数组合。

### 使用示例

```python
from engine import StrategyModelEngine

# 创建引擎
engine = StrategyModelEngine()

# 优化超参数（回归任务）
best_params = engine.optimize_hyperparameters(
    X=factors,
    y=forward_returns,
    model_type='lightgbm',
    task_type='regression',
    n_trials=50,
    val_size=0.2,
    direction='minimize'
)

print("最佳参数:")
print(best_params)

# 使用最佳参数训练模型
model = engine.train_stock_selection_model(
    X=factors,
    y=forward_returns,
    model_type='lightgbm',
    params=best_params
)

# 查看优化历史
study_summary = engine.optimizer.get_study_summary()
print(f"最佳值: {study_summary['best_value']:.4f}")
```

### 搜索的超参数

#### LightGBM
- n_estimators: 树的数量 (50-300)
- learning_rate: 学习率 (0.01-0.3，对数搜索)
- max_depth: 树的最大深度 (3-12)
- num_leaves: 叶子节点数 (20-100)
- subsample: 样本采样率 (0.5-1.0)
- colsample_bytree: 特征采样率 (0.5-1.0)
- reg_alpha: L1正则化系数 (0-10)
- reg_lambda: L2正则化系数 (0-10)

#### CatBoost
- iterations: 迭代次数 (50-300)
- learning_rate: 学习率 (0.01-0.3，对数搜索)
- max_depth: 树的最大深度 (3-10)
- subsample: 样本采样率 (0.5-1.0)
- colsample_bylevel: 层级特征采样率 (0.5-1.0)
- l2_leaf_reg: L2正则化系数 (0-10)

---

## 4. 完整工作流

### 示例

```python
from engine import StrategyModelEngine

# 创建引擎
engine = StrategyModelEngine()

# 运行完整工作流
results = engine.full_workflow(
    X=factors,
    y=forward_returns,
    model_type='lightgbm',
    industry=industry,
    factors=['momentum', 'value', 'quality', 'volatility', 'liquidity'],
    run_name='my_first_experiment',
    optimize=True,
    use_cv=True
)

# 获取结果
model = results['model']
feature_importance = results['feature_importance']
predictions = results['predictions']

print("完整工作流完成！")
```
