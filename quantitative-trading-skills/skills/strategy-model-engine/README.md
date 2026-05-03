# strategy-model-engine

量化策略模型引擎，提供策略模板生成、截面选股、择时模型、超参数优化、实验管理和过拟合防范等功能。

## 功能特性

- **策略模板生成**：趋势跟踪、均值回归、配对交易
- **截面选股模型**：LightGBM、CatBoost 多因子选股
- **择时模型**：基于分钟线技术指标的涨跌分类器
- **超参数优化**：Optuna 自动搜索最佳参数
- **实验管理**：MLflow 记录实验参数、指标和模型
- **过拟合防范**：Purged Group Time Series Split

## 快速开始

```python
from engine import StrategyModelEngine

# 创建引擎
engine = StrategyModelEngine()

# 生成策略模板
trend_template = engine.generate_strategy_template('trend_following')

# 训练截面选股模型
# model = engine.train_stock_selection_model(factors, forward_returns)

# 训练择时模型
# timing_model = engine.train_timing_model(minute_data)
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 文档

- [策略模板说明](references/strategy_templates.md)
- [模型使用指南](references/model_guide.md)
- [最佳实践](references/best_practices.md)

## 测试

```bash
cd tests
python test_engine.py
```
