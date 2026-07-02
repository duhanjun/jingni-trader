# API 参考文档

## BacktestEngine

主回测引擎类，提供完整的回测功能。

### 初始化

```python
from engine import BacktestEngine
from config import get_config

config = get_config()
engine = BacktestEngine(config)
```

### 方法

#### `run(strategy, start_date, end_date, initial_capital=1000000, **kwargs)`

运行回测。

**参数:**
- `strategy`: 策略函数
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)
- `initial_capital`: 初始资金 (默认 1,000,000)

**返回:**
- 回测结果字典，包含:
  - `trades_df`: 交易记录 DataFrame
  - `performance_metrics`: 绩效指标
  - `equity_curve`: 资金曲线
  - `benchmark_curve`: 基准曲线

#### `get_trades()`

获取交易记录。

**返回:**
- 交易记录 DataFrame

#### `get_equity_curve()`

获取资金曲线。

**返回:**
- 资金曲线 Series

#### `get_performance_metrics()`

获取绩效指标。

**返回:**
- 绩效指标字典

#### `generate_report(backtest_results, walk_forward_results=None)`

生成回测报告。

**参数:**
- `backtest_results`: 回测结果
- `walk_forward_results`: Walk-Forward 分析结果 (可选)

**返回:**
- 报告字典

#### `save_report(report, filename, output_dir=None, formats=["markdown", "json"])`

保存报告。

**参数:**
- `report`: 报告
- `filename`: 文件名
- `output_dir`: 输出目录 (可选)
- `formats`: 输出格式列表 (可选)

**返回:**
- 文件路径字典

#### `plot_equity_curve(results, save_path=None)`

绘制资金曲线。

**参数:**
- `results`: 回测结果
- `save_path`: 保存路径 (可选)

#### `plot_drawdown(results, save_path=None)`

绘制回撤曲线。

**参数:**
- `results`: 回测结果
- `save_path`: 保存路径 (可选)

#### `analyze_walk_forward(strategy, data, start_date, end_date, initial_capital=1000000)`

执行 Walk-Forward 分析。

**参数:**
- `strategy`: 策略函数
- `data`: 行情数据
- `start_date`: 开始日期
- `end_date`: 结束日期
- `initial_capital`: 初始资金

**返回:**
- Walk-Forward 分析结果

## AShareTradingRules

A股交易规则模拟类。

### 方法

#### `check_limit_up_down(code, price, pre_close, is_st=False)`

检查涨跌停限制。

#### `check_t1(code, sell_date)`

检查 T+1 规则。

#### `update_holding(code, quantity, date)`

更新持仓。

#### `get_holding_quantity(code)`

获取持仓数量。

#### `calculate_fee(amount, direction)`

计算交易费用。

## PerformanceMetrics

绩效指标计算类。

### 方法

#### `calculate(equity_curve, trades_df, config)`

计算绩效指标。

**返回的指标:**
- `total_return`: 总收益
- `annual_return`: 年化收益
- `volatility`: 波动率
- `sharpe_ratio`: 夏普比率
- `max_drawdown`: 最大回撤
- `calmar_ratio`: 卡玛比率
- `win_rate`: 胜率
- `profit_loss_ratio`: 盈亏比
- `total_trades`: 总交易次数
- `initial_value`: 初始价值
- `final_value`: 最终价值

## WalkForwardAnalyzer

Walk-Forward 过拟合检测类。

### 方法

#### `analyze(strategy, data, start_date, end_date, initial_capital=1000000)`

执行 Walk-Forward 分析。

**返回:**
- `total_periods`: 总周期数
- `avg_train_return`: 平均训练收益
- `avg_test_return`: 平均测试收益
- `performance_degradation`: 性能衰减
- `is_overfitted`: 是否过拟合
- `periods`: 各周期详细结果
