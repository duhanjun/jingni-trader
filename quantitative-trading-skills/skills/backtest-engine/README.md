# Backtest Engine

A股量化回测引擎，支持多引擎切换、完整A股规则模拟、绩效分析和可视化。

## 功能特性

- **多引擎支持**: RQAlpha、Backtrader、掘金量化
- **完整A股规则**: T+1、涨跌停、费用模型（佣金万2.5、最低5元、印花税1‰、过户费）
- **停牌处理**: 资产冻结，复牌后恢复
- **绩效指标**: 年化收益、夏普比率、最大回撤、胜率、盈亏比
- **过拟合检测**: Walk-Forward分析
- **回测报告**: Markdown/JSON格式，包含样本外指标
- **可视化**: 收益曲线、回撤曲线（matplotlib/plotly）

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from engine import BacktestEngine
from config import get_config

# 创建引擎
config = get_config()
engine = BacktestEngine(config)

# 定义策略
def my_strategy(context, data):
    if context.current_date.weekday() == 0:
        context.order("000001.SZ", 100)

# 运行回测
results = engine.run(
    strategy=my_strategy,
    start_date="2020-01-01",
    end_date="2024-01-01",
    initial_capital=1000000,
)

# 查看绩效
metrics = results["performance_metrics"]
print(f"总收益率: {metrics['total_return']:.2%}")
print(f"夏普比率: {metrics['sharpe_ratio']:.4f}")
print(f"最大回撤: {metrics['max_drawdown']:.2%}")

# 生成报告
report = engine.generate_report(results)
engine.save_report(report, "my_strategy_report")

# 可视化
engine.plot_equity_curve(results, "reports/equity_curve.png")
engine.plot_drawdown(results, "reports/drawdown.png")
```

## 项目结构

```
backtest-engine/
├── base/                # 基础抽象类
├── adapters/            # 各回测引擎适配器
├── rules/               # A股交易规则
├── performance/         # 绩效指标计算
├── overfitting/         # 过拟合检测
├── report/              # 报告生成
├── visualization/       # 可视化
├── references/          # 参考文档
├── tests/               # 测试用例
├── __init__.py
├── SKILL.md             # Skill 元数据
├── README.md            # 项目说明
├── requirements.txt     # 依赖
├── config.py            # 配置
└── engine.py            # 主引擎
```

## 文档

- [API 参考](./references/api_reference.md)
- [配置指南](./references/config_guide.md)
- [交易规则](./references/trading_rules.md)

## 许可证

MIT
