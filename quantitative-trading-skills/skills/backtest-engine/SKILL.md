---
name: backtest-engine
version: 1.0.0
description: A股量化回测引擎，支持多引擎切换、完整A股规则模拟、绩效分析和可视化
author: OpenClaw Team
license: MIT
tags:
  - quant-trading
  - A股
  - backtest
  - rqalpha
  - backtrader
  - 掘金
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - plotly>=5.14.0
  - quantstats>=0.0.59
  - empyrical>=0.5.5
  - rqalpha>=4.0.0
  - backtrader>=1.9.78
environment_variables:
  - name: GM_TOKEN
    description: 掘金量化API Token
    required: false
  - name: BACKTEST_ENGINE
    description: 默认回测引擎 (rqalpha/backtrader/gm)
    required: false
    default: "rqalpha"
  - name: REPORT_DIR
    description: 回测报告存储目录
    required: false
    default: "./reports"
language: python
python_version: "3.9+"
entry_point: engine.py
---

# backtest-engine Skill

## 概述

backtest-engine 是 A 股量化投研的统一回测引擎，提供：

1. **多引擎支持**：RQAlpha、Backtrader、掘金量化
2. **完整A股规则**：T+1、涨跌停、费用模型（佣金万2.5、最低5元、印花税1‰、过户费）
3. **停牌处理**：资产冻结，复牌后恢复
4. **绩效指标**：年化收益、夏普比率、最大回撤、胜率、盈亏比（quantstats/empyrical）
5. **过拟合检测**：Walk-Forward分析
6. **回测报告**：Markdown/JSON格式，包含样本外指标
7. **可视化**：收益曲线、回撤曲线（matplotlib/plotly）

## 使用示例

```python
from engine import BacktestEngine
from config import get_config
from rules import AShareTradingRules
from performance import PerformanceMetrics

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
    initial_capital=1000000
)

# 生成报告
report = engine.generate_report(results)
engine.save_report(report, "my_strategy_report")

# 可视化
engine.plot_equity_curve(results)
engine.plot_drawdown(results)
```

## 回测引擎

| 引擎 | 说明 | 需要Token |
|------|------|-----------|
| RQAlphaAdapter | RQAlpha 回测引擎 | 否 |
| BacktraderAdapter | Backtrader 回测引擎 | 否 |
| gmAdapter | 掘金量化回测引擎 | 是 |

## A股交易规则

- **T+1**：当日买入次日可卖
- **涨跌停**：±10%，ST股±5%，支持严格废单/排队模型可选
- **费用模型**：
  - 佣金：万分之2.5，最低5元
  - 印花税：千分之一（卖出时收取）
  - 过户费：按成交金额的十万分之二收取

## 绩效指标

| 指标 | 说明 |
|------|------|
| total_return | 总收益率 |
| annual_return | 年化收益率 |
| sharpe_ratio | 夏普比率 |
| max_drawdown | 最大回撤 |
| win_rate | 胜率 |
| profit_loss_ratio | 盈亏比 |

## 配置说明

详见 references/config_guide.md

## API 文档

详见 references/api_reference.md
