---
name: backtest-engine
version: 1.1.0
description: A股策略回测引擎。原生 pandas/numpy 实现，内置 T+1 交割、涨跌停板、停牌处理、A股真实费用模型（佣金/印花税/过户费双侧），自动计算年化收益、夏普比率、最大回撤、胜率等关键指标，生成回测报告和可视化图表。支持基准对比（alpha/beta/excess_return）与成本分离（gross/net return）。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - backtest-engine
  - T+1
  - 涨跌停
dependencies:
  - quantstats
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - plotly>=5.15.0
environment_variables:
  - name: QUANT_WORK_DIR
    description: 工作目录根路径
    required: false
    default: "./workspace"
  - name: BENCHMARK
    description: 基准指数
    required: false
    default: "000300.SH"
  - name: INIT_CAPITAL
    description: 初始资金
    required: false
    default: "1000000.0"
language: python
python_version: "3.9+"
entry_point: engine.py
trigger_keywords:
  - 回测
  - 回测引擎
  - T+1
  - 涨跌停
  - 绩效
  - 夏普比率
  - 最大回撤
---

# backtest-engine

## 概述

backtest-engine 是 A 股量化投研的**策略回测引擎**，提供：

1. **原生回测实现**：纯 pandas/numpy，无外部回测框架依赖
2. **A股规则模拟**：T+1、涨跌停、停牌、费用（佣金/印花税/过户费双侧）
3. **完整绩效指标**：年化收益、夏普、最大回撤、胜率、Calmar、Sortino 等
4. **基准对比**：alpha/beta/excess_return/benchmark_max_drawdown
5. **成本分离**：gross_return（不含费用）vs net_return（含费用），total_cost_drag
6. **可视化报告**：HTML/JSON 双格式输出

## A股特殊规则

- **T+1**：当日买入次日才能卖出（通过 Position.last_buy_date 严格校验）
- **涨跌停**：涨停无法买入，跌停无法卖出
- **停牌**：资产冻结，复牌后以开盘价恢复
- **费用**：佣金万2.5（最低5元）、印花税1‰（单边卖出）、过户费0.02‰（双侧）
- **滑点**：买入价 ×(1+slippage)，卖出价 ×(1-slippage)（双侧应用）
- **最小交易单位**：100股

## 使用示例

### Python API

```python
from engine import run
from context import Context

ctx = Context(
    task_id="task_001",
    user_intent="回测策略",
    current_stage="IDLE"
)
ctx.stock_pool = ["000001.SZ"]
ctx.start_date = "2021-01-01"
ctx.end_date = "2024-01-01"

result = run(ctx)
```

### CLI 运行

```bash
python engine.py -i "回测我的策略"
```

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)
