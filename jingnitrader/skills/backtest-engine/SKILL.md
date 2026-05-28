---
name: backtest-engine
version: 1.0.0
description: A股策略回测引擎。支持 RQAlpha、Backtrader、掘金(gm) 三种回测引擎，内置 T+1 交割、涨跌停板、停牌处理、A股真实费用模型（佣金/印花税/过户费），自动计算年化收益、夏普比率、最大回撤、胜率等关键指标，生成回测报告和可视化图表。支持 Walk-Forward 分析检测过拟合。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - backtest-engine
  - T+1
  - rqalpha
  - backtrader
dependencies:
  - rqalpha
  - backtrader
  - gm (掘金)
  - quantstats
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - plotly>=5.15.0
environment_variables:
  - name: BACKTEST_BACKEND
    description: 回测引擎后端
    required: false
    default: "rqalpha"
  - name: BACKTEST_DIR
    description: 回测结果存储目录
    required: false
    default: "./quant_workspace/backtest_results"
  - name: BENCHMARK
    description: 基准指数
    required: false
    default: "000300.SH"
language: python
python_version: "3.9+"
entry_point: engine.py
backends:
  - rqalpha
  - backtrader
  - gm
trigger_keywords:
  - 回测
  - 回测引擎
  - T+1
  - 涨跌停
  - 绩效
  - 夏普比率
  - 最大回撤
  - Walk-Forward
---

# backtest-engine

## 概述

backtest-engine 是 A 股量化投研的**策略回测引擎**，提供：

1. **多回测引擎支持**：RQAlpha、Backtrader、掘金量化
2. **A股规则模拟**：T+1、涨跌停、停牌、费用
3. **完整绩效指标**：年化收益、夏普、最大回撤、胜率
4. **可视化报告**：HTML/JSON 双格式输出
5. **Walk-Forward 分析**：过拟合检测

## A股特殊规则

- **T+1**：当日买入次日才能卖出
- **涨跌停**：涨停无法买入，跌停无法卖出
- **停牌**：资产冻结，复牌后以开盘价恢复
- **费用**：佣金万2.5（最低5元）、印花税1‰（单边卖出）、过户费0.02‰
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
