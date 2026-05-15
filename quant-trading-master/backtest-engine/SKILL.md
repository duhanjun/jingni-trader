---
name: backtest-engine
description: >
  A股策略回测引擎。支持 RQAlpha、Backtrader、掘金(gm) 三种回测引擎，
  内置 T+1 交割、涨跌停板、停牌处理、A股真实费用模型（佣金/印花税/过户费），
  自动计算年化收益、夏普比率、最大回撤、胜率等关键指标，生成回测报告和
  可视化图表。支持 Walk-Forward 分析检测过拟合。
trigger_keywords:
  - 回测
  - 回测引擎
  - T+1
  - 涨跌停
  - 绩效
  - 夏普比率
  - 最大回撤
  - Walk-Forward
version: 1.0.0
author: quant-team
dependencies:
  - rqalpha
  - backtrader
  - gm (掘金)
  - quantstats
  - pandas
  - numpy
  - matplotlib
  - plotly
backends:
  - rqalpha
  - backtrader
  - gm
---
# backtest-engine

## 职责

- 加载策略信号和数据，在指定回测引擎中模拟交易
- 严格遵循 A股规则：T+1、涨跌停、停牌、费用
- 输出完整交易记录和绩效指标
- 生成回测报告 HTML/JSON
- 支持多后端切换

## A股特殊规则

- T+1：当日买入次日才能卖出
- 涨跌停：涨停无法买入，跌停无法卖出
- 停牌：资产冻结，复牌后以开盘价恢复
- 费用：佣金万2.5（最低5元）、印花税1‰（单边卖出）、过户费0.02‰
- 最小交易单位：100股

## 使用方式

由 `quant-trading-master` 调度，调用 `run(ctx)` 函数。
