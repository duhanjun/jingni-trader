---
name: execution-monitor-engine
description: >
  A股实盘执行与监控引擎。支持模拟交易(paper)与实盘交易(live)两种模式，
  可对接 xtquant(miniQMT)、掘金(gm) 等券商接口。
  内置硬风控断路器（单日亏损限制、单笔金额上限、持仓集中度、订单频率限制），
  支持账户查询、订单发送与撤单、仓位同步，所有交易操作完整记录到审计日志。
trigger_keywords:
  - 实盘
  - 下单
  - 交易
  - 执行
  - 监控
  - 模拟
  - 风控
  - 断电器
  - 订单
  - 撤单
version: 1.0.0
author: quant-team
dependencies:
  - pandas
  - numpy
  - sqlalchemy
  - xtquant (可选)
  - gm (可选)
backends:
  - paper
  - xtquant
  - gm
---

# execution-monitor-engine

## 职责

- 根据目标组合权重生成实际买卖订单
- 支持模拟交易和实盘交易双模式
- 对接券商/交易终端接口
- 硬风控断路器独立于策略的检查层
- 账户管理：查询资产、可用资金、持仓
- 订单操作：发送（市价/限价）、撤单
- 审计日志：所有订单记录完整 JSONL 日志

## 支持模式

- `paper`: 模拟交易，本地虚拟账户
- `live`: 实盘交易，连接券商

## 硬风控断路器

- 单日累计亏损超过净值2% → 拒绝所有新开仓
- 单笔订单金额不超过净资产10%
- 持仓集中度限制（单票上限10%）
- 每秒订单频率限制（≤2笔）
- 交易日开始时自动重置每日损益

## 使用方式

由 `quant-trading-master` 调度，调用 `run(ctx)` 函数。