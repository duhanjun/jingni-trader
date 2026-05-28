---
name: execution-monitor-engine
version: 1.0.0
description: A股实盘执行与监控引擎。支持模拟交易(paper)与实盘交易(live)两种模式，可对接 xtquant(miniQMT)、掘金(gm) 等券商接口。内置硬风控断路器（单日亏损限制、单笔金额上限、持仓集中度、订单频率限制），支持账户查询、订单发送与撤单、仓位同步，所有交易操作完整记录到审计日志。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - execution-engine
  - 实盘交易
  - 风控
  - xtquant
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - sqlalchemy>=2.0.0
  - xtquant (可选)
  - gm (可选)
environment_variables:
  - name: TRADE_MODE
    description: 交易模式
    required: false
    default: "paper"
  - name: TRADE_BACKEND
    description: 交易接口后端
    required: false
    default: "xtquant"
  - name: EXECUTION_DIR
    description: 执行日志目录
    required: false
    default: "./quant_workspace/execution"
language: python
python_version: "3.9+"
entry_point: engine.py
backends:
  - paper
  - xtquant
  - gm
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
---

# execution-monitor-engine

## 概述

execution-monitor-engine 是 A 股量化投研的**实盘执行与监控引擎**，提供：

1. **双模式支持**：模拟交易(Paper)和实盘交易(Live)
2. **多券商对接**：xtquant、掘金量化
3. **硬风控断路器**：独立于策略的风险检查层
4. **账户管理**：查询资产、持仓、可用资金
5. **订单操作**：发送（市价/限价）、撤单
6. **审计日志**：完整的 JSONL 日志记录

## 硬风控断路器

- **单日亏损限制**：累计亏损超过净值2% → 拒绝新开仓
- **单笔金额上限**：不超过净资产10%
- **持仓集中度**：单票上限10%
- **订单频率**：每秒最多2笔

## 支持模式

- `paper`: 模拟交易，本地虚拟账户
- `live`: 实盘交易，连接券商

## 使用示例

### Python API

```python
from engine import run
from context import Context

ctx = Context(
    task_id="task_001",
    user_intent="执行交易",
    current_stage="IDLE"
)

result = run(ctx)
```

### CLI 运行

```bash
python engine.py -i "执行目标组合"
```

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)
