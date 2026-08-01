---
name: execution-monitor-engine
version: 1.0.0
description: A股实盘执行与监控引擎。支持模拟交易(paper)与实盘交易(live)两种模式，可对接 xtquant(miniQMT)、掘金(gm) 等券商接口。内置硬风控断路器（单日亏损限制、单笔金额上限、持仓集中度、订单频率限制），支持账户查询、订单发送与撤单、仓位同步，所有交易操作完整记录到审计日志。默认使用 paper 模拟交易模式（生产可用）。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - execution-engine
  - 实盘交易
  - 风控
  - xtquant
  - paper
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - sqlalchemy>=2.0.0
  - xtquant (可选)
  - gm (可选)
environment_variables:
  - name: TRADE_MODE
    description: 交易模式（paper / live）
    required: false
    default: "paper"
  - name: TRADE_BACKEND
    description: 交易接口后端（xtquant / gm）
    required: false
    default: "xtquant"
  - name: QUANT_WORK_DIR
    description: 工作目录根路径
    required: false
    default: "./workspace"
  - name: XTQUANT_PATH
    description: miniQMT userdata_mini 路径（live+xtquant 模式必需）
    required: false
  - name: XTQUANT_ACCOUNT
    description: miniQMT 资金账号（live+xtquant 模式必需）
    required: false
  - name: GM_TOKEN
    description: 掘金量化 API Token（live+gm 模式必需）
    required: false
  - name: GM_ACCOUNT_ID
    description: 掘金账户 ID（live+gm 模式必需，终端获取）
    required: false
  - name: INIT_CAPITAL
    description: 初始资金（paper 模式）
    required: false
    default: "1000000"
  - name: MAX_DAILY_LOSS_RATIO
    description: 单日最大亏损比例
    required: false
    default: "0.02"
  - name: MAX_SINGLE_ORDER_RATIO
    description: 单笔订单最大金额比例
    required: false
    default: "0.10"
  - name: MAX_ORDER_FREQUENCY
    description: 最大下单频率（笔/秒）
    required: false
    default: "2"
  - name: SLIPPAGE
    description: 滑点模拟比例（paper 模式）
    required: false
    default: "0.001"
  - name: COMMISSION_RATE
    description: 佣金费率
    required: false
    default: "0.00025"
  - name: STAMP_TAX_RATE
    description: 印花税率（卖出）
    required: false
    default: "0.001"
language: python
python_version: "3.9+"
entry_point: engine.py
trigger_keywords:
  - 实盘
  - 下单
  - 交易
  - 执行
  - 监控
  - 模拟
  - 风控
  - 断路器
  - 订单
  - 撤单
  - 量化断路器
---

# execution-monitor-engine

## 概述

execution-monitor-engine 是 A 股量化投研的**实盘执行与监控引擎**，提供：

1. **双模式支持**：模拟交易（Paper）和实盘交易（Live）
2. **多券商对接**：xtquant、掘金量化
3. **硬风控断路器**：独立于策略的风险检查层
4. **账户管理**：查询资产、持仓、可用资金
5. **订单操作**：发送（市价/限价）、撤单
6. **审计日志**：完整的 JSONL 日志记录

## 执行模式

| 模式 | 状态 | 说明 |
|------|------|------|
| `paper` | 生产可用 | 模拟交易，本地虚拟账户（滑点/T+1/数量校验/资金校验/断路器） |
| `live`（xtquant） | 生产可用 | 迅投 miniQMT 实盘，需本地客户端 + XTQUANT_PATH/XTQUANT_ACCOUNT |
| `live`（gm） | 生产可用 | 掘金量化实盘，需 GM_TOKEN/GM_ACCOUNT_ID |

> 三种模式均已通过实盘连通性验证（连接/查询/下单/撤单全流程）。

## 硬风控断路器

- **单日亏损限制**：累计亏损超过净值 3% → 拒绝新开仓
- **单笔金额上限**：不超过净资产 2%
- **持仓集中度**：单票上限 10%
- **订单频率**：每分钟最多 5 笔

## 量化断路器

`optimizations/quant_circuit_breaker.py` 提供增强版量化断路器：

- 支持多维度风控规则组合
- 可配置的触发阈值和恢复条件
- 与审计日志联动

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