---
name: execution-monitor-engine
version: 1.0.0
description: 交易执行监控引擎，支持多后端切换、模拟账户、硬断路器、审计日志和仓位管理
author: OpenClaw Team
license: MIT
tags:
  - quant-trading
  - execution
  - trading
  - xtquant
  - gm
  - vnpy
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - pydantic>=2.0.0
  - python-dotenv>=1.0.0
environment_variables:
  - name: PAPER_TRADE
    description: 是否启用模拟交易模式
    required: false
    default: "true"
  - name: EXECUTION_BACKEND
    description: 执行后端：xtquant/gm/vnpy/sim
    required: false
    default: "sim"
  - name: INITIAL_CAPITAL
    description: 初始资金
    required: false
    default: "1000000"
language: python
python_version: "3.9+"
entry_point: engine.py
---

# execution-monitor-engine Skill

## 概述

execution-monitor-engine 是量化交易的执行监控引擎，提供：

1. **多后端支持**：xtquant、gm（掘金）、vnpy、sim（模拟）
2. **模拟账户**：资金初始化、状态跟踪
3. **订单执行**：市价单、限价单、止损单模拟撮合
4. **硬断路器**：单日亏损>2%拒新开仓、单笔限额、订单频率限制、持仓集中度限制
5. **确认模式与演习模式**：仿真模式（真实行情+模拟撮合）、一键切换paper_trade
6. **审计日志完整性**：所有订单包括被风控拒绝都记录完整上下文
7. **仓位同步与成本计算**：实时仓位管理
8. **交易日志记录**：完整的交易记录

## 核心模块

| 模块 | 说明 |
|------|------|
| BaseTrader | 交易抽象基类 |
| SimAccount | 模拟账户管理 |
| OrderSimulator | 订单模拟执行 |
| CircuitBreaker | 硬断路器 |
| PositionManager | 仓位管理 |
| AuditLogger | 审计日志 |
| ExecutionEngine | 主引擎 |

## 使用示例

```python
from engine import ExecutionEngine
from config import get_config

# 创建引擎
config = get_config()
engine = ExecutionEngine(config)

# 初始化账户
engine.initialize_account(initial_capital=1000000)

# 下单
order = engine.place_order(
    symbol="000001.SZ",
    side="buy",
    order_type="market",
    quantity=100
)

# 撤单
engine.cancel_order(order_id=order.order_id)

# 获取持仓
positions = engine.get_positions()
```

## 后端适配器

| 适配器 | 说明 |
|--------|------|
| xtquantAdapter | 迅投量化交易接口 |
| gmAdapter | 掘金量化交易接口 |
| vnpyAdapter | vn.py交易接口 |
| SimAdapter | 模拟交易接口 |

## 硬断路器规则

详见 references/trading_rules.md

## 配置说明

详见 references/config_guide.md

## API 文档

详见 references/api_reference.md
