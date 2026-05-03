
---
name: quant-trading-master
version: 1.0.0
description: 量化交易全流程协调中枢Skill，负责管理任务状态机，调度子Skill完成A股量化投研全流程
author: OpenClaw Team
license: MIT
tags:
  - quant-trading
  - A股
  - master-skill
  - workflow
dependencies:
  - a-share-data-engine
  - a-share-factor-engine
  - strategy-model-engine
  - backtest-engine
  - portfolio-risk-engine
  - execution-monitor-engine
  - reports-engine
environment_variables:
  - name: TUSHARE_TOKEN
    description: Tushare Pro API Token
    required: false
  - name: GM_TOKEN
    description: 掘金量化API Token
    required: false
  - name: DATA_DIR
    description: 数据存储目录
    required: false
    default: "./data"
language: python
python_version: "3.9+"
entry_point: scripts/main_workflow.py
---

# quant-trading-master Skill

## 概述

quant-trading-master是量化交易Skill套件的**主协调中枢**，负责：

1. 解析用户意图，判断当前投研阶段
2. 管理任务状态机，按流程调度子Skill
3. 维护会话状态和任务上下文
4. 输出结构化的量化研究报告

## 阶段状态机

```
[数据获取] → [因子构建] → [模型训练] → [回测验证] → [组合优化] → [模拟／实盘] → [绩效报告]
```

### 分支逻辑

- 回测失败 → 返回因子调优
- 模型过拟合 → 触发样本外再验证

## Context对象

Context对象标准化定义，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| task_id | str | 当前任务ID |
| session_id | str | 当前会话ID |
| stock_pool | List[str] | 股票池（股票代码列表） |
| time_range | dict | 时间范围 {start_date, end_date} |
| artifacts | dict | 已完成阶段产物路径 |
| config | dict | 全局配置 |
| current_stage | str | 当前所处阶段 |
| stage_history | List[str] | 已完成阶段历史 |
| results | dict | 各阶段执行结果 |

## 使用示例

```python
from scripts.main_workflow import run, QuantTradingContext

# 创建Context
ctx = QuantTradingContext(
    task_id="task_001",
    session_id="session_001",
    stock_pool=["000001.SZ", "000002.SZ", "600000.SH"],
    time_range={"start_date": "2020-01-01", "end_date": "2024-01-01"},
    config={"data_backend": "tushare", "backtest_backend": "rqalpha"}
)

# 运行主流程
result = run(ctx)
```

## 子Skill映射

| 阶段 | 对应子Skill |
|------|------------|
| 数据获取 | a-share-data-engine |
| 因子构建 | a-share-factor-engine |
| 模型训练 | strategy-model-engine |
| 回测验证 | backtest-engine |
| 组合优化 | portfolio-risk-engine |
| 模拟／实盘 | execution-monitor-engine |
| 绩效报告 | reports-engine |

## 里程碑检查点

每个子Skill完成后自动检查：
- 产物完整性
- 基本合理性
- 失败时给出清晰错误码
- 支持从断点重试

## 错误处理

- 所有子Skill调用包含异常捕获
- 明确的错误信息和建议
- 优雅降级策略
