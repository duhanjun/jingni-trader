---
name: jingnitrader
version: 1.0.0
description: A股量化交易全流程主调度器。负责解析用户意图，管理投研阶段状态机，维护跨 Skill 的上下文对象，按流程依次调度七个子 Skill 完成从数据采集到绩效报告的全链路工作。本身不执行任何量化计算，只做编排。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - master-skill
  - workflow
  - 量化
  - 调度器
dependencies:
  - importlib (Python 标准库)
  - logging (Python 标准库)
  - json (Python 标准库)
environment_variables:
  - name: TUSHARE_TOKEN
    description: Tushare Pro API Token，用于获取A股数据
    required: false
  - name: GM_TOKEN
    description: 掘金量化API Token，用于实盘交易
    required: false
  - name: DATA_DIR
    description: 数据和工作目录
    required: false
    default: "./quant_workspace"
  - name: LOG_LEVEL
    description: 日志级别
    required: false
    default: "INFO"
language: python
python_version: "3.9+"
entry_point: engine.py
allowed_sub_skills:
  - a-share-data-engine
  - a-share-factor-engine
  - strategy-model-engine
  - backtest-engine
  - portfolio-risk-engine
  - execution-monitor-engine
  - reports-engine
trigger_keywords:
  - 量化
  - 回测
  - 选股
  - 因子
  - 实盘
  - 组合优化
  - A股
  - 策略开发
---

# jingnitrader

## 概述

jingnitrader 是量化交易 Skill 套件的**主协调中枢**，负责：

1. 解析用户自然语言意图，判断当前投研阶段
2. 管理任务状态机，按流程调度子 Skill
3. 维护会话状态和任务上下文
4. 输出结构化的量化研究报告

## 阶段状态机

```
[数据获取] → [因子构建] → [模型训练] → [回测验证] → [组合优化] → [模拟/实盘] → [绩效报告]
```

### 分支逻辑

- 回测失败 → 返回因子调优
- 模型过拟合 → 触发样本外再验证

## Context 对象

标准化的上下文对象，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| task_id | str | 当前任务ID |
| user_intent | str | 用户原始意图 |
| current_stage | str | 当前所处阶段 |
| target_stages | List[str] | 目标阶段列表 |
| stock_pool | List[str] | 股票池（股票代码列表） |
| start_date | str | 开始日期 |
| end_date | str | 结束日期 |
| artifacts | Dict[str, str] | 已完成阶段产物路径 |
| metadata | Dict[str, Any] | 各阶段元数据 |
| errors | List[str] | 错误记录 |

## 使用示例

### Python API

```python
from engine import run, MasterEngine
from context import Context

# 创建 Context
ctx = Context(
    task_id="task_001",
    user_intent="帮我用近3年A股数据做一个20日反转因子选股回测",
    current_stage="IDLE"
)

# 运行主流程
result = run(ctx)
print(result)
```

### CLI 运行

```bash
# 交互式输入
python engine.py -i "帮我用近3年A股数据做一个20日反转因子选股回测"

# 指定参数
python engine.py --task-id test001 --stock-pool 000001.SZ,600000.SH --start-date 2021-01-01 --end-date 2024-01-01

# 仅生成报告
python engine.py -i "生成上个月实盘绩效报告"
```

## 子 Skill 映射

| 阶段 | 对应子 Skill |
|------|-------------|
| DATA | a-share-data-engine |
| FACTOR | a-share-factor-engine |
| MODEL | strategy-model-engine |
| BACKTEST | backtest-engine |
| PORTFOLIO | portfolio-risk-engine |
| EXECUTION | execution-monitor-engine |
| REPORT | reports-engine |

## 里程碑检查点

每个子 Skill 完成后自动检查：

- 产物完整性
- 基本合理性
- 失败时给出清晰错误码
- 支持从断点重试

## 错误处理

- 所有子 Skill 调用包含异常捕获
- 明确的错误信息和建议
- 优雅降级策略

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)

## Context 协议

详见 [references/context_protocol.md](references/context_protocol.md)

## 工作流架构

详见 [references/workflow_architecture.md](references/workflow_architecture.md)
