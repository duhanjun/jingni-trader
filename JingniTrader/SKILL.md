---
name: JingniTrader
description: >
  A股量化交易全流程主调度器。负责解析用户意图，管理投研阶段状态机，
  维护跨 Skill 的上下文对象，按流程依次调度七个子 Skill 完成从数据
  采集到绩效报告的全链路工作。本身不执行任何量化计算，只做编排。
trigger_keywords:
  - 量化
  - 回测
  - 选股
  - 因子
  - 实盘
  - 组合优化
  - A股
  - 策略开发
allowed_sub_skills:
  - a-share-data-engine
  - a-share-factor-engine
  - strategy-model-engine
  - backtest-engine
  - portfolio-risk-engine
  - execution-monitor-engine
  - reports-engine
version: 1.0.0
author: quant-team
---

# JingniTrader

## 职责

- 理解用户自然语言，判断投研阶段
- 维护全局 `Context` 对象，包含股票池、时间范围、产物路径等
- 按阶段状态机调度子 Skill
- 校验每个阶段的产物完整性
- 支持断点续跑（已有产物可跳过）
- 最终返回结构化摘要

## 阶段状态机
[IDLE]
│ 用户输入
▼
[DATA] ──→ [FACTOR] ──→ [MODEL] ──→ [BACKTEST] ──→ [PORTFOLIO] ──→ [EXECUTION] ──→ [REPORT]
│ │ │ │ │ │ │
└────────────┴────────────┴────────────┴───────────────┴───────────────┴──────────────┘
可随时跳到 REPORT

text

## 使用方式

用户直接说出需求，例如：
- "帮我用近3年A股数据做一个20日反转因子选股回测"
- "优化当前组合，最大回撤控制在15%以内"
- "生成上个月实盘绩效报告"

## 依赖

无底层量化库依赖，仅依赖 Python 标准库 + `importlib` 用于动态加载子 Skill。
