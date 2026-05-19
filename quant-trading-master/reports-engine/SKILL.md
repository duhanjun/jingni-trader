---
name: reports-engine
description: >
  A股绩效归因与可视化报告引擎。整合全流程产物，生成包含净值曲线、
  绩效指标 TearSheet、月度收益热力图、申万行业归因、Brinson 分解、
  风格暴露分析的综合 HTML 报告。基于 QuantStats / Plotly / Matplotlib
  输出交互式图表和结构化数据 JSON。
trigger_keywords:
  - 报告
  - 绩效
  - 归因
  - 可视化
  - 热力图
  - 净值曲线
  - TearSheet
  - 行业归因
  - Brinson
  - 风格暴露
version: 1.0.0
author: quant-team
dependencies:
  - quantstats
  - plotly
  - matplotlib
  - pandas
  - numpy
  - jinja2
---

# reports-engine

## 职责

- 整合所有上游 Skill 产物，生成综合报告
- 基础绩效指标（年化收益、夏普、最大回撤、Calmar、胜率）
- QuantStats HTML 交互式 TearSheet
- Plotly 净值曲线图（含回撤子图）
- Plotly 月度收益热力图
- 风格暴露分析（大盘/小盘/成长/价值）
- 申万行业归因（利润贡献分解）
- Brinson 归因（配置效应 vs 选择效应）
- 输出 HTML 报告 + 结构化 JSON 数据

## 报告结构

1. 概览摘要（关键指标卡片）
2. 净值曲线与回撤图
3. 月度收益热力图
4. 绩效统计表
5. 风格暴露图
6. 行业归因图
7. Brinson 分解表
8. 完整交易记录（若有）

## 使用方式

由 `quant-trading-master` 调度，调用 `run(ctx)` 函数。