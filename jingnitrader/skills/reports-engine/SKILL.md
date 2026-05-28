---
name: reports-engine
version: 1.0.0
description: A股绩效归因与可视化报告引擎。整合全流程产物，生成包含净值曲线、绩效指标 TearSheet、月度收益热力图、申万行业归因、Brinson 分解、风格暴露分析的综合 HTML 报告。基于 QuantStats / Plotly / Matplotlib 输出交互式图表和结构化数据 JSON。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - reports-engine
  - 绩效
  - 可视化
  - quantstats
  - plotly
dependencies:
  - quantstats
  - plotly>=5.15.0
  - matplotlib>=3.7.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - jinja2>=3.1.0
environment_variables:
  - name: REPORT_DIR
    description: 报告存储目录
    required: false
    default: "./quant_workspace/reports"
  - name: REPORT_TITLE
    description: 报告标题
    required: false
    default: "A股量化策略绩效报告"
  - name: BENCHMARK
    description: 基准指数
    required: false
    default: "000300.SH"
language: python
python_version: "3.9+"
entry_point: engine.py
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
---

# reports-engine

## 概述

reports-engine 是 A 股量化投研的**绩效归因与可视化报告引擎**，提供：

1. **基础绩效指标**：年化收益、夏普、最大回撤、Calmar、胜率
2. **QuantStats TearSheet**：交互式 HTML 报告
3. **Plotly 图表**：净值曲线、月度热力图
4. **风格暴露分析**：大盘/小盘/成长/价值
5. **申万行业归因**：利润贡献分解
6. **Brinson 归因**：配置效应 vs 选择效应

## 报告结构

1. 概览摘要（关键指标卡片）
2. 净值曲线与回撤图
3. 月度收益热力图
4. 绩效统计表
5. 风格暴露图
6. 行业归因图
7. Brinson 分解表
8. 完整交易记录

## 使用示例

### Python API

```python
from engine import run
from context import Context

ctx = Context(
    task_id="task_001",
    user_intent="生成报告",
    current_stage="IDLE"
)

result = run(ctx)
```

### CLI 运行

```bash
python engine.py -i "生成我的策略报告"
```

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)
