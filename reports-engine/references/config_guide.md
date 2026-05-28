# 配置指南

本文档说明 reports-engine 的配置选项。

## 环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| REPORT_DIR | 报告存储目录 | 否 | "./quant_workspace/reports" |
| REPORT_TITLE | 报告标题 | 否 | "A股量化策略绩效报告" |
| BENCHMARK | 基准指数 | 否 | "000300.SH" |

## 配置文件

配置文件位于 `scripts/config.py`。

### 报告配置

```python
REPORT_DIR = os.path.expanduser(os.getenv("REPORT_DIR", "./quant_workspace/reports"))
REPORT_TITLE = os.getenv("REPORT_TITLE", "A股量化策略绩效报告")
REPORT_FORMAT = os.getenv("REPORT_FORMAT", "html")
INDUSTRY_STANDARD = "sw"  # sw=申万, zz=中信
BENCHMARK = os.getenv("BENCHMARK", "000300.SH")
RISK_FREE_RATE = 0.03
```

### 图表配置

```python
INCLUDE_TEARSHEET = True
INCLUDE_HEATMAP = True
INCLUDE_ATTRIBUTION = True
INCLUDE_TRADES = True
CHART_THEME = "plotly_white"
```

## 使用示例

```python
import os
os.environ['REPORT_TITLE'] = '我的策略绩效报告'

from engine import run, ReportGenerator

result = run(ctx)
```
