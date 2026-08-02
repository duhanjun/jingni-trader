# 配置指南

本文档说明 a-share-factor-engine 的配置选项。

## 环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| FACTOR_BACKEND | 因子计算后端 | 否 | "pandas_ta" |
| FACTOR_DIR | 因子数据存储目录 | 否 | "./workspace/factors" |
| IC_TYPE | IC计算方式 | 否 | "normal" |
| QUANT_WORK_DIR | 工作目录根路径 | 否 | "./workspace" |
| QUANT_ALPHALENS_REPORT | 是否生成 alphalens 因子分析报告（0/1） | 否 | "0" |

## 配置文件

配置文件位于 `scripts/config.py`。

### 因子计算配置

```python
FACTOR_BACKEND = os.getenv("FACTOR_BACKEND", "pandas_ta")
FACTOR_DIR = os.path.expanduser(os.getenv("FACTOR_DIR", "./workspace/factors"))
IC_TYPE = os.getenv("IC_TYPE", "normal")
```

### 中性化配置

```python
NEUTRALIZE_INDUSTRY = True
NEUTRALIZE_MARKET_CAP = True
```

### IC分析配置

```python
QUANTILES = 5
MIN_IC = 0.02
MIN_IC_IR = 0.3
MAX_CORRELATION = 0.8
```

## 后端选择

- `talib`: 使用 TA-Lib 计算技术指标（需要安装 ta-lib）
- `pandas_ta`: 使用 pandas-ta（纯 Python，默认）

## Alphalens 因子分析报告

通过 `QUANT_ALPHALENS_REPORT` 环境变量控制是否在 FACTOR 阶段自动生成 alphalens 因子分析报告。

### 启用方式

```bash
# 启用（每个入选因子输出 4 PNG + 1 HTML + 1 JSON）
export QUANT_ALPHALENS_REPORT=1

# 关闭（默认，不生成额外报告）
export QUANT_ALPHALENS_REPORT=0
```

### 依赖

- `alphalens-reloaded>=0.4.5`（已在 requirements.txt 声明，兼容 pandas 2.0）
- 未安装时自动降级到方案 C（自研轻量分层回测，仅输出 JSON + HTML）

### 输出位置

```
<QUANT_WORK_DIR>/reports/alphalens/<task_id>/
├── <factor>_returns.png       # 分层累积收益
├── <factor>_ic.png            # IC 时序图
├── <factor>_turnover.png      # 换手率分析
├── <factor>_summary.png       # 综合统计
├── <factor>_report.html       # 单因子 HTML 报告
└── <factor>_metrics.json      # 8 项关键指标
```

### 建议结论阈值

metrics.json 中的 `suggested_verdict` 字段基于以下阈值（参考 RuleJudge）：

| 结论 | 条件 |
|------|------|
| ACCEPT | `ic_ir >= 0.5` 且 `long_short_sharpe >= 0.8` |
| REVIEW | 不满足 ACCEPT 条件 |
| REJECT | （保留，当前未使用） |

## 使用示例

```python
import os
os.environ['FACTOR_BACKEND'] = 'pandas_ta'
os.environ['FACTOR_DIR'] = './factors'
os.environ['QUANT_ALPHALENS_REPORT'] = '1'  # 启用 alphalens 报告

from engine import run, FactorEngine
result = run(ctx)
```

