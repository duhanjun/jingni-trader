# 配置指南

## 概述

本指南详细说明了 `BacktestEngine` 的所有可配置参数。

## 配置参数

### 回测引擎配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| BACKTEST_ENGINE | str | "rqalpha" | 回测引擎类型: "rqalpha", "backtrader", "gm" |
| GM_TOKEN | str | None | 掘金量化 API Token |
| REPORT_DIR | str | "./reports" | 回测报告输出目录 |

### A股交易规则配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| ENABLE_T1 | bool | True | 是否启用 T+1 规则 |
| LIMIT_UP_DOWN_MODEL | str | "strict" | 涨跌停模型: "strict"(废单) 或 "queue"(排队) |
| COMMISSION_RATE | float | 0.00025 | 佣金率 (万分之2.5) |
| COMMISSION_MIN | float | 5.0 | 最低佣金 (元) |
| STAMP_DUTY_RATE | float | 0.001 | 印花税率 (千分之一, 卖出时收取) |
| TRANSFER_FEE_RATE | float | 0.00002 | 过户费率 (十万分之二) |
| ST_LIMIT_RATE | float | 0.05 | ST股涨跌停幅度 (5%) |
| NORMAL_LIMIT_RATE | float | 0.10 | 普通股涨跌停幅度 (10%) |

### 绩效分析配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| RISK_FREE_RATE | float | 0.03 | 无风险利率 (3%) |
| ANNUALIZATION_FACTOR | int | 252 | 年化因子 (252个交易日) |

### Walk-Forward 配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| WALK_FORWARD_TRAIN_WINDOW | int | 252 | 训练窗口期 (交易日数) |
| WALK_FORWARD_TEST_WINDOW | int | 63 | 测试窗口期 (交易日数) |

### 可视化配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| PLOT_BACKEND | str | "matplotlib" | 绘图后端: "matplotlib" 或 "plotly" |

## 配置方式

### 1. 代码方式

```python
from config import get_config

config = get_config(
    BACKTEST_ENGINE="backtrader",
    COMMISSION_RATE=0.0003,
    ENABLE_T1=True,
)
```

### 2. 环境变量方式

```bash
export BACKTEST_ENGINE="rqalpha"
export REPORT_DIR="./my_reports"
export GM_TOKEN="your_token_here"
```
