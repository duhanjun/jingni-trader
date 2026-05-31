---
name: data-engine
version: 1.0.0
description: A股数据采集与治理引擎。支持从 Tushare Pro、BaoStock、AkShare、xtquant、掘金(gm) 等多数据源获取日线/分钟线行情、财务、另类数据，完成复权、涨跌停标记、ST过滤、新股剔除、停牌处理等本土化清洗，并持久化到本地数据库或 Parquet 文件。优先使用 Agent 系统内置的金融数据获取工具（MCP/Skill/插件）。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - data-engine
  - tushare
  - akshare
  - baostock
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - tushare>=1.2.60
  - baostock>=0.8.8
  - akshare>=1.11.0
  - sqlalchemy>=2.0.0
environment_variables:
  - name: TUSHARE_TOKEN
    description: Tushare Pro API Token
    required: false
  - name: GM_TOKEN
    description: 掘金量化API Token
    required: false
  - name: QUANT_DATA_DIR
    description: 数据存储目录
    required: false
    default: "./workspace/data"
language: python
python_version: "3.9+"
entry_point: engine.py
trigger_keywords:
  - 数据获取
  - 行情
  - 日线
  - 分钟线
  - 财务
  - 数据清洗
  - 复权
---

# data-engine

## 概述

data-engine 是 A 股量化投研的**数据源统一引擎**，提供：

1. **多数据源支持**：Tushare、BaoStock、AkShare、xtquant、掘金量化
2. **统一接口**：BaseDataProvider 抽象基类，get_daily() 统一返回标准化 DataFrame
3. **数据清洗**：复权处理、停牌标记、涨跌停标记、ST/退市过滤、新股过滤
4. **数据存储**：支持 SQLite/MySQL/PostgreSQL 多种数据库
5. **Agent 系统工具优先**：自动检测并使用系统内置金融数据获取工具

## ⚠️ 数据源优先级策略

**获取数据时按以下优先级选择数据源：**

1. **系统内置工具提供的外部数据（最高优先级）**
   - 检查 Context 对象的 `external_data` 字段
   - 如果外部已通过 MCP/Skill/插件获取了数据，直接使用，跳过本地适配器
   - 外部数据格式：`{"daily": DataFrame, "stock_list": DataFrame, "source": "mcp_tushare"}`

2. **环境变量配置的数据源（第二优先级）**
   - 尝试通过 `TUSHARE_TOKEN` 使用 Tushare Pro API
   - 尝试通过 `GM_TOKEN` 使用掘金量化 API

3. **免费离线数据源（兜底方案）**
   - BaoStock（无需 Token）
   - AkShare（无需 Token）

**回退流程：**
```
external_data → Tushare/GM → BaoStock/AkShare
    ↓ 失败          ↓ 失败        ↓ 失败
  下一级         下一级        报错退出
```

## 数据结构

get_daily() 返回的 DataFrame 包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| code | str | 股票代码，格式如 000001.SZ 或 600000.SH |
| date | datetime | 交易日期 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 成交量（手） |
| amount | float | 成交额（元） |
| pre_close | float | 前收盘价 |
| change_pct | float | 涨跌幅（%） |
| turnover_rate | float | 换手率（%） |
| is_st | bool | 是否ST |
| is_limit_up | bool | 是否涨停 |
| is_limit_down | bool | 是否跌停 |

## 使用示例

### Python API

```python
from engine import run
from context import Context

# 创建 Context
ctx = Context(
    task_id="task_001",
    user_intent="获取数据",
    current_stage="IDLE"
)
ctx.stock_pool = ["000001.SZ", "600000.SH"]
ctx.start_date = "2021-01-01"
ctx.end_date = "2024-01-01"

# 方式1：使用内置适配器
result = run(ctx)

# 方式2：使用系统内置工具提供的外部数据（优先）
import pandas as pd
ctx.external_data = {
    "daily": pd.DataFrame(...),  # 系统内置工具已获取的日线数据
    "stock_list": pd.DataFrame(...),  # 系统内置工具已获取的股票列表
    "source": "mcp_tushare"
}
result = run(ctx)
```

### CLI 运行

```bash
python engine.py -i "获取2021年的000001.SZ行情数据"
```

## 数据源适配器

| 适配器 | 说明 | 需要Token |
|--------|------|-----------|
| TushareAdapter | Tushare Pro 数据源 | 是 |
| BaoStockAdapter | 宝盛数据源 | 否 |
| AkShareAdapter | AkShare 开源数据源 | 否 |
| xtquantAdapter | 迅投量化数据源 | 是 |
| gmAdapter | 掘金量化数据源 | 是 |

## 数据清洗规则

详见 [references/data_cleaning_rules.md](references/data_cleaning_rules.md)

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)