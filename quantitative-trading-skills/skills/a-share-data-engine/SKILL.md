---
name: a-share-data-engine
version: 1.0.0
description: A股数据源统一引擎，支持多数据源切换、统一数据清洗、复权处理和数据存储
author: OpenClaw Team
license: MIT
tags:
  - quant-trading
  - A股
  - data-engine
  - tushare
  - akshare
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - tushare>=1.2.60
  - baostock>=0.8.8
  - akshare>=1.11.0
  - sqlalchemy>=2.0.0
  - pymysql>=1.0.0
  - psycopg2-binary>=2.9.0
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
entry_point: engine.py
---

# a-share-data-engine Skill

## 概述

a-share-data-engine 是 A 股量化投研的数据源统一引擎，提供：

1. **多数据源支持**：Tushare、BaoStock、AkShare、xtquant、掘金量化
2. **统一接口**：BaseDataProvider 抽象基类，get_daily() 统一返回标准化 DataFrame
3. **数据清洗**：复权处理、停牌标记、涨跌停标记、ST/退市过滤、新股过滤
4. **数据存储**：支持 SQLite/MySQL/PostgreSQL 多种数据库
5. **快照管理**：支持创建和管理数据快照

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

```python
from engine import AShareDataEngine
from config import get_config

# 创建引擎
config = get_config()
engine = AShareDataEngine(config)

# 获取数据
df = engine.get_daily(
    codes=["000001.SZ", "600000.SH"],
    start_date="2020-01-01",
    end_date="2024-01-01"
)

# 保存数据
from storage import DataStorage
storage = DataStorage(config)
storage.save_daily(df)
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

详见 references/data_cleaning_rules.md

## 配置说明

详见 references/config_guide.md

## API 文档

详见 references/api_reference.md
