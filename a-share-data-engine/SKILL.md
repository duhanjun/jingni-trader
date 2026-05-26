---
name: a-share-data-engine
description: >
  A股数据采集与治理引擎。支持从 Tushare Pro、BaoStock、AkShare、xtquant、
  掘金(gm) 等多数据源获取日线/分钟线行情、财务、另类数据，完成复权、
  涨跌停标记、ST过滤、新股剔除、停牌处理等本土化清洗，并持久化到本地
  数据库或 Parquet 文件。
trigger_keywords:
  - 数据获取
  - 行情
  - 日线
  - 分钟线
  - 财务
  - 数据清洗
  - 复权
version: 1.0.0
author: quant-team
dependencies:
  - tushare
  - baostock
  - akshare
  - pandas
  - sqlalchemy
  - numpy
  - pyarrow (可选)
backends:
  - tushare
  - baostock
  - akshare
  - xtquant
  - gm
---

# a-share-data-engine

## 职责

- 根据 Context 中的股票池、时间范围、数据类型获取数据
- 数据标准化：统一列名、日期格式、复权处理
- A股特殊标记：`is_st`、`is_limit_up`、`is_limit_down`、`listed_days`
- 默认过滤：剔除 ST、上市不足60日新股、停牌日
- 存储：Parquet 文件 + 可选 SQL 数据库
- 返回清洗后数据产物路径

## 使用方式

由 `quant-trading-master` 调度，调用 `run(ctx)` 函数。
也可独立 CLI 运行：
```bash
python -m a_share_data_engine.engine --input ctx.json
```
