---
name: data-engine
version: 1.0.0
description: A股数据采集与治理引擎。支持从 Tushare Pro、BaoStock、AkShare、xtquant、掘金(gm)、通达信(tdxquant)、万得(Wind)、同花顺(iFinD) 等多数据源获取日线/分钟线行情、财务、估值数据，完成复权、涨跌停标记、ST过滤、新股剔除、停牌处理等本土化清洗，并持久化到本地 Parquet 文件。优先使用 Agent 系统内置的金融数据获取工具（MCP/Skill/插件）。支持精准降级、模拟数据兜底、依赖自动安装。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - data-engine
  - tushare
  - akshare
  - baostock
  - wind
  - ifind
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
  - name: IFIND_USERNAME
    description: 同花顺 iFinD 登录账号
    required: false
  - name: IFIND_PASSWORD
    description: 同花顺 iFinD 登录密码
    required: false
  - name: QUANT_WORK_DIR
    description: 工作目录根路径
    required: false
    default: "./workspace"
  - name: DATA_BACKENDS
    description: 数据源优先级链，逗号分隔
    required: false
    default: "baostock,akshare,websearch"
  - name: DATA_BACKEND
    description: 单源模式（不降级），与 DATA_BACKENDS 互斥
    required: false
  - name: ALLOW_SYNTHETIC_FALLBACK
    description: 全部数据源失败时生成模拟数据兜底
    required: false
    default: "true"
  - name: AUTO_INSTALL_BACKENDS
    description: 数据源依赖缺失时自动 pip install
    required: false
    default: "true"
  - name: DATA_FORMAT
    description: 数据落盘格式（parquet/csv/sql）
    required: false
    default: "parquet"
  - name: DATA_MAX_WORKERS
    description: 并行下载线程数
    required: false
    default: "4"
  - name: ADJUST_MODE
    description: 复权方式（前复权 hfq / 后复权 qfq）
    required: false
    default: "hfq"
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
  - 估值
  - 行业对比
---

# data-engine

## 概述

data-engine 是 A 股量化投研的**数据源统一引擎**，提供：

1. **多数据源支持**：Tushare、BaoStock、AkShare、xtquant、掘金量化、通达信、万得、同花顺 iFinD
2. **统一接口**：BaseDataProvider 抽象基类，`get_daily()` 和 `get_financial()` 统一返回标准化 DataFrame
3. **数据清洗**：复权处理、停牌标记、涨跌停标记、ST/退市过滤、新股过滤
4. **数据存储**：支持 Parquet/CSV/SQL 多种格式
5. **Agent 系统工具优先**：自动检测并使用系统内置金融数据获取工具
6. **精准降级**：每个数据源有明确的降级条件，按失败原因精准切换
7. **模拟数据兜底**：全部外部源失败时生成合成数据，跑通流程并如实告知用户
8. **依赖自动安装**：数据源库缺失时自动 `pip install`

## 数据源优先级策略

**获取数据时按以下优先级选择数据源：**

1. **系统内置工具提供的外部数据（最高优先级）**
   - 检查 Context 对象的 `external_data` 字段
   - 如果外部已通过 MCP/Skill/插件获取了数据，直接使用，跳过本地适配器
   - 外部数据格式：`{"daily": DataFrame, "stock_list": DataFrame, "source": "mcp_tushare"}`

2. **用户对话指定的数据源（ctx.data_sources）**
   - 用户通过对话明确指定（如"用 wind 取数据"），覆盖环境变量

3. **环境变量配置的数据源（DATA_BACKENDS）**
   - 默认 `"baostock,akshare,websearch"`（仅真正免费源）

4. **模拟数据兜底（synthetic）**
   - 全部外部源都失败时，生成几何布朗运动模拟数据，跑通流程并告知用户

**回退流程：**
```
external_data → ctx.data_sources → DATA_BACKENDS → 默认免费链 → synthetic
    ↓ 失败          ↓ 失败            ↓ 失败         ↓ 失败       ↓ 失败
  下一级         下一级           下一级        下一级       报错退出
```

## 精准降级规则

每个数据源有明确的降级条件（`DATA_FALLBACK_RULES`），只在特定异常类型触发时才切换：

| 数据源 | 降级条件 | 降级到 |
|--------|---------|--------|
| tushare | QuotaExceededError / RateLimitError（积分/权限/限频） | baostock |
| baostock | BlacklistedError / DataNotFoundError / NetworkError | akshare |
| akshare | NetworkError / BlacklistedError / DataNotFoundError | websearch |
| websearch | DataNotFoundError（搜索无结果） | synthetic（模拟数据） |

## 系统支持的全部数据源

> 全部 8 个数据源均已实现 `get_daily()` 和 `get_financial()` 接口，并通过实盘连通性测试。

### 默认免费源（自动参与降级链）
| 数据源 | 说明 | get_daily | get_financial | 需要配置 |
|--------|------|-----------|---------------|---------|
| baostock | 证券宝开源项目 | 支持 | 支持 | 无 |
| akshare | 聚合库爬虫 | 支持 | 支持 | 无 |
| websearch | WebSearch 工具查询 | 支持 | 支持 | 无 |

### opt-in 源（需显式启用）
| 数据源 | 说明 | get_daily | get_financial | 前置条件 |
|--------|------|-----------|---------------|---------|
| tushare | Tushare Pro | 支持 | 支持 | TUSHARE_TOKEN |
| xtquant | 迅投 QMT/xtp | 支持 | 支持 | 本地券商客户端 |
| gm | 掘金量化 | 支持 | 支持 | GM_TOKEN + SDK |
| tdxquant | 通达信量化 | 支持 | 支持 | 本地通达信终端 |
| wind | 万得 WindPy | 支持 | 支持 | Wind 金融终端 + WindPy |
| ifind | 同花顺 iFinD | 支持 | 支持 | iFinDPy + 账号密码 |

## 数据结构

### get_daily() 返回的 DataFrame 包含以下字段：

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

### get_financial() 返回的标准财务数据字段：

| 字段名 | 说明 |
|--------|------|
| code, report_date | 股票代码 + 报告期 |
| pe_ttm, pb, ps_ttm, dv_ratio | 估值指标 |
| roe, roa, gross_margin, net_margin | 盈利能力 |
| revenue_growth, profit_growth | 成长性 |
| debt_ratio, current_ratio, quick_ratio, ocf | 偿债能力与现金流 |
| industry, name | 行业与名称 |

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

# 使用内置适配器
result = run(ctx)

# 通过对话指定数据源优先级
ctx.data_sources = ["wind", "tushare", "baostock", "akshare", "websearch"]
result = run(ctx)

# 使用系统内置工具提供的外部数据（优先）
import pandas as pd
ctx.external_data = {
    "daily": pd.DataFrame(...),  # 系统内置工具已获取的日线数据
    "stock_list": pd.DataFrame(...),
    "source": "mcp_tushare"
}
result = run(ctx)
```

### CLI 运行

```bash
python engine.py -i "获取2021年的000001.SZ行情数据"
```

## 数据源适配器

| 适配器 | 数据源 | 需要Token/配置 |
|--------|--------|---------------|
| TushareAdapter | Tushare Pro | TUSHARE_TOKEN |
| BaoStockAdapter | 宝盛 | 无 |
| AkShareAdapter | AkShare | 无 |
| WebSearchAdapter | WebSearch | web_search_fn 注入 |
| XtQuantAdapter | 迅投 QMT | 本地客户端 |
| GmAdapter | 掘金量化 | GM_TOKEN |
| TdxQuantAdapter | 通达信 | 本地终端 |
| WindAdapter | 万得 | Wind 终端 |
| IfindAdapter | 同花顺 iFinD | IFIND_USERNAME/IFIND_PASSWORD |

## 数据清洗规则

详见 [references/data_cleaning_rules.md](references/data_cleaning_rules.md)

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)