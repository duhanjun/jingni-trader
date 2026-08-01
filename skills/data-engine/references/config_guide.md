# 配置指南

本文档说明 data-engine 的配置选项。

## 环境变量

### 数据源配置

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| DATA_BACKENDS | 数据源优先级链，逗号分隔 | 否 | "baostock,akshare,websearch" |
| DATA_BACKEND | 单源模式（不降级，与 DATA_BACKENDS 互斥） | 否 | 空 |
| ALLOW_SYNTHETIC_FALLBACK | 全部数据源失败时生成模拟数据兜底 | 否 | "true" |
| AUTO_INSTALL_BACKENDS | 数据源依赖缺失时自动 pip install | 否 | "true" |

### 数据源凭证

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| TUSHARE_TOKEN | Tushare Pro API Token | tushare 源必需 | 空 |
| GM_TOKEN | 掘金量化 API Token | gm 源必需 | 空 |
| IFIND_USERNAME | 同花顺 iFinD 登录账号 | ifind 源必需 | 空 |
| IFIND_PASSWORD | 同花顺 iFinD 登录密码 | ifind 源必需 | 空 |

### 数据存储与清洗

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| QUANT_WORK_DIR | 工作目录根路径 | "./workspace" |
| DATA_FORMAT | 数据落盘格式（parquet/csv/sql） | "parquet" |
| ADJUST_MODE | 复权方式（前复权 hfq / 后复权 qfq） | "hfq" |
| DATA_MAX_WORKERS | 并行下载线程数 | "4" |

## 数据源优先级策略

**获取数据时按以下优先级选择数据源：**

1. **系统内置工具提供的外部数据（最高优先级）** — ctx.external_data
2. **用户对话指定的数据源** — ctx.data_sources（如"用 wind 取数据"）
3. **环境变量配置的数据源** — DATA_BACKENDS
4. **模拟数据兜底** — synthetic

## 精准降级规则

| 数据源 | 降级条件 | 降级到 |
|--------|---------|--------|
| tushare | QuotaExceededError / RateLimitError | baostock |
| baostock | BlacklistedError / DataNotFoundError / NetworkError | akshare |
| akshare | NetworkError / BlacklistedError / DataNotFoundError | websearch |
| websearch | DataNotFoundError | synthetic |

## 数据源适配器

| 适配器 | 数据源 | 接口 | 需要Token/配置 |
|--------|--------|------|---------------|
| BaoStockAdapter | BaoStock | get_daily, get_financial | 无 |
| AkShareAdapter | AkShare | get_daily, get_financial | 无 |
| WebSearchAdapter | WebSearch | get_daily, get_financial | web_search_fn 注入 |
| TushareAdapter | Tushare Pro | get_daily, get_financial | TUSHARE_TOKEN |
| XtQuantAdapter | 迅投 QMT | get_daily, get_financial | 本地客户端 |
| GmAdapter | 掘金量化 | get_daily, get_financial | GM_TOKEN |
| TdxQuantAdapter | 通达信 | get_daily, get_financial | 本地终端 |
| WindAdapter | 万得 | get_daily, get_financial | Wind 终端 |
| IfindAdapter | 同花顺 iFinD | get_daily, get_financial | IFIND_USERNAME/IFIND_PASSWORD |

## 使用示例

```python
import os
os.environ['DATA_BACKENDS'] = 'baostock,akshare,websearch'

from engine import run
from context import Context

ctx = Context(task_id="task_001", stock_pool=["000001.SZ", "600000.SH"])
ctx.start_date = "2021-01-01"
ctx.end_date = "2024-01-01"

result = run(ctx)
```
