# API 参考文档

## run(ctx) -> dict

Skill 标准入口函数。

**参数：**
- ctx: Context 对象

**返回：**
```json
{
  "success": true,
  "artifact_path": "/path/to/cleaned_data.parquet",
  "metadata": {
    "rows": 1000000,
    "symbols_count": 4000,
    "date_range": "2021-01-01 ~ 2024-12-31"
  },
  "error": ""
}
```

## BaseDataProvider

数据源适配器抽象基类，所有数据源适配器均继承此类。

### get_daily(code, start_date, end_date) -> pd.DataFrame

获取日线行情数据。

**返回字段：**

| 字段名 | 类型 | 说明 |
|--------|------|------|
| code | str | 股票代码（000001.SZ / 600000.SH） |
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

### get_financial(code) -> pd.DataFrame

获取财务指标数据。

**返回字段：**

| 字段名 | 说明 |
|--------|------|
| code, report_date | 股票代码 + 报告期 |
| pe_ttm, pb, ps_ttm, dv_ratio | 估值指标 |
| roe, roa, gross_margin, net_margin | 盈利能力 |
| revenue_growth, profit_growth | 成长性 |
| debt_ratio, current_ratio, quick_ratio, ocf | 偿债能力与现金流 |
| industry, name | 行业与名称 |

## 已实现的适配器

| 适配器 | 数据源 | get_daily | get_financial |
|--------|--------|-----------|---------------|
| BaoStockAdapter | BaoStock | 支持 | 支持 |
| AkShareAdapter | AkShare | 支持 | 支持 |
| WebSearchAdapter | WebSearch | 支持 | 支持 |
| TushareAdapter | Tushare Pro | 支持 | 支持 |
| XtQuantAdapter | 迅投 QMT | 支持 | 支持 |
| GmAdapter | 掘金量化 | 支持 | 支持 |
| TdxQuantAdapter | 通达信 | 支持 | 支持 |
| WindAdapter | 万得 | 支持 | 支持 |
| IfindAdapter | 同花顺 iFinD | 支持 | 支持 |
