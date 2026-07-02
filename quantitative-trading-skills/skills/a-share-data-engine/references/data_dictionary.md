# 数据字典

## 日线数据字段

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| code | str | 股票代码 | 000001.SZ |
| date | datetime | 交易日期 | 2024-01-01 |
| open | float | 开盘价 | 10.50 |
| high | float | 最高价 | 11.00 |
| low | float | 最低价 | 10.20 |
| close | float | 收盘价 | 10.80 |
| volume | float | 成交量（手） | 100000 |
| amount | float | 成交额（元） | 108000000 |
| pre_close | float | 前收盘价 | 10.60 |
| change_pct | float | 涨跌幅（%） | 1.89 |
| turnover_rate | float | 换手率（%） | 2.5 |
| is_st | bool | 是否ST | False |
| is_limit_up | bool | 是否涨停 | False |
| is_limit_down | bool | 是否跌停 | False |
| is_suspended | bool | 是否停牌 | False |

## 股票列表字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| code | str | 股票代码 |
| symbol | str | 股票代码（不含后缀） |
| name | str | 股票名称 |
| area | str | 地区 |
| industry | str | 行业 |
| list_date | str | 上市日期 |

## 复权类型

| 类型 | 说明 |
|------|------|
| qfq | 前复权 - 以当前价格为基准，向前复权 |
| hfq | 后复权 - 以上市首日价格为基准，向后复权 |
| none | 不复权 - 原始价格 |

## 数据质量标准

- 数据缺失率 < 2% 为合格
- 关键字段（open, high, low, close, volume）缺失率 < 1%
