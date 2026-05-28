# 配置指南

本文档说明 a-share-data-engine 的配置选项。

## 环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| TUSHARE_TOKEN | Tushare Pro API Token | 否 | 空 |
| GM_TOKEN | 掘金量化API Token | 否 | 空 |
| DATA_DIR | 数据存储目录 | 否 | "./data" |
| DATA_BACKEND | 数据源后端 | 否 | "tushare" |

## 配置项

详见 `scripts/config.py`：

```python
DATA_BACKEND = "tushare"  # 可选: tushare/baostock/akshare
DATA_FORMAT = "parquet"   # 可选: parquet/csv/sql
ADJUST_MODE = "qfq"      # 复权模式: qfq/hfq/bfq
CACHE_DIR = "./cache"
MAX_MISSING_RATIO = 0.1  # 最大缺失比例
```
