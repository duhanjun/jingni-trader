# 配置指南

## 数据源配置

### Tushare

```python
config = get_config(
    DATA_BACKEND="tushare",
    TUSHARE_TOKEN="your_token_here",
)
```

获取 Token: https://tushare.pro/

### BaoStock

```python
config = get_config(
    DATA_BACKEND="baostock",
)
```

无需 Token，免费使用。

### AkShare

```python
config = get_config(
    DATA_BACKEND="akshare",
)
```

无需 Token，开源免费。

## 数据库配置

### SQLite（默认）

```python
config = get_config(
    DB_TYPE="sqlite",
    DATA_DIR="./data",
)
```

### MySQL

```python
config = get_config(
    DB_TYPE="mysql",
    DB_HOST="localhost",
    DB_PORT=3306,
    DB_NAME="a_share_data",
    DB_USER="root",
    DB_PASSWORD="password",
)
```

### PostgreSQL

```python
config = get_config(
    DB_TYPE="postgresql",
    DB_HOST="localhost",
    DB_PORT=5432,
    DB_NAME="a_share_data",
    DB_USER="postgres",
    DB_PASSWORD="password",
)
```

## 清洗配置

```python
config = get_config(
    CLEANING_ENABLED=True,
    FILTER_ST=False,
    FILTER_NEW_STOCK_DAYS=60,
    ADJ_TYPE="qfq",
)
```

## 环境变量

也可以通过环境变量配置：

```bash
export TUSHARE_TOKEN="your_token"
export DATA_DIR="./data"
export DB_TYPE="sqlite"
```

## 使用示例

### 完整示例

```python
from a_share_data_engine import (
    AShareDataEngine,
    get_config,
    DataStorage,
    SnapshotManager,
)

# 配置
config = get_config(
    DATA_BACKEND="baostock",
    DATA_DIR="./my_data",
)

# 创建引擎
engine = AShareDataEngine(config)
storage = DataStorage(config)
snapshot_mgr = SnapshotManager(config)

# 获取数据
df = engine.get_daily(
    codes=["000001.SZ", "600000.SH"],
    start_date="2024-01-01",
    end_date="2024-12-31",
)

# 检查质量
quality_report = engine.check_data_quality(df)
print(f"数据缺失率: {quality_report['missing_rate']:.2%}")

# 保存数据
storage.save_daily(df)

# 创建快照
snapshot_id = snapshot_mgr.create_snapshot(
    df,
    name="sample_data",
    description="2024年样例数据",
)

print(f"快照已创建: {snapshot_id}")
```
