# API 参考文档

## 核心类

### AShareDataEngine

主引擎类，提供统一的数据获取接口。

```python
from a_share_data_engine import AShareDataEngine, get_config

config = get_config(DATA_BACKEND="tushare")
engine = AShareDataEngine(config)
```

#### 方法

##### get_daily(codes, start_date, end_date, adj=None, clean=True)

获取日线数据。

**参数：**
- `codes`: List[str] - 股票代码列表
- `start_date`: str - 开始日期，格式 'YYYY-MM-DD'
- `end_date`: str - 结束日期，格式 'YYYY-MM-DD'
- `adj`: Optional[str] - 复权类型，'qfq'（前复权）、'hfq'（后复权）、'none'（不复权）
- `clean`: bool - 是否清洗数据

**返回：**
- DataFrame - 包含标准化字段的数据

##### get_stock_list()

获取股票列表。

**返回：**
- DataFrame - 股票基本信息

##### get_trading_calendar(start_date, end_date)

获取交易日历。

**参数：**
- `start_date`: str - 开始日期
- `end_date`: str - 结束日期

**返回：**
- List[str] - 交易日列表

##### check_data_quality(df)

检查数据质量。

**参数：**
- `df`: DataFrame - 数据

**返回：**
- dict - 质量报告

---

### Config

配置类。

```python
from a_share_data_engine import Config

config = Config(
    DATA_BACKEND="tushare",
    TUSHARE_TOKEN="your_token",
    DB_TYPE="sqlite",
)
```

**配置项：**
- `DATA_BACKEND`: 数据源后端
- `DATA_DIR`: 数据存储目录
- `TUSHARE_TOKEN`: Tushare Token
- `GM_TOKEN`: 掘金 Token
- `DB_TYPE`: 数据库类型
- `DB_HOST`: 数据库主机
- `DB_PORT`: 数据库端口
- `DB_NAME`: 数据库名
- `DB_USER`: 数据库用户
- `DB_PASSWORD`: 数据库密码
- `CLEANING_ENABLED`: 是否启用清洗
- `FILTER_ST`: 是否过滤ST
- `FILTER_NEW_STOCK_DAYS`: 新股过滤天数
- `ADJ_TYPE`: 默认复权类型

---

### DataStorage

数据存储类。

```python
from a_share_data_engine import DataStorage

storage = DataStorage(config)
```

#### 方法

##### save_daily(df, if_exists="replace")

保存日线数据到数据库。

##### load_daily(codes=None, start_date=None, end_date=None)

从数据库加载日线数据。

##### save_to_csv(df, filename)

保存为 CSV。

##### load_from_csv(filename)

从 CSV 加载。

---

### SnapshotManager

快照管理器。

```python
from a_share_data_engine import SnapshotManager

snapshot_mgr = SnapshotManager(config)
```

#### 方法

##### create_snapshot(df, name, description="")

创建快照。

##### load_snapshot(snapshot_id)

加载快照。

##### list_snapshots()

列出所有快照。

##### delete_snapshot(snapshot_id)

删除快照。

---

### DataCleaner

数据清洗器。

```python
from a_share_data_engine import DataCleaner

cleaner = DataCleaner(config)
```

#### 方法

##### clean(df)

执行完整的清洗流程。

##### calculate_missing_rate(df)

计算数据缺失率。
