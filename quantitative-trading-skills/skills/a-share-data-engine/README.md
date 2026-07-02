# A股数据引擎 (a-share-data-engine)

A股数据源统一引擎，支持多数据源切换、统一数据清洗、复权处理和数据存储。

## 功能特性

- **多数据源支持**: Tushare、BaoStock、AkShare、xtquant、掘金量化
- **统一接口**: BaseDataProvider 抽象基类，get_daily() 统一返回标准化 DataFrame
- **数据清洗**: 复权处理、停牌标记、涨跌停标记、ST/退市过滤、新股过滤
- **数据存储**: 支持 SQLite/MySQL/PostgreSQL 多种数据库
- **快照管理**: 支持创建和管理数据快照
- **数据质量检查**: 自动检查数据缺失率，确保缺失率 < 2%

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础使用

```python
from a_share_data_engine import AShareDataEngine, get_config

# 创建配置
config = get_config(DATA_BACKEND="baostock")

# 创建引擎
engine = AShareDataEngine(config)

# 获取数据
df = engine.get_daily(
    codes=["000001.SZ", "600000.SH"],
    start_date="2024-01-01",
    end_date="2024-12-31",
)

# 检查数据质量
quality_report = engine.check_data_quality(df)
print(f"数据缺失率: {quality_report['missing_rate']:.2%}")
```

## 项目结构

```
a-share-data-engine/
├── SKILL.md              # Skill 描述文件
├── README.md             # 项目说明
├── requirements.txt      # 依赖列表
├── __init__.py           # 包入口
├── engine.py             # 主引擎
├── config.py             # 配置管理
├── base/                 # 基础类
│   ├── __init__.py
│   └── base_data_provider.py
├── adapters/             # 数据源适配器
│   ├── __init__.py
│   ├── tushare_adapter.py
│   ├── baostock_adapter.py
│   ├── akshare_adapter.py
│   ├── xtquant_adapter.py
│   └── gm_adapter.py
├── cleaning/             # 数据清洗
│   ├── __init__.py
│   └── data_cleaner.py
├── storage/              # 数据存储
│   ├── __init__.py
│   └── data_storage.py
├── snapshots/            # 快照管理
│   ├── __init__.py
│   └── snapshot_manager.py
├── references/           # 文档
│   ├── api_reference.md
│   ├── data_dictionary.md
│   ├── data_cleaning_rules.md
│   └── config_guide.md
└── tests/                # 测试
    ├── __init__.py
    ├── test_data_cleaner.py
    └── test_engine.py
```

## 数据源选择

| 数据源 | Token | 免费 | 说明 |
|--------|-------|------|------|
| Tushare | 需要 | 部分 | 数据质量高 |
| BaoStock | 不需要 | 是 | 免费使用 |
| AkShare | 不需要 | 是 | 开源数据 |
| xtquant | 需要 | 部分 | 迅投量化 |
| 掘金 | 需要 | 部分 | 掘金量化 |

## 文档

- [API 参考](references/api_reference.md)
- [数据字典](references/data_dictionary.md)
- [数据清洗规则](references/data_cleaning_rules.md)
- [配置指南](references/config_guide.md)

## 测试

运行测试:

```bash
cd tests
python test_data_cleaner.py
python test_engine.py
```

## 许可证

MIT License
