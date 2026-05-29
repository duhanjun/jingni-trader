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

## DataEngine

数据引擎类。

### fetch_and_clean()

获取并清洗日线数据。
