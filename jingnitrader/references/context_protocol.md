# Context 协议

本文档定义 jingnitrader 中使用的 Context 对象的协议和规范。

## 概述

Context 是量化投研流程中的核心数据结构，用于在不同阶段之间传递状态和元数据。它确保了：

1. **状态传递**：各阶段可以访问之前阶段的产物和结果
2. **断点续跑**：可以根据已有的 Context 恢复执行
3. **错误追踪**：记录各阶段的错误信息

## 数据结构

### 必需字段

| 字段名 | 类型 | 描述 | 示例 |
|--------|------|------|------|
| task_id | str | 任务唯一标识 | "20240101120000" |
| user_intent | str | 用户原始意图 | "帮我做一个20日反转因子回测" |
| current_stage | str | 当前所处阶段 | "DATA" |
| target_stages | List[str] | 需要执行的目标阶段 | ["DATA", "BACKTEST", "REPORT"] |

### 可选字段

| 字段名 | 类型 | 描述 | 默认值 |
|--------|------|------|--------|
| stock_pool | List[str] | 股票池代码 | [] (全市场) |
| start_date | str | 数据开始日期 | "2021-01-01" |
| end_date | str | 数据结束日期 | "2024-12-31" |
| artifacts | Dict[str, str] | 各阶段产物路径 | {} |
| metadata | Dict[str, Any] | 各阶段元数据 | {} |
| errors | List[str] | 错误记录 | [] |

## 阶段定义

### 阶段列表

1. **IDLE** - 初始状态
2. **DATA** - 数据获取阶段
3. **FACTOR** - 因子构建阶段
4. **MODEL** - 模型训练阶段
5. **BACKTEST** - 回测验证阶段
6. **PORTFOLIO** - 组合优化阶段
7. **EXECUTION** - 执行交易阶段
8. **REPORT** - 报告生成阶段

### 阶段顺序

```
IDLE → DATA → FACTOR → MODEL → BACKTEST → PORTFOLIO → EXECUTION → REPORT
```

## 产物规范

### 产物文件命名

每个阶段完成后应生成对应的产物文件：

| 阶段 | 产物文件 | 格式 | 描述 |
|------|----------|------|------|
| DATA | cleaned_data.parquet | Parquet | 清洗后的行情数据 |
| FACTOR | factor_data.parquet | Parquet | 因子数据 |
| MODEL | model.pkl | Pickle | 训练好的模型 |
| BACKTEST | backtest_result.json | JSON | 回测结果 |
| PORTFOLIO | portfolio_weights.json | JSON | 组合权重配置 |
| EXECUTION | trade_log.json | JSON | 交易记录 |
| REPORT | report.html | HTML | 绩效报告 |

### 产物路径规范

产物文件应保存在对应的工作目录下：

```
workspace/
├── data/
│   └── cleaned_data.parquet      # DATA 阶段产物
├── factors/
│   └── factor_data.parquet       # FACTOR 阶段产物
├── models/
│   └── model.pkl                 # MODEL 阶段产物
├── backtest_results/
│   └── backtest_result.json      # BACKTEST 阶段产物
├── portfolio/
│   └── portfolio_weights.json     # PORTFOLIO 阶段产物
├── reports/
│   └── report.html               # REPORT 阶段产物
└── logs/
    └── *.log                      # 日志文件
```

## 元数据规范

### metadata 结构

```json
{
  "DATA": {
    "record_count": 1000000,
    "date_range": ["2021-01-01", "2024-12-31"],
    "data_source": "tushare"
  },
  "FACTOR": {
    "factor_names": ["reverse_20d", "volume_ratio"],
    "ic_mean": 0.05,
    "ic_std": 0.02
  },
  "MODEL": {
    "model_type": "lightgbm",
    "features": ["reverse_20d", "volume_ratio"],
    "accuracy": 0.65
  }
}
```

## 错误处理规范

### 错误记录格式

```python
ctx.add_error(f"{stage}: {error_message}")
```

### 错误类型

1. **数据错误** - 数据获取或处理失败
2. **计算错误** - 因子计算或模型训练失败
3. **系统错误** - 文件IO、网络等系统级错误

## Context 生命周期

### 1. 创建阶段

```python
ctx = Context(
    task_id="task_001",
    user_intent="帮我做一个回测",
    current_stage="IDLE"
)
```

### 2. 意图解析阶段

```python
ctx = engine.parse_intent(user_input)
# 设置 target_stages, stock_pool, start_date, end_date
```

### 3. 阶段执行阶段

```python
for stage in ctx.target_stages:
    success = engine.execute_stage(stage)
    if success:
        ctx.update_artifact(stage, artifact_path)
        ctx.metadata[stage] = stage_metadata
    else:
        ctx.add_error(f"{stage}: error_message")
```

### 4. 完成阶段

```python
result = {
    "success": len(ctx.errors) == 0,
    "context": ctx.to_dict(),
    "summary": engine._generate_summary()
}
```

## 序列化规范

### JSON 序列化

```python
# 转换为 JSON
json_str = json.dumps(ctx.to_dict(), ensure_ascii=False, indent=2)

# 从 JSON 恢复
ctx = Context.from_json(json_str)
```

### 文件持久化

```python
# 保存到文件
with open("context.json", "w", encoding="utf-8") as f:
    json.dump(ctx.to_dict(), f, ensure_ascii=False, indent=2)

# 从文件恢复
with open("context.json", "r", encoding="utf-8") as f:
    ctx = Context.from_json(f.read())
```

## 示例

### 完整示例

```python
from context import Context

# 创建 Context
ctx = Context(
    task_id="task_001",
    user_intent="帮我用近3年A股数据做一个20日反转因子选股回测",
    current_stage="IDLE"
)

# 意图解析后
ctx.target_stages = ["DATA", "FACTOR", "MODEL", "BACKTEST", "REPORT"]
ctx.stock_pool = ["000001.SZ", "600000.SH"]
ctx.start_date = "2021-01-01"
ctx.end_date = "2024-12-31"

# 阶段执行后
ctx.update_artifact("DATA", "workspace/data/cleaned_data.parquet")
ctx.metadata["DATA"] = {"record_count": 1000000}

# 错误记录
if error:
    ctx.add_error(f"DATA: 数据获取失败")

# 最终结果
result = ctx.to_dict()
```

## 最佳实践

1. **总是初始化必需字段**
2. **及时更新 artifacts 和 metadata**
3. **错误信息要清晰，包含阶段和具体原因**
4. **定期保存 Context 以支持断点续跑**
5. **验证阶段产物完整性后再进入下一阶段**
