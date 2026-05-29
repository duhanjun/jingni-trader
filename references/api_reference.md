# API 参考文档

本文档提供 jingnitrader 的完整 API 参考。

## 核心类

### MasterEngine

主调度引擎类，负责管理整个量化投研流程。

#### 方法

##### `__init__()`

初始化主调度引擎。

```python
engine = MasterEngine()
```

##### `parse_intent(user_input: str) -> Context`

解析用户自然语言输入，生成 Context 对象。

**参数：**
- `user_input` (str): 用户自然语言描述

**返回：**
- `Context`: 初始化后的上下文对象

**示例：**

```python
ctx = engine.parse_intent("帮我用近3年A股数据做一个20日反转因子选股回测")
```

##### `execute_stage(stage: str) -> bool`

执行单个阶段，调用对应的子 Skill。

**参数：**
- `stage` (str): 阶段名称（DATA/FACTOR/MODEL/BACKTEST/PORTFOLIO/EXECUTION/REPORT）

**返回：**
- `bool`: 执行是否成功

**示例：**

```python
success = engine.execute_stage("DATA")
```

##### `run_pipeline(user_input: str = None, ctx: Context = None) -> dict`

执行全流程管道。

**参数：**
- `user_input` (str, optional): 用户自然语言输入
- `ctx` (Context, optional): 已有的上下文对象

**返回：**
- `dict`: 执行结果，包含以下字段：
  - `success` (bool): 是否成功
  - `completed_stages` (List[str]): 已完成的阶段列表
  - `failed_stages` (List[str]): 失败的阶段列表
  - `summary` (str): 执行摘要
  - `context` (dict): 上下文对象

**示例：**

```python
result = engine.run_pipeline(user_input="帮我做一个回测")
print(result['success'])
print(result['summary'])
```

## 标准入口函数

### `run(ctx: Context = None, user_input: str = None) -> dict`

Skill 标准入口函数，所有 Skill 都应该实现此接口。

**参数：**
- `ctx` (Context, optional): 上下文对象
- `user_input` (str, optional): 用户自然语言

**返回：**
- `dict`: 执行结果

**示例：**

```python
from engine import run

# 使用自然语言
result = run(user_input="帮我做一个回测")

# 使用 Context
ctx = Context(...)
result = run(ctx=ctx)
```

## Context 类

上下文对象，用于在各个阶段之间传递状态。

### 属性

| 属性名 | 类型 | 描述 |
|--------|------|------|
| task_id | str | 任务ID |
| user_intent | str | 用户原始意图 |
| current_stage | str | 当前阶段 |
| target_stages | List[str] | 目标阶段列表 |
| stock_pool | List[str] | 股票池 |
| start_date | str | 开始日期 |
| end_date | str | 结束日期 |
| artifacts | Dict[str, str] | 各阶段产物路径 |
| metadata | Dict[str, Any] | 各阶段元数据 |
| errors | List[str] | 错误记录 |

### 方法

#### `update_artifact(stage: str, path: str)`

更新阶段产物路径。

```python
ctx.update_artifact("DATA", "/path/to/data.parquet")
```

#### `add_error(error: str)`

添加错误记录。

```python
ctx.add_error("数据获取失败：网络错误")
```

#### `to_dict() -> dict`

转换为字典格式。

```python
data = ctx.to_dict()
```

#### `from_json(json_str: str) -> Context`

从 JSON 字符串创建 Context 对象。

```python
ctx = Context.from_json(json_str)
```

## 常量

### STAGES

阶段列表：`["IDLE", "DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"]`

### STAGE_ORDER

阶段顺序映射：
```python
{
    "DATA": 1,
    "FACTOR": 2,
    "MODEL": 3,
    "BACKTEST": 4,
    "PORTFOLIO": 5,
    "EXECUTION": 6,
    "REPORT": 7,
}
```

### SKILL_MODULES

子 Skill 模块映射：
```python
{
    "DATA": "a_share_data_engine.scripts",
    "FACTOR": "a_share_factor_engine.scripts",
    "MODEL": "strategy_model_engine.scripts",
    "BACKTEST": "backtest_engine.scripts",
    "PORTFOLIO": "portfolio_risk_engine.scripts",
    "EXECUTION": "execution_monitor_engine.scripts",
    "REPORT": "reports_engine.scripts",
}
```

### EXPECTED_ARTIFACTS

各阶段预期产物文件：
```python
{
    "DATA": "cleaned_data.parquet",
    "FACTOR": "factor_data.parquet",
    "MODEL": "model.pkl",
    "BACKTEST": "backtest_result.json",
    "PORTFOLIO": "portfolio_weights.json",
    "EXECUTION": "trade_log.json",
    "REPORT": "report.html",
}
```

## 异常处理

所有方法都会捕获异常并记录到 `Context.errors` 中。如果执行失败，请检查：

1. `result['errors']` - 错误列表
2. `ctx.errors` - Context 中的错误记录
3. 日志文件 - 位于 `workspace/logs/` 目录

## 示例代码

### 完整流程

```python
from engine import run, MasterEngine
from context import Context

# 方式1：使用自然语言
result = run(user_input="帮我用近3年A股数据做一个20日反转因子选股回测")

# 方式2：使用 Context
ctx = Context(
    task_id="task_001",
    user_intent="帮我做一个回测",
    current_stage="IDLE",
    target_stages=["DATA", "BACKTEST", "REPORT"]
)
result = run(ctx=ctx)

# 检查结果
if result['success']:
    print("执行成功！")
    print(result['summary'])
else:
    print("执行失败：", result.get('errors'))
```

### CLI 使用

```bash
# 基础使用
python engine.py -i "帮我做一个回测"

# 保存结果
python engine.py -i "帮我做一个回测" -o result.json

# 使用已有的 Context
python engine.py -c context.json -i "继续执行"
```
