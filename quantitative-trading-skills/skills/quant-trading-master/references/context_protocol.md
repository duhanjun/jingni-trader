
# Context对象协议规范

## 概述

本文档定义了主Skill与子Skill之间传递Context对象的标准协议，确保各Skill之间能够正确通信和协作。

## Context对象字段说明

| 字段名 | 类型 | 是否必填 | 说明 |
|--------|------|----------|------|
| task_id | str | 是 | 当前任务唯一标识符 |
| session_id | str | 是 | 当前会话唯一标识符 |
| stock_pool | List[str] | 是 | 股票池，股票代码列表 |
| time_range | dict | 是 | 时间范围，包含 start_date 和 end_date |
| config | dict | 否 | 全局配置字典，包含各种配置项 |
| artifacts | dict | 否 | 已完成阶段产物路径映射 |
| current_stage | str/Stage | 否 | 当前所处阶段 |
| stage_history | List[str/Stage] | 否 | 已完成阶段历史记录 |
| results | dict | 否 | 各阶段执行结果存储 |

## 时间范围格式

```python
time_range = {
    "start_date": "2020-01-01",  # YYYY-MM-DD 格式
    "end_date": "2024-01-01"
}
```

## 股票代码格式

- 上海证券交易所：`600000.SH`
- 深圳证券交易所：`000001.SZ`

## 配置项规范

### 常用配置项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| data_backend | str | "tushare" | 数据源后端（tushare/baostock/akshare） |
| backtest_backend | str | "rqalpha" | 回测引擎后端（rqalpha/backtrader/gm） |
| trade_backend | str | "simulation" | 交易后端（simulation/xtquant/gm） |
| data_dir | str | "./data" | 数据存储目录 |
| log_level | str | "INFO" | 日志级别 |

## 产物路径规范

artifacts字典结构：

```python
artifacts = {
    "data_file": "/path/to/data.parquet",
    "factor_file": "/path/to/factors.parquet",
    "model_file": "/path/to/model.pkl",
    "backtest_report": "/path/to/backtest.md",
    "portfolio_config": "/path/to/portfolio.json",
    "trading_log": "/path/to/trading.log",
    "final_report": "/path/to/report.md"
}
```

## 阶段枚举值

```
"数据获取"
"因子构建"
"模型训练"
"回测验证"
"组合优化"
"模拟／实盘"
"绩效报告"
```

## 结果字典结构

results字典以阶段名为键，存储该阶段的执行结果：

```python
results = {
    "数据获取": {
        "success": True,
        "data_file": "/path/to/data.parquet",
        "message": "数据获取完成"
    },
    "因子构建": {
        "success": True,
        "factor_file": "/path/to/factors.parquet",
        "message": "因子构建完成",
        "factors": [...]
    },
    ...
}
```

## 序列化与反序列化

Context对象支持JSON序列化：

```python
# 序列化
ctx_dict = ctx.to_dict()
json_str = json.dumps(ctx_dict)

# 反序列化
ctx_dict = json.loads(json_str)
ctx = QuantTradingContext.from_dict(ctx_dict)
```

## 子Skill调用约定

每个子Skill应提供统一的入口函数：

```python
def run(ctx: QuantTradingContext) -&gt; Dict[str, Any]:
    """
    子Skill入口函数

    Args:
        ctx: 上下文对象

    Returns:
        结果字典，包含:
            - success: bool
            - (其他结果字段)
    """
    # 实现逻辑
    pass
```

## 错误处理约定

子Skill出错时应返回：

```python
{
    "success": False,
    "error": "错误描述",
    "error_code": "ERROR_CODE",
    "retryable": True  # 是否可以重试
}
```

## 检查点保存与加载

```python
from scripts.main_workflow import save_checkpoint, load_checkpoint

# 保存检查点
checkpoint_path = save_checkpoint(ctx)

# 从检查点恢复
ctx = load_checkpoint(checkpoint_path)
```
