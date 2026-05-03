# 量化交易主引擎 (Quant Trading Master)

量化交易Skill套件的协调中枢，负责管理任务状态机，调度子Skill完成A股量化投研全流程。

## 功能特性

- **意图解析**: 判断当前处于量化投研的哪个阶段
- **状态机管理**: 管理7个阶段的完整工作流
- **子Skill调度**: 按顺序调用各子Skill完成工作
- **上下文管理**: 维护任务状态和数据流转
- **里程碑检查**: 每个阶段完成后验证结果完整性
- **断点续传**: 支持从检查点恢复执行

## 阶段状态机

```
数据获取 → 因子构建 → 模型训练 → 回测验证 → 组合优化 → 模拟／实盘 → 绩效报告
```

### 分支逻辑

- 回测失败 → 返回因子调优
- 模型过拟合 → 样本外再验证

## 快速开始

### 安装依赖

```bash
cd ../..
pip install -r requirements.txt
```

### 基础使用

```python
from scripts.main_workflow import run, QuantTradingContext

# 创建上下文
ctx = QuantTradingContext(
    task_id="my_task_001",
    session_id="my_session_001",
    stock_pool=["000001.SZ", "000002.SZ", "600000.SH"],
    time_range={"start_date": "2020-01-01", "end_date": "2024-01-01"},
    config={"data_backend": "baostock", "backtest_backend": "backtrader"}
)

# 运行主流程
result = run(ctx)

# 查看结果
print(f"完成阶段: {result['completed_stages']}")
print(f"最终报告: {result['final_report']}")
```

### 从断点继续

```python
# 加载已有上下文
ctx = load_context("checkpoints/my_task_001.json")

# 继续执行
result = run(ctx, resume=True)
```

## 项目结构

```
quant-trading-master/
├── SKILL.md              # Skill 描述文件
├── README.md             # 项目说明
├── __init__.py           # 包入口
├── scripts/              # 脚本
│   └── main_workflow.py  # 主工作流脚本
└── references/           # 文档
    ├── context_protocol.md
    └── workflow_architecture.md
```

## 子Skill映射

| 阶段 | 对应子Skill |
|------|------------|
| 数据获取 | a-share-data-engine |
| 因子构建 | a-share-factor-engine |
| 模型训练 | strategy-model-engine |
| 回测验证 | backtest-engine |
| 组合优化 | portfolio-risk-engine |
| 模拟／实盘 | execution-monitor-engine |
| 绩效报告 | reports-engine |

## Context对象

Context对象标准化定义，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| task_id | str | 当前任务ID |
| session_id | str | 当前会话ID |
| stock_pool | List[str] | 股票池（股票代码列表） |
| time_range | dict | 时间范围 {start_date, end_date} |
| artifacts | dict | 已完成阶段产物路径 |
| config | dict | 全局配置 |
| current_stage | str | 当前所处阶段 |
| stage_history | List[str] | 已完成阶段历史 |
| results | dict | 各阶段执行结果 |

详见 [references/context_protocol.md](references/context_protocol.md)

## 里程碑检查点

每个子Skill完成后自动检查：
- 产物完整性
- 基本合理性
- 失败时给出清晰错误码
- 支持从断点重试

## 文档

- [Context协议](references/context_protocol.md)
- [工作流架构](references/workflow_architecture.md)

## 许可证

MIT License
