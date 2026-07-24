---
name: jingni-trader
version: 1.0.0
description: A股量化交易全流程主调度器。负责解析用户意图，管理投研阶段状态机，维护跨 Skill 的上下文对象，按流程依次调度七个子 Skill 完成从数据采集到绩效报告的全链路工作。本身不执行任何量化计算，只做编排。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - master-skill
  - workflow
  - 量化
  - 调度器
dependencies:
  - importlib (Python 标准库)
  - logging (Python 标准库)
  - json (Python 标准库)
environment_variables:
  - name: TUSHARE_TOKEN
    description: Tushare Pro API Token，用于获取A股数据
    required: false
  - name: GM_TOKEN
    description: 掘金量化API Token，用于实盘交易
    required: false
  - name: QUANT_WORK_DIR
    description: 数据和工作目录
    required: false
    default: "./workspace"
  - name: LOG_LEVEL
    description: 日志级别
    required: false
    default: "INFO"
language: python
python_version: "3.9+"
entry_point: engine.py
allowed_sub_skills:
  - data-engine
  - factor-engine
  - strategy-model-engine
  - backtest-engine
  - portfolio-risk-engine
  - execution-monitor-engine
  - reports-engine
included_skills:
  - skills/data-engine
  - skills/factor-engine
  - skills/strategy-model-engine
  - skills/backtest-engine
  - skills/portfolio-risk-engine
  - skills/execution-monitor-engine
  - skills/reports-engine
  - skills/jingni-datafeed
trigger_keywords:
  - 量化
  - 回测
  - 选股
  - 因子
  - 实盘
  - 组合优化
  - A股
  - 策略开发
---

# jingni-trader

## 概述

jingni-trader 是量化交易 Skill 套件的**主协调中枢**，负责：

1. 解析用户自然语言意图，判断当前投研阶段
2. 管理任务状态机，按流程调度子 Skill
3. 维护会话状态和任务上下文
4. 输出结构化的量化研究报告

## ⚠️ 数据源优先级策略

**在获取金融数据时，必须按以下优先级选择数据源：**

1. **Agent 系统内置工具（最高优先级）**
   - 检查当前 Agent 系统是否提供金融数据获取工具（如 MCP 连接器、系统内置插件、其他 Skill）
   - 常见系统内置工具类型：MCP Server（如 tushare MCP、akshare MCP）、系统级 market-data Skill、数据插件
   - 如果有系统内置工具可用，**优先使用系统内置工具获取数据**，避免重复安装依赖
   - 获取数据后通过 Context 对象的 `external_data` 字段传递给 data-engine

2. **环境变量配置的数据源（第二优先级）**
   - 如果系统内置工具的 Token 未配置或数据获取失败，回退到环境变量 `TUSHARE_TOKEN`、`GM_TOKEN` 等
   - 使用 data-engine 内置的适配器（tushare、baostock、akshare、xtquant、gm）获取数据

3. **免费离线数据源（兜底方案）**
   - BaoStock（无需 Token）
   - AkShare（无需 Token）

**数据源回退流程：**
```
系统内置工具 → 环境变量数据源 → 免费离线数据源
    ↓ 失败          ↓ 失败            ↓ 失败
  回退下一级      回退下一级        报错退出
```

## 运行归档机制

**每次运行完整流程时，自动创建归档目录保存所有过程和结果：**

- 在 workspace/archives/ 下创建 `YYYYMMDD_HHMMSS` 格式的运行归档目录
- 每个子任务在归档目录中创建 `step_N_<阶段名>` 格式的子文件夹
- 每个步骤保存 `summary.md` 子任务小结报告
- 全景汇总 `pipeline_summary.md` 保存在归档根目录

归档目录结构：
```
workspace/archives/20260529_143025/
├── pipeline_summary.md
├── step_1_DATA/
│   ├── summary.md
│   └── artifacts/
├── step_2_FACTOR/
│   ├── summary.md
│   └── artifacts/
├── step_3_MODEL/
│   ├── summary.md
│   └── artifacts/
...
```

## 阶段状态机

```
[数据获取] → [因子构建] → [模型训练] → [回测验证] → [组合优化] → [模拟/实盘] → [绩效报告]
```

### 分支逻辑

- 回测失败 → 返回因子调优
- 模型过拟合 → 触发样本外再验证

## Context 对象

标准化的上下文对象，包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| task_id | str | 当前任务ID（YYYYMMDDHHMMSS） |
| session_id | str | 会话ID |
| user_intent | str | 用户原始意图 |
| current_stage | str | 当前所处阶段 |
| target_stages | List[str] | 目标阶段列表 |
| stock_pool | List[str] | 股票池（股票代码列表，空列表=全市场） |
| benchmark | str | 基准指数代码（默认 000300.SH） |
| start_date | str | 开始日期 |
| end_date | str | 结束日期 |
| strategy_name | str | 策略名称 |
| strategy_params | Dict[str, Any] | 策略参数字典 |
| artifacts | Dict[str, str] | 已完成阶段产物路径 |
| external_data | Dict[str, Any] | 系统内置工具传入的外部数据 |
| run_dir | str | 当前运行归档目录路径 |
| step_dirs | Dict[str, str] | 各步骤归档子目录路径 |
| metadata | Dict[str, Any] | 各阶段元数据 |
| errors | List[str] | 错误记录 |

## 使用示例

### Python API

```python
from engine import run, MasterEngine
from context import Context

# 创建 Context
ctx = Context(
    task_id="task_001",
    user_intent="帮我用近3年A股数据做一个20日反转因子选股回测",
    current_stage="IDLE"
)

# 运行主流程
result = run(ctx)
print(result)
```

### CLI 运行

```bash
# 交互式输入
python engine.py -i "帮我用近3年A股数据做一个20日反转因子选股回测"

# 指定参数
python engine.py --task-id test001 --stock-pool 000001.SZ,600000.SH --start-date 2021-01-01 --end-date 2024-01-01

# 仅生成报告
python engine.py -i "生成上个月实盘绩效报告"
```

## 子 Skill 映射

| 阶段 | 对应子 Skill |
|------|-------------|
| DATA | data-engine |
| FACTOR | factor-engine |
| MODEL | strategy-model-engine |
| BACKTEST | backtest-engine |
| PORTFOLIO | portfolio-risk-engine |
| EXECUTION | execution-monitor-engine |
| REPORT | reports-engine |

## jingni-datafeed 自动部署

jingni-trader 可选依赖 `jingni-datafeed`（惊泥因子库 datafeed 服务），该子 skill 独立维护在 [duhanjun/jingni-datafeed](https://github.com/duhanjun/jingni-datafeed)。

**启动时自动检测**：MasterEngine 实例化时自动检查 `skills/jingni-datafeed/` 目录：
- **目录不存在** → 自动从 GitHub 克隆（`git clone --depth 1`），用户无需手动操作
- **目录已存在** → 运行正常的版本检查（只检测落后、不自动修改文件）

自动克隆失败时**不会阻断主流程**，仅输出警告日志。如需手动安装：

```bash
cd jingni-trader
git clone https://github.com/duhanjun/jingni-datafeed.git skills/jingni-datafeed
cp skills/jingni-datafeed/.env.example skills/jingni-datafeed/.env
# 编辑 skills/jingni-datafeed/.env 填入 JINGNI_URL / JINGNI_TOKEN
```

## 里程碑检查点

每个子 Skill 完成后自动检查：

- 产物完整性
- 基本合理性
- 失败时给出清晰错误码
- 支持从断点重试

## 错误处理

- 所有子 Skill 调用包含异常捕获
- 明确的错误信息和建议
- 优雅降级策略

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)

## Context 协议

详见 [references/context_protocol.md](references/context_protocol.md)

## 工作流架构

详见 [references/workflow_architecture.md](references/workflow_architecture.md)