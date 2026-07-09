# jingni-trader

A股量化交易全流程智能调度系统。基于大语言模型的量化投研工作流自动化引擎，通过自然语言交互即可完成从数据采集、因子构建、模型训练、回测验证到绩效报告的全链路工作。

## 核心特性

- 🧠 **自然语言驱动**：用户用中文描述需求，系统自动解析意图并执行对应投研流程
- 🔄 **断点续跑**：每个阶段的产物独立存储，已完成的阶段自动跳过
- 🛡️ **硬风控机制**：内置单日亏损限制、单笔金额上限等风险控制断路器
- 🔌 **模块化架构**：7个独立子引擎，支持按需组合和扩展
- 📊 **阶段状态机**：DATA → FACTOR → MODEL → BACKTEST → PORTFOLIO → EXECUTION → REPORT

## 系统架构

```
用户输入（自然语言）
      ↓
[ jingni-trader 主调度器 ]
      ↓
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ DATA│FACTOR│MODEL│BACK │PORT │EXEC │REPORT│
│     │     │     │TEST │FOLIO│     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┘
      ↓        ↓        ↓        ↓        ↓
[数据采集] [因子构建] [模型训练] [组合优化] [绩效报告]
```

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 数据采集 + 因子构建 + 回测 + 报告
python3 engine.py -i "帮我用近3年A股数据做一个20日反转因子选股回测"

# 仅生成报告
python3 engine.py -i "生成上个月实盘绩效报告"

# 组合优化
python3 engine.py -i "优化当前组合，最大回撤控制在15%以内"
```

## 项目结构

```
├── engine.py                    # 主调度引擎
├── SKILL.md                     # 主技能描述文件
├── requirements.txt             # Python 依赖
├── scripts/
│   ├── __init__.py
│   ├── context.py              # 上下文对象
│   ├── config.py               # 全局配置
│   └── archive.py              # 运行归档
├── skills/                      # 子技能目录
│   ├── data-engine/            # 数据采集引擎
│   ├── factor-engine/          # 因子计算引擎
│   ├── strategy-model-engine/  # 模型训练引擎
│   ├── backtest-engine/        # 回测引擎
│   ├── portfolio-risk-engine/  # 组合优化引擎
│   ├── execution-monitor-engine/ # 执行监控引擎
│   └── reports-engine/         # 报告生成引擎
├── references/                  # 参考文档
├── tests/                        # 集成测试
│   └── test_integration_e2e.py
└── README.md
```

## 引擎介绍

### 1. 数据采集引擎 (data-engine)

**职责**：从多数据源获取A股行情数据，完成复权、涨跌停标记、ST过滤等清洗工作。

**支持数据源**：
- Tushare Pro
- BaoStock
- AkShare
- xtquant（需券商渠道）
- 掘金量化（gm）

**输出产物**：`cleaned_data.parquet`

**数据源依赖自动安装**：

当某数据源适配器所需的第三方库尚未安装时，data-engine 不会直接跳过该数据源，而是先尝试用当前 Python 解释器自动安装依赖，安装成功后再加载并使用该数据源；仅当自动安装失败（如网络受限）时才会按降级链跳到下一个数据源。

- 开关：`AUTO_INSTALL_BACKENDS`（默认 `true`）。设为 `false` 可恢复旧行为（缺依赖即跳过）。
- 后端与 pip 包映射（见 `skills/data-engine/scripts/config.py` 的 `BACKEND_PIP_PACKAGES`）：
  - `tushare → tushare`
  - `baostock → baostock`
  - `akshare → akshare`
  - `xtquant → xtquant`
  - `gm → gm`
  - `tdxquant → tdxquant, pytdx`
  - `websearch` 无第三方依赖
- 自动安装结果会被缓存（`_INSTALL_CACHE`），避免在同一次运行的降级链里重复安装。
- 注意：`xtquant` / `gm` / `tdxquant` 即便 pip 包装好，仍需本地客户端或 Token 才能真正取到数据（属于运行时环境依赖，非安装问题）。

### 2. 因子计算引擎 (factor-engine)

**职责**：计算 Alpha 因子、技术指标，完成因子 IC 分析和预筛选。

**支持计算库**：
- pandas-ta
- TA-Lib

**输出产物**：`factor_data.parquet`

### 3. 模型训练引擎 (strategy-model-engine)

**职责**：基于机器学习模型进行选股预测，支持 LightGBM、CatBoost 等主流算法。

**输出产物**：`model.pkl`

### 4. 回测引擎 (backtest-engine)

**职责**：在历史数据上验证策略效果，计算收益率、夏普比、最大回撤等指标。

**支持回测框架**：
- 内置原生回测（native）— ✅ 生产可用
- RQAlpha — ⚠️ 预留/示例桩（仅返回模拟或空实现，非真实回测结果）
- Backtrader — ⚠️ 预留/示例桩（仅返回模拟或空实现，非真实回测结果）
- 掘金量化（gm）— ⚠️ 预留/示例桩（需本地 gm 客户端与 Token，未接入真实回测引擎）

> 生产可用状态（GAP-2/3 标注）：**仅 `backtest=native` 为真实回测实现**；rqalpha / backtrader / gm 适配器当前为预留/示例桩，运行它们不会产出真实历史回测结论，请勿据此下单或做投资决策。

**输出产物**：`backtest_result.json`

### 5. 组合优化引擎 (portfolio-risk-engine)

**职责**：基于量化模型输出，构建最优持仓组合，进行风险预算分配。

**输出产物**：`portfolio_weights.json`

### 6. 执行监控引擎 (execution-monitor-engine)

**职责**：对接券商接口，执行交易指令，实时监控仓位和盈亏。

**支持交易接口**：
- 模拟交易（paper）— ✅ 生产可用
- xtquant（需券商渠道）— ⚠️ 预留/示例桩（需本地券商客户端与 Token，未接入真实交易通道）
- 掘金量化（gm）— ⚠️ 预留/示例桩（需本地 gm 客户端与 Token，未接入真实交易通道）

> 生产可用状态（GAP-2/3 标注）：**仅 `execution=paper`（模拟交易）为真实可用**；xtquant / gm 执行接口当前为预留/示例桩，不连接真实券商通道，请勿用于实盘下单。

**输出产物**：`trade_log.json`

### 7. 报告生成引擎 (reports-engine)

**职责**：汇总各阶段产物，生成可视化的 HTML 绩效报告。

**输出产物**：`report.html`

## 配置说明

项目通过环境变量进行配置，无需额外配置文件。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QUANT_WORK_DIR` | `./workspace` | 工作目录（数据、产物、日志的根目录） |
| `DATA_BACKENDS` | `tushare,baostock,akshare,websearch` | 数据降级链顺序 |
| `DATA_BACKEND` | 无 | 单源模式（不降级） |
| `TUSHARE_TOKEN` / `GM_TOKEN` | 无 | API 令牌 |
| `ADJUST_MODE` | `hfq` | 复权方式（前复权 `qfq` / 后复权 `hfq`） |
| `ALLOW_SYNTHETIC_FALLBACK` | `true` | 全部数据源失败时生成模拟数据兜底 |
| `AUTO_INSTALL_BACKENDS` | `true` | 数据源依赖缺失时自动 `pip install` 后重试 |
| `DATA_FORMAT` | `parquet` | 数据落盘格式（csv/sql） |
| `DATA_MAX_WORKERS` | `4` | 并行下载线程数 |
| `BACKTEST_BACKEND` | `native` | 回测后端 |
| `FACTOR_BACKEND` | `pandas_ta` | 因子计算后端 |
| `TRADE_BACKEND` | `xtquant` | 交易后端 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## Context 对象

各引擎通过 Context 对象共享状态，包含以下字段：

```python
@dataclass
class Context:
    # 任务标识
    task_id: str                     # 任务ID（YYYYMMDDHHMMSS）
    session_id: str                  # 会话ID

    # 用户意图
    user_intent: str                 # 用户原始意图
    current_stage: str               # 当前阶段
    target_stages: List[str]         # 目标阶段列表

    # 股票与时间
    stock_pool: List[str]            # 股票池（空列表=全市场）
    benchmark: str                   # 基准指数（默认 000300.SH）
    start_date: str                  # 开始日期
    end_date: str                    # 结束日期

    # 策略参数
    strategy_name: str               # 策略名称
    strategy_params: Dict[str, Any]  # 策略参数字典

    # 产物与外部数据
    artifacts: Dict[str, str]         # 各阶段产物路径
    external_data: Dict[str, Any]    # 系统内置工具传入的外部数据

    # 运行归档
    run_dir: str                     # 运行归档目录路径
    step_dirs: Dict[str, str]        # 各步骤归档子目录路径

    # 元信息与错误
    metadata: Dict[str, Any]         # 各阶段元数据
    errors: List[str]                # 错误记录
```

## 开发指南

### 添加新的数据源适配器

1. 在 `data-engine/scripts/adapters/` 下创建新适配器
2. 继承 `BaseDataProvider` 类
3. 实现 `fetch_daily` 和 `fetch_minute` 方法
4. 在 `config.py` 中注册新的适配器

### 添加新的回测框架

1. 在 `backtest-engine/scripts/adapters/` 下创建新适配器
2. 继承 `BaseBacktest` 类
3. 实现 `run_backtest` 方法
4. 在 `config.py` 中注册新的框架

## 技术栈

- **数据处理**：pandas, numpy, scipy, pyarrow
- **数据源**：tushare, baostock, akshare
- **因子计算**：pandas-ta, TA-Lib
- **机器学习**：lightgbm, catboost, scikit-learn
- **回测框架**：native（内置原生回测）、vectorbt（可选）
- **组合优化**：PyPortfolioOpt, riskfolio-lib, cvxpy
- **可视化**：plotly, matplotlib, quantstats

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 联系方式

- 项目主页：https://github.com/duhanjun/jingni-trader
- 问题反馈：https://github.com/duhanjun/jingni-trader/issues
