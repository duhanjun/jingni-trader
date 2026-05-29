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

- Python 3.9+
- pip

### 安装方式

#### 方式一：Agent 系统直接导入（推荐）

从 GitHub 下载 ZIP 压缩包，直接导入 Agent 系统：

```bash
# 下载 ZIP 并解压
wget https://github.com/duhanjun/jingni-trader/archive/refs/heads/main.zip
unzip main.zip
```

解压后 `SKILL.md` 位于根目录，Agent 系统自动识别。

#### 方式二：使用安装脚本

```bash
bash install.sh
```

#### 方式三：手动安装依赖

```bash
pip install tushare baostock akshare pandas numpy
pip install pandas-ta talib scipy statsmodels
pip install lightgbm scikit-learn
pip install rqalpha backtrader
pip install PyPortfolioOpt riskfolio-lib
pip install quantstats plotly matplotlib
```

### 运行示例

#### CLI 命令行

```bash
# 数据采集 + 因子构建 + 回测 + 报告
python3 engine.py -i "帮我用近3年A股数据做一个20日反转因子选股回测"

# 仅生成报告
python3 engine.py -i "生成上个月实盘绩效报告"

# 组合优化
python3 engine.py -i "优化当前组合，最大回撤控制在15%以内"
```

#### Python API

```python
from engine import run
from scripts.context import Context

# 创建 Context
ctx = Context(
    task_id="task_001",
    user_intent="帮我用近3年A股数据做一个20日反转因子选股回测",
    current_stage="IDLE"
)

# 运行主流程
result = run(ctx=ctx)
print(result)
```

## 项目结构

```
├── engine.py                    # 主调度引擎（标准入口）
├── SKILL.md                     # Agent Skills 标准技能描述文件
├── scripts/
│   ├── __init__.py
│   ├── context.py               # 上下文对象（跨阶段状态传递）
│   └── config.py                # 全局配置（路径、风控、参数）
├── skills/                      # 7个子技能目录
│   ├── data-engine/             # 数据采集引擎
│   ├── factor-engine/           # 因子计算引擎
│   ├── strategy-model-engine/   # 模型训练引擎
│   ├── backtest-engine/         # 回测引擎
│   ├── portfolio-risk-engine/   # 组合优化引擎
│   ├── execution-monitor-engine/ # 执行监控引擎
│   └── reports-engine/          # 报告生成引擎
├── references/                  # 参考文档
│   ├── api_reference.md
│   ├── config_guide.md
│   ├── context_protocol.md
│   ├── context_schema.md
│   └── workflow_architecture.md
├── workspace/                   # 运行时工作目录（自动创建，.gitignore）
│   ├── data/                    # 清洗后数据
│   ├── factors/                 # 因子数据
│   ├── models/                  # 训练模型
│   ├── backtest_results/        # 回测结果
│   ├── portfolio/               # 组合配置
│   ├── reports/                 # 生成报告
│   └── logs/                    # 运行日志
├── install.sh                   # 安装脚本
├── LICENSE
└── README.md
```

## 引擎介绍

### 1. 数据采集引擎 (data-engine)

**职责**：从多数据源获取A股行情数据，完成复权、涨跌停标记、ST过滤、新股剔除、停牌处理等本土化清洗。

**支持数据源**：
- Tushare Pro
- BaoStock
- AkShare
- xtquant（需券商渠道）
- 掘金量化（gm）

**输出产物**：`cleaned_data.parquet`

### 2. 因子计算引擎 (factor-engine)

**职责**：计算 Alpha 因子、技术指标，完成因子 IC 分析（含行业中性化）、分层回测、相关性去冗余、多因子融合。

**支持计算库**：
- pandas-ta
- TA-Lib

**输出产物**：`factor_data.parquet`

### 3. 模型训练引擎 (strategy-model-engine)

**职责**：基于机器学习模型进行选股预测，支持 LightGBM、CatBoost、XGBoost 等主流算法，集成 Optuna 超参优化。

**输出产物**：`model.pkl`

### 4. 回测引擎 (backtest-engine)

**职责**：在历史数据上验证策略效果，计算收益率、夏普比、最大回撤、Calmar 比率等指标。

**支持回测框架**：
- RQAlpha
- Backtrader
- 掘金量化（gm）

**输出产物**：`backtest_result.json`

### 5. 组合优化引擎 (portfolio-risk-engine)

**职责**：基于量化模型输出，构建最优持仓组合，进行风险预算分配，支持 HRP、均值方差、Black-Litterman 等方法。

**输出产物**：`portfolio_weights.json`

### 6. 执行监控引擎 (execution-monitor-engine)

**职责**：对接券商接口，执行交易指令，实时监控仓位和盈亏，支持模拟/纸交易/实盘三种模式。

**支持交易接口**：
- 模拟交易（paper）
- xtquant（需券商渠道）
- 掘金量化（gm）

**输出产物**：`trade_log.json`

### 7. 报告生成引擎 (reports-engine)

**职责**：汇总各阶段产物，生成可视化的 HTML 绩效报告，含收益归因、因子收益热力图、交易明细等。

**输出产物**：`report.html`

## 配置说明

### 环境变量

| 变量名 | 说明 | 必需 |
|--------|------|------|
| `TUSHARE_TOKEN` | Tushare Pro API Token | 否 |
| `GM_TOKEN` | 掘金量化 API Token | 否 |
| `QUANT_WORK_DIR` | 工作目录（默认 `./workspace`） | 否 |
| `LOG_LEVEL` | 日志级别（默认 `INFO`） | 否 |

### 配置文件

配置文件位于 `~/.quant-trading/config.yaml`，示例：

```yaml
# 数据源配置
data_source:
  default: "tushare"
  tushare:
    token: "your_token_here"

# 回测配置
backtest:
  default: "rqalpha"
  commission: 0.0003
  slippage: 0.0001

# 执行配置
execution:
  default: "xtquant"
  mode: "paper"

# 风控配置
risk:
  max_position: 0.05
  max_loss_per_day: 0.02
```

详见 [references/config_guide.md](references/config_guide.md)

## Context 对象

各引擎通过 Context 对象共享状态：

```python
@dataclass
class Context:
    task_id: str                    # 任务ID
    user_intent: str                # 用户原始意图
    current_stage: str              # 当前阶段
    target_stages: List[str]        # 目标阶段列表
    stock_pool: List[str]           # 股票池
    start_date: str                 # 开始日期
    end_date: str                   # 结束日期
    artifacts: Dict[str, str]       # 各阶段产物路径
    metadata: Dict[str, Any]        # 各阶段元数据
    errors: List[str]               # 错误记录
```

详见 [references/context_protocol.md](references/context_protocol.md)

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

- **数据处理**：pandas, numpy, pyarrow
- **数据源**：tushare, baostock, akshare
- **因子计算**：pandas-ta, TA-Lib
- **机器学习**：lightgbm, catboost, scikit-learn, optuna
- **回测框架**：rqalpha, backtrader
- **组合优化**：PyPortfolioOpt, riskfolio-lib, cvxpy
- **可视化**：quantstats, plotly, matplotlib, seaborn

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 联系方式

- 项目主页：https://github.com/duhanjun/jingni-trader
- 问题反馈：https://github.com/duhanjun/jingni-trader/issues
