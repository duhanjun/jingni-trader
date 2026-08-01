# jingni-trader

A股量化交易全流程智能调度系统。基于大语言模型的量化投研工作流自动化引擎，通过自然语言交互即可完成从数据采集、因子构建、模型训练、回测验证到绩效报告的全链路工作。

## 核心特性

- **自然语言驱动**：用户用中文描述需求，系统自动解析意图并执行对应投研流程
- **统一意图解析**：自动识别用户意图，走量化策略或个股分析路径，按报告模板生成对应类型报告
- **断点续跑**：每个阶段的产物独立存储，已完成的阶段自动跳过
- **硬风控机制**：内置单日亏损限制、单笔金额上限等风险控制断路器
- **模块化架构**：7个独立子引擎，支持按需组合和扩展
- **多数据源降级**：支持 9 种数据源，用户可通过对话切换优先级，含精准降级与模拟数据兜底
- **运行归档**：每次运行自动创建时间戳归档目录，保存全部过程和产物

## 意图解析与路由

系统根据用户意图自动路由到不同的分析路径：

| 用户意图 | 触发关键词 | 阶段路径 | 报告类型 |
|------|-----------|---------|---------|
| **量化策略** | 回测/因子/策略/模型/组合/实盘 | DATA → FACTOR → MODEL → BACKTEST → PORTFOLIO → EXECUTION → REPORT | 量化策略绩效报告 |
| **个股分析** | 分析/怎么样/技术面/基本面/K线/诊股 | DATA → FACTOR → REPORT | 个股技术面 + 基本面深度分析报告 |

个股分析报告支持三种模板：`technical`（技术面）、`fundamental`（基本面）、`both`（同时生成两份报告，默认）。

## 系统架构

```
用户输入（自然语言）
      ↓
[ jingni-trader 主调度器（意图解析 + 统一路由）]
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

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 量化投资者：数据采集 + 因子构建 + 回测 + 报告
python engine.py -i "帮我用近3年A股数据做一个20日反转因子选股回测"

# 仅生成报告
python engine.py -i "生成上个月实盘绩效报告"

# 组合优化
python engine.py -i "优化当前组合，最大回撤控制在15%以内"

# 个股分析：技术面 + 基本面
python engine.py -i "分析 002594.SZ 比亚迪的技术面和基本面"

# 指定数据源优先级（通过对话即可切换）
python engine.py -i "用 wind 取数据，分析 000001.SZ 平安银行"
```

## 项目结构

```
├── engine.py                     # 主调度引擎（意图解析 + 统一路由 + LLM 注入）
├── SKILL.md                      # 主技能描述文件
├── README.md                     # 项目说明（本文件）
├── requirements.txt              # Python 依赖
├── skill-sync.yml                # Skill 版本同步配置
├── LICENSE                       # MIT 许可证
├── scripts/
│   ├── __init__.py
│   ├── context.py               # Context 上下文对象
│   ├── config.py                # 全局配置（路径/后端/风控阈值）
│   ├── archive.py               # 运行归档（时间戳目录 + 步骤小结）
│   └── skill_sync.py            # GitHub 版本检查与自动部署
├── skills/                       # 7 个子技能目录
│   ├── data-engine/             # 数据采集引擎（9 种数据源 + 精准降级）
│   ├── factor-engine/           # 因子计算引擎（表达式引擎 + Alpha158 + 惊泥因子库）
│   ├── strategy-model-engine/   # 模型训练引擎（LightGBM/CatBoost + Optuna）
│   ├── backtest-engine/         # 回测引擎（Native + 增强回测 + Walk-Forward）
│   ├── portfolio-risk-engine/   # 组合优化引擎（HRP + 风控断路器 + Walk-Forward）
│   ├── execution-monitor-engine/ # 执行监控引擎（Paper/Live + 量化断路器）
│   └── reports-engine/          # 报告生成引擎（统一路由 + K线图 + LLM 注入）
├── references/                   # 参考文档
│   ├── api_reference.md
│   ├── config_guide.md
│   ├── context_protocol.md
│   └── workflow_architecture.md
├── tests/                        # 测试套件（按子 Skill 边界组织，三级测试体系）
│   ├── conftest.py              # 全局 fixture 与 sys.path 配置
│   ├── pytest.ini               # 标记配置（contract/unit/integration + skill_*）
│   ├── fixtures/                # 共享合成数据与构造器
│   ├── master/                  # 主调度器测试
│   ├── data_engine/             # 各子 Skill 单元/契约测试
│   ├── factor_engine/
│   ├── strategy_model_engine/
│   ├── backtest_engine/
│   ├── portfolio_risk_engine/
│   ├── execution_monitor_engine/
│   ├── reports_engine/
│   └── integration/             # 跨 Skill 全链路集成测试
├── output/                       # 演示报告输出目录
└── workspace/                    # 运行时数据与归档目录（自动创建）
```

## 引擎介绍

### 1. 数据采集引擎 (data-engine)

**职责**：从多数据源获取A股行情数据，完成复权、涨跌停标记、ST过滤等清洗工作。

**支持数据源**（9 种，含对话式切换）：
- **默认免费源**（无需配置）：BaoStock、AkShare、WebSearch
- **opt-in 源**（需显式启用）：Tushare Pro（TUSHARE_TOKEN）、万得 Wind（WindPy）、同花顺 iFinD（账号密码）、迅投 xtquant（本地客户端）、掘金量化 gm（GM_TOKEN）、通达信 tdxquant（本地终端）

**用户可通过对话直接切换数据源**，例如说"用 wind 取数据"即可切换优先级，无需修改环境变量。

**数据源依赖自动安装**：当某数据源适配器所需的第三方库尚未安装时，自动 `pip install` 后重试（可通过 `AUTO_INSTALL_BACKENDS` 关闭）。

**输出产物**：`cleaned_data.parquet`

### 2. 因子计算引擎 (factor-engine)

**职责**：计算 Alpha 因子、技术指标，完成因子 IC 分析和预筛选。

**核心能力**：
- **双后端切换**：pandas_ta（默认，纯 Python）和 TA-Lib（C 依赖，性能更优），通过 `FACTOR_BACKEND` 切换
- **表达式引擎**：声明式因子定义，如 `RSI($close, 14)`、`MACD($close, 12, 26, 9)`
- **Alpha158 扩展因子库**：47 个因子，覆盖动量/波动率/成交量/技术指标/资金流向/其他 6 大类
- **提前期偏差检测**：自动检测因子计算中的未来数据泄露
- **IC 衰减分析**：分析因子预测能力随时间衰减曲线
- **惊泥因子库集成**：可选从 jingni-datafeed 获取已沉淀因子数据

**输出产物**：`factor_data.parquet`

### 3. 模型训练引擎 (strategy-model-engine)

**职责**：基于机器学习模型进行选股预测，支持 LightGBM、CatBoost 等主流算法。

**核心能力**：
- **多模型支持**：LightGBM（默认）、CatBoost、逻辑回归、随机森林
- **超参数优化**：Optuna 自动调参（支持搜索次数和超时限制）
- **实验追踪**：MLflow 模型版本管理
- **防过拟合**：Purged Group Time Series Split（分组时序交叉验证，含清洗期）
- **意图解析器**：从自然语言中解析策略意图和参数
- **Alpha 表达式引擎**：声明式因子表达式
- **动态权重分配**：基于因子表现的动态权重调整

**输出产物**：`model.pkl`

### 4. 回测引擎 (backtest-engine)

**职责**：在历史数据上验证策略效果，计算收益率、夏普比、最大回撤等指标。

**支持回测后端**：

| 后端 | 状态 | 说明 |
|------|------|------|
| `native` | 生产可用 | 内置原生回测，完整实现 |
| `rqalpha` | 预留/示例桩 | 仅返回模拟或空实现 |
| `backtrader` | 预留/示例桩 | 仅返回模拟或空实现 |
| `gm` | 预留/示例桩 | 需本地 gm 客户端与 Token |

**增强功能**：Walk-Forward 验证（过拟合检测）、向量化回测加速、IC 分析、扩展绩效指标

**A股规则**：T+1 交割、涨跌停板、停牌处理、真实费用模型（佣金万2.5/印花税1‰/过户费0.02‰）

**输出产物**：`backtest_result.json`

### 5. 组合优化引擎 (portfolio-risk-engine)

**职责**：基于量化模型输出，构建最优持仓组合，进行风险预算分配。

**优化方法**：最大夏普、最小方差、分层风险平价（HRP，默认）、Black-Litterman、CVaR

**A股约束**：个股权重上限 10%、行业偏离 ±5%、换手率控制 50%

**风控机制**：多层止损（组合层面单日亏损 3% + 个股层面破位止损）、VaR/CVaR 风险度量、Barra CNE5 风格因子归因

**增强功能**：Walk-Forward 稳健性验证、风险引擎、断路器 v2

**输出产物**：`portfolio_weights.json`

### 6. 执行监控引擎 (execution-monitor-engine)

**职责**：对接券商接口，执行交易指令，实时监控仓位和盈亏。

**执行模式**：

| 模式 | 状态 | 说明 |
|------|------|------|
| `paper` | 生产可用 | 模拟交易，本地虚拟账户（滑点/T+1/数量校验/资金校验/断路器/状态持久化） |
| `live`（xtquant） | 生产可用 | 迅投 miniQMT 实盘，需本地客户端 + XTQUANT_PATH/XTQUANT_ACCOUNT |
| `live`（gm） | 生产可用 | 掘金量化实盘，需 GM_TOKEN/GM_ACCOUNT_ID |

> 三种模式均已通过实盘连通性验证（连接/查询/下单/撤单全流程）。

**硬风控断路器**：单日亏损 2% 限制、单笔金额 10% 上限、持仓集中度 10%、订单频率 2笔/秒

**模拟交易增强**：滑点模拟（千1）、T+1 约束、100股整数倍校验、资金不足校验、审计日志（JSONL）、账户状态持久化

**量化断路器**：增强版多维度风控规则组合，可配置触发阈值和恢复条件

**输出产物**：`trade_log.json`

### 7. 报告生成引擎 (reports-engine)

**职责**：汇总各阶段产物，生成可视化 HTML 报告。根据意图自动路由到不同报告模板。

**量化策略报告**：净值曲线、绩效 TearSheet、月度收益热力图、申万行业归因、Brinson 分解、风格暴露分析

**个股分析报告**：
- 技术面深度分析：行情数据 + 多周期趋势 + 技术指标 + K线形态 + 量价关系 + 资金面 + 龙虎榜 + 深度解读（LLM） + 风险提示
- 基本面深度分析：行情数据 + 公司概况 + 盈利能力 + 成长性 + 估值分析 + 财务健康度 + 股东结构 + 深度解读（LLM） + 风险提示

**K线图功能**：基于 TradingView lightweight-charts，支持 8 种可切换技术指标（成交量/MACD/RSI/KDJ/BOLL/WR/CCI/OBV），显示 MA5/10/20/60 均线

**LLM 动态 Prompt 生成**：llm_analyst 模块根据模板配置中的 factor_groups 动态生成系统提示词，各因子分组含分析要点提示，自动注入到 LLM 深度解读 prompt 中

**LLM 内容注入**：报告模板包含 LLM 占位符，agent 可在 `run_pipeline()` 时传入 `llm_responses` 参数自动替换

**输出产物**：`technical_report.html` / `fundamental_report.html`（个股分析）或 `report.html`（量化策略）

## 配置说明

项目通过环境变量进行配置，无需额外配置文件。

### 通用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QUANT_WORK_DIR` | `./workspace` | 工作目录（数据、产物、日志、归档的根目录） |
| `QUANT_FORCE_REFRESH` | `0` | 强制刷新所有阶段，忽略缓存产物（设为 `1` 启用） |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### 数据采集

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATA_BACKENDS` | `baostock,akshare,websearch` | 数据源优先级链（仅真正免费源） |
| `DATA_BACKEND` | 无 | 单源模式（不降级，与 DATA_BACKENDS 互斥） |
| `TUSHARE_TOKEN` | 无 | Tushare Pro API Token |
| `GM_TOKEN` | 无 | 掘金量化 API Token |
| `IFIND_USERNAME` / `IFIND_PASSWORD` | 无 | 同花顺 iFinD 登录凭证 |
| `ALLOW_SYNTHETIC_FALLBACK` | `true` | 全部数据源失败时生成模拟数据兜底 |
| `AUTO_INSTALL_BACKENDS` | `true` | 数据源依赖缺失时自动 `pip install` 后重试 |
| `ADJUST_MODE` | `hfq` | 复权方式（前复权 `hfq` / 后复权 `qfq`） |
| `DATA_FORMAT` | `parquet` | 数据落盘格式（csv/sql） |
| `DATA_MAX_WORKERS` | `4` | 并行下载线程数 |

### 因子计算

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FACTOR_BACKEND` | `pandas_ta` | 因子计算后端（`pandas_ta` / `talib`） |
| `JINGNI_URL` | 无 | 惊泥因子库服务地址 |
| `JINGNI_TOKEN` | 无 | 惊泥因子库 API Token |

### 回测与模型

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BACKTEST_BACKEND` | `native` | 回测引擎后端（`native` / `rqalpha` / `backtrader` / `gm`） |
| `MODEL_TYPE` | `lightgbm` | 模型类型（`lightgbm` / `catboost` / `logistic_regression` / `random_forest`） |
| `OPTUNA_TRIALS` | `50` | Optuna 超参搜索次数 |
| `OPTUNA_TIMEOUT` | `600` | Optuna 超时时间（秒） |
| `BENCHMARK` | `000300.SH` | 基准指数 |
| `INIT_CAPITAL` | `1000000.0` | 初始资金 |

### 交易执行

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRADE_MODE` | `paper` | 交易模式（`paper` / `live`） |
| `TRADE_BACKEND` | `xtquant` | 交易接口后端（`xtquant` / `gm`） |
| `XTQUANT_PATH` | 无 | miniQMT userdata_mini 路径（live+xtquant 必需） |
| `XTQUANT_ACCOUNT` | 无 | miniQMT 资金账号（live+xtquant 必需） |
| `GM_TOKEN` | 无 | 掘金量化 API Token（live+gm 必需） |
| `GM_ACCOUNT_ID` | 无 | 掘金账户 ID（live+gm 必需，终端获取） |
| `INIT_CAPITAL` | `1000000` | 初始资金（paper 模式） |
| `SLIPPAGE` | `0.001` | 滑点模拟比例（paper 模式） |
| `MAX_DAILY_LOSS_RATIO` | `0.02` | 单日最大亏损比例 |
| `MAX_SINGLE_ORDER_RATIO` | `0.10` | 单笔订单最大金额比例 |

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
    artifacts: Dict[str, str]        # 各阶段产物路径
    external_data: Dict[str, Any]    # 系统内置工具传入的外部数据
    data_sources: Optional[List[str]] # data-engine 专用：用户对话指定的数据源优先级链

    # 运行归档
    run_dir: str                     # 运行归档目录路径
    step_dirs: Dict[str, str]        # 各步骤归档子目录路径

    # 元信息与错误
    metadata: Dict[str, Any]         # 各阶段元数据（含 report_template、factor_source 等）
    errors: List[str]                # 错误记录
```

## 数据源优先级策略

**对话优先 + 配置兜底**：用户通过自然语言对话即可切换数据源，无需修改环境变量。

完整优先级链（从高到低）：
```
1. ctx.external_data (Agent 系统内置工具/MCP) — 最高
2. ctx.data_sources (用户对话指定) — 用户说"用 wind 取数据"即可
3. 环境变量 DATA_BACKENDS — 高级用户/CI 配置
4. 代码默认值 "baostock,akshare,websearch" — 仅真正免费源
5. synthetic (模拟数据兜底) — 全部失败时告知用户
```

## jingni-datafeed 自动部署

jingni-trader 可选依赖 [jingni-datafeed](https://github.com/duhanjun/jingni-datafeed)（惊泥因子库 datafeed 服务）。

**启动时自动检测**：MasterEngine 实例化时自动检查 `skills/jingni-datafeed/` 目录：
- 目录不存在 → 自动从 GitHub 克隆（`git clone --depth 1`）
- 目录已存在 → 运行版本检查（只检测落后、不自动修改文件）

自动克隆失败时不会阻断主流程，仅输出警告日志。

## 运行归档

每次运行完整流程时，自动创建时间戳归档目录：

```
workspace/archives/20260529_143025/
├── pipeline_summary.md      # 全流程汇总报告
├── step_1_DATA/
│   ├── summary.md           # 子任务小结
│   └── artifacts/           # 产物副本
├── step_2_FACTOR/
│   ├── summary.md
│   └── artifacts/
├── step_3_REPORT/
│   ├── summary.md
│   └── artifacts/
...
```

## 测试套件

项目采用三级测试体系，按子 Skill 边界组织：

| 层级 | 标记 | 说明 |
|------|------|------|
| L1 契约测试 | `contract` | 验证 `run(ctx)` 接口契约 |
| L2 单元测试 | `unit` | 子 Skill 内部模块测试 |
| L3 集成测试 | `integration` | 跨 Skill 全链路真实对接 |

支持按标记和目录选择性运行：

```bash
# 运行全部测试
pytest tests/ -v

# 仅运行某个子 Skill 的测试
pytest tests/ -v -m skill_data_engine

# 仅运行契约测试
pytest tests/ -v -m contract

# 运行集成测试
pytest tests/ -v -m integration

# 按目录运行
pytest tests/data_engine/ -v
```

## LLM 内容注入

个股分析报告中包含 LLM 占位符，agent 可在 `run_pipeline()` 时传入 `llm_responses` 参数自动替换：

```python
from engine import MasterEngine

engine = MasterEngine()
result = engine.run_pipeline(
    user_input="分析 002594.SZ 比亚迪的技术面和基本面",
    llm_responses={
        "technical": {"overall_assessment": "...", "technical_score": 75, ...},
        "fundamental": {"overall_assessment": "...", "fundamental_score": 82, ...},
    }
)
```

## 开发指南

### 添加新的数据源适配器

1. 在 `skills/data-engine/scripts/adapters/` 下创建新适配器
2. 继承 `BaseDataProvider` 类
3. 实现 `fetch_daily` 和 `fetch_minute` 方法
4. 在 `config.py` 中注册新的适配器

### 添加新的回测框架

1. 在 `skills/backtest-engine/scripts/adapters/` 下创建新适配器
2. 继承 `BaseBacktest` 类
3. 实现 `run_backtest` 方法
4. 在 `config.py` 中注册新的框架

### 添加新的因子

1. 使用表达式引擎声明式定义：`engine.compute_expression_factors(data, {"my_factor": "RSI($close, 14)"})`
2. 或添加到 `skills/factor-engine/scripts/optimizations/alpha158_lib.py` 扩展因子库
3. 或添加到 `skills/factor-engine/factors/alphafactors.py` 内置因子

## 技术栈

- **数据处理**：pandas, numpy, scipy, pyarrow
- **数据源**：tushare, baostock, akshare, xtquant, gm, tdxquant, wind, ifind, websearch
- **因子计算**：pandas-ta, TA-Lib, alphalens
- **机器学习**：lightgbm, catboost, scikit-learn, optuna, mlflow
- **回测框架**：native（内置原生回测）
- **组合优化**：PyPortfolioOpt, riskfolio-lib, cvxpy
- **可视化**：plotly, matplotlib, quantstats, TradingView lightweight-charts
- **报告生成**：jinja2, HTML/CSS/JS

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 联系方式

- 项目主页：https://github.com/duhanjun/jingni-trader
- 问题反馈：https://github.com/duhanjun/jingni-trader/issues
