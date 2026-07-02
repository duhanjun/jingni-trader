# 量化交易Agent Skill技能套件 - 产品需求文档

## 概述
- **Summary**: 构建一套完整的量化交易Agent Skill技能套件，基于专业量化交易机构的系统架构，**专为A股市场优化**。整个系统基于主流Python开源项目构建，每个Skill负责整合调度特定的Python开源库，**支持在同一Skill内切换不同的底层库**。主Skill作为协调中枢调用各子Skill完成全流程工作。
- **Purpose**: 为OpenClaw系统提供专业级A股量化交易能力，通过模块化的Skill设计实现投研流程的自动化、标准化和可复用性。
- **Target Users**: 量化交易研究者、宽客、量化基金团队、个人投资者

## 技术栈选型原则
- **核心原则**: 每个Skill的scripts只负责整合和调度主流Python开源库，不重复造轮子
- **稳定性**: 优先选择GitHub Star高，社区活跃，维护良好的成熟项目
- **A股优先**: 优先选择支持A股市场、本土化程度高的库
- **多后端支持**: 同一功能支持多个底层库，可通过配置切换
- **扩展性**: 预留接口支持后续替换或升级底层库

## A股市场特点与架构调整

### A股三大核心差异（贯穿全流程）
1. **市场微观结构不同**: T+1交割、涨跌停板、可卖空受限、印花税/过户费等特殊成本
2. **数据源生态不同**: Tushare Pro/米筐/聚宽数据为核心，而非yfinance
3. **因子有效性与模型差异**: A股中反转效应更强、小市值溢价显著、散户驱动特征明显

### A股特殊处理要点
- **T+1交割**: 当日买入股票当日不可卖出
- **涨跌停板标记**: 涨停不可买入、跌停不可卖出
- **ST/退市风险标记**: 每日维护ST列表，策略池自动剔除
- **上市时间过滤**: 剔除上市不足60个交易日的新股
- **复权处理**: 必须使用后复权价格计算收益率
- **最小100股**: 持仓手数为100的整数倍

## 主流Python开源库技术栈（A股专属版）

### 数据获取层
| 库名 | 用途 | 特点 | 多后端支持 |
|------|------|------|-----------|
| **Tushare Pro** | A股全量日线/分钟线、财报、指数成分股、沪深港通 | 机构稳定首选，需要积分/付费 | 主要数据源 |
| **BaoStock** | 免费日线/分钟线历史数据 | 无需注册，适合快速回测 | 备选数据源 |
| **AkShare** | 财经数据聚合，龙虎榜、大宗交易、新股、基金另类数据 | 补充Tushare未覆盖数据 | 补充数据源 |
| **xtquant (xtdata)** | miniQMT行情数据 | 实时数据，支持Level2 | 可切换 |
| **gm (掘金)** | 掘金量化平台数据 | 云端数据，免费 | 可切换 |

### 数据存储层
| 库名 | 用途 | 特点 |
|------|------|------|
| **SQLAlchemy** | ORM框架 | 统一数据库接口 |
| **MySQL/PostgreSQL** | 结构化数据存储 | 清洗后的标准表 |
| **SQLite** | 轻量级本地存储 | 快速原型开发 |

### 因子与指标层
| 库名 | 用途 | 特点 | 多后端支持 |
|------|------|------|-----------|
| **TA-Lib** | 技术指标计算 | 150+指标、性能优 | 主要 |
| **pandas-ta** | 技术指标（纯Python） | 无需编译、安装简单 | 备选 |

### 回测与策略层
| 库名 | 用途 | 特点 | 多后端支持 |
|------|------|------|-----------|
| **RQAlpha** | A股专用回测引擎 | **专为A股设计**，内置涨跌停、T+1、手续费 | 主要 |
| **Backtrader** | 事件驱动回测 | 灵活度高，需自行添加A股规则 | 可切换 |
| **gm (掘金)** | 掘金回测引擎 | 云端回测，支持仿真 | 可切换 |
| **vnpy/VeighNa** | CTA回测框架 | 主要用于期货，股票需独立配置 | 可切换 |

### 模型训练层
| 库名 | 用途 | 特点 |
|------|------|------|
| **LightGBM** | 多因子选股模型 | 高维特征、训练快、对缺失值鲁棒 |
| **CatBoost** | 类别特征处理 | 无需独热编码、稳定性好 |
| **scikit-learn** | 基准模型 | 逻辑回归、SVM等 |
| **Optuna** | 超参搜索 | 内置剪枝功能 |
| **MLflow** | 实验管理 | 记录每一次因子组合和模型参数 |

### 绩效分析层
| 库名 | 用途 | 特点 |
|------|------|------|
| **quantstats** | 绩效分析报告 | 一键生成专业报告 |
| **PyPortfolioOpt** | 组合优化 | 现代投资组合理论 |

### 可视化层
| 库名 | 用途 | 特点 |
|------|------|------|
| **matplotlib** | 基础图表 | 灵活、广泛支持 |
| **plotly** | 交互图表 | Web风格、交互强 |

### 交易执行层（多后端切换）
| 库名 | 用途 | 特点 | 多后端支持 |
|------|------|------|-----------|
| **xtquant (xttrader)** | miniQMT交易接口 | 需配合miniQMT客户端使用 | ✅ 可切换 |
| **gm (掘金)** | 掘金量化交易 | 云端/本地仿真交易 | ✅ 可切换 |
| **vnpy/VeighNa** | CTA交易框架 | 支持CTP、XTP等 | ✅ 可切换 |

## 目标
1. 构建一个主Skill（quant-trading-master）作为协调中枢，负责全流程任务编排
2. 构建7个子Skill覆盖A股量化交易核心模块，每个Skill基于对应的主流Python开源库，**支持多后端切换**
3. 提供自动化安装脚本，支持自动检测和安装OpenClaw，自动导入所有Skill
4. 所有API密钥从环境变量统一读取（TUSHARE_TOKEN, XCSC_TUSHARE_TOKEN, GM_TOKEN等）

## 非目标
- 不包含具体的交易策略逻辑（策略由用户或AI自主研究生成）
- 不提供投资建议或财务分析服务
- 不实现交易所直连交易（仅通过券商API接口模拟/实盘执行）
- 不包含高频交易相关模块（Tick级撮合、内存撮合引擎等）

## 功能需求

### FR-1: 主Skill（quant-trading-master）
- 作为量化交易全流程的协调中枢
- 解析用户意图，判断当前处于哪个投研阶段
- 根据阶段调度相应子Skill完成工作
- 管理会话状态和任务上下文
- 输出结构化的量化研究报告和交易信号
- **阶段状态机**:
```
[数据获取] → [因子构建] → [模型训练] → [回测验证] → [组合优化] → [模拟／实盘] → [绩效报告]
```
每个阶段可分支：回测失败→返回因子调优；模型过拟合→触发样本外再验证
- **Context对象标准化**: 当前任务ID、会话ID、股票池、时间范围、已完成阶段产物路径、全局配置
- **里程碑检查点机制**: 每个子Skill完成时检验产物完整性和基本合理性，失败时给出清晰错误码，支持从断点重试
- **底层库**: 无（纯调度逻辑）

### FR-2: A股数据采集与治理（a-share-data-engine）
- 从Tushare Pro/BaoStock/AkShare/xtquant/gm获取A股行情数据
- 数据清洗：复权处理、停牌标记、涨跌停板标记
- ST/退市风险标记过滤
- 新股上市时间过滤（剔除上市不足60个交易日的新股）
- 本地数据仓库持久化（SQLite/MySQL/PostgreSQL）
- **数据版本管理**: 数据快照（Snapshot）概念，所有回测基于同一数据版本
- **复权处理规则**: 使用Tushare的`adj_factor`复权因子表，以`qfq`（前复权）计算收益率，以`hfq`（后复权）维持价格连续
- **多后端支持**: 可切换tushare/baostock/akshare/xtquant/gm作为数据源
- **验收标准补充**: 清洗后的日线数据缺失率 < 2%（以全A股过去5年为测试集），停牌/涨跌停标记无遗漏
- **底层库**: tushare, baostock, akshare, xtquant, gm, pandas, sqlalchemy

### FR-3: A股阿尔法因子库（a-share-factor-engine）
- 单Alpha因子构建与评估（动量、价值、质量、情绪等）
- A股专用因子：1个月反转、市值因子、换手率因子，资金流因子、事件因子
- Alpha因子IC（Information Coefficient）分析（需行业中性化处理）
- 因子相关性分析与去冗余
- 多Alpha融合与权重分配
- **因子存储与复用**: 因子计算结果集中存储为Parquet文件，支持增量更新
- **对抗性验证（可选）**: 训练/测试分布一致性检查
- **多后端支持**: 可切换ta-lib/pandas-ta计算技术指标
- **底层库**: ta-lib, pandas-ta, alphalens, pandas, scikit-learn

### FR-4: A股策略开发与模型训练（strategy-model-engine）
- 策略模板库（趋势跟踪、均值回归、配对交易等）
- 截面选股模型（LightGBM/CatBoost）
- 择时模型（基于分钟线技术指标）
- 参数优化（Optuna超参搜索）
- 实验管理与记录（MLflow）
- **过拟合防范（Purged Group Time Series Split）**:
  - 训练集、验证集、测试集比例：滚动窗口36个月训练，随后12个月验证，最后12个月测试
  - Purge间隔：训练与验证之间设置2个交易日间隔
  - 分组依据：按行业分组
- **强制样本外（OOS）输出**: 回测报告必须包含严格按时间切分的OOS绩效指标
- **底层库**: lightgbm, catboost, scikit-learn, optuna, mlflow

### FR-5: A股策略回测与仿真（backtest-engine）
- 事件驱动回测引擎（**支持RQAlpha/Backtrader/gm多后端切换**）
- A股规则模拟：T+1、涨跌停板、费用（佣金、印花税、过户费）
- **涨跌停精确模型**:
  - 买入策略可选：严格废单 / 排队模型
  - 卖出策略可选：严格废单 / 次日处理
- **分红配股转股处理**: 使用Tushare adj_factor复权因子
- **交易费用模型**: 可配置CommissionModel，内置A股标准模板（万2.5佣金、每笔最低5元、印花税1‰、过户费等）
- 停牌处理：资产冻结，复牌后恢复
- 绩效指标计算（年化收益、夏普比率、最大回撤、胜率、盈亏比等）
- 过拟合检测（Walk-Forward分析）
- 回测报告生成与可视化
- **验收标准补充**: 使用给定标准策略（如20日均线突破买入，T+1卖出）在已知基准数据集上回测，所得年化收益率、夏普比率与公开参照值误差不超过1%；在包含涨跌停的样本中，无任何一次突破规则成交
- **多后端支持**: 可切换rqalpha/backtrader/gm作为回测引擎
- **底层库**: rqalpha, backtrader, gm, quantstats, matplotlib, plotly

### FR-6: 组合优化与风控（portfolio-risk-engine）
- 风险模型构建（协方差矩阵估计）
- 组合权重优化（风险平价、最小方差、最大夏普）
- Barra风格因子归因（CNE5模型）
- A股特殊约束：单一股票持仓不超过10%、行业暴露偏离基准±5%
- 止损机制：单日回撤-3%硬止损、个股放量跌破20日均线止损
- 实时风险监控（VaR、CVaR）
- **底层库**: pypfopt, riskfolio-lib, numpy, scipy, pandas

### FR-7: A股实盘执行与监控（execution-monitor-engine）
- 模拟账户管理（资金初始化、状态跟踪）
- 订单模拟执行（市价单、限价单、止损单）
- **实盘接口框架（支持多后端切换）**:
  - xtquant：miniQMT/XTQuant接口
  - gm：掘金量化交易接口
  - vnpy：VeighNa框架（支持扩展）
- **本地硬断路器（Circuit Breaker）**:
  - 当日累计已执行订单的浮动盈亏 + 已实现盈亏，若超过`max_daily_loss`（例如净资产2%），直接拒绝所有新开仓订单，并平掉部分仓位
  - 单笔订单金额不得超过账户净资产的一定比例
  - 每秒/每分钟订单数限制（防刷单）
  - 持仓集中度限制（单票不超过配置上限）
- **确认模式与演习模式**:
  - 仿真模式：连接真实行情但下单到模拟撮合器
  - 一键切换到paper_trade，与实盘采用同一套代码路径
- **审计日志完整性**: 所有订单（包括被风控拒绝的单子）都必须记录完整上下文
- 仓位同步与成本计算
- 交易日志记录
- **验收标准补充**: 在xtquant/gm仿真环境中连续运行5个交易日，无因适配器异常导致的漏单、错单；每日结算后资金误差小于万分之一
- **多后端支持**: 可切换xtquant/gm/vnpy作为交易接口
- **底层库**: pandas, numpy（模拟逻辑），xtquant, gm, vnpy（实盘扩展）

### FR-8: 绩效归因与可视化报告（reports-engine）
- 日收益、滚动夏普、月度热力图报告
- A股风格暴露分析（大盘/小盘/成长/价值）
- 行业暴露图（申万一级）
- Brinson归因（超额收益分解）
- 动态净值曲线和因子衰减走势图
- **过拟合风险警示**: OOS夏普低于全样本夏普的60%时自动标记警告
- **底层库**: quantstats, plotly, matplotlib

### FR-9: 自动化安装脚本
- 检测本地是否已安装OpenClaw
- 如未安装，自动完成OpenClaw安装（Linux/macOS）
- 自动导入所有Skill到正确目录
- 环境变量配置指导（TUSHARE_TOKEN, GM_TOKEN等）
- 依赖检查与安装（自动安装Python依赖，支持选择安装xtquant/gm/vnpy等）
- **许可与合规提示**: 加入对Tushare、xtquant等库许可协议的重大提醒

### FR-10: Skill接口规范
- 每个Skill提供一个统一的`run(ctx)`函数入口，返回结构化结果
- 与OpenClaw的交互通过命令行或配置文件
- Context对象标准定义

## 多后端切换设计

### 统一接口抽象
为支持多后端切换，每个支持多库的Skill应遵循以下设计：

```
scripts/
├── base/                    # 抽象基类和统一接口
│   ├── base_data_provider.py    # 数据接口基类
│   ├── base_backtest.py         # 回测引擎基类
│   └── base_trader.py           # 交易接口基类
├── adapters/                # 各库适配器
│   ├── tushare_adapter.py       # Tushare适配器
│   ├── xtquant_adapter.py       # XtQuant适配器
│   ├── gm_adapter.py           # 掘金适配器
│   └── ...
├── engine.py                # 主引擎（根据配置选择适配器）
└── config.py                # 配置文件（选择使用的后端）
```

### 基类方法签名和返回值Schema

#### BaseDataProvider
```python
class BaseDataProvider:
    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "none"
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
        code, date, open, high, low, close, volume, amount,
        pre_close, change_pct, turnover_rate, is_st, is_limit_up, is_limit_down
        """
```

#### BaseBacktestEngine
```python
class BaseBacktestEngine:
    def run(
        self,
        strategy_code: str,
        data_path: str,
        config: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns tuple:
        (trades_df, performance_metrics)
        trades_df columns: datetime, code, action, price, amount
        performance_metrics: dict with keys like annual_return, sharpe_ratio, max_drawdown
        """
```

#### BaseTrader
```python
class BaseTrader:
    def place_order(self, symbol: str, action: str, amount: int, price: float) -> str:
        """Place an order, returns order_id"""

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""

    def get_positions(self) -> pd.DataFrame:
        """Get current positions"""
```

### 行为等价性声明
- AkShare 不原生提供后复权价格，适配器需使用后复权因子手动计算
- 掘金 `gm` 回测时停牌处理策略与 RQAlpha 不完全一致，需在文档中注明差异点
- 适配器应在发现差异时抛出明确警告

### 配置层验证
在`config.py`中不仅声明后端选择，还应包含初始化时对所选后端的可用性检查（网络连通、token有效性、版本兼容），并在启动时返回明确状态报告。

### 配置切换示例
```python
# config.py
DATA_BACKEND = "tushare"  # 可选: tushare, baostock, akshare, xtquant, gm
BACKTEST_BACKEND = "rqalpha"  # 可选: rqalpha, backtrader, gm
TRADE_BACKEND = "xtquant"  # 可选: xtquant, gm, vnpy
```

## 非功能需求

### NFR-1: 代码规范
- 所有Python代码遵循PEP 8规范
- 使用中文注释和文档字符串
- 类型提示（Type Hints）全覆盖
- 每个脚本包含详细docstring

### NFR-2: 错误处理
- 所有API调用包含异常处理
- 明确的错误信息和建议
- 日志记录机制（logging模块）
- 优雅降级策略（某后端不可用时切换到备选后端）

### NFR-3: 安全要求
- API密钥仅从环境变量读取
- 不在代码或日志中暴露敏感信息
- 交易操作包含确认机制
- 敏感操作日志审计

### NFR-4: 可扩展性
- 底层库可替换（预留接口）
- 支持自定义因子和策略模板
- 预留MCP服务接口扩展

## 约束

### 技术约束
- Python 3.9+
- 核心依赖：pandas, numpy, scipy, rqalpha, backtrader, lightgbm, tushare, akshare, xtquant, gm
- Skill文件遵循Anthropic标准格式（SKILL.md + scripts/ + references/）

### 业务约束
- 数据仅用于学习和研究
- 遵守Tushare服务条款和积分限制
- 模拟交易不涉及真实资金
- 实盘交易需要用户自行接入券商API

## 假设

1. 用户已具备Python基础环境
2. 用户拥有有效的Tushare Token（标准版）
3. 用户拥有有效的掘金量化Token（可选）
4. 用户使用OpenClaw作为AI助手运行环境
5. 用户主要关注中国A股市场（中低频策略）
6. 用户具备基本的量化投资知识

## 验收标准（量化阈值版）

### AC-1: 主Skill协调功能
- **Given**: 用户请求进行完整的量化投研任务
- **When**: 激活quant-trading-master Skill
- **Then**: 正确解析任务所处阶段，按流程依次调度子Skill完成工作，正确维护Context对象
- **Verification**: `programmatic` - 主Skill能正确解析任务类型并调用对应子Skill，Context正确传递

### AC-2: A股数据采集功能
- **Given**: 用户指定股票池、时间范围和数据类型
- **When**: 调用a-share-data-engine Skill获取数据
- **Then**: 返回清洗后的标准化数据，并包含涨跌停标记、ST标记等
- **Verification**: `programmatic` - 数据格式正确、A股特殊标记完整，清洗后的日线数据缺失率 < 2%（全A股过去5年测试集），支持多数据源切换

### AC-3: A股因子研究功能
- **Given**: 用户提供候选因子或要求生成Alpha
- **When**: 调用a-share-factor-engine Skill进行评估
- **Then**: 输出因子IC序列、相关性矩阵、融合建议，因子Parquet文件正确存储
- **Verification**: `programmatic` - IC计算正确（含行业中性化处理）、报告格式完整

### AC-4: 策略开发功能
- **Given**: 用户提供策略思路或选择策略模板
- **When**: 调用strategy-model-engine Skill开发策略
- **Then**: 生成可执行的策略代码和参数优化结果，包含分组时序交叉验证
- **Verification**: `programmatic` - 策略代码可运行、参数优化有结果、样本内外切分正确

### AC-5: 回测验证功能
- **Given**: 用户提供策略参数和回测时间范围
- **When**: 调用backtest-engine Skill执行回测
- **Then**: 生成包含关键指标的绩效报告，正确模拟T+1和涨跌停规则，支持切换回测引擎
- **Verification**: `programmatic` - 绩效指标计算正确、T+1规则生效、图表正确生成、多后端切换正常
- **量化验收**: 使用标准策略（20日均线突破买入，T+1卖出）在已知基准数据集上回测，年化收益率、夏普比率与公开参照值误差不超过1%；在包含涨跌停的样本中，无任何一次突破规则成交

### AC-6: 组合优化与风控功能
- **Given**: 用户提供候选股票池和风险偏好
- **When**: 调用portfolio-risk-engine Skill优化组合
- **Then**: 输出最优权重分配和风险分析，含A股特殊约束
- **Verification**: `programmatic` - 优化算法收敛、权重满足约束（单票≤10%）、VaR计算正确

### AC-7: 模拟交易功能
- **Given**: 用户启动模拟交易模式
- **When**: 调用execution-monitor-engine Skill（模拟模式）
- **Then**: 模拟账户正确执行信号并跟踪持仓
- **Verification**: `programmatic` - 订单执行正确、持仓计算准确

### AC-8: 实盘交易功能（多后端）
- **Given**: 用户选择实盘交易后端
- **When**: 调用execution-monitor-engine Skill（xtquant/gm/vnpy模式）
- **Then**: 正确对接所选接口发送订单，本地硬断路器生效
- **Verification**: `programmatic` - xtquant/gm/vnpy接口调用正确，支持切换
- **量化验收**: 在xtquant/gm仿真环境中连续运行5个交易日，无因适配器异常导致的漏单、错单；每日结算后资金误差小于万分之一

### AC-9: 报告生成功能
- **Given**: 用户需要绩效报告
- **When**: 调用reports-engine Skill生成报告
- **Then**: 输出包含A股风格暴露分析和行业归因的可视化报告
- **Verification**: `programmatic` - 报告格式完整、图表正确生成、过拟合风险警告触发正确

### AC-10: 安装脚本功能
- **Given**: 在未安装OpenClaw的Linux/macOS环境
- **When**: 运行自动化安装脚本
- **Then**: 自动完成OpenClaw安装和Skill导入，支持选择性安装xtquant/gm/vnpy
- **Verification**: `programmatic` - 脚本执行成功、Skill可被识别

### AC-11: Skill文件格式
- **Given**: 每个Skill项目
- **When**: 检查文件结构
- **Then**: 符合Anthropic标准（SKILL.md + scripts/ + references/）
- **Verification**: `programmatic` - 目录结构验证、YAML格式正确

## 开放问题

- [x] ~~实盘交易接口具体对接哪家券商？~~ **已确认：支持xtquant和gm，可扩展vnpy**
- [ ] 是否需要支持期权策略相关模块？
- [ ] 是否需要支持多策略组合同时运行？
