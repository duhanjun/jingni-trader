# 量化交易Agent Skill技能套件 - 产品需求文档

## 概述
- **Summary**: 构建一套完整的量化交易Agent Skill技能套件，基于专业量化交易机构的系统架构，**专为A股市场优化**。整个系统基于主流Python开源项目构建，每个Skill负责整合调度特定的Python开源库，主Skill作为协调中枢调用各子Skill完成全流程工作。
- **Purpose**: 为OpenClaw系统提供专业级A股量化交易能力，通过模块化的Skill设计实现投研流程的自动化、标准化和可复用性。
- **Target Users**: 量化交易研究者、宽客、量化基金团队、个人投资者

## 技术栈选型原则
- **核心原则**: 每个Skill的scripts只负责整合和调度主流Python开源库，不重复造轮子
- **稳定性**: 优先选择GitHub Star高，社区活跃，维护良好的成熟项目
- **A股优先**: 优先选择支持A股市场、本土化程度高的库
- **扩展性**: 预留接口支持后续替换或升级底层库

## A股市场特点与架构调整

### A股三大核心差异（贯穿全流程）
1. **市场微观结构不同**：T+1交割、涨跌停板、可卖空受限、印花税/过户费等特殊成本
2. **数据源生态不同**：Tushare Pro/米筐/聚宽数据为核心，而非yfinance
3. **因子有效性与模型差异**：A股中反转效应更强、小市值溢价显著、散户驱动特征明显

### A股特殊处理要点
- **T+1交割**：当日买入股票当日不可卖出
- **涨跌停板标记**：涨停不可买入、跌停不可卖出
- **ST/退市风险标记**：每日维护ST列表，策略池自动剔除
- **上市时间过滤**：剔除上市不足60个交易日的新股
- **复权处理**：必须采用后复权价格计算收益率
- **最小100股**：持仓手数为100的整数倍

## 主流Python开源库技术栈（A股专属版）

### 数据获取层
| 库名 | 用途 | 特点 |
|------|------|------|
| **Tushare Pro** | A股全量日线/分钟线、财报、指数成分股、沪深港通 | 机构稳定首选，需要积分/付费 |
| **BaoStock** | 免费日线/分钟线历史数据 | 无需注册，适合快速回测 |
| **AkShare** | 财经数据聚合，龙虎榜、大宗交易、新股、基金另类数据 | 补充Tushare未覆盖数据 |
| **pandas** | 时间序列对齐、面板构建、复权处理 | 核心数据处理层 |

### 数据存储层
| 库名 | 用途 | 特点 |
|------|------|------|
| **SQLAlchemy** | ORM框架 | 统一数据库接口 |
| **MySQL/PostgreSQL** | 结构化数据存储 | 清洗后的标准表 |
| **SQLite** | 轻量级本地存储 | 快速原型开发 |

### 因子与指标层
| 库名 | 用途 | 特点 |
|------|------|------|
| **TA-Lib** | 技术指标计算 | 150+指标、性能优 |
| **pandas-ta** | 技术指标（纯Python） | 无需编译、安装简单 |
| **Alphalens** | 因子IC/IR分析 | 需适配A股分组规则 |

### 回测与策略层
| 库名 | 用途 | 特点 |
|------|------|------|
| **RQAlpha** | A股专用回测引擎 | **专为A股设计**，内置涨跌停、T+1、手续费 |
| **Backtrader** | 事件驱动回测 | 灵活度高，需自行添加A股规则 |
| **vnpy/VeighNa** | 实盘框架 | 支持CTP、XTP等，也可用于回测 |

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

## 目标
1. 构建一个主Skill（quant-trading-master）作为协调中枢，负责全流程任务编排
2. 构建7个子Skill覆盖A股量化交易核心模块，每个Skill基于对应的主流Python开源库
3. 提供自动化安装脚本，支持自动检测和安装OpenClaw，自动导入所有Skill
4. 所有API密钥从环境变量统一读取（TUSHARE_TOKEN, XCSC_TUSHARE_TOKEN等）

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
- **底层库**: 无（纯调度逻辑）

### FR-2: A股数据采集与治理（a-share-data-engine）
- 从Tushare Pro/BaoStock/AkShare获取A股行情数据
- 数据清洗：复权处理、停牌标记、涨跌停板标记
- ST/退市风险标记过滤
- 新股上市时间过滤（剔除上市不足60个交易日的新股）
- 本地数据仓库持久化（SQLite/MySQL/PostgreSQL）
- **底层库**: tushare, baostock, akshare, pandas, sqlalchemy

### FR-3: A股阿尔法因子库（a-share-factor-engine）
- 单Alpha因子构建与评估（动量、价值、质量、情绪等）
- A股专用因子：1个月反转、市值因子、换手率因子、资金流因子、事件因子
- Alpha因子IC（Information Coefficient）分析（需行业中性化处理）
- 因子相关性分析与去冗余
- 多Alpha融合与权重分配
- **底层库**: ta-lib/pandas-ta, alphalens, pandas, scikit-learn

### FR-4: A股策略开发与模型训练（strategy-model-engine）
- 策略模板库（趋势跟踪、均值回归、配对交易等）
- 截面选股模型（LightGBM/CatBoost）
- 择时模型（基于分钟线技术指标）
- 参数优化（Optuna超参搜索）
- 实验管理与记录（MLflow）
- 过拟合防范（分组时序交叉验证）
- **底层库**: lightgbm, catboost, scikit-learn, optuna, mlflow

### FR-5: A股策略回测与仿真（backtest-engine）
- 事件驱动回测引擎（RQAlpha为主，Backtrader辅助）
- A股规则模拟：T+1、涨跌停板、费用（佣金、印花税、过户费）
- 停牌处理：资产冻结，复牌后恢复
- 绩效指标计算（年化收益、夏普比率、最大回撤、胜率、盈亏比等）
- 过拟合检测（Walk-Forward分析）
- 回测报告生成与可视化
- **底层库**: rqalpha, backtrader, quantstats, matplotlib, plotly

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
- 实盘接口框架（预留接口，支持扩展到vnpy/XTP）
- 仓位同步与成本计算
- 交易日志记录
- 风控守护进程：单日亏损超阈值报警、持仓集中度监控
- **底层库**: pandas, numpy（模拟逻辑），vnpy（实盘扩展）

### FR-8: 绩效归因与可视化报告（reports-engine）
- 日收益、滚动夏普、月度热力图报告
- A股风格暴露分析（大盘/小盘/成长/价值）
- 行业暴露图（申万一级）
- Brinson归因（超额收益分解）
- 动态净值曲线和因子衰减走势图
- **底层库**: quantstats, plotly, matplotlib

### FR-9: 自动化安装脚本
- 检测本地是否已安装OpenClaw
- 如未安装，自动完成OpenClaw安装（Linux/macOS）
- 自动导入所有Skill到正确目录
- 环境变量配置指导（TUSHARE_TOKEN等）
- 依赖检查与安装（自动安装Python依赖）

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
- 优雅降级策略

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
- 核心依赖：pandas, numpy, scipy, rqalpha, backtrader, lightgbm, tushare, akshare
- Skill文件遵循Anthropic标准格式（SKILL.md + scripts/ + references/）

### 业务约束
- 数据仅用于学习和研究
- 遵守Tushare服务条款和积分限制
- 模拟交易不涉及真实资金
- 实盘交易需要用户自行接入券商API

## 假设

1. 用户已具备Python基础环境
2. 用户拥有有效的Tushare Token（标准版）
3. 用户使用OpenClaw作为AI助手运行环境
4. 用户主要关注中国A股市场（中低频策略）
5. 用户具备基本的量化投资知识

## 验收标准

### AC-1: 主Skill协调功能
- **Given**: 用户请求进行完整的量化投研任务
- **When**: 激活quant-trading-master Skill
- **Then**: 正确解析任务所处阶段，按流程依次调度子Skill完成工作
- **Verification**: `programmatic` - 主Skill能正确解析任务类型并调用对应子Skill

### AC-2: A股数据采集功能
- **Given**: 用户指定股票池、时间范围和数据类型
- **When**: 调用a-share-data-engine Skill获取数据
- **Then**: 返回清洗后的标准化数据，并包含涨跌停标记、ST标记等
- **Verification**: `programmatic` - 数据格式正确、A股特殊标记完整

### AC-3: A股因子研究功能
- **Given**: 用户提供候选因子或要求生成Alpha
- **When**: 调用a-share-factor-engine Skill进行评估
- **Then**: 输出因子IC序列、相关性矩阵、融合建议
- **Verification**: `programmatic` - IC计算正确（含行业中性化处理）、报告格式完整

### AC-4: 策略开发功能
- **Given**: 用户提供策略思路或选择策略模板
- **When**: 调用strategy-model-engine Skill开发策略
- **Then**: 生成可执行的策略代码和参数优化结果
- **Verification**: `programmatic` - 策略代码可运行、参数优化有结果

### AC-5: 回测验证功能
- **Given**: 用户提供策略参数和回测时间范围
- **When**: 调用backtest-engine Skill执行回测
- **Then**: 生成包含关键指标的绩效报告，正确模拟T+1和涨跌停规则
- **Verification**: `programmatic` - 绩效指标计算正确、T+1规则生效、图表正确生成

### AC-6: 组合优化与风控功能
- **Given**: 用户提供候选股票池和风险偏好
- **When**: 调用portfolio-risk-engine Skill优化组合
- **Then**: 输出最优权重分配和风险分析，含A股特殊约束
- **Verification**: `programmatic` - 优化算法收敛、权重满足约束、VaR计算正确

### AC-7: 模拟交易功能
- **Given**: 用户启动模拟交易模式
- **When**: 调用execution-monitor-engine Skill（模拟模式）
- **Then**: 模拟账户正确执行信号并跟踪持仓
- **Verification**: `programmatic` - 订单执行正确、持仓计算准确

### AC-8: 报告生成功能
- **Given**: 用户需要绩效报告
- **When**: 调用reports-engine Skill生成报告
- **Then**: 输出包含A股风格暴露分析和行业归因的可视化报告
- **Verification**: `programmatic` - 报告格式完整、图表正确生成

### AC-9: 安装脚本功能
- **Given**: 在未安装OpenClaw的Linux/macOS环境
- **When**: 运行自动化安装脚本
- **Then**: 自动完成OpenClaw安装和Skill导入
- **Verification**: `programmatic` - 脚本执行成功、Skill可被识别

### AC-10: Skill文件格式
- **Given**: 每个Skill项目
- **When**: 检查文件结构
- **Then**: 符合Anthropic标准（SKILL.md + scripts/ + references/）
- **Verification**: `programmatic` - 目录结构验证、YAML格式正确

## 开放问题

- [ ] 实盘交易接口具体对接哪家券商？（需要用户确认，如vnpy+XTP、xtquant等）
- [ ] 是否需要支持期权策略相关模块？
- [ ] 是否需要支持多策略组合同时运行？
