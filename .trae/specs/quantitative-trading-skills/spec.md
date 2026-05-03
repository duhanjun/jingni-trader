# 量化交易Agent Skill技能套件 - 产品需求文档

## 概述
- **Summary**: 构建一套完整的量化交易Agent Skill技能套件，基于专业量化交易机构的系统架构，覆盖A股市场和股指期货的全流程量化投研与交易工作。**整个系统基于主流Python开源项目构建**，每个Skill负责整合调度特定的Python开源库，主Skill作为协调中枢调用各子Skill完成全流程工作。
- **Purpose**: 为OpenClaw系统提供专业级量化交易能力，通过模块化的Skill设计实现投研流程的自动化、标准化和可复用性。
- **Target Users**: 量化交易研究者、宽客、量化基金团队、个人投资者

## 技术栈选型原则
- **核心原则**: 每个Skill的scripts只负责整合和调度主流Python开源库，不重复造轮子
- **稳定性**: 优先选择GitHub Star高、社区活跃、维护良好的成熟项目
- **易用性**: 优先选择API简洁、文档完善、示例丰富的项目
- **扩展性**: 预留接口支持后续替换或升级底层库

## 目标
1. 构建一个主Skill（quant-trading-master）作为协调中枢，负责全流程任务编排
2. 构建8个子Skill覆盖量化交易核心模块，每个Skill基于对应的主流Python开源库
3. 提供自动化安装脚本，支持自动检测和安装OpenClaw，自动导入所有Skill
4. 所有API密钥从环境变量统一读取（TUSHARE_TOKEN, XCSC_TUSHARE_TOKEN等）

## 非目标
- 不包含具体的交易策略逻辑（策略由用户或AI自主研究生成）
- 不提供投资建议或财务分析服务
- 不实现交易所直连交易（仅通过券商API接口模拟/实盘执行）
- 不包含高频交易相关模块（Tick级撮合、内存撮合引擎等）

## 主流Python开源库技术栈

### 数据获取层
| 库名 | 用途 | 特点 |
|------|------|------|
| **tushare** | A股基础数据 | 主流中文金融数据接口 |
| **akshare** | A股/期货/宏观数据 | 免费、开源、数据种类丰富 |
| **yfinance** | 海外市场数据 | 简单易用、覆盖全球市场 |

### 数据处理层
| 库名 | 用途 | 特点 |
|------|------|------|
| **pandas** | 数据分析基础 | 事实标准、高效灵活 |
| **numpy** | 数值计算基础 | 底层高性能 |
| **scipy** | 科学计算 | 统计、优化算法 |

### 因子与指标层
| 库名 | 用途 | 特点 |
|------|------|------|
| **ta-lib** | 技术指标计算 | 150+指标、性能优 |
| **pandas-ta** | 技术指标（纯Python） | 无需编译、安装简单 |
| **alphalens** | 因子分析 | 专为因子研究设计 |

### 回测与策略层
| 库名 | 用途 | 特点 |
|------|------|------|
| **backtrader** | 事件驱动回测 | 轻量、灵活、文档丰富 |
| **zipline-reloaded** | Pipeline回测 | Quantopian生态、Pipeline API |
| **vectorbt** | 向量化回测 | 高性能、参数优化强 |

### 绩效分析层
| 库名 | 用途 | 特点 |
|------|------|------|
| **quantstats** | 绩效分析报告 | 一键生成专业报告 |
| **empyrical** | 风险指标计算 | Zipline核心组件 |

### 组合优化层
| 库名 | 用途 | 特点 |
|------|------|------|
| **pypfopt** | 组合优化 | 现代投资组合理论 |
| **cvxpy** | 凸优化 | 底层优化器 |

### 可视化层
| 库名 | 用途 | 特点 |
|------|------|------|
| **matplotlib** | 基础图表 | 灵活、广泛支持 |
| **plotly** | 交互图表 | Web风格、交互强 |
| **seaborn** | 统计图表 | 美观、统计友好 |

### 机器学习层（可选扩展）
| 库名 | 用途 | 特点 |
|------|------|------|
| **scikit-learn** | 传统ML | 全面、易用 |
| **lightgbm** | 梯度提升 | 高效、准确 |
| **statsmodels** | 统计建模 | 时间序列强 |

## 背景与上下文

### 专业量化交易系统架构（交叉验证后）
基于对vn.py、QUANTAXIS、AlphaOS、高频交易系统等开源和专业系统的深度调研，专业量化交易系统的标准架构分为以下层次：

```
┌─────────────────────────────────────────────────────────────┐
│                    监控与运维层                              │
├─────────────────────────────────────────────────────────────┤
│    风险管理层    │    组合优化层    │    交易执行层           │
├─────────────────────────────────────────────────────────────┤
│    策略开发层    │    回测验证层    │    绩效归因层           │
├─────────────────────────────────────────────────────────────┤
│    Alpha因子层   │   特征工程层     │   信号生成层            │
├─────────────────────────────────────────────────────────────┤
│                    数据工程层                                │
└─────────────────────────────────────────────────────────────┘
```

### 量化投研标准工作流程
```
阶段1: 市场研究与假设生成
    ↓
阶段2: 数据采集与清洗
    ↓
阶段3: 因子工程与IC评估
    ↓
阶段4: 多因子模型构建
    ↓
阶段5: 策略开发与参数优化
    ↓
阶段6: 回测验证与过拟合检测
    ↓
阶段7: 组合优化与风控
    ↓
阶段8: 模拟交易验证
    ↓
阶段9: 实盘部署与监控
```

## 功能需求

### FR-1: 主Skill（quant-trading-master）
- 作为量化交易全流程的协调中枢
- 解析用户意图，判断当前处于哪个投研阶段
- 根据阶段调度相应子Skill完成工作
- 管理会话状态和任务上下文
- 输出结构化的量化研究报告和交易信号
- **底层库**: 无（纯调度逻辑）

### FR-2: 市场研究Skill（market-researcher）
- 市场环境分析（大势判断、风格识别、情绪监控）
- 行业板块轮动分析
- 热点主题与概念股挖掘
- 宏观数据监控（CPI/PPI/PMI等）
- 监管政策与市场动态追踪
- **底层库**: tushare + akshare + matplotlib/plotly（可视化）

### FR-3: 数据工程Skill（data-engineering）
- 从tushare/akshare获取行情数据和基本面数据
- 数据清洗：缺失值处理、异常值检测（Winsorize）、复权处理
- 数据标准化：Point-in-Time一致性处理、因子对齐
- 本地存储：CSV/Parquet格式，规范化命名
- 支持日线、周线、分钟线数据
- **底层库**: pandas + numpy + scipy + tushare

### FR-4: Alpha因子研究Skill（alpha-researcher）
- 单Alpha因子构建与评估（动量、价值、质量、情绪等）
- Alpha因子IC（Information Coefficient）分析
- 因子相关性分析与去冗余
- 多Alpha融合与权重分配
- Alpha因子报告生成
- **底层库**: ta-lib/pandas-ta + alphalens + scipy

### FR-5: 策略开发Skill（strategy-developer）
- 策略模板库（趋势跟踪、均值回归、套利、配对交易等）
- 策略idea生成与评估框架
- 参数优化（网格搜索、随机搜索）
- 策略组合分析（相关性分析、风险贡献）
- 策略文档自动生成
- **底层库**: backtrader + scikit-learn（可选）

### FR-6: 回测验证Skill（backtest-validator）
- 事件驱动回测引擎
- 绩效指标计算（年化收益、夏普比率、最大回撤、胜率、盈亏比等）
- 过拟合检测（蒙特卡洛模拟、Walk-Forward分析）
- 交易成本建模（佣金、滑点、冲击成本）
- 回测报告生成与可视化
- **底层库**: backtrader + quantstats + empyrical + matplotlib/plotly

### FR-7: 组合优化Skill（portfolio-optimizer）
- 风险模型构建（协方差矩阵估计）
- 组合权重优化（风险平价、等权、最小方差、最大夏普）
- Barra风格因子归因
- 行业/风格暴露分析
- 换手率与交易成本优化
- **底层库**: pypfopt + cvxpy + scipy

### FR-8: 风险管理层Skill（risk-manager）
- 持仓风险计算（VaR、CVaR、Expected Shortfall）
- 实时风险监控（持仓限额、保证金、盈亏预警）
- 压力测试（历史情景、假设情景模拟）
- 合规检查（持仓集中度、交易频率、涨跌停限制）
- 风险归因分析
- **底层库**: numpy + scipy + pandas + empyrical

### FR-9: 交易执行Skill（execution-trader）
- 模拟账户管理（资金初始化、状态跟踪）
- 订单模拟执行（市价单、限价单、止损单）
- 实盘接口框架（预留接口，支持扩展到券商API）
- 仓位同步与成本计算
- 交易日志记录
- **底层库**: pandas + numpy（模拟逻辑）

### FR-10: 自动化安装脚本
- 检测本地是否已安装OpenClaw
- 如未安装，自动完成OpenClaw安装（Linux/macOS）
- 自动导入所有Skill到正确目录
- 环境变量配置指导
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
- 核心依赖：pandas, numpy, scipy, backtrader, quantstats, pypfopt, tushare, akshare
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
4. 用户主要关注中国A股市场和股指期货（中低频策略）
5. 用户具备基本的量化投资知识

## 验收标准

### AC-1: 主Skill协调功能
- **Given**: 用户请求进行完整的量化投研任务
- **When**: 激活quant-trading-master Skill
- **Then**: 正确解析任务所处阶段，按流程依次调度子Skill完成工作
- **Verification**: `programmatic` - 主Skill能正确解析任务类型并调用对应子Skill

### AC-2: 市场研究功能
- **Given**: 用户询问市场环境或板块机会
- **When**: 调用market-researcher Skill
- **Then**: 返回市场分析报告和关键指标
- **Verification**: `programmatic` - 能正确调用tushare/akshare接口并分析数据

### AC-3: 数据工程功能
- **Given**: 用户指定股票池、时间范围和数据类型
- **When**: 调用data-engineering Skill获取数据
- **Then**: 返回清洗后的标准化数据并保存到本地
- **Verification**: `programmatic` - 数据格式正确、缺失值处理合理

### AC-4: Alpha因子研究功能
- **Given**: 用户提供候选因子或要求生成Alpha
- **When**: 调用alpha-researcher Skill进行评估
- **Then**: 输出因子IC序列、相关性矩阵、融合建议
- **Verification**: `programmatic` - IC计算正确、报告格式完整

### AC-5: 策略开发功能
- **Given**: 用户提供策略思路或选择策略模板
- **When**: 调用strategy-developer Skill开发策略
- **Then**: 生成可执行的策略代码和参数优化结果
- **Verification**: `programmatic` - 策略代码可运行、参数优化有结果

### AC-6: 回测验证功能
- **Given**: 用户提供策略参数和回测时间范围
- **When**: 调用backtest-validator Skill执行回测
- **Then**: 生成包含关键指标的绩效报告和可视化图表
- **Verification**: `programmatic` - 绩效指标计算正确、图表正确生成

### AC-7: 组合优化功能
- **Given**: 用户提供候选股票池和风险偏好
- **When**: 调用portfolio-optimizer Skill优化组合
- **Then**: 输出最优权重分配和风险分析
- **Verification**: `programmatic` - 优化算法收敛、权重分配合理

### AC-8: 风险管理功能
- **Given**: 用户提供持仓数据或策略信号
- **When**: 调用risk-manager Skill进行评估
- **Then**: 输出风险指标和风控建议
- **Verification**: `programmatic` - VaR计算正确、告警阈值生效

### AC-9: 模拟交易功能
- **Given**: 用户启动模拟交易模式
- **When**: 调用execution-trader Skill（模拟模式）
- **Then**: 模拟账户正确执行信号并跟踪持仓
- **Verification**: `programmatic` - 订单执行正确、持仓计算准确

### AC-10: 安装脚本功能
- **Given**: 在未安装OpenClaw的Linux/macOS环境
- **When**: 运行自动化安装脚本
- **Then**: 自动完成OpenClaw安装和Skill导入
- **Verification**: `programmatic` - 脚本执行成功、Skill可被识别

### AC-11: Skill文件格式
- **Given**: 每个Skill项目
- **When**: 检查文件结构
- **Then**: 符合Anthropic标准（SKILL.md + scripts/ + references/）
- **Verification**: `programmatic` - 目录结构验证、YAML格式正确

## 开放问题

- [ ] 实盘交易接口具体对接哪家券商？（需要用户确认，如QMT、Ptrade、东方财富等）
- [ ] 是否需要支持期权策略相关模块？
- [ ] 是否需要集成机器学习模型训练框架（scikit-learn、lightgbm）？
- [ ] 是否需要支持多策略组合同时运行？
