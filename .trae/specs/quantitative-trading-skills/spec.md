# 量化交易Agent Skill技能套件 - 产品需求文档

## 概述
- **Summary**: 构建一套完整的量化交易Agent Skill技能套件，基于专业量化交易机构的标准架构，覆盖A股市场和股指期货的全流程量化交易工作。主Skill作为协调中枢，调用各个子Skill完成从数据处理、因子挖掘、策略回测、风险评估到模拟/实盘交易的全流程。
- **Purpose**: 为OpenClaw系统提供专业级量化交易能力，通过模块化的Skill设计实现策略研究的自动化、标准化和可复用性。
- **Target Users**: 量化交易研究者、宽客、量化基金团队

## 目标
1. 构建一个主Skill（quantitative-trading-master）作为协调中枢，负责全流程任务编排
2. 构建6个子Skill覆盖量化交易核心模块：
   - 数据处理Skill（data-processor）
   - 因子挖掘与回测Skill（factor-backtester）
   - 策略研究Skill（strategy-researcher）
   - 风险评估Skill（risk-evaluator）
   - 模拟交易Skill（paper-trader）
   - 实盘执行Skill（execution-trader）
3. 提供自动化安装脚本，支持自动检测和安装OpenClaw，自动导入所有Skill
4. 所有API密钥从环境变量统一读取（TUSHARE_TOKEN, XCSC_TUSHARE_TOKEN等）

## 非目标
- 不包含具体的交易策略逻辑（策略由用户或AI自主研究）
- 不提供投资建议或财务分析服务
- 不实现交易所直连交易（仅通过券商API接口模拟/实盘执行）

## 背景与上下文
- 基于Anthropic Agent Skills开放标准开发
- 兼容OpenClaw Skill系统
- 数据源以Tushare为主（xcsc-tushare作为补充）
- 专注于中国A股市场和股指期货

## 功能需求

### FR-1: 主Skill（quantitative-trading-master）
- 作为量化交易全流程的协调中枢
- 解析用户意图，分解任务并调度子Skill
- 管理会话状态和任务上下文
- 输出结构化的量化研究报告和交易信号

### FR-2: 数据处理Skill（data-processor）
- 从Tushare/xcsc-tushare获取行情数据和基本面数据
- 数据清洗、对齐、标准化处理
- 本地CSV/Parquet存储
- 支持日线、周线、分钟线数据

### FR-3: 因子挖掘与回测Skill（factor-backtester）
- 提供常用因子计算（动量、价值、质量、情绪等）
- 事件驱动/信号驱动的回测框架
- 绩效指标计算（年化收益、夏普比率、最大回撤、胜率等）
- 回测报告生成

### FR-4: 策略研究Skill（strategy-researcher）
- 策略idea生成和评估
- 策略参数优化
- 策略组合分析
- 策略文档生成

### FR-5: 风险评估Skill（risk-evaluator）
- 持仓风险计算
- 风险归因分析
- 压力测试
- VaR/CVaR计算
- 合规检查

### FR-6: 模拟交易Skill（paper-trader）
- 模拟账户管理
- 订单模拟执行
- 持仓跟踪
- 模拟绩效统计

### FR-7: 实盘执行Skill（execution-trader）
- 券商API接口对接（预留接口框架）
- 订单发送与跟踪
- 仓位同步
- 交易日志记录

### FR-8: 自动化安装脚本
- 检测本地是否已安装OpenClaw
- 如未安装，自动完成OpenClaw安装
- 自动导入所有Skill到正确目录
- 环境变量配置指导

## 非功能需求

### NFR-1: 代码规范
- 所有Python代码遵循PEP 8规范
- 使用中文注释和文档字符串
- 类型提示（Type Hints）全覆盖

### NFR-2: 错误处理
- 所有API调用包含异常处理
- 明确的错误信息和建议
- 日志记录机制

### NFR-3: 安全要求
- API密钥仅从环境变量读取
- 不在代码或日志中暴露敏感信息
- 交易操作包含二次确认机制

### NFR-4: 可扩展性
- 模块化设计，便于添加新功能
- 支持自定义因子和策略模板
- 预留MCP服务接口扩展

## 约束

### 技术约束
- Python 3.9+
- 主要依赖：pandas, numpy, akshare/tushare, backtrader
- Skill文件遵循Anthropic标准格式（SKILL.md + scripts/ + references/）

### 业务约束
- 数据仅用于学习和研究
- 遵守Tushare服务条款和积分限制
- 模拟交易不涉及真实资金

## 假设

1. 用户已具备Python基础环境
2. 用户拥有有效的Tushare Token
3. 用户使用OpenClaw作为AI助手运行环境
4. 用户主要关注中国A股和股指期货

## 验收标准

### AC-1: 主Skill功能
- **Given**: 用户请求进行量化研究任务
- **When**: 激活quantitative-trading-master Skill
- **Then**: 正确解析任务并调度相应子Skill完成工作
- **Verification**: `programmatic` - CLI命令验证

### AC-2: 数据获取功能
- **Given**: 用户指定股票代码和时间范围
- **When**: 调用data-processor Skill获取数据
- **Then**: 返回清洗后的CSV数据并保存到指定目录
- **Verification**: `programmatic` - 单元测试验证

### AC-3: 回测功能
- **Given**: 用户提供策略参数和回测时间范围
- **When**: 调用factor-backtester Skill执行回测
- **Then**: 生成包含关键指标的绩效报告
- **Verification**: `programmatic` - 回归测试验证

### AC-4: 风险评估功能
- **Given**: 用户提供持仓数据
- **When**: 调用risk-evaluator Skill进行评估
- **Then**: 输出风险指标和风险提示
- **Verification**: `programmatic` - 单元测试验证

### AC-5: 模拟交易功能
- **Given**: 用户启动模拟交易模式
- **When**: 调用paper-trader Skill
- **Then**: 模拟账户正确执行信号并跟踪持仓
- **Verification**: `programmatic` - 集成测试验证

### AC-6: 安装脚本功能
- **Given**: 在未安装OpenClaw的Linux/macOS环境
- **When**: 运行自动化安装脚本
- **Then**: 自动完成OpenClaw安装和Skill导入
- **Verification**: `programmatic` - 脚本执行验证

### AC-7: Skill文件格式
- **Given**: 每个Skill项目
- **When**: 检查文件结构
- **Then**: 符合Anthropic标准（SKILL.md + scripts/ + references/）
- **Verification**: `programmatic` - 目录结构验证

## 开放问题

- [ ] 实盘交易接口具体对接哪家券商？（需要用户确认）
- [ ] 是否需要支持多因子模型框架？（如alphalens风格）
- [ ] 是否需要回测结果可视化？（图表生成）
- [ ] 策略信号输出格式是否需要统一？（JSON/Excel/自定义模板）
