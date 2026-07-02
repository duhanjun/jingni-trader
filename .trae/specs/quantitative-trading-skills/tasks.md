# 量化交易Agent Skill技能套件 - 实施计划（A股专属版）

## 技术栈映射表

| Skill | 底层主要Python库 | A股适配要点 |
|-------|-------------------|-------------|
| quant-trading-master | 无（纯调度） | - |
| a-share-data-engine | tushare, baostock, akshare, xtquant, gm, pandas, sqlalchemy | 数据本土化、ST/涨跌停标记、多数据源切换 |
| a-share-factor-engine | ta-lib/pandas-ta, alphalens, scikit-learn | A股因子体系、行业中性化IC |
| strategy-model-engine | lightgbm, catboost, optuna, mlflow | 截面选股、择时模型 |
| backtest-engine | rqalpha, backtrader, gm, quantstats | T+1、涨跌停、费用模拟、多回测引擎切换 |
| portfolio-risk-engine | pypfopt, riskfolio-lib, numpy | A股约束、止损机制 |
| execution-monitor-engine | pandas, numpy（模拟）, xtquant, gm, vnpy（实盘） | 模拟交易、实盘接口框架（xtquant/gm/vnpy切换） |
| reports-engine | quantstats, plotly, matplotlib | A股风格暴露、Brinson归因 |

## [ ] Task 1: 项目结构设计与主Skill开发（quant-trading-master）
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 创建项目根目录结构
  - 开发主Skill（quant-trading-master）：协调中枢，负责全流程任务编排
  - 定义主Skill与子Skill的通信协议和数据格式
  - 开发阶段状态机实现
  - 开发Context对象标准化实现
  - 开发里程碑检查点机制
  - 编写主Skill的SKILL.md、scripts/main_workflow.py、references/流程说明
- **Acceptance Criteria Addressed**: AC-1, AC-11
- **底层库**: 无（纯调度逻辑）
- **Test Requirements**:
  - `programmatic` TR-1.1: 验证主Skill的SKILL.md格式符合Anthropic标准
  - `programmatic` TR-1.2: 验证项目目录结构正确
  - `programmatic` TR-1.3: 阶段状态机能正确执行流程
  - `programmatic` TR-1.4: Context对象能正确传递
  - `human-judgement` TR-1.5: 检查主Skill的工作流说明是否清晰完整
- **Notes**: 主Skill是整个系统的核心，需要精心设计任务分解逻辑和阶段判断机制

## [ ] Task 2: A股数据采集与治理Skill开发（a-share-data-engine）
- **Priority**: P0
- **Depends On**: Task 1
- **Description**:
  - 开发BaseDataProvider抽象基类（统一接口）
  - 开发各数据源适配器：
    - Tushare适配器
    - BaoStock适配器
    - AkShare适配器
    - xtquant适配器
    - gm适配器
  - 开发数据清洗脚本：复权处理、停牌标记、涨跌停板标记（limit_flag）
  - 开发ST/退市风险过滤脚本：每日维护st_list
  - 开发新股过滤脚本：剔除上市不足60个交易日的新股
  - 开发数据存储脚本：SQLite/MySQL/PostgreSQL持久化，规范化命名
  - 开发数据快照（Snapshot）管理
  - 编写references文档：API调用说明、数据字典、A股清洗规则
- **Acceptance Criteria Addressed**: AC-2, AC-11
- **底层库**: tushare, baostock, akshare, xtquant, gm, pandas, sqlalchemy, numpy
- **Test Requirements**:
  - `programmatic` TR-2.1: 数据获取脚本能正确获取日线数据（tushare）
  - `programmatic` TR-2.2: 涨跌停标记字段正确生成
  - `programmatic` TR-2.3: ST过滤能正确剔除ST股票
  - `programmatic` TR-2.4: 新股过滤能正确剔除上市不足60日的新股
  - `programmatic` TR-2.5: 数据库存储功能正常
  - `programmatic` TR-2.6: 清洗后的日线数据缺失率 < 2%
  - `programmatic` TR-2.7: 支持多数据源（tushare/baostock/akshare/xtquant/gm）切换
- **Notes**: 复用已有的tushare skill设计，整合baostock/akshare/xtquant/gm作为补充

## [ ] Task 3: A股阿尔法因子库Skill开发（a-share-factor-engine）
- **Priority**: P0
- **Depends On**: Task 2
- **Description**:
  - 开发单Alpha因子构建脚本：动量、价值、质量、情绪等因子
  - 开发A股专用因子脚本：1个月反转、lncap市值、换手率、资金流、事件因子
  - 开发因子IC分析脚本：IC序列计算、ICIR分析（行业中性化处理）
  - 开发因子相关性分析脚本：相关性矩阵、去冗余建议
  - 开发多Alpha融合脚本：等权融合、IC加权、风险归因加权
  - 开发因子存储与复用：Parquet文件，支持增量更新
  - 开发对抗性验证（可选）：训练/测试分布一致性检查
  - 编写references文档：因子说明、IC分析指南（A股分组规则）、多因子融合方法
- **Acceptance Criteria Addressed**: AC-3, AC-11
- **底层库**: ta-lib, pandas-ta, alphalens, pandas, scikit-learn, numpy, scipy
- **Test Requirements**:
  - `programmatic` TR-3.1: 因子计算结果与已知公式一致（ta-lib/pandas-ta切换）
  - `programmatic` TR-3.2: IC计算正确（含行业中性化处理）
  - `programmatic` TR-3.3: 相关性矩阵正确
  - `programmatic` TR-3.4: A股专用因子（反转、市值、换手率）正确计算
  - `programmatic` TR-3.5: 因子Parquet文件正确存储，支持增量更新
  - `human-judgement` TR-3.6: 检查因子分析报告格式完整性
- **Notes**: 参考WorldQuant Alpha表达式和Alphalens框架，需修改分组函数按中证全指/沪深300分行业分层

## [ ] Task 4: A股策略开发与模型训练Skill开发（strategy-model-engine）
- **Priority**: P1
- **Depends On**: Task 3
- **Description**:
  - 开发策略模板生成脚本：趋势跟踪、均值回归、配对交易等模板
  - 开发截面选股模型脚本：LightGBM/CatBoost多因子选股
  - 开发择时模型脚本：基于分钟线技术指标的3日涨跌分类器
  - 开发参数优化脚本：Optuna超参搜索
  - 开发实验管理脚本：MLflow记录因子组合和模型参数
  - 开发过拟合防范脚本：分组时序交叉验证（Purged Group Time Series Split）
  - 编写references文档：策略模板说明、模型使用指南、最佳实践
- **Acceptance Criteria Addressed**: AC-4, AC-11
- **底层库**: lightgbm, catboost, scikit-learn, optuna, mlflow, numpy, pandas
- **Test Requirements**:
  - `programmatic` TR-4.1: 能生成至少3种策略模板
  - `programmatic` TR-4.2: LightGBM选股模型能正常运行
  - `programmatic` TR-4.3: Optuna参数优化能找到预设最优解
  - `programmatic` TR-4.4: MLflow能正确记录实验
  - `programmatic` TR-4.5: 分组时序交叉验证正确实现
  - `human-judgement` TR-4.6: 检查策略文档结构完整性
- **Notes**: 策略逻辑由AI根据市场情况生成，此Skill提供框架和模板

## [ ] Task 5: A股策略回测与仿真Skill开发（backtest-engine）
- **Priority**: P0
- **Depends On**: Task 4
- **Description**:
  - 开发BaseBacktestEngine抽象基类（统一接口）
  - 开发各回测引擎适配器：
    - RQAlpha适配器
    - Backtrader适配器
    - gm适配器
  - 开发A股规则模拟脚本：T+1、涨跌停板、费用（佣金万2.5、印花税1‰、过户费、最低5元佣金）
  - 开发停牌处理脚本：资产冻结，复牌后恢复
  - 开发绩效指标计算脚本：年化收益、夏普比率、最大回撤、胜率、盈亏比（quantstats/empyrical）
  - 开发过拟合检测脚本：Walk-Forward分析
  - 开发回测报告生成脚本：Markdown/JSON格式报告
  - 开发回测可视化脚本：收益曲线、回撤曲线（matplotlib/plotly）
  - 编写references文档：RQAlpha配置说明、回测框架说明、绩效指标定义
- **Acceptance Criteria Addressed**: AC-5, AC-11
- **底层库**: rqalpha, backtrader, gm, quantstats, empyrical, matplotlib, plotly, numpy, pandas
- **Test Requirements**:
  - `programmatic` TR-5.1: RQAlpha能正确执行回测（T+1规则生效）
  - `programmatic` TR-5.2: 涨跌停板限制正确生效（严格废单/排队模型）
  - `programmatic` TR-5.3: 绩效指标计算与标准公式一致
  - `programmatic` TR-5.4: 回测报告能正确生成，包含样本外（OOS）指标
  - `programmatic` TR-5.5: 图表能正确生成（matplotlib/plotly）
  - `programmatic` TR-5.6: 支持多回测引擎（rqalpha/backtrader/gm）切换
  - `programmatic` TR-5.7: 使用标准策略回测，误差 < 1%，无规则突破成交
  - `human-judgement` TR-5.8: 检查回测报告格式完整性
- **Notes**: RQAlpha是A股回测首选，内置涨跌停、T+1、手续费；Backtrader/gm作为辅助

## [ ] Task 6: 组合优化与风控Skill开发（portfolio-risk-engine）
- **Priority**: P0
- **Depends On**: Task 5
- **Description**:
  - 开发协方差矩阵估计脚本：历史协方差、指数加权协方差 shrinkage（pypfopt）
  - 开发组合权重优化脚本：风险平价、最小方差、最大夏普（pypfopt）
  - 开发Barra风格因子归因脚本：行业暴露、风格暴露分析（CNE5模型）
  - 开发A股约束处理脚本：单一股票持仓不超过10%、行业暴露偏离基准±5%
  - 开发止损机制脚本：单日回撤-3%硬止损、个股放量跌破20日均线止损
  - 开发VaR/CVaR计算脚本：历史模拟法、参数法（numpy/scipy）
  - 开发实时风险监控脚本：持仓限额、保证金、盈亏预警
  - 编写references文档：优化算法说明、风险模型说明、风控规则
- **Acceptance Criteria Addressed**: AC-6, AC-11
- **底层库**: pypfopt, riskfolio-lib, numpy, scipy, pandas, empyrical
- **Test Requirements**:
  - `programmatic` TR-6.1: 协方差矩阵计算正确（pypfopt）
  - `programmatic` TR-6.2: 优化算法收敛，权重满足A股约束
  - `programmatic` TR-6.3: Barra归因分析结果合理
  - `programmatic` TR-6.4: 止损机制能正确触发
  - `programmatic` TR-6.5: VaR计算结果与标准方法一致
  - `human-judgement` TR-6.6: 检查优化报告格式完整性
- **Notes**: 风控是量化交易的生命线，A股约束需要严格遵守

## [ ] Task 7: A股实盘执行与监控Skill开发（execution-monitor-engine）
- **Priority**: P1
- **Depends On**: Task 6
- **Description**:
  - 开发BaseTrader抽象基类（统一接口）
  - 开发各交易接口适配器：
    - xtquant适配器
    - gm适配器
    - vnpy适配器
  - 开发模拟账户管理脚本：资金初始化、账户状态跟踪（pandas）
  - 开发订单模拟执行脚本：市价单、限价单、止损单
  - 开发本地硬断路器（Circuit Breaker）：
    - 单日累计已执行订单的浮动盈亏 + 已实现盈亏，若超过`max_daily_loss`（例如净资产2%），直接拒绝所有新开仓订单，并平掉部分仓位
    - 单笔订单金额不得超过账户净资产的一定比例
    - 每秒/每分钟订单数限制（防刷单）
    - 持仓集中度限制（单票不超过配置上限）
  - 开发确认模式与演习模式：
    - 仿真模式：连接真实行情但下单到模拟撮合器
    - 一键切换到paper_trade，与实盘采用同一套代码路径
  - 开发审计日志完整性：所有订单（包括被风控拒绝的单子）都必须记录完整上下文
  - 开发仓位同步与成本计算
  - 开发交易日志脚本：完整交易记录、审计日志
  - 开发风控守护进程：单日亏损超阈值报警、持仓集中度监控
  - 编写references文档：模拟交易规则、订单类型说明、xtquant/gm/vnpy对接说明
- **Acceptance Criteria Addressed**: AC-7, AC-8, AC-11
- **底层库**: pandas, numpy（模拟逻辑），xtquant, gm, vnpy（实盘扩展）
- **Test Requirements**:
  - `programmatic` TR-7.1: 模拟账户能正确执行买入/卖出信号
  - `programmatic` TR-7.2: 持仓计算正确（平均成本、盈亏）
  - `programmatic` TR-7.3: 模拟绩效统计与实际交易一致
  - `programmatic` TR-7.4: 交易日志记录完整
  - `programmatic` TR-7.5: 本地硬断路器能正确触发
  - `programmatic` TR-7.6: 风控守护进程能正确触发报警
  - `programmatic` TR-7.7: 支持多交易接口（xtquant/gm/vnpy）切换
  - `programmatic` TR-7.8: 在xtquant/gm仿真环境连续运行5个交易日，无漏单错单；资金误差 < 万分之一
  - `human-judgement` TR-7.9: 检查实盘接口框架设计合理性
- **Notes**: 模拟交易是实盘前的重要验证环节，实盘接口需要用户根据实际券商API实现

## [ ] Task 8: 绩效归因与可视化报告Skill开发（reports-engine）
- **Priority**: P1
- **Depends On**: Task 6
- **Description**:
  - 开发日收益报告脚本：quantstats一键生成
  - 开发滚动夏普、月度热力图脚本
  - 开发A股风格暴露分析脚本：大盘/小盘/成长/价值暴露（沪深300基准）
  - 开发行业暴露图脚本：申万一级行业暴露
  - 开发Brinson归因脚本：超额收益分解
  - 开发动态净值曲线脚本：plotly交互图表
  - 开发因子衰减走势图脚本：matplotlib静态图表
  - 开发过拟合风险警示：OOS夏普低于全样本夏普的60%时自动标记警告
  - 编写references文档：报告说明、归因方法、图表使用指南
- **Acceptance Criteria Addressed**: AC-9, AC-11
- **底层库**: quantstats, plotly, matplotlib, pandas, numpy
- **Test Requirements**:
  - `programmatic` TR-8.1: quantstats报告正确生成
  - `programmatic` TR-8.2: A股风格暴露分析正确
  - `programmatic` TR-8.3: Brinson归因计算正确
  - `programmatic` TR-8.4: plotly图表能正确生成
  - `programmatic` TR-8.5: 过拟合风险警告正确触发
  - `human-judgement` TR-8.6: 检查报告格式完整性
- **Notes**: 报告需要符合国内投资人审阅习惯

## [ ] Task 9: 自动化安装脚本开发
- **Priority**: P0
- **Depends On**: Task 1-8
- **Description**:
  - 开发OpenClaw检测脚本：检测系统是否已安装OpenClaw
  - 开发OpenClaw安装脚本：支持Linux/macOS自动安装
  - 开发Skill导入脚本：将所有Skill复制到正确目录
  - 开发环境变量配置脚本：指导用户配置TUSHARE_TOKEN、GM_TOKEN等
  - 开发依赖检查脚本：检查Python版本、必需包，支持选择性安装xtquant/gm/vnpy等
  - 编写许可与合规提示：加入对Tushare、xtquant、gm等库许可协议的重大提醒
  - 编写README说明：完整使用指南
- **Acceptance Criteria Addressed**: AC-10
- **Test Requirements**:
  - `programmatic` TR-9.1: 检测脚本能正确识别OpenClaw安装状态
  - `programmatic` TR-9.2: 安装脚本能在干净环境完成安装
  - `programmatic` TR-9.3: Skill导入脚本能将所有Skill导入到正确位置
  - `programmatic` TR-9.4: 依赖检查能正确识别缺失项，支持选择性安装
  - `human-judgement` TR-9.5: 检查README文档完整性
- **Notes**: 安装脚本是用户体验的关键，需要健壮的错误处理

## [ ] Task 10: 环境变量和安全配置
- **Priority**: P0
- **Depends On**: Task 9
- **Description**:
  - 设计统一的环境变量规范
  - 创建环境变量模板文件（.env.example）
  - 开发环境变量检查脚本
  - 编写安全配置说明文档（SECURITY.md）
- **Acceptance Criteria Addressed**: NFR-3
- **Test Requirements**:
  - `programmatic` TR-10.1: 所有脚本能从环境变量正确读取API密钥
  - `programmatic` TR-10.2: 缺少环境变量时脚本能给出明确提示
- **Notes**: 安全是底线，API密钥不能硬编码

## [ ] Task 11: 项目文档与测试
- **Priority**: P1
- **Depends On**: Task 1-10
- **Description**:
  - 编写完整项目README
  - 编写每个Skill的使用说明
  - 创建测试用例和测试数据
  - 进行代码审查和优化
  - 创建requirements.txt（包含所有Python依赖）
- **Acceptance Criteria Addressed**: All
- **Test Requirements**:
  - `programmatic` TR-11.1: 所有脚本能正常运行
  - `programmatic` TR-11.2: 文档示例代码能正确执行
  - `programmatic` TR-11.3: requirements.txt包含所有依赖（pandas, numpy, scipy, rqalpha, backtrader, lightgbm, tushare, akshare, xtquant, gm, quantstats, pypfopt等）
  - `human-judgement` TR-11.4: 代码审查通过
  - `human-judgement` TR-11.5: README完整且易于理解
- **Notes**: 好的文档能大大降低使用门槛
