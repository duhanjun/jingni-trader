# 量化交易Agent Skill技能套件 - 验证清单

## Task 1: 项目结构设计与主Skill开发（quant-trading-master）
- [ ] Checkpoint 1.1: 项目根目录 `quantitative-trading-skills/` 创建成功
- [ ] Checkpoint 1.2: 主Skill目录 `skills/quant-trading-master/SKILL.md` 格式正确
- [ ] Checkpoint 1.3: 主Skill包含 `scripts/main_workflow.py`
- [ ] Checkpoint 1.4: 主Skill包含 `references/` 目录
- [ ] Checkpoint 1.5: SKILL.md包含正确的YAML frontmatter（name, description字段）
- [ ] Checkpoint 1.6: 工作流程说明文档完整

## Task 2: 市场研究Skill开发（market-researcher）
- [ ] Checkpoint 2.1: Skill目录 `skills/market-researcher/SKILL.md` 存在
- [ ] Checkpoint 2.2: `scripts/market_analyzer.py` 能分析市场环境
- [ ] Checkpoint 2.3: `scripts/sector_analyzer.py` 能分析行业板块
- [ ] Checkpoint 2.4: `scripts/macro_monitor.py` 能获取宏观数据
- [ ] Checkpoint 2.5: `references/market_analysis_guide.md` 市场分析框架完整
- [ ] Checkpoint 2.6: 市场分析功能测试通过

## Task 3: 数据工程Skill开发（data-engineering）
- [ ] Checkpoint 3.1: Skill目录 `skills/data-engineering/SKILL.md` 存在
- [ ] Checkpoint 3.2: `scripts/data_fetcher.py` 能获取Tushare数据
- [ ] Checkpoint 3.3: `scripts/data_cleaner.py` 能处理缺失值和异常值
- [ ] Checkpoint 3.4: `scripts/data_standardizer.py` 能处理复权和PIT一致性
- [ ] Checkpoint 3.5: `scripts/data_storage.py` 能导出CSV/Parquet格式
- [ ] Checkpoint 3.6: `references/tushare_api.md` API文档完整
- [ ] Checkpoint 3.7: 数据清洗功能测试通过

## Task 4: Alpha因子研究Skill开发（alpha-researcher）
- [ ] Checkpoint 4.1: Skill目录 `skills/alpha-researcher/SKILL.md` 存在
- [ ] Checkpoint 4.2: `scripts/factor_builder.py` 包含至少5种因子计算
- [ ] Checkpoint 4.3: `scripts/ic_analyzer.py` 能计算IC和ICIR
- [ ] Checkpoint 4.4: `scripts/correlation_analyzer.py` 能计算相关性矩阵
- [ ] Checkpoint 4.5: `scripts/alpha_fusion.py` 能融合多个Alpha
- [ ] Checkpoint 4.6: `references/factor_guide.md` 因子说明文档完整
- [ ] Checkpoint 4.7: 因子分析报告能正确生成

## Task 5: 策略开发Skill开发（strategy-developer）
- [ ] Checkpoint 5.1: Skill目录 `skills/strategy-developer/SKILL.md` 存在
- [ ] Checkpoint 5.2: `scripts/strategy_template.py` 包含至少3种策略模板
- [ ] Checkpoint 5.3: `scripts/strategy_generator.py` 能生成策略idea
- [ ] Checkpoint 5.4: `scripts/optimizer.py` 能进行参数优化
- [ ] Checkpoint 5.5: `scripts/portfolio_analyzer.py` 能分析策略组合
- [ ] Checkpoint 5.6: `scripts/doc_generator.py` 能生成策略文档
- [ ] Checkpoint 5.7: `references/strategy_guide.md` 策略指南完整

## Task 6: 回测验证Skill开发（backtest-validator）
- [ ] Checkpoint 6.1: Skill目录 `skills/backtest-validator/SKILL.md` 存在
- [ ] Checkpoint 6.2: `scripts/backtest_engine.py` 能执行回测
- [ ] Checkpoint 6.3: `scripts/performance.py` 能计算夏普比率、最大回撤等
- [ ] Checkpoint 6.4: `scripts/overfit_detector.py` 能检测过拟合
- [ ] Checkpoint 6.5: `scripts/cost_model.py` 能建模交易成本
- [ ] Checkpoint 6.6: `scripts/report_generator.py` 能生成回测报告
- [ ] Checkpoint 6.7: `scripts/visualizer.py` 能生成图表
- [ ] Checkpoint 6.8: `references/backtest_guide.md` 回测说明完整

## Task 7: 组合优化Skill开发（portfolio-optimizer）
- [ ] Checkpoint 7.1: Skill目录 `skills/portfolio-optimizer/SKILL.md` 存在
- [ ] Checkpoint 7.2: `scripts/covariance.py` 能估计协方差矩阵
- [ ] Checkpoint 7.3: `scripts/optimizer.py` 能优化组合权重
- [ ] Checkpoint 7.4: `scripts/risk_attribution.py` 能进行Barra归因
- [ ] Checkpoint 7.5: `scripts/turnover.py` 能优化换手率
- [ ] Checkpoint 7.6: `references/portfolio_guide.md` 组合优化说明完整

## Task 8: 风险管理层Skill开发（risk-manager）
- [ ] Checkpoint 8.1: Skill目录 `skills/risk-manager/SKILL.md` 存在
- [ ] Checkpoint 8.2: `scripts/var_calculator.py` 能计算VaR/CVaR
- [ ] Checkpoint 8.3: `scripts/realtime_monitor.py` 能实时监控风险
- [ ] Checkpoint 8.4: `scripts/stress_test.py` 能执行压力测试
- [ ] Checkpoint 8.5: `scripts/compliance_check.py` 能检查合规
- [ ] Checkpoint 8.6: `scripts/risk_attribution.py` 能进行风险归因
- [ ] Checkpoint 8.7: `references/risk_guide.md` 风控说明完整

## Task 9: 交易执行Skill开发（execution-trader）
- [ ] Checkpoint 9.1: Skill目录 `skills/execution-trader/SKILL.md` 存在
- [ ] Checkpoint 9.2: `scripts/account_manager.py` 能管理模拟账户
- [ ] Checkpoint 9.3: `scripts/order_executor.py` 能执行模拟订单
- [ ] Checkpoint 9.4: `scripts/position_tracker.py` 能跟踪持仓
- [ ] Checkpoint 9.5: `scripts/broker_interface.py` 券商接口框架完整
- [ ] Checkpoint 9.6: `scripts/trade_logger.py` 交易日志记录完整
- [ ] Checkpoint 9.7: `references/execution_guide.md` 执行说明完整

## Task 10: 自动化安装脚本开发
- [ ] Checkpoint 10.1: `scripts/install.sh` 能检测OpenClaw是否安装
- [ ] Checkpoint 10.2: `scripts/install.sh` 能自动安装OpenClaw（Linux/macOS）
- [ ] Checkpoint 10.3: `scripts/import_skills.sh` 能导入所有Skill
- [ ] Checkpoint 10.4: `scripts/setup_env.sh` 能指导环境变量配置
- [ ] Checkpoint 10.5: `scripts/check_dependencies.py` 能检查依赖
- [ ] Checkpoint 10.6: README.md 包含完整使用指南

## Task 11: 环境变量和安全配置
- [ ] Checkpoint 11.1: `.env.example` 文件存在且包含所有必需变量
- [ ] Checkpoint 11.2: 所有脚本从环境变量读取API密钥
- [ ] Checkpoint 11.3: 缺少环境变量时有明确错误提示
- [ ] Checkpoint 11.4: SECURITY.md 安全配置说明完整

## Task 12: 项目文档与测试
- [ ] Checkpoint 12.1: `requirements.txt` 包含所有Python依赖
- [ ] Checkpoint 12.2: 每个Skill包含测试脚本
- [ ] Checkpoint 12.3: 代码遵循PEP 8规范
- [ ] Checkpoint 12.4: 所有Python脚本包含中文注释
- [ ] Checkpoint 12.5: 项目README完整且易于理解
- [ ] Checkpoint 12.6: 所有SKILL.md格式符合Anthropic标准
