# 量化交易Agent Skill技能套件 - 验证清单（A股专属版）

## Task 1: 项目结构设计与主Skill开发（quant-trading-master）
- [ ] Checkpoint 1.1: 项目根目录 `quantitative-trading-skills/` 创建成功
- [ ] Checkpoint 1.2: 主Skill目录 `skills/quant-trading-master/SKILL.md` 格式正确
- [ ] Checkpoint 1.3: 主Skill包含 `scripts/main_workflow.py`
- [ ] Checkpoint 1.4: 主Skill包含 `references/` 目录
- [ ] Checkpoint 1.5: SKILL.md包含正确的YAML frontmatter（name, description字段）
- [ ] Checkpoint 1.6: 工作流程说明文档完整（包含9个阶段）

## Task 2: A股数据采集与治理Skill开发（a-share-data-engine）
- [ ] Checkpoint 2.1: Skill目录 `skills/a-share-data-engine/SKILL.md` 存在
- [ ] Checkpoint 2.2: `scripts/data_fetcher.py` 能获取Tushare数据
- [ ] Checkpoint 2.3: `scripts/data_cleaner.py` 能处理复权和涨跌停标记
- [ ] Checkpoint 2.4: `scripts/st_filter.py` 能过滤ST股票
- [ ] Checkpoint 2.5: `scripts/new_stock_filter.py` 能过滤新股
- [ ] Checkpoint 2.6: `scripts/data_storage.py` 能存储到数据库
- [ ] Checkpoint 2.7: `references/` API文档完整
- [ ] Checkpoint 2.8: 数据清洗功能测试通过

## Task 3: A股阿尔法因子库Skill开发（a-share-factor-engine）
- [ ] Checkpoint 3.1: Skill目录 `skills/a-share-factor-engine/SKILL.md` 存在
- [ ] Checkpoint 3.2: `scripts/factor_builder.py` 包含至少5种因子计算（ta-lib）
- [ ] Checkpoint 3.3: `scripts/a_share_factors.py` 包含A股专用因子（反转、市值、换手率）
- [ ] Checkpoint 3.4: `scripts/ic_analyzer.py` 能计算IC和ICIR（含行业中性化）
- [ ] Checkpoint 3.5: `scripts/correlation_analyzer.py` 能计算相关性矩阵
- [ ] Checkpoint 3.6: `scripts/alpha_fusion.py` 能融合多个Alpha
- [ ] Checkpoint 3.7: `references/factor_guide.md` 因子说明文档完整
- [ ] Checkpoint 3.8: 因子分析报告能正确生成

## Task 4: A股策略开发与模型训练Skill开发（strategy-model-engine）
- [ ] Checkpoint 4.1: Skill目录 `skills/strategy-model-engine/SKILL.md` 存在
- [ ] Checkpoint 4.2: `scripts/strategy_template.py` 包含至少3种策略模板
- [ ] Checkpoint 4.3: `scripts/stock_selector.py` LightGBM选股模型能正常运行
- [ ] Checkpoint 4.4: `scripts/timing_model.py` 择时模型能正常运行
- [ ] Checkpoint 4.5: `scripts/optimizer.py` Optuna参数优化能找到最优解
- [ ] Checkpoint 4.6: `scripts/experiment_tracker.py` MLflow能记录实验
- [ ] Checkpoint 4.7: `scripts/overfit_prevention.py` 过拟合防范脚本存在
- [ ] Checkpoint 4.8: `references/strategy_guide.md` 策略指南完整

## Task 5: A股策略回测与仿真Skill开发（backtest-engine）
- [ ] Checkpoint 5.1: Skill目录 `skills/backtest-engine/SKILL.md` 存在
- [ ] Checkpoint 5.2: `scripts/rqalpha_backtest.py` RQAlpha回测能正确执行
- [ ] Checkpoint 5.3: `scripts/backtrader_backtest.py` Backtrader回测能正确执行
- [ ] Checkpoint 5.4: `scripts/a_share_rules.py` A股规则（T+1、涨跌停）正确模拟
- [ ] Checkpoint 5.5: `scripts/cost_model.py` 交易成本模型正确
- [ ] Checkpoint 5.6: `scripts/performance.py` 绩效指标计算正确（quantstats/empyrical）
- [ ] Checkpoint 5.7: `scripts/walk_forward.py` Walk-Forward分析正确
- [ ] Checkpoint 5.8: `scripts/report_generator.py` 回测报告能正确生成
- [ ] Checkpoint 5.9: `scripts/visualizer.py` 图表能正确生成（matplotlib/plotly）
- [ ] Checkpoint 5.10: `references/backtest_guide.md` 回测说明完整

## Task 6: 组合优化与风控Skill开发（portfolio-risk-engine）
- [ ] Checkpoint 6.1: Skill目录 `skills/portfolio-risk-engine/SKILL.md` 存在
- [ ] Checkpoint 6.2: `scripts/covariance.py` 协方差矩阵计算正确（pypfopt）
- [ ] Checkpoint 6.3: `scripts/optimizer.py` 组合优化算法收敛，权重满足A股约束
- [ ] Checkpoint 6.4: `scripts/barra_attribution.py` Barra风格归因正确
- [ ] Checkpoint 6.5: `scripts/a_share_constraints.py` A股约束处理正确
- [ ] Checkpoint 6.6: `scripts/stop_loss.py` 止损机制能正确触发
- [ ] Checkpoint 6.7: `scripts/var_calculator.py` VaR/CVaR计算正确
- [ ] Checkpoint 6.8: `scripts/realtime_monitor.py` 实时风险监控正常
- [ ] Checkpoint 6.9: `references/portfolio_guide.md` 组合优化说明完整

## Task 7: A股实盘执行与监控Skill开发（execution-monitor-engine）
- [ ] Checkpoint 7.1: Skill目录 `skills/execution-monitor-engine/SKILL.md` 存在
- [ ] Checkpoint 7.2: `scripts/account_manager.py` 模拟账户管理正常
- [ ] Checkpoint 7.3: `scripts/order_executor.py` 订单执行正确
- [ ] Checkpoint 7.4: `scripts/position_tracker.py` 持仓计算正确
- [ ] Checkpoint 7.5: `scripts/broker_interface.py` 实盘接口框架完整（vnpy/XTP）
- [ ] Checkpoint 7.6: `scripts/trade_logger.py` 交易日志记录完整
- [ ] Checkpoint 7.7: `scripts/risk_guardian.py` 风控守护进程正常
- [ ] Checkpoint 7.8: `references/execution_guide.md` 执行说明完整

## Task 8: 绩效归因与可视化报告Skill开发（reports-engine）
- [ ] Checkpoint 8.1: Skill目录 `skills/reports-engine/SKILL.md` 存在
- [ ] Checkpoint 8.2: `scripts/daily_report.py` quantstats日报告正确生成
- [ ] Checkpoint 8.3: `scripts/rolling_sharpe.py` 滚动夏普计算正确
- [ ] Checkpoint 8.4: `scripts/style_analysis.py` A股风格暴露分析正确（沪深300基准）
- [ ] Checkpoint 8.5: `scripts/industry_attribution.py` 行业暴露分析正确（申万一级）
- [ ] Checkpoint 8.6: `scripts/brinson_attribution.py` Brinson归因正确
- [ ] Checkpoint 8.7: `scripts/equity_curve.py` plotly动态净值曲线正确
- [ ] Checkpoint 8.8: `scripts/factor_decay.py` 因子衰减图正确
- [ ] Checkpoint 8.9: `references/report_guide.md` 报告说明完整

## Task 9: 自动化安装脚本开发
- [ ] Checkpoint 9.1: `scripts/install.sh` 能检测OpenClaw是否安装
- [ ] Checkpoint 9.2: `scripts/install.sh` 能自动安装OpenClaw（Linux/macOS）
- [ ] Checkpoint 9.3: `scripts/import_skills.sh` 能导入所有Skill
- [ ] Checkpoint 9.4: `scripts/setup_env.sh` 能指导环境变量配置
- [ ] Checkpoint 9.5: `scripts/check_dependencies.py` 能检查依赖并自动安装
- [ ] Checkpoint 9.6: README.md 包含完整使用指南

## Task 10: 环境变量和安全配置
- [ ] Checkpoint 10.1: `.env.example` 文件存在且包含所有必需变量
- [ ] Checkpoint 10.2: 所有脚本从环境变量读取API密钥
- [ ] Checkpoint 10.3: 缺少环境变量时有明确错误提示
- [ ] Checkpoint 10.4: SECURITY.md 安全配置说明完整

## Task 11: 项目文档与测试
- [ ] Checkpoint 11.1: `requirements.txt` 包含所有Python依赖
- [ ] Checkpoint 11.2: 每个Skill包含测试脚本
- [ ] Checkpoint 11.3: 代码遵循PEP 8规范
- [ ] Checkpoint 11.4: 所有Python脚本包含中文注释
- [ ] Checkpoint 11.5: 项目README完整且易于理解
- [ ] Checkpoint 11.6: 所有SKILL.md格式符合Anthropic标准
