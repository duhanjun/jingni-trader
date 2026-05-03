# 量化交易Agent Skill技能套件 - 验证清单（A股专属版）

## Task 1: 项目结构设计与主Skill开发（quant-trading-master）
- [ ] Checkpoint 1.1: 项目根目录 `quantitative-trading-skills/` 创建成功
- [ ] Checkpoint 1.2: 主Skill目录 `skills/quant-trading-master/SKILL.md` 格式正确
- [ ] Checkpoint 1.3: 主Skill包含 `scripts/main_workflow.py`
- [ ] Checkpoint 1.4: 主Skill包含 `references/` 目录
- [ ] Checkpoint 1.5: SKILL.md包含正确的YAML frontmatter（name, description字段）
- [ ] Checkpoint 1.6: 阶段状态机能正确执行流程
- [ ] Checkpoint 1.7: Context对象能正确传递
- [ ] Checkpoint 1.8: 里程碑检查点机制能正确验证产品完整性

## Task 2: A股数据采集与治理Skill开发（a-share-data-engine）
- [ ] Checkpoint 2.1: Skill目录 `skills/a-share-data-engine/SKILL.md` 存在
- [ ] Checkpoint 2.2: BaseDataProvider抽象基类正确实现，统一接口定义完整
- [ ] Checkpoint 2.3: Tushare适配器正确实现
- [ ] Checkpoint 2.4: BaoStock适配器正确实现
- [ ] Checkpoint 2.5: AkShare适配器正确实现
- [ ] Checkpoint 2.6: xtquant适配器正确实现
- [ ] Checkpoint 2.7: gm适配器正确实现
- [ ] Checkpoint 2.8: 数据清洗脚本能正确处理复权、停牌标记、涨跌停板标记
- [ ] Checkpoint 2.9: ST过滤能正确剔除ST股票
- [ ] Checkpoint 2.10: 新股过滤能正确剔除上市不足60日的新股
- [ ] Checkpoint 2.11: 数据库存储功能正常（SQLite/MySQL/PostgreSQL）
- [ ] Checkpoint 2.12: 数据快照（Snapshot）管理功能正常
- [ ] Checkpoint 2.13: 清洗后的日线数据缺失率 < 2%（全A股过去5年测试）
- [ ] Checkpoint 2.14: 停牌/涨跌停标记无遗漏
- [ ] Checkpoint 2.15: 支持多数据源（tushare/baostock/akshare/xtquant/gm）切换
- [ ] Checkpoint 2.16: references/API文档完整

## Task 3: A股阿尔法因子库Skill开发（a-share-factor-engine）
- [ ] Checkpoint 3.1: Skill目录 `skills/a-share-factor-engine/SKILL.md` 存在
- [ ] Checkpoint 3.2: 单Alpha因子（动量、价值、质量、情绪等）能正确计算
- [ ] Checkpoint 3.3: A股专用因子（1个月反转、lncap市值、换手率等）正确计算
- [ ] Checkpoint 3.4: ta-lib/pandas-ta两种技术指标库支持切换
- [ ] Checkpoint 3.5: 因子IC分析能正确计算IC序列、ICIR（含行业中性化处理）
- [ ] Checkpoint 3.6: 因子相关性分析能正确计算相关性矩阵、去冗余建议
- [ ] Checkpoint 3.7: 多Alpha融合能正确计算（等权融合、IC加权、风险归因加权）
- [ ] Checkpoint 3.8: 因子Parquet文件正确存储，支持增量更新
- [ ] Checkpoint 3.9: 对抗性验证（可选）能正确检查训练/测试分布一致性
- [ ] Checkpoint 3.10: references/因子文档完整

## Task 4: A股策略开发与模型训练Skill开发（strategy-model-engine）
- [ ] Checkpoint 4.1: Skill目录 `skills/strategy-model-engine/SKILL.md` 存在
- [ ] Checkpoint 4.2: 能生成至少3种策略模板（趋势跟踪、均值回归、配对交易等）
- [ ] Checkpoint 4.3: LightGBM/CatBoost选股模型能正常运行
- [ ] Checkpoint 4.4: 择时模型（基于分钟线技术指标）能正常运行
- [ ] Checkpoint 4.5: Optuna参数优化能找到预设最优解
- [ ] Checkpoint 4.6: MLflow能正确记录实验
- [ ] Checkpoint 4.7: 分组时序交叉验证（Purged Group Time Series Split）正确实现
- [ ] Checkpoint 4.8: references/策略文档完整

## Task 5: A股策略回测与仿真Skill开发（backtest-engine）
- [ ] Checkpoint 5.1: Skill目录 `skills/backtest-engine/SKILL.md` 存在
- [ ] Checkpoint 5.2: BaseBacktestEngine抽象基类正确实现
- [ ] Checkpoint 5.3: RQAlpha适配器正确实现，能正确执行回测
- [ ] Checkpoint 5.4: Backtrader适配器正确实现
- [ ] Checkpoint 5.5: gm适配器正确实现
- [ ] Checkpoint 5.6: A股规则模拟脚本正确实现（T+1、涨跌停板、费用模型）
- [ ] Checkpoint 5.7: 涨跌停板限制正确生效（严格废单/排队模型可选）
- [ ] Checkpoint 5.8: 停牌处理脚本正确实现（资产冻结，复牌后恢复）
- [ ] Checkpoint 5.9: 交易费用模型正确实现（佣金万2.5、印花税1‰、过户费、最低5元佣金）
- [ ] Checkpoint 5.10: 绩效指标计算与标准公式一致（quantstats/empyrical）
- [ ] Checkpoint 5.11: Walk-Forward分析能正确执行
- [ ] Checkpoint 5.12: 回测报告能正确生成，包含样本外（OOS）指标
- [ ] Checkpoint 5.13: 图表能正确生成（matplotlib/plotly）
- [ ] Checkpoint 5.14: 使用标准策略回测，年化收益、夏普比率与公开参照值误差 < 1%
- [ ] Checkpoint 5.15: 在包含涨跌停的样本中，无任何一次突破规则成交
- [ ] Checkpoint 5.16: 支持多回测引擎（rqalpha/backtrader/gm）切换
- [ ] Checkpoint 5.17: references/回测文档完整

## Task 6: 组合优化与风控Skill开发（portfolio-risk-engine）
- [ ] Checkpoint 6.1: Skill目录 `skills/portfolio-risk-engine/SKILL.md` 存在
- [ ] Checkpoint 6.2: 协方差矩阵估计正确（pypfopt）
- [ ] Checkpoint 6.3: 组合权重优化算法收敛（风险平价、最小方差、最大夏普）
- [ ] Checkpoint 6.4: 权重满足A股约束（单票不超过10%、行业暴露偏离基准±5%）
- [ ] Checkpoint 6.5: Barra风格因子归因分析结果合理（CNE5模型）
- [ ] Checkpoint 6.6: 止损机制能正确触发（单日回撤-3%硬止损、个股放量跌破20日均线止损）
- [ ] Checkpoint 6.7: VaR/CVaR计算结果与标准方法一致
- [ ] Checkpoint 6.8: 实时风险监控脚本正常运行（持仓限额、保证金、盈亏预警）
- [ ] Checkpoint 6.9: references/风控文档完整

## Task 7: A股实盘执行与监控Skill开发（execution-monitor-engine）
- [ ] Checkpoint 7.1: Skill目录 `skills/execution-monitor-engine/SKILL.md` 存在
- [ ] Checkpoint 7.2: BaseTrader抽象基类正确实现
- [ ] Checkpoint 7.3: xtquant适配器正确实现
- [ ] Checkpoint 7.4: gm适配器正确实现
- [ ] Checkpoint 7.5: vnpy适配器正确实现
- [ ] Checkpoint 7.6: 模拟账户管理正常（资金初始化、账户状态跟踪）
- [ ] Checkpoint 7.7: 模拟订单执行正确（市价单、限价单、止损单）
- [ ] Checkpoint 7.8: 持仓跟踪正确（实时持仓更新、成本计算、盈亏计算）
- [ ] Checkpoint 7.9: 本地硬断路器能正确触发
- [ ] Checkpoint 7.10: 确认模式与演习模式正常（仿真模式、一键切换paper_trade）
- [ ] Checkpoint 7.11: 审计日志完整性（所有订单包括被风控拒绝的都记录完整上下文）
- [ ] Checkpoint 7.12: 风控守护进程正常运行（单日亏损超阈值报警、持仓集中度监控）
- [ ] Checkpoint 7.13: 交易日志记录完整
- [ ] Checkpoint 7.14: 支持多交易接口（xtquant/gm/vnpy）切换
- [ ] Checkpoint 7.15: 在xtquant/gm仿真环境连续运行5个交易日，无漏单错单
- [ ] Checkpoint 7.16: 每日结算后资金误差 < 万分之一
- [ ] Checkpoint 7.17: references/实盘文档完整

## Task 8: 绩效归因与可视化报告Skill开发（reports-engine）
- [ ] Checkpoint 8.1: Skill目录 `skills/reports-engine/SKILL.md` 存在
- [ ] Checkpoint 8.2: quantstats日收益报告能正确生成
- [ ] Checkpoint 8.3: 滚动夏普、月度热力图能正确生成
- [ ] Checkpoint 8.4: A股风格暴露分析正确（大盘/小盘/成长/价值，沪深300基准）
- [ ] Checkpoint 8.5: 行业暴露图能正确生成（申万一级行业）
- [ ] Checkpoint 8.6: Brinson归因计算正确（超额收益分解）
- [ ] Checkpoint 8.7: plotly动态净值曲线能正确生成
- [ ] Checkpoint 8.8: matplotlib因子衰减走势图能正确生成
- [ ] Checkpoint 8.9: 过拟合风险警告正确触发（OOS夏普低于全样本夏普的60%时自动标记警告）
- [ ] Checkpoint 8.10: references/报告文档完整

## Task 9: 自动化安装脚本开发
- [ ] Checkpoint 9.1: `scripts/install.sh` 能检测OpenClaw是否安装
- [ ] Checkpoint 9.2: `scripts/install.sh` 能自动安装OpenClaw（Linux/macOS）
- [ ] Checkpoint 9.3: `scripts/import_skills.sh` 能导入所有Skill
- [ ] Checkpoint 9.4: `scripts/setup_env.sh` 能指导环境变量配置（TUSHARE_TOKEN、GM_TOKEN等）
- [ ] Checkpoint 9.5: `scripts/check_dependencies.py` 能检查依赖，支持选择性安装xtquant/gm/vnpy等
- [ ] Checkpoint 9.6: 许可与合规提示完整（Tushare、xtquant、gm等库许可协议）
- [ ] Checkpoint 9.7: README.md 包含完整使用指南

## Task 10: 环境变量和安全配置
- [ ] Checkpoint 10.1: `.env.example` 文件存在且包含所有必需变量
- [ ] Checkpoint 10.2: 所有脚本能从环境变量正确读取API密钥
- [ ] Checkpoint 10.3: 缺少环境变量时脚本能给出明确提示
- [ ] Checkpoint 10.4: SECURITY.md 安全配置说明完整

## Task 11: 项目文档与测试
- [ ] Checkpoint 11.1: `requirements.txt` 包含所有Python依赖
- [ ] Checkpoint 11.2: 每个Skill包含测试脚本
- [ ] Checkpoint 11.3: 代码遵循PEP 8规范
- [ ] Checkpoint 11.4: 所有Python脚本包含中文注释和类型提示
- [ ] Checkpoint 11.5: 项目README完整且易于理解
- [ ] Checkpoint 11.6: 所有SKILL.md格式符合Anthropic标准
