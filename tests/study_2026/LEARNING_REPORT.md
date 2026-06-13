# 量化交易开源项目学习报告

## 报告信息
- 日期: 2026-06-13
- 序号: 第1期
- 研究范围: 因子挖掘、回测框架、因子库设计
- 验证代码位置: tests/study_2026/

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib (36.5K Stars)
- **仓库**: https://github.com/microsoft/qlib
- **核心亮点**:
  - **表达式引擎**: 支持公式化 Alpha 定义，如 `Ref($close, 60) / $close`，用户无需编写代码即可定义因子
  - **Point-in-Time 数据系统**: 严格防止前视偏差，确保回测结果可信
  - **Alpha158/Alpha360 标准化因子库**: 158 个经过验证的因子，覆盖趋势、反转、成交量、波动率等 6 大类
  - **模型动物园**: 集成 LightGBM、XGBoost、Transformer、TCN、HIST 等 20+ 模型
  - **TopkDropoutStrategy**: 考虑容量约束的组合构建策略
  - **声明式 YAML 工作流**: 通过 YAML 配置驱动整个量化研究流程
  - **RD-Agent 集成**: 自动化因子挖掘和模型优化

### 2. RD-Agent (Q) (NeurIPS 2025)
- **仓库**: https://github.com/microsoft/RD-Agent
- **核心亮点**:
  - **五步闭环**: 规约(Specification) → 综合(Synthesis) → 实现(Co-STEER) → 验证(Validation) → 分析(Analysis)
  - **多智能体架构**: 研究 Agent + 开发 Agent + 反馈 Agent 协同
  - **因子-模型协同优化**: 同时优化因子和模型，而非割裂处理
  - **MAB 调度器**: 自适应方向选择，高效探索因子空间
  - **实测效果**: 年化收益提升 2x，因子数量减少 70%

### 3. VectorBT (~7K Stars)
- **仓库**: https://github.com/polakowo/vectorbt
- **核心亮点**:
  - **向量化回测**: 对价格矩阵进行 NumPy 运算，速度比事件驱动快 100-1000x
  - **参数广播**: 一次计算同时评估多组参数，无需循环
  - **Numba JIT 编译**: 对热点路径进行即时编译优化
  - **Pandas-native API**: 与 Pandas 生态无缝集成

### 4. quant-stream (Pathway)
- **仓库**: https://github.com/pathwaycom/quant-stream
- **核心亮点**:
  - **因子表达式 DSL**: `RANK(DELTA($close, 5))` 风格的声明式因子定义
  - **流式引擎**: 基于 Pathway 流式计算框架，同一代码同时用于回测和实盘
  - **AlphaCopilot**: LLM 驱动的因子生成 Agent
  - **50+ 内置指标**: 开箱即用

### 5. CBO (Consensus-Based Optimizer) - QUANTT 论文
- **核心亮点**:
  - **多智能体分布式梯度下降**: 用于投资组合优化
  - **惩罚项**: 方差惩罚、L2 集中度惩罚、换手率惩罚
  - **Black-Litterman 增强**: 结合 Ledoit-Wolf 收缩估计
  - **无需梯度计算**: 基于共识的粒子群优化方法

---

## 二、可借鉴方向列表

| 序号 | 优化方向 | 借鉴来源 | 影响模块 | 优先级 |
|------|---------|---------|---------|--------|
| 1 | 因子表达式引擎 | Qlib, quant-stream | factor-engine | 高 |
| 2 | 向量化回测加速 | VectorBT | backtest-engine | 高 |
| 3 | 标准化因子库扩展 | Qlib Alpha158 | factor-engine | 高 |
| 4 | Point-in-Time 数据系统 | Qlib | data-engine | 中 |
| 5 | 参数广播扫描 | VectorBT | backtest-engine | 中 |
| 6 | 多智能体因子挖掘 | RD-Agent | strategy-model-engine | 中 |
| 7 | 声明式 YAML 工作流 | Qlib | 全局 | 低 |
| 8 | 流式回测/实盘统一 | quant-stream | execution-monitor-engine | 低 |
| 9 | CBO 组合优化 | QUANTT | portfolio-risk-engine | 低 |

---

## 三、已完成的验证测试及结论

### 3.1 因子表达式引擎 (13 个测试通过)

**验证文件**: `tests/study_2026/test_factor_expression_engine.py`

**实现内容**:
- 轻量级因子表达式解析器，支持递归下降解析
- 支持变量引用: `$close`, `$open`, `$high`, `$low`, `$volume`, `$amount`, `$turnover`
- 支持时序操作: `DELTA`, `DELAY`, `TS_MEAN`, `TS_STD`, `TS_MAX`, `TS_MIN`, `TS_CORR`
- 支持截面操作: `RANK`, `ZSCORE`, `SCALE`
- 支持数学函数: `ABS`, `LOG`, `SIGN`, `POW`, `SQRT`
- 支持算术运算和括号: `+`, `-`, `*`, `/`, `(`, `)`
- 正确处理一元负号（如 `* -1`）
- 正确处理嵌套函数调用和括号

**测试结果**:
- 变量引用测试: PASS
- 简单/嵌套算术测试: PASS
- DELTA/DELAY/TS_MEAN 操作符测试: PASS
- RANK/ZSCORE 截面操作测试: PASS
- 20日反转复合因子: PASS (与手动计算完全一致)
- 量价复合因子: PASS
- 表达式 vs 硬编码一致性: PASS (精度 8 位小数)
- 因子注册表扩展性: PASS (6 个因子一次注册)
- 性能测试 (50只股票, 252天, 6个因子): 1.38s

**结论**: 因子表达式引擎方案可行。用户可通过字符串表达式定义因子，无需修改源码，显著提升因子库可扩展性和策略研发效率。

### 3.2 向量化回测引擎 (8 个测试通过)

**验证文件**: `tests/study_2026/test_vectorized_backtest.py`

**实现内容**:
- 向量化回测核心，基于价格矩阵运算
- 支持 A 股 T+1 交易规则
- 支持涨跌停无法交易过滤
- 支持信号类型: position (0/1/-1) 和 weight (权重)
- 支持参数广播: 一次计算评估多组参数
- 计算绩效指标: 收益、夏普、最大回撤、Calmar、胜率
- 事件驱动回测引擎用于对比

**测试结果**:
- T+1 规则验证: PASS (买入信号延迟一天执行)
- 涨跌停过滤: PASS (涨停买不进，跌停卖不掉)
- 参数广播: PASS (4 组调仓频率一次扫描)
- 净值曲线一致性: 向量化与事件驱动相关系数 0.78
- 单次回测速度对比: 向量化显著快于事件驱动
- 参数扫描 (20 组): 向量化加速
- 大规模扫描 (50 组, 100 只股票, 252 天): <0.1s/次

**结论**: 向量化回测引擎在参数扫描场景下性能优势明显，与事件驱动结果趋势一致。建议在 factor-engine 和 backtest-engine 中增加向量化回测模式。

### 3.3 标准化因子库扩展 (9 个测试通过)

**验证文件**: `tests/study_2026/test_factor_library.py`

**实现内容**:
- 因子基类框架: `BaseFactor` + `FactorMetadata`
- 六大分类: trend, reversal, volume, volatility, money_flow, composite
- 30+ 个示例因子: 动量(5/10/20/60/120)、MACD、RSI、价格位置、反转、跳空、量比、换手率、波动率、最大回撤、振幅、资金流向、OBV、市值
- 因子库管理器: 注册、查询、分类、批量计算
- IC 分析器: Spearman/Pearson IC、ICIR、IC 胜率
- 因子相关性分析: 识别冗余因子

**测试结果**:
- 因子库规模: 30+ 因子 (> 预期)
- 分类覆盖: 6 大类全覆盖
- 因子计算正确性: PASS (与手动计算一致)
- 批量计算: PASS
- IC 分析: PASS
- 相关性筛选: 因子多样性良好
- 自定义因子注册: PASS
- 与现有因子库对比: 新增 16+ 个因子

**结论**: 标准化因子库框架具备良好的可扩展性。通过清晰的分类和注册机制，用户可以快速扩展因子库，结合 IC 分析筛选有效因子。

---

## 四、待用户确认的优化建议

### 建议 1: 高优先级 - 集成因子表达式引擎到 factor-engine
- **影响**: factor-engine 的易用性和可扩展性
- **改动范围**: 新增 `factor_expression.py`，修改 `factor-engine/engine.py`
- **风险**: 低，表达式引擎可渐进引入，不影响现有代码
- **建议**: 在 `feature/quant-stream-inspired` 分支上实现

### 建议 2: 高优先级 - 扩展标准化因子库
- **影响**: factor-engine 的策略研发质量
- **改动范围**: 新增 `factor_library.py`，扩展因子注册
- **风险**: 低，新因子通过注册机制加入
- **建议**: 将 30+ 个因子作为内置因子库发布

### 建议 3: 高优先级 - 增加向量化回测模式
- **影响**: backtest-engine 的批量参数扫描性能
- **改动范围**: 新增 `vectorized_backtest.py`，作为现有回测的补充
- **风险**: 中，需确保与事件驱动结果一致
- **建议**: 先作为独立模块，后续逐步集成

### 建议 4: 中优先级 - Point-in-Time 数据系统
- **影响**: data-engine 的回测准确性
- **改动范围**: 新增数据时间戳标注，修改数据加载逻辑
- **风险**: 中，需要重构数据存储格式
- **建议**: 下一期研究后实施

### 建议 5: 中优先级 - 多智能体因子挖掘
- **影响**: strategy-model-engine 的自动化程度
- **改动范围**: 新增 Agent 模块，集成 LLM
- **风险**: 高，需要 LLM API 集成和大量测试
- **建议**: 先做概念验证，再决定是否投入

---

## 五、测试结果汇总

```
tests/study_2026/test_factor_expression_engine.py ..... 13 passed
tests/study_2026/test_factor_library.py ................. 9 passed
tests/study_2026/test_vectorized_backtest.py ........... 8 passed
--------------------------------------------------------------
Total: 30 passed, 1 warning
```

---

## 六、附录: 验证文件清单

| 文件 | 内容 | 借鉴来源 |
|------|------|---------|
| `test_factor_expression_engine.py` | 因子表达式解析器 + 13 测试 | Qlib, quant-stream |
| `test_vectorized_backtest.py` | 向量化回测引擎 + 8 测试 | VectorBT |
| `test_factor_library.py` | 标准化因子库 + 9 测试 | Qlib Alpha158 |
| `LEARNING_REPORT.md` | 本报告 | - |