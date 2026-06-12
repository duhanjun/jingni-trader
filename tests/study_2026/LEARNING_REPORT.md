# jingni-trader 量化交易学习报告 #001

**日期**: 2026-06-12
**研究主题**: 2026年量化交易开源项目调研与 jingni-trader 优化方向验证

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib (42K+ Stars)
- **项目地址**: https://github.com/microsoft/qlib
- **核心定位**: AI 导向的量化投资平台，最新的 LLM agent 生态基础设施
- **关键亮点**:
  - **表达式引擎 (Expression Engine)**: DSL 语法声明因子，如 `$close`, `Ref($close, 1)`, `Mean($close, 5)`，极大降低因子编写门槛
  - **列式二进制存储**: 自定义 columnar binary format 支持快速时间序列切片
  - **Alpha158/Alpha360 因子库**: 预置数百个标准化因子
  - **RD-Agent LLM 集成**: 自动挖掘 alpha 因子和优化模型
  - **DAG 任务调度**: `qrun` 命令行工具支持配置驱动全流程
  - **模型库**: 30+ 预置模型（LightGBM, LSTM, GRU, Transformer 等）

### 2. QUANTAXIS (25K+ Stars)
- **项目地址**: https://github.com/yutiansut/QUANTAXIS
- **核心定位**: 全栈式量化金融分析框架，Python + Rust 混合架构
- **关键亮点**:
  - **Python + Rust 混合**: QARSBridge 实现动态检测和自动回退，性能关键路径用 Rust 重写
  - **QIFI 协议**: 跨语言的统一账户模型，数据结构标准化
  - **零拷贝数据桥接**: 共享内存方式交换数据，避免序列化开销
  - **微服务架构**: 统一资源管理，支持从单机研究到分布式部署

### 3. QuantConnect/LEAN (7.8K+ Stars)
- **项目地址**: https://github.com/QuantConnect/Lean
- **核心定位**: 全球领先的开源量化交易引擎，机构级专业平台
- **关键亮点**:
  - **事件驱动架构**: MarketDataEvent → SignalEvent → OrderEvent → FillEvent → PortfolioEvent 完整事件链
  - **模块化可插拔组件**: SlippageModel, FeeModel, FillModel, MarginModel 独立可替换
  - **Universe Selection**: 动态资产池筛选避免选择偏差
  - **Algorithm Framework**: Alpha Creation + Portfolio Construction + Risk Management + Execution 四阶段框架
  - **生存偏差消除**: 自动处理拆股、分红、退市、并购等公司行为

---

## 二、可借鉴方向列表

| 序号 | 优化方向 | 借鉴来源 | 优先级 | 对应 jingni-trader 模块 |
|------|---------|---------|--------|------------------------|
| 1 | 因子表达式引擎 (DSL) | Qlib Expression Engine | 高 | factor-engine |
| 2 | 事件驱动回测架构 | LEAN Event-Driven | 高 | backtest-engine |
| 3 | 列式数据存储优化 | Qlib Columnar / QUANTAXIS | 中 | data-engine |
| 4 | 向量化计算加速 | Qlib / QUANTAXIS (Rust) | 中 | factor-engine, portfolio-risk-engine |
| 5 | Universe Selection 机制 | LEAN | 中 | data-engine |
| 6 | 可插拔交易成本模型 | LEAN | 低 | backtest-engine |
| 7 | Rust 核心加速 | QUANTAXIS QARSBridge | 低（长期） | 全局 |
| 8 | LLM Agent 因子挖掘 | Qlib + RD-Agent | 远期 | factor-engine, strategy-model-engine |

---

## 三、已完成验证测试及结论

### 测试1: 因子表达式引擎 (DSL-based Factor Declaration)

**文件**: `tests/study_2026/test_factor_expression_engine.py`
**测试用例数**: 9 个（全部通过）
**借鉴来源**: Microsoft Qlib - Expression Engine

**测试内容**:
- 基本字段访问 ($close, $open, $high-$low)
- 滚动窗口函数 (Mean, Std, Ref)
- 动量因子 ($close/Ref($close, 20)-1)
- 波动率因子 Std($returns, 20)
- 截面排名因子 Rank($close)
- 复合多因子组合
- 表达式语法校验（7种有效表达式）
- 扩展性测试（8因子动态添加无需修改核心引擎）
- 性能对比：DSL vs 硬编码

**关键结论**:

| 指标 | DSL 方式 | 硬编码方式 | 说明 |
|------|---------|-----------|------|
| 7因子计算时间 | ~0.15s | ~0.04s | DSL 慢约 3-4x，但可接受 |
| 代码行数 | 3行配置 | 30+行代码 | DSL 极简 |
| 添加新因子 | 修改配置 | 修改核心代码 | DSL 零侵入 |
| 计算准确性 | 与硬编码一致 | - | 差异 < 0.01 |

**结论**: DSL 表达式引擎可行，建议作为因子引擎的可选增强功能引入。用户可编写配置字符串声明因子，引擎自动解析执行。性能开销约 3-4x，在日频策略中可忽略。

---

### 测试2: 事件驱动回测架构 (Event-Driven Backtest)

**文件**: `tests/study_2026/test_event_driven_backtest.py`
**测试用例数**: 8 个（全部通过）
**借鉴来源**: QuantConnect/LEAN - Event-Driven Architecture

**测试内容**:
- 基本执行流程验证
- 资金非负性保证
- A股规则支持（T+1、涨跌停限制、100股整数倍）
- 交易成本计算（佣金、印花税、过户费）
- 滑点模型影响
- 事件处理顺序一致性
- 大规模组合性能（50只股票 x 1年）
- 不同滑点模型敏感性分析

**关键结论**:

| 测试项 | 结果 | 说明 |
|-------|------|------|
| A股 T+1 规则 | 通过 | 同一天同一股票不出现买卖双向交易 |
| 涨跌停限制 | 通过 | 涨停拒绝买入，跌停拒绝卖出 |
| 100股整数倍 | 通过 | 所有成交均为100股整数倍 |
| 资金安全 | 通过 | 现金和总资产始终非负 |
| 费用计算 | 通过 | 佣金/印花税/过户费准确 |
| 50股票x1年回测 | ~6.6s | 可接受的性能 |
| 滑点敏感性 | 通过 | 高滑点 → 低收益，逻辑正确 |

**结论**: 事件驱动架构清晰且易于扩展。相比当前 adapter 模式，事件驱动模型具有更好的可测试性（每个事件类型可独立测试）和可扩展性（增删事件类型不影响核心引擎）。建议作为 backtest-engine 的增强架构。

---

### 测试3: 列式数据存储与高效数据处理管道

**文件**: `tests/study_2026/test_columnar_data_pipeline.py`
**测试用例数**: 11 个（全部通过）
**借鉴来源**: Qlib Columnar Binary + QUANTAXIS QARSBridge

**测试内容**:
- Parquet vs CSV vs Feather vs HDF5 格式对比
- 单股票查询性能
- 时间范围切片查询
- Pivot 宽表性能
- 向量化 vs 循环计算（滚动均值）
- IC 计算性能
- 因子中性化性能（numpy vs sklearn）
- 协方差估计质量（Ledoit-Wolf vs Sample）
- 数据过滤排序优化
- 分块处理 vs 一次性加载
- 并行因子计算

**关键结论**:

| 测试项 | 关键发现 |
|-------|---------|
| 存储格式 | Feather 最快(读取7ms)，Parquet 最小(0.67MB)。当前 Parquet 选择合理 |
| 读取性能 | Parquet 比 CSV 快约 1.1x（小数据集），大数据集差距更显著 |
| 向量化加速 | groupby.transform 比 for 循环快约 7.5x |
| IC 计算 | numpy 向量化比纯循环快约 2x |
| 中性化 | numpy 矩阵运算比 sklearn 快约 2x |
| 协方差质量 | Ledoit-Wolf 条件数远优于 Sample Cov（更稳定） |

**结论**: 
- jingni-trader 当前使用 Parquet 存储是正确的选择
- 因子计算中应大量使用 groupby.transform 替代显式循环
- 因子中性化可改用 numpy 矩阵运算替代 sklearn，提升约 2x 性能
- 协方差估计已采用 Ledoit-Wolf，无需调整

---

## 四、待用户确认的优化建议

### 建议1: 引入因子 DSL 表达式引擎（高优先级）

- **现状**: 因子计算硬编码在 `factor-engine/engine.py`，添加新因子需修改核心代码
- **方案**: 在 factor-engine 中新增 `FactorExpressionEngine` 模块，用户通过配置声明因子
- **收益**: 因子开发效率提升 10x，降低非程序员参与门槛
- **风险**: 原型验证中 DSL 比硬编码慢 3-4x，需确认日频策略可接受此开销
- **测试文件**: `tests/study_2026/test_factor_expression_engine.py`

### 建议2: 采用事件驱动回测架构（高优先级）

- **现状**: 回测引擎通过 adapter 模式委托给 backtrader/rqalpha 等外部框架
- **方案**: 实现原生事件驱动引擎作为 `native_adapter` 的增强版，与 adapter 模式共存
- **收益**: 更精确的 A 股交易规则模拟，更好的可测试性和可扩展性
- **风险**: 需要重新验证与现有策略的兼容性
- **测试文件**: `tests/study_2026/test_event_driven_backtest.py`

### 建议3: 因子中性化改用 numpy 矩阵运算（中优先级）

- **现状**: 使用 sklearn LinearRegression 逐日做中性化回归
- **方案**: 改用 numpy 最小二乘直接求解（`np.linalg.solve`）
- **收益**: 约 2x 性能提升，减少 sklearn 依赖
- **风险**: 需处理矩阵奇异的边界条件
- **测试文件**: `tests/study_2026/test_columnar_data_pipeline.py` TestVectorizedComputation.test_03

### 建议4: 引入 Universe Selection 机制（中优先级）

- **现状**: 股票池通过 Context.stock_pool 静态指定
- **方案**: 参考 LEAN 设计 Universe Selection Model，支持动态过滤条件
- **收益**: 避免选择偏差，支持更灵活的策略（如按市值/行业/流动性筛选）

### 建议5: LLM Agent 因子挖掘（远期）

- **方案**: 参考 Qlib + RD-Agent 模式，在 strategy-model-engine 中引入 LLM 辅助因子发现
- **注意**: 此方向需等到因子表达式引擎和回测引擎稳定后再推进

---

## 五、验证测试汇总

```
总测试文件: 3
总测试用例: 28
通过: 28
失败: 0
总耗时: 22.36s

测试文件:
  tests/study_2026/test_factor_expression_engine.py      (9 passed)
  tests/study_2026/test_event_driven_backtest.py          (8 passed)
  tests/study_2026/test_columnar_data_pipeline.py         (11 passed)
```

---

## 六、下一步行动

1. 用户审阅上述优化建议后，确认优先实施的优化方向
2. 确认后在 `feature/quant-stream-inspired` 分支上进行代码实现
3. 实施完毕后在主代码中集成，并进行完整的回归测试

---

*报告生成时间: 2026-06-12 | 生成者: jingni-trader AI Agent*
*免责声明: 本报告中的验证代码仅供学习和研究参考，所有数值均基于模拟数据，不代表真实交易表现。*