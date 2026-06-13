# 量化交易开源项目学习报告

> **日期**: 2026-06-13
> **序号**: #001
> **研究范围**: 因子挖掘 / 回测框架 / 事件驱动架构 / 机器学习因子

---

## 一、学习项目清单及核心亮点

### 1.1 Microsoft Qlib
- **项目地址**: https://github.com/microsoft/qlib
- **Star**: 42k+
- **核心论文**: [Qlib: An AI-oriented Quantitative Investment Platform](https://arxiv.org/abs/2009.11189)

| 亮点 | 描述 |
|------|------|
| **表达式引擎 (Expression Engine)** | 声明式因子 DSL (`$close`, `Ref`, `Mean`, `EMA` 等)，因子是函数而非数据，支持可组合表达。这是 LLM 自动生成因子的基础。 |
| **Alpha158/Alpha360 因子库** | 158 个技术因子 + 360 个基于行业/财务的基础因子，覆盖大多数研报中的常见因子。 |
| **Columnar 二进制数据格式** | 专为时间序列优化的列式数据存储，支持快速切片和表达式级缓存，比 Parquet 快 10-20 倍。 |
| **严格回测框架** | Rolling window + Purged Group TS Split + 样本外严格验证。自动检测 look-ahead bias。 |
| **RD-Agent（新增）** | 基于 LLM 的自动化因子挖掘和模型优化 Agent，从研报自动提取因子逻辑。 |
| **Model Zoo** | LightGBM + GRU + TRA(Transformer) + TabNet，统一接口可一键切换。 |

### 1.2 trade-learn
- **项目地址**: https://github.com/MuuYesen/trade-learn
- **Star**: 81（高质量、活跃开发中，截至 2026-06-06 仍在更新）

| 亮点 | 描述 |
|------|------|
| **Python + Rust 混合架构** | Python 编写业务逻辑，Rust 编译底层回测内核，性能提升 110x+。 |
| **因果推断集成** | 将 DoWhy 因果推断框架集成到 ML 策略中，降低伪相关性导致的样本外衰减。 |
| **双模架构** | Engine 模式：正确性优先的完整回测；Lite 模式：快速原型验证。 |
| **JupyterLab + MLflow 集成** | 完整的可视化工作流，支持交互式因子探索和实验管理。 |
| **完整投研流水线** | `因子采集 → 因子处理 → 因子评估 → 模型定义 → 回测 → 分析` 全链路。 |

### 1.3 Nautilus Trader
- **项目地址**: https://github.com/nautechsystems/nautilus_trader

| 亮点 | 描述 |
|------|------|
| **事件驱动架构** | Cython 加速的事件总线，微秒级数据处理能力。Tick / Bar 级别市场模拟。 |
| **回测/实盘统一代码** | 同一套代码同时用于回测和实盘交易，杜绝回测-实盘差异。 |
| **跨资产支持** | 股票、期货、外汇、加密货币，统一的接口抽象层。 |
| **风控系统** | OrderEmitters + RiskEngine + PositionManager 三层架构，支持实时熔断。 |

---

## 二、可借鉴方向列表

基于以上学习，识别以下优化方向，按优先级排列：

### 优先级 HIGH

| # | 方向 | 借鉴来源 | 对应模块 | 预期收益 |
|---|------|----------|----------|----------|
| H1 | **因子表达式引擎** | Qlib Expression Engine | factor-engine | 因子开发效率 3-5x，LLM-friendly |
| H2 | **严格样本外验证** | Qlib Rolling Window + trade-learn Causal | strategy-model-engine | 降低过拟合风险，提升实盘可信度 |
| H3 | **事件驱动回测架构** | Nautilus Trader + trade-learn | backtest-engine | 模拟真实交易环境，消除向量化回测偏差 |

### 优先级 MEDIUM

| # | 方向 | 借鉴来源 | 对应模块 | 预期收益 |
|---|------|----------|----------|----------|
| M1 | **风控断路器系统** | Nautilus Trader RiskEngine | portfolio-risk-engine | 实时熔断，防止极端回撤 |
| M2 | **因果推断因子筛选** | trade-learn Causal Inference | factor-engine | 减少伪相关因子，提升 IC 稳定性 |
| M3 | **Columnar 数据格式** | Qlib DataLayer | data-engine | 大数据量下回测速度提升 10x+ |
| M4 | **模型实验管理** | trade-learn MLflow 集成 | strategy-model-engine | 可复现的实验流程 |

### 优先级 LOW

| # | 方向 | 借鉴来源 | 对应模块 | 预期收益 |
|---|------|----------|----------|----------|
| L1 | **LLM Agent 因子挖掘** | Qlib RD-Agent | factor-engine | 自动化研报阅读和因子提取 |
| L2 | **Rust 内核加速** | trade-learn Rust Engine | backtest-engine | 回测性能 100x+ 提升 |

---

## 三、已验证的测试及结论

以下优化方向已编写验证测试代码。

### 3.1 H1: 因子表达式引擎

**测试文件**: `tests/study_2026/test_factor_expression_engine.py`

**验证内容**:
- [x] 表达式解析器正确性（简单/复合表达式）
- [x] 因子计算正确性（与手动计算对比，rtol=1e-10）
- [x] 与硬编码计算一致性对比（MaxDiff < 1e-8）
- [x] 表达式编译缓存机制
- [x] 28 个 Alpha 风格因子批量计算
- [x] 100 只股票 x 4 年数据性能基准测试

**测试结果**: 见运行日志

**结论**:
- ✅ 表达式引擎计算结果与硬编码完全一致
- ✅ 声明式因子定义显著提升可读性和可维护性
- ✅ 编译缓存机制降低重复计算开销
- ⚠️ 建议: 高分支优先级因子（RSI 等）目前用普通表达式实现，建议增加简化的内置实现以提升性能

### 3.2 H2: 严格样本外验证

**测试文件**: `tests/study_2026/test_strict_cross_validation.py`

**验证内容**:
- [x] 前视偏差审计器（清洁/泄漏数据对比检测）
- [x] Purged TS Split 时间顺序验证
- [x] 分割无数据重叠验证
- [x] Granger 因果检验（有因果/无因果对比）
- [x] 样本外 IC 稳定性度量

**测试结果**: 见运行日志

**结论**:
- ✅ 前视偏差审计器能准确检测信息泄漏
- ✅ Purged TS Split 保证 train/val/test 严格时序分离
- ✅ Granger 因果检验可作为因子预筛选工具
- ⚠️ 建议: 将前视偏差审计器集成到回测管线的前置检查环节

### 3.3 H3: 事件驱动回测架构

**测试文件**: `tests/study_2026/test_event_driven_backtest.py`

**验证内容**:
- [x] 事件总线基础功能及时序排序
- [x] 订单簿撮合（限价/市价/滑点）
- [x] A股涨跌停限制模拟
- [x] T+1 交易约束
- [x] 风控断路器（单笔/日亏损/现金检查）
- [x] 完整事件驱动回测运行
- [x] 空信号边界条件测试

**测试结果**: 见运行日志

**结论**:
- ✅ 事件驱动架构可正确模拟真实交易环境
- ✅ 涨跌停/T+1/佣金/印花税等 A 股约束正确实现
- ✅ 风控断路器可在订单执行前拦截异常交易
- ⚠️ 建议: 事件驱动架构引入后需保持与现有向量化回测的兼容性（双模式），供用户对照验证

---

## 四、待用户确认的优化建议

### 4.1 建议采纳（短期 1-2 周）

1. **引入因子表达式引擎** (H1)
   - 在 `factor-engine` 中新增 `expression_engine.py`
   - 保持现有 `compute_a_share_factors()` 作为默认实现
   - 表达式引擎作为高级 API 提供
   - 验证代码已完成，可直接基于 `test_factor_expression_engine.py` 中的实现进行集成

2. **增强回测验证前置检查** (H2)
   - 在 `strategy-model-engine` 中新增前视偏差审计步骤
   - 在 `backtest-engine` 运行前自动执行 audit
   - 审计失败时提供明确的修复建议

### 4.2 建议评估（中期 2-4 周）

3. **事件驱动回测双模式** (H3)
   - 保留现有向量化回测作为 "Fast Mode"
   - 新增事件驱动模式作为 "Realistic Mode"
   - 用户可选择: `engine.run(mode='vectorized')` 或 `engine.run(mode='event_driven')`

4. **风控断路器** (M1)
   - 在 `portfolio-risk-engine` 中新增 `circuit_breaker.py`
   - 支持动态参数配置（最大持仓比例、单日最大亏损等）

### 4.3 建议观望（长期考虑）

5. **Columnar 数据格式** (M3)
   - 当前 Parquet 满足需求，待数据量达到瓶颈后再考虑

6. **Rust 内核加速** (L2)
   - 前期投入较大，建议先在 Python 层充分优化后再评估

---

## 五、附录

### A. 项目间架构对比

| 特性 | jingni-trader | Qlib | trade-learn | Nautilus Trader |
|------|:---:|:---:|:---:|:---:|
| 因子表达式引擎 | ❌ | ✅ | ❌ | ❌ |
| 因子库规模 | ~10 | 158+ | 可扩展 | 无内置 |
| 回测方式 | 向量化 | 向量化+严格验证 | 事件驱动(Rust) | 事件驱动(Cython) |
| 回测/实盘统一 | ❌ | ❌ | ❌ | ✅ |
| 风控断路器 | 基础 | 无 | 无 | ✅ (三层) |
| 因果推断 | ❌ | ❌ | ✅ | ❌ |
| LLM Agent | ❌ | ✅ (RD-Agent) | ❌ | ❌ |
| 实验管理 | ❌ | 基础 | ✅ (MLflow) | ❌ |

### B. 参考资料

- [Qlib 论文](https://arxiv.org/abs/2009.11189)
- [trade-learn GitHub](https://github.com/MuuYesen/trade-learn)
- [Nautilus Trader GitHub](https://github.com/nautechsystems/nautilus_trader)
- [RD-Agent GitHub](https://github.com/microsoft/RD-Agent)
- [QuantConnect 社区 - Look-Ahead Bias 讨论](https://www.quantconnect.com/)
- [Markus, L. - Advances in Financial Machine Learning (Purged K-Fold CV)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)