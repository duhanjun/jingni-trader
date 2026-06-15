# jingni-trader 量化交易学习报告

> 日期: 2026-06-15 | 序号: #1
> 本次学习周期: 2026年6月

---

## 一、学习项目清单及核心亮点

### 1.1 Microsoft Qlib (github.com/microsoft/qlib)

| 维度 | 说明 |
|------|------|
| **Stars** | 36,500+ |
| **定位** | AI-oriented 量化投资平台 |
| **语言** | Python |
| **许可证** | MIT |

**核心亮点:**

1. **Point-in-Time (PIT) 数据系统**
   - 通过 `DataHandler` 和 `Provider` 抽象层，确保在回测的每个时间点只能访问当时已知的数据
   - 自动检测并防止前视偏差（look-ahead bias）
   - 支持增量数据更新，新增数据不会影响历史计算结果

2. **声明式工作流 (qrun)**
   - 使用 YAML 配置文件定义完整量化工作流，从数据加载到模型训练、回测一气呵成
   - 配置即文档，便于复现和版本管理

3. **Model Zoo 注册表模式**
   - 20+ 内置模型（LightGBM, CatBoost, TabNet, GRU, Transformer 等），通过注册表动态加载
   - 用户可以在不修改核心代码的情况下注册自定义模型
   - 统一的 `fit/predict` 接口

4. **表达式引擎 (Expression Engine)**
   - 通过 DSL 表达式定义因子计算，如 `"Ref($close, -5) / $close - 1"`
   - 支持表达式缓存和向量化计算
   - 天然 Pit 安全

5. **RD-Agent (LLM-driven Factor Mining)**
   - 使用 LLM 自动发现和实现新的量化因子
   - 自动生成因子代码、测试和文档

---

### 1.2 FinRL-X (github.com/AI4Finance-Foundation/FinRL)

| 维度 | 说明 |
|------|------|
| **Stars** | 5,000+ |
| **定位** | 模块化金融强化学习框架 |
| **语言** | Python |
| **许可证** | MIT |

**核心亮点:**

1. **Weight-Centric 统一接口**
   - 核心创新：使用目标权重向量 `wt ∈ R^n` 作为所有策略组件的统一输出契约
   - 回测和实盘使用相同的接口语义，消除部署偏差
   - 支持多资产、多时间尺度的组合管理

2. **Composable Strategy Pipeline（可组合策略管道）**
   - 四层设计：Selection（选股）→ Allocation（分配）→ Timing（择时）→ Risk Overlay（风控覆盖）
   - 每层独立可替换，支持 A/B 测试
   - 每层输出权重向量，作为下一层的输入

3. **Deployment Consistency（部署一致性）**
   - 回测和实盘使用完全相同的代码路径
   - 通过 `Env` 抽象层统一回测环境和实盘环境

4. **ML/DRL/LLM 多范式集成**
   - 支持传统 RL（PPO, DDPG, SAC）和 LLM-based 代理
   - 统一的 `Agent` 接口

---

### 1.3 QUANTAXIS (github.com/QUANTAXIS/QUANTAXIS)

| 维度 | 说明 |
|------|------|
| **Stars** | 25,000+ |
| **定位** | 全栈量化交易系统（A股聚焦） |
| **语言** | Python + Rust |
| **许可证** | MIT |

**核心亮点:**

1. **Python + Rust 混合架构 (QARSBridge)**
   - 核心计算引擎由 Rust 实现，Python 提供用户接口
   - 通过 Apache Arrow 和 Shared Memory 实现零拷贝数据桥接
   - 性能提升：单票分钟线2年回测 500ms，单指标计算 70ns

2. **QIFI 协议（统一账户模型）**
   - 定义标准化的账户、订单、持仓、成交数据结构
   - 回测和实盘使用完全相同的账户模型
   - 支持多账户、多策略、多市场

3. **微服务架构 (v2.1)**
   - 数据服务、回测服务、交易服务独立部署
   - gRPC 通信 + 消息队列
   - 支持水平扩展

4. **全市场数据覆盖**
   - A股、期货、期权、基金、债券、港股、美股、加密货币
   - 分钟级到日级多粒度数据
   - 复权、除权除息自动处理

---

## 二、可借鉴方向列表

| 序号 | 借鉴方向 | 来源项目 | 优先级 | 影响模块 |
|------|----------|----------|--------|----------|
| 1 | **PIT 数据安全检查** | Qlib | 高 | factor-engine |
| 2 | **因子注册表模式** | Qlib Model Zoo | 高 | factor-engine |
| 3 | **Weight-Centric 接口** | FinRL-X | 中 | strategy-model-engine, backtest-engine |
| 4 | **声明式 YAML 工作流** | Qlib qrun | 中 | engine.py (主调度) |
| 5 | **Rust 核心加速** | QUANTAXIS | 低 | backtest-engine |
| 6 | **QIFI 统一账户协议** | QUANTAXIS | 低 | backtest-engine, execution-monitor-engine |
| 7 | **表达式引擎** | Qlib | 中 | factor-engine |
| 8 | **微服务架构** | QUANTAXIS | 低 | 整体架构 |

---

## 三、已完成的验证测试及结论

### 3.1 测试环境

- **测试文件**: `tests/study_2026/test_point_in_time_safety.py`
- **依赖**: numpy, pandas, scipy
- **测试数据**: 模拟 A 股日线数据 (10-100 只股票, 252 个交易日)

### 3.2 测试结果汇总

#### 测试1: PIT 前视偏差泄漏检测 ✅ 通过

| 因子 | IC_mean | 判断 |
|------|---------|------|
| cheat_factor（使用 t+1 close） | 1.0000 | ⚠️ 可疑 - 成功检测 |
| normal_factor（使用 t-1 close） | 0.0395 | ✅ 安全 |

**结论**: `PITSafetyChecker.detect_leakage_via_future_return()` 成功识别了使用未来数据的作弊因子（IC=1.0 vs 正常因子 IC=0.04）。该检测器可作为 jingni-trader 因子开发流程中的自动化检查工具。

#### 测试2: Rolling 操作 PIT 安全性 ✅ 通过

**结论**: 前向滚动窗口（pandas rolling 默认行为）与后向滚动窗口（前视偏差）之间存在显著差异（max_diff 1.0-7.2），确认了 PIT 安全实现的重要性。jingni-trader 现有的 `groupby.transform(rolling)` 模式是 PIT 安全的。

#### 测试3: PIT 安全数据处理器 ✅ 通过

**结论**: `PITSafeDataHandler` 提供了显式的 PIT 安全计算接口，包含：
- `compute_rolling_feature()` - 按股票独立计算滚动特征
- `compute_cross_sectional_rank()` - 按时间截面排名
- `validate_time_alignment()` - 验证时间对齐性

#### 测试4: 截面排名 PIT 一致性 ✅ 通过

**结论**: 验证了截面排名在不同时间窗口下的行为差异。不同窗口下同一天排名可能不同，这是 PIT 的正确行为——回测时每个时间点只能看到当时完整的截面数据。

#### 测试5: 因子注册表模式 ✅ 通过

**测试结果**:
- 成功注册 5 个因子（动量 2 个、波动率 1 个、成交量 2 个）
- 按类别查询功能正常
- 批量计算 1000 行数据，非空率 90%-96%

**与现有实现对比**:

| 特性 | 现有硬编码方式 | 注册表模式 |
|------|---------------|-----------|
| 添加新因子 | 修改核心 engine.py | 外部注册，无需修改核心代码 |
| 因子分类 | 无 | 按类别自动分组 |
| 可发现性 | 需阅读源码 | `list_by_category()` 查询 |
| 可测试性 | 需复杂 mock | 每个因子独立可测 |
| 热插拔 | 不支持 | 支持 |

#### 测试6: Weight-Centric 信号接口 ✅ 通过

**测试结果**:

| 度量 | 二元信号 (现有) | 权重向量 (FinRL-X) |
|------|----------------|-------------------|
| 持仓股票数 | 4/20 (20%) | 9/20 (45%) |
| 信息熵 | 1.3863 | 0.8154 |
| 分数幅值信息 | 丢失 | 保留 |
| 最大权重 | 0.250 (等权) | 0.8136 (softmax) |

**结论**: 权重向量接口保留了因子得分的幅值信息，信息熵更低（更集中），适合需要细粒度仓位管理的场景。同时支持 FinRL-X 的四层管道设计（Selection → Allocation → Timing → Risk Overlay）。

#### 测试7: 回测引擎性能基准 ✅ 通过

| 规模 | 耗时 | Rust 10x 估算 |
|------|------|--------------|
| 10只/年 | 0.2581s | ~0.026s |
| 50只/年 | 0.2003s | ~0.020s |
| 100只/年 | 0.2136s | ~0.021s |

**结论**: 当前纯 Python 实现在小规模测试中表现良好，但 QUANTAXIS 的 Rust 核心展示了 10x-100x 的加速潜力。对于分钟级数据或全市场回测场景，Rust 加速是可行的优化方向。

---

## 四、待用户确认的优化建议

### 建议1: 引入 PIT 安全检查器（优先级: 高）

**来源**: Microsoft Qlib

**具体方案**:
- 在 `factor-engine` 中集成 `PITSafetyChecker` 作为因子计算后的自动验证步骤
- 在 `backtest-engine` 的回测前添加 PIT 检查钩子
- 若检测到异常 IC（> 0.15），发出警告或阻止回测

**影响范围**: `skills/factor-engine/engine.py`, `skills/backtest-engine/engine.py`

**工作量**: 小（1-2 个文件修改）

---

### 建议2: 因子引擎重构为注册表模式（优先级: 高）

**来源**: Microsoft Qlib Model Zoo

**具体方案**:
- 将 `compute_a_share_factors()` 中的硬编码因子计算函数迁移到注册表
- 在 `factor-engine/engine.py` 中引入 `FactorRegistry` 类
- 保留现有 12 个 alpha 因子作为注册表默认项
- 添加 `--factor` 参数支持按名称选择因子子集

**影响范围**: `skills/factor-engine/engine.py`

**工作量**: 中（需重构现有因子计算逻辑）

---

### 建议3: 支持 Weight-Centric 信号输出（优先级: 中）

**来源**: FinRL-X

**具体方案**:
- 在 `strategy-model-engine` 中添加 `weight_mode` 参数
- 当 `weight_mode=True` 时，输出权重向量而非二元信号
- 在 `backtest-engine` 的 `native_adapter` 中支持权重向量输入
- 添加 FinRL-X 风格的 Selection → Allocation → Timing → Risk Overlay 四层管道

**影响范围**: `skills/strategy-model-engine/engine.py`, `skills/backtest-engine/scripts/adapters/native_adapter.py`

**工作量**: 中（需新增信号转换逻辑，适配回测引擎）

---

### 建议4: 声明式 YAML 工作流（优先级: 中）

**来源**: Microsoft Qlib qrun

**具体方案**:
- 在 `scripts/config.py` 基础上扩展 YAML 配置格式
- 支持从 YAML 文件一键启动完整流水线
- 配置中包含数据源、因子列表、模型参数、回测参数等

**影响范围**: `scripts/config.py`, `engine.py`

**工作量**: 中（需设计 YAML schema）

---

### 建议5: Rust 核心加速（优先级: 低，长期）

**来源**: QUANTAXIS QARSBridge

**具体方案**:
- 使用 PyO3/maturin 将回测核心循环用 Rust 重写
- 通过 Apache Arrow 在 Python 和 Rust 之间传递数据
- 优先加速因子计算和回测引擎中的热点循环

**影响范围**: 新增 `rust-core/` 目录

**工作量**: 大（需 Rust 开发和 FFI 集成）

---

## 五、附录

### A. 测试文件清单

| 文件 | 说明 |
|------|------|
| `tests/study_2026/test_point_in_time_safety.py` | PIT 安全检查、因子注册表、权重向量、性能基准测试 |

### B. 参考链接

- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [FinRL-X](https://github.com/AI4Finance-Foundation/FinRL)
- [QUANTAXIS](https://github.com/QUANTAXIS/QUANTAXIS)
- [Qlib 论文: "Qlib: An AI-oriented Quantitative Investment Platform"](https://arxiv.org/abs/2009.11189)
- [FinRL 论文: "FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance"](https://arxiv.org/abs/2011.09607)

### C. 运行测试命令

```bash
cd /workspace
python tests/study_2026/test_point_in_time_safety.py
```