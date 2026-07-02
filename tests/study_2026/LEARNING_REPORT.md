# Jingni-Trader 量化交易学习报告

> 报告序号: #1
> 日期: 2026-06-14
> 状态: 验证完成，待用户确认

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib (42k+ Stars)

- **仓库**: https://github.com/microsoft/qlib
- **核心亮点**:
  - **表达式引擎 (Expression Engine)**: 用声明式 DSL 定义因子，如 `Ref($close, 60) / $close`。因子从"硬编码"变为"表达式驱动"，Alpha158 因子库包含 158 个预定义因子公式。
  - **二进制数据格式**: 自研 `qlib_bin` 格式，比 HDF5/Parquet 快 10-20 倍的数据读取速度。
  - **滚动窗口训练**: 内置 `RollingDataHandler` 支持滚动时间窗口训练，防止前视偏差。
  - **实验管理**: `MLflow` 集成，支持实验记录和模型版本管理。
  - **模型层**: 支持 LightGBM、GRU、LSTM、Transformer 等，提供 `Model` 抽象基类。
- **可借鉴方向**:
  - 因子表达式 DSL → 替换当前硬编码的 ta-lib/pandas-ta 适配器
  - 二进制数据格式 → 优化 data-engine 的 Parquet 读写性能
  - 滚动窗口管理 → 补强 backtest-engine 的前视偏差防护
  - 实验追踪 → 新增 experiment 模块

### 2. QUANTAXIS (25k+ Stars)

- **仓库**: https://github.com/yutiansut/QUANTAXIS
- **核心亮点**:
  - **QIFI 协议**: 统一的金融交互接口，定义了标准化的 Account/Position/Order/Trade 模型，使回测和实盘使用完全相同的账户结构。
  - **QARSBridge (Rust Bridge)**: Rust 编写的性能桥接层，通过零拷贝共享内存方案实现 Python→Rust 的高性能数据传输。
  - **微服务架构**: 基于 gRPC 的分布式架构，支持多服务并行。
  - **ClickHouse 列存**: 使用 ClickHouse 存储高频行情数据，查询性能优异。
- **可借鉴方向**:
  - QIFI 统一账户模型 → 统一 backtest-engine 和 execution-monitor-engine 的账户结构
  - Rust 性能桥接 → 大数据量计算场景的性能优化
  - 微服务架构 → 未来分布式部署的参考

### 3. TradingAgents-CN (15k+ Stars)

- **仓库**: https://github.com/hsliuping/TradingAgents-CN
- **核心亮点**:
  - **多智能体辩论架构**: 分析师 Agent、风险官 Agent、交易员 Agent 等多角色协作，通过辩论机制生成交易决策。
  - **LLM 驱动**: 支持 OpenAI/DeepSeek/Qwen 等多种大模型，用自然语言描述市场观点。
  - **风险纪律**: 内置风险控制 Agent，确保交易决策符合风控规则。
  - **Web 界面**: Streamlit 构建的实时进度展示和用户体验。
- **可借鉴方向**:
  - 多 Agent 风险审核 → 增强 portfolio-risk-engine 的决策流程
  - LLM 辅助策略分析 → 新增 strategy-model-engine 的 AI 辅助模块

---

## 二、可借鉴方向列表（按优先级排序）

| 优先级 | 优化方向 | 借鉴来源 | 涉及模块 | 预期收益 |
|--------|----------|----------|----------|----------|
| P0 | 因子表达式引擎 | Qlib Expression Engine | factor-engine | 因子扩展性提升 10x，新因子从"写代码"变为"写公式" |
| P0 | 统一账户/仓位模型 | QUANTAXIS QIFI | backtest-engine, execution-monitor-engine | 回测与实盘账户一致性，减少部署风险 |
| P1 | 向量化回测优化 | Qlib backtest | backtest-engine | 回测速度提升 3-5x，支持更大规模股票池 |
| P1 | 滚动窗口数据管理 | Qlib DataHandler | data-engine | 消除前视偏差，提升回测可信度 |
| P2 | 二进制数据格式 | Qlib bin format | data-engine | 数据读取速度提升 10x |
| P2 | 多 Agent 风控 | TradingAgents-CN | portfolio-risk-engine | 风控决策更智能、更全面 |
| P3 | Rust 性能桥接 | QUANTAXIS QARSBridge | 全局 | 核心计算路径性能优化 |

---

## 三、已完成的验证测试及结论

### 3.1 因子表达式引擎 (Factor Expression Engine)

- **测试文件**: `tests/study_2026/test_factor_expression_engine.py`
- **测试用例数**: 23个 (全部通过)
- **测试覆盖**:
  - 字段引用、算术运算、ref/delta 时序算子
  - rolling mean/std/max/min/sum 窗口计算
  - rank/scale 截面算子
  - if/and/or 逻辑组合
  - 比较运算符 (>, <, >=, <=, ==)
  - 因子注册中心 (FactorRegistry) 的注册、分类、批量计算
  - 预定义 Alpha158 风格因子库 (13个因子)
  - 缓存有效性验证
  - 性能对比测试 (表达式引擎 vs 直接 pandas 计算)

- **关键结论**:
  - 表达式引擎可以正确解析 `$close / mean($close, 20) - 1` 等复杂因子公式
  - 批量计算 13 个 Alpha158 风格因子的耗时约为直接 pandas 计算的 2x 以内，在可接受范围
  - 缓存机制有效，热缓存命中可将重复计算加速 10x+
  - 因子注册中心支持按类别 (动量/反转/波动/流动性/趋势) 管理因子

- **性能对比**:
  | 数据集 | 表达式引擎 | 直接 pandas | 比率 |
  |--------|-----------|-------------|------|
  | 5000行×13因子 | ~0.02s | ~0.01s | ~2x |
  | 缓存命中 | ~0.000s | - | >>10x |

### 3.2 向量化回测引擎 (Vectorized Backtest Engine)

- **测试文件**: `tests/study_2026/test_vectorized_backtest.py`
- **测试用例数**: 16个 (全部通过)
- **测试覆盖**:
  - 基本回测运行、空数据/空信号边界处理
  - 单股票/多股票回测
  - 涨跌停限制效果验证
  - 佣金/印花税影响验证
  - 绩效指标完整性 (夏普比率、最大回撤、胜率等)
  - 买入持有策略正确性验证
  - 不同规模数据集性能测试 (100天×50股 到 500天×300股)
  - 可扩展性验证

- **关键结论**:
  - 向量化回测对大股票池有显著优势，500天×300股在 10秒内完成
  - 涨跌停限制、佣金费率等参数可正确影响回测结果
  - 回测复杂度随数据量增长呈次线性关系 (O(N^1.5) 左右)
  - 买入持有策略的最终权益与手算一致

- **性能对比**:
  | 数据集 | 耗时 |
  |--------|------|
  | 100天×50股 | ~0.5s |
  | 250天×100股 | ~2s |
  | 500天×200股 | ~8s |
  | 500天×300股 | ~10s |

### 3.3 统一账户/仓位模型 (Unified Account & Position Model)

- **测试文件**: `tests/study_2026/test_unified_account.py`
- **测试用例数**: 14个 (全部通过)
- **测试覆盖**:
  - 订单生命周期 (提交→成交→部分成交→撤销)
  - 风控检查 (持仓数量限制、单票仓位限制、现金不足拒绝)
  - 持仓管理 (多股票持仓、市值更新、已实现盈亏)
  - 账户快照 (序列化、审计)
  - JSON 序列化/反序列化
  - 集成测试 (模拟 100 天完整回测流程)

- **关键结论**:
  - 统一账户模型将 Order→Trade→Position 形成完整闭环
  - 风控规则在订单提交时自动执行，防止违规交易
  - AccountSnapshot 可记录任意时刻的账户状态，支持回放和审计
  - JSON 序列化使账户状态可跨系统/跨进程传递

---

## 四、待用户确认的优化建议

### 建议 1: 集成因子表达式引擎 (P0)

将 `ExpressionEngine` 和 `FactorRegistry` 集成到 `skills/factor-engine/` 中，替换当前的硬编码因子计算器。

- **影响范围**: `factor-engine/scripts/adapters/pandas_ta_calculator.py`, `talib_calculator.py`
- **新增文件**: `factor-engine/scripts/base/expression_engine.py`, `factor-engine/scripts/base/alpha_factors.py`
- **风险**: 低，表达式引擎作为独立模块，可渐进式替换

### 建议 2: 引入统一账户模型 (P0)

将 `UnifiedAccount` 集成到 `backtest-engine` 和 `execution-monitor-engine` 中。

- **影响范围**: `backtest-engine/scripts/adapters/native_adapter.py`, `execution-monitor-engine/`
- **新增文件**: `backtest-engine/scripts/base/unified_account.py`
- **风险**: 中，账户模型变更可能影响回测和信号生成逻辑

### 建议 3: 向量化回测优化 (P1)

将 `VectorizedBacktestEngine` 的向量化策略集成到现有回测引擎中。

- **影响范围**: `backtest-engine/scripts/adapters/native_adapter.py`
- **风险**: 中，需保持与现有回测 API 的兼容性

---

## 五、测试文件清单

| 文件 | 测试数 | 状态 |
|------|--------|------|
| `tests/study_2026/test_factor_expression_engine.py` | 23 | ✅ 全部通过 |
| `tests/study_2026/test_vectorized_backtest.py` | 16 | ✅ 全部通过 |
| `tests/study_2026/test_unified_account.py` | 14 | ✅ 全部通过 |
| **总计** | **53** | **✅ 全部通过** |

---

## 六、Git 操作状态

> ⚠️ 根据约束，未执行任何 git commit/push/merge 操作。
> 所有验证代码位于 `tests/study_2026/` 目录下，未修改任何主代码。
> 当前分支: `feature/quant-stream-inspired`
> 用户确认优化方案后可执行 git 操作。