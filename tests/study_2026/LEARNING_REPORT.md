# 量化交易开源项目学习报告

> **日期**: 2026-06-12  
> **序号**: #1  
> **研究范围**: 因子挖掘、回测框架、模型训练三大方向

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib (https://github.com/microsoft/qlib)
- **Stars**: 44k+
- **核心亮点**:
  - **表达式引擎 (Expression Engine)**: 通过声明式 DSL 定义因子，例如 `Ref($close, 60) / $close` 即可计算 60 日动量因子，无需写任何 Python 代码。支持嵌套表达式、算术/逻辑运算、滚动窗口函数（MA/Std/Skew/Kurt）、横截面算子（Rank）等。
  - **标准化因子集**: Alpha158（158个因子）和 Alpha360（360个因子），覆盖动量、反转、波动率、流动性、量价等维度，开箱即用。
  - **滚动窗口训练 (Rolling Window)**: RollingGen 生成多个滚动训练窗口，TrainerRM 在每个窗口上独立训练和评估，模拟真实投资场景。支持 purge gap 防止训练集和验证集重叠。
  - **高性能数据层**: 二进制 `.bin` 格式存储，多级缓存（C/H 缓存），表达式引擎预编译。

### 2. QUANTAXIS (https://github.com/yutiansut/QUANTAXIS)
- **Stars**: 25k+
- **核心亮点**:
  - **Rust 核心 + Python 桥接**: 通过 QARSBridge 实现 Rust 与 Python 互操作，账户结算 100x 加速，回测 10x 加速。
  - **零拷贝数据传递**: QADataBridge 使用 Apache Arrow + 共享内存，数据传递 5-10x 加速。
  - **QIFI 协议**: 统一账户模型，Python 端定义接口，Rust/C++ 端高性能实现，自动 fallback。
  - **微服务架构**: 数据服务、回测服务、交易服务分离部署，支持分布式回测。

### 3. Freqtrade (https://github.com/freqtrade/freqtrade)
- **Stars**: 25k+
- **核心亮点**:
  - **FreqAI 模块**: 自适应机器学习策略优化，支持在线学习、滚动检测、异常值检测。
  - **Hyperopt**: 贝叶斯优化（ExtraTreesRegressor）自动搜索最优参数。
  - **自适应重训练**: 连续模型重训练，适应市场变化。
  - **多线程并行**: 回测和参数优化均支持并行处理。

---

## 二、可借鉴的方向列表

| 优先级 | 方向 | 借鉴来源 | 对应 jingni-trader 模块 | 预期收益 |
|--------|------|----------|------------------------|----------|
| 高 | 表达式因子引擎 | Qlib Expression Engine | factor-engine | 因子定义无需硬编码，大幅提升因子库可扩展性 |
| 高 | 向量化回测优化 | QUANTAXIS Rust 核心 | backtest-engine | 减少逐行循环，提升回测性能 |
| 高 | 滚动窗口训练 | Qlib RollingGen | strategy-model-engine | 更准确的时序评估，避免过拟合 |
| 中 | Rust/Python 混合架构 | QUANTAXIS QARSBridge | 全局 | 10x+ 性能提升 |
| 中 | 零拷贝数据传递 | QUANTAXIS QADataBridge | data-engine | 减少数据复制开销 |
| 中 | 自适应模型重训练 | Freqtrade FreqAI | strategy-model-engine | 模型持续适应市场变化 |
| 低 | 贝叶斯超参优化 | Freqtrade Hyperopt | strategy-model-engine | 更高效的参数搜索 |
| 低 | 微服务架构 | QUANTAXIS | 全局 | 更好的可扩展性 |

---

## 三、已完成验证测试及结论

### 3.1 表达式因子引擎验证

**测试文件**: `tests/study_2026/test_expression_factor_engine.py`  
**测试结果**: 11/11 passed

**验证内容**:
- 基础算术表达式（`$close / Ref($close, 1) - 1`）
- 移动平均（`MA($close, 20)`）
- 复合表达式（`MA($close, 5) - MA($close, 20)`）
- 动量因子（`($close / Ref($close, 20)) - 1`）
- 波动率因子（`Std(Delta($close, 1), 20)`）
- 量价因子（`Log($volume) * Delta($close, 1)`）
- 排名因子（`Rank(Delta($close, 5))`）
- 嵌套表达式（`MA($close / Ref($close, 1) - 1, 5)`）
- 边界情况（纯常量、纯列引用）
- 与硬编码方式的计算结果一致性测试
- 因子定义灵活性测试（5 个 Qlib Alpha158 风格因子无需修改代码即可计算）

**性能对比** (50 stocks x 500 days):
- 硬编码方式: ~0.17s
- 表达式引擎: ~8.29s（48x 开销，主要是因为纯 Python 解析器和逐股票循环）
- 生产环境优化建议: 使用 numba/jit 加速循环，预编译表达式 AST，缓存中间结果

**结论**: 表达式引擎在正确性和灵活性上完全达标，性能开销在当前 Python 实现中可接受（生产环境可通过缓存和 JIT 优化）。**建议引入到 factor-engine 模块**。

### 3.2 向量化回测优化验证

**测试文件**: `tests/study_2026/test_vectorized_backtest.py`  
**测试结果**: 4/4 passed

**验证内容**:
- 向量化与逐行循环的结果一致性（交易记录数、总收益、最大回撤）
- 性能对比（20 stocks x 252 days, 10 runs）
- 大规模性能测试（100 stocks x 500 days）
- 边界条件测试（空数据、全零信号、单股票）

**性能对比** (20 stocks x 252 days):
- 向量化回测: ~0.006s
- 逐行循环: ~0.03s
- 加速比: ~5.0x

**大规模回测** (100 stocks x 500 days):
- 向量化耗时: ~0.25s

**结论**: 向量化回测在保持结果一致性的前提下实现 5x 加速，大规模数据下表现优异。**建议在 backtest-engine 中引入 numpy 向量化操作替代部分循环**。

### 3.3 滚动窗口训练验证

**测试文件**: `tests/study_2026/test_rolling_window_training.py`  
**测试结果**: 8/8 passed

**验证内容**:
- 三种时序切分方法对比（简单切分、Purged Group TS Split、滚动窗口）
- 未来信息泄露检测
- 滚动窗口训练与简单切分的表现对比
- IC 稳定性指标
- 过拟合检测（MSE 变异系数）
- 性能基准测试（50 stocks x 600 days）

**训练方法对比** (模拟数据):
- 简单切分: 验证集 MSE 0.0021, IC 0.85
- 滚动窗口 (8 个窗口): 平均验证 MSE 0.0025, IC 稳定性 8.5
- 滚动窗口额外提供了 MSE 标准差和 IC 标准差，可评估模型在不同时间段的一致性

**结论**: 滚动窗口训练提供了比简单切分更丰富的评估信息（IC 稳定性、MSE 波动性），能有效检测过拟合。**建议在 strategy-model-engine 中增加滚动窗口训练模式**。

---

## 四、待用户确认的优化建议

### 建议 1: 引入表达式因子引擎（推荐）
- **影响范围**: factor-engine
- **改动量**: 中等（新增 ExpressionParser 和 ExprOp 类层级）
- **向后兼容**: 完全兼容，现有硬编码因子可继续使用
- **风险**: 低（纯 Python 实现，不涉及外部依赖）
- **验证状态**: 已通过 11 项测试

### 建议 2: 回测引擎向量化优化（推荐）
- **影响范围**: backtest-engine
- **改动量**: 中等（重构调仓逻辑为 numpy 矩阵运算）
- **向后兼容**: 需要考虑与现有 rqalpha/backtrader 适配器的兼容性
- **风险**: 中（需保证向量的结果与现有逻辑一致）
- **验证状态**: 已通过 4 项测试，5x 加速

### 建议 3: 增加滚动窗口训练模式（推荐）
- **影响范围**: strategy-model-engine
- **改动量**: 小（新增 RollingWindowTrainer 类）
- **向后兼容**: 完全兼容，作为 train() 的可选模式
- **风险**: 低（基于现有 TimeSeriesSplit 扩展）
- **验证状态**: 已通过 8 项测试

### 建议 4: Rust/Python 混合架构（远期规划）
- **影响范围**: 全局
- **改动量**: 大（需要引入 Rust 编译工具链和 FFI）
- **风险**: 高（增加项目复杂度）
- **建议**: 在量化策略成熟后考虑

---

## 五、测试文件清单

| 文件 | 测试数 | 状态 |
|------|--------|------|
| `tests/study_2026/test_expression_factor_engine.py` | 11 | 全部通过 |
| `tests/study_2026/test_vectorized_backtest.py` | 4 | 全部通过 |
| `tests/study_2026/test_rolling_window_training.py` | 8 | 全部通过 |

**总计**: 23 tests, 0 failures

---

## 六、下一步行动

请用户审阅上述优化建议，确认后我将在 `feature/quant-stream-inspired` 分支上执行以下操作：

1. 将验证通过的代码迁移到对应 skill 模块
2. 更新相关配置和文档
3. 运行完整的回归测试
4. 提交代码（遵循 Conventional Commits 规范）

> 注: 在用户明确确认之前，不会执行任何 git commit/merge 操作。