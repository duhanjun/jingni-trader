# jingni-trader 量化交易学习报告

> 日期: 2026-06-14 | 序号: #1 | 学习周期: 2026-Q2

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib (42K+ stars)
- **仓库**: https://github.com/microsoft/qlib
- **最新提交**: 2026-04-22 (持续活跃)
- **核心亮点**:
  - **Expression Engine**: 声明式因子表达式 DSL，如 `$close / Ref($close, 1) - 1`，自动解析为 pandas 操作
  - **Alpha158 因子集**: 标准化 158 个 Alpha 因子，覆盖 K 线、价格、成交量等维度
  - **列式数据存储**: 专为量化优化的二进制格式 `qlib_data`，支持高性能读取
  - **RD-Agent**: 自动因子挖掘与强化学习驱动的因子发现
  - **模型 Zoo**: 内置 LightGBM、GRU、LSTM、Transformer 等模型

### 2. QUANTAXIS (25K+ stars)
- **仓库**: https://github.com/yutiansut/QUANTAXIS
- **最新提交**: 2026-02-28 (持续活跃)
- **核心亮点**:
  - **QIFI 协议**: 统一账户数据接口，Python/Rust 双版本零拷贝数据交换
  - **Python+Rust 混合架构**: 性能关键路径用 Rust 实现，Python 做业务逻辑
  - **全栈设计**: 数据采集 → 因子计算 → 回测 → 实盘，一站式解决方案
  - **多数据库支持**: MongoDB + ClickHouse + InfluxDB 时序优化

### 3. Freqtrade (25K+ stars)
- **仓库**: https://github.com/freqtrade/freqtrade
- **核心亮点**:
  - **FreqAI 模块**: 自适应 ML 训练管线，支持 rolling/expanding window 训练
  - **自动特征选择**: 基于模型 feature importance 的特征筛选
  - **离群值检测**: MAD/IQR 方法的异常值清洗
  - **事件驱动架构**: 基于消息总线的模块间通信
  - **完善的回测+模拟+实盘**: 统一的交易接口抽象

---

## 二、可借鉴方向列表

| 方向 | 来源 | 目标模块 | 优先级 | 可行性 |
|------|------|----------|--------|--------|
| 因子表达式 DSL | Qlib Expression Engine | factor-engine | 高 | 已验证 |
| Alpha158 因子集 | Qlib Alpha158 | factor-engine | 中 | 已验证 |
| 统一账户模型 | QUANTAXIS QIFI | execution-monitor-engine | 高 | 已验证 |
| 跨模块事件总线 | Freqtrade Event Bus | 全局架构 | 高 | 已验证 |
| 自适应 ML 训练管线 | Freqtrade FreqAI | strategy-model-engine | 中 | 已验证 |
| Rolling/Expanding Window | Freqtrade FreqAI | strategy-model-engine | 中 | 已验证 |
| 自动特征选择 | Freqtrade FreqAI | strategy-model-engine | 中 | 已验证 |
| 离群值检测 | Freqtrade FreqAI | data-engine | 低 | 已验证 |
| 列式二进制存储 | Qlib data storage | data-engine | 低 | 待验证 |
| Python+Rust 混合 | QUANTAXIS | backtest-engine | 低 | 待验证 |

---

## 三、已完成的验证测试及结论

### 测试环境
- Python 3.12, numpy 2.4, pandas 3.0, scikit-learn 1.9, lightgbm 4.x
- 测试文件: `tests/study_2026/test_*.py`
- 测试结果: **37 passed, 2 skipped, 0 failed**

### 优化方向 1: 因子表达式 DSL 引擎

**文件**: `tests/study_2026/test_factor_dsl.py` (17 tests, 16 passed, 1 skipped)

**验证内容**:
1. DSL 解析器: 支持列引用 `$close`、算术运算、函数调用 `Ref()`, `Mean()`, `Std()`, `Max()`, `Min()`, `Sum()`, `Log()`, `Abs()`, `Sign()`, `Rank()`, `Delay()`, `Delta()`, `PctChange()`
2. 18 个 Alpha158 因子子集全部可通过 DSL 声明式定义
3. 编译器将 AST 编译为 pandas 操作序列
4. 与硬编码计算对比: 结果完全一致，性能接近（DSL 编译一次后可重复执行）

**性能数据**:
- 18 个因子批量计算 (100 只股票 × 500 日): 约 0.5-1.5s，平均每个因子 < 0.1s
- 与硬编码方案的性能差异在可接受范围内（< 2x）

**结论**: 因子 DSL 方案可行，可显著提升因子库的可扩展性。用户只需声明因子表达式而无需修改核心引擎代码。

### 优化方向 2: 统一账户模型 (QIFI 风格)

**文件**: `tests/study_2026/test_unified_account.py` (14 tests, 13 passed, 1 skipped)

**验证内容**:
1. 定义了 `Position`, `Order`, `Trade`, `Account` 四个统一数据模型
2. 实现了 `AccountEventBus` 事件总线，支持跨模块订阅/发布
3. 序列化往返测试: JSON/dict 序列化正确，支持从字典恢复
4. 模拟了「回测 → 风控 → 执行」三模块协作流程
5. 内存效率: dataclass 方案与字典方案内存占用相当，但类型安全性更高

**关键指标**:
- 50 只持仓的账户模型: 内存占用 < 100KB (dataclass)
- 序列化速度: to_dict() ~ 3us/次, to_json() ~ 30us/次

**结论**: 统一账户模型可有效解决当前各模块数据模型不一致的问题，事件总线机制可解耦模块间通信。

### 优化方向 3: 自适应 ML 训练管线

**文件**: `tests/study_2026/test_automl_pipeline.py` (8 tests, 8 passed)

**验证内容**:
1. Rolling Window 和 Expanding Window 两种训练窗口生成
2. 离群值检测 (MAD 和 IQR 方法)
3. 基于 feature importance 的自动特征选择
4. 重训练触发条件判断
5. 自适应 vs 固定窗口的 IC 对比（在时间漂移数据上自适应窗口胜出）

**关键指标**:
- 特征选择后预测速度提升约 2-3x (20 特征 → 5 特征)
- 在有时间漂移的数据上，自适应窗口 IC 优于固定窗口
- 训练时间: 5000 样本 × 20 特征，一次训练 < 0.5s

**结论**: 自适应 ML 管线可有效应对市场风格漂移，自动特征选择可降低模型复杂度并提升推理速度。

---

## 四、待用户确认的优化建议

### 高优先级 (建议尽快实施)

1. **因子 DSL 集成到 factor-engine**
   - 在 `FactorEngine` 中增加 `register_factor()` 和 `compute_dsl_factors()` 方法
   - 保持现有 `compute_a_share_factors()` 兼容，新旧并行
   - 涉及的 scope: `factor-engine`

2. **统一账户模型替换分散模型**
   - 将 `Account` dataclass 移至 `skills/execution-monitor-engine/` 作为共享模型
   - 其他模块通过事件总线订阅账户变更
   - 涉及的 scope: `execution-monitor-engine`, `portfolio-risk-engine`, `backtest-engine`

3. **事件总线引入**
   - 将 `AccountEventBus` 提升为全局架构组件
   - 各引擎模块通过事件总线实现松耦合通信
   - 涉及的 scope: 全局架构

### 中优先级 (建议列入 roadmap)

4. **Alpha158 因子集补齐**
   - 基于 DSL 将因子库从 ~15 个扩展到 50+ 个
   - 涉及的 scope: `factor-engine`

5. **自适应 ML 训练管线**
   - 在 `strategy-model-engine` 中增加 `AdaptiveMLPipeline` 类
   - 支持 Rolling Window 和 Expanding Window 训练模式
   - 涉及的 scope: `strategy-model-engine`

### 低优先级 (可后续评估)

6. **列式二进制存储优化**
   - 评估 qlib 的二进制存储格式是否适合 jingni-trader
   - 涉及的 scope: `data-engine`

7. **Rust 加速关键路径**
   - 评估 backtest 回测引擎中 Rust 加速的可行性
   - 涉及的 scope: `backtest-engine`

---

## 五、文件清单

```
tests/study_2026/
├── LEARNING_REPORT.md          ← 本报告
├── test_factor_dsl.py           ← 因子 DSL 验证测试 (17 tests)
├── test_unified_account.py      ← 统一账户模型验证测试 (14 tests)
└── test_automl_pipeline.py      ← 自适应 ML 管线验证测试 (8 tests)
```

---

> **重要提醒**: 所有优化代码在用户明确确认之前，不得执行 git commit/push/merge 操作。
> 验证代码位于独立的 `tests/study_2026/` 目录中，未修改任何主代码文件。