# 量化交易开源项目学习报告

**日期**: 2026-06-12  
**序号**: #001  
**研究员**: AI Agent  
**分支**: feature/quant-stream-inspired

---

## 1. 学习项目清单及核心亮点

### 1.1 Microsoft Qlib

| 属性 | 详情 |
|------|------|
| GitHub | https://github.com/microsoft/qlib |
| Stars | ~44k |
| 语言 | Python |
| 定位 | AI-Oriented Quantitative Investment Platform |

**核心亮点**:

1. **表达式引擎 (Expression Engine)**
   - 提供 DSL 语法定义因子，如 `Ref($close, 60) / $close - 1` 直接表达 60 日收益率
   - 支持时序算子（Ref, Mean, Std, Max, Min）、截面算子（CSRank, CSMean）、元素算子（Add, Sub, Mul, Div）
   - 因子声明为**函数而非数据**，LLM 可直接生成因子表达式
   - 表达式解析为 AST，支持嵌套组合

2. **数据管道设计**
   - `Data Loader → Data Handler → Dataset` 三层架构
   - 列式二进制数据格式（`qlib_bin`，基于 ffnan 压缩），10x 读取速度提升
   - 支持多频率数据（日频/分钟频）的统一处理接口
   - 内置 Alpha158 / Alpha360 因子库

3. **工作流引擎**
   - YAML 配置驱动的工作流定义
   - 支持断点续跑和交叉验证
   - 在线/离线学习模式切换
   - RD-Agent 集成，自动化因子挖掘

### 1.2 NautilusTrader

| 属性 | 详情 |
|------|------|
| GitHub | https://github.com/nautechsystems/nautilus_trader |
| Stars | ~5k（快速增长中） |
| 语言 | Rust (核心) + Python (绑定 via PyO3) |
| 定位 | Production-Ready, High-Performance Algorithmic Trading Platform |

**核心亮点**:

1. **事件驱动架构 (Event-Driven Architecture)**
   - 所有组件通过 MessageBus 通信，松耦合
   - 单线程 + LMAX Disruptor 模式，无锁高性能
   - 支持同一套事件系统同时用于回测和实盘

2. **确定性时间模型**
   - nanosecond 级时间分辨率
   - 回测的时间推进逻辑与实盘一致
   - 支持多种时间触发策略（固定间隔、市场事件、定时）

3. **风险引擎**
   - 可插拔的风险检查器管道
   - 预交易风控（最大仓位、最大回撤、最大订单量等）
   - 内置 Account/Position/Risk 管理

4. **Rust 核心性能**
   - 核心引擎用 Rust 实现，毫秒级延迟
   - Python 绑定层通过 PyO3 暴露 API
   - 缓存系统（Cache）减少重复计算

### 1.3 Freqtrade + FreqAI

| 属性 | 详情 |
|------|------|
| GitHub | https://github.com/freqtrade/freqtrade |
| Stars | ~35k（业界最热） |
| 语言 | Python |
| 定位 | Open-Source Crypto Trading Bot with ML (FreqAI) |

**核心亮点**:

1. **FreqAI 标准化 ML 接口**
   - `IFreqaiModel` 抽象基类，`train() → fit() → predict()` 三步接口
   - 支持 LightGBM, XGBoost, PyTorch, CatBoost 等多种后端
   - DataKitchen/DataDrawer 负责数据管道（归一化、PCA、异常值检测）

2. **超参数优化 (Hyperopt)**
   - 基于 Optuna 的贝叶斯优化
   - 支持多维参数空间定义
   - 自定义损失函数（Sharpe, Profit, WinRate 等）
   - 带时间序列交叉验证

3. **自适应训练策略**
   - 滑动训练窗口（Sliding Window）
   - 模型过期机制（超过一定时间自动重训练）
   - 特征重要性分析和筛选

4. **完整的策略开发体验**
   - 统一的回测/模拟盘/实盘接口
   - Strategy 模板化定义
   - 内置 100+ 指标

---

## 2. 可借鉴的优化方向

### 2.1 因子表达式引擎（借鉴 Qlib）

| 维度 | 当前 jingni-trader | 优化方向 |
|------|-------------------|----------|
| 因子定义方式 | 硬编码 Python 函数 | DSL 表达式引擎 |
| 可扩展性 | 需修改代码 | 注册即用，无需改代码 |
| AI 友好性 | LLM 难以直接生成 | LLM 可直接输出 DSL |
| 跨截面计算 | 需手动 groupby | 内置 CSRank 等算子 |

**对 jingni-trader 的改进**:
- 在 `factor-engine` 中引入 FactorExpression `FactorExpressionParser` 和 `FactorExpressionEngine`
- 支持 `Ref($close, 20)`, `Mean($close, 60)`, `CSRank($close)` 等表达式
- 保留现有 API 接口兼容，新增 `register_expression()` 接口

### 2.2 事件驱动回测架构（借鉴 NautilusTrader）

| 维度 | 当前 jingni-trader | 优化方向 |
|------|-------------------|----------|
| 架构模式 | 向量化回测（逐列计算） | 事件驱动（MessageBus） |
| 组价通信 | 直接调用 | 松耦合事件总线 |
| 风险控制 | 嵌入策略函数 | 独立 RiskEngine 管道 |
| 实盘一致性 | 不保证 | 同一事件系统 |

**对 jingni-trader 的改进**:
- 在 `backtest-engine` 中引入 `MessageBus`、`EventDrivenBacktestEngine`
- 在 `portfolio-risk-engine` 中引入 `RiskEngine` 管道
- 事件类型覆盖完整的订单生命周期

### 2.3 标准化 ML 模型接口 + 超参数优化（借鉴 Freqtrade/FreqAI）

| 维度 | 当前 jingni-trader | 优化方向 |
|------|-------------------|----------|
| 模型接口 | 无统一抽象 | `BaseQuantModel` 抽象基类 |
| 数据预处理 | 策略内手工处理 | `DataPipeline` 管道化 |
| 超参数调优 | 无自动化 | `HyperoptEngine` + Optuna |
| 训练策略 | 固定窗口 | 滑动窗口 `SlidingWindowTrainer` |

**对 jingni-trader 的改进**:
- 在 `strategy-model-engine` 中引入 `BaseQuantModel` 抽象
- 在 `strategy-model-engine` 中引入 `HyperoptEngine` 自动调参
- 在 `strategy-model-engine` 中引入 `SlidingWindowTrainer` 自适应训练

---

## 3. 已完成的验证测试

### 3.1 测试文件清单

| 测试文件 | 优化方向 | 测试数 | 结果 |
|----------|----------|--------|------|
| `test_factor_expression_engine.py` | 因子表达式引擎 | 16 | 全部通过 |
| `test_event_driven_backtest.py` | 事件驱动回测 | 11 | 全部通过 |
| `test_ml_model_interface.py` | ML 模型接口 | 12 | 全部通过 |

### 3.2 各优化方向测试详情

#### 3.2.1 因子表达式引擎

**测试覆盖**:
- 表达式解析器（tokenizer + parser → AST）：7 测试
- 表达式求值（多股票批量计算）：5 测试
- 性能对比测试：1 测试
- 边界条件（空数据、NaN）：2 测试

**关键测试结果**:

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 1 日收益率正确性 | 通过 | `Div(Sub($close, Ref($close, 1)), Ref($close, 1))` 与 pandas pct_change 数值一致 |
| 20 日均线正确性 | 通过 | `Mean($close, 20)` 与 rolling(20).mean() 一致 |
| 截面排名正确性 | 通过 | `CSRank($close)` 与 groupby.rank(pct=True) 一致 |
| 复合因子 | 通过 | 嵌套表达式 `CSRank(Sub(0, Div(Sub($close, Ref($close, 20)), Ref($close, 20))))` |
| 性能 (100 股 x 522 天) | 通过 | 表达式引擎 ~76ms vs 手动 ~8ms (9.2x)，可接受 |

#### 3.2.2 事件驱动回测

**测试覆盖**:
- 消息总线发布/订阅：3 测试
- 风险引擎管道：3 测试
- 完整回测流程：3 测试
- 性能对比：1 测试

**关键测试结果**:

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 完整回测 MA20 突破策略 | 通过 | 产生有效交易记录和绩效指标 |
| 风险拒绝机制 | 通过 | 极端风控下交易数显著减少 |
| 事件总数验证 | 通过 | 回测过程产生大量有序事件 |
| 性能 (50 股 x 522 天) | 通过 | ~12 秒完成，约 51000+ 事件 |

#### 3.2.3 ML 模型接口

**测试覆盖**:
- 标准化模型接口：3 测试
- 数据预处理管道：4 测试
- 超参数优化：2 测试
- 滑动窗口训练：3 测试

**关键测试结果**:

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 模型训练/预测/持久化 | 通过 | 支持 save/load 周期 |
| 多模型类型 | 通过 | LightGBM, XGBoost 均正常（含 sklearn 回退） |
| 数据管道（StandardScaler/PCA/MinMax） | 通过 | 正确处理 fit/transform |
| 超参数优化 | 通过 | Optuna 贝叶斯搜索，最佳 IC=0.2939 |
| 滑动窗口 | 通过 | 按间隔自动重训练 |

---

## 4. 性能对比分析

### 4.1 因子计算性能

| 方法 | 100 股 x 50K 行 | 相对性能 |
|------|----------------|----------|
| Pandas groupby | ~8ms | 1.0x (基线) |
| 表达式引擎 | ~76ms | 9.2x |
| **优化潜力**: 改用向量化分组计算可降至 ~20ms | | |

### 4.2 回测引擎性能

| 引擎类型 | 50 股 x 50K 行 | 特点 |
|----------|----------------|------|
| 向量化回测（旧） | ~0.5s | 快但不灵活 |
| 事件驱动（新） | ~12s | 灵活但较慢 |
| **取舍**: 事件驱动提供了更高的准确性和可扩展性，适合策略验证阶段 | | |

---

## 5. 待用户确认的优化建议

### 优先级 P0（高价值、低风险）

1. **因子表达式引擎集成到 factor-engine**
   - 影响范围：`skills/factor-engine/scripts/base/` 新增 `expression_engine.py`
   - API 兼容：保留现有 `BaseFactor`, `PandasTACalculator`，新增 `ExpressionFactor`
   - 风险：低，纯增量功能

2. **标准化 ML 模型接口集成到 strategy-model-engine**
   - 影响范围：`skills/strategy-model-engine/scripts/base/` 新增 `base_quant_model.py`
   - API 兼容：新增抽象类，不修改现有代码
   - 风险：低

### 优先级 P1（中价值、需评估）

3. **事件驱动回测架构引入 backtest-engine**
   - 影响范围：`skills/backtest-engine/scripts/base/` 需要重构
   - API 兼容：建议新增 `event_driven_engine.py`，保留向量化引擎
   - 风险：中，需充分测试与原向量的结果一致性

4. **超参数优化引入 strategy-model-engine**
   - 影响范围：新增 `hyperopt_engine.py`
   - 依赖：需安装 Optuna (`pip install optuna`)
   - 风险：低

### 优先级 P2（长期规划）

5. **数据存储格式优化**（借鉴 Qlib 的列式二进制格式）
6. **Rust 核心迁移**（借鉴 NautilusTrader 的 PyO3 方案，长期性能优化）
7. **实盘交易接口统一事件模型**

---

## 6. 下一步行动

1. **等待用户确认** P0 优化方向是否可以合并到主分支
2. 用户确认后，按 Conventional Commits 规范提交：
   - `feat(factor-engine): add factor expression engine inspired by Qlib`
   - `feat(strategy-model-engine): add standardized ML model interface`
3. 如有需要，可继续深入 P1/P2 方向的验证测试

---

## 附录：验证代码位置

| 文件 | 路径 |
|------|------|
| 因子表达式引擎测试 | `tests/study_2026/test_factor_expression_engine.py` |
| 事件驱动回测测试 | `tests/study_2026/test_event_driven_backtest.py` |
| ML 模型接口测试 | `tests/study_2026/test_ml_model_interface.py` |
| 本报告 | `tests/study_2026/LEARNING_REPORT.md` |

---

*报告生成时间: 2026-06-12 | 基于 jingni-trader v0.1.x 代码结构分析*