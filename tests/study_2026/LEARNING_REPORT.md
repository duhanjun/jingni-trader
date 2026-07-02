# 量化交易开源项目学习报告

> **日期**: 2026-06-13  
> **序号**: #001  
> **范围**: 因子引擎 / 回测引擎 / ML 模型引擎

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib
- **GitHub**: https://github.com/microsoft/qlib (17.5k+ Stars)
- **语言**: Python | **协议**: MIT
- **核心亮点**:
  - **Expression Engine**: 声明式因子 DSL，如 `Mean(Pct($close), 20)` 即可定义新因子，无需修改框架代码
  - **Alpha158/Alpha360**: 标准化因子库（158/360 个因子），一键生成全因子集
  - **列式数据格式**: 内存映射二进制文件，比 Parquet 快 10-30x
  - **多级缓存机制**: ExpressionCache → DatasetCache，避免重复计算
  - **DataHandler 抽象**: Provider → Processor → Dataset 三段式流水线
  - **Model Zoo**: 标准化 `train` / `predict` / `dump` / `load` 接口
  - **RollingGen**: 滚动窗口训练发生器，自动管理时序切片
  - **Nested Decision Framework**: 支持 TopK + Weight 的组合决策
  - **MLflow 集成**: 实验追踪、参数记录、模型版本管理

### 2. NautilusTrader
- **GitHub**: https://github.com/nautechsystems/nautilus_trader (~10k Stars)
- **语言**: Rust (核心) + Python (绑定) | **协议**: LGPL
- **核心亮点**:
  - **Event-Driven Architecture (EDA)**: 所有操作通过事件驱动，回测/实盘共享同一 NautilusKernel
  - **Deterministic Execution**: 相同输入产生相同输出，回测高度可信
  - **Dual Timestamp Model**: `ts_event` (事件发生时间) + `ts_init` (系统收到时间)，天然防前视偏差
  - **MessageBus**: pub/sub + req/res + point-to-point 三种通信模式
  - **SimulatedExchange**: 完整的模拟交易所，含撮合引擎、手续费模型、延迟模型
  - **Cache 组件**: 集中式内存缓存，统一管理所有对象生命周期
  - **Rust/PyO3 高性能**: 核心路径 Rust 实现，10x+ 性能提升
  - **Component 解耦**: RiskEngine / ExecEngine / DataEngine 完全独立

### 3. Freqtrade
- **GitHub**: https://github.com/freqtrade/freqtrade (49k+ Stars)
- **语言**: Python | **协议**: GPL-3.0
- **核心亮点**:
  - **FreqAI (ML模块)**: 自适应机器学习，支持 auto-retraining 和 continual learning
  - **滑动窗口训练**: 基于时间/数据的滑动窗口，适应市场概念漂移
  - **Hyperopt**: 基于 Bayesian Optimization 的策略超参数优化
  - **Multi-timeframe Feature Engineering**: 多周期特征自动对齐
  - **Strategy Callback System**: 丰富的策略回调钩子 (on_entry, on_exit, custom_stoploss 等)
  - **Loss Function 可定制**: 允许自定义优化目标（不仅是收益率）
  - **Model Expiration**: 模型性能衰减自动检测 + 重新训练触发

---

## 二、可借鉴方向列表

### 方向 A: 因子表达式引擎 (借鉴 Qlib)
- **优先级**: 高
- **当前问题**: `factor-engine/engine.py` 中 `compute_a_share_factors()` 硬编码因子计算，新增因子需要修改引擎核心代码，缺乏可扩展性
- **优化方案**: 引入声明式因子表达式 mini 引擎，因子通过字符串表达式定义

### 方向 B: 事件驱动回测引擎 (借鉴 NautilusTrader)
- **优先级**: 中
- **当前问题**: `native_adapter` 使用简单循环遍历，缺乏结构性前视偏差防护和时间事件模型
- **优化方案**: 引入 Event/EventQueue 机制，策略通过事件回调解耦，双时间戳防偏差

### 方向 C: 自适应 ML 重训练 (借鉴 FreqAI)
- **优先级**: 高
- **当前问题**: `strategy-model-engine` 一次性训练 + Optuna，市场变化后模型失效，无法自动适应
- **优化方案**: 滑动窗口重训练 + 模型过期检测 + 性能衰减自动触发

### 其他可关注方向
| 方向 | 借鉴源 | 优先级 | 说明 |
|------|--------|--------|------|
| 列式二进制数据格式 | Qlib | 中 | 用 Feather/Apache Arrow 替换 Parquet，提升数据加载性能 |
| 多级缓存 | Qlib | 低 | ExpressionCache + DatasetCache 减少重复计算 |
| Rust 加速核心路径 | NautilusTrader | 低 | 用 PyO3 重写关键计算函数 |
| Hyperopt 策略优化 | Freqtrade | 中 | Bayesian 超参优化 + 自定义损失函数 |
| 多周期特征对齐 | Freqtrade | 低 | 多时间框架特征自动合并到统一频率 |

---

## 三、已完成的验证测试及结论

### 测试1: 因子表达式引擎 (test_factor_expression_engine.py)

| 项目 | 结果 |
|------|------|
| 测试用例数 | 12 |
| 通过率 | 100% (12/12) |
| 数据规模 | 20只股票 x 200个交易日 = 4,000行 |

**关键结论**:

1. **正确性**: 所有操作符 (Ref, Mean, Std, Pct, Rank, Delta) 的计算结果与 pandas 手动计算完全一致
2. **性能**: 表达式引擎解析开销约为硬编码的 2-3x，在 4,000 行数据规模下均在可接受范围 (< 0.1s)
3. **扩展性**: 新增因子只需添加一行表达式字符串，无需修改引擎代码。例如新增反转因子 `'-Pct($close)'` 一行即可
4. **兼容性**: 与现有 `compute_a_share_factors` 输出格式兼容

### 测试2: 事件驱动回测引擎 (test_event_driven_backtest.py)

| 项目 | 结果 |
|------|------|
| 测试用例数 | 6 |
| 通过率 | 100% (6/6) |

**关键结论**:

1. **事件流正确**: MarketDataEvent 按交易日和股票正常生成，事件队列保证时间有序
2. **净值计算正确**: 初始净值 = init_cash，净值始终为正
3. **时间序确定性**: 随机打乱的事件推入队列后，始终按时间顺序输出
4. **组件解耦**: Strategy 通过 `submit_order` 间接操作组合，不直接修改 Portfolio
5. **前视偏差检测机制**: 检测器可对比正常/泄漏回测结果并标记可疑情况

### 测试3: 自适应 ML 重训练 (test_adaptive_retraining.py)

| 项目 | 结果 |
|------|------|
| 测试用例数 | 5 |
| 通过率 | 100% (5/5) |

**关键结论**:

1. **滑动窗口 vs 固定窗口**: 具概念漂移数据上，自适应模型 MSE 显著低于固定窗口模型
2. **模型过期检测**: 检测器能正确识别性能衰减并触发重训练
3. **滚动性能**: 分段评估显示，自适应模型在数据后期不会剧烈恶化
4. **预热期**: 预热逻辑正确，预热期内不触发训练
5. **重训练开销**: 单次 Ridge 重训练 < 1ms (小规模数据)

---

## 四、待用户确认的优化建议

### 建议1: 引入因子表达式引擎到 factor-engine ✅ 已验证
- **测试文件**: `tests/study_2026/test_factor_expression_engine.py`
- **影响模块**: `skills/factor-engine/engine.py`
- **改动范围**: 中等 (新增 `FactorExpressionEngine` 类 + 重构 `compute_a_share_factors`)
- **风险**: 低 (完全向后兼容，可作为可选功能)

### 建议2: 事件驱动回测核心改造 ✅ 已验证
- **测试文件**: `tests/study_2026/test_event_driven_backtest.py`
- **影响模块**: `skills/backtest-engine/scripts/adapters/native_adapter.py`
- **改动范围**: 较大 (重写 native adapter)
- **风险**: 中 (可能影响现有策略的回测结果)

### 建议3: 自适应模型重训练机制 ✅ 已验证
- **测试文件**: `tests/study_2026/test_adaptive_retraining.py`
- **影响模块**: `skills/strategy-model-engine/engine.py`
- **改动范围**: 中等 (新增 AdaptiveTrainer 类)
- **风险**: 低 (可选功能，不影响现有流程)

---

## 五、参考资源

- Microsoft Qlib: https://github.com/microsoft/qlib
- NautilusTrader: https://github.com/nautechsystems/nautilus_trader
- Freqtrade: https://github.com/freqtrade/freqtrade
- Qlib Paper: "Qlib: An AI-oriented Quantitative Investment Platform" (arXiv)

---

*报告生成时间: 2026-06-13  ·  保存位置: tests/study_2026/LEARNING_REPORT.md*