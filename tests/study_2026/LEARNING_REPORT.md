# jingni-trader 量化交易学习报告

---

## 日期: 2026-06-14
## 序号: #1
## 学习周期: 2026年第1轮

---

## 一、学习项目清单

| 序号 | 项目名称 | GitHub Stars | 语言 | 核心特色 |
|------|----------|-------------|------|----------|
| 1 | [Microsoft Qlib](https://github.com/microsoft/qlib) | 15k+ | Python | AI量化平台，表达式引擎，PIT数据库 |
| 2 | [QUANTAXIS](https://github.com/yutiansut/QUANTAXIS) | 25k+ | Python/Rust | 混合架构，分布式，QIFI协议 |
| 3 | [Freqtrade/FreqAI](https://github.com/freqtrade/freqtrade) | 25k+ | Python | 自适应ML训练，策略-模型解耦 |

---

## 二、各项目核心亮点与可借鉴之处

### 2.1 Microsoft Qlib

**核心亮点:**

1. **因子表达式引擎**
   - 用户通过字符串表达式定义因子，如 `"Ref($close, -5) / $close - 1"`
   - 内置 Alpha158（158个因子）和 Alpha360（360个因子）预置因子库
   - 支持截面运算（Rank, Scale）、时序运算（Ref, Mean, Std）、算术运算
   - 表达式引擎自动处理数据对齐和 NaN 填充

2. **Point-in-Time (PIT) 数据库**
   - 每个数据点关联知识时间（knowledge_time），确保不泄露未来信息
   - 财务数据按公告日对齐，而非报告期日期
   - DataHandler 内置 PIT 支持

3. **模型 Zoo**
   - 集成 LightGBM、CatBoost、LSTM、Transformer、GRU 等
   - 统一的模型训练和预测接口
   - 支持滚动训练窗口

4. **RD-Agent**
   - LLM 驱动的自动化因子挖掘
   - 自动生成因子表达式并验证

**可借鉴方向:**
- factor-engine: 引入表达式引擎，降低因子编写成本 ~93%
- data-engine: 引入 PIT 数据对齐，杜绝未来数据泄露
- 因子库标准化: 预置 Alpha158/Alpha360 风格因子集

### 2.2 QUANTAXIS

**核心亮点:**

1. **Python + Rust 混合架构**
   - 核心计算模块用 Rust 实现（QARSBridge）
   - 回测速度提升 10-20x vs 纯 Python
   - 零拷贝数据传输（Apache Arrow 格式）

2. **QIFI 统一账户协议**
   - 标准化账户信息格式
   - 跨券商、跨市场统一接口

3. **微服务架构**
   - 数据服务、回测服务、交易服务独立部署
   - 分布式任务调度（QAScheduler）

4. **全栈覆盖**
   - 从数据采集到实盘交易的全流程覆盖
   - 支持 A股、期货、数字货币等多市场

**可借鉴方向:**
- backtest-engine: 考虑关键路径用 NumPy 向量化或 Cython 加速
- execution-monitor-engine: 借鉴 QIFI 统一协议设计
- 项目架构: 微服务化方向参考

### 2.3 Freqtrade / FreqAI

**核心亮点:**

1. **IFreqaiModel 接口（模型-策略解耦）**
   - 策略代码不直接依赖具体模型实现
   - 通过接口切换不同模型（LightGBM、XGBoost、PyTorch等）
   - 支持自定义模型适配器

2. **自适应重训管道**
   - 实盘中持续用最新数据重训模型
   - 重训在后台线程进行，不阻塞预测和交易
   - 可配置重训频率（train_period_days、live_retrain_hours）

3. **市场状态检测**
   - 支持多种市场状态检测方法
   - 不同状态可切换不同模型

4. **自动特征工程**
   - 自动数据归一化、异常值剔除
   - 特征重要性分析和特征选择

**可借鉴方向:**
- strategy-model-engine: 引入 IAdaptiveModel 接口，实现模型-策略解耦
- 新增自适应重训管道（后台线程模式）
- 新增模型注册中心（ModelRegistry）
- 新增市场状态检测模块

---

## 三、已完成的验证测试

### 测试1: 表达式驱动的因子引擎

**测试文件:** `tests/study_2026/test_factor_expression_engine.py`
**借鉴来源:** Microsoft Qlib Alpha158/Alpha360

**测试结果:**

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 基础表达式计算 | PASS | Ref, Mean, 四则运算均正确 |
| 截面运算 | PASS | Rank 值域 [0.020, 1.000], Scale 均值≈0 |
| 条件表达式 | PASS | If 匹配率 100% |
| 与手写因子对比 | PASS | MA20_DEV 最大差异 0.0 |
| Alpha158 批量计算 | PASS | 20个因子，0.025s，1.2ms/因子 |
| 缓存性能 | PASS | 二次计算速度提升显著 |
| 扩展性对比 | PASS | 减少代码量 ~93%（20行 vs 300行） |

**结论:** 表达式引擎方案可行，显著降低因子编写成本，提升代码可维护性。

### 测试2: Point-in-Time 数据管理

**测试文件:** `tests/study_2026/test_point_in_time_data.py`
**借鉴来源:** Microsoft Qlib PIT Database

**测试结果:**

| 测试项 | 结果 | 说明 |
|--------|------|------|
| PIT 正确性 | PASS | 2024-02-15 查询仅返回2022年报 |
| 年报公告后可见 | PASS | 2024-05-15 可获取2023年报 |
| 未来数据泄露影响量化 | PASS | 虚增收益 94.26%，Sharpe 偏差 0.78 |
| PIT 管道集成 | PASS | 管道流程设计合理 |

**结论:** 未来数据泄露对回测结果影响显著（本例中虚增收益 94%），PIT 对齐是回测准确性的必要条件。

### 测试3: 自适应模型重训管道

**测试文件:** `tests/study_2026/test_adaptive_model_retraining.py`
**借鉴来源:** Freqtrade FreqAI IFreqaiModel

**测试结果:**

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 自适应 vs 固定窗口 | PASS | 概念漂移后 MAE 改善 84.8% |
| 模型版本管理 | PASS | 8个模型版本，含元数据 |
| 市场状态检测 | PASS | 能区分趋势/震荡/高波动 |
| 后台训练 | PASS | 不阻塞预测 |

**结论:** 自适应重训在概念漂移场景下显著优于固定窗口训练，模型版本管理对生产环境至关重要。

---

## 四、待用户确认的优化建议

### 高优先级（建议立即实施）

| 编号 | 优化方向 | 目标模块 | 借鉴来源 | 预计工作量 |
|------|----------|----------|----------|-----------|
| H1 | 表达式驱动因子引擎 | factor-engine | Qlib | 3-5天 |
| H2 | PIT 数据对齐 | data-engine | Qlib | 2-3天 |
| H3 | 模型-策略解耦接口 | strategy-model-engine | FreqAI | 2-3天 |
| H4 | 自适应重训管道 | strategy-model-engine | FreqAI | 3-5天 |

### 中优先级（建议下一迭代实施）

| 编号 | 优化方向 | 目标模块 | 借鉴来源 | 预计工作量 |
|------|----------|----------|----------|-----------|
| M1 | 预置因子库（Alpha158风格） | factor-engine | Qlib | 2天 |
| M2 | 因子元数据标准化 | factor-engine | Qlib | 1天 |
| M3 | PIT 验证器 | data-engine | Qlib | 1-2天 |
| M4 | 模型注册中心 | strategy-model-engine | FreqAI | 1-2天 |
| M5 | 市场状态检测模块 | strategy-model-engine | FreqAI | 2天 |

### 低优先级（长期规划）

| 编号 | 优化方向 | 目标模块 | 借鉴来源 |
|------|----------|----------|----------|
| L1 | 因子表达式验证器 | factor-engine | Qlib |
| L2 | 财务数据修正系列 | data-engine | Qlib |
| L3 | 持续学习/增量训练 | strategy-model-engine | FreqAI |
| L4 | 分布式训练支持 | strategy-model-engine | QUANTAXIS |
| L5 | 核心路径向量化加速 | backtest-engine | QUANTAXIS |
| L6 | 微服务化架构 | 全局 | QUANTAXIS |

---

## 五、验证代码位置

```
tests/study_2026/
├── test_factor_expression_engine.py    # 优化方向1: 表达式因子引擎
├── test_point_in_time_data.py          # 优化方向2: PIT数据管理
├── test_adaptive_model_retraining.py   # 优化方向3: 自适应模型重训
└── LEARNING_REPORT.md                  # 本报告
```

---

## 六、下一步行动

1. 等待用户审阅上述优化建议，确认优先级
2. 用户确认后，在 `feature/quant-stream-inspired` 分支上实施
3. 实施完成后，在对应的 tests/study_2026/ 下生成集成测试
4. 遵循 Conventional Commits 规范提交代码

---

*报告生成时间: 2026-06-14*
*当前分支: feature/quant-stream-inspired*