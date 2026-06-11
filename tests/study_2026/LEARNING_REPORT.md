# jingni-trader 量化领域学习报告

> 日期: 2026-06-11 | 序号: #1
> 学习来源: GitHub 开源项目 + 量化交易社区
> 状态: 已完成验证测试，待用户确认优化方向

---

## 一、学习项目清单及核心亮点

### 1.1 Microsoft Qlib (42,000+ Stars)
- **仓库**: https://github.com/microsoft/qlib
- **定位**: AI 驱动的量化投资平台，覆盖数据→因子→模型→回测全流程
- **核心亮点**:
  1. **表达式引擎 (Expression Engine)**: 使用 DSL 语法声明因子，如 `Ref($close, 20) / $close - 1`，因子作为函数而非数据定义，天然适配 LLM 自动生成
  2. **Alpha158 因子库**: 内置 158 个标准化 Alpha 因子，覆盖动量/反转/波动率/流动性/估值等维度
  3. **列式二进制存储**: 自研 `.bin` 格式，支持快速切片，内存效率远优于 pandas
  4. **多级缓存系统**: 表达式缓存 + 数据缓存，避免重复计算
  5. **Config-Driven Pipeline**: 通过 YAML 配置驱动整个 ML 流水线
  6. **RD-Agent 集成**: LLM Agent 自动挖掘 Alpha 因子 → 回测验证 → 迭代进化

### 1.2 NautilusTrader (21,500+ Stars | 评分 10/10)
- **仓库**: https://github.com/nautechsystems/nautilus_trader
- **定位**: 面向生产的高性能算法交易平台，Rust 核心 + Python 控制平面
- **核心亮点**:
  1. **事件驱动架构**: MessageBus 解耦所有组件，支持 Pub/Sub、点对点、请求/响应三种通信模式
  2. **确定性时间模型**: 回测和实盘使用完全一致的执行语义，消除代码重写
  3. **订单生命周期管理**: Created → Submitted → Accepted → PartiallyFilled → Filled → Cancelled
  4. **Crash-Only Design**: 崩溃后快速恢复，统一恢复路径
  5. **Rust 核心 + PyO3 绑定**: 性能关键路径用 Rust，策略开发用 Python
  6. **多交易所同时交易**: 原生支持多 venue 并发回测和实盘

### 1.3 QUANTAXIS (25,000+ Stars)
- **仓库**: https://github.com/yutiansut/QUANTAXIS
- **定位**: 全栈式量化金融分析框架，Python + Rust 混合架构
- **核心亮点**:
  1. **QARSBridge**: Rust 性能核心桥接，自动检测 Rust 库，不可用时回退 Python 实现
  2. **QIFI 协议**: 统一账户模型，跨语言二进制兼容
  3. **零拷贝数据桥接**: Python ↔ Rust 数据交换避免序列化开销
  4. **渐进式迁移**: 用户无需修改代码，自动选择最优实现

---

## 二、可借鉴的方向列表

### 方向 1: 向量化回测引擎 (借鉴 qlib + NautilusTrader)
| 维度 | 当前状态 | 优化目标 |
|------|---------|---------|
| 计算模式 | 逐日循环遍历 (O(n_stocks × n_days)) | 向量化矩阵运算 (O(1) 批量操作) |
| 性能 (全A 5000股×5年) | 预估 10-30 分钟 | 目标 30 秒 - 2 分钟 |
| 内存使用 | pandas DataFrame in-memory | 列式存储 + 分块计算 |
| 模式切换 | 无 | 快速模式 (因子筛选) / 精确模式 (最终验证) |

### 方向 2: 因子表达式引擎 (借鉴 qlib Expression Engine)
| 维度 | 当前状态 | 优化目标 |
|------|---------|---------|
| 因子定义 | 硬编码在 `compute_a_share_factors()` | DSL 表达式声明 |
| 新增因子 | 修改源码 | 配置文件 / API 调用 |
| LLM 友好性 | 不支持 | 直接生成 DSL 表达式 |
| 因子库规模 | ~15 个 | Alpha158 风格 (100+) |
| 缓存机制 | 无 | 表达式结果缓存 |

### 方向 3: 事件驱动回测架构 (借鉴 NautilusTrader)
| 维度 | 当前状态 | 优化目标 |
|------|---------|---------|
| 架构模式 | 过程式 (单一函数) | 事件驱动 (MessageBus) |
| 组件耦合 | 高 (回测/策略/执行耦合) | 低 (Actor 模式独立通信) |
| 订单生命周期 | 无 | Created→Filled/Rejected |
| 策略扩展 | 修改回测代码 | 实现 Strategy 子类 |
| 多策略并行 | 不支持 | 天然支持 |
| 实盘迁移 | 代码分离 | 统一事件模型 |
| 风控热插拔 | 不支持 | MessageBus 订阅 |

---

## 三、已完成的验证测试及结论

### 测试 1: 向量化回测引擎性能对比
- **文件**: `tests/study_2026/test_vectorized_backtest.py`
- **测试数据**: 50 只股票 × 126 天 (小规模) / 500 只股票 × 252 天 (大规模)
- **结果**:

| 方法 | 小规模耗时 | 加速比 | 大规模耗时 | 精度 |
|------|-----------|--------|-----------|------|
| 逐日循环 (当前) | 0.35s | 1.0x (基准) | 估算 ~2.3s | 基准 |
| 纯向量化 | 0.04s | **9.9x** | 0.11s | 中 (牺牲部分交易成本细节) |
| 分块向量化 (推荐) | 0.19s | 1.8x | 预估 ~0.5s | 高 (保留交易成本) |

- **结论**: 纯向量化在大规模场景下优势巨大 (预估 5000 股可 50x 加速)。分块向量化是平衡精度与速度的最佳方案。

### 测试 2: 因子表达式引擎原型
- **文件**: `tests/study_2026/test_expression_engine.py`
- **实现**: 递归下降解析器，支持 7 种滚动窗口函数 (Ref/Mean/Std/Sum/Max/Min/PctChange)
- **结果**:

| 测试项 | 结果 |
|--------|------|
| 与硬编码一致性 (volume_ratio) | corr=1.000, max_diff=0.000 |
| 动态添加新因子 | 7/7 成功，无需修改源码 |
| LLM 生成表达式成功率 | **7/7 (100%)** |
| 表达式引擎性能 | ~4x 慢于硬编码 (可接受，探索阶段灵活性优先) |

- **结论**: DSL 表达式引擎完全可行，100% LLM 友好。生产环境可将表达式编译为优化代码。

### 测试 3: 事件驱动回测架构
- **文件**: `tests/study_2026/test_event_driven_backtest.py`
- **实现**: MessageBus + Order 生命周期 + TopKDropoutStrategy + 热插拔风控
- **结果**:

| 指标 | 值 |
|------|-----|
| 126 天事件总数 | 1,320 events |
| 订单成交 | 534 filled |
| 总收益率 | -14.51% |
| 夏普比率 | -1.10 |
| 风控拦截 | 55/143 orders (38%) |
| 执行耗时 | 0.38s |

- **结论**: 事件驱动架构正确实现了订单生命周期和热插拔组件。与 NautilusTrader 设计理念对齐。

### 架构对比总结

```
                   当前 jingni-trader           优化后 (借鉴方向)
─────────────────────────────────────────────────────────────────
回测引擎           过程式逐日循环               双模式: 向量化快速 + 事件驱动精确
因子引擎           硬编码 15 个因子             DSL 表达式 + 预编译 + Alpha158 库
策略 API           无统一接口                   Strategy Actor 基类
风险管理           独立 RiskManager             事件驱动热插拔风控
数据存储           Parquet 文件                 列式二进制 + 多级缓存
代码架构           Pipeline 模式                事件驱动 MessageBus
```

---

## 四、待用户确认的优化建议

### 优先级高 (建议优先实施)
1. **向量化回测双模式** (`backtest-engine`)
   - 新增 `VectorizedAdapter` 快速模式 + 优化 `NativeAdapter` 精确模式
   - 预期收益: 全A回测从 10 分钟降至 30 秒
   - 风险: 低 (不修改现有逻辑，增量添加)

2. **因子表达式引擎** (`factor-engine`)
   - 引入 `FactorExpressionEngine` 类 + 预定义因子库配置文件
   - 预期收益: 因子扩展效率 10x，LLM Agent 可直接生成
   - 风险: 低 (现有硬编码方式保留)

### 优先级中 (建议逐步推进)
3. **事件驱动层** (`backtest-engine`)
   - 新增 `EventDrivenBacktestEngine` 作为可选后端
   - 引入 `MessageBus` + `Order` 领域模型
   - 预期收益: 回测保真度提升，实盘迁移成本降低
   - 风险: 中 (架构改动较大，建议先在新分支实验)

4. **风控组件热插拔** (`portfolio-risk-engine`)
   - 基于 MessageBus 订阅模式重构风控
   - 预期收益: 风控策略可独立测试和组合
   - 风险: 低 (依赖方向 3)

### 优先级低 (长期规划)
5. **Rust 性能核心** (`所有模块`)
   - 借鉴 QUANTAXIS 的 QARSBridge 模式
   - 在性能瓶颈处引入 Rust 实现，Python 自动回退
   - 风险: 高 (需建立 Rust 工具链和 CI)

6. **列式数据存储** (`data-engine`)
   - 借鉴 qlib `.bin` 格式，支持 memory-mapping
   - 风险: 中 (格式变更需数据迁移)

---

## 五、备注

- 所有验证代码位于 `tests/study_2026/` 目录
- 验证代码均为独立文件，未修改项目主代码
- 根据约束，尚未执行任何 git 操作
- 待用户确认后，可选择优化的方向创建对应的 feature 分支实施

### 测试文件列表
```
tests/study_2026/
├── test_vectorized_backtest.py     # 向量化回测性能对比
├── test_expression_engine.py       # 因子表达式引擎原型
├── test_event_driven_backtest.py   # 事件驱动回测架构原型
└── LEARNING_REPORT.md              # 本报告
```