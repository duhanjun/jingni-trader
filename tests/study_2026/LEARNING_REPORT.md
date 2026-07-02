# 量化交易开源项目学习报告

---

## 报告元信息

| 字段 | 值 |
|------|-----|
| 报告序号 | #001 |
| 学习日期 | 2026-06-13 |
| 分支 | feature/quant-stream-inspired |
| 测试目录 | tests/study_2026/ |
| 状态 | 验证完成，待用户确认 |

---

## 一、学习项目清单及核心亮点

### 1.1 Microsoft Qlib (17.5K+ Stars)
- **仓库**: https://github.com/microsoft/qlib
- **语言**: Python
- **核心亮点**:
  - **表达式引擎 (Expression Engine)**: 支持声明式因子定义语法，如 `Ref($close, 60) / $close`，将因子从"硬编码逻辑"提升为"可配置表达式"
  - **DataHandlerLP 架构**: "配置即代码"设计，数据处理管道通过 YAML/JSON 配置驱动，无需修改源码即可调整数据流
  - **Alpha158/Alpha360 因子库**: 按动量/反转/波动率/流动性等分组组织 158 种因子，分组管理理念清晰
  - **嵌套决策框架 (NDF)**: 多层策略嵌套架构，策略可组合、可复用
  - **缓存机制**: 自动缓存中间计算结果，避免重复计算

### 1.2 FinRL-X (AI4Finance Foundation)
- **仓库**: https://github.com/AI4Finance-Foundation/FinRL-Trading
- **论文**: FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading (arXiv:2603.21330)
- **核心亮点**:
  - **Weight-Centric 架构**: 目标投资组合权重向量 `w_t` 作为策略层与下游模块之间的唯一接口，统一合约
  - **SATR 管道**: `S(Selection) → A(Allocation) → T(Timing) → R(Risk Overlay)` 四个可组合的保合同变换
  - **合同保持变换 (Contract-Preserving Transform)**: 每个阶段的输入输出都是权重向量，保证模块可替换
  - **部署一致性 (Deployment-Consistent)**: 同一代码可在回测/模拟盘/实盘环境中运行，只需切换执行代理
  - **RL/ML 友好**: 权重向量格式天然适配强化学习 Agent 输出

### 1.3 VectorBT (4K+ Stars)
- **仓库**: https://github.com/polakowo/vectorbt
- **语言**: Python / Numba
- **核心亮点**:
  - **矩阵化计算范式**: 将数千种策略配置打包到多维 NumPy 数组，单次操作同时评估所有参数组合
  - **Numba 加速**: 核心计算路径使用 Numba JIT 编译，接近 C 级别性能
  - **参数扫描**: `run_combs()` 方法能一次性测试所有参数组合，加速比可达 10-100 倍
  - **Purged Walk-Forward CV**: 支持带 purge + embargo 的交叉验证，防止过拟合
  - **多维广播**: NumPy 广播机制实现多维参数空间的穷举搜索

---

## 二、jingni-trader 现状分析与可借鉴方向

### 2.1 模块评估矩阵

| 模块 | 当前设计 | 存在问题 | 借鉴方向 |
|------|----------|----------|----------|
| factor-engine | 硬编码 `compute_a_share_factors()` | 新增因子需改核心代码，无分组管理，无依赖管理 | Qlib 表达式引擎 + 注册表模式 |
| strategy-model-engine | ML 模型训练 → 离散信号 {-1,0,1} | 信号格式不统一，模块间耦合 | FinRL-X weight-centric 接口 |
| backtest-engine | 适配器模式接入 RQAlpha/Backtrader | 参数扫描效率低，需多次独立回测 | VectorBT 矩阵化批量回测 |
| portfolio-risk-engine | PyPortfolioOpt + 独立风险检查 | 风控与策略分离，缺乏组合风险覆盖 | FinRL-X RiskOverlay 抽象 |
| data-engine | 多源 Fallback + 合成数据 | 设计良好，暂时无需改动 | — |
| execution-monitor-engine | 模拟交易 + 熔断 | 功能基础 | NautilusTrader 事件驱动架构 |
| reports-engine | 绩效报告 + 归因 | 功能完整 | 可增加 Walk-Forward CV 报告 |

### 2.2 已完成的验证测试

以下三个优化方向已完成代码编写、测试和性能对比：

#### 优化方向 1: 因子注册与自动发现机制（借鉴 Qlib）

- **测试文件**: `tests/study_2026/test_factor_registry.py`
- **测试结果**: 9/9 通过
- **关键发现**:
  - 注册表模式性能与硬编码几乎无差异（0.90x，略快）
  - 拓扑排序正确处理依赖关系
  - 循环依赖检测有效
  - 按分组批量计算因子可行

#### 优化方向 2: 可组合策略管道（借鉴 FinRL-X）

- **测试文件**: `tests/study_2026/test_composable_strategy.py`
- **测试结果**: 8/8 通过
- **关键发现**:
  - SATR 管道的保合同变换可行，所有阶段输出统一权重向量
  - 策略组件可跨管道复用
  - 风险覆盖层（个股权重上限 + 行业分散化）正确约束最终权重
  - 边界条件（空选股、缺数据）处理正确

#### 优化方向 3: 向量化快速回测加速器（借鉴 VectorBT）

- **测试文件**: `tests/study_2026/test_vectorized_backtest.py`
- **测试结果**: 8/8 通过
- **性能数据**:
  - 向量化 vs 逐股循环: 1.17x 加速（20 只 x 1000 天）
  - 批量参数回测: 12 个组合 / 0.99 秒（平均 0.0825 秒/组合）
  - Purged Walk-Forward CV: 5-fold 正确工作
  - A 股整手约束正确（100 股倍数）

---

## 三、待用户确认的优化建议

### 建议 1: 引入 FactorRegistry 重构 factor-engine（优先级：高）

借鉴 Qlib 的注册表模式，将当前 `factor-engine/engine.py` 中的硬编码因子迁移到注册表管理。

**预计影响**:
| 文件 | 变更 |
|------|------|
| `skills/factor-engine/engine.py` | 重构：用 FactorRegistry 替代 `compute_a_share_factors()` |
| `skills/factor-engine/registry.py` | 新增：FactorRegistry 实现 |
| `skills/factor-engine/factors/` | 新增：各分组因子定义文件 |

**风险**: 中等（需要回归测试确保现有因子计算结果一致）
**收益**: 因子扩展性显著提升，新增因子只需添加装饰器

### 建议 2: 引入 ComposableStrategy 统一策略接口（优先级：中）

借鉴 FinRL-X 的 weight-centric 设计，统一因子 → 信号 → 权重的完整数据流。

**预计影响**:
| 文件 | 变更 |
|------|------|
| `skills/strategy-model-engine/engine.py` | 重构：输出从 `signal` 改为 `weight vector` |
| `skills/strategy-model-engine/pipeline.py` | 新增：SATR 管道实现 |
| `skills/backtest-engine/engine.py` | 修改：直接接收权重向量 |
| `skills/portfolio-risk-engine/` | 整合：作为 RiskOverlay 嵌入管道 |

**风险**: 较高（接口变更影响多个模块）
**收益**: 模块解耦、策略可组合、RL 友好

### 建议 3: VectorizedBacktester 作为回测加速层（优先级：低）

在当前事件驱动回测之上增加向量化加速层，用于参数扫描阶段。

**预计影响**:
| 文件 | 变更 |
|------|------|
| `skills/backtest-engine/vectorized.py` | 新增：VectorizedBacktester 实现 |
| `skills/backtest-engine/engine.py` | 修改：增加 `batch_backtest()` 入口 |

**风险**: 低（作为独立模块，不影响现有流程）
**收益**: 参数优化效率 10x+ 提升

### 建议 4: 引入 Purged Walk-Forward CV（优先级：中）

借鉴 VectorBT + Lopez de Prado，在回测报告中增加 Purged Walk-Forward 交叉验证章节。

---

## 四、测试结果汇总

```
测试套件                       测试数  通过  失败  耗时
─────────────────────────────────────────────────────────
test_factor_registry.py           9      9     0    0.334s
test_composable_strategy.py       8      8     0    0.029s
test_vectorized_backtest.py       8      8     0    1.779s
─────────────────────────────────────────────────────────
合计                             25     25     0    2.142s
```

### 性能基准数据

| 对比项 | 旧方式 | 新方式 | 结论 |
|--------|--------|--------|------|
| 因子计算 (50股×500天) | 0.0855s (硬编码) | 0.0768s (注册表) | 0.90x，无性能损失 |
| 多股回测 (20股×1000天) | 0.0206s (逐股) | 0.0176s (向量化) | 1.17x 加速 |
| 批量参数扫描 (12组合) | ~12s (独立运行) | 0.99s (一次运行) | ~12x 加速 |

---

## 五、其他未验证但有价值的参考项目

| 项目 | Stars | 亮点 | 适用场景 |
|------|-------|------|----------|
| NautilusTrader | 4K+ | Rust核心 + Python绑定、事件驱动架构、研究到生产一致性 | 实盘部署、生产级交易系统 |
| Zipline-Reloaded | 3K+ | 经典Pipeline API、DataBundle | 回测框架参考 |
| btgym | 1K+ | OpenAI Gym 回测环境 | RL策略开发 |
| empyrical | 1K+ | 专业金融绩效指标库 | 丰富风险评估指标 |

---

## 六、下一步行动

1. **请用户审查上述 4 个优化建议**，确认优先级和实施范围
2. 确认后可开始将 FactorRegistry 或 ComposableStrategy 集成到主代码
3. 所有 git 操作（commit/push/merge）将等待用户明确确认后执行

---

*报告生成于: 2026-06-13 | 分支: feature/quant-stream-inspired | 作者: AI Agent*