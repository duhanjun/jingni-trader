# 量化交易开源项目学习报告

> **日期**: 2026-06-11 | **序号**: #1
> **项目**: jingni-trader
> **当前分支**: feature/quant-stream-inspired

---

## 一、学习项目清单及核心亮点

### 1.1 Microsoft Qlib (44K+ Stars)

- **仓库**: https://github.com/microsoft/qlib
- **核心亮点**:
  - **表达式引擎**: 基于 DSL 的因子定义（`$close`, `Ref($close, 5)`, `Mean($close, 20)`），开发者无需写复杂代码即可快速定义因子
  - **Alpha158/Alpha360**: 预定义因子集，覆盖动量、反转、波动率、成交量、技术指标等 158/360 个因子，标准化因子命名与分类
  - **Point-in-Time 数据库**: 防止前视偏差的时序数据库设计，按真实信息到达时间存储数据
  - **列式二进制存储**: 高效的二进制数据格式，支持快速列切片和时间切片
  - **模型 Zoo**: 内置 LightGBM、GRU、LSTM、Transformer、TRA 等模型，统一训练/预测接口
  - **YAML 工作流**: 通过 `qrun` 命令 + YAML 配置一键运行全流程
- **关联论文**: RD-Agent-Quant (NeurIPS 2025) — 多智能体框架实现因子-模型联合优化，使用 Co-STEER 代码生成器 + MAB 调度器，2x 年化收益 + 70% 更少因子

### 1.2 VectorBT (6.5K+ Stars)

- **仓库**: https://github.com/polakowo/vectorbt
- **核心亮点**:
  - **向量化回测**: 用 NumPy 矩阵运算替代逐 K 线循环，避免 Python 循环开销
  - **参数网格扫描**: 笛卡尔积参数扫描一次性计算，100-1000x 加速
  - **内置 57+ 绩效指标**: Sharpe、Sortino、Calmar、最大回撤、胜率等
  - **可视化**: 丰富的热力图、参数曲面图，直观展示参数空间
  - **多资产支持**: 支持投资组合层面的向量化回测

### 1.3 FactorEngine (arXiv:2603.16365, 2026年3月)

- **论文**: https://arxiv.org/abs/2603.16365
- **核心亮点**:
  - **程序级因子挖掘**: 将因子定义为图灵完备的代码程序，表达能力远超符号回归
  - **三分离架构**: 逻辑修正 vs 参数优化 / LLM引导搜索 vs 贝叶斯超参搜索 / LLM使用 vs 本地计算
  - **知识注入引导**: 从财报、研报等非结构化文本中提取可执行因子程序（多智能体闭环 pipeline）
  - **经验知识库**: 轨迹感知优化，包括从失败中学习，跨市场环境复用经验
  - **因子衰减分析**: 通过 IC 随滞后期变化估计因子半衰期，辅助因子轮换决策
- **效果**: IC/ICIR、Rank IC/ICIR 全面提升，AR/Sharpe 优于基线方法

---

## 二、可借鉴的方向列表

基于以上学习，对照 jingni-trader 现有架构，梳理以下优化方向：

| 优先级 | 优化方向 | 借鉴来源 | 目标模块 | 预期收益 |
|--------|---------|---------|---------|---------|
| **高** | 表达式因子定义 DSL | Qlib | factor-engine | 因子开发效率提升 10x+，因子库从 ~10 扩展到 158+ |
| **高** | 向量化回测模式 | VectorBT | backtest-engine | 参数优化加速 50-100x |
| **高** | 因子衰减分析 | FactorEngine | factor-engine | 淘汰失效因子，提升因子组合稳定性 |
| 中 | Point-in-Time 数据库 | Qlib | data-engine | 消除前视偏差，提升回测可信度 |
| 中 | 因子-模型联合优化 | RD-Agent | strategy-model-engine | 自动超参搜索，提升模型表现 |
| 中 | 57+ 绩效指标 | VectorBT | backtest-engine | 丰富回测报告维度 |
| 低 | 经验知识库 | FactorEngine | factor-engine | 跨市场环境因子经验复用 |
| 低 | LLM 辅助因子挖掘 | RD-Agent + FactorEngine | factor-engine | 自动化因子发现，降低人工依赖 |

---

## 三、已完成的验证测试及结论

### 3.1 向量化回测引擎测试

**测试文件**: `tests/study_2026/test_vectorized_backtest.py`

| 测试项 | 结果 | 详情 |
|--------|------|------|
| 正确性验证 | ✓ 通过 | 向量化与事件驱回测收益方向一致 |
| 单次回测性能 | 15.2x 加速 | 100次平均：事件驱动 1.04s vs 向量化 0.07s |
| 参数网格扫描 | **71.3x 加速** | 64组参数：事件驱动 0.69s vs 向量化 0.01s |
| 参数优化验证 | ✓ 通过 | 正确找到最优 MA 参数 (10, 20)，Sharpe 0.855 |
| 边界条件 | ✓ 通过 | 空数据、单日数据、全零信号、NaN 信号均正确处理 |

**结论**: 向量化回测在参数优化场景下有显著性能优势（71.3x），建议作为 backtest-engine 的可选加速模式，可在现有事件驱动引擎基础上增加 `VectorizedBacktest` 包装器。

### 3.2 表达式因子定义测试

**测试文件**: `tests/study_2026/test_expression_factor.py`

| 测试项 | 结果 | 详情 |
|--------|------|------|
| 表达式引擎基本功能 | ✓ 通过 | 支持 $close, Ref, Mean, Std, RSI, BB_upper/lower 等 17 个内置函数 |
| 因子库批量计算 | ✓ 通过 | 22 个因子全部计算成功，覆盖 7 大分类 |
| 截面因子计算 | ✓ 通过 | 支持 Rank 等截面操作 |
| 错误处理 | ✓ 通过 | 未知函数/字段抛出 ValueError，除零保护 |
| 性能对比 | ✓ 通过 | 表达式方式（折算6因子）比硬编码快约 2x（小规模）到 7x（中规模） |

**结论**: 表达式引擎在保持灵活性的同时性能优于硬编码方式。建议将 `ExpressionEngine` 集成到 factor-engine，同时将 `AlphaFactorLibrary` 的 22 个因子作为内置因子库。

### 3.3 因子衰减分析测试

**测试文件**: `tests/study_2026/test_factor_decay.py`

| 测试项 | 结果 | 详情 |
|--------|------|------|
| 因子衰减分析 | ✓ 通过 | 成功区分强/弱/噪声因子，强因子半衰期 >= 弱因子 |
| 因子分类 | ✓ 通过 | 4 类分类（stable/medium/fast_decay/invalid）正常工作 |
| 因子轮换策略 | ✓ 通过 | 基于半衰期的动态权重分配，淘汰弱因子 |
| 边界条件 | ✓ 通过 | 空数据、单日数据、全NaN因子均正确处理 |
| 报告生成 | ✓ 通过 | 生成结构化衰减分析报告 |

**结论**: 因子衰减分析能有效识别因子有效期，辅助因子筛选与权重调整。建议集成到 factor-engine 的 IC 分析模块中，作为因子质量评估的补充维度。

---

## 四、待用户确认的优化建议

### 建议1: 集成表达式引擎到 factor-engine（高优先级）

- **内容**: 将 `ExpressionEngine` 和 `AlphaFactorLibrary` 集成到 `skills/factor-engine/`
- **影响范围**: factor-engine（新增模块，不影响现有逻辑）
- **工作量**: 约 2-3 天
- **收益**: 因子库从 10 个扩展到 22+，支持用户自定义因子表达式，开发效率提升 10x+

### 建议2: 新增向量化回测模式（高优先级）

- **内容**: 在 `skills/backtest-engine/` 下新增 `vectorized_backtest.py`
- **影响范围**: backtest-engine（新增模块，与现有引擎并行）
- **工作量**: 约 3-4 天
- **收益**: 参数优化场景 50-100x 加速

### 建议3: 集成因子衰减分析（高优先级）

- **内容**: 将 `FactorDecayAnalyzer` 和 `FactorRotationStrategy` 集成到 factor-engine
- **影响范围**: factor-engine（扩展 IC 分析模块）
- **工作量**: 约 2 天
- **收益**: 量化因子有效期，自动淘汰失效因子

---

## 五、附录：验证代码文件清单

```
tests/study_2026/
├── test_vectorized_backtest.py    # 向量化回测引擎验证（450行）
├── test_expression_factor.py      # 表达式引擎因子定义验证（660行）
├── test_factor_decay.py           # 因子衰减分析验证（610行）
└── LEARNING_REPORT.md             # 本报告（追加写入）
```

---

> **Git 约束提醒**: 所有优化代码在用户明确确认之前，禁止执行 git commit/push/merge。当前验证代码位于 `tests/study_2026/` 独立测试目录，不影响主代码。