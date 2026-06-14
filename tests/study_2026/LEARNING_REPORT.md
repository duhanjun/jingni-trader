# 量化交易开源项目学习报告

## 报告信息

| 字段 | 内容 |
|------|------|
| 日期 | 2026-06-14 |
| 序号 | #1 (首次学习报告) |
| 研究人 | jingni-trader AI Agent |
| 当前分支 | feature/quant-stream-inspired |

---

## 一、学习项目清单及核心亮点

### 1.1 Microsoft Qlib (⭐ 42K+)

- **仓库**: https://github.com/microsoft/qlib
- **语言**: Python
- **许可证**: MIT

**核心亮点**:
1. **Point-in-Time (PIT) 数据系统**: 严格防止前视偏差。每次查询数据时，系统自动确保只返回当前时间点之前已知的数据。
2. **表达式引擎 (Expression Engine)**: 用 DSL 表达式定义因子计算，如 `Ref($close, -5) / $close - 1`，自动处理 PIT 约束。
3. **模型动物园 (Model Zoo)**: 内置 20+ SOTA 量化模型（LightGBM、GRU、GATs、TabNet、Transformer、Localformer 等），统一接口 `model.fit()` / `model.predict()`。
4. **RD-Agent**: 基于 LLM 的自动因子挖掘框架，自动发现和验证新因子。
5. **分层架构**: `Data Layer → Model Layer → Strategy Layer → Execution Layer`，各层独立可替换。

**对 jingni-trader 的启发**:
- PIT 数据安全检查器可集成到 `data-engine` 和 `strategy-model-engine` 中
- 表达式引擎可提升 `factor-engine` 的因子定义灵活性
- 考虑引入 Qlib 模型作为 `strategy-model-engine` 的备选模型

### 1.2 Riskfolio-Lib (⭐ 3.4K)

- **仓库**: https://github.com/dcajasn/Riskfolio-Lib
- **语言**: Python
- **许可证**: BSD-3-Clause

**核心亮点**:
1. **24 种凸风险度量**: 分散性风险、下行风险、回撤风险三大类，支持 VaR/CVaR/EVaR/RLVaR/DaR/CDaR/EDaR/RLDaR 等。
2. **分层优化方法**: HRP (Hierarchical Risk Parity)、HERC (Hierarchical Equal Risk Contribution)、NCO (Nested Clustered Optimization)。
3. **松弛风险平价 (Relaxed Risk Parity)**: 允许正则化参数，使风险贡献更平衡。
4. **多目标优化**: 支持均值-风险、风险-回报、风险-风险多目标。

**对 jingni-trader 的启发**:
- **直接增强 `portfolio-risk-engine`**: 当前仅支持基础 HRP 和均值-方差，可引入 HERC/NCO 和扩展风险度量
- 风险归因可增加下行风险/回撤风险维度

### 1.3 AKQuant (⭐ 1.3K)

- **仓库**: https://github.com/akfamily/akquant
- **语言**: Rust + Python
- **许可证**: Apache-2.0

**核心亮点**:
1. **Rust+Python 混合架构**: 核心计算用 Rust 实现，Python 提供 API 层，兼顾性能与易用性。
2. **Walk-forward Validation**: 完整的滚动训练验证框架，模拟真实交易中的定期重训练场景。
3. **Signal vs. Action 分离**: 模型产生信号，策略层将信号转为交易动作，两者解耦便于独立测试。
4. **LLM 辅助策略生成**: 支持自然语言描述策略逻辑。
5. **多时间框架 Feed API**: 同时处理不同频率的数据（日线/分钟线）。

**对 jingni-trader 的启发**:
- Walk-forward 验证框架可集成到 `strategy-model-engine`
- Signal-Action 分离模式可提升策略模块的可测试性
- 长远看，Rust 重写核心计算模块可提升性能

### 1.4 其他值得关注的项目

| 项目 | Stars | 核心价值 |
|------|-------|----------|
| cvxportfolio | 2K+ | 校园级组合优化，文档极佳，多周期再平衡 |
| Zipline-Reloaded | 1.5K+ | 回测引擎标杆，事件驱动架构 |
| vnpy | 25K+ | 实际交易接口丰富，CTP/XTP 等 |
| QUANTAXIS | 8K+ | 全栈量化框架，微服务架构 |
| TradeMaster | 2K+ | 强化学习交易，13+ RL 算法 |
| FinRobot | 2K+ | LLM Agent 驱动的金融分析 |
| FinGPT | 14K+ | 金融大模型，情感分析 |
| TradingAgents | 4K+ | 多 Agent 交易系统 |

---

## 二、可借鉴的优化方向

### 方向 1: 分层组合优化增强 (优先级: ⭐⭐⭐⭐⭐)

**借鉴来源**: Riskfolio-Lib
**对照模块**: `portfolio-risk-engine`

**现状**: 当前 `portfolio-risk-engine` 仅支持均值-方差、最大夏普、最小方差、简化版 HRP 和 CVaR（等权兜底）。HRP 实现不完整，缺少 HERC/NCO。

**优化建议**:
1. 完善 HRP 实现（基于协方差矩阵聚类，递归二分权重分配）
2. 新增 HERC（分层等风险贡献）方法
3. 新增 NCO（嵌套聚类优化）方法
4. 通过配置切换不同优化方法，提供统一接口

**验证状态**: ✅ 已完成验证测试，详见 [test_hierarchical_portfolio.py](test_hierarchical_portfolio.py)

### 方向 2: 扩展风险度量 (优先级: ⭐⭐⭐⭐)

**借鉴来源**: Riskfolio-Lib
**对照模块**: `portfolio-risk-engine`

**现状**: 仅实现 VaR（历史模拟法）和 CVaR，缺少 EVaR、回撤风险度量。

**优化建议**:
1. 新增 EVaR（熵风险价值）计算
2. 新增回撤风险族：DaR (在险回撤)、CDaR (条件在险回撤)、EDaR (熵回撤风险)
3. 新增 Ulcer Index、Sortino Ratio、Calmar Ratio
4. 支持基于下行风险的风险平价优化

**验证状态**: ✅ 已完成验证测试，详见 [test_extended_risk_measures.py](test_extended_risk_measures.py)

### 方向 3: Walk-forward Validation 框架 (优先级: ⭐⭐⭐⭐)

**借鉴来源**: AKQuant + MS Qlib
**对照模块**: `strategy-model-engine`

**现状**: 已有 `purged_group_ts_split`，但缺少完整的滚动训练验证框架。

**优化建议**:
1. 实现 `WalkForwardValidator` 类，支持滚动窗口生成、重训练、评估
2. 实现 `PointInTimeChecker` 数据安全检查器
3. 引入 Signal-Action 分离设计模式
4. 与现有 `purged_group_ts_split` 互补，提供不同粒度的验证

**验证状态**: ✅ 已完成验证测试，详见 [test_walkforward_validation.py](test_walkforward_validation.py)

### 方向 4: 因子表达式引擎 (优先级: ⭐⭐⭐)

**借鉴来源**: MS Qlib
**对照模块**: `factor-engine`

**现状**: 因子计算依赖 pandas_ta/talib 计算器，因子定义较固定。

**优化建议**:
1. 引入 DSL 表达式引擎，支持如 `Ref($close, -5) / Ref($close, -20) - 1` 的因子定义
2. 自动处理 PIT 约束
3. 支持用户自定义因子组合

**验证状态**: ⏳ 待验证（需更多设计讨论）

### 方向 5: 核心计算模块性能优化 (优先级: ⭐⭐)

**借鉴来源**: AKQuant
**对照模块**: 全局

**现状**: 纯 Python 实现，大数据量下性能瓶颈。

**优化建议**:
1. 使用 Cython/Numba 加速关键计算路径
2. 长远考虑 Rust 重写核心模块（回测引擎、因子计算）
3. 引入并行计算（多进程/多线程因子计算）

**验证状态**: ⏳ 待验证（需性能基线和基准测试）

---

## 三、已完成验证测试及结论

### 测试 1: 分层组合优化 (HRP/HERC/NCO)

**测试文件**: `tests/study_2026/test_hierarchical_portfolio.py`
**测试结果**: ✅ 全部通过

**测试内容**:
- HRP 实现验证：10 资产，权重和=1.0，夏普比率 1.02 vs 等权 0.25
- HERC 实现验证：12 资产，两种聚类内权重方法对比
- NCO 实现验证：15 资产，嵌套聚类优化
- 四种方法综合对比：等权/HRP/HERC/NCO
- 边界条件：单资产、双资产、高共线性、聚类数>资产数

**关键结论**:
- HRP 在聚类结构数据上显著优于等权（夏普 1.02 vs 0.25）
- HERC 两种类内权重方法表现接近，逆方差略优
- 单资产边界条件需特殊处理（scipy linkage 限制）

### 测试 2: 扩展风险度量

**测试文件**: `tests/study_2026/test_extended_risk_measures.py`
**测试结果**: ✅ 全部通过

**测试内容**:
- EVaR 计算：95%/99% 置信度验证
- 回撤风险族：DaR/CDaR/EDaR/Ulcer Index 计算与单调性验证
- 下行风险：半标准差、Sortino、Calmar 比率
- 组合风险画像：等权 vs 集中组合的风险对比
- 下行风险平价优化：基于半标准差的权重优化
- 边界条件：空数组、单元素、常量、极端回撤

**关键结论**:
- 等风险度量单调性成立：99% EVaR > 95% EVaR
- 回撤风险度量单调性成立：CDaR >= DaR, EDaR >= DaR
- 下行风险平价确实降低了组合的下半标准差

### 测试 3: Walk-forward Validation 框架

**测试文件**: `tests/study_2026/test_walkforward_validation.py`
**测试结果**: ✅ 全部通过

**测试内容**:
- PIT 安全检查器：前视偏差检测、训练/测试边界验证
- Walk-forward 窗口生成：滚动训练窗口自动划分
- 完整的训练-预测-评估管道
- Signal-Action 分离模式：分位数策略、做多策略、换手率计算
- 静态训练 vs 滚动训练对比

**关键结论**:
- PIT 检查器能正确检测训练/测试边界违规
- Walk-forward 框架生成合理的滚动窗口
- 滚动训练提供比静态训练更真实的样本外评估
- Signal-Action 分离模式使策略逻辑可独立测试

---

## 四、待用户确认的优化建议

### 建议 1: 将 HRP/HERC/NCO 集成到 portfolio-risk-engine ⭐⭐⭐⭐⭐

- **影响模块**: `skills/portfolio-risk-engine/engine.py`
- **改动量**: 中等（新增 3 个方法类，约 200 行代码）
- **风险**: 低（纯新增功能，不影响现有逻辑）
- **建议分支**: `feature/riskfolio-inspired`

### 建议 2: 将扩展风险度量集成到 portfolio-risk-engine ⭐⭐⭐⭐

- **影响模块**: `skills/portfolio-risk-engine/engine.py`
- **改动量**: 中等（新增风险度量函数，约 150 行代码）
- **风险**: 低（纯新增功能）
- **建议分支**: 可与建议 1 合并

### 建议 3: 将 Walk-forward 验证集成到 strategy-model-engine ⭐⭐⭐⭐

- **影响模块**: `skills/strategy-model-engine/engine.py`
- **改动量**: 较大（新增验证框架类，约 250 行代码）
- **风险**: 中等（需与现有 purged_ts_split 配合）
- **建议分支**: `feature/walkforward-validation`

### 建议 4: 引入因子表达式引擎 ⭐⭐⭐

- **影响模块**: `skills/factor-engine/engine.py`
- **改动量**: 大（需设计 DSL 和表达式解析器）
- **风险**: 中等（需重构因子计算管道）
- **建议分支**: 待设计讨论后确定

---

## 五、测试文件索引

| 文件 | 优化方向 | 借鉴来源 | 状态 |
|------|----------|----------|------|
| `test_hierarchical_portfolio.py` | 分层组合优化 | Riskfolio-Lib | ✅ 通过 |
| `test_extended_risk_measures.py` | 扩展风险度量 | Riskfolio-Lib | ✅ 通过 |
| `test_walkforward_validation.py` | Walk-forward 验证 | AKQuant + Qlib | ✅ 通过 |

---

## 六、下一步计划

1. 等待用户审阅本报告，确认优先实施的优化方向
2. 用户确认后，在独立 feature 分支上实施代码集成
3. 集成后运行完整项目测试套件
4. 撰写集成后的验证报告

---

> **重要提示**: 根据约束要求，所有优化代码已放置在独立测试文件中，未执行任何 git commit/push/merge 操作。用户确认后方可进行代码合并。