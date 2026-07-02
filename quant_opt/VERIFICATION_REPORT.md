# jingni-trader 量化优化验证报告

**执行日期**: 2026-06-20
**分支**: `feat/quant-opt-20260620`
**执行人**: 自动化学习与验证流程

---

## 一、学习项目清单及核心亮点

本次联网调研覆盖 GitHub、arXiv、Papers with Code、QuantConnect 等平台，重点考察
2025-2026 年活跃的量化交易开源项目。筛选出以下 4 个最具借鉴价值的项目：

### 1. Microsoft Qlib（GitHub 42k+ stars）
- **定位**: AI 驱动的量化投研平台，LLM 时代量化基础设施
- **核心亮点**:
  - **表达式引擎**: 用公式字符串定义因子（如 `Ref($close, 20) / $close - 1`），
    支持算子嵌套，因子库可无限扩展
  - **Alpha158/Alpha360 预置因子库**: 158/360 个开箱即用的标准化因子
  - **高性能数据层**: 专用 `.bin` 二进制格式，列式存储，支持表达式缓存
  - **分层架构**: Data → Model → Strategy → Backtest → Workflow，组件松耦合
  - **实验管理**: QlibRecorder + MLflow，支持滚动训练、在线 serving
  - **RD-Agent 集成**: LLM 自动挖掘 Alpha 因子并验证
- **借鉴价值**: ⭐⭐⭐⭐⭐（表达式引擎、因子库设计直接可借鉴）

### 2. vectorbt（GitHub 7k+ stars）
- **定位**: 向量化回测框架，性能比事件驱动快 100-1000 倍
- **核心亮点**:
  - **向量化范式**: 用 NumPy/Numba JIT 一次性计算所有指标、信号、组合，
    替代逐 bar 循环
  - **多维数组**: 单次回测可测试数千组参数组合（参数扫描）
  - **Pandas 原生 API**: `df.vbt.signals` 等访问器，数据科学家友好
  - **Walk-forward 优化**: 内置滚动窗口优化
- **借鉴价值**: ⭐⭐⭐⭐⭐（向量化回测思想直接解决 jingni-trader 性能瓶颈）

### 3. NautilusTrader（GitHub 24k+ stars）
- **定位**: 生产级事件驱动交易引擎，Rust 核心 + Python 控制面
- **核心亮点**:
  - **研究-实盘一致性**: 同一 NautilusKernel 在回测和实盘运行，策略代码零修改
  - **确定性设计**: 单线程事件循环，纳秒级时间戳，可复现
  - **消息总线架构**: MessageBus pub/sub + Cache + Portfolio + RiskEngine
  - **替换边界**: 回测用 SimulatedExchange，实盘用真实 Venue Adapter，
    边界以下代码完全相同
- **借鉴价值**: ⭐⭐⭐⭐（研究-实盘一致性理念，架构设计参考）

### 4. AlphaGen / Alpha² / FactorMiner（学术前沿）
- **定位**: 基于 RL/LLM/MCTS 的公式化 Alpha 因子自动挖掘
- **核心亮点**:
  - **AlphaGen (KDD 2023)**: RL 生成因子集合，优化 IC 与多样性
  - **Alpha²**: DRL + 维度分析，确保因子逻辑合理性
  - **FactorMiner (ICLR 2026)**: 自进化 Agent + 经验记忆，Ralph Loop 范式
  - **AlphaForge**: 生成-预测网络 + 动态权重组合
  - **共同特征**: RPN 表示公式，相关性去冗余，IC 筛选
- **借鉴价值**: ⭐⭐⭐⭐（因子自动挖掘方向，未来增强）

---

## 二、可借鉴的方向列表

对照 jingni-trader 现有代码结构，识别出以下改进空间：

| 模块 | 现有问题 | 借鉴来源 | 优化方向 | 优先级 |
|------|---------|---------|---------|--------|
| backtest-engine | `native_adapter.py` 逐日 Python for 循环，`data[data['date']==dt]` 循环内重复过滤，O(n×m) 复杂度 | vectorbt | 向量化回测，预透视矩阵，目标权重调仓 | 高 |
| factor-engine | 因子硬编码在 `compute_a_share_factors`，新增因子需改源码 | Qlib 表达式引擎 | 公式字符串定义因子，AST 解析，算子注册表 | 高 |
| factor-engine | `_calc_ic` 逐日循环调用 `scipy.spearmanr`，每因子每周期重复 | Qlib/vectorbt | `groupby('date').corr()` 向量化 IC | 中 |
| backtest-engine | 回测与执行适配器逻辑分离，无一致性保证 | NautilusTrader | 研究-实盘共享核心引擎，替换边界设计 | 中 |
| data-engine | 无表达式缓存，重复计算 | Qlib 数据层 | 因子计算结果缓存，`.bin` 格式 | 低 |
| factor-engine | 无因子自动挖掘能力 | AlphaGen/FactorMiner | RL/LLM 因子生成（未来方向） | 低 |

---

## 三、已完成的验证测试及结论

本次在 `feat/quant-opt-20260620` 分支实现了 3 个优化模块，共 22 个测试全部通过。

### 优化点 1: 表达式因子引擎（借鉴 Qlib）

**文件**: [quant_opt/expression_factor_engine.py](file:///workspace/quant_opt/expression_factor_engine.py)

**实现内容**:
- 基于 Python `ast` 模块的表达式解析器，支持公式字符串定义因子
- 算子注册表：时序算子（Ref/Delta/Mean/Sum/Std/Max/Min/WMA/EMA）、
  双序列时序算子（Corr/Cov）、横截面算子（CSRank/CSZScore）、算术运算
- 预置 Alpha158 子集因子库（18 个因子）
- 表达式缓存机制，避免重复计算
- 严格无前视偏差：时序算子按 code 分组 rolling，仅用历史数据

**测试结果**（10/10 通过）:
```
✓ 字段引用正确（$close 与原始列一致，误差 0）
✓ Ref 算子无前视偏差（首 5 日为 NaN）
✓ 算术运算正确（Ref($close,5)/$close-1 误差 0）
✓ 滚动算子 Mean/Std/Max/Min 全部正确（误差 < 1e-8）
✓ 横截面算子 CSRank/CSZScore 正确（误差 < 1e-8）
✓ 嵌套表达式计算成功（非空率 80%）
✓ Alpha158 子集 18 个因子全部有有效值
✓ 缓存机制有效（首次 3.6ms，缓存 0.001ms）
✓ 无效表达式正确抛出异常
✓ 性能：表达式引擎 18 因子 358ms，单因子 19.9ms（硬编码 11.6ms）
```

**结论**: 表达式引擎在提供 3.6 倍可扩展性（18 因子 vs 硬编码 5 因子）的同时，
单因子性能仅慢 1.7 倍，且支持缓存。**新增因子无需修改源码，只需写公式字符串**，
可扩展性显著提升。

### 优化点 2: 向量化回测引擎（借鉴 vectorbt）

**文件**: [quant_opt/vectorized_backtest.py](file:///workspace/quant_opt/vectorized_backtest.py)

**实现内容**:
- 预透视数据为 (date × code) 矩阵，消除循环内 `data[data['date']==dt]` 过滤
- 目标权重调仓模型（等权分配，受 max_position_pct 上限）
- T+1 滞后：信号次日生效
- 涨跌停过滤：涨停不买入，跌停不卖出
- 持仓资金随价格变动（pos_capital × price_ratio），正确反映盈亏
- 向量化交易成本计算（佣金 + 印花税 + 滑点）

**测试结果**（7/7 通过）:
```
✓ 基本回测流程正常（96 交易日，139 笔交易，收益 4.83%）
✓ 空数据/无信号正确处理
✓ 资金守恒（权益 = 现金 + 市值，差异 0）
✓ T+1 规则生效（信号日持仓 0，次日建仓）
✓ 涨跌停过滤执行
✓ 绩效指标合理（收益 14.94%，回撤 -4.35%，波动率 7.33%，胜率 56%）
```

**性能对比**（向量化 vs native_adapter）:
| 规模 | native_adapter | 向量化 | 加速比 |
|------|---------------|--------|--------|
| 30 股 × 120 日 | 53.7ms | 41.1ms | 1.3x |
| 50 股 × 250 日 | 162.0ms | 66.2ms | 2.4x |
| 100 股 × 250 日 | 230.1ms | 75.5ms | 3.0x |

**结论**: 向量化回测性能提升随规模增大而显著（1.3x → 3.0x），
因为预透视矩阵消除了 O(n×m) 的循环内过滤。两者绩效指标有差异
（仓位模型不同：目标权重 vs 等额预算），但趋势一致、范围合理。
**建议作为 native_adapter 的高性能替代，用于因子筛选和参数扫描阶段**。

### 优化点 3: 向量化 IC 分析（借鉴 Qlib/vectorbt）

**文件**: [quant_opt/vectorized_ic.py](file:///workspace/quant_opt/vectorized_ic.py)

**实现内容**:
- 用 `groupby('date').corr()` 替代逐日 `scipy.spearmanr` 循环
- Spearman IC = Pearson(rank(factor), rank(forward_return)) 按日分组
- 一次性 merge，避免每因子重复 merge
- 多因子批量 IC 汇总（ic_mean/ic_std/ic_ir/ic_positive_ratio/ic_t_stat）

**测试结果**（5/5 通过）:
```
✓ 正确性：向量化 IC 与 scipy 逐日计算一致（误差 < 1.11e-16）
✓ Pearson IC 计算正确
✓ 多因子 IC 汇总：factor_0（有效因子）IC=0.699，其他因子 IC≈0.02
✓ 边界条件：空数据、样本不足正确处理
```

**性能对比**（5 因子 × 3 周期）:
| 方法 | 耗时 | 加速比 |
|------|------|--------|
| 逐日循环（scipy.spearmanr） | 6319ms | 1.0x |
| 向量化（groupby.corr） | 3200ms | 2.0x |

**结论**: 向量化 IC 分析结果与 scipy 完全一致（误差 < 1e-16），
性能提升 2 倍。**建议替换 factor-engine._calc_ic 的逐日循环实现**。

---

## 四、待用户确认的优化建议

以下优化方案已在 `feat/quant-opt-20260620` 分支验证通过，**等待用户确认后**
方可合并到 main 分支：

### 建议 1: 集成表达式因子引擎到 factor-engine（高优先级）
- **改动**: 在 `factor-engine/scripts/` 新增 `expression_engine.py`，
  将 `compute_a_share_factors` 中的硬编码因子迁移为表达式配置
- **收益**: 新增因子无需改源码，支持 Alpha158 子集，便于因子实验
- **风险**: 低（新增模块，不破坏现有接口）
- **验证状态**: ✅ 已验证（10/10 测试通过）

### 建议 2: 新增向量化回测适配器（高优先级）
- **改动**: 在 `backtest-engine/scripts/adapters/` 新增 `vectorized_adapter.py`，
  继承 `BaseBacktestEngine`，注册为 `BACKTEST_BACKEND="vectorized"`
- **收益**: 大规模回测性能提升 1.3-3 倍，适合因子筛选阶段
- **风险**: 中（仓位模型与 native_adapter 不同，需文档说明）
- **验证状态**: ✅ 已验证（7/7 测试通过）

### 建议 3: 优化 factor-engine 的 IC 分析（中优先级）
- **改动**: 用 `groupby('date').corr()` 替换 `_calc_ic` 的逐日循环
- **收益**: IC 分析性能提升 2 倍，结果完全一致
- **风险**: 低（纯性能优化，接口不变）
- **验证状态**: ✅ 已验证（5/5 测试通过）

### 建议 4: 引入研究-实盘一致性架构（长期方向）
- **借鉴**: NautilusTrader 的 NautilusKernel 设计
- **方向**: 回测引擎与执行监控引擎共享核心状态机（Portfolio/RiskEngine），
  仅在数据/执行客户端边界替换实现
- **收益**: 消除回测-实盘行为差异，降低部署风险
- **风险**: 高（架构重构，需分阶段实施）
- **验证状态**: ⏳ 待规划

### 建议 5: 探索 LLM/RL 因子自动挖掘（前沿方向）
- **借鉴**: AlphaGen / FactorMiner / AlphaForge
- **方向**: 基于 RL 或 LLM+MCTS 自动生成公式化因子，IC 筛选 + 相关性去冗余
- **收益**: 突破人工因子库限制，发现非直觉 Alpha
- **风险**: 高（研究性质，需 GPU 资源）
- **验证状态**: ⏳ 待规划

---

## 五、代码结构

```
quant_opt/                          # 优化验证包（独立于 main 代码）
├── __init__.py
├── expression_factor_engine.py     # 表达式因子引擎（借鉴 Qlib）
├── vectorized_backtest.py          # 向量化回测引擎（借鉴 vectorbt）
├── vectorized_ic.py                # 向量化 IC 分析（借鉴 Qlib/vectorbt）
└── tests/
    ├── __init__.py
    ├── _test_utils.py              # 合成数据生成器
    ├── test_expression_factor.py   # 表达式引擎测试（10 项）
    ├── test_vectorized_backtest.py # 向量化回测测试（7 项）
    └── test_vectorized_ic.py       # 向量化 IC 测试（5 项）
```

## 六、复现方式

```bash
# 切换到验证分支
git checkout feat/quant-opt-20260620

# 安装依赖
pip install pandas numpy scipy scikit-learn pyarrow

# 运行全部测试
python quant_opt/tests/test_expression_factor.py
python quant_opt/tests/test_vectorized_ic.py
python quant_opt/tests/test_vectorized_backtest.py
```

---

## 七、约束遵守说明

- ✅ 所有新代码位于 `feat/quant-opt-20260620` 分支的 `quant_opt/` 目录，**未修改 main 分支任何代码**
- ✅ 分支已推送到 GitHub 远程仓库（仅 push，未 merge）
- ✅ 未执行任何 `git merge` 操作
- ⏳ 等待用户确认后，方可执行 PR 合入 main 分支
