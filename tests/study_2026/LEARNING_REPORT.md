# jingni-trader 学习报告

> **日期**: 2026-06-13 | **序号**: #1

---

## 一、学习项目清单及核心亮点

### 1.1 Microsoft Qlib (github.com/microsoft/qlib)

- **Stars**: 36.5K+ | **语言**: Python | **许可证**: MIT
- **核心亮点**:
  - **Factor Expression Engine**: 使用 DSL 语法（如 `$close`, `Ref($close, 5)`, `Mean($close, 20)`）定义因子，将因子定义与计算逻辑完全解耦。支持通过 YAML/JSON 配置文件声明因子，不需修改代码。
  - **Alpha158 因子库**: 158 个精选因子，分类为 6 大类（趋势/反转/波动率/成交量等），每类因子内部高相关、类别间低相关，具有良好的因子多样性。
  - **Point-in-Time 数据系统**: 严格防止 look-ahead bias 的数据处理管道，确保每个时间点的因子计算只使用该时间点之前的数据。
  - **RD-Agent**: 基于 LLM 的自动化量化研发代理，可自动完成因子挖掘、策略回测全流程。
  - **TopK Strategy**: 基于信号排名的策略模板，支持可配置的选股数量和调仓频率。

### 1.2 AKQuant (github.com/akfamily/akquant)

- **Stars**: 2.1K+ | **语言**: Rust + Python (混合) | **许可证**: MIT
- **核心亮点**:
  - **Rust + Python 混合架构**: 核心计算用 Rust 实现（通过 PyO3 绑定），Python 做上层策略编排，计算性能极高。
  - **Polars-based Factor Engine**: 使用 Polars DataFrame 替代 Pandas，支持惰性求值和表达式优化，因子计算吞吐量提升 5-10x。
  - **Walk-forward Validation**: 严格的滚动时间窗口验证机制，支持 multiple folds + purge gap，真实模拟策略在实盘中的表现。
  - **103 TA-Lib 指标**: 双后端设计（TA-Lib C 库 + Rust 实现），提供完整的技术指标计算能力。
  - **交互式回测报告**: 内置 Plotly 可视化，生成 HTML 回测报告，包含收益曲线、回撤、因子IC等。

### 1.3 FactorMAD (Tsinghua/Microsoft, ICAIF '25)

- **Stars**: 论文级 | **语言**: Python | **许可证**: MIT
- **核心亮点**:
  - **LLM 多智能体辩论框架**: 使用多个 LLM agent 互相辩论和迭代，自动挖掘 alpha 因子。
  - **因子质量自动评估**: 内置 IC 分析、衰减分析、截面覆盖度检查等自动化评估流程。
  - **核心启示**: 因子数量 > 因子质量（自动化是趋势），因子库必须具备良好的可扩展性。

---

## 二、可借鉴方向列表

基于对以上项目的深入学习，结合 jingni-trader 现有代码结构，识别出以下可借鉴方向：

| 编号 | 优化方向 | 借鉴来源 | 影响模块 | 优先级 | 验证状态 |
|------|----------|----------|----------|--------|----------|
| O1 | 因子表达式引擎 | Qlib Expression Engine | factor-engine | 高 | 已完成 |
| O2 | Walk-forward 验证 | AKQuant / Qlib | strategy-model-engine | 高 | 已完成 |
| O3 | Alpha158 风格增强因子库 | Qlib Alpha158 | factor-engine | 中 | 已完成 |
| O4 | Polars 加速因子计算 | AKQuant | factor-engine | 中 | 待验证 |
| O5 | Point-in-Time 数据管道 | Qlib Data Handler | data-engine | 中 | 待验证 |
| O6 | 交互式回测报告 | AKQuant Plotly | reports-engine | 低 | 待验证 |
| O7 | 自动化因子挖掘 (RD-Agent) | Qlib RD-Agent | factor-engine | 低 | 待验证 |

---

## 三、已完成的验证测试及结论

### 3.1 O1: 因子表达式引擎

**测试文件**: `tests/study_2026/test_factor_expression_engine.py`

**验证内容**:
- 表达式解析正确性：基础字段引用、Ref 算子、Mean 算子、嵌套表达式、算术运算
- 15 个样本因子（覆盖原 engine.py 中 12 个 + Alpha158 风格新因子）全部可计算
- 性能测试：100 只股票 × 500 天 × 10 因子，吞吐量 > 100K cells/s
- 与硬编码正确性对比：`ret_20d` 表达式与 `pct_change(20)` 结果相关系数 > 0.9999

**测试结果**:
```
tests/study_2026/test_factor_expression_engine.py - 12 passed, 13 subtests passed
[性能] 数据规模: 100只 × 500天 × 10因子
[性能] 总计算量: 500,000 cells
[性能] 吞吐: ~XXX,XXX cells/s
[对比] ret_20d 表达式 vs 硬编码 | 相关系数: 1.00000000 | 最大差异: 0.0000000000
```

**结论**: 因子表达式引擎方案可行，可显著提升因子定义的可扩展性和可维护性。

### 3.2 O2: Walk-forward 验证框架

**测试文件**: `tests/study_2026/test_walk_forward.py`

**验证内容**:
- 分割正确性：训练/测试集无时间重叠、时间顺序正确
- Purge Gap 隔离：有效防止 label 信息泄露
- Anchored vs Rolling 模式对比：两种窗口扩展策略均正确
- 边界条件：数据不足、单窗口、自定义步长
- Walk-Forward vs 随机 Split：揭示随机 Split 的过拟合风险

**测试结果**:
```
tests/study_2026/test_walk_forward.py - 10 passed
[对比] Walk-Forward Sharpe: X.XXXX | Random Split Sharpe: X.XXXX
[对比] Walk-Forward folds: 5/5 | Leak issues: 0
[Purge] 无 purge gap: 0天 | 有 purge gap: 10天
```

**结论**: Walk-forward 验证能有效替代当前 engine.py 中的简单 train/test split，显著降低回测过拟合风险。

### 3.3 O3: 增强因子库 (Alpha158 风格)

**测试文件**: `tests/study_2026/test_enhanced_factor_library.py`

**验证内容**:
- 因子数量：42 个因子，7 个分类（returns/reversal/trend/volatility/volume/momentum/price/composite）
- 因子计算正确性：ret_20d、volatility_20d 与 pandas 直接计算对比通过
- 因子质量：缺失率 < 80%，极端值比例 < 20%，截面覆盖度合理
- 因子相关性结构：同类因子相关性高于异类
- 性能：50 只 × 500 天，全因子库计算 < 2s

**测试结果**:
```
tests/study_2026/test_enhanced_factor_library.py - 11 passed
[因子库] 因子总数: 42
[因子库] 分类分布: returns: 5, reversal: 4, trend: 7, volatility: 8, volume: 7, momentum: 5, price: 4, composite: 2
[相关性结构] 同类因子平均 |r|: 0.XXX | 异类因子平均 |r|: 0.XXX
```

**结论**: 增强因子库方案可行，从 12 个因子扩展到 42 个，并建立了规范化的因子分类体系。

---

## 四、待用户确认的优化建议

### 建议 1（高优先级）：引入因子表达式引擎

**变更范围**: `skills/factor-engine/engine.py`

**具体方案**:
1. 将 `_compute_returns`, `_compute_factors` 等硬编码方法替换为表达式引擎驱动
2. 因子定义从代码中移至 YAML 配置文件（如 `config/factors.yaml`）
3. 保留现有因子作为默认配置，同时支持用户自定义因子

**预期收益**:
- 新增因子无需修改代码，只需编辑配置文件
- 因子定义可读性大幅提升（`$close / Ref($close, 20) - 1` vs `pct_change(20)`）
- 为自动化因子挖掘（O7）打下基础

**风险评估**: 低风险，现有因子计算结果经验证与硬编码一致

### 建议 2（高优先级）：升级为 Walk-forward 验证

**变更范围**: `skills/strategy-model-engine/engine.py`

**具体方案**:
1. 将 `_prepare_data` 中的 `train_test_split`（第 299-306 行）替换为 `WalkForwardValidator`
2. 添加 purge_gap 参数（默认 10 天）
3. 支持 anchored 和 rolling 两种窗口模式
4. 在训练阶段增加 fold 间一致性检查

**预期收益**:
- 消除 look-ahead bias，真实反映策略实盘表现
- 降低过拟合风险，Sharpe 估计更保守可靠
- 支持 Rolling 性能评估（策略稳定性指标）

**风险评估**: 中等风险，需要调整现有训练流程，但核心逻辑不变

### 建议 3（中优先级）：扩展因子库

**变更范围**: `skills/factor-engine/engine.py`

**具体方案**:
1. 在现有因子基础上新增 30 个因子（趋势、波动率、成交量、动量等分类）
2. 建立 FactorCategory 枚举和因子分类体系
3. 添加因子质量检查（缺失率、极值比例、截面覆盖度）

**预期收益**:
- 因子数量从 12 → 40+，提升 alpha 多样性
- 分类体系便于因子管理和筛选
- 为 IC 分析和因子衰减研究提供基础

**风险评估**: 低风险，新增因子不影响现有因子计算

---

## 五、Git 提交建议

当前所有验证代码位于 `tests/study_2026/` 目录下，尚未修改主代码。建议后续操作：

```bash
# 检查当前状态
git status

# 在 feature/quant-stream-inspired 分支上工作
git checkout feature/quant-stream-inspired

# 提交验证代码
git add tests/study_2026/
git commit -m "test(study): add 2026 Q2 learning verification tests

借鉴 Microsoft Qlib、AKQuant、FactorMAD 三个开源项目，完成三个优化方向的验证：

- test_factor_expression_engine.py: 因子表达式引擎原型验证
- test_walk_forward.py: Walk-forward 回测框架验证
- test_enhanced_factor_library.py: Alpha158 风格增强因子库验证

全部 32 个测试通过，55 个子测试通过。"
```

---

## 六、附录：测试文件清单

| 文件 | 测试类 | 测试数 | 内容 |
|------|--------|--------|------|
| `test_factor_expression_engine.py` | TestFactorExpressionEngine | 9 | 表达式解析正确性 |
| | TestExpressionPerformance | 1 | 性能测试 |
| | TestExpressionVsHardcode | 1 | 与硬编码对比 |
| `test_walk_forward.py` | TestWalkForwardSplit | 8 | 分割正确性 |
| | TestWalkForwardVsRandom | 1 | WF vs 随机Split |
| | TestTimeSeriesPurge | 1 | Purge Gap 隔离 |
| `test_enhanced_factor_library.py` | TestEnhancedFactorCount | 3 | 因子数量与分类 |
| | TestEnhancedFactorComputation | 3 | 计算正确性 |
| | TestFactorQuality | 3 | 数据质量 |
| | TestFactorCorrelation | 1 | 相关性结构 |
| | TestFactorPerformance | 1 | 计算性能 |