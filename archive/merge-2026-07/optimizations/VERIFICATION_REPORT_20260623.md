# jingni-trader 量化优化验证报告（2026-06-23 执行）

- **执行日期**: 2026-06-23
- **分支**: `feat/quant-opt-20260623`
- **本次提交目录**: `optimizations/`（与历史 `quant_opt/`、`quant_opt_20260623/` 并存，互不冲突）
- **测试结果**: 25 项通过 / 0 项失败
- **状态**: 已推送至 GitHub 远程，**未合并 main**（等待用户确认）

---

## 一、学习项目清单及核心亮点

通过联网调研 GitHub、arXiv、QuantConnect、技术社区等，筛选出以下 3 个最具借鉴价值的开源项目：

### 1. Microsoft Qlib（15k+ Star，AI 量化平台）
- **仓库**: https://github.com/microsoft/qlib
- **论文**: https://arxiv.org/abs/2009.11189
- **核心亮点**:
  - **表达式引擎**：声明式因子定义，将数学表达式（如 `MA(Close,20)-MA(Close,5)`）解析为 AST 并映射为可执行算子树，无需写 Python 代码即可定义因子
  - **算子注册器模式**：`ElemOperator`/`PairOperator`/`Rolling` 类层次，统一接口 + 自动发现
  - **Alpha158/Alpha360 因子库**：开箱即用的标准化因子集，含元信息（方向、依赖）
  - **高性能基础设施**：列式 `.bin` 存储 + 缓存机制避免重复计算
  - **Walk-forward 验证框架**：滚动训练/测试，防过拟合
  - **分层架构**：Data → Feature → Model → Signal → Strategy → Portfolio 清晰解耦

### 2. VectorBT（向量化回测引擎）
- **官网**: https://vectorbt.dev/
- **核心亮点**:
  - **矩阵化回测**：将策略表示为 NumPy 多维数组，参数组合放入列维度，单次运算完成数千回测
  - **Numba/Rust 加速**：JIT 编译解决向量化中的路径依赖问题
  - **`Portfolio.from_signals`**：信号直接转组合，支持广播机制多资产
  - **性能**：10,000 次双均线回测 ~15 秒（Backtrader 需 1 小时+）
  - **研究优先定位**：明确区分“研究阶段快速验证”与“实盘精确执行”

### 3. TradingAgents（74k+ Star，多智能体 LLM 交易）
- **仓库**: https://github.com/TauricResearch/TradingAgents
- **核心亮点**: 7 个专业化 AI Agent 模拟对冲基金决策结构，含牛熊对抗辩论
- **借鉴价值**: 偏 AI/研究方向，对核心引擎优化参考较小，暂未纳入本次代码验证

---

## 二、可借鉴的方向列表

对照 jingni-trader 现有代码，识别出以下改进空间：

| # | 模块 | 现有问题 | 借鉴来源 | 优化方向 | 优先级 |
|---|------|---------|---------|---------|--------|
| 1 | 回测引擎 `native_adapter.py` | `for dt in dates` + `iterrows` 逐行循环，大股票池极慢 | VectorBT | 向量化矩阵回测 | 高 |
| 2 | 因子引擎 `compute_a_share_factors` | 所有因子硬编码在一个方法，扩展需改源码 | Qlib Alpha158 | 装饰器注册表 + 依赖拓扑排序 | 高 |
| 3 | 因子引擎 `_calc_ic` | 逐日循环调用 `scipy.spearmanr` | Qlib 基础设施 | groupby + rank 向量化 | 高 |
| 4 | 因子引擎 `neutralize` | 逐日循环拟合 `LinearRegression` | Qlib 基础设施 | groupby + 闭式 OLS 向量化 | 高 |
| 5 | 回测引擎 | 同日收盘价同时用于信号与执行，存在前视偏差风险 | Qlib/VectorBT 默认 | 信号 shift(1)，T+1 次日执行 | 中 |
| 6 | 数据引擎 | 无因子缓存，重复计算 | Qlib 缓存机制 | 因子结果缓存 | 中 |
| 7 | 策略 API | 无声明式因子表达式 | Qlib 表达式引擎 | 表达式 DSL（远期） | 低 |

---

## 三、已完成的验证测试及结论

本次针对前 4 项高优先级方向编写了验证代码，全部位于 `optimizations/` 目录，**未修改任何 main 分支原有文件**。

### 验证代码结构

```
optimizations/
├── __init__.py
├── VERIFICATION_REPORT_20260623.md   # 本报告
├── vectorized_backtest_adapter.py    # 向量化回测引擎（VectorBT 思想）
├── factor_registry.py                # 因子注册表框架（Qlib Alpha158 思想）
├── vectorized_factor_analysis.py     # 向量化 IC/中性化（Qlib 基础设施思想）
└── tests/
    ├── __init__.py
    ├── test_all.py                   # 正确性 + 性能 + 边界测试
    └── test_results.json             # 测试结果（机器可读）
```

### 测试结果汇总（25/25 通过）

| 测试类别 | 测试项 | 结果 | 关键数据 |
|---------|--------|------|---------|
| **向量化回测正确性** | 净值列完整 / 首日净值 / 无NaN / 持仓数 / 指标完整 / 回撤非正 | 6/6 PASS | 首日净值=1,000,000 |
| **向量化回测边界** | 空数据 / 单日 / 全NaN信号 / 全涨停过滤 | 4/4 PASS | - |
| **向量化回测性能** | 300股×500日 vs 原生事件驱动 | PASS | **加速 10.3x**（1.23s vs 12.68s） |
| **因子注册表正确性** | 注册数 / 分类 / 方向 / 计算 / 依赖等式 / 拓扑排序 | 6/6 PASS | 13 内置因子，6 分类 |
| **因子注册表扩展性** | 自定义因子一行注册 + 计算 | 2/2 PASS | - |
| **向量化 IC 正确性+性能** | 与原循环数学等价 + 加速比 | 3/3 PASS | 相关系数=1.000000，**加速 5.3x** |
| **向量化中性化正确性+性能** | 与原循环等价 + 市值相关性降低 + 加速比 | 3/3 PASS | 相关系数=1.000000，**加速 34.5x** |

### 关键性能对比

| 模块 | 原循环实现 | 向量化实现 | 加速比 | 数学等价性 |
|------|-----------|-----------|--------|-----------|
| 回测引擎（300股×500日） | 12.681s | 1.230s | **10.3x** | 方法差异（等权目标组合 vs 现金路径依赖） |
| IC 分析（100股×200日） | 0.4226s | 0.0803s | **5.3x** | 完全等价（相关系数=1.0，差异=0） |
| 中性化（80股×100日） | 0.5342s | 0.0155s | **34.5x** | 完全等价（相关系数=1.0） |

### 重要说明

1. **IC 分析与中性化**：向量化实现与原循环**数学完全等价**（相关系数 1.0，最大数值差异 0），可安全替换。
2. **向量化回测**：采用“等权目标组合”法（Qlib TopkDropout 同款），消除现金路径依赖以实现整体向量化；与原 native_adapter 的“现金预算分配”法**非完全等价**，但同为合法回测方法，适合研究阶段快速验证。实盘/精确资金曲线仍建议用事件驱动引擎。
3. **因子注册表**：完全兼容原 `compute_a_share_factors` 输出结构，新增因子仅需一行 `@FactorRegistry.register(...)` 装饰器，无需修改核心代码。

---

## 四、待用户确认的优化建议

以下优化方案已在 `feat/quant-opt-20260623` 分支验证通过，**等待您确认后方可合并 main**：

### 建议 A：合入向量化 IC 分析与中性化（强烈推荐）
- **风险**: 极低（数学完全等价，相关系数=1.0）
- **收益**: IC 分析 5.3x 加速，中性化 34.5x 加速
- **改动**: `skills/factor-engine/engine.py` 的 `_calc_ic` 与 `neutralize` 方法替换为 `VectorizedFactorAnalysis` 实现

### 建议 B：引入因子注册表框架（推荐）
- **风险**: 低（新增模块，原 `compute_a_share_factors` 可逐步迁移）
- **收益**: 因子库可扩展性大幅提升，支持依赖管理、元信息查询、自动文档
- **改动**: `skills/factor-engine/scripts/` 新增 `factor_registry.py`，`engine.py` 增加注册表调用入口

### 建议 C：新增向量化回测后端（推荐，作为研究阶段加速器）
- **风险**: 中（与原生回测方法不同，需明确使用场景）
- **收益**: 大股票池回测 10x+ 加速，适合参数扫描/因子快速验证
- **改动**: `skills/backtest-engine/scripts/adapters/` 新增 `vectorized_adapter.py`，`config.py` 的 `BACKTEST_BACKEND` 增加 `"vectorized"` 选项
- **建议保留**原生事件驱动引擎用于实盘/精确回测

### 建议 D：前视偏差防护（推荐）
- **风险**: 低
- **收益**: 消除回测前视偏差，结果更贴近实盘
- **改动**: 回测引擎信号执行增加 `shift(1)`，明确“T 日收盘信号 → T+1 开盘执行”

---

## 五、复现方式

```bash
git fetch origin
git checkout feat/quant-opt-20260623
pip install pandas numpy scipy scikit-learn
python -m optimizations.tests.test_all
```

测试结果 JSON: `optimizations/tests/test_results.json`
