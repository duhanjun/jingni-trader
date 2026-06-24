# jingni-trader 量化优化验证报告

> **执行日期**: 2026-06-24
> **分支**: `feat/quant-opt-20260624`（基于 main，仅 push 未合并）
> **执行人**: 自动化学习与验证流程

---

## 一、联网学习成果

### 1.1 学习项目清单

通过 GitHub Trending、Awesome Quant、QuantConnect、python.financial 等渠道，调研了 2025-2026 年活跃的量化交易开源项目，精选以下 3 个最具借鉴价值的项目深入分析：

| 项目 | Star | 核心定位 | 借鉴价值 |
|------|------|----------|----------|
| **VectorBT / VectorBT PRO** | 4k+ | 向量化回测引擎，NumPy 数组运算替代逐 bar 循环 | ★★★★★ 回测性能 |
| **Microsoft Qlib (Alpha158)** | 17.5k+ | AI 量化平台，表达式化因子定义 + 统一算子语义 | ★★★★★ 因子可扩展性 |
| **Quantopian Alphalens** | 3k+ | 因子 IC 分析与分层回测标准库 | ★★★★ IC 分析向量化 |

辅助参考项目：**AKQuant**（Rust+Python 混合，Polars 因子表达式引擎）、**NautilusTrader**（事件驱动 + 编译内核）、**TradingAgents**（LLM 自然语言策略生成，77.9k star）。

### 1.2 核心亮点总结

**VectorBT 的关键设计**：
- 将回测逻辑从「逐 bar 事件循环」改为「全量 NumPy 数组运算」，参数扫描速度提升 100x+
- 提出 Research→Validation→Live 三阶段工作流：向量化引擎做发现，事件驱动引擎做验证
- 核心洞察：**研究吞吐量**与**执行保真度**应分离，用不同引擎服务不同阶段

**Qlib Alpha158 的关键设计**：
- 因子用表达式字符串定义（如 `Rank(Ts_Mean(Close, 5))`），可序列化、可配置、可组合
- 所有算子基于 `groupby.transform` 实现，**零逐股票 Python 循环**
- 统一算子集（Ts_Mean / Delta / Rank / Corr ...），覆盖 Alpha101 风格公式

**Alphalens 的关键设计**：
- IC 分析本质是「截面 rank + 截面相关系数」，可用 groupby transform 完全向量化
- 公式：`corr = Σ((x-x̄)(y-ȳ)) / √(Σ(x-x̄)²·Σ(y-ȳ)²)`，无需逐日 scipy 调用

---

## 二、jingni-trader 现状分析与优化方向

### 2.1 现有代码性能瓶颈定位

通过逐文件阅读主仓库代码，定位到 3 处显著的 Python 循环瓶颈：

| 位置 | 文件 | 瓶颈代码 | 问题 |
|------|------|----------|------|
| 回测引擎 | [native_adapter.py](file:///workspace/skills/backtest-engine/scripts/adapters/native_adapter.py#L44-L55) | `for dt in dates: for _, row in day_signal.iterrows()` | 逐日 + 逐股票双层 Python 循环 |
| 因子计算 | [pandas_ta_calculator.py](file:///workspace/skills/factor-engine/scripts/adapters/pandas_ta_calculator.py#L62-L72) | `for code in data['code'].unique():` | 逐股票 Python 循环计算因子 |
| IC 分析 | [factor-engine/engine.py](file:///workspace/skills/factor-engine/engine.py#L250-L261) | `for dt in dates: stats.spearmanr(...)` | 逐日 Python 循环 + 逐日 scipy 调用 |
| 中性化 | [factor-engine/engine.py](file:///workspace/skills/factor-engine/engine.py#L148-L178) | `for dt in dates: LinearRegression().fit(...)` | 逐日 Python 循环 |

### 2.2 可借鉴的优化方向

| 优化方向 | 借鉴来源 | 预期收益 | 可行性 |
|----------|----------|----------|--------|
| 向量化回测引擎 | VectorBT | 回测速度 3-13x | 高（语义等价，可对比验证） |
| 因子表达式引擎 | Qlib Alpha158 / AKQuant | 因子计算 7-23x，可扩展性大幅提升 | 高 |
| 向量化 IC 分析 | Alphalens | IC 计算 1.5-3.5x | 高 |
| Walk-forward 验证 | Qlib / AKQuant | 防止过拟合，模型层增强 | 中（需改 model-engine） |
| LLM 策略生成 | TradingAgents / RD-Agent | 自然语言写策略 | 中（需 LLM 接入） |

---

## 三、已完成的验证测试

### 3.1 验证代码结构

所有新代码位于 `quant_opt_20260624/` 独立目录，**未修改 main 分支任何代码**：

```
quant_opt_20260624/
├── synthetic_data.py            # 合成数据生成器（可复现，不依赖 tushare token）
├── vectorized_backtest.py       # 向量化回测引擎（借鉴 VectorBT）
├── factor_expression_engine.py  # 因子表达式引擎（借鉴 Qlib Alpha158）
├── vectorized_ic_analysis.py    # 向量化 IC 分析（借鉴 Alphalens）
├── tests/
│   ├── conftest.py              # 共享 fixtures + 主仓库 NativeAdapter 加载
│   ├── test_correctness.py      # 正确性测试（15 项）
│   ├── test_performance.py      # 性能对比测试（6 项）
│   └── test_edge_cases.py       # 边界条件测试（14 项）
└── results/
    ├── test_output.txt          # 原始测试输出
    └── verification_report.md   # 本报告
```

### 3.2 测试结果总览

```
============================= 35 passed in 38.46s ==============================
```

**全部 35 项测试通过**，分布如下：
- 正确性测试：15 项 ✅
- 性能对比测试：6 项 ✅
- 边界条件测试：14 项 ✅

### 3.3 正确性验证结论

#### 3.3.1 向量化回测 vs 原生回测（NativeAdapter）

`VectorizedAdapter` 实现了与 `NativeAdapter` **完全相同的交易语义**（先卖后买、等额分配、T+1、涨跌停、佣金、印花税、滑点），仅将日内操作向量化。

| 验证项 | 结果 | 容差 |
|--------|------|------|
| 净值曲线一致性 | ✅ 通过 | 相对误差 < 1e-6 |
| 交易笔数一致性 | ✅ 通过 | 完全相等 |
| 最终净值一致性 | ✅ 通过 | 相对误差 < 1e-6 |
| 绩效指标一致性（total_return/max_drawdown/sharpe/win_rate） | ✅ 通过 | 绝对误差 < 1e-6 |
| 不同策略（ma_cross / reversal）一致性 | ✅ 通过 | 相对误差 < 1e-6 |

**结论**：向量化实现与原生实现在数值上完全等价，可安全替换。

#### 3.3.2 因子表达式引擎正确性

| 验证项 | 基准 | 结果 |
|--------|------|------|
| `Delta(Close, 5)` | pandas `groupby.diff(5)` | ✅ rtol < 1e-10 |
| `Ts_Mean(Close, 20)` | pandas `rolling(20).mean` | ✅ rtol < 1e-10 |
| `Rank(Ts_Mean(Close, 5))` | 截面排名 ∈ [0,1] | ✅ 通过 |
| `Div(Sub(High,Low),Close)` | 手工算术 | ✅ rtol < 1e-10 |
| 11 个内置因子可计算 | 无异常 | ✅ 通过 |
| 结果行数 == 输入行数 | 无股票丢弃 | ✅ 通过 |

#### 3.3.3 向量化 IC 分析正确性

| 验证项 | 基准 | 结果 |
|--------|------|------|
| Spearman IC | scipy.stats.spearmanr 逐日 | ✅ 偏差 < 1e-9 |
| Pearson IC | scipy.stats.pearsonr 逐日 | ✅ 偏差 < 1e-9 |
| IC 统计量与序列一致 | 手工校验 | ✅ 通过 |
| 分层收益形状 | 5 分位 × N 日 | ✅ 通过 |

**关键洞察**：向量化 Spearman IC（先截面 rank 再做向量化 Pearson）与 scipy.stats.spearmanr 数值上等价，偏差 < 1e-9。

### 3.4 性能对比结果

| 模块 | 数据规模 | 原生实现 | 向量化实现 | 加速比 |
|------|----------|----------|------------|--------|
| 回测引擎 | 50 股 × 250 日 | 1058.5 ms | 280.5 ms | **3.77x** |
| 回测引擎 | 200 股 × 500 日 | 10338.5 ms | 771.4 ms | **13.40x** |
| 因子计算 | 100 股 × 250 日 | 264.1 ms | 36.5 ms | **7.24x** |
| 因子计算 | 300 股 × 500 日 | 3166.0 ms | 134.6 ms | **23.51x** |
| IC 分析 | 100 股 × 250 日 | 108.3 ms | 30.5 ms | **3.55x** |
| IC 分析 | 300 股 × 500 日 | 286.0 ms | 181.5 ms | **1.58x** |

**关键发现**：
1. **加速比随数据规模增大而提升**——这正是向量化的特征，Python 循环的固定开销在大数据下被摊薄，而 C 层向量化运算的吞吐优势凸显。
2. **回测引擎在大规模下加速最显著（13.4x）**——因为 `iterrows` 的开销与股票数×日数成正比。
3. **因子引擎加速比最高（23.5x）**——逐股票循环的开销在 300 股时极为突出。
4. **IC 分析在大规模下加速比下降（1.58x）**——因为向量化版本中 `groupby.transform` 在大数据下也有一定开销，但仍有正向收益。

### 3.5 边界条件测试结论

14 项边界测试全部通过，覆盖：
- 空数据、单只股票、单日数据
- 全涨停/全跌停（价格限制过滤生效）
- 信号全 0（净值不变）
- T+1 同日买卖顺序
- 因子引擎：未知字段、缺失字段、深层嵌套表达式
- IC 分析：截面股票不足、全 NaN 因子、恒定因子（零方差）

**结论**：三个模块在边界条件下均不抛异常，优雅降级。

---

## 四、待用户确认的优化建议

以下优化方向已通过验证，**等待用户确认后方可合并到 main 分支**：

### 建议 1：替换回测引擎 NativeAdapter 为 VectorizedAdapter（高优先级）

- **改动**：将 `skills/backtest-engine/scripts/adapters/native_adapter.py` 的逐日循环替换为向量化实现
- **收益**：大规模回测 13x 加速，语义完全等价（已验证）
- **风险**：低——正确性测试已证明数值一致
- **建议**：可作为新 adapter 并存，或直接替换

### 建议 2：引入因子表达式引擎替换 PandasTaCalculator（高优先级）

- **改动**：将 `skills/factor-engine/scripts/adapters/pandas_ta_calculator.py` 的逐股票循环替换为表达式引擎
- **收益**：因子计算 23x 加速；因子可配置化（YAML 定义表达式），可扩展性大幅提升
- **风险**：中——需迁移现有因子定义到表达式格式
- **建议**：新增 `expression_calculator.py` adapter，与现有 pandas_ta adapter 并存

### 建议 3：向量化 IC 分析（中优先级）

- **改动**：将 `skills/factor-engine/engine.py` 的 `_calc_ic` 逐日循环替换为向量化实现
- **收益**：IC 计算 1.5-3.5x 加速
- **风险**：低——数值等价已验证
- **建议**：直接替换 `_calc_ic` 方法

### 建议 4（待研究）：引入 Walk-forward 验证机制

- **借鉴**：Qlib / AKQuant 的滚动训练框架
- **目标**：在 `strategy-model-engine` 中加入滚动 train/test，防止过拟合
- **状态**：尚未实现，需进一步设计

### 建议 5（待研究）：LLM 自然语言策略生成

- **借鉴**：TradingAgents（77.9k star）/ Microsoft RD-Agent
- **目标**：用自然语言描述策略，自动生成因子表达式与回测代码
- **状态**：尚未实现，需评估 LLM 接入成本

---

## 五、合规性说明

- ✅ 所有新代码位于 `feat/quant-opt-20260624` 分支的独立目录 `quant_opt_20260624/`
- ✅ **未修改 main 分支任何代码**
- ✅ **未执行 git merge**，仅 `git push` 新分支
- ✅ 测试数据为合成生成，不依赖外部 token / 网络，可复现
- ⏳ 等待用户明确确认后，方可执行 merge / PR 合入 main

---

## 六、复现方式

```bash
# 切换到优化分支
git checkout feat/quant-opt-20260624

# 安装依赖（若未安装）
pip install numpy pandas scipy scikit-learn pytest

# 运行全部测试
cd quant_opt_20260624
python3 -m pytest tests/ -v -s

# 单独运行性能测试查看加速比
python3 -m pytest tests/test_performance.py -v -s
```