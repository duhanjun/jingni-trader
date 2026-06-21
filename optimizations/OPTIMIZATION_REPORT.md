# jingni-trader 量化优化学习与验证报告

> **执行日期**: 2026-06-21
> **分支**: `feat/quant-opt-20260621`
> **状态**: 已推送至 GitHub，**未合并到 main**（待用户确认）

---

## 一、联网学习成果

### 1.1 学习项目清单

通过搜索 GitHub、arXiv、QuantConnect、python.financial 等平台，筛选出以下近期活跃、高 Star、有借鉴价值的量化交易开源项目：

| 项目 | Star | 核心亮点 | 借鉴方向 |
|------|------|---------|---------|
| **AKQuant** | 1.5k+ | Rust 内核 + Python 接口；Polars 驱动的因子表达式引擎（`Rank(Ts_Mean(Close,5))`）；Walk-forward Validation；TA-Lib 双后端 | 因子计算的 Polars 向量化 |
| **Microsoft Qlib** | 15k+ | AI 驱动量化研究；内置大量因子库与模型模板（LightGBM/Transformer）；向量化因子计算与 IC 分析 | 向量化 IC 分析、批量因子评估 |
| **VectorBT / VectorBT PRO** | — | 向量化回测范式（NumPy 数组运算，无逐 bar 循环）；大规模参数搜索；全面绩效指标体系 | 向量化回测指标、绩效指标扩展 |
| **NautilusTrader** | — | Rust/C++ 核心事件驱动引擎；订单簿真实感；回测与实盘行为一致性 | 高性能架构设计参考 |
| **FactorEngine** (arXiv:2603.16365) | — | LLM 引导的因子挖掘；程序级因子（Turing-complete）；贝叶斯超参搜索 | 因子挖掘自动化思路 |
| **vn.py** | 23k+ | 国产量化框架；CTP/IB/加密货币多交易所支持；社区活跃 | 实盘接口设计参考 |

### 1.2 核心亮点总结

1. **向量化范式成为主流**：2026 年 Python 回测生态明显分化为「向量化研究」与「事件驱动实盘」两阵营。研究阶段用向量化引擎（VectorBT/Polars）做快速假设检验，实盘阶段用事件驱动引擎（NautilusTrader）保证执行真实感。

2. **Polars 替代 Pandas 趋势明确**：基于 Rust + Apache Arrow 的 Polars 在大规模数据处理上比 Pandas 快 3-30 倍，内存占用减少 50%+。AKQuant、Qlib 等新项目已采用 Polars 作为因子计算引擎。

3. **因子计算从「逐日循环」转向「向量化分组」**：传统 pandas `for dt in dates` 逐日计算 IC/中性化的方式，被 Polars `group_by + agg` 的多线程向量化计算取代，性能提升 1-2 个数量级。

4. **LLM + 因子挖掘兴起**：FactorEngine 等学术工作探索用 LLM 将非结构化研报转化为可执行因子程序，结合贝叶斯优化做参数调优。

---

## 二、jingni-trader 现状分析与优化方向

### 2.1 对照分析

| 模块 | 现状 | 改进空间 | 借鉴来源 |
|------|------|---------|---------|
| **factor-engine** IC 分析 | `_calc_ic` 逐日 Python 循环 + scipy | 向量化 group_by + corr | AKQuant / Qlib |
| **factor-engine** 中性化 | `neutralize` 逐日循环 + sklearn OLS | FWL 定理向量化 | Qlib |
| **backtest-engine** 指标 | `_calc_metrics` 仅 7 个基础指标 | 扩展至 20+ 专业指标 | VectorBT |
| **backtest-engine** 回测 | 仅事件驱动（逐日循环） | 增加向量化回测路径 | VectorBT |
| **data-engine** | pandas 处理 | Polars 加速数据清洗 | AKQuant |
| **strategy-model-engine** | 无 Walk-forward | 滚动训练验证 | AKQuant |

### 2.2 本次实施的优化（已验证）

本次选取**改进空间最大、可独立验证、不依赖外部数据源**的 3 个方向实施：

1. **Polars 向量化 IC 分析** — 替换 factor-engine 的逐日循环
2. **Polars 向量化因子中性化（FWL 定理）** — 替换 factor-engine 的逐日 OLS
3. **增强版向量化绩效指标** — 扩展 backtest-engine 的指标体系

---

## 三、优化实现详情

### 3.1 Polars 向量化 IC 分析

**文件**: `optimizations/polars_ic_analysis.py`

**优化点**: 原 `_calc_ic` 通过 `for dt in dates` 逐日调用 `scipy.stats.spearmanr`，每次都做 pandas 布尔索引拷贝。

**向量化方案**:
- Spearman IC = `group_by("date")` 内对因子和收益率分别 `rank()`，再用 `pl.corr()` 计算 Pearson 相关
- Pearson IC = `group_by("date").agg(pl.corr(factor, forward))`
- 所有日期的 IC 在 Rust 层多线程并行计算，无 Python 循环
- 增加 NaN 过滤，处理常数因子等边界情况

### 3.2 Polars 向量化因子中性化（FWL 定理）

**文件**: `optimizations/polars_neutralize.py`

**优化点**: 原 `neutralize` 通过 `for dt in dates` 逐日构建 sklearn `LinearRegression` 对象拟合。

**向量化方案（Frisch-Waugh-Lovell 定理）**:

完整 OLS `factor = a + b*lncap + Σ c_k*industry_k + e` 的残差 `e`，可由 FWL 定理等价计算：
1. 按 `(date, industry)` 对 factor 和 lncap 去均值（剥离行业效应）— 向量化 `group_by + mean`
2. 对去均值后的变量做市值回归 — 向量化 `cov/var` 公式

FWL 定理保证残差与完整 OLS **数值完全一致**（测试验证最大偏差 1.15e-14）。

### 3.3 增强版向量化绩效指标

**文件**: `optimizations/vectorized_metrics.py`

**优化点**: 原 `_calc_metrics` 仅 7 个指标（total_return, annual_return, volatility, sharpe, max_drawdown, win_rate, calmar）。

**扩展至 24 个指标**（全 numpy 向量化）:
- 收益类: total_return, annual_return, best_day, worst_day
- 风险类: volatility, downside_deviation, max_drawdown, max_dd_duration, VaR_95, CVaR_95
- 风险调整: sharpe, sortino, calmar, omega
- 胜率盈亏: win_rate, profit_loss_ratio, avg_win, avg_loss, max_consec_win/loss
- 尾部风险: skewness, kurtosis, tail_ratio
- 稳定性: recovery_factor, annual_turnover

---

## 四、验证测试结果

### 4.1 测试概览

| 测试套件 | 通过 | 失败 |
|---------|------|------|
| 正确性测试 | 7 | 0 |
| 性能对比测试 | 4 | 0 |
| 边界条件测试 | 12 | 0 |
| **总计** | **23** | **0** |

### 4.2 正确性验证

将 Polars 实现与复刻原 engine.py 逻辑的 pandas/scipy 参考实现逐项对比：

| 测试项 | 对比方式 | 最大偏差 | 结论 |
|--------|---------|---------|------|
| Spearman IC | 逐日 IC 值对比 | 5.55e-17 | 机器精度内一致 |
| Pearson IC | 逐日 IC 值对比 | 1.11e-16 | 机器精度内一致 |
| IC 统计摘要 | mean/IR/正比例对比 | < 1e-3 | 一致 |
| 市值中性化 | 逐样本残差对比 | 2.66e-15 | 机器精度内一致 |
| 行业+市值中性化 (FWL) | 逐样本残差对比 | 1.15e-14 | FWL 定理验证通过 |
| 绩效指标基础 | 6 项基础指标对比 | < 1e-3 | 一致 |
| 换手率计算 | 含交易记录 | — | 正确 |

### 4.3 性能对比

#### IC 分析加速比

| 规模 (股票×天) | pandas (s) | polars (s) | 加速比 |
|---------------|-----------|-----------|-------|
| 100×100 | 0.179 | 0.0023 | **77x** |
| 300×250 | 0.551 | 0.0117 | **47x** |
| 500×250 | 0.610 | 0.0212 | **29x** |
| 1000×250 | 0.710 | 0.0443 | **16x** |

#### 市值中性化加速比

| 规模 (股票×天) | pandas (s) | polars (s) | 加速比 |
|---------------|-----------|-----------|-------|
| 100×100 | 0.206 | 0.0010 | **200x** |
| 300×250 | 0.601 | 0.0030 | **200x** |
| 500×250 | 0.619 | 0.0043 | **145x** |
| 1000×250 | 0.643 | 0.0077 | **84x** |

#### 行业+市值中性化加速比 (FWL 向量化 vs 逐日 OLS)

| 规模 (股票×天) | pandas (s) | polars (s) | 加速比 |
|---------------|-----------|-----------|-------|
| 100×100 | 0.505 | 0.0020 | **256x** |
| 300×250 | 1.350 | 0.0064 | **211x** |
| 500×250 | 1.488 | 0.0097 | **154x** |
| 1000×250 | 1.598 | 0.0168 | **95x** |

#### 绩效指标

增强版 24 个指标 vs 原基础版 7 个指标，计算耗时仅增加约 2ms（0.0025s vs 0.0006s），可忽略。

### 4.4 边界条件验证

全部 12 项边界测试通过：空数据、单行数据、全 NaN、部分缺失、常数因子、单一行业、样本不足、极端值、短净值序列、空净值序列、无波动序列、单日净值。

---

## 五、待用户确认的优化建议

以下优化方向已验证可行，**需用户确认后**方可合并到 main 分支：

### 建议合并的优化（高置信度）

1. **将 Polars IC 分析集成到 factor-engine**
   - 替换 `skills/factor-engine/engine.py` 的 `_calc_ic` 方法
   - 性能提升 16-77x，数值结果完全一致
   - 需增加 `polars` 依赖

2. **将 FWL 向量化中性化集成到 factor-engine**
   - 替换 `skills/factor-engine/engine.py` 的 `neutralize` 方法
   - 性能提升 95-256x，FWL 定理保证数值一致
   - 行业+市值中性化场景收益最大

3. **将增强版绩效指标集成到 backtest-engine**
   - 替换 `skills/backtest-engine/engine.py` 的 `_calc_metrics` 方法
   - 指标从 7 个扩展到 24 个，耗时几乎无增加
   - 新增 Sortino、VaR、CVaR、Omega 等专业风控指标

### 后续可探索的方向（需进一步研究）

4. **向量化回测引擎** — 借鉴 VectorBT，为 native_adapter 增加向量化回测路径（适合简单策略快速验证）
5. **Walk-forward Validation** — 借鉴 AKQuant，在 strategy-model-engine 增加滚动训练验证
6. **Polars 数据管道** — 在 data-engine 引入 Polars 加速数据清洗
7. **因子表达式引擎** — 借鉴 AKQuant，支持 `Rank(Ts_Mean(Close,5))` 风格的因子表达式

---

## 六、代码结构

```
optimizations/
├── __init__.py                    # 模块说明
├── polars_ic_analysis.py          # Polars 向量化 IC 分析
├── polars_neutralize.py           # Polars 向量化中性化 (FWL)
├── vectorized_metrics.py          # 增强版绩效指标 (24个)
├── OPTIMIZATION_REPORT.md         # 本报告
└── tests/
    ├── __init__.py                # 测试数据生成器
    ├── test_correctness.py        # 正确性测试 (7项)
    ├── test_performance.py        # 性能对比测试 (4项)
    ├── test_boundary.py           # 边界条件测试 (12项)
    ├── run_all_tests.py           # 测试运行器
    └── test_reports/              # 测试输出与摘要
```

---

## 七、复现方式

```bash
# 切换到优化分支
git checkout feat/quant-opt-20260621

# 安装依赖
pip install polars pandas numpy scipy scikit-learn pyarrow

# 运行全部验证测试
python optimizations/tests/run_all_tests.py
```

---

## 八、重要约束说明

- 所有优化代码位于 `feat/quant-opt-20260621` 分支，**未修改 main 分支任何代码**
- 分支已推送到 GitHub 远程仓库，**未执行 git merge**
- 待用户明确确认后，方可合并到 main 分支
