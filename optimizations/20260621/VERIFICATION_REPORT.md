# jingni-trader 量化优化验证报告

> **执行日期**：2026-06-21
> **分支**：`feat/quant-opt-20260621`（基于 `main` 创建，仅 push，未 merge）
> **执行人**：自动化学习与验证流程
> **状态**：✅ 验证完成，等待用户确认是否合入 main

---

## 一、学习项目清单及核心亮点

本轮通过 GitHub、arXiv、Papers with Code、QuantConnect 等渠道检索近期活跃的量化交易开源项目，筛选出 3 个最具借鉴价值的项目深入阅读。

### 1. Microsoft Qlib（⭐ 16k+，最活跃）

- **定位**：面向 AI 的量化投研全流程平台，覆盖数据→因子→模型→回测→组合→执行。
- **核心亮点**：
  - **表达式引擎（Expression Engine）**：用 DSL 描述因子，如 `Ref($close, 60) / $close`、`Mean($close, 20)`，支持任意嵌套；底层用 Cython/NumPy 向量化计算。
  - **Alpha158 / Alpha360 数据集**：标准化因子库，避免重复造轮子。
  - **DataHandler / Processor 分层**：数据加载、清洗、归一化、标签生成解耦，Processor 链式组合（ZScoreNorm / Fillna / DropnaLabel 等）。
  - **RollingGen**：walk-forward 滚动训练切分器，支持 rolling / expanding 两种模式，可配置 step。
  - **TopkDropoutStrategy**：组合层经典实现，可作参考。
- **可借鉴方向**：因子 DSL、Processor 链、RollingGen。

### 2. akquant（Rust 内核 + Python API，国产新锐）

- **定位**：高性能量化研究框架，Rust 核心保证速度，Python API 保证易用。
- **核心亮点**：
  - **Polars 因子表达式引擎**：基于 Polars Lazy API，算子语法接近 Alpha101，如 `Rank(Ts_Mean(Close, 5))`。
  - **`over('code')` / `over('date')` 双轴**：时序算子按 `code` 分组（时间序列），截面算子按 `date` 分组（横截面），语义清晰。
  - **Walk-forward Validation 框架**：内置 purge gap + embargo，防止前视偏差与样本泄漏。
  - **Rust 加速的回测内核**：百万级 bar 回测秒级完成。
- **可借鉴方向**：Polars 因子引擎架构、walk-forward purge/embargo 设计。

### 3. vn.py / Backtrader / Zipline 等老牌框架（对比阅读）

- **vn.py**：CTP 实盘接口封装完善，事件驱动引擎成熟，但偏实盘而非研究。
- **Backtrader**：Strategy/Cerebro/Analyzer 分层清晰，指标计算与回测解耦。
- **Zipline**：`TradingAlgorithm` API 简洁，但已停止维护。
- **可借鉴方向**：Analyzer 指标体系、事件驱动 vs 向量化回测的取舍。

### 其他参考

- **empyrical**（量化指标参考实现）、**pyfolio**（绩效分析报告）、**alphalens**（因子分析）——用于指标口径与因子 IC/IR 分析的对标。
- **FinRL**（强化学习量化）——AI 应用方向参考，本轮未深入。

---

## 二、可借鉴的方向列表（对照 jingni-trader 现状）

| # | 借鉴方向 | 来源 | jingni-trader 现状问题 | 本轮是否验证 |
|---|---------|------|----------------------|------------|
| A | **因子表达式 DSL（Polars 内核）** | Qlib Expression + akquant Polars | `FactorEngine.compute_a_share_factors` 硬编码 12 个因子，新增因子需改源码，无 DSL，无横截面算子 | ✅ 已验证 |
| B | **向量化回测 + 指标口径修正** | empyrical/pyfolio + Backtrader Analyzer | `native_adapter` 逐日 `data[date==dt]` 过滤 O(n) ；`base_backtest.calc_sharpe` 用算术年化但 `calc_annual_return` 用几何年化，口径不一致；`calc_win_rate` 把买单（pnl 为负）也算进去，污染胜率 | ✅ 已验证 |
| C | **Walk-forward 滚动验证（含 purge + embargo）** | akquant + Qlib RollingGen + 金融 ML 文献 | `ModelEngine.purged_group_ts_split` 仅 expanding 模式，无 rolling，无 purge gap，无 embargo，存在前视偏差与样本泄漏风险 | ✅ 已验证 |

### 待后续轮次探索（本轮未做）

- D. **DataHandler/Processor 链式预处理**（Qlib）——jingni-trader 的 `DataEngine` 已有 v3 多源回退，但缺标准化 Processor 链。
- E. **TopkDropoutStrategy 组合层**（Qlib）——`PortfolioRiskEngine` 可参考。
- F. **Rust 加速回测内核**（akquant）——性能再上一个数量级，但工程成本高。
- G. **FinRL 强化学习策略**——AI 方向探索。

---

## 三、已完成的验证测试及结论

### 测试代码

- 测试套件：[`test_optimizations.py`](file:///workspace/optimizations/20260621/test_optimizations.py)
- 优化 A 实现：[`factor_expression_engine.py`](file:///workspace/optimizations/20260621/factor_expression_engine.py)
- 优化 B 实现：[`vectorized_backtest_metrics.py`](file:///workspace/optimizations/20260621/vectorized_backtest_metrics.py)
- 优化 C 实现：[`walk_forward_split.py`](file:///workspace/optimizations/20260621/walk_forward_split.py)

### 测试结果总览

```
======================================================================
jingni-trader 优化验证测试 (feat/quant-opt-20260621)
======================================================================
总计: 38  通过: 38  失败: 0
======================================================================
```

| 类别 | 测试数 | 通过 | 失败 |
|------|-------|------|------|
| A. 因子表达式引擎（正确性/性能/边界） | 14 | 14 | 0 |
| B. 向量化回测 + 指标修正（正确性/性能/指标/边界） | 12 | 12 | 0 |
| C. Walk-forward 滚动分割（正确性/边界） | 12 | 12 | 0 |
| **合计** | **38** | **38** | **0** |

### 优化 A：Polars 因子表达式引擎

**实现要点**：
- 算子集：时序 `Ref/Delta/Mean/Std/Sum/Max/Min/Ts_Rank/Corr/Slope/WMA`（`over('code')`），截面 `Rank/ZScore/Quantile`（`over('date')`）。
- 用 Python `ast` 解析公式字符串为表达式树。
- **两阶段编译**解决 Polars 不能嵌套 `over()` 的限制：截面算子作用于复合时序表达式时，先把内层时序表达式物化为中间列 `__expr_N`，再对中间列施加截面算子。
- `Corr` 算子用动态 `k = both_valid.rolling_sum(...)` 而非硬编码 `n`，正确处理 `min_samples=2` 导致的实际样本数变化。
- 预置 12 个因子（reversal_5d/20d、momentum_20/60、volatility_5/20、vol_ratio、amount_ratio、slope_20、corr_pv_20、cs_rank_mom、cs_zscore_vol）。

**关键测试结论**：

| 测试 | 结果 | 关键数据 |
|------|------|---------|
| 正确性 vs pandas（8 个算子） | ✅ | Ref/Delta/Mean/Std/Rank/复合Rank/Corr/Slope 全部与 pandas 参考实现一致 |
| 性能 polars vs pandas | ✅ | 200 股 × 1000 日 × 12 因子：polars=2.768s，pandas=5.490s，**加速比 1.98x** |
| 边界（空/单股/超窗/未知算子/非法公式） | ✅ | 5 项边界全部不崩溃，超窗返回全 NaN |

### 优化 B：向量化回测 + 指标修正

**实现要点**：
- `DateIndexedBacktester`：预索引 `data.groupby('date')` 为 dict，将逐日 O(n) 过滤降为 O(1) 查表；回测逻辑与 `native_adapter` 等价以保证可比性。
- `CorrectedMetrics`：
  - 同时输出 `annual_return_arith`（算术）与 `annual_return_geom`（几何），Sharpe 同样输出两种口径，让用户自选。
  - `calc_win_rate_corrected` 仅统计 `action == 'sell'` 的平仓单，剔除买单污染。
  - 新增 `calc_benchmark_metrics`：excess_return / beta / alpha / information_ratio / tracking_error。
- `OriginalMetricsReplica`：忠实复刻 `base_backtest.py` 的原版指标，用于对比测试。

**关键测试结论**：

| 测试 | 结果 | 关键数据 |
|------|------|---------|
| 净值曲线与原版一致 | ✅ | max_diff = 1.16e-10（浮点误差级别） |
| 成交笔数一致 | ✅ | opt=809 orig=809 |
| 性能 日期预索引 vs 逐日过滤 | ✅ | 100 股 × 500 日：opt=1.136s orig=1.612s，**加速比 1.42x** |
| 诊断：原版算术年化 ≠ 几何年化 | ✅ | 确认问题存在：arith=-0.0114 vs geom=-0.0231 |
| 修正版输出两种 Sharpe 口径 | ✅ | sharpe_arith=-0.2661，sharpe_geom=-0.3412 |
| 原版 Sharpe == 修正版 sharpe_arith | ✅ | orig=-0.266090 == arith=-0.266090（复刻一致） |
| 胜率：原版被买单污染 vs 修正版仅平仓单 | ✅ | orig=0.2（错） vs corrected=0.5（对） |
| benchmark 相对指标齐全 | ✅ | excess/beta/alpha/IR/tracking_error 全部输出 |
| 边界（空数据/无信号） | ✅ | 2 项边界正确返回空结果 |

### 优化 C：Walk-forward 滚动验证切分

**实现要点**：
- `RollingWindowSplit` 参数：`train_window / test_window / step / valid_window / purge_days / embargo_days / mode`。
- `mode='rolling'`：训练窗口滑动；`mode='expanding'`：训练起点固定，窗口扩张。
- `purge_days`：训练集尾部与测试集之间留间隔，防止标签（前视收益）泄漏到训练集。
- `embargo_days`：下一折训练集起点必须在上一折测试集结束后再加 embargo 天，防止测试集样本通过滑动窗口泄漏到下一折训练集。
- `WindowSegment` dataclass 明确每折的 train/test/valid 起止索引。
- `iter_masks(dates)` 直接产出布尔 mask，便于与 DataFrame 对齐。

**关键修复**：
- 初版锚定 `cursor=0` 为 test_start，导致 `train_end_idx = -1 - purge_days` 非法，生成 0 折。改为锚定 `test_start_idx = train_window + purge_days`，并用 `prev_test_end_idx` 跟踪 embargo。

**关键测试结论**：

| 测试 | 结果 | 关键数据 |
|------|------|---------|
| 生成折数 > 0 | ✅ | 2 折 |
| 训练集严格早于测试集 | ✅ | 全部折满足 |
| 折间测试集不重叠 | ✅ | 全部折满足 |
| embargo 使下一折训练集在测试集之后 | ✅ | 验证生效 |
| purge 使训练集尾部与测试集有间隔 | ✅ | 验证生效 |
| expanding 模式训练集起点固定 | ✅ | 8 折，起点均为 0 |
| valid 窗口正确生成 | ✅ | 2 折含 valid |
| 边界（不足/刚好/非法参数/mask 互斥） | ✅ | 5 项边界全部正确 |

---

## 四、对比分析

### 性能对比汇总

| 场景 | 原版 | 优化版 | 加速比 |
|------|------|--------|-------|
| 因子计算（200 股 × 1000 日 × 12 因子） | pandas 5.490s | polars 2.768s | **1.98x** |
| 回测（100 股 × 500 日） | 逐日过滤 1.612s | 日期预索引 1.136s | **1.42x** |

### 正确性对比汇总

| 场景 | 原版 | 优化版 | 一致性 |
|------|------|--------|-------|
| 因子值（8 个算子） | pandas 参考 | polars 实现 | ✅ 全部一致 |
| 回测净值曲线 | native_adapter | DateIndexedBacktester | ✅ max_diff=1.16e-10 |
| 回测成交笔数 | 809 | 809 | ✅ 完全一致 |

### 指标口径问题诊断

| 指标 | 原版问题 | 修正版方案 |
|------|---------|-----------|
| Sharpe 年化 | 用算术年化 `returns.mean() * trading_days`，与 `calc_annual_return` 的几何年化口径不一致 | 同时输出 `sharpe_arith` 与 `sharpe_geom`，用户自选 |
| 胜率 | 把买单（pnl 为负）也算交易，污染胜率（0.2） | 仅统计 `action=='sell'` 平仓单（0.5） |
| 相对指标 | 缺失 | 新增 excess_return / beta / alpha / IR / tracking_error |

### Walk-forward 对比

| 维度 | 原版 `purged_group_ts_split` | 优化版 `RollingWindowSplit` |
|------|----------------------------|---------------------------|
| 模式 | 仅 expanding | rolling + expanding |
| purge gap | ❌ 无 | ✅ 可配置 |
| embargo | ❌ 无 | ✅ 可配置 |
| 前视偏差风险 | 存在 | 已消除 |
| 样本泄漏风险 | 存在 | 已消除 |

---

## 五、待用户确认的优化建议

以下优化方案已通过验证测试，**尚未合并到 main**，等待用户确认：

### 建议 1：引入因子表达式 DSL（优化 A）

- **替换范围**：`skills/factor-engine/engine.py` 的 `compute_a_share_factors` 方法。
- **方式**：新增 `FactorExpressionEngine` 作为底层计算引擎，原硬编码因子改写为 `PRESET_FACTORS` 字典中的公式字符串。
- **收益**：新增因子只需写一行公式（如 `Rank(-Ref($close,5)/Ref($close,1))`），无需改 Python 代码；性能提升约 2x。
- **风险**：Polars 依赖（项目已有）；DSL 学习成本（低，算子语义透明）。
- **建议**：✅ 推荐合入。

### 建议 2：回测引擎向量化 + 指标口径修正（优化 B）

- **替换范围**：
  - `skills/backtest-engine/scripts/adapters/native_adapter.py`：改用日期预索引。
  - `skills/backtest-engine/scripts/base/base_backtest.py`：修正 Sharpe 口径、胜率统计，新增 benchmark 相对指标。
- **方式**：保持原 API 不变，内部替换实现；新增 `sharpe_geom` / `win_rate_corrected` / benchmark 指标字段。
- **收益**：性能提升约 1.4x；指标口径正确；新增相对指标便于策略评价。
- **风险**：指标口径变化可能影响历史报告对比——建议保留 `sharpe_arith`（原口径）与 `sharpe_geom` 并存，过渡期不破坏旧报告。
- **建议**：✅ 推荐合入，但需在 CHANGELOG 标注指标口径变化。

### 建议 3：Walk-forward 滚动验证（优化 C）

- **替换范围**：`skills/strategy-model-engine/engine.py` 的 `purged_group_ts_split` 方法。
- **方式**：新增 `RollingWindowSplit` 作为切分器，原方法保留为 `mode='expanding'` 的特例。
- **收益**：支持 rolling 模式；消除前视偏差与样本泄漏；提升模型评估可信度。
- **风险**：rolling 模式下训练样本数固定，早期折数样本量较少——已通过 `min_samples` 与边界测试覆盖。
- **建议**：✅ 推荐合入，默认 `mode='rolling'`，保留 expanding 作为可选项。

### 后续轮次建议（本轮未做）

- 引入 Qlib DataHandler/Processor 链式预处理。
- 参考 TopkDropoutStrategy 重构组合层。
- 评估 Rust 加速回测内核的可行性。
- 探索 FinRL 强化学习策略。

---

## 六、合规性声明

- ✅ 所有新代码位于 `feat/quant-opt-20260621` 分支的 `optimizations/20260621/` 独立目录，**未修改 main 分支任何代码**。
- ✅ 分支已推送至 GitHub 远程（仅 push，未 merge）。
- ✅ **未执行任何 `git merge` 操作**，等待用户明确确认后方可合入 main。
- ✅ 验证测试 38/38 全部通过，包含正确性、性能、边界三类测试。

---

## 七、附录：测试输出原文

```
======================================================================
jingni-trader 优化验证测试 (feat/quant-opt-20260621)
======================================================================

A. 因子表达式引擎
  A1. 正确性 (vs pandas): 8/8 PASS
  A2. 性能 (polars vs pandas): polars=2.768s pandas=5.490s 加速比=1.98x
  A3. 边界条件: 5/5 PASS

B. 向量化回测 + 修正指标
  B1. 正确性 (vs 原版逐日过滤): 净值 max_diff=1.16e-10, 成交笔数一致
  B2. 性能: opt=1.136s orig=1.612s 加速比=1.42x
  B3. 指标修正: Sharpe 口径/胜率/benchmark 全部诊断与修正通过
  B4. 边界条件: 2/2 PASS

C. Walk-Forward 滚动分割
  C1. 正确性: 7/7 PASS（含 rolling/expanding/purge/embargo/valid）
  C2. 边界条件: 5/5 PASS

总计: 38  通过: 38  失败: 0
```
