# jingni-trader 量化优化验证报告

> 生成时间：2026-06-20 03:26:12
> 分支：`feat/quant-opt-20260620`
> 测试状态：**通过**

---

## 一、优化点说明

本次优化基于对开源量化项目的研究，针对 jingni-trader 现有代码的 3 个改进方向，
在独立分支 `feat/quant-opt-20260620` 的 `optimizations/20260620/` 目录下实现，
**未修改 main 分支任何代码**。

### 优化 1：向量化因子 IC 分析与中性化（性能优化）

**借鉴来源**：
- [AlphaPurify](https://pypi.org/project/alphapurify/)（2026-05 发布，Polars 向量化，4M 行 25 秒）
- [Microsoft Qlib](https://github.com/microsoft/qlib)（高性能 DataServer，比 Pandas 快 10x）

**问题**：
jingni-trader 现有 `FactorEngine._calc_ic` 与 `FactorEngine.neutralize` 对每个日期循环：
- IC 分析：逐日调用 `scipy.stats.spearmanr`，D 个日期 = D 次 Python 调用
- 中性化：逐日构造 `sklearn.LinearRegression` 对象并 fit/predict，开销大

**方案**：
- IC：`groupby + transform(rank)` + 向量化 Pearson 公式，将逐日循环压缩为几次整表运算
- 中性化：`groupby.apply` + numpy `lstsq` 替代 sklearn，避免对象构造开销
- 仅市值场景：用 Frisch-Waugh-Lovell 定理完全向量化（无需逐日 apply）

**文件**：
- `factor_engine_opt/vectorized_ic.py`
- `factor_engine_opt/vectorized_neutralize.py`

### 优化 2：因子预处理（去极值 + 标准化）

**借鉴来源**：
- AlphaPurify（40+ 预处理方法：Winsorization / Neutralization / Standardization）
- Qlib processor（Normalize / RobustZScoreNorm / Fillna 声明式处理器）

**问题**：
jingni-trader 现有因子引擎在 IC 分析与融合前未做去极值和标准化，
极端值会扭曲 IC 与 IC-IR 加权，且不同量纲因子无法直接加权融合。

**方案**：
- `winsorize_mad`：MAD 法去极值（抗异常值，比 3σ 更稳健）
- `winsorize_quantile`：分位数法去极值（1%/99% 截断）
- `standardize_zscore`：Z-score 标准化（截面均值 0、标准差 1）
- `preprocess_factor`：一站式 pipeline（去极值 → 标准化）

所有操作均按 date 分组向量化。

**文件**：
- `factor_engine_opt/preprocessing.py`

### 优化 3：增强回测绩效指标 + 前视偏差检测

**借鉴来源**：
- [Qlib backtest.performance](https://qlib.readthedocs.io/)（turnover / alpha / beta / IR 完整体系）
- [QuantStats](https://github.com/ranaroussi/quantstats)（max_drawdown_duration / profit_factor）
- [Jesse](https://jesse.trade/)（零 look-ahead bias 设计）
- [Qlib Point-in-Time Data](https://deepwiki.com/microsoft/qlib)（PIT 数据避免未来信息泄漏）

**问题**：
jingni-trader 现有 `BaseBacktestMetrics.calc_all_metrics` 缺少：
- 换手率（衡量交易成本敏感度）
- Alpha/Beta（相对基准的 CAPM 指标）
- 信息比率（主动策略核心评估指标）
- 最大回撤持续期（资金被套时间）
- 前视偏差校验（回测可信度保障）

**方案**：
- `calc_turnover`：从持仓明细计算年化换手率
- `calc_alpha_beta`：CAPM 回归，年化 alpha 与 beta
- `calc_information_ratio`：超额收益 / 跟踪误差
- `calc_max_drawdown_duration`：净值创新高到回到前高的最长天数
- `check_forward_return_leakage`：检测特征是否泄漏未来收益
- `check_signal_timestamp_order`：检测信号是否使用了未来数据
- `check_feature_alignment`：检测因子与未来收益对齐是否正确

**文件**：
- `backtest_engine_opt/enhanced_metrics.py`
- `backtest_engine_opt/look_ahead_guard.py`

---

## 二、借鉴来源清单

| 项目 | 类型 | 核心亮点 | 借鉴方向 |
|------|------|----------|----------|
| [Microsoft Qlib](https://github.com/microsoft/qlib) | 开源平台 (11k+ stars) | 高性能 DataServer、PIT 数据、Alpha158 因子库、表达式引擎 | 向量化、增强指标、PIT 设计 |
| [AlphaPurify](https://pypi.org/project/alphapurify/) | Python 库 (2026-05) | Polars 向量化、40+ 预处理方法、4M 行 25 秒 | 向量化 IC、去极值/标准化 |
| [FactorEngine (arXiv:2603.16365)](https://arxiv.org/abs/2603.16365) | 论文 (2026-04) | LLM 程序级因子挖掘、知识注入、经验库 | 因子挖掘方向（后续） |
| [RD-Agent(Q)](https://github.com/microsoft/RD-Agent) | 多智能体框架 | 因子-模型协同优化、2x 收益 70% 更少因子 | 自动化研究（后续） |
| [Jesse](https://jesse.trade/) | 回测框架 | 零 look-ahead bias、Monte Carlo 压力测试 | 前视偏差检测 |
| [QuantStats](https://github.com/ranaroussi/quantstats) | 绩效分析库 | 丰富绩效归因与可视化 | 增强指标 |

---

## 三、测试结果

### 测试统计

| 指标 | 数值 |
|------|------|
| 测试总数 | 57 |
| 通过 | 57 |
| 失败 | 0 |
| 错误 | 0 |
| 跳过 | 0 |
| pytest 运行时间 | 5.99s |

### 测试覆盖

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_vectorized_ic.py` | IC 正确性（vs 逐日循环）、性能对比、边界条件 |
| `test_vectorized_neutralize.py` | 中性化正确性（vs sklearn）、性能对比、FWL、边界 |
| `test_preprocessing.py` | 去极值/标准化正确性、边界条件、pipeline 一致性 |
| `test_enhanced_metrics.py` | Alpha/Beta/IR/换手率/回撤持续期正确性、边界 |
| `test_look_ahead_guard.py` | 前视偏差检测正向/反向、边界条件 |

### pytest 完整输出

```
_abnormal_forward_return_detected PASSED [ 45%]
tests/test_look_ahead_guard.py::TestFeatureAlignment::test_missing_forward_col PASSED [ 47%]
tests/test_preprocessing.py::TestWinsorizeMAD::test_extreme_values_clipped PASSED [ 49%]
tests/test_preprocessing.py::TestWinsorizeMAD::test_normal_values_preserved PASSED [ 50%]
tests/test_preprocessing.py::TestWinsorizeMAD::test_per_date_grouping PASSED [ 52%]
tests/test_preprocessing.py::TestWinsorizeQuantile::test_quantile_clip PASSED [ 54%]
tests/test_preprocessing.py::TestStandardize::test_mean_zero_std_one PASSED [ 56%]
tests/test_preprocessing.py::TestStandardize::test_constant_value_no_div_zero PASSED [ 57%]
tests/test_preprocessing.py::TestPreprocessPipeline::test_pipeline_equivalent_to_steps PASSED [ 59%]
tests/test_preprocessing.py::TestPreprocessPipeline::test_no_winsorize PASSED [ 61%]
tests/test_preprocessing.py::TestPreprocessBoundary::test_all_nan PASSED [ 63%]
tests/test_preprocessing.py::TestPreprocessBoundary::test_single_value PASSED [ 64%]
tests/test_vectorized_ic.py::TestICCorrectness::test_spearman_matches_loop PASSED [ 66%]
tests/test_vectorized_ic.py::TestICCorrectness::test_pearson_matches_loop PASSED [ 68%]
tests/test_vectorized_ic.py::TestICCorrectness::test_ic_stats_consistency PASSED [ 70%]
tests/test_vectorized_ic.py::TestICCorrectness::test_ic_analysis_batch PASSED [ 71%]
tests/test_vectorized_ic.py::TestICPerformance::test_vectorized_faster_than_loop PASSED [ 73%]
tests/test_vectorized_ic.py::TestICBoundary::test_empty_data PASSED      [ 75%]
tests/test_vectorized_ic.py::TestICBoundary::test_missing_column PASSED  [ 77%]
tests/test_vectorized_ic.py::TestICBoundary::test_insufficient_samples PASSED [ 78%]
tests/test_vectorized_ic.py::TestICBoundary::test_all_nan_factor PASSED  [ 80%]
tests/test_vectorized_ic.py::TestICBoundary::test_single_date PASSED     [ 82%]
tests/test_vectorized_neutralize.py::TestNeutralizeCorrectness::test_mcap_industry_matches_sklearn PASSED [ 84%]
tests/test_vectorized_neutralize.py::TestNeutralizeCorrectness::test_mcap_only_matches_sklearn PASSED [ 85%]
tests/test_vectorized_neutralize.py::TestNeutralizeCorrectness::test_fwl_mcap_only_matches_sklearn PASSED [ 87%]
tests/test_vectorized_neutralize.py::TestNeutralizeCorrectness::test_residual_uncorrelated_with_mcap PASSED [ 89%]
tests/test_vectorized_neutralize.py::TestNeutralizePerformance::test_vectorized_faster_than_sklearn PASSED [ 91%]
tests/test_vectorized_neutralize.py::TestNeutralizePerformance::test_fwl_faster_than_apply PASSED [ 92%]
tests/test_vectorized_neutralize.py::TestNeutralizeBoundary::test_empty_data PASSED [ 94%]
tests/test_vectorized_neutralize.py::TestNeutralizeBoundary::test_no_regressors PASSED [ 96%]
tests/test_vectorized_neutralize.py::TestNeutralizeBoundary::test_insufficient_samples PASSED [ 98%]
tests/test_vectorized_neutralize.py::TestNeutralizeBoundary::test_no_neutralization PASSED [100%]

============================== 57 passed in 5.42s ==============================

```

---

## 四、性能对比结果

### 1. 因子 IC 分析（Spearman Rank IC）

| 实现 | 数据规模 | 耗时 | 加速比 |
|------|----------|------|--------|
| 逐日循环 scipy.spearmanr（现有） | 200 日 × 300 股 | 0.5047s | 1.00x（基准） |
| 向量化 groupby+rank（优化） | 200 日 × 300 股 | 0.0421s | **11.99x** |

### 2. 因子中性化（市值 + 行业）

| 实现 | 数据规模 | 耗时 | 加速比 |
|------|----------|------|--------|
| 逐日 sklearn LinearRegression（现有） | 100 日 × 300 股 × 10 行业 | 0.66s | 1.00x（基准） |
| 向量化 groupby.apply+numpy.lstsq（优化） | 100 日 × 300 股 × 10 行业 | 0.2649s | **2.49x** |

### 3. 仅市值中性化（FWL 完全向量化）

| 实现 | 数据规模 | 耗时 | 加速比 |
|------|----------|------|--------|
| 逐日 sklearn LinearRegression（现有） | 100 日 × 300 股 | 0.66s | 1.00x（基准） |
| FWL 定理完全向量化（优化） | 100 日 × 300 股 | 0.012s | **55.1x** |

---

## 五、对比分析

### 正确性

- **IC 分析**：向量化 Spearman/Pearson IC 与逐日循环结果在 `1e-6` 容差内一致，
  验证了向量化实现的正确性。
- **中性化**：向量化残差与逐日 sklearn 回归残差在 `1e-6` 容差内一致；
  中性化后因子与市值相关性 < 0.05，验证了中性化效果。
- **预处理**：去极值后极端值被截断、正常值保留；标准化后截面均值≈0、标准差≈1。
- **增强指标**：Alpha/Beta 与 CAPM 理论值一致（beta=1/2 的构造策略回归正确）；
  信息比率与定义一致；换手率与已知持仓变化一致。

### 性能

- IC 分析向量化实现获得 **11.99x** 加速，主要来自避免逐日 Python 循环
  与 scipy 函数调用开销。
- 中性化向量化获得 **2.49x** 加速，主要来自
  避免 sklearn 对象构造与 fit/predict 开销，改用 numpy lstsq。
- 仅市值场景的 FWL 完全向量化获得 **55.1x** 加速，
  因为完全消除了逐日 apply，所有运算通过 groupby+transform 一次性完成。

### 边界条件

所有模块均通过边界条件测试：空数据、单日、样本不足、全 NaN、单值、无自变量等，
验证了实现的健壮性。

---

## 六、待用户确认的优化建议

以下优化方向已验证可行，**待用户确认后**可合并到 main 分支：

1. **因子引擎**：用 `vectorized_ic` / `vectorized_neutralize` 替换现有逐日循环实现，
   保持接口不变，获得 11.99x~2.49x 性能提升。
2. **因子引擎**：在 IC 分析与融合前增加 `preprocess_factor`（去极值 + 标准化）预处理步骤，
   提升因子质量与融合稳健性。
3. **回测引擎**：在 `BaseBacktestMetrics.calc_all_metrics` 中集成增强指标
   （turnover / alpha / beta / IR / max_drawdown_duration）。
4. **回测引擎**：在回测入口增加 `look_ahead_guard` 前视偏差校验，提升回测可信度。

### 后续可探索方向（本次未实现，需更大改动）

- **PIT 数据层**：借鉴 Qlib Point-in-Time 设计，重构 data-engine 数据存储，从根本上避免前视偏差。
- **因子缓存**：借鉴 Qlib 多级缓存，避免重复计算因子。
- **LLM 因子挖掘**：借鉴 FactorEngine / RD-Agent(Q)，引入 LLM 自动因子发现与模型协同优化。
- **声明式因子表达式**：借鉴 Qlib 表达式引擎，支持 `Ref($close, 20)/$close` 等声明式因子定义。

---

## 七、文件清单

```
optimizations/20260620/
├── VERIFICATION_REPORT.md              # 本报告
├── factor_engine_opt/
│   ├── __init__.py
│   ├── vectorized_ic.py                # 优化1：向量化 IC
│   ├── vectorized_neutralize.py        # 优化1：向量化中性化 + FWL
│   └── preprocessing.py                # 优化2：去极值 + 标准化
├── backtest_engine_opt/
│   ├── __init__.py
│   ├── enhanced_metrics.py             # 优化3：增强绩效指标
│   └── look_ahead_guard.py             # 优化3：前视偏差检测
└── tests/
    ├── __init__.py
    ├── test_vectorized_ic.py
    ├── test_vectorized_neutralize.py
    ├── test_preprocessing.py
    ├── test_enhanced_metrics.py
    ├── test_look_ahead_guard.py
    └── run_all_tests.py                # 本运行器
```
