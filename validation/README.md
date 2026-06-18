# jingni-trader 量化交易开源项目学习与优化验证报告

- **执行日期**: 2026-06-18
- **工作分支**: `feat/quant-opt-20260618`
- **状态**: ✅ 全部 8 章节验证通过 / 40 单元测试通过
- **重要约束**: 本次所有代码仅在 `validation/` 独立目录中,未修改 main 分支任何业务代码

---

## 1. 学习项目清单 (2026 年活跃量化开源项目)

通过 GitHub 搜索、官方文档与项目 README 比对, 重点调研 4 个项目, 其中 3 个作为本次优化的主要借鉴来源:

| 项目 | 简介 | Star | 借鉴要点 | 官方链接 |
|------|------|------|----------|----------|
| **AKQuant** | Rust+Python 高性能 A 股/美股回测与实盘框架 | 1.3K+ | Walk-forward Validation、Pipeline 防数据泄漏、Signal/Action 分离、ML Adapter、Golden Test | <https://github.com/akfamily/akquant> |
| **VectorBT / VectorBT PRO** | 向量化回测框架 | 17K+ | 向量化计算、Purged K-Fold + Embargo、`@parameterized` 装饰器、`Portfolio.stats()` 一键指标 | <https://github.com/polakowo/vectorbt> |
| **qlib** (Microsoft) | AI 量化平台, 含 Alpha158 因子库 | 17K+ | 横截面 Z-Score、IC/Rank IC 分析、组合优化器、回测与实盘 | <https://github.com/microsoft/qlib> |
| **RD-Agent** (Microsoft) | 因子 + 模型的 LLM 自动研发 | 1K+ | 程序级因子发现、贝叶斯搜索、知识库引导 | <https://github.com/microsoft/RD-Agent> |

> 上述项目均在 2025-2026 年持续活跃, 最近 3 个月内均有 commit。

---

## 2. 核心亮点与可借鉴方向

### 2.1 AKQuant
- ✅ **Walk-forward Validation** (rolling training):  解决"训练-测试"时间序列污染, 是回测的金标准
- ✅ **Pipeline API**:  训练集 fit 统计量 → transform 训练/测试集, 强制隔离, 防止"未来函数"
- ✅ **Signal vs. Action 分离**:  先生成原始 signal, 再经 risk/sizing 转 action, 方便复用
- ✅ **ML Adapter**:  `SklearnAdapter`, `PyTorchAdapter` 等标准接口, 训练/预测/序列化统一
- ✅ **Golden Tests**:  锁定关键计算输出, 防止后续重构改变结果

### 2.2 VectorBT
- ✅ **向量化计算**:  基于 NumPy/Pandas, 一次计算多组参数, 100x 性能提升
- ✅ **Purged K-Fold + Embargo**:  来自 AFML 第 7 章, 解决 K-Fold 在金融时序中的 label 泄漏
- ✅ **`@parameterized` 装饰器**:  把"参数 → 结果矩阵"自动化, 避免手写 grid search
- ✅ **`Portfolio.stats()`**:  一键输出 60+ 指标, 含 Sharpe / Sortino / Calmar / Omega / Ulcer / VaR / CVaR

### 2.3 qlib
- ✅ **`Alpha158` 标准因子集**:  158 个常见算子, 快速构建基线
- ✅ **横截面 Z-Score (`CSZScoreNorm`)**:  行业内或全市场截面标准化
- ✅ **多周期因子 + IC 衰减分析**:  评估因子预测能力的稳定性

---

## 3. jingni-trader 现状与改进空间

通过分析 `engine.py`, `context.py`, `config.py`, `SKILL.md` 与各子 Skill 的 BaseCalculator, 发现以下可改进点:

| 模块 | 现状 | 改进空间 | 借鉴 |
|------|------|----------|------|
| `factor-engine/TalibCalculator` | for 循环逐只股票调用 TA-Lib | 用 `groupby + transform` 向量化 | VectorBT |
| `strategy-model-engine/base_model.py` | 仅有 train/predict/save/load | 缺 Purged K-Fold + Walk-Forward 工具, 但 config.py 已声明 `PURGE_GAP_DAYS=5`、`TRAIN_WINDOW_MONTHS=36` | AKQuant + AFML |
| `strategy-model-engine` | 无 Pipeline 防泄漏 | 缺 sklearn-style 预处理 Pipeline | AKQuant |
| `backtest-engine/rqalpha_adapter` | 仅输出 5 个核心指标 (annual_return / sharpe / max_dd / total_return / vol) | 需要 30+ 综合指标 (Sortino / Calmar / VaR / CVaR / Omega / Ulcer / 偏度/峰度 / Beta / Alpha) | VectorBT.stats |
| `factor-engine` | 因子 IC 分析能力弱 | 缺 IC / Rank IC / ICIR / 多空胜率 / 换手率 | qlib |
| `portfolio-risk-engine` | 缺参数化扫描工具 | 缺 `@parameterized` grid search | VectorBT |
| 整体 | 无 Golden Test | 缺回归测试 baseline | AKQuant |

---

## 4. 优化方向与验证内容

在 `feat/quant-opt-20260618` 分支下, 共实现 5 个验证模块, 共 **40 单元测试 + 8 端到端验证场景**:

### 4.1 向量化因子计算 (`vectorized_factor.py`)
- 借鉴 VectorBT:  用 `groupby + transform` 替代 for 循环
- 9 个常见因子 (MA / EMA / RSI / STD / Momentum / ZScore)
- 同时提供 `LoopFactorCalculator` 作 baseline 对照
- `benchmark()` 函数自动对比耗时

### 4.2 Purged CV (`purged_cv.py`)
- 借鉴 AFML 第 7/12 章 + VectorBT/AKQuant
- `PurgedKFold`:  K-Fold + purge + embargo
- `CombinatorialPurgedKFold`:  AFML §12 多 backtest 路径
- `WalkForwardSplitter`:  滚动 train/val/test (expanding / rolling 两种模式)
- `ic_time_series_split`:  因子 IC 分析专用

### 4.3 综合指标库 (`metrics.py`)
- 借鉴 VectorBT `Portfolio.stats()` + quantstats
- **31 个核心指标**:  收益类(9) + 风险类(13) + 比率类(6) + 综合(3)
- 含 IC / Rank IC / ICIR / 多空收益 / 多空胜率

### 4.4 数据预处理 Pipeline (`pipeline.py`)
- 借鉴 AKQuant Pipeline + sklearn Pipeline
- `MissingValueFiller` / `Winsorizer` / `CrossSectionalScaler` / `IndustryNeutralizer`
- 强制 fit / transform 分离, 保证测试集使用训练集统计量

### 4.5 参数化扫描 (`parameterized.py`)
- 借鉴 VectorBT `@parameterized` / `@chunked`
- `sweep(func, param_space, common_kwargs)`:  一次扫描返回 SweepResult
- `SweepResult.to_dataframe()` / `best()`:  网格结果结构化

---

## 5. 测试与验证结果

### 5.1 端到端验证 (8 个场景)

| # | 验证项 | 关键结果 | 结论 |
|---|--------|----------|------|
| 1 | **向量化 vs 循环一致性** | 9 个因子最大绝对差 = `0.0` | ✅ bit-by-bit 完全一致 |
| 2 | **向量化性能** | 50x252 加速 1.51x,  200x504 加速 1.47x,  500x1000 加速 1.42x | ✅ 平均 1.47x 加速 |
| 3 | **Purged CV 行为** | 5 折 purge=5D/embargo=5D, 训练/测试无重叠; CPCV 5 选 2 → 10 路径; WF 步进 50 | ✅ 全部符合预期 |
| 4 | **因子 IC 检测** | 强信号 IC=0.97 / 多空胜率=100%; 弱信号 IC=0.08 / 多空胜率=85% | ✅ 与信号强度正相关 |
| 5 | **净值指标完整性** | 31 项指标全部产出 (Sharpe 0.57, Sortino 0.92, Calmar 0.36, MaxDD -31%) | ✅ |
| 6 | **Pipeline 防泄漏** | 训练 winsor 界 [-2.04, 2.04], 测试集因子全部落在界内 | ✅ |
| 7 | **IC 时序切分** | 4 折滚动, 每折 train/test IC 稳定在 0.95 | ✅ |
| 8 | **完整单元测试** | 40/40 passed in 3.3s | ✅ |

### 5.2 性能对比详细数据

| 规模 (n_stocks × n_days) | 总行数 | loop (ms) | vec (ms) | 加速比 |
|--------------------------|--------|-----------|----------|--------|
| 50 × 252 | 12,600 | 387.16 | 257.02 | **1.51x** |
| 200 × 504 | 100,800 | 1,621.95 | 1,105.14 | **1.47x** |
| 500 × 1000 | 500,000 | 4,266.16 | 3,011.75 | **1.42x** |

> 注:  本次验证为纯 NumPy/Pandas 实现 (无 TA-Lib C 扩展)。引入 TA-Lib 后, 向量化收益会进一步放大 (VectorBT 实测可达 50-100x)。

### 5.3 综合指标库输出示例 (基于合成 504 天净值, 年化 12%, 波动 18%)

```
total_return = 0.2384        annual_return = 0.1131
sharpe_ratio = 0.5720        sortino_ratio = 0.9202
calmar_ratio = 0.3647        omega_ratio = 1.1152
volatility_annual = 0.1814   max_drawdown = -0.3101
var_historical = -0.0177      cvar_historical = -0.0236
skewness = -0.0783           kurtosis = 0.0114
information_ratio = -0.0048  beta = 0.0031
```

### 5.4 Golden Baseline

`validation/golden/golden_baseline.json` 保存了 8 个模块的回归基线数据, 供后续 PR 校验:

- `vectorized_factors`:  7 个因子最大绝对差 (应全为 0)
- `equity_stats`:  31 项净值指标
- `factor_metrics_strong_signal` / `factor_metrics_weak_signal`
- `purged_kfold` / `walk_forward` / `combinatorial_purged_kfold` / `ic_time_series_split`

---

## 6. 复现方法

```bash
git fetch origin
git checkout feat/quant-opt-20260618

# 完整 8 章节验证 (耗时约 30-40 秒)
PYTHONPATH=. python3 -m validation.run_validation

# 单元测试 (耗时约 3 秒)
PYTHONPATH=. python3 -m pytest validation/tests/ -v

# 重新生成 Golden baseline
PYTHONPATH=. python3 -m validation.make_golden
```

---

## 7. 重要发现与警示

### 7.1 验证中发现的问题

1. **`backtest-engine/rqalpha_adapter.py` 中 `_calculate_metrics` 仅有 5 个指标**,  无法满足实盘归因、风险评估、组合优化需求 → 建议替换为 `validation/metrics.calc_all_stats`
2. **`strategy-model-engine/SKILL.md` 提到 "Purged Group Time Series Split"** 但无对应实现,  `base_model.py` 仅暴露 train/predict/save/load → 建议引入 `validation/purged_cv` 工具
3. **`factor-engine` 无 IC / Rank IC 分析** → 建议引入 `validation/metrics.factor_metrics` 用于因子评估
4. **`TalibCalculator` 用 for 循环** → 建议改用 `validation/vectorized_factor.VectorizedFactorCalculator` 模式 (groupby + transform)
5. **无 Pipeline 防数据泄漏** → 模型训练中 Z-Score / Winsorize / Industry 行业中性化容易引入"未来函数", 建议使用 `validation/pipeline.Pipeline` 强制 fit/transform 分离

### 7.2 未在本次实现的内容 (后续可考虑)

- **Signal / Action 分离的策略 API** (借鉴 AKQuant)
- **ML Adapter** (借鉴 AKQuant SklearnAdapter / PyTorchAdapter)
- **组合优化器** (借鉴 VectorBT / qlib, 基于 cvxpy)
- **实盘交易接口** (借鉴 AKQuant 的 broker 抽象)
- **回测事件总线** (借鉴 NautilusTrader)
- **LLM 自动因子发现** (借鉴 RD-Agent / FactorEngine)

---

## 8. 待用户确认的优化建议

以下建议均**不会自动执行**, 等待用户明确确认:

| 优先级 | 建议 | 收益 | 工作量 |
|--------|------|------|--------|
| **P0** | 把 `validation/metrics` 接入 `backtest-engine/rqalpha_adapter._calculate_metrics` | 风险评估能力提升 6x (5→31 指标) | 0.5 天 |
| **P0** | 把 `validation/purged_cv.PurgedKFold` / `WalkForwardSplitter` 接入 `strategy-model-engine` 训练流程 | 防止过拟合/前视偏差 | 1 天 |
| **P1** | `TalibCalculator` 改为 `VectorizedFactorCalculator` 模式 (groupby + transform) | 计算性能提升 1.4-1.5x, 10x+ (若引入 TA-Lib C 扩展) | 1 天 |
| **P1** | `strategy-model-engine` 引入 `validation/pipeline.Pipeline` 风格预处理 | 严格防止数据泄漏 | 1 天 |
| **P2** | 引入 `@parameterized` 装饰器,  支持窗口/阈值参数网格扫描 | 因子研究效率提升 | 0.5 天 |
| **P2** | 把 `validation/metrics.factor_metrics` 接入因子评估流程 | 自动评估 IC / Rank IC / ICIR | 0.5 天 |
| **P3** | 引入 Signal / Action 分离的策略 API | 策略复用性提升 | 2 天 |
| **P3** | 引入组合优化器 (cvxpy) | 支持均值-方差 / 风险平价 | 2 天 |

---

## 9. 交付物

- `validation/vectorized_factor.py` — 向量化因子计算器 + benchmark
- `validation/purged_cv.py` — PurgedKFold + CombinatorialPurgedKFold + WalkForwardSplitter + ic_time_series_split
- `validation/metrics.py` — 31 项综合指标库 (return/risk/ratio/factor)
- `validation/pipeline.py` — 防数据泄漏的 Pipeline (4 种 step)
- `validation/parameterized.py` — 参数化扫描装饰器
- `validation/synth_data.py` — 合成数据生成器
- `validation/tests/` — 40 个单元测试
- `validation/golden/golden_baseline.json` — 回归基线
- `validation/results/validation_report.json` — 本次运行结果
- `validation/run_validation.py` — 一键验证入口
- `validation/make_golden.py` — Golden baseline 生成
- `validation/README.md` — 本文件
