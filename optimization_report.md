# Quant Optimization Verification Report

**Branch:** `feat/quant-opt-20260617`
**Date:** 2026-06-17
**Scope:** jingni-trader (A股量化交易全流程主调度器)
**Status:** ✅ 16/16 tests pass — ready for user review (no merge to `main`)

---

## 1. 学习项目清单及核心亮点

本次主要研究并借鉴了以下三个具有代表性的开源项目：

### 1.1 Microsoft **Qlib** (`microsoft/qlib`, ~44k★)

| 模块 | 借鉴要点 |
|------|----------|
| `qlib.data.ops` 表达式引擎 | 把因子定义为 `MA(Close, 20) - MA(Close, 5)` 的纯函数式表达式，而非命令式 pandas 代码 |
| `DataHandler` + processors | 因子值的标准化、缺失值处理与缓存分离，Data Layer 与 Feature Layer 解耦 |
| `Alpha158` / `Alpha360` | 标准化因子集，让研究结果可复现、可对比 |
| `RD-Agent` (2024) | LLM 驱动的自动因子挖掘思路 |

**亮点：** 表达式引擎是 Qlib 最被工业界复用的部分，几乎所有内部研究模块都基于 `qlib.data.ops` 编写。

### 1.2 **AKQuant** (`cloudQuant/akquant`, 2026 活跃)

| 模块 | 借鉴要点 |
|------|----------|
| `akquant.factor` 因子表达式引擎 | Polars 驱动的 Alpha101 DSL，`Rank(Ts_Mean(Close, 5))` 风格 |
| `akquant.ml` 训练框架 | Walk-Forward Validation + 信号/动作分离设计 |
| `akquant.akfamily.xyz` 文档 | A股特定：涨跌停、停牌、T+1 规则的工程化处理 |

**亮点：** walk-forward 中明确把"模型只产 signal"和"signal→order"两段解耦，规避了把过拟合的阈值规则泄漏到训练集。

### 1.3 **VeighNa vnpy.alpha** (`vnpy/vnpy`, 4.0 在 2025 重写)

| 模块 | 借鉴要点 |
|------|----------|
| 四层架构 dataset / model / strategy / lab | 把投研流水线模块化，每层接口稳定 |
| `AlphaDataset` + `AlphaModel` | train/valid/test 段切割、模型统一接口 |
| 内置 Alpha158 + Alpha101 | 标准化因子库 + 行业/动量/价值 6 维划分 |

**亮点：** 4.0 重写后与 Qlib 思路完全一致，是中文社区最易被 A股研究员复用的实现。

---

## 2. 可借鉴的方向列表（按 ROI 排序）

| # | 借鉴方向 | 来源 | jingni-trader 现状 | 改进空间 |
|---|---------|------|-------------------|---------|
| 1 | **声明式因子表达式引擎** | Qlib / AKQuant | `compute_a_share_factors` 写死 ~15 个因子，新因子需改 Python 源码 | **本次已实现** — 用户可写 `Rank(MA($close, 5) - MA($close, 20))` |
| 2 | **Walk-Forward Validation 框架** | AKQuant | `purged_group_ts_split` 仅做一次性切分，没有滚动训练 + embargo | **本次已实现** — 完整 rolling / expanding + purge_gap + embargo |
| 3 | **Alpha158 标准化因子库** | Qlib / vnpy.alpha | 缺乏标准化因子库，因子间无法直接对比 | **本次已实现** — 38 个因子，含 A股本土化扩展（换手率、量比） |
| 4 | **信号 / 动作分离** | AKQuant | `compute_a_share_signals` 与 `signal_to_action` 耦合 | 建议下一阶段改造 |
| 5 | **统一模型接口 (AlphaModel)** | vnpy.alpha | 不同子模型各写一套适配代码 | 建议下一阶段改造 |
| 6 | **Polars / DuckDB 后端** | AKQuant | 全部基于 pandas | 待评估（数据量级 ≤10k 行时收益有限） |

---

## 3. 已实现的验证 1：声明式因子表达式引擎

**文件：**
```
quant_opt/factor_expression/
├── __init__.py
├── parser.py         # Pratt 顶层解析器 + Tokenizer
├── operators.py      # 25 个内置算子（ts / cs / el 三组）
└── engine.py         # 算子调度 + 缓存
```

**核心 API：**
```python
from quant_opt.factor_expression import FactorEngine
engine = FactorEngine()
out = engine.calc(df, "Rank(MA($close, 5) - MA($close, 20))")
out = engine.calc_many(df, ["MA($close, 5)", "Rank(Delta($close, 5))"], ["fast", "slow"])
```

**亮点：**
- 兼容 Qlib `MA / Ref / Std / Rank / Zscore` 命名
- 兼容 AKQuant `Ts_Mean / CsRank` 命名
- 子表达式缓存（Qlib `FeatureRowName` 模式）
- 自定义算子注册接口

**关键测试：**

| 用例 | 验证内容 | 结论 |
|------|---------|------|
| `engine_ref_matches_baseline` | `Ref(x, 1)` 与 `groupby.shift(1)` bit-级一致 | ✅ 通过 |
| `engine_ma_matches_baseline` | `MA(x, 5)` 与 `groupby.rolling(5).mean()` bit-级一致 | ✅ 通过 |
| `engine_rank_matches_baseline` | `Rank(x)` 与 `groupby.rank(pct=True)` bit-级一致 | ✅ 通过 |
| `engine_alpha101_combo` | 复合 Alpha101 表达式结果与手写 baseline 一致 | ✅ 通过 |
| `engine_calc_many_with_cache` | 3 个表达式共享子节点缓存 | ✅ 通过 |
| `engine_perf_baseline` | 5 个复合表达式 × 6400 行 < 1.5s | ✅ 实际 58ms |

---

## 4. 已实现的验证 2：Walk-Forward Validation 框架

**文件：**
```
quant_opt/walk_forward/
└── __init__.py     # WalkForwardConfig / WalkForward / WindowResult / WalkForwardResult
```

**核心 API：**
```python
from quant_opt.walk_forward import WalkForward, WalkForwardConfig
cfg = WalkForwardConfig(train_period=252*2, val_period=63, step=63,
                        purge_gap=5, embargo=5,
                        feature_cols=[...], label_col="forward_5d")
wf = WalkForward(model_factory=lambda: Ridge(alpha=1.0))
result = wf.run(df, cfg)
print(result.summary)        # mean_ic / ic_ir / n_windows
print(result.signals.head()) # 逐日 (code, date, signal, label, window_id)
```

**亮点：**
- **rolling** + **expanding** 两种训练窗口模式
- **purge_gap**（训练/验证间空窗）防 look-ahead bias
- **embargo**（验证后空窗）防自相关泄漏
- 任意 sklearn 兼容模型（model_factory 模式）
- 信号 / 动作分离（AKQuant 风格）
- 自动输出 `signals` DataFrame，可直接喂给下一阶段 backtest

**关键测试：**

| 用例 | 验证内容 | 结论 |
|------|---------|------|
| `walk_forward_runs_and_emits_signals` | 基本执行 + 产出正确的 signals 格式 | ✅ 通过 (2 windows, mean_ic=-0.052) |
| `walk_forward_embargo_skips_overlap` | embargo 越大窗口越少 | ✅ 通过 (embargo=0 → 14, embargo=60 → 4) |
| `walk_forward_expanding_mode` | expanding 模式下训练集逐窗增大 | ✅ 通过 (4 windows) |
| `walk_forward_aggregate_ic_method` | IC/IR 聚合指标正确 | ✅ 通过 |

---

## 5. 已实现的验证 3：Alpha158 风格因子库

**文件：**
```
quant_opt/alpha158/
└── __init__.py     # 38 个因子 + FactorLibrary 注册器
```

**覆盖维度：**

| Family | 数量 | 示例 |
|--------|-----|------|
| momentum | 6 | `ret_1d`, `ret_5d`, `ret_20d`, `reversal_5d` |
| ma | 5 | `ma_bias_5`, `ma_bias_20`, `ma_cross_5_20` |
| volatility | 5 | `std_5`, `std_20`, `range_5` |
| volume | 4 | `volume_ma_5`, `volume_ratio_5_20` |
| cross_section | 5 | `rank_close`, `rank_volume`, `rank_ret_20d` |
| alpha101 | 7 | `alpha_001`, `alpha_006`, `alpha_021` |
| a_share（本土化） | 6 | `turnover_ma_5`, `intraday_ret`, `hl_range_1` |
| **合计** | **38** | |

**亮点：**
- 全部以 **声明式表达式字符串** 存储（`FactorSpec.expression`），可与 `FactorEngine.calc_many` 无缝对接
- 包含 `direction` 字段（+1 / -1 / 0），便于后续做因子有效性自动评估
- `FactorLibrary.register()` 允许用户/团队扩展自己的因子族

**关键测试：**

| 用例 | 验证内容 | 结论 |
|------|---------|------|
| `alpha158_library_runs` | 库包含 ≥30 因子，前 10 个可在 200 天面板上全部算出 | ✅ 38 factors, 10 computed |
| `alpha101_alpha_001_matches_baseline` | Alpha#001 公式与手写 baseline bit-级一致 | ✅ 通过 |
| `integration_engine_plus_library_plus_wf` | **端到端**：library → engine → walk-forward | ✅ mean_ic=0.033 |

---

## 6. 端到端集成测试

```
[PASS] integration_engine_plus_library_plus_wf        156ms  end-to-end OK, 2 windows, mean_ic=0.033
```

完整链路：
1. 从 `FactorLibrary` 取出 6 个因子表达式
2. 用 `FactorEngine.calc_many` 在 40 支股票 × 180 天面板上算出 6 列特征
3. 构造 5 日 forward-return 标签
4. `WalkForward.run` 用 80/20 滚动窗口 + 中位数填充 + Ridge 训练
5. 输出 IC / IR

---

## 7. 性能对比

| 操作 | 行数 | 耗时 | 备注 |
|------|------|------|------|
| 5 个复合表达式（最大 20 步） | 6,400 | 58ms | < 1.5s 阈值 |
| 单次 factor library 算 10 因子 | 2,000 | 28ms | - |
| Walk-Forward 2 窗口 × Ridge | 6,000 | 113ms | - |
| 端到端（6 因子 + 2 窗口） | 7,200 | 156ms | - |

---

## 8. 待用户确认的优化建议

> ⚠️ 以下建议**不**自动执行；需用户明确确认后才合入 `main`。

| # | 建议 | 工作量 | 风险 |
|---|------|-------|------|
| A | 把 `compute_a_share_factors` 重构为调用 `FactorEngine`，对外保持兼容 | 中 | 低 — 引擎是纯增量 |
| B | 在 `model-engine` 中把 hard-coded 模型列表改为 `WalkForward` 工厂模式 | 中 | 中 — 需配合每个模型的重写 |
| C | 在 `backtest-engine` 的回测主循环里改用 `WalkForward.signals` 作为 signal 源 | 中 | 中 — 需改 backtest 协议 |
| D | 将 `Alpha158_LIKE` 因子库注入 `factor-engine` 的默认发现列表 | 小 | 低 |
| E | 在 CI 中加 `quant_opt/tests/run_all.py` 作为必跑套件 | 小 | 极低 |

**推荐优先落 A + D**：A 把研究入口与表达式引擎绑定（用户体感最强），D 把标准化因子库开放给所有下游模块。

---

## 9. 文件清单

```
quant_opt/                                    # 新增包，约 970 行代码
├── __init__.py                               # 23 行
├── factor_expression/
│   ├── __init__.py                           # 23 行
│   ├── parser.py                             # 162 行 — Pratt 解析器
│   ├── operators.py                          # 247 行 — 25 个算子
│   └── engine.py                             # 234 行 — 调度 + 缓存
├── walk_forward/
│   └── __init__.py                           # 269 行 — rolling/expanding + purge + embargo
├── alpha158/
│   └── __init__.py                           # 199 行 — 38 因子 + 注册器
├── tests/
│   ├── __init__.py                           # 470 行 — 16 个测试
│   └── run_all.py                            # 19 行 — 独立 runner
└── reports/
    └── 20260617_optimization_report.md       # 本报告
```

---

## 10. 测试运行方式

```bash
cd /data/user/skills/jingni-trader
python3 quant_opt/tests/run_all.py
# === 16/16 passed ===
```

---

## 11. Git / 分支状态

| 项目 | 状态 |
|------|------|
| 仓库初始化 | ✅ `git init` + baseline commit `dd49151` 在 `main` |
| 新分支 | ✅ `feat/quant-opt-20260617`（基于 `main`） |
| 工作分支提交 | ✅ `3476be1` + `d21a136` 共 2 个 commit |
| 远程 `origin` | 已配置为 `https://github.com/duhanjun/jingni-trader.git` |
| `git push origin feat/quant-opt-20260617` | ⚠️ **沙箱无 GitHub 凭据，push 失败**（`could not read Username`） |
| 离线补丁 | ✅ `quant_opt/reports/feat-quant-opt-20260617.patch`（2181 行）已生成并提交 |

**用户可选推送方式：**

```bash
# 方式 1：使用 gh CLI（推荐，需要 GH_TOKEN）
GH_TOKEN=<your_token> gh auth login --with-token
cd /data/user/skills/jingni-trader
git push -u origin feat/quant-opt-20260617

# 方式 2：直接推送（输入凭据）
cd /data/user/skills/jingni-trader
git push -u origin feat/quant-opt-20260617

# 方式 3：通过 patch 应用到已存在的 fork/clone
git apply quant_opt/reports/feat-quant-opt-20260617.patch
git push -u <your-remote> feat/quant-opt-20260617
```

---

**本分支仅为实验性优化，所有变更位于 `quant_opt/` 子目录，未触及 `main` 上的任何现有文件。**
**`feat/quant-opt-20260617` 不会自动合入 `main`；等待用户明确确认。**
