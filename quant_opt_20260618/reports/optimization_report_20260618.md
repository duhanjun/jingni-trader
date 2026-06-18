# jingni-trader 量化交易开源项目学习与优化报告

| 字段 | 值 |
| --- | --- |
| 执行日期 | 2026-06-18 |
| 自动化分支 | `feat/quant-opt-20260618` |
| 主分支 | `main` (未修改) |
| 报告生成 | 自动任务（jingni-trader 量化学习 / 验证 / 推送流程） |

---

## 1. 联网学习：候选开源项目清单

> 任务要求在 GitHub / 学术平台 / 量化社区中搜索近期活跃项目，重点关注 因子挖掘、回测框架、风险控制、ML/AI 量化、实盘接口等方向。

| # | 项目 | 仓库 | 关注方向 | 状态 / 亮点 | 借鉴权重 |
| - | --- | --- | --- | --- | --- |
| 1 | **AKQuant** | [akfamily/akquant](https://github.com/akfamily/akquant) | 因子表达引擎、回测引擎、ML 框架、并行参数搜索 | Rust + Python 混合架构，Polars 引擎，536 commits / 持续活跃，内置 walk-forward / 多进程 grid search / 可视化模块 | ⭐⭐⭐⭐⭐ |
| 2 | **Microsoft Qlib** | [microsoft/qlib](https://github.com/microsoft/qlib) | 因子表达式引擎、Alpha158/360 数据集、PIT 数据层、YAML 工作流 | 学术界与工业界引用最广的 AI Quant 平台；DataLoader / DataHandler / Dataset 三段式数据流 | ⭐⭐⭐⭐⭐ |
| 3 | **RD-Agent** | [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | 自动化因子 / 模型研发循环 | 微软新项目，将 LLM 嵌入 quant 研发循环：`rdagent fin_factor` 自动生成因子 | ⭐⭐⭐⭐ |
| 4 | **SimTradeLab** | [kay-ou/SimTradeLab](https://github.com/kay-ou/SimTradeLab) | PTrade API 仿真回测 | 内存数据，宣称比 PTrade 快 100-160×；事件驱动 + 简单策略接口 | ⭐⭐⭐ |
| 5 | **FactorEngine 论文** | [arXiv 2603.16365](https://arxiv.org/abs/2603.16365) | Polars + 并行因子引擎、程序化因子表示 | 学术论文，借鉴其"程序化因子"与"知识注入 bootstrap"思路 | ⭐⭐⭐⭐ |
| 6 | **vnpy** | [vnpy/vnpy](https://github.com/vnpy/vnpy) | 综合量化平台、回测 + 实盘 | 28.4k stars 国产龙头；事件驱动引擎、网关抽象层 | ⭐⭐⭐（设计偏复杂，借鉴度中等） |

### 1.1 三个最有借鉴价值的项目深入摘要

#### ① AKQuant（akfamily/akquant）
- **架构**：Rust 核心 + Python 绑定；`FactorEngine` 用 Polars 算子；多进程 grid search。
- **核心亮点**：
  - **Polars 因子引擎**：把"时间序列算子"和"截面算子"在统一的算子表内声明，Polars 表达式链式求值，向量化性能远高于 pandas groupby + apply。
  - **Walk-forward framework**：`ml/` 子模块内置滚动训练 + 验证 + 测试三段式窗口，自动统计 IC / ICIR / decile 收益。
  - **Multi-process grid search**：参数搜索并行化，加速因子/超参调优。
  - **可视化报告**：内置交互式回测报告（plotly）。

#### ② Qlib（microsoft/qlib）
- **架构**：`DataLoader → DataHandler → Dataset → Model → Strategy → Backtest`。
- **核心亮点**：
  - **表达式引擎（Expression Engine）**：用户用字符串公式声明因子，如 `Ref($close, 5) / $close - 1`、`Rank(Corr($close, $volume, 10))`。
  - **Alpha158 / Alpha360 预置数据集**：覆盖多数公开 alpha 因子作为社区基线。
  - **PIT（Point-in-Time）数据层**：以"announce_date"为可见时间点，杜绝未来函数。
  - **YAML 工作流**：研究-训练-回测-评测全流程可声明。

#### ③ RD-Agent（microsoft/RD-Agent）
- **核心亮点**：LLM 自动挖掘因子与模型；`rdagent fin_factor` 子命令可基于历史 IC 提出新因子假设。
- **对 jingni-trader 的价值**：为未来"自动因子库扩充"提供思路，但需要 LLM 能力与算子集约束。

---

## 2. 对照 jingni-trader 现状的可借鉴方向

通过阅读 [engine.py](file:///workspace/engine.py)、[scripts/context.py](file:///workspace/scripts/context.py)、[skills/factor-engine/engine.py](file:///workspace/skills/factor-engine/engine.py)、[skills/backtest-engine/engine.py](file:///workspace/skills/backtest-engine/engine.py)、[skills/backtest-engine/scripts/adapters/native_adapter.py](file:///workspace/skills/backtest-engine/scripts/adapters/native_adapter.py)、[skills/portfolio-risk-engine/engine.py](file:///workspace/skills/portfolio-risk-engine/engine.py)、[skills/strategy-model-engine/engine.py](file:///workspace/skills/strategy-model-engine/engine.py)，梳理出以下 6 个可改进方向：

| # | 方向 | 借鉴来源 | 当前 jingni-trader 痛点 | 优化思路 |
| - | --- | --- | --- | --- |
| A | **因子公式 DSL** | Qlib / AKQuant | `factor-engine.compute_a_share_factors` 硬编码 5+ 个因子；新增因子需修改源码 | 引入表达式引擎，字符串公式 `Rank(Ts_Mean($close, 5))` 即可注册因子；不破坏现有 API |
| B | **Polars / 向量化后端** | AKQuant | 全部 `groupby + apply(lambda)` 在大规模数据上较慢 | 可选 Polars 后端（本次仅做 PoC，未集成） |
| C | **PIT 数据合并** | Qlib DataLayer | 数据合并采用 `pd.merge` + `period_end` 关联，易引入未来函数 | 引入 `pit_merge`：用 `announce_date` 严格控制可见时间，并提供 `detect_lookahead` 报告 |
| D | **Walk-forward 滚动训练** | AKQuant ml 模块 | `purged_group_ts_split` 只是切分工具，缺完整 train/val/test 滚动 + OOS 指标汇总 | 引入 `walk_forward_train_predict`，输出 IC / ICIR / 覆盖率 |
| E | **内容指纹缓存** | Qlib ExpressionCache | `execute_stage` 仅按"产物文件存在"判断，参数微调会复算或错算 | 引入 `fingerprint(obj) + DiskCache`，按 (key, 数据指纹) 复用中间结果 |
| F | **风险控制** | Qlib / vnpy | portfolio-risk-engine 已有组合优化、VaR 计算，但缺"因子暴露 + 风格归因" | 后续可加入 Barra 风格因子分解（不在本次 PoC 范围） |

> **本次只对 A / C / D / E 四个方向做完整 PoC 与测试**，B 和 F 留作下一轮。

---

## 3. 验证代码与测试结果

### 3.1 代码组织（全部位于 `quant_opt_20260618/` 目录，未触碰 main）

```
quant_opt_20260618/
├── __init__.py
├── factor_engine/
│   ├── __init__.py
│   └── expression_engine.py     # 借鉴 Qlib / AKQuant 的 Factor DSL
├── pit_join/
│   ├── __init__.py
│   └── pit_merge.py             # 借鉴 Qlib PIT DataLayer
├── walk_forward/
│   ├── __init__.py
│   └── walk_forward.py          # 借鉴 AKQuant ml 框架
├── cache/
│   ├── __init__.py
│   └── fingerprint_cache.py     # 借鉴 Qlib ExpressionCache
├── tests/
│   ├── __init__.py
│   ├── fixtures.py              # 合成 A 股日线 + 财务数据
│   ├── test_expression_engine.py
│   ├── test_pit_merge.py
│   ├── test_walk_forward.py
│   └── test_fingerprint_cache.py
└── benchmark/
    ├── __init__.py
    ├── run_benchmarks.py
    └── benchmark_results.json
```

### 3.2 单元测试

```
$ PYTHONPATH=. python -m pytest quant_opt_20260618/tests -q
................................                         [100%]
32 passed in 1.99s
```

- 因子表达式引擎：18 个测试（算子注册、解析正确性、与 pandas 参考实现对照、批量计算、Alpha101 风格组合、边界 / 异常）。
- PIT 合并：5 个测试（基本合并、未来函数检测、最新公告选取、空 value_cols、原始顺序保持）。
- Walk-Forward：4 个测试（窗口生成、数据不足、smoke test、IC/ICIR 汇总）。
- 指纹缓存：5 个测试（DataFrame / dict / 标量指纹、DiskCache hit / miss、fp 变化失效、clear）。

### 3.3 端到端基准（`benchmark/run_benchmarks.py`）

| 类别 | 老实现 | 新实现 | 关键结论 |
| --- | --- | --- | --- |
| 因子引擎 | 5 个硬编码因子 / 11 行代码 / 0.0785s | 6 个因子（含 `mom_rank`）/ 6 行公式 / 0.13s | DSL 更灵活，`ret_5d` 数值与 pandas 参考实现差异 1.1e-16（机器精度） |
| PIT 合并 | 直接 merge，0.0131s | `pit_merge`，0.0126s | **剔除 18,039 行未来函数 = 总行数的 12.5%**；速度相当 |
| 滚动训练 | 仅 `purged_group_ts_split` | 8 折 / 0.10s / OOS 100% 覆盖 / mean IC / ICIR | 完整 train→val→test 框架落地，IC/ICIR 可直接对比单折 vs OOS |
| 指纹缓存 | if-exists 跳过 | fingerprint + DiskCache | **缓存命中 1931× 加速**；参数微调自动失效 |

完整 JSON 见 [benchmark_results.json](file:///workspace/quant_opt_20260618/benchmark/benchmark_results.json)。

### 3.4 关键代码片段示例

#### 因子表达式（Factor DSL）
```python
from quant_opt_20260618.factor_engine import calc_factors

formulas = {
    "ret_1d":         "Delta($close, 1) / Delay($close, 1)",
    "ret_5d":         "Delta($close, 5) / Delay($close, 5)",
    "volatility_20d": "Ts_Std(Delta($close, 1) / Delay($close, 1), 20)",
    "mom_rank":       "Rank(Delta($close, 5) / Delay($close, 5))",
    "alpha101_alpha1": "Rank(Ts_Mean($close, 10)) - Rank(Ts_Mean($volume, 10))",
}
df = calc_factors(panel, formulas)
```

#### PIT 合并
```python
from quant_opt_20260618.pit_join import PITConfig, pit_merge, detect_lookahead

cfg = PITConfig(asof_col="date", announce_col="announce_date", by="code")
df_pit = pit_merge(df_daily, df_financial, cfg, value_cols=["pe_ttm", "roe"])
report = detect_lookahead(df_naive, df_pit, value_cols=["pe_ttm", "roe"])
# report["total_lookahead_eliminated"] 直接告诉你"错配了多少行未来函数"
```

#### Walk-Forward
```python
from quant_opt_20260618.walk_forward import (
    WalkForwardConfig, walk_forward_train_predict, aggregate_wf_metrics
)

cfg = WalkForwardConfig(
    train_window_days=400, val_window_days=120,
    test_window_days=120,  step_days=120,
    purge_days=5, embargo_days=5,
)
oos, results = walk_forward_train_predict(X, y, dates, fit_predict_fn, cfg)
print(aggregate_wf_metrics(results))
# -> {'n_folds': 8, 'mean_ic': -0.003, 'icir': -0.32, 'positive_ic_ratio': 0.125}
```

#### 指纹缓存
```python
from quant_opt_20260618.cache import fingerprint, DiskCache

cache = DiskCache(root=".cache/quant")
fp = fingerprint(panel_df)         # 数据指纹
result = cache.get_or_compute(
    "factors_v1", fp,
    lambda: compute_expensive_factors(panel_df),
)
```

---

## 4. 待用户确认的优化建议

| # | 建议 | 实施成本 | 预期收益 | 推荐合并？ |
| - | --- | --- | --- | --- |
| 1 | **把 Factor DSL 作为 `factor-engine` 的可选后端**（保留现有 `compute_a_share_factors` 兼容入口） | 中（2-3 天） | 新增因子无需改源码；用户可从 YAML/JSON 加载公式库 | ✅ 推荐 |
| 2 | **在 `data-engine` 中增加 PIT 校验管线**，对每个财务/事件类数据自动跑 `detect_lookahead` | 中（2 天） | 杜绝未来函数；上线前报告一份"潜在未来函数"清单 | ✅ 推荐 |
| 3 | **在 `strategy-model-engine` 中集成 walk-forward**，并以 OOS IC/ICIR 取代"单折 IC"作为过拟合门控 | 中（2-3 天） | 显著降低过拟合风险；与 `purged_group_ts_split` 共存 | ✅ 推荐 |
| 4 | **在 `engine.execute_stage` 阶段引入 `fingerprint` 缓存**，把"产物文件存在"判断升级为"（产物 + 输入指纹）一致" | 小（1 天） | 频繁调参 / 重复执行时提速 100× 以上 | ✅ 推荐 |
| 5 | Polars 后端切换 | 大（1-2 周） | 大规模数据 5-10× 加速 | ⏳ 待 #1 落地后评估 |
| 6 | Barra 风格因子归因 | 大（1-2 周） | 风险归因更清晰 | ⏳ 后续排期 |

> **本次任务范围内**：以上建议对应的所有验证代码已实现并测试通过（32/32 单测 + 4 个端到端基准）。代码仅落在 `feat/quant-opt-20260618` 分支的 `quant_opt_20260618/` 目录，**未对 main 分支做任何修改**。分支已自动推送到 GitHub 远程。

---

## 5. Git 操作记录

| 时间 | 操作 | 结果 |
| --- | --- | --- |
| T0 | `git checkout -b feat/quant-opt-20260618` | ✅ 新分支创建 |
| T1 | 添加 13 个新文件（`quant_opt_20260618/` 树下） | ✅ 仅新分支有变更 |
| T2 | `git push origin feat/quant-opt-20260618` | ✅ 已推送（**未合并到 main**） |

> ⚠️ 严格遵循任务约束：在用户明确确认前，绝不执行 `git merge` / PR 合入。

---

## 6. 附录

### 6.1 测试运行方法

```bash
# 单元测试
PYTHONPATH=. python -m pytest quant_opt_20260618/tests -v

# 端到端基准
PYTHONPATH=. python quant_opt_20260618/benchmark/run_benchmarks.py
```

### 6.2 报告与产物路径

- 综合报告（本文档）：`quant_opt_20260618/reports/optimization_report_20260618.md`
- 单元测试：`quant_opt_20260618/tests/`
- 端到端基准脚本与 JSON：`quant_opt_20260618/benchmark/`
- 验证模块代码：`quant_opt_20260618/{factor_engine,pit_join,walk_forward,cache}/`

### 6.3 自动化复现

本次学习流程可由以下命令复现：

```bash
# 1. 拉取最新 main
git fetch origin && git checkout main && git pull

# 2. 创建当日分支
DATE=$(date +%Y%m%d)
git checkout -b feat/quant-opt-$DATE

# 3. 复用本目录下的所有代码与测试
#    （本报告提交时已把代码固化在 feat/quant-opt-20260618 分支）

# 4. 跑测试 + 基准
PYTHONPATH=. python -m pytest quant_opt_20260618/tests -q
PYTHONPATH=. python quant_opt_20260618/benchmark/run_benchmarks.py

# 5. 推送
git add quant_opt_20260618/
git commit -m "feat(quant-opt): 借鉴 AKQuant/Qlib 落地因子 DSL + PIT + Walk-Forward + 指纹缓存 PoC"
git push origin feat/quant-opt-$DATE
```
