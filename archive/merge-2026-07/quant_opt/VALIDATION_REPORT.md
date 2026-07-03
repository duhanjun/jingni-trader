# jingni-trader 量化优化验证报告

**执行日期**：2026-06-18
**分支**：`feat/quant-opt-20260618`
**验证模块**：`/workspace/quant_opt/`

---

## 一、本次学习项目清单

| 项目 | 仓库 | 关注度 | 借鉴重点 |
|------|------|--------|----------|
| **Microsoft Qlib** | github.com/microsoft/qlib | 36.5k⭐ / MIT | 表达式因子引擎、Alpha158/360 因子库、PIT 数据、Rolling 训练框架 |
| **Microsoft RD-Agent** | github.com/microsoft/RD-Agent | 11.1k⭐ / MIT | 研究-开发双 Agent 闭环、CoSTEER 代码生成、因子-模型协同进化 |
| **ClawQuant (agent-quant)** | github.com/clawquant/agentquant | PyPI 0.2.57 / MIT | A 股 Skill 化封装、MongoDB 行情底座、CLI 工具设计 |

### 关键亮点摘录

**Qlib** ([核心架构](https://deepwiki.com/microsoft/qlib))
- 4 层架构：Core Infrastructure → Data Pipeline → Model Layer → Strategy & Backtesting → Workflow
- 表达式引擎支持 `Ref($close, 5) / $close - 1` 这种 DSL 写法，运行时再编译为算子
- 多级缓存（Expression Cache + Dataset Cache）避免重复计算
- Point-in-Time (PIT) 数据库杜绝时间穿越
- YAML workflow config + `qrun` CLI 让回测可一行命令复现

**RD-Agent** ([NeurIPS 2025 论文](https://papers.nips.cc/paper_files/paper/2025/file/ac5c2b6e423883cbcacbcccf88491b78-Paper-Datasets_and_Benchmarks_Track.pdf))
- R (Research) + D (Development) 双 Agent 闭环
- 5 步循环：假设生成 → 任务分解 → 代码实现 → 回测反馈 → 下轮假设
- Co-STEER 引擎：最多 10 轮"写代码—测试—修正"自动调试
- 多臂赌博机调度器自适应选择进化方向
- 成功 / 失败案例库做 RAG 检索提升首次生成成功率

**ClawQuant** ([PyPI](https://pypi.org/project/agent-quant/))
- 专为 OpenClaw / jingni-trader 类 Skill 框架设计
- CLI 化数据更新 / 查询 / 因子 / 回测
- 因子 + 策略开箱即用

---

## 二、jingni-trader 现状对照

| 模块 | 现状问题 | 借鉴方向 |
|------|----------|----------|
| `skills/factor-engine/engine.py::compute_a_share_factors` | 因子硬编码，新增需改源码；只有约 15 个 A 股因子 | Qlib 表达式引擎 + Alpha158 因子库（30+ 公式） |
| `skills/factor-engine/engine.py::ic_analysis` / `_calc_ic` | Python 循环逐日计算 Spearman，速度慢 | Qlib 向量化 rank-IC 计算 |
| `skills/factor-engine/engine.py::ic_analysis` | 缺少 PIT 漏检 / 重复日期检测 | Qlib Point-in-Time + Lopez de Prado Purged K-Fold |
| `scripts/config.py::MODEL_TYPE` 等 | 模型训练未做时序切分（只有 train/valid/test 月份配置） | Qlib `RollingGen` + 自带 purge gap |
| `engine.py::parse_intent` | 关键词匹配，过简 | 长期可借鉴 RD-Agent LLM 驱动解析 |

---

## 三、已实现的优化模块

### 3.1 表达式因子引擎 (Qlib 借鉴)

**文件**：`quant_opt/expression_engine/expr_engine.py`、`alpha_catalog.py`

- **Tokenizer + AST Parser + Evaluator** 三段式，纯 Python / pandas 实现
- 内置 22 个算子：时序 (Ref/Mean/Std/Delta/EMA/TsRank/Corr/Max/Min/Sum) + 截面 (Rank/Scale/Zscore/Quantile/CsMean/CsStd/Neutralize) + 标量 (Abs/Log/Sign/SignedPower)
- **30 个预制 Alpha 因子**（momentum / reversal / volatility / volume / trend / cross-section / log-return 等）
- 用户加因子只改一行字符串，无需改代码

### 3.2 向量化 IC 分析 (Qlib 借鉴)

**文件**：`quant_opt/ic_analysis/vectorized_ic.py`

- 逐日 Spearman 改为 `groupby(level="date").rank() + transform("mean")` 向量化
- 一次性输出 `ic_mean / ic_std / ic_ir / ic_positive_ratio / ic_t_stat / n_periods`
- `rank_ic_decay(factor, target, max_lag=N)` 因子衰减曲线
- **`batch_ic()`** 批量计算多因子

**性能对比（200 股票 × 750 日）**：

| 方案 | 耗时 | 加速比 |
|------|------|--------|
| 原版 jingni-trader 循环 | 1.32 s | 1.0× |
| 新版向量化 | 0.10 s | **~13×** |

数值一致性：max diff < 1.2e-3，corr = 1.0000

### 3.3 时序 CV + PIT 漏检 (Qlib PIT 借鉴)

**文件**：`quant_opt/cv_splitter/purged_cv.py`

- `TimeSeriesCV`：滑窗 train/valid/test，支持 `purge_gap` + `embargo`
- `PurgedKFold`：K 折 + purge gap（Lopez de Prado 风格）
- `leakage_check()` 检测：
  - `(code, date)` 重复
  - 未来日期
  - 恒定列
  - 特征与 forward return 完全相关

---

## 四、测试与基准

### 4.1 单元测试

**文件**：`quant_opt/tests/test_quant_opt.py`
**结果**：**25/25 passed**（详情见 `reports/test_results.json`）

| 维度 | 测试数 | 通过 |
|------|--------|------|
| Tokenizer / Parser | 5 | 5 |
| Evaluator | 5 | 5 |
| Alpha catalog | 2 | 2 |
| Vectorized IC | 4 | 4 |
| TimeSeries CV / PurgedKFold | 4 | 4 |
| Leakage check | 4 | 4 |
| Operator registry | 1 | 1 |

### 4.2 性能基准

**文件**：`quant_opt/benchmarks/bench_engine.py`、`bench_ic.py`、`demo_pipeline.py`、`run_all.py`

| 规模 | 总耗时 (31 因子) | 单因子 |
|------|------------------|--------|
| 50 股 × 252 日 | 1.38 s | 44.6 ms |
| 100 股 × 504 日 | 5.08 s | 164.0 ms |
| 200 股 × 750 日 | 14.51 s | 468.2 ms |
| 500 股 × 1000 日 | 47.40 s | 1528.9 ms |

IC 计算加速 ~13×

### 4.3 端到端 Pipeline

`demo_pipeline.py` 走完：
1. 构造 50 股 × 252 日合成数据
2. 用表达式引擎算出 31 个因子
3. 跑 PIT 漏检（clean）
4. 批量 IC → 选最优因子 REV_60（IC_IR = -0.1533）
5. 因子衰减分析（10 期内保持稳定负 IC）
6. 时序 CV：3 个 fold，每个 train/valid/test = 120/30/30
7. Purged K-Fold：5 fold

---

## 五、待用户确认的优化建议

| 建议 | 优先级 | 改动量 | 风险 |
|------|--------|--------|------|
| **A. 把表达式引擎集成进 `factor-engine`**，替换 `compute_a_share_factors` 中的硬编码因子 | 高 | 中 | 低（保留原代码，新增可选项） |
| **B. 把向量化 IC 集成进 `factor-engine.ic_analysis`**，循环代码作为 fallback | 高 | 小 | 低（已验证数值等价） |
| **C. 在 `data-engine` 增加 `leakage_check()` 钩子** | 中 | 小 | 极低（纯检查） |
| **D. 在 `model-engine` 增加 `TimeSeriesCV` 训练切分器**，替换固定 36/12/12 月窗口 | 中 | 中 | 中（需确保不影响现有 LightGBM 训练） |
| **E. 因子引擎的 alpha catalog 扩到 158 个**（Qlib Alpha158 完整版） | 低 | 大 | 低 |
| **F. 引入 RD-Agent 风格的 LLM 自动因子挖掘**（高级玩法） | 低 | 极大 | 高（需 LLM 凭据、计算成本） |

---

## 六、文件结构

```
/workspace/quant_opt/
├── __init__.py
├── expression_engine/
│   ├── __init__.py
│   ├── expr_engine.py        # Tokenizer + Parser + Evaluator
│   └── alpha_catalog.py      # 30 个预制 alpha 公式
├── ic_analysis/
│   ├── __init__.py
│   └── vectorized_ic.py      # 向量化 IC + 衰减分析
├── cv_splitter/
│   ├── __init__.py
│   └── purged_cv.py          # TimeSeriesCV + PurgedKFold + leakage_check
├── tests/
│   ├── __init__.py
│   └── test_quant_opt.py     # 25 个单元测试
├── benchmarks/
│   ├── __init__.py
│   ├── bench_ic.py           # 性能基准：循环 vs 向量化
│   ├── bench_engine.py       # 表达式引擎扩展性
│   ├── demo_pipeline.py      # 端到端 demo
│   └── run_all.py            # 一键跑全部 + 写报告
└── reports/
    ├── summary.json          # 聚合摘要
    ├── bench_ic.json         # IC 基准结果
    ├── bench_engine.json     # 引擎基准结果
    ├── demo_pipeline.json    # pipeline demo 结果
    └── test_results.json     # 完整测试结果
```

---

## 七、复现命令

```bash
# 安装依赖
pip install numpy pandas scipy

# 跑测试
PYTHONPATH=/workspace python3 quant_opt/tests/test_quant_opt.py

# 跑基准
PYTHONPATH=/workspace python3 quant_opt/benchmarks/run_all.py
```

---

## 八、结论

本次基于对 Qlib / RD-Agent / ClawQuant 的深度调研，**实现并验证了 3 个可立即在 jingni-trader 上借鉴的优化点**：

1. **表达式因子引擎** — 把因子维护从改代码变成改字符串，把因子库规模从 ~15 提升到 30+（并可继续扩展到 158/360）
2. **向量化 IC 计算** — 性能提升 13×，数值与原版一致
3. **时序 CV + PIT 漏检** — 防止 look-ahead leakage 标准化

所有代码与测试已 commit 到 `feat/quant-opt-20260618` 分支，**未合入 main**。等待用户确认后（建议优先 A、B、C 三项）再做集成与合并。
