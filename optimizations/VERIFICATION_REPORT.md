# jingni-trader 量化优化验证报告

> **分支**: `feat/quant-opt-20260624`
> **执行日期**: 2026-06-24
> **测试结果**: 33/33 通过 (100%)
> **状态**: 已推送至 GitHub 远程，**未合并** main（等待用户确认）

---

## 一、学习项目清单及核心亮点

本次联网调研覆盖 GitHub、arXiv、PyPI、量化社区，筛选出以下 3 个最有借鉴价值的开源项目/研究：

### 1. VectorBT — 向量化回测框架
- **仓库**: https://vectorbt.dev/ (开源版) / VectorBT PRO
- **Star/活跃度**: Python 回测领域性能标杆，2026 年仍在活跃迭代
- **核心亮点**:
  - **完全向量化**设计：将策略状态表示为 2D NumPy 数组 (日期 × 资产)，用矩阵运算替代逐 bar 循环
  - **Numba + Rust 双后端**：关键路径预编译，处理 100 万订单仅需 ~50ms
  - **参数扫描**：可在数秒内测试数千组参数组合
  - 性能对比：相同 SMA 策略，Backtrader 需 12 秒，VectorBT < 200ms (60x+)
- **可借鉴方向**: 向量化回测引擎设计、矩阵化持仓/收益计算

### 2. alphalens-reloaded — 因子分析标准库
- **仓库**: https://github.com/stefan-jansen/alphalens-reloaded
- **维护者**: stefan-jansen (原 Quantopian alphalens 的社区维护版)
- **核心亮点**:
  - **IC 衰减分析** (IC Decay)：计算因子对 1/5/10/20 多个前视期的 IC，揭示因子预测能力的衰减速度
  - **因子换手率** (Turnover)：因子排名的自相关性，衡量信号稳定性与实盘交易成本
  - **分层收益** (Quantile Returns)：按因子值分 N 层，计算每层平均收益与多空收益
  - **Rank IC vs Normal IC**：同时提供 Spearman (鲁棒) 与 Pearson 两种 IC
  - 完整的 `create_full_tear_sheet` 可视化报告
- **可借鉴方向**: IC 衰减、换手率、分层收益 — 现有 factor-engine 完全缺失

### 3. Microsoft Qlib — AI 量化投研平台
- **仓库**: https://github.com/microsoft/qlib (17.5K+ Star)
- **核心亮点**:
  - **Alpha158 / Alpha360** 标准因子集 + 20+ 预置模型 benchmark
  - **滚动训练** (Walk-forward) 框架：防止前视偏差，适配非平稳金融数据
  - 完整的 **数据-特征-模型-回测-报告** 闭环
  - Benchmark 显示 LightGBM/DoubleEnsemble 在 CSI300 上 ICIR 可达 0.37-0.42
- **可借鉴方向**: 滚动训练验证、防泄漏窗口设计、样本外评估

### 其他参考项目（未深入实现，记录备查）
| 项目 | 价值点 |
|------|--------|
| AKQuant (Rust+Python) | Rust 内核 + 因子表达式引擎 (Polars) |
| Hubble (arXiv 2604.09601) | LLM 驱动的自动因子挖掘 + AST 沙箱 |
| NautilusTrader | 事件驱动 + 实盘一致性回测 |
| vnpy | 国内实盘接口集成 (CTP/富途等 40+) |

---

## 二、可借鉴的方向列表

对照 jingni-trader 现有代码，识别出以下改进空间（按优先级排序）：

| # | 优化方向 | 现有问题 | 借鉴来源 | 可行性 |
|---|---------|---------|---------|--------|
| 1 | **向量化回测引擎** | `native_adapter.py` 用 `for dt: for code:` 双重 Python 循环，252天×100股需 ~97ms | VectorBT | 高 ✅ 已验证 |
| 2 | **IC 衰减 + 换手率 + 分层收益** | `factor-engine` 只算 IC mean/std/IR，缺衰减/换手/分层；且 IC 用逐日循环 | alphalens-reloaded | 高 ✅ 已验证 |
| 3 | **滚动训练验证器** | `strategy-model-engine` 有 `purged_group_ts_split` 但 `train()` 只单次训练，无滚动重训 + 无泄漏校验 | Qlib | 高 ✅ 已验证 |
| 4 | HR P 修复 | `portfolio-risk-engine._optimize_hrp` 传空 returns，HRP 必然失败 | PyPortfolioOpt | 中 (待后续) |
| 5 | CVaR 优化实现 | `_optimize_cvar` 是 stub，直接返回等权 | Riskfolio-lib | 中 (待后续) |
| 6 | Barra 归因 | `barra_style_attribution` 返回全 0 | 业界 Barra CNE5 | 低 (需因子数据) |
| 7 | 主引擎模块重载 hack | `engine.py` 用 `del sys.modules` 强制重载，脆弱 | 标准包管理 | 中 (待后续) |

本次聚焦 **前 3 项**（高可行性、高价值、可独立验证），已完成代码与测试。

---

## 三、已完成的验证测试及结论

### 测试总览

| 模块 | 测试数 | 通过 | 通过率 | 关键性能 |
|------|--------|------|--------|---------|
| 向量化回测引擎 | 10 | 10 | 100% | **8.6x 加速** |
| 增强因子分析 | 13 | 13 | 100% | **7.3x 加速** (IC) |
| 滚动训练验证器 | 10 | 10 | 100% | 滚动 IC 0.83 vs 单次 -0.17 |
| **合计** | **33** | **33** | **100%** | — |

### 3.1 向量化回测引擎 (借鉴 VectorBT)

**文件**: `optimizations/vectorized_backtest/vectorized_adapter.py`

**核心设计**:
- 将价格、目标权重表示为 `DataFrame(date × code)` 矩阵
- 持仓权重 `ffill` 后，组合日收益 = `(held_weights * daily_returns).sum(axis=1)` 一次性矩阵运算
- 交易成本仅在调仓日计入，按换手率向量计算
- 权益曲线 = `init_capital * (1 + portfolio_net_return).cumprod()`

**正确性验证** (手算小样本):
```
价格: A=[10,11,11], B=[20,20,22]
调仓: Day0 全仓A → Day1 切换全仓B
无成本预期: 10000 → 11000 (A涨10%) → 12100 (B涨10%)
实际输出: 12100.00 ✓  总收益 21.00% ✓
含成本终值 12032.85 < 12100 ✓
```

**性能对比** (252天 × 100股, 月度调仓):
| 引擎 | 耗时 | 终值 |
|------|------|------|
| 现有 native 循环 | 97.1 ms | 1,765,776 |
| 向量化 (本实现) | 11.3 ms | 1,816,476 |
| **加速比** | **8.6x** | 终值差异 2.8% (策略语义略异) |

**边界测试**: 单股票 ✓ / 全现金 ✓ / 价格含 NaN ✓

### 3.2 增强因子分析 (借鉴 alphalens-reloaded)

**文件**: `optimizations/factor_analysis/enhanced_factor_analysis.py`

**新增能力** (现有 factor-engine 均缺失):
1. **IC 衰减**: 一次性计算 [1,5,10,20] 期 IC，揭示因子适用周期
2. **因子换手率**: 排名自相关性，衡量实盘交易成本
3. **分层收益**: 5 层分位收益 + 多空收益
4. **Rank IC (Spearman)**: 对极值鲁棒，业界首选

**IC 衰减验证** (有预测力的合成因子):
| 前视期 | IC均值 | ICIR | t统计量 |
|--------|--------|------|---------|
| 1天 | 0.413 | 4.40 | 高显著 |
| 5天 | **0.959** | **79.9** | 1116 |
| 10天 | 0.664 | 9.71 | 高显著 |
| 20天 | 0.462 | 5.02 | 高显著 |

→ 5期 IC 最高 (0.959)，随期数衰减 ✓ 符合预期

**分层收益验证** (5层):
| 分位 | 平均收益 | 观测数 |
|------|---------|--------|
| Q1 (最低) | -4.02% | 3120 |
| Q2 | -1.40% | 3120 |
| Q3 | +0.25% | 3120 |
| Q4 | +2.02% | 3120 |
| Q5 (最高) | **+4.82%** | 3120 |

→ 单调递增 ✓ 多空收益 +8.84% ✓

**性能对比** (200天 × 80股, Spearman IC):
| 方法 | 耗时 | IC均值 |
|------|------|--------|
| 现有逐日循环 (scipy.spearmanr) | 101.9 ms | 0.9594 |
| 向量化 (rank + 协方差矩阵) | 13.9 ms | 0.9594 |
| **加速比** | **7.3x** | 完全一致 ✓ |

**边界测试**: 纯随机因子 |IC|=0.014 < 0.1 ✓ (无虚假预测力)

### 3.3 滚动训练验证器 (借鉴 Qlib)

**文件**: `optimizations/walk_forward/walk_forward_validator.py`

**核心设计**:
- `generate_windows()`: 生成 (train, test) 滚动窗口序列，带 embargo_gap
- `validate_no_leakage()`: 严格校验 ① train_end < test_start ② test 区间不重叠 ③ 覆盖率
- `run()`: 接收任意 sklearn 风格模型，返回聚合样本外预测 (OOF) + 每 fold 指标

**泄漏校验** (train=120天, test=30天, gap=5天):
- 生成 13 个 fold ✓
- 泄漏校验通过 (violations=[]) ✓
- 所有 fold: train_end < test_start ✓
- 测试集覆盖率 100% ✓

**滚动 vs 单次训练对比** (非平稳数据: f1 系数从 +0.5 → -0.5):
| 方法 | 样本外 IC |
|------|----------|
| 单次训练 (前60%训练, 后40%测试) | **-0.169** (失效) |
| 滚动训练 (13 fold) | **+0.833** (有效) |

→ 非平稳数据上滚动训练显著优于单次训练 ✓

**边界测试**: gap=0 仍无重叠 ✓ / 数据不足返回空窗口 ✓

---

## 四、测试代码结构

```
optimizations/
├── __init__.py
├── vectorized_backtest/
│   ├── __init__.py
│   └── vectorized_adapter.py      # 向量化回测引擎 (VectorBT 风格)
├── factor_analysis/
│   ├── __init__.py
│   └── enhanced_factor_analysis.py # IC衰减/换手率/分层收益 (alphalens 风格)
├── walk_forward/
│   ├── __init__.py
│   └── walk_forward_validator.py   # 滚动训练+泄漏校验 (Qlib 风格)
├── run_all_tests.py                # 综合测试套件 (33 项)
├── test_results.json               # 测试结果 (机器可读)
└── VERIFICATION_REPORT.md          # 本报告
```

**运行方式**:
```bash
cd /workspace
python -m optimizations.run_all_tests
```

---

## 五、待用户确认的优化建议

以下优化方案已在 `feat/quant-opt-20260624` 分支验证通过，**等待用户确认后**方可合并到 main：

### 建议 1: 集成向量化回测引擎 (高优先级)
- **动作**: 将 `vectorized_adapter.py` 作为 `backtest-engine` 的新后端 `BACKTEST_BACKEND=vectorized`
- **收益**: 回测速度提升 8x+，参数扫描场景受益更大
- **风险**: 低。仅新增后端，不修改现有 native/rqalpha/backtrader 后端
- **适用**: 周期性调仓策略 (多因子选股主流场景)

### 建议 2: 增强 factor-engine 的 IC 分析 (高优先级)
- **动作**: 将 `enhanced_factor_analysis.py` 的 IC 衰减/换手率/分层收益集成到 `factor-engine.ic_analysis`
- **收益**: 因子评估维度从 5 个指标扩展到 15+，IC 计算加速 7x
- **风险**: 低。新增分析方法，现有 IC 输出字段保持兼容
- **关键新增**: IC 衰减 (判断因子周期)、换手率 (判断实盘成本)、分层收益 (判断单调性)

### 建议 3: 为 strategy-model-engine 增加滚动训练模式 (中优先级)
- **动作**: 将 `walk_forward_validator.py` 作为 `MODEL` 阶段的可选训练模式
- **收益**: 非平稳数据上样本外 IC 从 -0.17 提升到 +0.83；提供泄漏校验保障
- **风险**: 中。需调整 `train()` 接口，但保留单次训练为默认模式
- **建议**: 通过 `strategy_params.walk_forward=True` 开启

### 后续可探索方向 (本次未实现)
- 修复 `portfolio-risk-engine._optimize_hrp` 的空 returns bug
- 实现 `_optimize_cvar` (当前为 stub)
- 用 Rust/Polars 重写因子表达式引擎 (借鉴 AKQuant)
- 探索 LLM 驱动因子挖掘 (借鉴 Hubble)

---

## 六、合规说明

- ✅ 所有新代码位于 `feat/quant-opt-20260624` 分支独立目录 `optimizations/`
- ✅ **未修改** main 分支任何现有文件
- ✅ **未执行** git merge / PR 合入操作
- ✅ 已推送分支至 GitHub 远程 (仅 push，不合并)
- ⏳ 等待用户明确确认后方可合并到 main

---

*报告生成时间: 2026-06-24 | 测试环境: Python 3.12.13, pandas 3.0.3, numpy 2.5.0, scipy 1.18.0*
