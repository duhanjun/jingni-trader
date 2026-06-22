# 量化交易开源项目学习与 jingni-trader 优化验证报告

> **执行日期**: 2026-06-23
> **分支**: `feat/quant-opt-20260623`（基于 main，未合并）
> **执行人**: 自动化学习与验证流程

---

## 一、联网学习成果

### 1.1 学习项目清单

通过搜索 GitHub Trending、Awesome Quant、CSDN/技术社区、python.financial 等，筛选出以下高价值开源项目：

| 项目 | Star | 核心定位 | 借鉴价值 |
|------|------|---------|---------|
| **microsoft/qlib** | 15k+ | AI 导向量化投资平台 | Point-in-Time 数据库、Alpha158 因子表达式、ML Model Zoo |
| **polakowo/vectorbt** | - | 向量化回测框架 | NumPy 数组运算替代逐 bar 循环，研究阶段极速 |
| **nautilustrader** | - | 事件驱动高频交易 | 回测/实盘行为一致性、Rust 编译内核 |
| **akquant** | 新兴 | Rust+Python 混合框架 | 因子表达式引擎（Polars 驱动）、Walk-forward 验证 |
| **quantopian/alphalens** | - | 因子分析工具 | 分层回测（quantile returns）、IC 标准分析 |
| **vnpy** | 19k+ | 国产量化交易框架 | 实盘接口、CTA 策略模块化 |

### 1.2 深入研究的 3 个项目及核心亮点

#### ① Qlib（微软）— AI 量化全链路
- **Point-in-Time 数据库**：按时间点组织数据，防止未来数据泄露（look-ahead bias），这是严肃量化的关键
- **Alpha158/Alpha360 因子库**：用表达式 DSL 定义因子，如 `Rank(Ts_Mean(Close, 5))`，无需写代码即可扩展因子
- **qrun 工作流**：YAML 配置 + 一条命令跑完整实验（数据→特征→模型→回测→评估）
- **RD-Agent 集成**：LLM 自动挖因子、调参

#### ② VectorBT — 向量化回测范式
- **核心思想**：将回测逻辑推向 NumPy 数组运算，避免 Python 逐 bar 循环
- **两阶段工作流**：研究阶段用向量化快速筛选 → 验证阶段用事件驱动做执行级回测
- **性能**：大规模参数扫描比传统事件驱动快 10-100x

#### ③ Alphalens — 因子分析标准
- **IC 分析**：Spearman Rank IC 时间序列 + ICIR + t 统计量
- **分层回测**：按因子值分 5/10 组，观察各组收益单调性
- **因子衰减分析**：不同持有周期的 IC 变化

---

## 二、jingni-trader 现状分析与改进空间

### 2.1 现有架构

```
engine.py (主调度器)
  ├─ skills/data-engine          数据采集（tushare/akshare/baostock）
  ├─ skills/factor-engine        因子计算 + IC分析 + 中性化 + 融合
  ├─ skills/strategy-model-engine 模型训练
  ├─ skills/backtest-engine      回测（native/backtrader/rqalpha/gm 适配器）
  ├─ skills/portfolio-risk-engine 组合优化
  ├─ skills/execution-monitor-engine 实盘执行
  └─ skills/reports-engine       报告生成
```

### 2.2 发现的问题（对照开源项目）

| 模块 | 问题 | 借鉴来源 | 严重度 |
|------|------|---------|--------|
| **backtest native_adapter** | `iterrows()` 逐行循环，O(日×股) Python 调用，慢 | VectorBT | 高 |
| **backtest native_adapter** | `t_plus_1` 参数存在但**未真正生效**，当日买入可当日卖出 | NautilusTrader | 高（正确性） |
| **backtest native_adapter** | 滑点仅作用于买入，卖出无滑点 | NautilusTrader | 中（正确性） |
| **factor-engine `_calc_ic`** | 对每个日期循环调用 `scipy.stats.spearmanr`，O(日) 次 Python 调用 | Qlib/Alphalens | 高 |
| **factor-engine `neutralize`** | 对每个日期循环拟合 `LinearRegression`，O(日) 次 sklearn 调用 | Qlib | 高 |
| **factor-engine** | 因子硬编码在 `compute_a_share_factors()`，新增因子需改源码 | Qlib Alpha158/akquant | 中（扩展性） |
| **factor-engine** | 无分层回测（quantile returns）能力 | Alphalens | 中 |
| **engine.py** | `parse_intent` 关键词匹配，脆弱 | - | 低 |
| **engine.py** | `sys.modules` 删除式重载 hack | - | 低（可维护性） |

---

## 三、已完成的优化验证

### 3.1 优化点一：向量化回测引擎

**文件**: `optimizations/vectorized_backtest/__init__.py`

**借鉴来源**: VectorBT（向量化范式）+ NautilusTrader（行为一致性）

**优化内容**:
1. 用 `pivot_table` 一次性构建 (date × code) 宽表，日内全部用 NumPy 布尔掩码 + 向量运算，消除 `iterrows()`
2. **严格实现 T+1**：记录每只股票买入日 `buy_day`，当日买入不可卖出（原实现该参数未生效）
3. **滑点对称**：买入 `close*(1+slippage)`，卖出 `close*(1-slippage)`（原实现卖出无滑点）
4. **停牌过滤**：close 为 NaN 或 volume==0 视为停牌，跳过交易

**性能对比**:

| 数据规模 | 原生实现 | 向量化实现 | 加速比 |
|---------|---------|-----------|--------|
| 20股×60日 | 0.097s | 0.029s | **3.34x** |
| 50股×120日 | 0.302s | 0.063s | **4.77x** |
| 100股×120日 | 0.483s | 0.055s | **8.74x** |
| 200股×120日 | 0.855s | 0.076s | **11.22x** |

> 加速比随数据规模增大而提升（向量化优势更明显）。
> 注：最终净值存在差异是**预期的**——因为向量化版本正确实现了 T+1 和对称滑点，原实现存在正确性缺陷。

**正确性测试**（10 项全通过）:
- 基本运行返回结构完整
- T+1 强制生效（当日买入不可当日卖出）
- 关闭 T+1 后允许日内交易
- 滑点同时作用于买卖两侧
- 空数据/空信号边界处理
- 涨停无法买入
- 停牌股票被跳过
- 绩效指标完整性

### 3.2 优化点二：向量化因子 IC 分析

**文件**: `optimizations/vectorized_factor/__init__.py`

**借鉴来源**: Qlib（批量化 IC）+ Alphalens（分层回测）

**优化内容**:
1. `calc_ic_series`：原 `for dt: scipy.spearmanr()` → `groupby('date')` + rank + 向量化相关，一次性计算全部截面
2. `neutralize_vectorized`：原 `for dt: LinearRegression.fit()` → `groupby('date')` + `numpy.lstsq` 批量求解残差
3. 新增 `quantile_returns`：分层回测，按因子值分 N 组计算各组远期收益（借鉴 Alphalens）

**性能对比**:

| 数据规模 | scipy 逐日 | 向量化 | 加速比 | IC 结果差异 |
|---------|-----------|--------|--------|-----------|
| 30股×80日 | 0.0300s | 0.0110s | **2.72x** | 0.00e+00 |
| 100股×120日 | 0.0480s | 0.0154s | **3.13x** | 0.00e+00 |
| 200股×200日 | 0.0873s | 0.0331s | **2.64x** | 0.00e+00 |

> **IC 结果完全一致**（差异为 0），证明向量化实现正确无误。

**正确性测试**（6 项全通过）:
- IC 序列非空
- 向量化 IC 与 scipy 参考实现一致（误差 < 1e-6）
- IC 统计结构完整
- 中性化保持数据形状
- 中性化后残差与市值因子正交（|corr| < 0.1）
- 分层回测输出正确

### 3.3 优化点三：因子表达式引擎

**文件**: `optimizations/factor_expression/__init__.py`

**借鉴来源**: Qlib Alpha158（表达式 DSL）+ akquant（Alpha101 风格公式）

**优化内容**:
- 提供表达式 DSL，用户用字符串定义因子，无需写代码：
  - `Rank(Ts_Mean(Close, 5))` — 5日均量排名
  - `Div(Sub(High, Low), Close)` — 振幅
  - `Mul(-1, Ts_Mean(Return, 20))` — 20日反转
- 支持算子：截面（Rank/Zscore/Scale）、时序（Ts_Mean/Std/Max/Min/Sum/Rank/Delta/Delay）、算术（Add/Sub/Mul/Div/Abs/Log/Sign）
- 算子大小写不敏感（`Ts_std` 等同 `Ts_Std`）
- 预定义 Alpha158 风格因子库（8 个因子）

**性能**:

| 数据规模 | 因子数 | 耗时 | 吞吐 |
|---------|--------|------|------|
| 20股×60日 (1200行) | 8 | 0.059s | 134.6 因子/秒 |
| 50股×100日 (5000行) | 8 | 0.125s | 63.8 因子/秒 |
| 100股×120日 (12000行) | 8 | 0.254s | 31.5 因子/秒 |

**正确性测试**（11 项全通过）:
- 表达式解析（字段/函数/嵌套）
- Close 字段计算与原数据一致
- Ts_Mean 与 pandas rolling 参考一致
- Rank 截面排名正确
- 嵌套表达式 Rank(Ts_Mean(Close,5)) 正确
- 算术运算 Div(Sub(High,Low),Close) 正确
- 批量计算
- Alpha158 预定义库全部可计算
- 未知算子/字段报错

---

## 四、测试总结

```
======================== 31 passed, 2 warnings in 3.62s ========================
```

| 测试类别 | 测试数 | 通过 | 覆盖内容 |
|---------|--------|------|---------|
| 向量化回测 | 10 | 10 | 正确性、T+1、滑点、边界、涨跌停、停牌 |
| 向量化因子 | 6 | 6 | IC 正确性、中性化、分层回测 |
| 因子表达式 | 11 | 11 | 解析、计算、批量、预定义库、错误处理 |
| 性能测试 | 4 | 4 | 大规模数据性能 |
| **合计** | **31** | **31** | — |

---

## 五、待用户确认的优化建议

以下优化已在本分支验证通过，**未合并 main**，等待用户确认：

### 建议合并的优化（高价值、低风险）

1. **向量化回测引擎** — 性能提升 3-11x，且修复了 T+1 未生效、滑点不对称的正确性缺陷
2. **向量化 IC 分析** — 性能提升 2.6-3.1x，结果与原实现完全一致（零差异）
3. **因子表达式引擎** — 提供因子扩展能力，不破坏现有代码

### 后续可探索方向（本次未实现）

| 方向 | 借鉴来源 | 说明 |
|------|---------|------|
| Point-in-Time 数据库 | Qlib | 防止未来数据泄露，需改造 data-engine |
| Walk-forward 验证 | akquant | 滚动训练验证，防过拟合 |
| ML Model Zoo | Qlib | 内置 LightGBM/LSTM/Transformer 模板 |
| Rust 加速核心 | akquant/NautilusTrader | 性能敏感路径用 Rust 重写 |
| LLM 因子挖掘 | Qlib RD-Agent | 自动生成候选因子 |

---

## 六、文件清单

```
feat/quant-opt-20260623 分支新增文件:
├── optimizations/
│   ├── vectorized_backtest/__init__.py   向量化回测引擎
│   ├── vectorized_factor/__init__.py     向量化因子分析
│   ├── factor_expression/__init__.py     因子表达式引擎
│   ├── benchmark.py                      性能基准脚本
│   ├── benchmark_results.json            基准结果数据
│   └── VERIFICATION_REPORT.md            本报告
└── tests/
    └── test_optimizations.py             31 项测试（全通过）
```

**重要约束遵守**:
- 所有新代码位于 `feat/quant-opt-20260623` 分支独立目录，**未修改 main 分支任何代码**
- 仅执行 `git push`，**未执行 git merge**
- 等待用户确认后方可合并
