# jingni-trader 量化优化学习与验证报告

> 执行日期：2026-06-21
> 分支：`feat/quant-opt-20260621`
> 执行人：自动化学习流程（GLM-5.2）

---

## 一、联网学习成果

### 1.1 学习项目清单

通过 GitHub、PyPI、arXiv、python.financial 等渠道，调研了 2026 年活跃的量化交易开源项目，重点挑选以下 3 个最具借鉴价值的项目深入分析：

| 项目 | 类型 | Star/活跃度 | 核心亮点 |
|------|------|------------|---------|
| **VectorBT** | 向量化回测框架 | 6.5k+ / 活跃 | 将回测逻辑表示为 NumPy 数组运算，用 Numba/Rust 加速，比传统事件驱动快 100-1000 倍；支持海量参数并行扫描 |
| **AKQuant** | Rust+Python 混合框架 | 1.5k+ / 2026 活跃 | Rust 零拷贝内核；内置 Polars 驱动的因子表达式引擎，支持 `Rank(Ts_Mean(Close, 5))` 等 Alpha101 风格公式；原生 Walk-forward ML 验证 |
| **Qlib (Microsoft)** | AI 量化研究平台 | 15k+ / 活跃 | 表达式 DSL 将因子定义为可组合算子树；内置 LightGBM/Transformer 模型模板；动态 universe 的 Pipeline 研究 |

辅助参考：NautilusTrader（事件驱动+Rust 内核，回测/实盘一致性）、FactorEngine（arXiv 2603.16365，LLM 引导的程序级因子挖掘）。

### 1.2 核心亮点提炼

**VectorBT 的关键设计**：
- 「向量化范式」vs「事件驱动范式」的清晰分工：研究阶段用向量化求速度，实盘阶段用事件驱动求真实
- 将每个策略实例表示为多维数组的一个切片，一次处理数千实例
- 路径依赖问题用编译内核（Numba/Rust）解决，而非纯 Python

**AKQuant 的关键设计**：
- 因子表达式引擎：用户用字符串定义因子，无需写代码，自动并行计算与数据对齐
- 时序算子（Ts_Mean/Ts_Std）按标的分组，横截面算子（Rank/Zscore）按日期分组

**Qlib 的关键设计**：
- 算子树可任意嵌套组合，新增因子零代码改动
- 表达式即文档，可审计、可复现

---

## 二、jingni-trader 现状分析与优化方向

### 2.1 现有代码瓶颈定位

通过阅读 jingni-trader 源码，发现以下性能与可扩展性瓶颈：

| 模块 | 文件 | 问题 | 复杂度 |
|------|------|------|--------|
| 回测引擎 | `skills/backtest-engine/scripts/adapters/native_adapter.py` | `for dt in dates: signals[signals['date']==dt]` 逐日过滤 | O(n_days × n_rows) |
| 因子IC分析 | `skills/factor-engine/engine.py` `_calc_ic` | `for dt in dates: data[data['date']==dt]` 逐日循环算相关 | O(n_days × n_rows) |
| 因子中性化 | `skills/factor-engine/engine.py` `neutralize` | `for dt in dates:` 逐日拟合回归 | O(n_days × n_rows) |
| 因子计算器 | `skills/factor-engine/scripts/adapters/pandas_ta_calculator.py` `_calc_single` | `for code in unique(): data[mask]` 逐标的循环 | O(n_codes × n_rows) |
| 因子定义 | 同上 | if/elif 硬编码每个因子，新增需改源码 | 可扩展性差 |

### 2.2 可借鉴的优化方向

| 优化方向 | 借鉴来源 | 预期收益 | 本次验证 |
|---------|---------|---------|---------|
| 向量化回测引擎 | VectorBT | 性能 10-30x | ✅ 已验证 |
| 因子表达式引擎 | AKQuant / Qlib | 可扩展性，零代码新增因子 | ✅ 已验证 |
| 向量化 IC 分析 | VectorBT 数组化思想 | 性能 3x+ | ✅ 已验证 |
| 因子中性化向量化 | 同上 | 性能提升 | ⏳ 待后续 |
| 因子计算器 groupby 化 | 同上 | 性能提升 | ⏳ 待后续 |

---

## 三、已完成的验证测试

### 3.1 验证代码结构

所有新代码位于 `quant_opt_20260621/` 独立目录，不修改 main 分支任何现有文件：

```
quant_opt_20260621/
├── __init__.py
├── vectorized_backtest.py        # 向量化回测引擎（借鉴 VectorBT）
├── factor_expression_engine.py   # 因子表达式引擎（借鉴 AKQuant/Qlib）
├── vectorized_ic.py              # 向量化 IC 分析
├── run_verification.py           # 验证测试主入口
└── verification_results.json     # 测试结果数据
```

### 3.2 测试结果汇总

**6 项测试全部通过（PASS=6, FAIL=0, ERROR=0）**：

| 测试 | 状态 | 耗时 | 关键结论 |
|------|------|------|---------|
| backtest_correctness | PASS | 0.158s | 交易笔数完全一致(83=83)，终值相对误差仅 0.15% |
| backtest_performance | PASS | 2.097s | 平均加速 **20.65x**，大规模(100股×200天)达 **30.8x** |
| backtest_edge_cases | PASS | 0.041s | 空数据/单股/无信号/全涨停 均正确处理 |
| expression_parser | PASS | 0.000s | 嵌套表达式 `Rank(Ts_Mean(Close,5))` 解析正确 |
| expression_calculation | PASS | 0.054s | 与手写 pandas 实现完全一致(误差 0) |
| vectorized_ic | PASS | 0.312s | IC 误差 6.94e-17(机器精度)，加速 **3.28x** |

### 3.3 性能对比详情

**回测引擎性能（3 种数据规模）**：

| 规模 | 数据行数 | 原生耗时 | 向量化耗时 | 加速比 |
|------|---------|---------|-----------|--------|
| 10股×60天 | 600 | 0.087s | 0.010s | 9.16x |
| 50股×120天 | 6,000 | 0.363s | 0.017s | 22.00x |
| 100股×200天 | 20,000 | 0.999s | 0.032s | **30.80x** |

> 结论：数据规模越大，向量化优势越明显（消除 O(n²) 过滤开销）。

**IC 分析性能**：
- 原生逐日循环：0.152s（110 个周期）
- 向量化 groupby：0.046s（110 个周期）
- 加速比：3.28x，IC 值误差 6.94e-17（机器精度，完全等价）

### 3.4 正确性验证详情

**回测终值对比**（20股×60天，5日反转策略）：
- 原生终值：903,910.05
- 向量化终值：902,516.73
- 相对误差：0.1541%（< 2% 阈值，PASS）
- 交易笔数：83 = 83（完全一致）

> 微小差异来源：买卖候选在同日同预算下，遍历顺序的细微差异导致成交股数偶有 100 股级别差别，业务逻辑完全等价。

**因子表达式正确性**：
- `Ts_Mean(Close, 5)` vs 手写 rolling mean：最大误差 0.00e+00
- `Rank(-Returns(Close, 5))` vs 手写横截面排名：最大误差 0.00e+00
- 5 个预置因子批量计算：全部成功

---

## 四、优化方案设计与待确认建议

### 4.1 已验证可行的优化（建议合入 main）

**优化点 1：向量化回测引擎**
- 替换 `native_adapter.py` 的逐日循环为 `vectorized_backtest.py` 的 pivot+NumPy 方案
- 保持 A 股 T+1、涨跌停、印花税、滑点规则不变
- 收益：10-30x 性能提升，结果等价（误差 <0.2%）
- 风险：低，已通过正确性+边界测试

**优化点 2：因子表达式引擎**
- 新增 `factor_expression_engine.py` 作为 factor-engine 的新后端
- 支持 Alpha101 风格表达式，预置 12 个因子
- 收益：新增因子零代码改动（只需写字符串），可扩展性大幅提升
- 风险：低，与手写实现完全等价

**优化点 3：向量化 IC 分析**
- 替换 `engine.py` `_calc_ic` 的逐日循环为 `vectorized_ic.py` 的 groupby 方案
- 收益：3x+ 性能提升，结果机器精度等价
- 风险：极低

### 4.2 待用户确认事项

1. **是否将上述 3 个优化点合并到 main 分支？**
   - 当前所有代码在 `feat/quant-opt-20260621` 分支，未合并
   - 用户确认后我可执行 git merge / PR 合入

2. **向量化回测的微小终值差异（0.15%）是否可接受？**
   - 差异源于同日多候选股的遍历顺序，非逻辑错误
   - 如需完全一致，可对齐候选股排序顺序

3. **因子表达式引擎是否作为 factor-engine 的默认后端？**
   - 可与现有 pandas_ta/talib 后端并存
   - 建议新增 `FACTOR_BACKEND=expression` 配置项

4. **后续优化方向（本次未验证，待优先级确认）**：
   - 因子中性化向量化（`neutralize` 方法的逐日循环）
   - 因子计算器 groupby 化（`_calc_single` 的逐标的循环）
   - 引入 Numba JIT 编译进一步加速（如 VectorBT）
   - 引入 Polars 替代 pandas 提升大数据处理（如 AKQuant）

---

## 五、附录

### 5.1 验证测试运行方式

```bash
cd /workspace
python -m quant_opt_20260621.run_verification
```

### 5.2 因子表达式引擎支持的算子

| 类别 | 算子 |
|------|------|
| 字段 | Close, Open, High, Low, Volume, Amount |
| 时序 | Ts_Mean, Ts_Sum, Ts_Std, Ts_Max, Ts_Min, Ts_Rank, Delta, Delay, WMA, EMA |
| 横截面 | Rank, Zscore, Scale |
| 二元 | Add, Sub, Mul, Div, Max, Min, +, -, *, / |
| 二元时序 | Corr, Cov |
| 其他 | Returns, 一元正负号 |

### 5.3 借鉴来源链接

- VectorBT: https://vectorbt.dev/
- AKQuant: https://github.com/akfamily/akquant
- Qlib: https://github.com/microsoft/qlib
- Python Backtesting Landscape 2026: https://python.financial/
- FactorEngine 论文: https://arxiv.org/pdf/2603.16365

### 5.4 分支信息

- 分支名：`feat/quant-opt-20260621`
- 远程：`origin` (github.com/duhanjun/jingni-trader)
- 状态：已推送，**未合并到 main**（等待用户确认）
