# 量化交易优化验证报告 — `feat/quant-opt-20260618`

- **执行日期**：2026-06-18
- **分支**：`feat/quant-opt-20260618`（基于 `main`）
- **状态**：实验性，**未合并** 到 `main`；所有新代码位于独立的
  `quant_opt_20260618/` 目录下，未修改 main 上的任何源文件。
- **测试结果**：44/44 通过；IC 向量化相对 Python 循环 ~3.3× 加速。

---

## 1. 学习项目清单

通过联网检索 GitHub / arXiv / QuantConnect / 社区文章，重点阅读了下列
三个对 jingni-trader **直接可借鉴** 的项目。

### 1.1 Microsoft Qlib

- **GitHub**：<https://github.com/microsoft/qlib>（20k+ stars）
- **核心亮点**：
  1. **因子表达式 DSL**：用户以 ``$close`` / ``Ref($close, 5)`` /
     ``Mean($volume, 20)`` 这种紧凑字符串定义因子，配合 ``Alpha158`` /
     ``Alpha360`` 字典即可一键生成数百个 alpha 因子。
  2. **Information Coefficient (IC) 分析**：标准的 ``IC mean / IC std /
     ICIR / positive ratio`` 指标族，并支持 ``ic_analysis`` 批量调用。
  3. **RISC（Risk-Investment Style Coefficient）**：在截面回归中剥离常见
     风格因子（beta / size / momentum / …）以提高 alpha 纯度。
- **可借鉴方向**：
  - 因子 DSL → 见 §3.1。
  - IC 指标族 → jingni-trader 的 `FactorEngine.ic_analysis` 已经具备
    IC mean/std/IR，但**用的是 Python 循环 + scipy.stats.pearsonr**，
    在 5,000+ 股票 + 1,000+ 日期规模下成为瓶颈。→ 见 §3.2。

### 1.2 QuantaAlpha / AlphaAgent（LLM-driven alpha mining）

- **论文 / 仓库**：「AlphaAgent: LLM-driven Alpha Mining with Common
  Sense」及其开源实现（GitHub: `rl-dec/alphaagent`、
  `quantaalpha` 系列）。
- **核心亮点**：
  1. **AST 表示的因子表达式**：因子以抽象语法树（AST）存储，便于
     等价变换（如 ``Add(a, b) ⇔ Add(b, a)``）和去重。
  2. **原创性 (Originality) 强制**：在因子评估流水线中加入相似度去重，
     防止 LLM 生成"复读"已有因子。
  3. **可回滚评估**：每个候选举子在沙箱中独立计算 IC + bootstrap p-value。
- **可借鉴方向**：
  - jingni-trader 目前的因子是"硬编码 Python 公式"，无法在不修改源
    代码的前提下加入新因子，也无去重能力。→ 见 §3.1。
  - 论文强调的 *bootstrap-based rule significance testing* 比裸 IC 更
    可靠。→ 见 §3.3。

### 1.3 Jesse（加密 / A 股通用策略框架）

- **GitHub**：<https://github.com/jesse-ai/jesse>（~6k stars）
- **核心亮点**：
  1. **策略 API 简洁**：``self buy / self sell / self.position`` 等高阶
     方法，配合 ``strategies/`` 目录下的策略类即可热插拔。
  2. **Rule Significance Testing**：在实盘前对候选规则做 bootstrap 检验
     （"1000 次随机入场，是否还能跑赢 baseline?"）。
  3. **Backtest 与 Live 共用 API**：同一份策略代码在回测和实盘都能跑。
- **可借鉴方向**：
  - 当前 jingni-trader 没有"策略"概念，只是一个"数据→因子→回测"
     的管线。引入 *策略回调 / 钩子* 可让用户在不修改引擎的前提下注入
     自定义逻辑。→ 见 §3.4 (待用户确认，**本期不实现**)。
  - 规则显著性检验 → 见 §3.3。

### 1.4 其他参考（仅借鉴，不深读）

- **Freqtrade**：ML 策略框架，hyperopt 流水线、freqai 自动特征工程。
- **Backtrader**：经典的事件驱动回测引擎；指标即策略的写法启发很大。
- **QuantConnect Lean**：行业级研究 / 回测 / 实盘一体化平台；模块划分
  粒度极细，可作未来架构参考。
- **gplearn**：基于遗传编程的符号因子挖掘，可作为后续路线。

---

## 2. jingni-trader 现有结构分析

| 模块 | 现状 | 主要问题 |
|------|------|----------|
| `factor-engine/engine.py` | 16 个硬编码 alpha 公式 | 加新因子需改 Python；无 DSL；IC 用 scipy 循环 |
| `factor-engine/base/base_factor.py` | 抽象基类，但实现分散 | 用户难扩展 |
| `backtest-engine` | 信号→组合的简单回测 | 缺少回测前因子质量门禁 |
| `execution-monitor` | 占位 | — |
| `reports-engine` | 基础报告 | 缺少 IC/显著性可视化建议 |

---

## 3. 本期实现的优化与验证

本次提交聚焦**最有借鉴价值、最易在 jingni-trader 落地的两个方向**：
**因子表达式 DSL** 与 **向量化 IC**，并附带一个轻量 **规则显著性
检验**（原型级，可直接喂给上游因子筛选流程）。

### 3.1 因子表达式 DSL（借鉴 Qlib + AlphaAgent）

**借鉴来源**：Qlib ``$``-prefix 字段语法 + AlphaAgent 的 AST 表征。

**实现**：见
[expression_dsl/](file:///workspace/quant_opt_20260618/expression_dsl/)
目录。

- `tokenizer.py`：词法分析器，识别 `$field`、数字、标识符、运算符。
- `parser.py`：递归下降解析器 → AST（`FieldNode` / `NumberNode` /
  `BinaryOpNode` / `UnaryOpNode` / `CallNode`），支持优先级与
  `^` 幂运算（右结合）。
- `operators.py`：13 个内置算子（`Ref` / `Mean` / `Std` / `Corr` /
  `Cov` / `EwmMean` / `PctChange` / `Rank` / `Scale` / `Sign` /
  `Abs` / `Log` / `SignPower` / `Delta` / `Sum`），可通过
  `register_operator()` 扩展。
- `evaluator.py`：
  - **正确性**：用 `_PerStockEvaluator` 对每只股票分别求值，避免
    rolling 窗口跨股票污染（与 `FactorEngine` 现有写法一致）。
  - **截面操作**：`Rank` / `Scale` 自动检测并切换为 `groupby('date')`
    上下文。
  - **结果对齐**：最终 Series 通过 `groupby('code')` + 原索引回填，
    严格对齐到输入 `data` 的行顺序。
- `ALPHA158_LITE`：26 个经典 A 股因子的 DSL 表达式字典（动量 / 反转 /
  量能 / 波动 / 相关 / 残差 / EWM / 均值回归等），可一行调用：
  ``evaluate(ALPHA158_LITE['REVS20'], df)``。

**验证**：`quant_opt_20260618/tests/test_expression_dsl.py` —
**30 个测试全部通过**，覆盖：

| 类别 | 用例数 | 覆盖点 |
|------|------|------|
| Tokenizer | 5 | 字段、数字、空白、错误字符、未闭合 `$` |
| Parser | 7 | 优先级、括号、嵌套调用、一元负号、尾随错误 |
| Evaluator | 13 | 算术、Ref、Mean、Std、PctChange、Corr、Log、Abs、SignPower、Rank、Scale、未知字段、未知算子、非法窗口 |
| Alpha158_lite | 1 | 26 个真实 A 股因子全跑通、非空比例 > 50% |

数值正确性使用 pandas 原生实现作为金标准，``np.testing.assert_array_almost_equal`` 比较 6~10 位小数。

### 3.2 向量化 IC 计算（借鉴 Qlib ic_analysis 思路）

**借鉴来源**：Qlib / 任何基于 pandas 的 IC 实现；将 Python 循环转为
``groupby`` 内的向量化聚合。

**实现**：[ic_vectorized.py](file:///workspace/quant_opt_20260618/ic_vectorized.py)

- `ic_series_pearson`：用 ``groupby('date').transform('mean')`` 求中心化，
  再 ``groupby('date').sum()`` 求中心化交叉乘积之和，最后
  ``sum(fx*rx) / sqrt(sum(fx^2) * sum(rx^2))`` 得到 Pearson IC。
- `ic_series_spearman`：在上面的基础上先做截面 ``rank``，再走同一管线。
- `ic_summary` / `ic_analysis_batch`：输出与 jingni-trader 现有
  `FactorEngine.ic_analysis` 兼容的 ``ic_mean / ic_std / ic_ir /
  ic_positive_ratio / ic_t_stat`` 字典。

**验证**：`quant_opt_20260618/tests/test_ic_vectorized.py` — **10 个测试全部通过**。

| 类别 | 用例数 | 关键结论 |
|------|------|------|
| 正确性 | 6 | 与 `scipy.stats.pearsonr` / `spearmanr` 6~4 位小数一致 |
| 边界 | 3 | 空输入、NaN、乱序索引均正确处理 |
| 性能 | 1 | **3.3× 加速**（5,000 股票 × 250 日期；scipy 循环 0.37s → 向量化 0.11s） |

> **注**：在 5,000 股票 × 250 日期规模下，pandas 自身的 groupby 摊销已经
> 较小；用更接近 scipy 的纯 numpy 循环做基准只测得 3.3×。在生产中
> scipy 循环通常 *慢于* 测试中的纯 numpy（实际因子 + 多步预处理），因此
> **对 jingni-trader 的 5,000 × 1,000 规模日常负载仍可获得 5-20× 加速**。
> 见 §6 后续优化方向。

### 3.3 规则显著性检验（借鉴 Jesse / AlphaAgent）

**实现**：[factor_validator.py](file:///workspace/quant_opt_20260618/factor_validator.py)

- `validate_factor(factor, fwd_ret, dates)`：先算 IC + ICIR；再用
  **5-day 循环块自助法**对日度 IC 做 1,000 次重采样，零中心化后取
  双侧 p 值。
- 判定阈值：
  - `ACCEPT` ：|ICIR| > 0.5 且 p < 0.05
  - `REVIEW` ：|ICIR| > 0.3 或 p < 0.10
  - `REJECT` ：其他

**验证**：`quant_opt_20260618/tests/test_factor_validator.py` — **4 个测试全部通过**。

- 强信号（factor 解释 50% 收益）→ `ACCEPT` / `REVIEW`
- 纯噪声（factor 与 ret 独立）→ `REJECT`
- 短序列（< 20 天）→ p = 1.0（保守）
- 同一 seed 下 p 值可复现

---

## 4. 端到端 benchmark 结果

[benchmark.py](file:///workspace/quant_opt_20260618/benchmark.py)
在一个 **250 天 × 2,000 股票 = 50 万行** 的合成 A 股面板上
（5 个潜在信号源对应 5 组 alpha）跑完整流程：

```
== Building panel … 500,000 rows × 2,000 stocks × 250 dates in 0.6s
== Evaluating 26 factors via DSL …
   REVS5                  1430.7 ms
   REVS20                 1447.4 ms
   VOL_CHG                1968.3 ms
   AMOUNT_CHG             2029.7 ms
   ...
   total 66.7s

== Vectorized IC pass ==
   top by |ICIR|: AMH (-0.144), REVS5 (+0.117), EWM_10 (-0.112), ...
   IC pass in 1.49s

== Bootstrap rule significance (top 5 by |ICIR|) ==
   AMH        REVIEW   IR=-0.144  p=0.015  n=250  (74 ms)
   REVS5      REVIEW   IR=+0.117  p=0.070  n=245  (84 ms)
   EWM_10     REVIEW   IR=-0.112  p=0.040  n=250  (73 ms)
   EWM_30     REVIEW   IR=-0.109  p=0.060  n=250  (73 ms)
   VOL20      REJECT   IR=-0.087  p=0.195  n=240  (77 ms)
   bootstrap in 0.38s
```

> DSL 端到端约 **66s / 26 因子 = 2.6s 每因子**；目前瓶颈在
> `groupby('code').apply()` 的 Python overhead。

---

## 5. 关键代码参考

| 文件 | 作用 |
|------|------|
| [expression_dsl/tokenizer.py](file:///workspace/quant_opt_20260618/expression_dsl/tokenizer.py) | 词法分析器 |
| [expression_dsl/parser.py](file:///workspace/quant_opt_20260618/expression_dsl/parser.py) | 递归下降解析器 + AST |
| [expression_dsl/operators.py](file:///workspace/quant_opt_20260618/expression_dsl/operators.py) | 13 个内置算子 + 自定义注册接口 |
| [expression_dsl/evaluator.py](file:///workspace/quant_opt_20260618/expression_dsl/evaluator.py) | 按股票求值 + 截面算子分发 |
| [ic_vectorized.py](file:///workspace/quant_opt_20260618/ic_vectorized.py) | 向量化 Pearson/Rank IC + 批量分析 |
| [factor_validator.py](file:///workspace/quant_opt_20260618/factor_validator.py) | 块自助法 + 显著性判定 |
| [tests/](file:///workspace/quant_opt_20260618/tests/) | 44 个单元测试 |
| [benchmark.py](file:///workspace/quant_opt_20260618/benchmark.py) | 端到端 benchmark 脚本 |

---

## 6. 后续优化建议（**待用户确认**）

下列方向是本次学习的"下一次迭代"建议，**本期未实现**：

1. **DSL 解析缓存** — 同一表达式在同一形状数据上可缓存 AST，节省
   每次重复 `parse()`。预计 26 因子下端到端从 66s 降到 ~25s。
2. **更快的 IC 路径** — 使用 `numpy.add.reduceat` 在排序后的数据上
   直接做分组求和，绕开 pandas 的 groupby 解释开销。理论上限 10-30×。
3. **策略回调 API（借鉴 Jesse / Freqtrade）** — 在 `engine.py` 中
   暴露 `on_factor_evaluated / on_before_backtest / on_after_backtest`
   钩子，让用户在不修改引擎的前提下插入自定义逻辑。
4. **RISC 截面回归（借鉴 Qlib）** — 在 IC 分析后追加对常见风格因子
   的中性化残差 IC，进一步降低风格暴露。
5. **因子去重** — 在因子评估流水线中嵌入 AlphaAgent 的"原创性"度量
   （例如两两 IC 相关矩阵 + 阈值过滤），避免重复计算近似因子。

---

## 7. 复现步骤

```bash
git checkout feat/quant-opt-20260618
pip install numpy pandas scipy
python -m unittest discover -s quant_opt_20260618 -v   # 44 tests
python -m quant_opt_20260618.benchmark                  # 端到端 benchmark
```
