# jingni-trader 量化优化验证报告

**执行日期**: 2026-06-21
**分支**: `feat/quant-opt-20260621`
**执行人**: 自动化学习与优化流程

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib (GitHub Star: 15k+)
- **定位**: AI 驱动的量化投研平台，覆盖数据→因子→模型→回测→组合全流程
- **核心亮点**:
  - **因子表达式 DSL**: Alpha158/Alpha101 因子库，支持 `Ref/Mean/Std/Rank/Corr` 等算子组合
  - **Walk-forward Validation**: 滚动训练验证框架，防止过拟合
  - **二进制数据存储**: 列式存储加速大规模数据读取
  - **ML Pipeline**: 内置 LightGBM/Transformer 模型模板
- **可借鉴方向**: 因子表达式引擎、滚动验证框架

### 2. VectorBT / VectorBT PRO
- **定位**: 向量化回测框架，研究阶段极致性能
- **核心亮点**:
  - **全向量化**: NumPy 数组运算替代逐 bar Python 循环
  - **Portfolio.from_signals()**: 信号驱动的快速回测
  - **参数扫描**: 跨多资产大规模参数搜索
- **可借鉴方向**: 向量化回测引擎、性能优化思路

### 3. akquant (Rust + Python, akfamily, Star: 1.5k+)
- **定位**: 下一代高性能混合框架，Rust 内核 + Python 接口
- **核心亮点**:
  - **因子表达式引擎**: Polars 驱动，支持 `Rank(Ts_Mean(Close, 5))` 风格公式
  - **Walk-forward Validation**: 内置滚动训练框架
  - **TA-Lib 双后端**: python/rust 兼容，103 个指标
  - **多进程网格搜索**: 策略参数并行优化
- **可借鉴方向**: 因子表达式引擎、Polars 加速

### 4. NautilusTrader
- **定位**: 事件驱动回测/实盘一体化，Rust 核心强调回测/实盘一致性
- **核心亮点**:
  - **订单簿真实性**: 严格的订单类型、成交、延迟建模
  - **回测/实盘对等**: 同一策略代码可无缝切换回测与实盘
  - **前视偏差严格防护**: 信号 T 日生成 → T+1 日成交
- **可借鉴方向**: 前视偏差防护、回测/实盘一致性

### 5. FactorEngine (arXiv:2603.16365, 2026)
- **定位**: LLM 驱动的程序级因子挖掘框架
- **核心亮点**:
  - **因子即代码**: 因子表示为 Turing-complete 代码，可审计
  - **LLM 引导搜索**: 大模型提出宏观变异 + 贝叶斯优化参数
  - **知识注入**: 从研报自动提取因子并转化为可执行代码
- **可借鉴方向**: 未来因子自动挖掘方向（本次未实现，作为长期规划）

---

## 二、jingni-trader 现状分析与可借鉴方向

### 2.1 现有架构
```
engine.py (主调度器)
├── skills/data-engine/          数据获取 (tushare/baostock/akshare/...)
├── skills/factor-engine/        因子计算 (pandas_ta/talib)
├── skills/strategy-model-engine/ 策略建模
├── skills/backtest-engine/      回测引擎 (native/backtrader/rqalpha/gm)
├── skills/portfolio-risk-engine/ 组合优化
├── skills/execution-monitor-engine/ 实盘执行
└── skills/reports-engine/       报告生成
```

### 2.2 发现的问题

| 模块 | 文件 | 问题 | 严重度 |
|------|------|------|--------|
| 回测引擎 | [native_adapter.py](file:///workspace/skills/backtest-engine/scripts/adapters/native_adapter.py) | 逐行 `iterrows` 遍历，性能差 | 高 |
| 回测引擎 | native_adapter.py | 信号当日 close 生成 + 当日 close 成交，**前视偏差** | 高 |
| 回测引擎 | native_adapter.py | `t_plus_1` 参数传入但未实际使用 | 中 |
| 回测引擎 | native_adapter.py | 未处理停牌（仅处理涨跌停） | 中 |
| 回测引擎 | native_adapter.py | `pnl` 字段 buy 记为负、sell 记为正，胜率计算无意义 | 中 |
| 因子引擎 | [pandas_ta_calculator.py](file:///workspace/skills/factor-engine/scripts/adapters/pandas_ta_calculator.py) | 因子硬编码在 if-elif 链，添加新因子需改源码 | 高 |
| 因子引擎 | [engine.py](file:///workspace/skills/factor-engine/engine.py) | 中性化逐日循环 + LinearRegression，性能差 | 中 |
| 因子引擎 | engine.py | 无法表达复合因子（如 `Rank(Ts_Mean(Close,5))`） | 中 |
| 绩效指标 | [base_backtest.py](file:///workspace/skills/backtest-engine/scripts/base/base_backtest.py) | `calc_sharpe` 用 `returns.mean()*252`，与 `calc_annual_return` 不一致 | 高 |
| 绩效指标 | base_backtest.py | `calc_win_rate` 基于单笔 pnl，buy 的 pnl 是负的，无意义 | 高 |
| 绩效指标 | base_backtest.py | 缺少 Information Ratio、Tracking Error、分月收益 | 中 |
| 主调度器 | [engine.py](file:///workspace/engine.py) | `parse_intent` 硬编码日期（"近3年"→2021-2024，但今天是 2026） | 中 |
| 主调度器 | engine.py | 每次执行 stage 都 `del sys.modules['scripts']`，可能丢状态 | 低 |

### 2.3 可借鉴方向列表

| # | 优化方向 | 借鉴来源 | 价值 | 实现难度 |
|---|---------|---------|------|---------|
| 1 | 向量化回测引擎 + 前视偏差修复 | VectorBT + NautilusTrader | 高 | 中 |
| 2 | 因子表达式引擎 | Qlib + akquant | 高 | 中 |
| 3 | 回测指标修正与扩展 | empyrical + quantstats | 高 | 低 |
| 4 | Walk-forward 滚动验证 | Qlib + akquant | 中 | 中 |
| 5 | LLM 因子自动挖掘 | FactorEngine | 中 | 高 |
| 6 | 主调度器动态日期解析 | 通用最佳实践 | 中 | 低 |

---

## 三、本次已完成的验证测试

本次在 `feat/quant-opt-20260621` 分支的 `experiments/quant_opt_20260621/` 目录下实现了 3 个优化模块，并完成全部验证测试。

### 3.1 优化模块清单

| 文件 | 行数 | 说明 |
|------|------|------|
| [vectorized_backtest.py](file:///workspace/experiments/quant_opt_20260621/vectorized_backtest.py) | ~330 | 向量化回测引擎，修复前视偏差、T+1、涨跌停 |
| [factor_expression_engine.py](file:///workspace/experiments/quant_opt_20260621/factor_expression_engine.py) | ~440 | 因子表达式引擎，支持 Alpha101 风格 DSL |
| [metrics_fix.py](file:///workspace/experiments/quant_opt_20260621/metrics_fix.py) | ~280 | 回测指标修正，新增 IR/TE/盈亏比/分月收益 |
| [test_vectorized_backtest.py](file:///workspace/experiments/quant_opt_20260621/test_vectorized_backtest.py) | ~360 | 向量化回测测试（正确性/前视偏差/T+1/涨跌停/性能/边界） |
| [test_factor_expression.py](file:///workspace/experiments/quant_opt_20260621/test_factor_expression.py) | ~240 | 因子表达式测试（解析/算子/复合/Corr/Alpha101/边界） |
| [test_metrics_fix.py](file:///workspace/experiments/quant_opt_20260621/test_metrics_fix.py) | ~260 | 指标修正测试（一致性/Sharpe/MDD/IR/胜率/边界/对比） |
| [run_validation.py](file:///workspace/experiments/quant_opt_20260621/run_validation.py) | ~160 | 端到端集成验证脚本 |

### 3.2 测试结果汇总

#### 3.2.1 向量化回测引擎测试

```
[Test 1] 基本正确性测试          [PASS]  净值合理, 指标完整
[Test 2] 前视偏差测试            [PASS]  信号 T 日生成, T+1 日 open 成交
[Test 3] T+1 限制测试            [PASS]  sell-first 顺序自然保证 T+1
[Test 4] 涨跌停限制测试          [PASS]  涨停拒绝买入, 跌停拒绝卖出
[Test 5] 性能对比测试            [PASS]  向量化 142.8ms vs 逐行 1449.8ms, 加速 10.15x
[Test 6] 边界条件测试            [PASS]  空数据/单股/全零信号/缺失日期
```

**性能对比** (50 只股票 × 500 日 = 25000 行):
- 向量化回测: **142.8 ms**
- 逐行循环回测 (模拟旧 native_adapter): **1449.8 ms**
- **加速比: 10.15x**

#### 3.2.2 因子表达式引擎测试

```
[Test 1] 表达式解析测试          [PASS]  字段/函数/嵌套/二元/双参数
[Test 2] 简单因子计算正确性      [PASS]  Ref/Ts_Mean/Ts_Std 与手算一致
[Test 3] 复合表达式测试          [PASS]  动量/反转/Rank 截面排名
[Test 4] Corr 双参数算子测试     [PASS]  Corr(Volume, Close, 10) 正确
[Test 5] Alpha101 风格表达式     [PASS]  alpha_006/012/101 全部计算成功
[Test 6] 边界条件测试            [PASS]  空数据/单股/未知字段/未知算子
```

**支持的算子**: `Ref, Delta, Ts_Mean, Ts_Std, Ts_Max, Ts_Min, Ts_Sum, Ts_Rank, EMA, Corr, Cov, Abs, Log, Sign, Rank`

**支持的字段**: `Open, High, Low, Close, Volume, Amount, Turnover, Return` (可扩展注册)

#### 3.2.3 回测指标修正测试

```
[Test 1] 年化收益一致性测试      [PASS]  total_return 与 annual_return 一致
[Test 2] Sharpe 一致性测试       [PASS]  使用几何年化, 与 annual_return 一致
[Test 3] 最大回撤正确性测试      [PASS]  峰值 120 -> 谷底 85, MDD=-29.17%
[Test 4] Information Ratio 测试  [PASS]  TE/IR 与手算一致
[Test 5] Trade pair 胜率测试     [PASS]  修复旧实现 buy/sell pnl 无意义问题
[Test 6] 边界条件测试            [PASS]  单点/全零/空 trades/safe_div
[Test 7] 与旧实现对比测试        [PASS]  新实现修正了 Sharpe 与 annual_return 一致性
[Test 8] 完整指标计算测试        [PASS]  全部新增指标正确
```

**关键 bug 修复对比** (504 日数据, annual_ret=12%, vol=18%):
| 指标 | 旧实现 (base_backtest.py) | 新实现 (metrics_fix.py) | 差异说明 |
|------|--------------------------|------------------------|---------|
| annual_return | -23.38% | -23.38% | 一致 |
| Sharpe | -1.6683 (算术年化) | -1.5585 (几何年化) | **修正**: 与 annual_return 一致 |

#### 3.2.4 端到端集成验证

```
数据规模: 30 只股票 × 300 日 = 9000 行
因子计算: 9 个 Alpha101 风格因子, 耗时 90.4 ms
回测执行: 670 笔交易, 耗时 90.5 ms
绩效指标: 22 项指标全部计算成功
```

**端到端绩效摘要**:
- 总收益: -4.95%
- 年化收益: -4.18%
- Sharpe: -0.5227
- 最大回撤: -22.10%
- 信息比率: -0.6510 (相对基准)
- 交易胜率: 51.21% (基于 trade pair)
- 盈亏比: 0.9168

---

## 四、对比分析

### 4.1 回测引擎对比

| 特性 | 旧 native_adapter | 新 vectorized_backtest |
|------|------------------|----------------------|
| 成交价 | 当日 close (前视偏差) | T+1 open (无前视偏差) |
| T+1 限制 | 参数未使用 | sell-first 顺序自然保证 |
| 涨跌停 | 仅过滤 | 涨停拒买 + 跌停拒卖 |
| 停牌 | 未处理 | 拒绝成交 |
| 性能 (25k 行) | 1449.8 ms | 142.8 ms (**10x 加速**) |
| 仓位模式 | 仅等权 | 等权 + target_weight |
| 滑点 | 买入加 | 买入加 + 卖出减 |

### 4.2 因子引擎对比

| 特性 | 旧 PandasTaCalculator | 新 FactorExpressionEngine |
|------|----------------------|--------------------------|
| 添加新因子 | 改源码 if-elif | 写表达式字符串 |
| 复合因子 | 不支持 | 支持 `Rank(Ts_Mean(Close,5))` |
| 算子库 | 19 个技术指标 | 15 个时序/截面算子 + 可扩展 |
| Alpha101 | 不支持 | 支持 (alpha_006/012/101 验证通过) |
| 自定义算子 | 不支持 | `register_func()` 注册 |
| 自定义字段 | 不支持 | `register_field()` 注册 |

### 4.3 绩效指标对比

| 指标 | 旧 BaseBacktestMetrics | 新 metrics_fix |
|------|----------------------|---------------|
| Sharpe 年化 | 算术 (`mean*252`) | 几何 (与 annual_return 一致) |
| 胜率 | 单笔 pnl (无意义) | trade pair 配对 (真实盈亏) |
| Information Ratio | 无 | 有 |
| Tracking Error | 无 | 有 |
| 盈亏比 | 无 | 有 |
| 分月收益 | 无 | 有 |
| 滚动 Sharpe | 无 | 有 |
| Sortino | 有 (算术年化) | 有 (几何年化, 一致) |

---

## 五、待用户确认的优化建议

以下优化方向已通过验证测试，**等待用户确认后**方可合并到 main 分支：

### 建议 1: 替换 native_adapter 为向量化回测引擎 (高优先级)
- **改动**: 将 `vectorized_backtest.py` 集成为 `native_adapter.py` 的替代
- **收益**: 10x 性能提升 + 消除前视偏差 + 修复 T+1
- **风险**: 旧回测结果不可复现（成交价从 close 改为 next_open）
- **建议**: 提供 `trade_on` 配置项，默认 `next_open`，保留 `same_close` 兼容

### 建议 2: 引入因子表达式引擎 (高优先级)
- **改动**: 在 `factor-engine` 中新增 `expression_calculator.py`，与现有 `pandas_ta_calculator` 并存
- **收益**: 因子可扩展性大幅提升，支持 Alpha101 风格 DSL
- **风险**: 低（新增模块，不影响现有代码）
- **建议**: 优先合并，作为因子库扩展的基础设施

### 建议 3: 修正回测指标计算 (高优先级)
- **改动**: 修正 `base_backtest.py` 的 `calc_sharpe` 和 `calc_win_rate`，新增 IR/TE/盈亏比
- **收益**: 修复 Sharpe 不一致 bug，胜率计算有意义
- **风险**: 旧报告的 Sharpe 数值会变化（从算术年化改为几何年化）
- **建议**: 优先合并，bug 修复必须做

### 建议 4: 主调度器动态日期解析 (中优先级，未实现)
- **改动**: `engine.py` 的 `parse_intent` 用 `datetime.now()` 动态计算日期
- **收益**: 修复"近3年"硬编码为 2021-2024 的 bug
- **风险**: 低
- **建议**: 下次迭代实现

### 建议 5: Walk-forward 滚动验证框架 (中优先级，未实现)
- **改动**: 在 `strategy-model-engine` 中新增滚动训练验证
- **收益**: 防止模型过拟合
- **风险**: 中（需要改造模型训练流程）
- **建议**: 下次迭代实现

---

## 六、约束遵守说明

- ✅ 所有新代码位于 `feat/quant-opt-20260621` 分支的 `experiments/quant_opt_20260621/` 目录
- ✅ 未修改 main 分支任何代码
- ✅ 未执行 git merge 操作
- ✅ 分支已推送到 GitHub 远程仓库（仅 push，不合并）
- ✅ 验证报告已保存到本地文件系统

---

## 七、附录

### 7.1 测试运行命令
```bash
cd /workspace/experiments/quant_opt_20260621
python3 test_vectorized_backtest.py
python3 test_factor_expression.py
python3 test_metrics_fix.py
python3 run_validation.py
```

### 7.2 验证结果数据
详见同目录下 `validation_results.json`

### 7.3 参考资料
- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [VectorBT](https://vectorbt.dev/)
- [akquant](https://github.com/akfamily/akquant)
- [NautilusTrader](https://nautilustrader.io/)
- [FactorEngine 论文 (arXiv:2603.16365)](https://arxiv.org/abs/2603.16365)
- [The Python Backtesting Landscape 2026](https://python.financial/)
