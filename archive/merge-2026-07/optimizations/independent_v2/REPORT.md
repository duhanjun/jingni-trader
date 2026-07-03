# jingni-trader 量化交易优化验证报告 (Independent v2)

**执行日期**: 2026-06-21
**分支**: `feat/quant-opt-20260621`
**验证目录**: `optimizations/independent_v2/`
**测试结果**: 37/37 全部通过

> 本报告为独立验证版本 v2，与同分支已有的 `optimizations/expression_factors/`、
> `optimizations/enhanced_metrics/` 等模块并存，互不干扰。

---

## 一、学习项目清单及核心亮点

本次联网调研覆盖 GitHub、arXiv、QuantConnect、Papers with Code、CSDN、juejin 等平台，
筛选出以下 3 个最具借鉴价值的开源项目/研究方向：

### 1. Microsoft Qlib (GitHub Star: 15k+)
- **定位**: AI 驱动的量化投研平台，聚焦 A 股
- **核心亮点**:
  - **Alpha158 因子库**: 158 个标准化因子，覆盖 K 线形态、趋势、波动、位置、价量统计 6 大类
  - **IC/RankIC 分析**: 标准化的因子有效性评估流程
  - **Workflow 实验管理**: 基于 YAML 配置的实验追踪
- **借鉴价值**: jingni-trader 现有 factor-engine 仅 ~20 个基础技术指标，缺少系统性 Alpha 因子库

### 2. VectorBT (GitHub Star: ~7k)
- **定位**: 向量化回测框架，100-1000x 快于事件驱动框架
- **核心亮点**:
  - **纯向量化**: 用 NumPy/Numba 一次性计算整个时间序列
  - **参数扫描**: 可在秒级测试数千种参数组合
  - **Portfolio.from_signals**: 信号驱动的组合回测
- **借鉴价值**: jingni-trader 现有 native_adapter.py 用 `iterrows()` 逐日循环，性能瓶颈明显

### 3. CogAlpha / AlphaBench / LLM-MCTS Alpha Mining (arXiv 2025-2026)
- **定位**: LLM 驱动的公式化 Alpha 因子自动挖掘
- **核心亮点**:
  - **LLM + MCTS**: 用大语言模型生成可解释的因子公式
  - **代码级因子表示**: 因子以可执行代码形式存在
  - **IC 反馈驱动**: 用回测 IC 作为搜索奖励信号
- **借鉴价值**: 与 jingni-trader 的 LLM 驱动哲学一致，未来可作为 factor-engine 的智能扩展

### 4. 风控最佳实践 (综合来源)
- **Kelly Criterion**: f* = (bp - q) / b，半凯利降低破产风险
- **ATR 动态止损**: 用波动率自适应调整止损距离
- **回撤断路器**: 分级降仓机制（5%预警 / 10%降仓 / 20%清仓）

---

## 二、可借鉴的方向列表

| # | 优化方向 | 借鉴来源 | 现有问题 | 预期收益 | 优先级 |
|---|---------|---------|---------|---------|-------|
| 1 | 向量化回测引擎 | VectorBT | native_adapter 用 iterrows() 逐日循环 | 17x+ 加速 | 高 |
| 2 | Alpha158 因子库 | Qlib | 仅 ~20 个基础技术指标 | 64 个标准化因子 | 高 |
| 3 | IC 因子分析模块 | Qlib | 无因子有效性评估 | IC/RankIC/ICIR 分析 | 高 |
| 4 | 凯利仓位管理 | Kelly Criterion | 无仓位管理 | 自适应仓位 | 中 |
| 5 | ATR 动态止损 | Wilder ATR | 无止损机制 | 波动率自适应止损 | 中 |
| 6 | 回撤断路器 | 风控最佳实践 | 无回撤控制 | 分级降仓 | 中 |
| 7 | LLM 因子挖掘 | CogAlpha/AlphaBench | 因子需手工编写 | 自动生成因子（未来） | 低 |

---

## 三、已完成的验证测试

### 3.1 向量化回测引擎验证

**文件**: `optimizations/independent_v2/vectorized_backtest/vectorized_adapter.py`
**测试**: `optimizations/independent_v2/vectorized_backtest/test_vectorized_vs_native.py`

**测试结果** (8/8 通过):
- test_vectorized_basic_metrics_nonzero: PASS
- test_vectorized_vs_native_metrics_comparable: PASS
- test_t_plus_1_constraint_respected: PASS
- test_performance_vectorized_faster: PASS (**17.3x 加速**)
- test_empty_data: PASS
- test_empty_signals: PASS
- test_single_stock_single_day: PASS
- test_no_rebalance_signal: PASS

**性能对比** (20 股票 × 300 天 × 日频调仓):
```
[perf] native=448.1ms, vectorized=25.9ms, speedup=17.3x
```

### 3.2 Alpha158 因子库 + IC 分析验证

**文件**:
- `optimizations/independent_v2/alpha158_factors/alpha158_calculator.py`
- `optimizations/independent_v2/alpha158_factors/ic_analysis.py`

**测试结果** (9/9 通过):
- test_all_factors_calculable: PASS (64 个因子全部可计算)
- test_kline_factors_correctness: PASS
- test_trend_factor_ma5_correctness: PASS
- test_factor_info_returns_metadata: PASS
- test_ic_analysis_momentum_effective: PASS (动量因子被正确判定有效)
- test_ic_analysis_random_factor_not_effective: PASS (随机因子被正确判定无效)
- test_ic_summary_empty_series: PASS
- test_forward_returns_correctness: PASS
- test_factor_calculation_performance: PASS (64 因子×10股×250天: 268.7ms)

**因子分类** (64 个因子):
| 类别 | 因子数 | 示例 |
|------|-------|------|
| K线形态 | 9 | KMID, KLEN, KUP, KLOW |
| 趋势 | 10 | ROC5-60, MA5-60 |
| 波动 | 15 | STD5-60, MAX5-60, MIN5-60 |
| 位置 | 10 | IMAX5-60, IMIN5-60 |
| 价量统计 | 10 | CORR5-60, CORD5-60 |
| 成交量 | 10 | VSTD5-60, VMA5-60 |

### 3.3 风险控制模块验证

**文件**: `optimizations/independent_v2/risk_control/risk_manager.py`

**测试结果** (20/20 通过):
- KellySizer: 7 个测试（公式正确性、负期望、上限、估计、空数据、全胜、整手）
- ATRStopLoss: 4 个测试（ATR计算、多头止损、空头止损、追踪止损只上移）
- DrawdownCircuitBreaker: 7 个测试（正常/预警/紧急/清仓/回撤计算/权重调整/状态字典）
- RiskManager: 2 个集成测试（综合评估、空数据）

**风控规则**:
| 回撤幅度 | 状态 | 仓位乘数 |
|---------|------|---------|
| < 5% | NORMAL | 1.0 |
| 5%-10% | WARN_REDUCE | 0.75 |
| 10%-20% | EMERGENCY_REDUCE | 0.50 |
| ≥ 20% | HALT | 0.0 |

---

## 四、待用户确认的优化建议

以下优化方案已在 `feat/quant-opt-20260621` 分支验证通过，**未合并到 main**，等待用户确认：

### 建议 1: 将向量化回测适配器合入 backtest-engine（高优先级）
- **操作**: 将 `vectorized_adapter.py` 移至 `skills/backtest-engine/scripts/adapters/`
- **收益**: 17x+ 回测加速，更完整的净值曲线
- **风险**: 低，接口与现有 BaseBacktestEngine 兼容

### 建议 2: 将 Alpha158 因子库合入 factor-engine（高优先级）
- **操作**: 将 `alpha158_calculator.py` 移至 `skills/factor-engine/scripts/adapters/`
- **收益**: 因子数量从 20 → 84，新增 IC 分析能力
- **风险**: 低，作为新增计算器并存

### 建议 3: 将风控模块合入 portfolio-risk-engine（中优先级）
- **操作**: 将 `risk_manager.py` 移至 `skills/portfolio-risk-engine/scripts/`
- **收益**: 新增凯利仓位、ATR 止损、回撤断路器
- **风险**: 中，需与现有组合优化器集成测试

### 建议 4: 探索 LLM 因子挖掘（低优先级，未来方向）
- **操作**: 参考 CogAlpha/AlphaBench，在 factor-engine 中新增 LLM 驱动的因子生成器
- **风险**: 高，需 LLM API 成本

---

## 五、文件清单

```
optimizations/independent_v2/
├── __init__.py
├── data_fixtures.py                          # 合成数据生成器
├── jingni_compat.py                          # 兼容包装层
├── run_all_tests.py                          # 统一测试运行器
├── REPORT.md                                 # 本报告
├── vectorized_backtest/
│   ├── __init__.py
│   ├── vectorized_adapter.py                 # 向量化回测适配器
│   └── test_vectorized_vs_native.py          # 8 个测试
├── alpha158_factors/
│   ├── __init__.py
│   ├── alpha158_calculator.py                # Alpha158 因子计算器 (64 因子)
│   ├── ic_analysis.py                        # IC/RankIC/ICIR 分析
│   └── test_alpha158.py                      # 9 个测试
└── risk_control/
    ├── __init__.py
    ├── risk_manager.py                       # Kelly + ATR + 回撤断路器
    └── test_risk_control.py                  # 20 个测试
```

**测试统计**: 37 个测试全部通过

---

## 六、参考来源

- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [VectorBT](https://vectorbt.dev/)
- [CogAlpha (arXiv:2511.18850)](https://arxiv.org/pdf/2511.18850v3)
- [AlphaBench (ICLR 2026)](http://www.cs.cityu.edu.hk/~cliu644/HomePage/doc/AlphaBench/AlphaBench_PDF.pdf)
- [Alpha158 因子公式](https://fund.bigquant.com/wiki/doc/nODcNAKYPJ)
- [Kelly Criterion](https://licai.cofool.com/ask/qa_4683597.html)
- [ATR 动态止损](https://tradinghack.net/risk-management/max-drawdown-complete-guide/)
- [量化风控清单](https://gov.capital/the-ironclad-quants-checklist-7-mandatory-trading-rules-to-crank-down-your-losses-by-85/)
