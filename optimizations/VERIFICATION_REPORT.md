# 量化优化验证报告 (feat/quant-opt-20260624)

- **生成时间**: 2026-06-24 09:14:36
- **分支**: `feat/quant-opt-20260624`
- **测试结果**: 4/4 模块通过

> 本报告由 `optimizations/run_all_tests.py` 自动生成。所有验证代码位于 `/workspace/optimizations/`，未修改仓库任何既有文件。

## 一、优化项与借鉴来源

### OPT1：向量化/分组回测引擎
- **问题**: `skills/backtest-engine/scripts/adapters/native_adapter.py` 逐日 `signals[signals['date']==dt]` / `data[data['date']==dt]` 两次 O(n) 布尔掩码，整体 O(n_days × n_rows) ≈ O(n²)。
- **优化**: 进入循环前一次性 `groupby('date')` 构建 `data_by_date` / `signals_by_date` 字典，循环体内 O(1) 查找；保留逐日循环（cash/positions 状态机路径依赖，无法纯向量化，且不引入 numba）。
- **借鉴**: VectorBT「向量化优先、避免朴素循环」哲学；交易逻辑（卖先买后、T+1、涨跌停、佣金/印花税）与原始**逐字一致**，仅替换数据获取方式。
- **代码**: `optimizations/vectorized_backtest.py`（含 `run_original_backtest` 基准 + `VectorizedBacktest` 优化版）

### OPT2：因子表达式引擎
- **问题**: `skills/factor-engine` 在 `compute_a_share_factors` 中硬编码因子，扩展性差。
- **优化**: 用 Python `ast` 模块实现 Parser（白名单校验）+ Executor，支持字段/算术/函数，时序算子组内(groupby code)、横截面算子(groupby date)。
- **借鉴**: Qlib 表达式 DSL、WorldQuant Alpha101 算子集、AKQuant 轻量解析。
- **代码**: `optimizations/factor_expression_engine.py`

### OPT3：向量化 IC 分析 + 胜率修复
- **问题1**: `skills/factor-engine/engine.py` `_calc_ic` 逐日布尔掩码 O(n²)。
- **优化1**: `groupby('date').apply` + `scipy.stats.spearmanr` 向量化。
- **问题2**: `skills/backtest-engine/.../base_backtest.py` `calc_win_rate` 把买入成交（pnl 恒负）计入分母，胜率被低估。
- **优化2**: 仅统计 `action=='sell'` 成交：`win_rate = (sell 且 pnl>0)/sell 总数`。
- **借鉴**: Qlib 向量化 IC；QuantConnect LEAN 的 round-trip 平仓盈亏统计口径。
- **代码**: `optimizations/vectorized_ic.py`、`optimizations/metrics_fix.py`

## 二、测试结果

| 优化项 | 测试模块 | 结果 | 备注 |
|---|---|---|---|
| OPT1 | `test_vectorized_backtest.py` | ✅ PASS | 正确性+性能+边界，加速比 1.44x |
| OPT2 | `test_factor_expression.py` | ✅ PASS | 正确性+算术+错误处理 |
| OPT3a | `test_vectorized_ic.py` | ✅ PASS | 正确性+性能+边界，加速比 2.63x |
| OPT3b | `test_metrics_fix.py` | ✅ PASS | 修复正确性+边界 |

## 三、性能对比

| 优化项 | 原始耗时 | 优化耗时 | 加速比 | 数据规模 |
|---|---|---|---|---|
| OPT1 | 0.905s | 0.628s | **1.44x** | 80 stocks × 400 days |
| OPT3a | 0.408s | 0.155s | **2.63x** | 100 stocks × 300 days |

> 性能测试均在同一沙箱环境运行（pandas/numpy/scipy 原生实现，未使用 numba/vectorbt/qlib）。

**关于 OPT1 加速比说明**: OPT1 仅消除 O(n²) 布尔掩码（约节省 0.2s），逐日循环内的路径依赖逻辑（卖出/买入/市值计算中的 `day_data_map.loc[code]` 逐标的取价）在两版中完全一致、无法在不引入 numba 的前提下进一步向量化，因此是剩余主要耗时。cProfile 显示该部分 `.loc` 取价约占向量化版 60%+ 耗时。OPT3a 加速比更高，因其 IC 计算无路径依赖、可整体 groupby 向量化。

## 四、正确性结论

- **OPT1**: 原始与向量化回测在相同 data+signals 下，`equity_curve` 逐日一致（`np.allclose` rtol=1e-9）、`trades` 笔数与金额一致、最终现金一致；浮点误差 ~1e-10。
- **OPT2**: `MA/STD/SUM/REF/DELTA/TS_MAX/TS_MIN/CORR/COV/RANK/ABS/LOG` 与手动 pandas 实现最大相对差 < 1e-9；复合表达式 `RANK(-MA(Close,5))` 语义正确；未知函数/字段/节点抛 `ValueError`。
- **OPT3a**: 向量化 IC 与原始逐日 IC 在相同数据上点数一致、最大绝对差 < 1e-9。
- **OPT3b**: 修正胜率仅统计 sell 成交，等于 `(pnl>0 的 sell)/sell 总数`；原始口径因含买入(pnl恒负)而低估。

## 五、待用户确认事项

1. **OPT1 是否合并入 `native_adapter.py`**: 当前仅在 `optimizations/` 验证，未改动原文件。确认无误后可替换原 `run_backtest` 的数据获取部分（交易逻辑保持不变）。
2. **OPT2 是否作为 factor-engine 的新后端**: 表达式引擎目前独立，是否接入 `compute_a_share_factors` / 配置化因子定义需确认。
3. **OPT3 IC 向量化是否替换 `_calc_ic`**: 确认后可直接替换 `engine.py` 中 `_calc_ic`。
4. **OPT3 胜率修复是否替换 `BaseBacktestMetrics.calc_win_rate`**: 修正会改变历史胜率口径，需确认是否同时提供 `calc_win_rate_round_trip`（按完整买卖对统计）作为更严谨版本。
5. **性能基准**: 加速比受沙箱 CPU/数据规模影响，生产数据规模下建议复测。

## 六、文件清单

```
optimizations/
├── data_generator.py            # 合成 A 股数据 + MA 交叉信号生成器
├── vectorized_backtest.py       # OPT1: 原始 + 向量化回测
├── factor_expression_engine.py  # OPT2: ast 表达式引擎
├── vectorized_ic.py             # OPT3a: 向量化 IC
├── metrics_fix.py               # OPT3b: 胜率修复
├── run_all_tests.py             # 总测试入口 + 报告生成
├── VERIFICATION_REPORT.md       # 本报告
└── tests/
    ├── test_vectorized_backtest.py
    ├── test_factor_expression.py
    ├── test_vectorized_ic.py
    └── test_metrics_fix.py
```
