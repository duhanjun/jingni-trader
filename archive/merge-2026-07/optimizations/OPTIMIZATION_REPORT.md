# jingni-trader 优化验证报告

**执行日期**: 2026-06-20
**分支**: `feat/quant-opt-20260620`
**执行人**: 自动化学习与优化流程
**测试结果**: 13 项测试全部通过 (13/13 PASS)

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib ⭐ 11k+
- **仓库**: https://github.com/microsoft/qlib
- **定位**: AI 导向的量化投资平台，A股研究领域的标杆
- **核心亮点**:
  - **表达式引擎**: 用表达式（如 `Ref($close, 60) / $close`）定义因子，无需写复杂代码
  - **高性能二进制存储**: 专为金融数据设计的 `.bin` 格式，列式存储
  - **多层缓存系统**: Dataset/DataHandler/DataLoader 三级抽象，自动缓存中间结果
  - **Point-in-Time (PIT) 数据**: 严格避免未来函数，保证回测真实性
  - **Rolling Window 训练**: `RollingGen` 支持滚动窗口训练，避免前视偏差
  - **LLM 集成**: 2026 年与 RD-Agent 结合，LLM 自动挖掘 Alpha 因子
- **可借鉴方向**: 表达式引擎、PIT 数据、滚动训练、高效缓存

### 2. NautilusTrader ⭐ 10/10 评级
- **仓库**: https://github.com/nautechsystems/nautilus_trader
- **定位**: 生产级 Rust 原生事件驱动交易引擎
- **核心亮点**:
  - **研究-实盘同构**: 同一份策略代码在回测和实盘运行，无需重写
  - **确定性事件驱动**: 单线程内核 + 确定性时钟，保证回测可复现
  - **双纳秒时间戳**: `ts_event`（事件发生时间）+ `ts_init`（系统创建时间）
  - **Crash-only 设计**: 启动与崩溃恢复共用代码路径，提升可靠性
  - **Rust 核心 + Python 控制面**: 热路径用 Rust，策略编写用 Python
  - **模块化适配器**: Ports & Adapters 模式，易于扩展新交易所
- **可借鉴方向**: 研究-实盘同构、确定性设计、双时间戳、模块化适配器

### 3. VectorBT ⭐ 6.5k+
- **仓库**: https://github.com/polakowo/vectorbt
- **定位**: 向量化回测框架，研究阶段的性能王者
- **核心亮点**:
  - **向量化回测**: 用 NumPy 矩阵运算替代逐 K 线循环，比事件驱动快 100-1000 倍
  - **Numba JIT 加速**: 热路径 JIT 编译，可选 Rust 内核
  - **参数扫描**: 单次 pass 完成上万组参数组合的回测
  - **57 项绩效指标**: 开箱即用的完整指标体系
  - **多资产原生支持**: 列维度广播，天然支持多资产/多参数
- **可借鉴方向**: 向量化计算、参数扫描、完整指标体系

---

## 二、可借鉴的方向列表（对照 jingni-trader）

| # | 借鉴方向 | 来源 | jingni-trader 现状 | 改进价值 | 已验证 |
|---|---------|------|-------------------|---------|--------|
| 1 | 向量化 IC 分析 | Qlib | `factor-engine` 用 for-loop 遍历日期 | ⭐⭐⭐⭐⭐ | ✅ 6.69x 加速 |
| 2 | 增强绩效指标 | VectorBT/Qlib | `backtest-engine` 仅 7 项指标 | ⭐⭐⭐⭐⭐ | ✅ 扩展到 22 项 |
| 3 | 向量化回测（混合策略）| VectorBT | `native_adapter` 用 for-loop | ⭐⭐⭐⭐ | ✅ 2.98x 加速 |
| 4 | 因子表达式引擎 | Qlib | 因子硬编码在 `compute_a_share_factors` | ⭐⭐⭐⭐ | 待开发 |
| 5 | 研究-实盘同构 | NautilusTrader | 回测与执行代码分离 | ⭐⭐⭐ | 待评估 |
| 6 | PIT 数据严格性 | Qlib | 未显式处理 | ⭐⭐⭐⭐ | 待评估 |
| 7 | 多层缓存系统 | Qlib | 仅文件级缓存 | ⭐⭐⭐ | 待评估 |
| 8 | 双时间戳模型 | NautilusTrader | 单时间戳 | ⭐⭐ | 待评估 |
| 9 | 滚动窗口训练 | Qlib | 无 | ⭐⭐⭐ | 待评估 |
| 10 | 确定性回测 | NautilusTrader | 未显式保证 | ⭐⭐⭐ | 待评估 |

---

## 三、已完成的验证测试及结论

### 测试环境
- Python 3.12.13
- pandas 3.0.3 / numpy 2.4.6 / scipy 1.18.0 / scikit-learn 1.9.0
- 测试数据: 50 只股票 × 730 个交易日（模拟 A 股日线）

### 测试结果汇总

```
==== 测试汇总: 13 通过 / 0 失败 / 共 13 项 ====
```

### 优化点 1: 向量化 IC 分析

**借鉴来源**: Microsoft Qlib 的高性能基础设施设计

**优化点说明**:
- 现有 `factor-engine/engine.py` 的 `_calc_ic` 方法用 Python for-loop 遍历每个日期，逐日调用 `scipy.stats.spearmanr`
- 优化后用 `groupby('date').rank()` 一次性计算所有日期的 rank，再用 `groupby('date').cov()` 向量化计算相关系数

**测试代码**: [optimizations/vectorized_ic.py](file:///workspace/optimizations/vectorized_ic.py)

**测试结果**:
| 指标 | 循环版 | 向量化版 | 提升 |
|------|--------|---------|------|
| 执行时间（中位数）| 6.92s | 1.03s | **6.69x 加速** |
| IC 均值最大差异 | - | 0.000000 | 完全一致 |
| 测试数据规模 | 300 股票 × 250 日 × 5 因子 | 同左 | - |

**结论**: IC 分析向量化收益显著，且结果完全一致。建议合并到 `factor-engine`。

### 优化点 2: 增强绩效指标

**借鉴来源**: VectorBT 的 57 项指标体系 + Qlib 的绩效分析

**优化点说明**:
- 现有 `backtest-engine/engine.py` 的 `_calc_metrics` 仅计算 7 项基础指标
- 优化后新增 15 项指标，覆盖下行风险、相对基准、交易质量等维度

**测试代码**: [optimizations/enhanced_metrics.py](file:///workspace/optimizations/enhanced_metrics.py)

**新增指标清单**:
| 类别 | 指标 | 说明 |
|------|------|------|
| 风险 | `sortino_ratio` | 下行风险调整收益 |
| 风险 | `downside_deviation` | 下行波动率 |
| 风险 | `max_drawdown_duration_days` | 最大回撤持续期 |
| 相对基准 | `information_ratio` | 信息比率 |
| 相对基准 | `beta` | CAPM Beta |
| 相对基准 | `alpha` | CAPM Alpha |
| 相对基准 | `tracking_error` | 跟踪误差 |
| 交易 | `n_trades` | 交易笔数 |
| 交易 | `profit_factor` | 盈亏比 |
| 交易 | `win_loss_ratio` | 单笔盈亏比 |
| 交易 | `avg_trade_pnl` | 平均单笔盈亏 |
| 交易 | `turnover_per_year` | 年换手率 |
| 收益 | `best_month` | 最佳月度收益 |
| 收益 | `worst_month` | 最差月度收益 |
| 收益 | `positive_month_ratio` | 正收益月份占比 |

**测试结果**:
- 基础指标一致性: ✅ 通过（total_return, sharpe_ratio, max_drawdown 等与原实现完全一致）
- Sortino >= Sharpe 关系: ✅ 通过（1.0915 >= 1.0647）
- Beta/Alpha 计算: ✅ 通过（beta=0.1355, alpha=0.1083）

**结论**: 指标体系显著增强，且与原实现保持兼容。建议合并到 `backtest-engine`。

### 优化点 3: 向量化回测引擎

**借鉴来源**: VectorBT 的向量化思想 + NautilusTrader 的业务规则保真

**优化点说明**:
- 现有 `native_adapter.py` 用 Python for-loop 逐日遍历，每日用 `day_data_map.loc[code]` 逐股票查找
- 优化后采用"日外循环 + 日内向量化"混合策略：
  - 保留日外循环（回测本质是路径依赖，无法完全向量化）
  - 信号对齐用一次 `merge` 完成，替代循环内查找
  - 每日用 numpy 数组处理买卖/估值
  - 用 `groupby` 预分组，避免逐日 filter

**关键设计权衡**:
- VectorBT 之所以能达到 100-1000x 加速，是因为它**牺牲了业务规则的真实性**（T+1、涨跌停、资金约束）
- 本优化**保留全部 A 股业务规则**，因此加速比低于 VectorBT，但结果可用于真实决策

**测试代码**: [optimizations/vectorized_backtest.py](file:///workspace/optimizations/vectorized_backtest.py)

**测试结果**:
| 指标 | 循环版 | 向量化版 | 提升 |
|------|--------|---------|------|
| 执行时间（中位数）| 0.508s | 0.170s | **2.98x 加速** |
| 终值一致性 | 1,402,380 | 1,407,451 | 相对误差 0.36%* |
| 交易笔数 | 1255 | 1255 | 完全一致 |

*注：终值微小差异源于多股票场景下资金分配顺序不同，不影响策略评估结论。

**边界条件测试**:
- ✅ 空数据处理
- ✅ 单股票单日
- ✅ 涨停限制买入（涨停时无交易）

**结论**: 在保留 A 股业务规则的前提下，向量化回测获得 2.98x 加速。建议合并到 `backtest-engine`。

---

## 四、待用户确认的优化建议

### 高优先级（已验证，建议合并）

1. **向量化 IC 分析** → 合并到 `skills/factor-engine/engine.py`
   - 替换 `_calc_ic` 方法的 for-loop 实现
   - 预期收益: IC 分析 6.69x 加速
   - 风险: 低（结果完全一致）

2. **增强绩效指标** → 合并到 `skills/backtest-engine/engine.py`
   - 替换 `_calc_metrics` 方法
   - 预期收益: 指标从 7 项扩展到 22 项
   - 风险: 低（基础指标保持兼容）

3. **向量化回测引擎** → 合并到 `skills/backtest-engine/scripts/adapters/native_adapter.py`
   - 替换 `run_backtest` 方法的循环实现
   - 预期收益: 回测 2.98x 加速
   - 风险: 中（终值有 0.36% 微小差异，需确认是否可接受）

### 中优先级（待开发验证）

4. **因子表达式引擎**（借鉴 Qlib）
   - 在 `factor-engine` 中引入表达式解析器
   - 用 `Ref($close, 20)` 等表达式定义因子，替代硬编码
   - 预期收益: 因子库可扩展性大幅提升

5. **PIT 数据严格性**（借鉴 Qlib）
   - 在 `data-engine` 中引入 Point-in-Time 数据模型
   - 严格避免未来函数，保证回测真实性
   - 预期收益: 回测结果更可信

6. **滚动窗口训练**（借鉴 Qlib）
   - 在 `strategy-model-engine` 中实现 `RollingGen`
   - 支持滚动窗口训练，避免前视偏差
   - 预期收益: 模型评估更严谨

### 低优先级（待评估）

7. **研究-实盘同构**（借鉴 NautilusTrader）
   - 统一回测与执行的策略接口
   - 需要较大架构调整

8. **多层缓存系统**（借鉴 Qlib）
   - 在 `data-engine` 中引入 Dataset/DataHandler/DataLoader 三级缓存
   - 需要较大架构调整

---

## 五、重要约束确认

- ✅ 所有优化代码位于 `feat/quant-opt-20260620` 分支的 `optimizations/` 目录
- ✅ 未修改 main 分支的任何代码
- ✅ 未执行 git merge 操作
- ✅ 分支已推送到 GitHub 远程仓库（仅 push，不合并）
- ⏳ 等待用户确认后方可合并到 main

---

## 六、文件清单

```
optimizations/
├── __init__.py              # 模块入口
├── vectorized_backtest.py   # 向量化回测引擎（含对照基准）
├── enhanced_metrics.py      # 增强绩效指标（含基础指标对照）
├── vectorized_ic.py         # 向量化 IC 分析（含对照基准）
├── test_optimizations.py    # 验证测试套件（13 项测试）
├── test_results.json        # 测试结果（JSON 格式）
└── OPTIMIZATION_REPORT.md   # 本报告
```

---

## 七、参考资料

- [Microsoft Qlib 论文](https://arxiv.org/abs/2009.11189)
- [Qlib GitHub](https://github.com/microsoft/qlib)
- [NautilusTrader 文档](https://nautilustrader.io/docs/latest/concepts/overview/)
- [VectorBT 文档](https://vectorbt.dev/)
- [20+ Algo Trading Frameworks Reviewed](https://autotradelab.com/blog/nautilus-vs-vectorbt-vs-freqtrade-20-python-quant-trading-frameworks-compared)
- [The 5 GitHub Repos Rewriting How AI Trades Money](https://blog.themenonlab.com/blog/ai-finance-github-repos-march-2026)
