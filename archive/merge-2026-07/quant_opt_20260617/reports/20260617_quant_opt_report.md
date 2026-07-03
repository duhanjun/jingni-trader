# 量化交易优化验证报告 (feat/quant-opt-20260617)

> 日期: 2026-06-17
> 分支: `feat/quant-opt-20260617`（基于 `main`，未合并）
> 范围: 借鉴 [polakowo/vectorbt](https://github.com/polakowo/vectorbt) 和 [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader)
> 对应 jingni-trader 子模块: `backtest-engine` / `execution-monitor-engine`

---

## 1. 联网调研与项目清单

通过 WebSearch 在 GitHub、arXiv Papers with Code、QuantConnect 社区、BigQuant 等渠道调研，整理出近期高 Star / 活跃的量化交易开源项目。完整列表与亮点见下表：

| # | 项目 | Star | 核心亮点 | 借鉴价值 |
|---|------|------|----------|---------|
| 1 | [microsoft/qlib](https://github.com/microsoft/qlib) | 28k+ | AI quant 投资平台；Point-in-Time 数据 + 多范式 ML；RD-Agent 自动因子挖掘 | 数据完整性、因子库治理 |
| 2 | [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | 6.5k+ | 全数据向量化回测（NumPy/Numba/Rust）；100-1000x 加速 | 回测引擎性能 |
| 3 | [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | 9k+ | 生产级事件驱动；RiskEngine + TradingState 状态机 + Throttler | 风控体系 |
| 4 | [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | 43k+ | 多 Agent LLM 投研框架 | 决策可解释性 |
| 5 | [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) | 35k+ | FreqAI 机器学习策略 | ML 集成 |
| 6 | [mementum/backtrader](https://github.com/mementum/backtrader) | 17k+ | 老牌事件驱动框架 | 生态成熟 |
| 7 | [quantopian/zipline](https://github.com/quantopian/zipline) | 18k+ | PyData 系，回测标杆 | 数据管线 |
| 8 | [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | 11k+ | 深度强化学习做交易 | 强化学习 |

### 重点借鉴项目（深入阅读）

#### A. polakowo/vectorbt
**核心架构**：
- 不用 Python 事件循环，而是把整段行情 × 投资组合构建为 NumPy 矩阵 `(T, N)`；
- 信号、目标权重、实际持仓、现金、权益全部向量化；
- 支持 Numba JIT / Rust 后端进一步加速；
- 大量使用 `np.cumsum / np.where / 广播` 而非 `.iterrows()`。

**关键代码片段**（在 `vectorbt/generic/nb.py` 中）：
```python
@njit
def cs_rank_nb(x):  # 跨标的横截面排序
    ...
@njit
def signal_to_positions_nb(signals, init_cash, ...):
    cash = init_cash
    for col in range(signals.shape[1]):  # 仍是外层 for 列
        ...
```
外层仍需按列循环，但内层完全向量化。

#### B. nautechsystems/nautilus_trader
**核心架构**：
- `RiskEngine` 是预交易校验链，按固定顺序检查 free balance / notional / rate；
- 全局 `TradingState { ACTIVE, HALTED, REDUCING }`；
- `Throttler` 用 token bucket 控制 submit / modify / cancel 速率；
- 所有 `OrderDenied` 事件含 `reason` 枚举，便于上层订阅处理。

**关键代码片段**（在 `nautilus_trader/risk/engine.py`）：
```python
self._check_orders_risk(orders)  # 预交易校验链
if self.trading_state == TradingState.HALTED:
    self._deny(orders, "Trading is halted")
```

---

## 2. jingni-trader 现有代码审计要点

通过阅读 `engine.py` / `skills/*/engine.py` / 各子引擎 `base/*.py`，识别出以下可优化点：

| 模块 | 现状 | 问题 |
|------|------|------|
| `backtest-engine/scripts/adapters/native_adapter.py` | 双层 `for dt in dates` + `for code, row in day_data.iterrows()` | Python 循环 + `iterrows()`，20 标的 × 500 日 ≈ 1.5s，规模化时性能差 |
| `execution-monitor-engine/engine.py::CircuitBreaker` | 单点检查：日亏 / 单笔金额 / 频率 | 缺少全局交易状态机 (HALTED / REDUCING)、单标的最大持仓、reduce_only 校验、行业偏离、审计日志 |
| `data-engine` | 长格式 DataFrame 接口 | 缺 PIT（Point-in-Time）数据可用性检查，留有未来数据泄漏风险 |
| `factor-engine/engine.py` | 因子合成 OK，但缺少 alpha 表达式 AST 化能力 | 拓展性受限 |
| 整体 | 缺乏统一的 `RiskEngine.check_order(...)` 返回结构 | 上层调用方需要自己解析 reason 字符串 |

> 注：本期工作聚焦"向量化回测 + 多层风控"两条最高 ROI 路径；PIT 与因子 AST 化留作后续轮次。

---

## 3. 本次优化方向与验证代码

### 优化 1：向量化回测引擎 (借鉴 vectorbt)

**目标**：避免双层 Python 循环，将回测推进改写为 NumPy 数组运算。

**文件**：[vectorized_engine.py](file:///workspace/quant_opt_20260617/vectorized_backtest/vectorized_engine.py)

**关键设计**：
- 把行情 pivot 成 `(T, N)` 二维矩阵；
- `holdings` 是 `(T+1, N)` 整型数组，`np.cumsum` / `np.where` 推进；
- T+1 通过 `signals[t-1]` 移到下一天处理；
- 涨跌停 mask 在数据准备阶段就构造好；
- 输出 schema 与 `native_adapter` 一致：`trades / positions / equity_curve / metrics`。

**保留 A 股特性**：佣金率 / 最低佣金 / 印花税 / 滑点 / 100 股一手取整 / 涨跌停过滤 / 买入预算等比分配。

### 优化 2：多层风控引擎 (借鉴 NautilusTrader)

**目标**：把"全局状态机 + 预交易校验链 + 限速 + 审计"组合成一个 `MultiLayerRiskEngine`。

**文件**：[multi_layer_risk.py](file:///workspace/quant_opt_20260617/risk_engine/multi_layer_risk.py)

**关键设计**：
- `TradingState` 三态枚举：`ACTIVE` / `REDUCING` / `HALTED`，通过 `halt() / reduce_only() / resume()` 切换；
- `check_order()` 校验链顺序：state → cash/reduce_only → 单笔金额 → 单票最大持仓 → 标的 notional 上限 → 日亏 → 行业偏离 → 频率节流；
- `TokenBucketThrottler`：可配置的每秒下单 / 撤单速率；
- `AccountSnapshot` 把"账户 + 行业映射 + 基准权重 + 价格快照"打包传入；
- `OrderDecision` 统一返回结构，含 `allowed / reason / reason_detail / trading_state`；
- 审计日志 JSONL 记录所有 ALLOW / DENY / HALT / RESUME 事件。

**对现有 CircuitBreaker 的兼容**：保留了日亏 / 单笔金额 / 频率三个核心行为，扩展为完整校验链（见 test_16_alignment_with_existing_circuit_breaker）。

---

## 4. 验证测试结果

### 4.1 向量化引擎（10 个测试用例）

| # | 测试用例 | 结论 |
|---|----------|------|
| 01 | 基础回测能跑出指标且指标合理 | OK |
| 02 | 权益恒等式（现金 + 持仓市值 = 权益） | OK，最大误差 < 1e-6 |
| 03 | 成交记录一致性（买入 pnl < 0，卖出 amount > 0） | OK |
| 04 | T+1 模式：信号当日不成交 | OK |
| 05 | 非 T+1 模式：信号次日有持仓 | OK |
| 06 | 性能对比 vs 原生 native_adapter | **18.97x 加速** |
| 07 | 空数据 | OK |
| 08 | 单只股票 | OK |
| 09 | 涨停过滤 | OK |
| 10 | 手数取整（100 的倍数） | OK |

**性能数据**（[perf_vectorized_vs_native.json](file:///workspace/quant_opt_20260617/reports/perf_vectorized_vs_native.json)）：

```
数据集: 20 标的 × 500 日
向量化引擎耗时: 0.077s
原生 native_adapter 耗时: 1.463s
加速比: 18.97x
```

> 注：在更大规模（如 1000 标的 × 2500 日）下，vectorbt 官方公布的加速比可达 100-1000x，本验证的 18.97x 来自 20×500 的中小数据集，且为首次执行（含 Python 启动开销）。

### 4.2 多层风控引擎（19 个测试用例）

| # | 测试用例 | 结论 |
|---|----------|------|
| 01-04 | 状态机：ACTIVE / HALTED / REDUCING / RESUME | OK |
| 05 | 单笔金额上限 | OK |
| 06 | 单票最大持仓 | OK |
| 07 | 日亏止损 | OK |
| 08 | reduce_only 超卖保护 | OK |
| 09 | 资金不足 | OK |
| 10 | 频率节流（Token Bucket） | OK，5+ 单/秒拒 |
| 11 | 行业偏离触发 | OK |
| 12 | 行业偏离可控 | OK |
| 13 | 缺行业数据时跳过 | OK |
| 14 | 审计日志写入 | OK |
| 15 | 统计接口 | OK |
| 16 | 与现有 CircuitBreaker 行为对齐 | OK（同一笔订单两边都拒） |
| 17-19 | Token Bucket 单元测试 | OK |

**审计日志样本**（[risk_audit.jsonl](file:///workspace/quant_opt_20260617/reports/risk_audit.jsonl)）：
```json
{"timestamp":"2026-06-17T20:13:38.499299","kind":"DENY","reason":"DAILY_LOSS","detail":"日亏 -5.00% 超过阈值 -3.00%","trading_state":"ACTIVE"}
{"timestamp":"2026-06-17T20:13:38.499438","kind":"DENY","reason":"SINGLE_ORDER_SIZE","detail":"单笔金额 78000 超过上限 20000","trading_state":"ACTIVE"}
```

### 4.3 边界 / 健壮性

- **空数据**：两套引擎均能优雅返回空 result，不抛异常；
- **单标的**：向量化引擎可正常运行（虽然失去向量化优势，但保证正确性）；
- **涨跌停 mask**：手造涨停日不成交，权益曲线长度等于交易日数；
- **跨日期停牌**：通过 close 为 NaN 的 mask 隐式处理；
- **手数取整**：所有买入股数均为 100 的整数倍。

---

## 5. 待用户确认的优化建议

下列建议希望与用户确认后再决定是否合并到 main：

### 5.1 短期（建议采纳）
1. **将 `VectorizedBacktestEngine` 作为 `native_adapter` 的可选 backend**
   - 在 `backtest-engine/engine.py` 中通过 `BACKTEST_BACKEND=vectorized` 环境变量切换
   - 保留 `native` 作为 fallback，避免向量化引擎在复杂订单类型下的兼容性风险
   - 评估：风险可控，收益显著

2. **将 `MultiLayerRiskEngine` 接入 `PaperExecutor`**
   - 用新引擎替换 `CircuitBreaker`，增加：状态机 / reduce_only / 持仓 / 行业偏离 / 审计
   - 评估：API 表面略有变化（`check_order(code, side, volume, price, account)` vs `check_send_order(account, code, order_value)`），需做包装层

### 5.2 中期（需进一步验证）
3. **引入 PIT（Point-in-Time）数据可用性字段**
   - 在 `data-engine` 中为每条数据加 `available_at` 字段，因子合成时强制按"信号生成时刻是否已发布"过滤
   - 借鉴 Qlib 的 PIT 设计，可避免使用财务因子时的未来数据泄漏
   - 评估：需要重写 data-engine 部分接口

4. **风险规则可视化配置**
   - 把 `MultiLayerRiskEngine` 的参数外置到 `risk_rules.yaml`
   - 评估：低风险

### 5.3 长期（路线图）
5. **多进程 / Numba 加速**
   - 给向量化引擎加上 `@njit` 装饰，把内层循环编译到机器码
   - 借鉴 vectorbt 的 nb.py 设计

6. **AI 因子自动挖掘**
   - 引入 Qlib 的 RD-Agent 思路，用 LLM 在已有 alpha 库基础上做"算子组合 + 回测评估"的闭环
   - 风险：需要算力、依赖外部 LLM API

---

## 6. 操作记录

- **分支创建**：`git checkout -b feat/quant-opt-20260617`（基于 main）
- **新增文件**：
  - `quant_opt_20260617/vectorized_backtest/vectorized_engine.py`
  - `quant_opt_20260617/risk_engine/multi_layer_risk.py`
  - `quant_opt_20260617/tests/test_vectorized_engine.py`
  - `quant_opt_20260617/tests/test_risk_engine.py`
  - `quant_opt_20260617/README.md`
  - `quant_opt_20260617/reports/20260617_quant_opt_report.md`（本文件）
- **测试通过数**：向量化 10 / 10，风控 19 / 19
- **不修改 main**：所有改动均在 `quant_opt_20260617/` 独立目录下
- **分支推送**：本次任务结束后会推送到 GitHub 远程 `feat/quant-opt-20260617`，**不合并**

---

## 7. 结论

- 通过对 vectorbt 与 NautilusTrader 的借鉴与验证，证明向量化回测和多层风控在 jingni-trader 中的可行性与显著收益；
- 向量化引擎在 20×500 数据集上实现 **18.97x 加速**，指标语义与原生一致；
- 多层风控引擎在保留现有 CircuitBreaker 核心行为的同时，新增了 6 项检查 + 状态机 + 审计日志；
- **请用户审阅本报告**，决定是否在下一轮合并到 main。

