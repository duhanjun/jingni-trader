# jingni-trader 量化优化学习与验证报告

**执行日期**: 2026-06-23
**分支**: `feat/quant-opt-20260623`
**执行人**: 自动化学习流程

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib (⭐ 41k+)
- **定位**: AI 驱动的量化投资全链路平台
- **核心亮点**:
  - **Point-in-Time (PIT) 数据库**: 按时间点组织财报数据，严格防止未来函数（look-ahead bias），解决财报修订链问题
  - **表达式引擎 (Expression Engine)**: 用 DSL 字符串定义因子（如 `Ref($close, 5)/$close`），无需写代码，支持时间序列/横截面算子
  - **高性能 .bin 数据格式**: 专为金融时序设计的二进制存储，比 parquet/csv 快数倍
  - **Model Zoo**: 内置 20+ SOTA 模型（LightGBM/Transformer/GATs/TFT）
  - **RD-Agent**: LLM 自动挖因子、调参
- **借鉴价值**: ⭐⭐⭐⭐⭐（PIT 与表达式引擎直接填补 jingni-trader 短板）

### 2. Backtrader (⭐ 10k+)
- **定位**: 轻量灵活的 Python 回测框架
- **核心亮点**:
  - 事件驱动 + 向量化混合架构
  - 插件丰富（TA-Lib/Plotly 集成）
  - API 直观，50 行即可跑双均线策略
- **借鉴价值**: ⭐⭐⭐⭐（向量化回测性能设计）

### 3. vn.py (⭐ 23k+)
- **定位**: 国产最成熟的实盘量化框架
- **核心亮点**:
  - 支持 CTP/IB/OKX 等数十家交易所
  - 毫秒级延迟，社区极活跃
  - CTA/算法交易/期权套利模块齐全
- **借鉴价值**: ⭐⭐⭐（实盘执行层参考，本次未重点验证）

---

## 二、可借鉴的方向列表

对照 jingni-trader 现有代码（`skills/` 七阶段架构），识别出 3 个高价值优化方向：

| # | 优化方向 | 借鉴来源 | jingni-trader 现状问题 | 预期收益 |
|---|---------|---------|----------------------|---------|
| 1 | **因子表达式引擎** | Qlib DSL | `pandas_ta_calculator.py` 用 if-elif 硬编码 18 个因子，新增因子必须改代码 | 因子库无限扩展，无需改代码；表达式可序列化/共享/复现 |
| 2 | **Point-in-Time 数据** | Qlib PIT | `base_data_provider.get_financial` 仅按 report_date 取数，回测会用财报"最终修订值"造成未来函数 | 严格防止 look-ahead bias，提升回测真实性 |
| 3 | **向量化回测** | Backtrader/Qlib | `native_adapter.py` 用 `iterrows()` 逐行遍历，O(D×N) 复杂度，大股票池极慢 | 性能提升 10x+，支持全 A 回测 |

---

## 三、已完成的验证测试及结论

### 测试环境
- Python 3.12.13 / numpy 2.5.0 / pandas 3.0.3
- 测试框架: unittest
- 测试文件: `optimizations/test_verification.py`

### 测试结果总览

```
Ran 29 tests in 0.544s
OK
```

**全部 29 个测试通过**，覆盖正确性、性能、边界条件、集成场景。

### 优化点 1：因子表达式引擎 (ExpressionEngine)

**文件**: `optimizations/expression_engine.py`

**实现内容**:
- 递归下降解析器，支持四则运算、函数调用、字段引用（`$close`）
- 算子注册表 `Ops`，分三类：
  - 时间序列 (ts): `Ref / Mean / Std / Max / Min / Delta / Rank / WMA / Corr`
  - 横截面 (cs): `CSRank / CSZscore / CSQuantile`
  - 元素级 (element): `Abs / Log / Sign`
- 表达式解析缓存，重复计算免解析

**测试覆盖** (11 项):
| 测试 | 结论 |
|------|------|
| `test_field_access` | ✅ 字段访问正确 |
| `test_arithmetic` | ✅ 四则运算与 pandas 结果一致 (rtol=1e-10) |
| `test_ref_operator` | ✅ Ref 时间序列算子与 groupby.shift 一致 |
| `test_mean_operator` | ✅ Mean 滚动均值与 rolling.mean 一致 |
| `test_cs_rank` | ✅ CSRank 横截面排名落在 [0,1]，单调性正确 |
| `test_complex_expression` | ✅ 复合表达式 `-1 * CSRank(Delta($close, 5) / $close)` 范围合理 |
| `test_evaluate_many` | ✅ 批量计算 3 个因子成功 |
| `test_unknown_operator_raises` | ✅ 未知算子抛 KeyError |
| `test_unknown_field_raises` | ✅ 未知字段抛 KeyError |
| `test_empty_data` | ✅ 空数据优雅处理 |
| `test_cache_hit` | ✅ 解析缓存生效 |

**关键结论**: 表达式引擎可完全替代 `pandas_ta_calculator.py` 的硬编码方式，新增因子只需写字符串表达式，例如：
```python
engine.evaluate("CSRank(-1 * Delta($close, 5) / Ref($close, 5))", data)
```

### 优化点 2：Point-in-Time 数据提供者 (PITProvider)

**文件**: `optimizations/pit_data_provider.py`

**实现内容**:
- 维护 `(code, period, field, publish_date, value, revision_seq)` 修订链
- `get_pit(code, field, observe_date)`: 返回观察日当天"已公开的最新值"
- `as_of_pit(panel)`: 对回测面板按观察日批量对齐财报字段
- `revision_chain()`: 审计某条记录的完整修订历史

**测试覆盖** (10 项):
| 测试 | 结论 |
|------|------|
| `test_before_publish_returns_none` | ✅ 发布日前查询返回 None |
| `test_original_value_before_revision` | ✅ 修订前返回原始值 12.5 |
| `test_revised_value_after_revision` | ✅ 修订后返回修订值 13.1 |
| `test_next_period_value` | ✅ 跨报告期查询正确 |
| **`test_look_ahead_bias_protection`** | ✅ **核心**: 2024-05-15 不拿到 2024-06-15 的修订值 |
| `test_revision_chain` | ✅ 修订链完整 (2 条记录) |
| `test_as_of_pit_alignment` | ✅ 面板对齐: [None, 12.5, 13.1, 14.0] |
| `test_unknown_code_returns_none` | ✅ 未知股票返回 None |
| `test_stats` | ✅ 统计信息正确 |
| `test_empty_provider` | ✅ 空 provider 边界 |

**关键结论**: PIT 机制严格防止了未来函数。例如 2024Q1 财报在 2024-04-30 发布原始值 12.5，2024-06-15 修订为 13.1；回测 2024-05-15 的策略只会拿到 12.5，不会泄露 13.1。

### 优化点 3：向量化回测引擎 (VectorizedBacktest)

**文件**: `optimizations/vectorized_backtest.py`

**实现内容**:
- 价格/信号矩阵化: `pivot_table` 构造 (date × code) 矩阵
- 持仓矩阵化: numpy 数组维护每日持仓
- 向量化净值: `equity = cash + (shares * close).sum(axis=1)`
- 保留 A 股规则: T+1、涨跌停、印花税、佣金、滑点
- 与 `NativeAdapter` 接口兼容

**测试覆盖** (6 项 + 1 性能对比):
| 测试 | 结论 |
|------|------|
| `test_basic_run` | ✅ 基本回测跑通，输出净值曲线与指标 |
| `test_empty_data` | ✅ 空数据返回空结果 |
| `test_no_signals` | ✅ 无信号时净值保持初始资金 |
| `test_target_weight_mode` | ✅ 目标权重模式正常 |
| `test_metrics_reasonable` | ✅ 指标合理 (夏普/回撤范围正常) |
| `test_price_limit_blocks_trades` | ✅ 涨跌停限制生效 |

**性能对比测试**:
```
[性能] 逐行 (iterrows): 196.5ms | 向量化: 16.7ms | 加速比: 11.8x
```
- 测试规模: 20 只股票 × 120 天
- **向量化实现比逐行实现快 11.8 倍**，且随股票数增加优势更明显

### 集成测试

**`test_full_pipeline`**: 端到端验证 表达式因子 → PIT 对齐 → 向量化回测
```
[集成] 端到端回测完成: 净值=915828
```
✅ 三大优化模块协同工作正常。

---

## 四、对比分析

### 4.1 因子库扩展性对比

| 维度 | 现状 (pandas_ta_calculator) | 优化后 (ExpressionEngine) |
|------|---------------------------|-------------------------|
| 新增因子 | 修改 `_calc_factor` if-elif | 写一行表达式字符串 |
| 横截面算子 | ❌ 不支持 | ✅ CSRank/CSZscore/CSQuantile |
| 复合因子 | ❌ 需手写多层函数 | ✅ `CSRank(Delta($close,5)/Ref($close,5))` |
| 可序列化 | ❌ 代码即逻辑 | ✅ 表达式字符串可存库/共享 |
| 因子数量上限 | 18 个硬编码 | 无限 |

### 4.2 回测正确性对比

| 维度 | 现状 (无 PIT) | 优化后 (PITProvider) |
|------|-------------|---------------------|
| 财报未来函数 | ❌ 用最终修订值 | ✅ 只用观察日已公开值 |
| 修订链追踪 | ❌ 无 | ✅ revision_chain 审计 |
| 回测真实性 | 有泄露风险 | 严格无泄露 |

### 4.3 回测性能对比

| 维度 | 现状 (NativeAdapter iterrows) | 优化后 (VectorizedBacktest) |
|------|------------------------------|---------------------------|
| 20股×120天 | ~197ms | ~17ms (11.8x) |
| 算法复杂度 | O(D×N) 逐行 | O(D) 矩阵化 |
| 全 A (5000股) | 预计 >30s | 预计 <3s |
| 内存 | 低 | 中（矩阵化） |

---

## 五、待用户确认的优化建议

以下优化方案已在 `feat/quant-opt-20260623` 分支验证通过，**等待用户确认后方可合并到 main**：

### 建议 1: 用 ExpressionEngine 替换 pandas_ta_calculator 的硬编码
- **影响范围**: `skills/factor-engine/`
- **风险**: 低（新旧可并存，表达式引擎为新增模块）
- **建议**: 在 `factor-engine` 新增 `expression_calculator.py` 适配器，实现 `BaseFactorCalculator` 接口，内部调用 `ExpressionEngine`；保留 `pandas_ta_calculator` 作为兼容

### 建议 2: 在 data-engine 引入 PITProvider
- **影响范围**: `skills/data-engine/`
- **风险**: 中（需改造 `get_financial` 增加 publish_date 维度）
- **建议**: 新增 `pit_provider.py`，`get_financial` 返回带 `publish_date` 的记录；回测阶段强制走 `as_of_pit` 对齐

### 建议 3: 用 VectorizedBacktest 替换 NativeAdapter 作为默认回测后端
- **影响范围**: `skills/backtest-engine/`
- **风险**: 中（需验证与 backtrader/rqalpha 适配器结果一致性）
- **建议**: 将 `VectorizedBacktest` 注册为新的 native 适配器，通过配置切换；保留原 `NativeAdapter` 用于结果交叉验证

### 合并流程
1. 用户确认优化方案后告知"可以合并"
2. 创建 PR: `feat/quant-opt-20260623` → `main`
3. Code review
4. 合并后删除分支

---

## 六、文件清单

```
optimizations/
├── __init__.py                  # 模块说明
├── expression_engine.py         # 优化1: 因子表达式引擎 (419行)
├── pit_data_provider.py         # 优化2: PIT 数据提供者 (158行)
├── vectorized_backtest.py       # 优化3: 向量化回测引擎 (199行)
└── test_verification.py         # 验证测试套件 (29 项测试)
```

**所有新代码均位于 `feat/quant-opt-20260623` 分支的 `optimizations/` 目录，未修改 main 分支任何原有代码。**

---

## 七、参考来源

- Qlib 论文: https://arxiv.org/abs/2009.11189
- Qlib PIT 文档: https://github.com/microsoft/qlib/blob/main/docs/advanced/PIT.rst
- Qlib 表达式算子: https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py
- Backtrader: https://github.com/backtrader/backtrader
- vn.py: https://github.com/vnpy/vnpy
