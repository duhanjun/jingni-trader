# jingni-trader 量化交易学习报告

## 日期: 2026-06-11 | 序号: #1

---

## 一、学习项目清单

本次研究聚焦三个高影响力量化交易开源项目，从因子引擎、数据处理性能、ML管道三个维度为 jingni-trader 寻找优化方向。

### 1.1 Qlib (Microsoft)

| 项目 | 详情 |
|------|------|
| GitHub | https://github.com/microsoft/qlib |
| Stars | ~15,000+ |
| 语言 | Python |
| 核心亮点 | Expression Engine, Alpha因子库, 配置驱动工作流 |

**关键设计:**

1. **Expression Engine（表达式引擎）** — 支持声明式因子定义，如 `Ref($close, 1)`、`Mean($close, 20)`、`Corr($close, $volume, 10)`。用户通过 DSL 定义因子，引擎自动解析依赖关系和计算。这极大降低了编写新因子的门槛。

2. **Alpha158/Alpha360** — 预定义的标准化因子库，覆盖158或360个常见Alpha因子，所有因子经过对齐、标准化、中性化处理。

3. **Point-in-Time 数据管理** — 确保回测无未来信息泄露。使用 `handler` 串行处理数据加载、标准化、Alpha处理等流水线。

4. **qrun + YAML 配置驱动** — 整个研究流程（数据→特征→模型→回测）通过 YAML 文件一键配置执行。

### 1.2 QUANTAXIS

| 项目 | 详情 |
|------|------|
| GitHub | https://github.com/yutiansut/QUANTAXIS |
| Stars | ~8,000+ |
| 语言 | Python + Rust |
| 核心亮点 | Rust核心性能优化, QIFI统一协议, 零拷贝数据桥 |

**关键设计:**

1. **Rust核心 (QAPRO-RS)** — 计算密集型任务（数据处理、因子计算、回测迭代）使用Rust重写，获得10~100x性能提升。

2. **QADataBridge（零拷贝数据桥）** — 基于Apache Arrow的共享内存数据交换，避免了Python与Rust间的频繁数据拷贝。

3. **QIFI协议** — 统一的账户/持仓/订单/成交协议，屏蔽不同券商API差异。这对jingni-trader的多券商对接有直接参考价值。

4. **Polars集成** — 从Pandas逐步迁移到Polars，利用其惰性执行（LazyFrame）和查询优化的特性。

### 1.3 Freqtrade (+ FreqAI)

| 项目 | 详情 |
|------|------|
| GitHub | https://github.com/freqtrade/freqtrade |
| Stars | ~35,000+ |
| 语言 | Python |
| 核心亮点 | FreqAI自适应ML管道, Optuna超参优化(NSGA-III), 实盘就绪 |

**关键设计:**

1. **FreqAI自适应重训练** — 支持多种重训练策略：
   - 时间窗口训练（每N天）
   - 性能退化触发训练（回测指标低于阈值）
   - 每N次完整循环训练
   - 模型过期自动替换

2. **预测置信度系统** — 基于最近N次预测误差（Dissimilarity Index / RMSE）计算置信度，低置信度时自动降低仓位或切换备用策略。

3. **HyperOpt NSGA-III** — 多目标优化（Pareto最优前沿），同时优化收益、最大回撤、胜率等多个目标，而非简单的加权融合。

4. **市场状态检测** — 内置市场regime检测，不同regime使用不同模型参数。

---

## 二、可借鉴方向列表

| 序号 | 优化方向 | 借鉴来源 | 当前状态 | 优先级 | 对应模块 |
|------|---------|---------|---------|--------|---------|
| 1 | 声明式因子表达式引擎 | Qlib Expression Engine | 已编写验证代码并测试通过 | **高** | factor-engine |
| 2 | Polars替代Pandas核心操作 | QUANTAXIS | 已编写性能对比验证代码 | **高** | data-engine, factor-engine |
| 3 | 自适应ML重训练管道 | FreqAI | 已编写验证代码并测试通过 | **高** | strategy-model-engine |
| 4 | 预测置信度评估 | FreqAI | 已编写验证代码并测试通过 | 中 | strategy-model-engine |
| 5 | 配置驱动工作流 (YAML) | Qlib qrun | 待验证 | 中 | engine (主引擎) |
| 6 | NSGA-III多目标优化 | Freqtrade HyperOpt | 已编写验证代码 | 低 | portfolio-risk-engine |
| 7 | 统一券商协议接口 | QIFI (QUANTAXIS) | 待验证 | 低 | execution-monitor-engine |
| 8 | Point-in-Time数据管理 | Qlib | 待验证 | 低 | data-engine |

---

## 三、已完成的验证测试

### 3.1 因子表达式引擎 — test_factor_expression_engine.py

**测试结果: 全部通过**

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 基本计算正确性 | PASS | 与硬编码方式计算结果完全一致 |
| 表达式可组合性 | PASS | 支持嵌套复合因子 (如MomVolAdj = 动量/波动率) |
| 扩展性 | PASS | 自定义函数注册 (如ZScore) 正常工作 |
| 边界条件 | PASS | 空数据、单股票、NaN、零成交量正确处理 |

**对比数据:**
- 声明式引擎: 0.042s
- 硬编码方式: 0.023s
- 性能差异约 1.8x，但声明式提供了更高的灵活性和可维护性

### 3.2 Polars性能对比 — test_polars_performance.py

**测试结果: Polars 在核心操作上有显著优势（特定场景除外）**

| 基准测试 | Pandas | Polars | 加速比 |
|----------|--------|--------|--------|
| 分组滚动计算 | 0.766s | 0.079s | **9.63x** |
| 截面排名排序 | 0.095s | 0.021s | **4.60x** |
| 透视+协方差 | 0.037s | 3.704s | 0.01x (注) |
| Parquet写 | 0.130s | 0.024s | **5.38x** |
| Parquet读 | 0.024s | 0.013s | 1.91x |

**注:** 透视+协方差场景下Polars的逐元素cov计算循环远慢于Pandas的matrix cov操作。这说明不是所有操作Polars都有优势，需要针对性优化。

**正确性:** Pandas vs Polars 因子计算结果一致性PASS，最大浮点误差 < 0.00004。

### 3.3 自适应ML管道 — test_adaptive_ml_pipeline.py

**测试结果: 全部通过**

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 滑动窗口管理 | PASS | 自动分割训练/验证窗口，purge日期间隔 |
| 自适应重训练 | PASS | 2个训练窗口各生成模型，性能稳定 |
| 采样器对比 | PASS | TPE和NSGA-III性能接近，均正常收敛 |
| 预测置信度 | PASS | 可区分好/中/差模型质量 |
| 模型过期检测 | PASS | 按配置周期正确判断过期 |

**置信度测试数据:**
- 好模型: 0.838
- 中等模型: 0.501
- 差模型: 0.230
- 模型漂移: 0.000 (误差持续增大)

---

## 四、优化建议（待用户确认）

### 建议1: 引入因子表达式引擎层 (factor-engine) — 推荐优先采纳

在当前`skills/factor-engine/engine.py`中，因子计算为硬编码`groupby().pct_change()`等操作。建议新增一层表达式引擎，允许：

```
# 当前方式 (硬编码)
df['ret_20d'] = df.groupby('code')['close'].pct_change(20)

# 优化后 (声明式)
engine.register_expression("ret_20d", "PctChange($close, 20)")
```

收益:
- 因子定义与计算逻辑解耦
- 支持从YAML/JSON配置文件热加载因子定义  
- 新增因子无需修改引擎代码，仅注册即可
- 自动缓存和依赖解析

### 建议2: data-engine引入Polars可选后端 — 推荐渐进采纳

在`config.py`中添加 `DATA_BACKEND` 配置项，允许切换 pandas/polars。对于分组滚动计算（因子引擎核心操作），Polars提供**9.63x加速**。建议先从 data-engine 的ETL模块入手，逐步扩展到factor-engine。

### 建议3: strategy-model-engine增加自适应重训练 — 推荐纳入规划

当前模型训练为一次性，完成后不再更新。建议引入：
- 定期重训练（每30/60/90天）
- 性能监控（IC值持续下降触发重训练）
- 模型版本管理（保留最近N个模型）
- 预测置信度输出（`predict_with_confidence()`）

### 建议4: 配置驱动工作流 — 可列入中长期规划

借鉴Qlib的YAML配置方式，将整个量化研究流程从"手动调用"改为"配置声明"。降低使用门槛，提升可复现性。

---

## 五、测试文件清单

| 文件 | 说明 | 借鉴来源 |
|------|------|---------|
| `tests/study_2026/test_factor_expression_engine.py` | 因子表达式引擎验证 | Microsoft/qlib |
| `tests/study_2026/test_polars_performance.py` | Polars vs Pandas性能对比 | QUANTAXIS |
| `tests/study_2026/test_adaptive_ml_pipeline.py` | 自适应ML管道+置信度验证 | freqtrade/freqtrade |

所有测试均可通过 `python3 tests/study_2026/test_*.py` 独立运行。

---

## 六、合规声明

- 未执行任何 `git commit`、`git push`、`git merge` 操作
- 所有验证代码仅存在于 `tests/study_2026/` 目录中
- 未对主项目代码 (`skills/`、`scripts/`、`engine.py`) 进行任何修改
- 待用户确认优化方案后，方可在 feature 分支中实施修改