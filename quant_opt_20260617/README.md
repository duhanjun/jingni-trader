# 量化交易优化模块 - 2026-06-17

> **本目录为 jingni-trader 项目优化实验模块，所有代码在 `feat/quant-opt-20260617` 分支**
> **未经用户确认前，不会合并到 main 分支**

## 目录结构

```
quant_opt_20260617/
├── README.md                          # 本文件
├── __init__.py                        # 包初始化
├── factor_expression_engine.py        # 优化 1: 因子表达式引擎
├── walk_forward.py                    # 优化 2: Walk-Forward 滚动训练
├── dynamic_factor_fusion.py           # 优化 3: 动态因子权重融合
├── ic_analysis.py                     # 优化 4: 增强 IC 分析
├── verify_all.py                      # 统一验证脚本
├── tests/
│   ├── __init__.py
│   ├── _synthetic_data.py             # 测试数据生成器
│   ├── test_factor_expression.py      # 14 个测试
│   ├── test_walk_forward.py           # 9 个测试
│   ├── test_dynamic_fusion.py         # 10 个测试
│   └── test_ic_analysis.py            # 8 个测试
└── results/                           # 验证结果
    ├── verification_report_*.md
    └── verification_report_*.json
```

## 借鉴的开源项目

| 项目 | Star | 借鉴点 | 对应模块 |
|------|------|--------|----------|
| [microsoft/qlib](https://github.com/microsoft/qlib) | 15k+ | Alpha158 因子库、Pipeline 编排、Recorder | 因子表达式引擎 + IC 分析 |
| [akquant](https://github.com/akfamily/akquant) | 1.3k | Polars 因子引擎、Walk-Forward、Trait 接口 | 表达式引擎 + Walk-Forward |
| [AlphaForge](https://arxiv.org/abs/2406.18394) | 论文 | 动态因子权重、生成-预测网络 | 动态因子融合 |
| [vectorbt](https://github.com/polakowo/vectorbt) | 7k+ | 向量化回测、Numba JIT | (未来扩展) |

## 优化模块概览

### 1. 因子表达式引擎（`factor_expression_engine.py`）

**借鉴自**: qlib 的 DataLayer Expression Engine + akquant 的 Polars 因子引擎

**解决的问题**: jingni-trader 当前因子开发需要手写 `groupby + rolling` 循环，效率低且难复用

**核心特性**:
- 轻量级 DSL：`Mean($close, 20)`、`Rank(Delta($close, 1))`、`If(...)` 等
- 24 个内置算子：10 个时间序列、4 个截面、10 个数学/逻辑
- 安全 AST 求值（白名单节点类型，禁用 `eval/import`）
- 大小写不敏感

**性能**: 30 股票 × 800 天，10 个因子，0.24s，约 100 万行/秒

**与 jingni-trader 原实现的差异**:
- 原: `factor-engine/engine.py` 的 `compute_a_share_factors` 是硬编码因子公式
- 现: 声明式公式，新增因子无需修改 Python 代码

**测试**: 14 个测试全部通过

### 2. Walk-Forward 滚动训练（`walk_forward.py`）

**借鉴自**: akquant ML Guide 的 Walk-Forward Validation

**解决的问题**: jingni-trader 的 `strategy-model-engine` 当前只做"一次性全量训练"，没有 OOS 验证

**核心特性**:
- 滚动窗口：每个 fold 训练 + 测试，支持扩展 / 固定窗口
- Purge gap：训练/测试之间留 gap 防 look-ahead
- 多评估器：rank_ic, mse, accuracy, f1, custom
- OOS 预测拼接 + 整体 OOS 指标

**关键防御**:
- 训练集和测试集索引不重叠（test_05_oos_no_overlap_with_own_train）

**测试**: 9 个测试全部通过

**与 jingni-trader 原实现的差异**:
- 原: `train()` 方法一次性 fit 全量数据，无 OOS 评估
- 现: 滚动训练，每个 fold 一个独立 model，输出 OOS 预测

### 3. 动态因子权重融合（`dynamic_factor_fusion.py`）

**借鉴自**: AlphaForge 论文 + qlib IC 加权

**解决的问题**: jingni-trader 的 `factor_fusion` 是静态 IC_IR 加权，无法处理因子时效性

**核心特性**:
- 4 种融合方法：STATIC_IC_WEIGHTED（baseline）、EMA_IC_WEIGHTED、ADAPTIVE_TOPK（AlphaForge）、EQUAL_WEIGHT
- 自适应死因子过滤（`ic_floor`）
- 权重平滑（混入均匀权重避免跳变）
- 一键对比方法（`compare_methods`）

**测试**: 10 个测试全部通过

**与 jingni-trader 原实现的差异**:
- 原: 静态 IC_IR 加权（`factor-engine/engine.py:345-364`）
- 现: EMA IC 时序衰减 + TopK 动态筛选

### 4. 增强 IC 分析（`ic_analysis.py`）

**借鉴自**: qlib + alphalens

**解决的问题**: jingni-trader 的 IC 分析只有简单的 IC mean/IR，缺少衰减/分位数/换手/半衰期

**核心特性**:
- **IC Decay**: 不同 forward period（1/5/10/20/40/60d）的 IC
- **Quantile Returns**: 因子分 5/10 组看各组收益
- **Turnover**: 相邻两期因子排名变化程度
- **Half-life**: 用 AR(1) 估计因子衰减半衰期

**测试**: 8 个测试全部通过

**与 jingni-trader 原实现的差异**:
- 原: 只有 IC mean, IC IR, IC t-stat
- 现: IC Decay + 分位收益 + 换手 + 半衰期

## 使用方式

### 单元测试

```bash
cd /workspace
python3 -m unittest discover -s quant_opt_20260617 -p "test_*.py" -v
```

预期结果: 41 个测试全部通过

### 端到端验证

```bash
# 快速模式 (10 stocks × 300 days, ~10s)
python3 -m quant_opt_20260617.verify_all --quick

# 全量模式 (30 stocks × 800 days, ~45s)
python3 -m quant_opt_20260617.verify_all
```

报告输出到 `results/verification_report_*.md`

### 在业务代码中使用

```python
from quant_opt_20260617.factor_expression_engine import FactorEngine
from quant_opt_20260617.walk_forward import WalkForwardCV
from quant_opt_20260617.dynamic_factor_fusion import DynamicFactorFusion, FusionConfig, FusionMethod
from quant_opt_20260617.ic_analysis import ICAnalyzer

# 1) 表达式引擎
engine = FactorEngine()
df = engine.compute(ohlcv_df, [
    "Mean($close, 20)",
    "Rank(Delta($close, 5))",
    "Std(Mean($volume, 5))",
])

# 2) Walk-Forward 训练
cv = WalkForwardCV(
    model_factory=lambda: LGBMRegressor(),
    scorer="rank_ic",
    train_window_days=504,
    test_window_days=63,
    purge_gap_days=20,
)
result = cv.run(X, y, dates)
print(result.oos_predictions)  # 全部 OOS 预测

# 3) 动态因子融合
fuser = DynamicFactorFusion(FusionConfig(
    method=FusionMethod.EMA_IC_WEIGHTED,
    ema_halflife_days=60,
))
fused = fuser.fuse(factor_df, forward_returns=fwd_df)

# 4) 增强 IC 分析
analyzer = ICAnalyzer(forward_periods=[1, 5, 10, 20, 40])
report = analyzer.run(factor_df, price_df, factor_cols)
```

## 验证结果（2026-06-17）

| 场景 | 关键指标 | 结论 |
|------|----------|------|
| 因子表达式引擎 | 10 因子, 99.5 万行/秒 | PASS |
| Walk-Forward 滚动训练 | 15 窗口, OOS rank_ic=0.0363±0.064 | PASS |
| 动态因子融合 | 4 方法对比 | PASS |
| 增强 IC 分析 | 5 因子, IC Decay 25 行, 换手 3961 行 | PASS |
| 端到端集成 | 夏普=1.01, 累计收益=473.63% | PASS |

详细报告见 `results/` 目录。

## 与 jingni-trader 现有模块的集成建议

### factor-engine 集成
- `compute_a_share_factors()` 改为调用 `FactorEngine().compute(df, [...formula])`
- `factor_fusion()` 改为调用 `DynamicFactorFusion().fuse(...)`
- `ic_analysis()` 改为调用 `ICAnalyzer().run(...)`

### strategy-model-engine 集成
- `train()` 改为调用 `WalkForwardCV.run()`
- 输出 `oos_predictions` 给到 reports-engine 做归因

### 风险与限制
1. **依赖**: pandas, numpy, scipy, scikit-learn（jingni-trader 已具备）
2. **数据格式**: 与 jingni-trader 现有 schema 完全兼容（code, date, OHLCV...）
3. **性能**: 100 万行/秒，远超原实现（建议在原代码加 profile 后再决定）
4. **未实现**: 多核并行、分布式、GPU 加速（属于后续优化方向）

## 待用户确认的优化建议

1. **优先级高**: 将 `factor_expression_engine.py` 集成到 `factor-engine`，替换手写循环
2. **优先级高**: 将 `walk_forward.py` 集成到 `strategy-model-engine`，作为 OOS 验证
3. **优先级中**: 将 `dynamic_factor_fusion.py` 替换原静态 IC_IR 加权
4. **优先级中**: 将 `ic_analysis.py` 集成到 `reports-engine`，增加 IC Decay / 半衰期报告
5. **优先级低**: 引入并行/分布式计算（参考 qlib 的 TaskRunner）

## 重要约束

**未经用户确认前，本分支代码不会合并到 main 分支**

允许操作：
- ✓ 编译
- ✓ 单元测试
- ✓ 端到端验证
- ✓ git push 本分支到 GitHub

禁止操作：
- ✗ git merge / PR 合入 main
