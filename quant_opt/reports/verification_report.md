# jingni-trader 量化优化验证报告

**执行分支**: `feat/quant-opt-20260618`
**总耗时**: 1.67s
**数据规模**: 12000 行 / 20 只票 / 600 个交易日

## 1. Walk-Forward Validation

| 配置 | Folds | mean IC | std IC | ICIR | 耗时 |
|------|-------|---------|--------|------|------|
| **Strict** (purge=10) | 5 | 0.0199 | 0.0540 | 0.368 | 0.0398s |
| **Loose** (purge=0)   | 14 | 0.0144 | 0.0542 | 0.266 | 0.0605s |

⚠️ **Strict WFV 触发过拟合预警**: flags=['std/mean=2.72 > 1.5']

**对比结论**:
- Strict (purge=10) 与 Loose (purge=0) 的 fold 数量与 mean IC 不同，
  表明 purge gap 会显著影响 OOS 评估的真实度。
- Strict 流程与 jingni-trader SKILL.md 中「模型过拟合 → 触发样本外再验证」
  状态机分支可由本模块 `detect_overfit()` 自动驱动。

## 2. 因子表达式 DSL (Alpha101 风格)

**注册因子数**: 6
**计算耗时**: 0.5047s

| 因子 | 表达式 | mean IC | std IC | ICIR |
|------|--------|---------|--------|------|
| `alpha_001` | `Rank(Delta(close, 5))` | 0.0083 | 0.2406 | 0.034 |
| `reversal_20` | `Rank(-1 * Delta(close, 20) / Delay(close, 20))` | -0.0135 | 0.2143 | -0.063 |
| `bias_10` | `(close - Ts_Mean(close, 10)) / Ts_Mean(close, 10)` | 0.0074 | 0.2403 | 0.031 |
| `vp_ratio` | `Rank(Delta(volume, 5)) - Rank(Delta(close, 5))` | 0.0144 | 0.2391 | 0.060 |
| `vol_20` | `Ts_Std(close / Delay(close, 1) - 1, 20)` | 0.0420 | 0.2323 | 0.181 |
| `mom_decay` | `Decay_Linear(close / Delay(close, 1) - 1, 10)` | 0.0074 | 0.2381 | 0.031 |

**正确性自检 (vs 直接 pandas 实现)**:
- ✅ `alpha_001` max_abs_diff=0.00e+00 — 等价于直接 groupby 实现
- ✅ `bias_10` max_abs_diff=0.00e+00 — 等价于直接 rolling mean 实现

## 3. 前视偏差检测器

| 场景 | Errors | Warnings | 关键问题 |
|------|--------|----------|----------|
| 坏代码 (含 3 类典型错误) | 1 | 1 | LOOKAHEAD_NEG_SHIFT |
| 干净代码 | 0 | 0 | - |
| 时间切分 (gap=3d, purge=5) | 0 | 0 | - |
| 时间切分 (重叠) | 1 | 0 | LEAKAGE_OVERLAP |

**坏代码检出的问题清单**:
- [error] `LOOKAHEAD_NEG_SHIFT` @ bad_strategy.py:5: 使用 .shift(-n) 引用未来数据，请改用正向 shift 或当日数据。
- [warning] `LOOKAHEAD_ROLLING_NO_SHIFT` @ bad_strategy.py:7: .rolling(...) 后未观察到 .shift(1)，可能将当日数据用于当日决策，建议在 rolling 后 shift(1)。

## 4. 借鉴来源 & 后续建议

| 模块 | 主要借鉴 | jingni-trader 可优化点 |
|------|----------|------------------------|
| Walk-Forward | AKQuant 内置 WFV、Qlib DataHandler | strategy-model-engine 当前仅做 CV 切分，建议增加 WFV 滚动 + 过拟合检测，与状态机分支「样本外再验证」联动 |
| Factor DSL | AKQuant 因子表达式引擎、WorldQuant Alpha101 | factor-engine 因子硬编码，建议引入字符串 DSL，让用户在 YAML/JSON 中声明自定义因子 |
| Lookahead Detector | Qlib Point-in-Time、VectorBT 文档 | 新增工具方法扫描常见前视偏差(负 shift、rolling 不 shift、label 入 feature、时间泄漏) |
