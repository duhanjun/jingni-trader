# 量化交易优化验证报告 - 2026-06-17 04:21:15

**总耗时**: 10.3s  
**验证场景数**: 5  
**借鉴项目**: microsoft/qlib, akquant, AlphaForge, vectorbt  
**分支**: feat/quant-opt-20260617  

## 验证场景汇总

| 场景 | 关键指标 | 结论 |
|------|----------|------|
| 因子表达式引擎 | 10 因子, 326696 行/秒, 所有因子有数据=True | PASS
| 动态因子融合 | 4 方法对比, 最佳方法: ema_ic_weighted | PASS |
| 增强 IC 分析 | 5 因子, IC Decay 0 行, 换手 1461 行 | PASS |
| 端到端集成 | 表达式引擎→融合→回测, 夏普=0.51, 累计收益=49.65% | PASS |

---

## factor_expression_engine

```json
{
  "scenario": "factor_expression_engine",
  "n_factors": 10,
  "n_factors_with_data": 10,
  "elapsed_sec": 0.092,
  "rows_per_sec": 326696,
  "all_factors_have_data": true,
  "operator_count": 24,
  "operators": {
    "time_series": [
      "decaylinear",
      "delta",
      "ema",
      "max",
      "mean",
      "min",
      "ref",
      "std",
      "sum",
      "tsrank"
    ],
    "cross_section": [
      "mad",
      "quantile",
      "rank",
      "scale"
    ],
    "math": [
      "abs",
      "and",
      "equal",
      "greater",
      "if",
      "less",
      "log",
      "or",
      "sign",
      "sqrt"
    ]
  },
  "sample_factors": [
    "Mean_close_5",
    "Mean_close_10",
    "Mean_close_20",
    "Std_close_20",
    "Delta_close_5"
  ]
}
```

---

## scenario_2_walk_forward

```json
{
  "scenario": "scenario_2_walk_forward",
  "status": "FAILED",
  "error": "无法生成任何 walk-forward 窗口，请检查日期范围或调整参数"
}
```

---

## dynamic_factor_fusion

```json
{
  "scenario": "dynamic_factor_fusion",
  "elapsed_sec": 3.686,
  "methods_compared": 4,
  "comparison": [
    {
      "method": "static_ic_weighted",
      "status": "OK",
      "rank_ic": -0.057,
      "rank_ic_std": 0.3309,
      "ic_ir": -0.1723,
      "long_short_return": NaN,
      "n_days": 275
    },
    {
      "method": "ema_ic_weighted",
      "status": "OK",
      "rank_ic": -0.0545,
      "rank_ic_std": 0.3431,
      "ic_ir": -0.1588,
      "long_short_return": NaN,
      "n_days": 275
    },
    {
      "method": "adaptive_topk",
      "status": "OK",
      "rank_ic": -0.057,
      "rank_ic_std": 0.3309,
      "ic_ir": -0.1723,
      "long_short_return": NaN,
      "n_days": 275
    },
    {
      "method": "equal_weight",
      "status": "OK",
      "rank_ic": -0.057,
      "rank_ic_std": 0.3309,
      "ic_ir": -0.1723,
      "long_short_return": NaN,
      "n_days": 275
    }
  ],
  "best_method": "ema_ic_weighted"
}
```

---

## ic_analysis

```json
{
  "scenario": "ic_analysis",
  "elapsed_sec": 5.24,
  "factors_analyzed": 5,
  "n_ic_decay_rows": 0,
  "n_quantile_rows": 30,
  "n_turnover_rows": 1461,
  "n_half_life_rows": 0,
  "top_ic_combo": {},
  "fastest_decay_factor": {},
  "mean_ic_decay": null
}
```

---

## integration

```json
{
  "scenario": "integration",
  "n_factors_expressed": 6,
  "n_factors_generated": 6,
  "fuse_elapsed_sec": 0.98,
  "n_alpha_dates": 275,
  "avg_long_return_5d": 0.001769,
  "std_long_return_5d": 0.024591,
  "sharpe_ratio_5d_hold": 0.5106,
  "cumulative_return": 0.4965
}
```
