# 量化交易优化验证报告 - 2026-06-17 04:22:40

**总耗时**: 43.8s  
**验证场景数**: 5  
**借鉴项目**: microsoft/qlib, akquant, AlphaForge, vectorbt  
**分支**: feat/quant-opt-20260617  

## 验证场景汇总

| 场景 | 关键指标 | 结论 |
|------|----------|------|
| 因子表达式引擎 | 10 因子, 995179 行/秒, 所有因子有数据=True | PASS
| Walk-Forward 滚动训练 | 15 窗口, OOS rank_ic=0.0363±0.0640, OOS 覆盖率=58.1% | PASS |
| 动态因子融合 | 4 方法对比, 最佳方法: static_ic_weighted | PASS |
| 增强 IC 分析 | 5 因子, IC Decay 25 行, 换手 3961 行 | PASS |
| 端到端集成 | 表达式引擎→融合→回测, 夏普=1.01, 累计收益=473.63% | PASS |

---

## factor_expression_engine

```json
{
  "scenario": "factor_expression_engine",
  "n_factors": 10,
  "n_factors_with_data": 10,
  "elapsed_sec": 0.241,
  "rows_per_sec": 995179,
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

## walk_forward

```json
{
  "scenario": "walk_forward",
  "n_windows": 15,
  "elapsed_sec": 0.091,
  "oos_coverage": 0.5806,
  "oos_rank_ic_mean": 0.0363,
  "oos_rank_ic_std": 0.064,
  "oos_rank_ic_ir": 0.5676,
  "train_window_days": 300,
  "test_window_days": 30,
  "purge_gap_days": 10,
  "n_features": 4,
  "config": {
    "train_window_days": 300,
    "test_window_days": 30,
    "step_days": 30,
    "purge_gap_days": 10,
    "expanding": true,
    "scorer": "auto",
    "n_windows": 15
  }
}
```

---

## dynamic_factor_fusion

```json
{
  "scenario": "dynamic_factor_fusion",
  "elapsed_sec": 11.489,
  "methods_compared": 4,
  "comparison": [
    {
      "method": "static_ic_weighted",
      "status": "OK",
      "rank_ic": 0.0034,
      "rank_ic_std": 0.1769,
      "ic_ir": 0.0194,
      "long_short_return": -0.0007,
      "n_days": 775
    },
    {
      "method": "ema_ic_weighted",
      "status": "OK",
      "rank_ic": -0.0043,
      "rank_ic_std": 0.1922,
      "ic_ir": -0.0222,
      "long_short_return": -0.0008,
      "n_days": 775
    },
    {
      "method": "adaptive_topk",
      "status": "OK",
      "rank_ic": 0.0034,
      "rank_ic_std": 0.1769,
      "ic_ir": 0.0194,
      "long_short_return": -0.0007,
      "n_days": 775
    },
    {
      "method": "equal_weight",
      "status": "OK",
      "rank_ic": 0.0034,
      "rank_ic_std": 0.1769,
      "ic_ir": 0.0194,
      "long_short_return": -0.0007,
      "n_days": 775
    }
  ],
  "best_method": "static_ic_weighted"
}
```

---

## ic_analysis

```json
{
  "scenario": "ic_analysis",
  "elapsed_sec": 27.998,
  "factors_analyzed": 5,
  "n_ic_decay_rows": 25,
  "n_quantile_rows": 30,
  "n_turnover_rows": 3961,
  "n_half_life_rows": 5,
  "top_ic_combo": {
    "factor": "ret_5d",
    "forward_period": 1,
    "ic_mean": 0.0275,
    "ic_std": 0.1922,
    "ic_ir": 0.1431,
    "ic_pos_ratio": 0.568,
    "ic_t_stat": 4.033,
    "n_days": 794
  },
  "fastest_decay_factor": {
    "factor": "momentum_volume",
    "ic_half_life": 0.8259,
    "n_days": 790,
    "ic_mean": -0.0152
  },
  "mean_ic_decay": 0.0033
}
```

---

## integration

```json
{
  "scenario": "integration",
  "n_factors_expressed": 6,
  "n_factors_generated": 6,
  "fuse_elapsed_sec": 3.02,
  "n_alpha_dates": 775,
  "avg_long_return_5d": 0.002397,
  "std_long_return_5d": 0.01678,
  "sharpe_ratio_5d_hold": 1.0142,
  "cumulative_return": 4.7363
}
```
