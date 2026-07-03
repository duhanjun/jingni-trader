# jingni-trader 量化优化模块（feat/quant-opt-20260618）

本目录包含对 jingni-trader 项目的优化验证代码，基于 2026-06-18 的联网学习成果。

## 模块组成

| 子目录 | 借鉴来源 | 作用 |
|--------|---------|------|
| `pit_checker/` | Microsoft Qlib | Point-in-Time 数据完整性检查 |
| `factor_dsl/` | Qlib + AKQuant | 因子表达式 DSL（公式化因子定义） |
| `wf_validator/` | Qlib Rolling Training | Walk-Forward 验证器（样本外评估） |
| `tests/` | - | 单元测试 + 性能基准 |
| `reports/` | - | 验证报告与基准结果 |

## 快速开始

```python
# 1) PIT 检查
from quant_opt_20260618.pit_checker.checker import check_pit
report = check_pit(df, pit_columns=["announce_date"])
print(report.is_clean, report.violations)

# 2) 因子 DSL
from quant_opt_20260618.factor_dsl.engine import FactorEngine, FactorExpression
engine = FactorEngine()
engine.register(FactorExpression("mom_5", "Mean($close, 5)"))
engine.register(FactorExpression("alpha_1", "Rank(mom_5) - Rank(Std($close, 20))"))
factors = engine.compute(price_df)

# 3) WFA 验证
from quant_opt_20260618.wf_validator.splitter import TimeSeriesSplitter, WalkForwardValidator
splitter = TimeSeriesSplitter(train_period_days=252, test_period_days=63, step_days=63)
folds = splitter.split(data)
validator = WalkForwardValidator(factor_col="alpha_score", ret_col="ret_forward_1d")
report = validator.run(data, folds)
print(report.print_summary())
```

## 运行测试

```bash
# 全部测试
python3 quant_opt_20260618/tests/run_all.py

# 单独模块
python3 -m pytest quant_opt_20260618/tests/test_pit_checker.py -v
python3 -m pytest quant_opt_20260618/tests/test_factor_dsl.py -v
python3 -m pytest quant_opt_20260618/tests/test_wf_validator.py -v

# 性能基准
python3 quant_opt_20260618/tests/benchmark.py
```

## 测试结果

- ✅ PIT Checker: 10/10
- ✅ Factor DSL: 15/15
- ✅ WFA Validator: 11/11
- **合计：36/36 全部通过**

详细报告见 `reports/optimization_report_2026-06-18.md`。
