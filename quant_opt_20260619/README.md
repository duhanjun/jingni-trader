# quant_opt_20260619

jingni-trader 项目学习与优化验证目录.

## 目录内容

| 文件 | 作用 |
|---|---|
| `factor_dsl_engine.py` | 优化 A: 因子表达式 DSL 引擎（借鉴 qlib + KunQuant） |
| `vectorized_metrics.py` | 优化 B: 向量化绩效指标（借鉴 vectorbt） |
| `test_validation.py` | 10 个测试用例（正确性/性能/边界） |
| `test_output.log` | 测试运行日志 |
| `VERIFICATION_REPORT.md` | 完整验证报告 |

## 快速运行

```bash
cd quant_opt_20260619
pip install numpy pandas scipy
python3 test_validation.py
```

## 集成到 jingni-trader (待用户确认)

- `factor_dsl_engine.py` → `skills/factor-engine/scripts/adapters/dsl_calculator.py`
- `vectorized_metrics.py` → 合并到 `skills/backtest-engine/scripts/base/base_backtest.py`

详见 [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) 第 6 节.