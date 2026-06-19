# quant_opt: jingni-trader 优化验证模块

本目录是 jingni-trader 项目在 `feat/quant-opt-20260619` 分支上进行的优化验证。

**不修改 main 分支的任何代码**, 全部代码均位于本目录下的独立模块中。

## 模块结构

```
quant_opt/
├── factor_expression/      # 因子表达式引擎 (借鉴 AKQuant / Qlib)
├── vectorized_backtest/     # 向量化回测引擎 (借鉴 VectorBT)
├── extended_metrics/        # 扩展绩效指标 (借鉴 VectorBT 60+ 指标)
├── tests/                   # 单元测试 (31 个, 全部通过)
└── reports/
    ├── REPORT_20260619.md   # 完整学习与验证报告
    └── test_run_output.txt  # 测试输出
```

## 快速使用

```bash
# 运行所有测试
cd /workspace && python3 quant_opt/tests/run_all.py

# 单独测试
python3 quant_opt/tests/test_factor_expression.py
python3 quant_opt/tests/test_extended_metrics.py
python3 quant_opt/tests/test_vectorized_backtest.py
```

详见 [`reports/REPORT_20260619.md`](reports/REPORT_20260619.md)
