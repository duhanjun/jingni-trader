# jingni-trader 量化优化实验模块

**分支**: `feat/quant-opt-20260615`
**执行日期**: 2026-06-15
**状态**: 实验性代码，不直接修改 main 分支

## 模块概览

| 模块 | 文件 | 借鉴来源 | 测试 |
|------|------|---------|------|
| 安全因子表达式引擎 | [`factor_expression_engine.py`](file:///workspace/skills/quant_opt/factor_expression_engine.py) | AKQuant, 国内多因子系统 | 19 项 |
| 向量化回测引擎 | [`vectorized_backtest.py`](file:///workspace/skills/quant_opt/vectorized_backtest.py) | AKQuant, simtradelab, Qlib | 7 项 |
| Walk-forward 滚动框架 | [`walk_forward.py`](file:///workspace/skills/quant_opt/walk_forward.py) | AKQuant, FinRL-X, Qlib | 4 项 |
| 性能/精度对比基准 | [`benchmark.py`](file:///workspace/skills/quant_opt/benchmark.py) | 自研 | - |

## 快速开始

### 运行所有测试

```bash
cd /workspace
python -m unittest discover -s skills/quant_opt/tests -v
```

### 运行性能/精度对比基准

```bash
cd /workspace
python -m skills.quant_opt.benchmark
```

### 因子表达式引擎示例

```python
from skills.quant_opt.factor_expression_engine import FactorEngine
import pandas as pd

engine = FactorEngine()
df = pd.read_parquet("your_data.parquet")  # 含 code, date, close, volume 等列

# 单因子
out = engine.compute(df, "Rank(Ts_Mean(Delta($close, 1), 20))", name="alpha1")

# 批量因子
out = engine.compute_many(df, {
    "mom_5": "Delta($close, 5) / $close",
    "turnover_rank": "Rank($volume)",
    "vol_20d": "StdDev(Delta($close, 1), 20)",
})
```

### 向量化回测示例

```python
from skills.quant_opt.vectorized_backtest import VectorizedBacktest

bt = VectorizedBacktest(init_capital=1_000_000)
result = bt.run(data=df, signals=signals_df)
print(result.metrics)
print(f"耗时: {result.runtime_seconds:.3f}s")
```

### Walk-forward 示例

```python
from skills.quant_opt.walk_forward import WalkForwardRunner, generate_windows

def my_train(data, window):
    return {"mean_close": data["close"].mean()}

def my_backtest(model, data, window):
    return {"return": (data["close"].iloc[-1] / data["close"].iloc[0] - 1)}

runner = WalkForwardRunner(train_fn=my_train, backtest_fn=my_backtest)
windows = generate_windows("2018-01-01", "2024-01-01", train_months=12,
                           valid_months=3, test_months=3, step_months=6)
report = runner.run(data, windows)
print(report["summary"])
```

## 详细报告

完整学习成果与优化分析见 [`report.md`](file:///workspace/skills/quant_opt/report.md)。
