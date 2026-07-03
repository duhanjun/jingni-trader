# 量化交易优化验证（feat/quant-opt-20260617）

本目录是 jingni-trader 的"开箱验证"实验区，所有代码不修改 main 分支。

## 目录结构

```
quant_opt_20260617/
├── vectorized_backtest/
│   └── vectorized_engine.py    # 向量化回测引擎（借鉴 VectorBT）
├── risk_engine/
│   └── multi_layer_risk.py     # 多层风控引擎（借鉴 NautilusTrader）
├── tests/
│   ├── test_vectorized_engine.py
│   └── test_risk_engine.py
└── reports/                    # 测试输出（性能 / 审计日志）
```

## 借鉴来源

| 项目 | 借鉴点 | 本目录对应模块 |
|------|--------|----------------|
| [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | 100x-1000x 向量化回测 | `vectorized_backtest/` |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | RiskEngine + TradingState + Throttler | `risk_engine/` |

## 运行测试

```bash
# 进入项目根
cd /workspace

# 向量化引擎
PYTHONPATH=. python quant_opt_20260617/tests/test_vectorized_engine.py

# 风控引擎
PYTHONPATH=. python quant_opt_20260617/tests/test_risk_engine.py
```

测试结果会输出到 `reports/` 目录：
- `perf_vectorized_vs_native.json`：性能对比
- `risk_audit.jsonl`：风控审计日志
