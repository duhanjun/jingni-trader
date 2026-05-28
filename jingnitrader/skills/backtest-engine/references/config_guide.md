# 配置指南

本文档说明 backtest-engine 的配置选项。

## 环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| BACKTEST_BACKEND | 回测引擎后端 | 否 | "rqalpha" |
| BACKTEST_DIR | 回测结果存储目录 | 否 | "./quant_workspace/backtest_results" |
| BENCHMARK | 基准指数 | 否 | "000300.SH" |
| INIT_CAPITAL | 初始资金 | 否 | 1000000 |
| RISK_FREE_RATE | 无风险利率 | 否 | 0.03 |

## 配置文件

配置文件位于 `scripts/config.py`。

### 费用配置

```python
COMMISSION_RATE = 0.00025      # 佣金 万2.5
MIN_COMMISSION = 5.0            # 最低佣金 5元
STAMP_TAX_RATE = 0.001         # 印花税 千1（卖出）
TRANSFER_FEE_RATE = 0.00002    # 过户费 万0.2
SLIPPAGE = 0.0001              # 滑点 万1
```

### 引擎配置

```python
BACKTEST_BACKEND = os.getenv("BACKTEST_BACKEND", "rqalpha")
BACKTEST_DIR = os.path.expanduser(os.getenv("BACKTEST_DIR", "./quant_workspace/backtest_results"))
INIT_CAPITAL = float(os.getenv("INIT_CAPITAL", "1000000"))
BENCHMARK = os.getenv("BENCHMARK", "000300.SH")
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.03"))
```

## 后端选择

- `rqalpha`: 锐汇量化回测框架
- `backtrader`: Backtrader 回测框架
- `gm`: 掘金量化

## 使用示例

```python
import os
os.environ['BACKTEST_BACKEND'] = 'rqalpha'
os.environ['BENCHMARK'] = '000300.SH'

from engine import run, BacktestEngine
result = run(ctx)
```
