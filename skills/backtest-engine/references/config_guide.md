# 配置指南

本文档说明 backtest-engine 的配置选项。

## 环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| BACKTEST_DIR | 回测结果存储目录 | 否 | "./workspace/backtest_results" |
| QUANT_WORK_DIR | 工作目录根路径 | 否 | "./workspace" |
| BENCHMARK | 基准指数 | 否 | "000300.SH" |
| INIT_CAPITAL | 初始资金 | 否 | 1000000 |
| RISK_FREE_RATE | 无风险利率 | 否 | 0.03 |
| COMMISSION_RATE | 佣金费率 | 否 | 0.00025 |
| MIN_COMMISSION | 最低佣金（元） | 否 | 5.0 |
| STAMP_TAX_RATE | 印花税率（卖出） | 否 | 0.001 |
| TRANSFER_FEE_RATE | 过户费率（双侧） | 否 | 0.00002 |
| SLIPPAGE | 滑点比例 | 否 | 0.001 |

## 配置文件

配置文件位于 `scripts/config.py`。

### 费用配置

```python
COMMISSION_RATE = 0.00025      # 佣金 万2.5
MIN_COMMISSION = 5.0            # 最低佣金 5元
STAMP_TAX_RATE = 0.001         # 印花税 千1（卖出）
TRANSFER_FEE_RATE = 0.00002    # 过户费 万0.2（双侧）
SLIPPAGE = 0.001               # 滑点 0.1%（双侧应用）
```

### 引擎配置

```python
BACKTEST_DIR = os.getenv("BACKTEST_DIR", "./workspace/backtest_results")
INIT_CAPITAL = float(os.getenv("INIT_CAPITAL", "1000000"))
BENCHMARK = os.getenv("BENCHMARK", "000300.SH")
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.03"))
```

## 使用示例

```python
import os
os.environ['BENCHMARK'] = '000300.SH'

from engine import run, BacktestEngine
result = run(ctx)
```
