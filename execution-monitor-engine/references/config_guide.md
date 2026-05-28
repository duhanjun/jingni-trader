# 配置指南

本文档说明 execution-monitor-engine 的配置选项。

## 环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| TRADE_MODE | 交易模式 | 否 | "paper" |
| TRADE_BACKEND | 交易接口后端 | 否 | "xtquant" |
| EXECUTION_DIR | 执行日志目录 | 否 | "./quant_workspace/execution" |
| INIT_CAPITAL | 初始资金 | 否 | 1000000 |

## 配置文件

配置文件位于 `scripts/config.py`。

### 风控配置

```python
MAX_DAILY_LOSS_RATIO = 0.02          # 单日最大亏损 2%
MAX_SINGLE_ORDER_RATIO = 0.10        # 单笔订单不超过净资产10%
MAX_SINGLE_STOCK_WEIGHT = 0.10       # 单票最大持仓10%
MAX_ORDER_FREQUENCY = 2              # 每秒最多2笔
```

### 费用配置

```python
COMMISSION_RATE = 0.00025            # 佣金 万2.5
MIN_COMMISSION = 5.0                 # 最低佣金 5元
STAMP_TAX_RATE = 0.001              # 印花税 千1
SLIPPAGE = 0.0001                   # 滑点 万1
```

### 路径配置

```python
EXECUTION_DIR = os.path.expanduser(os.getenv("EXECUTION_DIR", "./quant_workspace/execution"))
AUDIT_LOG_PATH = os.path.join(EXECUTION_DIR, "audit.log")
ACCOUNT_STATE_PATH = os.path.join(EXECUTION_DIR, "account_state.json")
```

## 模式选择

- `paper`: 模拟交易，本地虚拟账户运行
- `live`: 实盘交易，需要配置券商接口

## 使用示例

```python
import os
os.environ['TRADE_MODE'] = 'paper'

from engine import run, PaperExecutor, CircuitBreaker

result = run(ctx)
```
