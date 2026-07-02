# 配置指南

## 基础配置

### 执行后端配置
- `EXECUTION_BACKEND`: 交易执行后端，可选值：
  - `sim`: 模拟交易（默认）
  - `xtquant`: 迅投量化
  - `gm`: 掘金量化
  - `vnpy`: vn.py

### 模拟交易模式
- `PAPER_TRADE`: 是否启用模拟交易模式，默认 `true`

### 账户配置
- `INITIAL_CAPITAL`: 初始资金，默认 `1000000.0`

## 风险控制配置

### 单日亏损限制
- `MAX_DAILY_LOSS_RATE`: 单日最大亏损比例，默认 `0.02` (2%)

### 订单限制
- `MAX_SINGLE_ORDER_AMOUNT`: 单笔最大订单金额，默认 `100000.0`
- `MAX_ORDER_FREQUENCY_PER_MINUTE`: 每分钟最大下单次数，默认 `10`

### 持仓限制
- `MAX_POSITION_CONCENTRATION`: 最大持仓集中度，默认 `0.3` (30%)

## 交易成本配置

### 手续费
- `COMMISSION_RATE`: 手续费率，默认 `0.00025` (万分之2.5)
- `MIN_COMMISSION`: 最低手续费，默认 `5.0`

### 滑点
- `SLIPPAGE`: 模拟滑点比例，默认 `0.001` (千分之1)

## 日志配置

- `LOG_DIR`: 审计日志目录，默认 `./logs`
- `TRADE_LOG_DIR`: 交易日志目录，默认 `./trade_logs`

## 环境变量

所有配置项都可以通过环境变量设置，例如：
```bash
export PAPER_TRADE=true
export EXECUTION_BACKEND=sim
export INITIAL_CAPITAL=1000000
```

## 使用示例

```python
from engine import ExecutionEngine
from config import get_config

# 自定义配置
config = get_config(
    EXECUTION_BACKEND="sim",
    INITIAL_CAPITAL=2000000,
    MAX_DAILY_LOSS_RATE=0.01
)

engine = ExecutionEngine(config)
```
