# 配置指南

本文档说明 execution-monitor-engine 的配置选项。

## 环境变量

### 交易模式

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| TRADE_MODE | 交易模式（paper / live） | 否 | "paper" |
| TRADE_BACKEND | 交易接口后端（xtquant / gm） | 否 | "xtquant" |
| QUANT_WORK_DIR | 工作目录根路径 | 否 | "./workspace" |
| EXECUTION_DIR | 执行日志目录 | 否 | "{QUANT_WORK_DIR}/execution" |

### 实盘交易后端配置

#### xtquant (miniQMT)

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| XTQUANT_PATH | miniQMT 安装目录下的 userdata_mini 路径 | live+xtquant 必需 | 空 |
| XTQUANT_ACCOUNT | miniQMT 资金账号 | live+xtquant 必需 | 空 |

#### gm (掘金量化)

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| GM_TOKEN | 掘金量化 API Token | live+gm 必需 | 空 |
| GM_ACCOUNT_ID | 掘金账户 ID（终端获取） | live+gm 必需 | 空 |

### 风控配置

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| MAX_DAILY_LOSS_RATIO | 单日最大亏损比例 | 0.02 (2%) |
| MAX_SINGLE_ORDER_RATIO | 单笔订单最大金额比例 | 0.10 (10%) |
| MAX_SINGLE_STOCK_WEIGHT | 单票最大持仓比例 | 0.10 (10%) |
| MAX_ORDER_FREQUENCY | 每秒最大下单笔数 | 2 |

### 费用配置（paper 模式）

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| INIT_CAPITAL | 初始资金 | 1000000 |
| COMMISSION_RATE | 佣金费率 | 0.00025 (万2.5) |
| MIN_COMMISSION | 最低佣金 | 5.0 元 |
| STAMP_TAX_RATE | 印花税率（卖出） | 0.001 (千1) |
| SLIPPAGE | 滑点模拟比例 | 0.001 (千1) |

### 路径配置

```python
EXECUTION_DIR = os.path.join(QUANT_WORK_DIR, "execution")
AUDIT_LOG_PATH = os.path.join(EXECUTION_DIR, "trade_log.jsonl")
ACCOUNT_STATE_PATH = os.path.join(EXECUTION_DIR, "account_state.json")
```

## 模式选择

### paper（模拟交易）

本地虚拟账户，支持滑点模拟、T+1约束、数量校验、资金校验、断路器风控、审计日志、状态持久化。

```python
import os
os.environ['TRADE_MODE'] = 'paper'
```

### live + xtquant（miniQMT 实盘）

需本地运行 miniQMT 客户端，连接 xtdata 数据服务和 XtQuantTrader 交易接口。

```bash
# 环境变量配置
TRADE_MODE=live
TRADE_BACKEND=xtquant
XTQUANT_PATH=D:\gszq\qmt\userdata_mini
XTQUANT_ACCOUNT=你的资金账号
```

### live + gm（掘金量化实盘）

需配置掘金 Token 和账户 ID，通过 set_token + set_account_id 连接。

```bash
# 环境变量配置
TRADE_MODE=live
TRADE_BACKEND=gm
GM_TOKEN=你的掘金Token
GM_ACCOUNT_ID=你的掘金账户ID
```

## 使用示例

```python
import os
os.environ['TRADE_MODE'] = 'paper'

from engine import run, PaperExecutor, CircuitBreaker

result = run(ctx)
```
