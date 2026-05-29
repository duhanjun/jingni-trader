# 配置指南

本文档说明 jingnitrader 的配置选项和使用方法。

## 环境变量

### 必需的环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| TUSHARE_TOKEN | Tushare Pro API Token | 否 | 空 |
| GM_TOKEN | 掘金量化API Token | 否 | 空 |
| DATA_DIR | 数据和工作目录 | 否 | "./workspace" |
| LOG_LEVEL | 日志级别 | 否 | "INFO" |

### 可选的环境变量

| 变量名 | 描述 | 可选值 | 默认值 |
|--------|------|--------|--------|
| DATA_BACKEND | 数据源后端 | tushare/baostock/akshare/xtquant/gm | "tushare" |
| BACKTEST_BACKEND | 回测框架 | rqalpha/backtrader/gm | "rqalpha" |
| TRADE_BACKEND | 交易接口 | xtquant/gm | "xtquant" |
| FACTOR_BACKEND | 因子计算库 | talib/pandas_ta | "talib" |

## 配置文件

配置文件位于 `scripts/config.py`，包含以下配置项：

### 工作目录

```python
WORK_DIR = "./workspace"
DATA_DIR = os.path.join(WORK_DIR, "data")
FACTOR_DIR = os.path.join(WORK_DIR, "factors")
MODEL_DIR = os.path.join(WORK_DIR, "models")
BACKTEST_DIR = os.path.join(WORK_DIR, "backtest_results")
PORTFOLIO_DIR = os.path.join(WORK_DIR, "portfolio")
REPORT_DIR = os.path.join(WORK_DIR, "reports")
LOG_DIR = os.path.join(WORK_DIR, "logs")
```

### A股市场配置

```python
A_SHARE_COMMISSION_RATE = 0.00025    # 佣金 万2.5
A_SHARE_STAMP_TAX = 0.001           # 印花税 千1（卖出）
A_SHARE_MIN_COMMISSION = 5.0        # 最低佣金 5元
A_SHARE_MIN_LOT = 100               # 最小交易单位
A_SHARE_T_PLUS_1 = True             # T+1 交易
```

### 风控阈值

```python
MAX_DAILY_LOSS_RATIO = 0.03         # 单日最大亏损 3%
MAX_SINGLE_STOCK_WEIGHT = 0.10      # 单票最大持仓 10%
MAX_INDUSTRY_DEVIATION = 0.05       # 行业偏离基准 ±5%
NEW_STOCK_EXCLUDE_DAYS = 60         # 新股保护期
```

## 使用示例

### 1. 设置环境变量

```bash
export TUSHARE_TOKEN="your_token_here"
export GM_TOKEN="your_gm_token_here"
export DATA_BACKEND="tushare"
export LOG_LEVEL="DEBUG"
```

### 2. 在 Python 中使用

```python
import os
os.environ['TUSHARE_TOKEN'] = 'your_token_here'

from engine import run, MasterEngine
from context import Context

# 创建 Context
ctx = Context(
    task_id="task_001",
    user_intent="帮我用近3年A股数据做一个20日反转因子选股回测",
    current_stage="IDLE"
)

# 运行
result = run(ctx=ctx)
```

### 3. CLI 使用

```bash
# 设置环境变量
export TUSHARE_TOKEN="your_token_here"

# 运行
python engine.py -i "帮我用近3年A股数据做一个20日反转因子选股回测"
```

## 故障排除

### 常见问题

1. **Token 未设置**
   - 确保已设置 `TUSHARE_TOKEN` 环境变量
   - 或在代码中设置 `os.environ['TUSHARE_TOKEN'] = 'your_token'`

2. **目录权限错误**
   - 确保 `DATA_DIR` 目录存在且有写入权限
   - 或使用默认目录 `./workspace`

3. **后端加载失败**
   - 检查相应的包是否已安装
   - 如使用 tushare 后端，确保 `pip install tushare`
