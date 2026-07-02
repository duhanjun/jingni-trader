# 实盘对接指南

本指南帮助您将量化交易系统连接到真实交易接口。

---

## 📋 对接方案总览

| 方案 | 难度 | 需要账号 | 特点 |
|------|------|---------|------|
| **模拟交易** | ⭐ | ❌ | 立即可用，无需配置 |
| **掘金量化(gm)** | ⭐⭐ | ✅ | 云端服务，支持回测+实盘 |
| **迅投QMT(xtquant)** | ⭐⭐⭐ | ✅ | 专业级，仅Windows |

---

## 🚀 快速开始：模拟交易

### 无需任何配置，直接使用

```python
import sys
sys.path.insert(0, 'skills/execution-monitor-engine')
from engine import ExecutionEngine
from config import get_config

# 创建引擎（默认就是模拟模式）
config = get_config()
engine = ExecutionEngine(config)

# 初始化账户
engine.initialize_account(initial_capital=1000000)

# 下单
order = engine.place_order(
    symbol='000001.SZ',
    side='buy',
    order_type='market',
    quantity=100
)
```

### 模拟交易功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 市价单 | ✅ | 立即成交 |
| 限价单 | ✅ | 条件成交 |
| 止损单 | ✅ | 止损触发 |
| 资金管理 | ✅ | 自动扣费 |
| 持仓管理 | ✅ | 成本价计算 |
| 硬断路器 | ✅ | 风险控制 |

---

## 🔧 掘金量化(gm)对接

### 1. 注册账号

访问 https://www.myquant.cn/register 注册

### 2. 获取API Token

登录后在个人中心获取Token

### 3. 配置环境变量

```bash
export GM_TOKEN="your_token_here"
```

或在 `.env` 文件中添加：
```
GM_TOKEN=your_token_here
```

### 4. 修改配置

编辑 `skills/execution-monitor-engine/config.py`：

```python
class Config(BaseModel):
    EXECUTION_BACKEND: str = "gm"  # 改为gm
    PAPER_TRADE: bool = False       # 关闭模拟模式
```

### 5. 测试连接

```python
from engine import ExecutionEngine
from config import get_config

config = get_config()
engine = ExecutionEngine(config)

# 获取真实账户信息
account = engine.get_account()
print(account)
```

---

## 🔧 迅投QMT(xtquant)对接

### 1. 安装QMT

1. 联系券商获取QMT客户端
2. 安装到本地（仅支持Windows）
3. 登录QMT账号

### 2. 配置Python环境

QMT使用自己的Python环境，需要安装xtquant包：

```bash
# 在QMT的Python环境中安装
pip install xtquant
```

### 3. 配置环境变量

```bash
export XTQUANT_PATH="C:/QMT"  # QMT安装路径
```

### 4. 修改配置

```python
class Config(BaseModel):
    EXECUTION_BACKEND: str = "xtquant"
    PAPER_TRADE: bool = False
```

### 5. 测试连接

```python
from engine import ExecutionEngine
from config import get_config

config = get_config()
engine = ExecutionEngine(config)

# 获取真实持仓
positions = engine.get_positions()
print(positions)
```

---

## 🛡️ 风控配置

### 硬断路器规则

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_DAILY_LOSS_RATE` | 2% | 单日亏损超过2%拒新开仓 |
| `MAX_SINGLE_ORDER_AMOUNT` | 100,000 | 单笔最大金额 |
| `MAX_ORDER_FREQUENCY_PER_MINUTE` | 10 | 每分钟最大下单次数 |
| `MAX_POSITION_CONCENTRATION` | 30% | 单只股票最大仓位占比 |

### 修改风控参数

```python
config = Config(
    MAX_DAILY_LOSS_RATE=0.03,      # 3%
    MAX_SINGLE_ORDER_AMOUNT=50000,  # 5万
    PAPER_TRADE=False
)
engine = ExecutionEngine(config)
```

---

## 📝 订单类型说明

| 类型 | 代码 | 说明 |
|------|------|------|
| 市价单 | `market` | 以市场价格立即成交 |
| 限价单 | `limit` | 指定价格成交 |
| 止损单 | `stop` | 触发止损价格时成交 |

### 使用示例

```python
# 市价单
engine.place_order('000001.SZ', 'buy', 'market', 100)

# 限价单（10.00元买入）
engine.place_order('000001.SZ', 'buy', 'limit', 100, price=10.00)

# 止损单（跌破9.50元卖出）
engine.place_order('000001.SZ', 'sell', 'stop', 100, stop_price=9.50)
```

---

## ⚠️ 注意事项

1. **实盘风险**：实盘交易存在真实亏损风险，请充分测试后再使用
2. **T+1规则**：A股当日买入的股票次日才能卖出
3. **涨跌停**：涨停时无法买入，跌停时无法卖出
4. **资金充足**：确保账户有足够资金
5. **网络稳定**：实盘交易需要稳定的网络连接

---

## 🔍 故障排除

### 问题1：Token无效

```
Error: Invalid token
```

**解决**：检查GM_TOKEN是否正确，或重新获取

### 问题2：连接超时

```
Error: Connection timeout
```

**解决**：检查网络，或使用代理

### 问题3：资金不足

```
Order rejected: Insufficient funds
```

**解决**：确保账户有足够资金

### 问题4：QMT未启动

```
Error: QMT not running
```

**解决**：确保QMT客户端已启动并登录

---

## 📞 技术支持

如遇问题，请检查：
1. 环境变量配置是否正确
2. API Token是否有效
3. 网络连接是否正常
4. 账户资金是否充足