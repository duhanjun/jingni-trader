# API 参考文档

本文档提供 execution-monitor-engine 的完整 API 参考。

## 核心类

### Account

虚拟账户类。

#### 方法

##### `reset_daily()`

重置每日状态。

##### `get_current_nav(prices) -> float`

获取当前净值。

##### `apply_buy(code, price, volume, commission)`

执行买入。

##### `apply_sell(code, price, volume, commission, stamp_tax)`

执行卖出。

##### `calc_commission(amount, is_sell) -> float`

计算手续费。

### CircuitBreaker

硬风控断路器。

#### 方法

##### `check_send_order(account, code, order_value, prices) -> Dict`

检查是否允许发单。

**返回：**
```python
{"allowed": bool, "reason": str}
```

### PaperExecutor

模拟交易执行器。

#### 方法

##### `query_account() -> Dict`

查询账户状态。

##### `send_order(code, side, volume, price, order_type) -> Dict`

发送订单。

##### `cancel_order(order_id) -> Dict`

撤单。

##### `query_positions() -> pd.DataFrame`

查询持仓。

##### `sync_positions(target_weights, prices) -> List[Dict]`

同步目标仓位。

## 标准入口函数

### `run(ctx) -> Dict[str, Any]`

Skill 标准入口函数。

**参数：**
- `ctx` (Context): 上下文对象

**返回：**
```python
{
    "success": bool,
    "artifact_path": str,      # 审计日志路径
    "metadata": {
        "orders_executed": int,
        "orders_failed": int,
        "account_snapshot": {...},
        "mode": str,
    },
    "error": str
}
```

**示例：**

```python
from engine import run

result = run(ctx)
if result['success']:
    print(f"执行成功: {result['metadata']['orders_executed']} 笔订单")
```

## CLI 使用

```bash
python engine.py context.json
```
