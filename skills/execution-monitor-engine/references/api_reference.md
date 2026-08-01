# API 参考文档

本文档提供 execution-monitor-engine 的完整 API 参考。

## 核心类

### Account

虚拟账户类（paper 模式）。

#### 方法

##### `reset_daily()`

重置每日状态，T+1 持仓解冻（available_volume 恢复为 volume）。

##### `get_current_nav(prices) -> float`

获取当前净值。

##### `apply_buy(code, price, volume, commission)`

执行买入，T+1 约束（买入当日 available_volume 不变）。

##### `apply_sell(code, price, volume, commission, stamp_tax)`

执行卖出，校验可用持仓。

##### `calc_commission(amount, is_sell) -> float`

计算手续费（佣金 + 印花税）。

### CircuitBreaker

硬风控断路器。

#### 方法

##### `check_send_order(account, code, order_value, prices) -> Dict`

检查是否允许发单（单日亏损/单笔金额/频率限制）。

**返回：**
```python
{"allowed": bool, "reason": str}
```

### PaperExecutor

模拟交易执行器。继承 BaseExecutor，内置滑点模拟、T+1 约束、数量校验、断路器风控、审计日志、状态持久化。

#### 方法

##### `query_account() -> Dict`

查询账户状态。

##### `send_order(code, side, volume, price, order_type) -> Dict`

发送订单。数量需为 100 的整数倍且 >= 100，限价单必须指定价格。

**返回：**
```python
{"success": True, "order_id": str, "status": "filled", "fill_price": float}
# 或
{"success": False, "error": str}
```

##### `cancel_order(order_id) -> Dict`

撤单。

##### `query_positions() -> pd.DataFrame`

查询持仓。

##### `sync_positions(target_weights, prices) -> List[Dict]`

同步目标仓位，生成调仓订单列表。

##### `save_state() / load_state() -> bool`

账户状态持久化（JSON 格式）。

### XtQuantExecutor

迅投 miniQMT 实盘交易执行器。继承 BaseExecutor，通过 xtdata + XtQuantTrader 连接本地 miniQMT 客户端。

#### 方法

##### `connect(path="", session_id=0) -> bool`

连接 miniQMT（xtdata.connect + XtQuantTrader.connect + subscribe）。失败返回 False，不抛异常。

##### `available -> bool`

执行器是否可用（已连接）。

##### `query_account() -> Dict`

查询账户资产。返回 `{total_assets, available_cash, market_value, frozen_cash, account_id}`。

##### `send_order(code, side, volume, price, order_type) -> Dict`

发送订单（限价/市价）。未连接返回 `{"success": False, "error": "miniQMT 未连接"}`。

##### `cancel_order(order_id) -> Dict`

撤单。

##### `query_positions() -> pd.DataFrame`

查询当前持仓。

##### `sync_positions(target_weights, prices) -> List[Dict]`

同步目标仓位，生成调仓订单列表。

### GMExecutor

掘金量化实盘交易执行器。继承 BaseExecutor，通过 set_token + set_account_id 连接掘金终端。

#### 方法

##### `_try_connect() -> bool`

连接掘金终端（set_token + set_account_id）。失败返回 False，不抛异常。

##### `available -> bool`

执行器是否可用（已连接）。

##### `query_account() -> Dict`

查询账户资产（get_cash）。返回 `{total_assets, available_cash, market_value, frozen_cash, account_id}`。

##### `send_order(code, side, volume, price, order_type) -> Dict`

发送订单（order_volume）。代码格式自动转换（600000.SH -> SHSE.600000）。

##### `cancel_order(order_id) -> Dict`

撤单（order_cancel）。

##### `query_positions() -> pd.DataFrame`

查询当前持仓（get_position）。

##### `sync_positions(target_weights, prices) -> List[Dict]`

同步目标仓位，生成调仓订单列表。

## 标准入口函数

### `run(ctx) -> Dict[str, Any]`

Skill 标准入口函数。

**参数：**
- `ctx` (Context): 上下文对象，需包含 `artifacts['PORTFOLIO']`（目标权重 JSON 路径）

**返回：**
```python
{
    "success": bool,
    "artifact_path": str,      # 审计日志路径
    "metadata": {
        "orders_executed": int,
        "orders_failed": int,
        "account_snapshot": {...},
        "mode": str,           # "paper" / "live"
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
