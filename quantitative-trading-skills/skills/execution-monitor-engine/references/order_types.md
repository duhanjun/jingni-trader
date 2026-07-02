# 订单类型说明

## 订单类型

### 1. 市价单 (MARKET)
- 以当前市场最优价格立即成交的订单
- 特点：成交速度快，但成交价格不确定
- 适用场景：需要立即成交，对价格不敏感

```python
order = engine.place_order(
    symbol="000001.SZ",
    side="buy",
    order_type="market",
    quantity=100
)
```

### 2. 限价单 (LIMIT)
- 以指定价格或更优价格成交的订单
- 特点：成交价格确定，但可能无法立即成交
- 适用场景：对价格有明确要求

```python
order = engine.place_order(
    symbol="000001.SZ",
    side="buy",
    order_type="limit",
    quantity=100,
    price=10.50
)
```

### 3. 止损单 (STOP)
- 当价格达到指定止损价时，转为市价单成交
- 特点：用于控制风险，锁定亏损或保护利润
- 适用场景：止损止盈

```python
order = engine.place_order(
    symbol="000001.SZ",
    side="sell",
    order_type="stop",
    quantity=100,
    stop_price=9.50
)
```

## 买卖方向

- `buy`: 买入
- `sell`: 卖出

## 订单状态

- `pending`: 待处理
- `submitted`: 已提交
- `partial_filled`: 部分成交
- `filled`: 完全成交
- `canceled`: 已撤销
- `rejected`: 已拒绝
- `failed`: 失败
