# A股交易规则

本文档详细说明了 backtest-engine 中模拟的 A股交易规则。

## T+1 交易规则

A股实行 T+1 交易制度：
- 当日买入的股票，当日不能卖出
- 需在下一个交易日（T+1）才能卖出

**配置:**
```python
config = get_config(ENABLE_T1=True)  # 启用 T+1
config = get_config(ENABLE_T1=False)  # 禁用 T+1（T+0 模式）
```

## 涨跌停限制

### 涨跌停幅度

| 股票类型 | 涨跌幅 |
|----------|--------|
| 普通股 | ±10% |
| ST股 | ±5% |

### 交易模型

#### 严格模型 ("strict")

当委托价格超出涨跌停范围时，订单直接被拒绝（废单）。

#### 排队模型 ("queue")

当委托价格超出涨跌停范围时，订单以涨跌停价格进入排队队列。

**配置:**
```python
config = get_config(LIMIT_UP_DOWN_MODEL="strict")  # 严格模型
config = get_config(LIMIT_UP_DOWN_MODEL="queue")  # 排队模型
```

## 费用模型

### 费用构成

| 费用类型 | 费率 | 收取方 | 说明 |
|----------|------|--------|------|
| 佣金 | 万分之2.5 | 券商 | 双向收取，最低5元 |
| 印花税 | 千分之一 | 国家 | 仅卖出时收取 |
| 过户费 | 十万分之二 | 交易所 | 双向收取 |

### 费用计算示例

假设买入 1000 股，每股 10 元：
- 成交金额：10,000 元
- 佣金：max(10,000 × 0.00025, 5) = 5 元
- 印花税：0 元（买入时不收取）
- 过户费：10,000 × 0.00002 = 0.2 元
- 总费用：5.2 元

假设卖出 1000 股，每股 11 元：
- 成交金额：11,000 元
- 佣金：max(11,000 × 0.00025, 5) = 5 元
- 印花税：11,000 × 0.001 = 11 元
- 过户费：11,000 × 0.00002 = 0.22 元
- 总费用：16.22 元

**配置:**
```python
config = get_config(
    COMMISSION_RATE=0.00025,  # 佣金率
    COMMISSION_MIN=5.0,       # 最低佣金
    STAMP_DUTY_RATE=0.001,    # 印花税率
    TRANSFER_FEE_RATE=0.00002 # 过户费率
)
```

## 停牌处理

当股票处于停牌状态时：
- 无法买入或卖出该股票
- 持仓被冻结
- 复牌后恢复交易

**使用方式:**
```python
suspension_handler = SuspensionHandler()

# 设置停牌日期
suspension_handler.set_suspension_dates("000001.SZ", [date1, date2])

# 检查是否可以交易
can_trade = suspension_handler.can_trade("000001.SZ", date)
```
