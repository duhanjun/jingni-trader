# 策略模板说明

## 概述

strategy-model-engine 提供了三种常用的量化策略模板，涵盖了趋势跟踪、均值回归和配对交易等主流策略类型。

## 策略类型

### 1. 趋势跟踪策略 (trend_following)

**核心思想**：追随市场趋势，在趋势开始时入场，在趋势结束时出场。

**包含技术指标**：
- 双均线系统（短期MA、长期MA）
- 布林带（上轨、中轨、下轨）
- 唐奇安通道（最高价、最低价）

**交易信号**：
- 买入：短期MA上穿长期MA，或价格突破布林带上轨/唐奇安通道上沿
- 卖出：短期MA下穿长期MA，或价格跌破布林带下轨/唐奇安通道下沿

**默认参数**：
```python
{
    "ma_fast_period": 5,
    "ma_slow_period": 20,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "donchian_period": 20,
    "stop_loss": 0.02,
    "take_profit": 0.05,
    "position_size": 0.1
}
```

**使用场景**：
- 市场处于明显的上升或下降趋势
- 波动率适中的市场环境
- 适合中长期持仓

---

### 2. 均值回归策略 (mean_reversion)

**核心思想**：价格围绕均值波动，当价格偏离均值过大时会回归。

**包含技术指标**：
- RSI（相对强弱指标）
- 布林带

**交易信号**：
- 买入：RSI < 30（超卖），或价格跌破布林带下轨
- 卖出：RSI > 70（超买），或价格突破布林带上轨

**默认参数**：
```python
{
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "stop_loss": 0.02,
    "take_profit": 0.04,
    "position_size": 0.1
}
```

**使用场景**：
- 震荡市场（无明显趋势）
- 价格波动有规律地围绕均值
- 适合短期持仓

---

### 3. 配对交易策略 (pair_trading)

**核心思想**：寻找两只价格相关性高的股票，当它们的价差偏离历史均值时进行交易。

**包含技术指标**：
- 价差（价格比或价格差）
- 价差的移动平均线
- Z-score（标准化价差）

**交易信号**：
- 买入价差：Z-score < -2（价差过小，预期扩大）
- 卖出价差：Z-score > 2（价差过大，预期收窄）
- 平仓：|Z-score| < 0.5

**默认参数**：
```python
{
    "spread_method": "price_ratio",
    "spread_ma_period": 20,
    "spread_std_period": 20,
    "z_score_period": 20,
    "z_score_threshold": 2,
    "z_score_close": 0.5,
    "cointegration_pvalue": 0.05,
    "stop_loss": 0.03,
    "take_profit": 0.05,
    "position_ratio": 1.0
}
```

**使用场景**：
- 两只股票有稳定的历史相关性
- 属于同一行业或板块
- 适合市场中性策略

---

## 使用示例

### 生成策略模板

```python
from engine import StrategyModelEngine

engine = StrategyModelEngine()

# 生成趋势跟踪策略
trend_template = engine.generate_strategy_template("trend_following", ma_fast_period=10)

# 生成均值回归策略
mean_template = engine.generate_strategy_template("mean_reversion", rsi_period=7)

# 生成配对交易策略
pair_template = engine.generate_strategy_template("pair_trading", z_score_threshold=1.5)
```

### 策略模板结构

每个策略模板包含以下部分：
```python
{
    "name": "策略名称",
    "description": "策略描述",
    "indicators": [  # 技术指标列表
        {
            "name": "指标名称",
            "type": "指标类型",
            ...  # 其他参数
        }
    ],
    "signals": [  # 交易信号列表
        {
            "name": "信号名称",
            "condition": "触发条件",
            "action": "操作类型"
        }
    ],
    "parameters": {  # 策略参数
        ...
    },
    "risk_management": {  # 风险管理参数
        "stop_loss": 0.02,
        "take_profit": 0.05,
        ...
    }
}
```
