# 多因子融合方法

## 融合方法概述

### 1. 等权融合 (Equal Weight)

**原理**: 所有因子赋予相同权重

**优点**:
- 简单直观
- 不容易过拟合
- 对单因子表现不敏感

**缺点**:
- 没有利用因子的预测能力差异
- 没有考虑因子间的相关性

**适用场景**:
- 因子表现都比较好且相近
- 因子之间相关性较低

---

### 2. IC 加权融合 (IC Weighted)

**原理**: 根据因子的历史 IC 均值加权，IC 越高权重越大

**权重计算公式**:
```
weight_i = max(IC_i, 0) / sum(max(IC_j, 0) for all j)
```

**优点**:
- 利用了因子的历史表现
- 表现好的因子获得更高权重

**缺点**:
- 可能过拟合历史数据
- 没有考虑因子的稳定性
- 没有考虑因子间的相关性

**适用场景**:
- 因子的 IC 有显著差异
- 因子表现较为稳定

---

### 3. 风险平价融合 (Risk Weighted)

**原理**: 根据因子的波动率倒数加权，波动率越低权重越大

**权重计算公式**:
```
weight_i = (1 / volatility_i) / sum(1 / volatility_j for all j)
```

**优点**:
- 降低风险较高因子的权重
- 因子风险贡献相对均衡

**缺点**:
- 没有直接考虑因子的预测能力
- 假设因子收益率服从正态分布

**适用场景**:
- 风险控制优先
- 因子波动率差异较大

---

## 融合最佳实践

### 1. 因子筛选

在融合前，建议先筛选因子：
- IC 均值 > 0.01
- ICIR > 0.2
- 相关性过高的因子去冗余（如 |corr| > 0.7）

### 2. 滚动窗口训练

避免使用整个历史期的 IC，使用滚动窗口：
```
rolling_window = 60  # 60个交易日
```

### 3. 组合多种融合方法

可以结合多种融合方法的结果：
```
combined = 0.5 * equal_weight + 0.5 * ic_weighted
```

### 4. 动态权重调整

根据市场环境动态调整权重：
- 牛市：动量因子权重高
- 熊市：质量、低波因子权重高

---

## 使用示例

```python
from engine import AShareFactorEngine
from config import get_config

config = get_config()
engine = AShareFactorEngine(config)

# 计算因子
factors = engine.compute_factors(price_data)

# IC 分析
ic_report = engine.analyze_ic(factors, forward_returns)

# 等权融合
equal_combined = engine.combine_factors(factors, method="equal")

# IC 加权融合
ic_combined = engine.combine_factors(
    factors, 
    method="ic_weighted",
    ic_mean=ic_report["ic_mean"]
)

# 风险平价融合
risk_combined = engine.combine_factors(factors, method="risk_weighted")
```

---

## 融合效果评估

评估融合后的因子表现：
1. **IC 分析**: 融合后因子的 IC、ICIR
2. **回测**: 基于融合因子的回测表现
3. **换手率**: 融合因子的换手率
4. **稳健性**: 在不同市场环境下的表现
