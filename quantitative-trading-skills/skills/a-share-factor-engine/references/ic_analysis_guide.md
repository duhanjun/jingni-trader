# IC 分析指南

## 什么是 IC

IC (Information Coefficient，信息系数) 衡量因子预测能力的指标，通常计算为：
- 因子值与未来收益率的 Spearman 秩相关系数
- Pearson 相关系数（可选）

## IC 的解读

### IC 均值
- **> 0.05**: 强预测能力
- **0.02 - 0.05**: 中等预测能力
- **0.01 - 0.02**: 弱预测能力
- **< 0.01**: 预测能力较弱或不稳定

### ICIR (IC 均值 / IC 标准差)
- **> 0.5**: 稳定的好因子
- **0.3 - 0.5**: 较稳定的因子
- **< 0.3**: 不稳定的因子

### IC 胜率
- IC 为正的日期占比
- **> 60%**: 胜率较高

## 行业中性化

### 为什么需要行业中性化
1. 某些行业长期表现好，因子可能只是暴露了行业风险
2. 降低策略在行业上的偏离
3. 使因子的选股能力在行业内更纯净

### 如何做行业中性化
1. 在每个截面内，按行业分组
2. 在每个行业内对因子进行去均值（或 z-score 标准化）
3. 合并各行业的结果

## IC 分析报告解读

```python
ic_report = engine.analyze_ic(factors, forward_returns, industry_data)
```

报告包含：
- `ic_series`: IC 时间序列
- `ic_mean`: 各因子 IC 均值
- `ic_std`: 各因子 IC 标准差
- `icir`: 各因子 ICIR

## 分析示例

```python
import matplotlib.pyplot as plt

# 绘制 IC 时间序列
ic_series = ic_report["ic_series"]
ic_series.set_index("date").plot(figsize=(12, 6))
plt.title("IC Time Series")
plt.ylabel("IC")
plt.show()

# 绘制 IC 均值柱状图
ic_mean = pd.Series(ic_report["ic_mean"]).sort_values()
ic_mean.plot(kind="barh", figsize=(10, 6))
plt.title("IC Mean by Factor")
plt.xlabel("IC Mean")
plt.show()
```

## 注意事项

1. **前瞻性偏差**: 确保使用真实可获得的数据计算因子
2. **过拟合**: IC 高但样本外表现差，可能是过拟合
3. **换手率**: 高 IC 但换手率过高可能导致交易成本侵蚀收益
4. **多空表现**: 分析因子在多头和空头端的表现差异
