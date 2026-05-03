# reports-engine - A股量化策略报告引擎

## 功能特性

- **QuantStats 一键报告**: 自动生成完整HTML报告
- **滚动夏普与月度热力图**: 风险收益时变分析
- **风格暴露分析**: 大盘/小盘/成长/价值暴露
- **行业暴露分析**: 申万一级行业权重分析
- **Brinson归因**: 超额收益分解（配置/选择/交互）
- **Plotly动态净值曲线**: 交互式可视化
- **因子衰减分析**: 因子有效性半衰期
- **过拟合风险警示**: 样本内外夏普比率检查

## 快速开始

```python
import pandas as pd
from engine import ReportsEngine

# 创建引擎
engine = ReportsEngine()

# 生成报告
report = engine.generate_full_report(
    portfolio_returns=portfolio_returns,
    benchmark_returns=benchmark_returns,
    factor_exposures=factor_exposures,
    industry_weights=industry_weights
)
```
