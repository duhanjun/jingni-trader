# 配置指南

## 环境变量

- `REPORT_DIR`: 报告输出目录，默认 `./reports`
- `BENCHMARK_CODE`: 基准指数代码，默认 `000300.SH`

## 配置类参数

```python
from config import get_config

config = get_config(
    REPORT_DIR="./my_reports",
    BENCHMARK_CODE="000905.SH",
    RISK_FREE_RATE=0.03,
    ROLLING_WINDOW=252,
    DECAY_PERIODS=20,
    OVERFITTING_THRESHOLD=0.6,
    PLOT_STYLE="seaborn-v0_8",
    FIGURE_SIZE=(14, 10)
)
```
