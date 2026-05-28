# 配置指南

本文档说明 a-share-factor-engine 的配置选项。

## 环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| FACTOR_BACKEND | 因子计算后端 | 否 | "talib" |
| FACTOR_DIR | 因子数据存储目录 | 否 | "./quant_workspace/factors" |
| IC_TYPE | IC计算方式 | 否 | "spearman" |

## 配置文件

配置文件位于 `scripts/config.py`。

### 因子计算配置

```python
FACTOR_BACKEND = os.getenv("FACTOR_BACKEND", "talib")
FACTOR_DIR = os.path.expanduser(os.getenv("FACTOR_DIR", "./quant_workspace/factors"))
IC_TYPE = os.getenv("IC_TYPE", "spearman")
```

### 中性化配置

```python
NEUTRALIZE_INDUSTRY = True
NEUTRALIZE_MARKET_CAP = True
```

### IC分析配置

```python
QUANTILES = 5
MIN_IC = 0.02
MIN_IC_IR = 0.3
MAX_CORRELATION = 0.8
```

## 后端选择

- `talib`: 使用 TA-Lib 计算技术指标（需要安装 ta-lib）
- `pandas_ta`: 使用 pandas-ta（纯 Python）

## 使用示例

```python
import os
os.environ['FACTOR_BACKEND'] = 'pandas_ta'
os.environ['FACTOR_DIR'] = './factors'

from engine import run, FactorEngine
result = run(ctx)
```
