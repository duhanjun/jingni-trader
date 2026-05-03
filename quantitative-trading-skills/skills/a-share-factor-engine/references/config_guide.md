# 配置指南

## 配置参数详解

### 数据路径配置

```python
config.FACTOR_DATA_DIR = "./factor_data"
```
- **描述**: 因子数据存储目录
- **默认值**: `./factor_data`
- **说明**: 保存和加载 Parquet 格式因子数据的目录

---

### 技术分析库选择

```python
config.TA_LIBRARY = "pandas-ta"  # 或 "talib"
```
- **描述**: 技术分析库选择
- **选项**:
  - `pandas-ta`: 使用 pandas-ta 库（推荐，无需安装 TA-Lib）
  - `talib`: 使用 TA-Lib 库（需要安装）
- **默认值**: `pandas-ta`

---

### IC 分析配置

```python
config.IC_LAG = 1
```
- **描述**: IC 计算的收益滞后阶数
- **默认值**: `1`
- **说明**: 用 t 期因子预测 t+1 期收益

```python
config.NEUTRALIZE_INDUSTRY = False
```
- **描述**: 是否进行行业中性化
- **默认值**: `False`
- **说明**: 若为 True，需要传入 industry_data

---

### 因子预处理配置

```python
config.WINSORIZE_THRESHOLD = 0.01
```
- **描述**: 缩尾处理的分位数阈值
- **默认值**: `0.01`
- **说明**: 上下各 1% 的极值被缩尾

```python
config.STANDARDIZE = True
```
- **描述**: 是否进行截面标准化（z-score）
- **默认值**: `True`
- **说明**: 每个交易日对因子做标准化

---

### 融合方法配置

```python
config.ENSEMBLE_METHOD = "ic_weighted"
```
- **描述**: 默认的因子融合方法
- **选项**:
  - `equal`: 等权融合
  - `ic_weighted`: IC 加权融合
  - `risk_weighted`: 风险平价融合
- **默认值**: `ic_weighted`

---

## 配置使用示例

### 方式一: 从环境变量加载

```bash
export FACTOR_DATA_DIR="/data/factors"
export TA_LIBRARY="pandas-ta"
export WINSORIZE_THRESHOLD=0.02
```

```python
from config import get_config
config = get_config()
```

### 方式二: 直接传入参数

```python
from config import get_config

config = get_config(
    FACTOR_DATA_DIR="/data/factors",
    TA_LIBRARY="pandas-ta",
    WINSORIZE_THRESHOLD=0.02,
    STANDARDIZE=True,
    ENSEMBLE_METHOD="ic_weighted"
)
```

### 方式三: 修改已有配置

```python
config = get_config()
config.WINSORIZE_THRESHOLD = 0.02
config.ENSEMBLE_METHOD = "equal"
```

---

## 推荐配置

### 稳健型配置

```python
config = get_config(
    WINSORIZE_THRESHOLD=0.02,
    STANDARDIZE=True,
    NEUTRALIZE_INDUSTRY=True,
    ENSEMBLE_METHOD="equal"
)
```

### 积极型配置

```python
config = get_config(
    WINSORIZE_THRESHOLD=0.01,
    STANDARDIZE=True,
    NEUTRALIZE_INDUSTRY=False,
    ENSEMBLE_METHOD="ic_weighted"
)
```

### 风险控制型配置

```python
config = get_config(
    WINSORIZE_THRESHOLD=0.02,
    STANDARDIZE=True,
    NEUTRALIZE_INDUSTRY=True,
    ENSEMBLE_METHOD="risk_weighted"
)
```
