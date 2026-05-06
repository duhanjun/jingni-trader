# Qlib研究引擎

基于微软Qlib的AI量化因子研究引擎。

## 功能特性

- 因子库：Alpha158/Alpha360
- 模型训练：LightGBM、XGBoost
- 因子分析：IC分析、SHAP解释性分析
- 与Backtrader无缝对接

## 快速开始

```python
from engine import QlibResearchEngine
from config import get_config

config = get_config()
engine = QlibResearchEngine(config)

# 初始化Qlib
engine.init_qlib()

# 获取Alpha360因子处理器
handler = engine.get_alpha360_handler(
    start_time="2020-01-01",
    end_time="2024-12-31"
)

# 训练模型
model = engine.train_model(handler)

# 生成信号
predictions = engine.generate_signals(model, handler)
```

## 配置

环境变量：
- `QLIB_DATA_DIR`: Qlib数据目录（默认: ./qlib_data）
- `QLIB_CACHE_DIR`: Qlib缓存目录（默认: ./qlib_cache）