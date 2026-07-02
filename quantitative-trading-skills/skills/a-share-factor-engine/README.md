# A股多因子选股引擎 (a-share-factor-engine)

## 概述

这是一个功能完整的 A 股多因子选股引擎 Skill，提供从因子计算、IC 分析、相关性分析、因子融合到存储复用的全流程支持。

## 功能特性

1. **多因子库**
   - 技术因子：动量、反转、波动率、RSI、MACD、布林带、KDJ、威廉指标
   - A 股专用因子：对数市值、换手率
   - 可扩展框架，易于添加新因子

2. **IC 分析**
   - IC 时间序列计算
   - ICIR 分析
   - 行业中性化处理

3. **相关性分析**
   - 相关性矩阵
   - 冗余因子识别
   - 因子精简建议

4. **多 Alpha 融合**
   - 等权融合
   - IC 加权融合
   - 风险平价融合

5. **因子存储**
   - Parquet 格式高效存储
   - 支持增量更新

6. **对抗性验证**
   - 训练/测试分布一致性检查
   - KS 检验、t 检验
   - 因子稳定性分析

## 快速开始

### 安装依赖

```bash
pip install pandas numpy pandas-ta pyarrow fastparquet scipy scikit-learn
```

### 基本使用

```python
from engine import AShareFactorEngine
from config import get_config

# 创建引擎
config = get_config()
engine = AShareFactorEngine(config)

# 计算因子
factors = engine.compute_factors(
    price_data,
    factor_list=["momentum_1m", "lncap", "turnover"]
)

# IC 分析
ic_report = engine.analyze_ic(factors, forward_returns)

# 相关性分析
corr_report = engine.analyze_correlation(factors, ic_report["ic_mean"])

# 因子融合
combined = engine.combine_factors(factors, method="ic_weighted", ic_mean=ic_report["ic_mean"])

# 保存因子
engine.save_factors(factors, "my_factors.parquet")

# 完整工作流
result = engine.full_workflow(
    price_data,
    forward_returns,
    save_filename="factors.parquet"
)
```

## 目录结构

```
a-share-factor-engine/
├── SKILL.md              # Skill 定义文件
├── README.md             # 说明文档
├── requirements.txt      # 依赖列表
├── config.py             # 配置模块
├── engine.py             # 主引擎
├── factors/              # 因子计算模块
│   ├── __init__.py
│   └── factor_builder.py
├── ic_analysis/          # IC 分析模块
│   ├── __init__.py
│   └── ic_analyzer.py
├── correlation/          # 相关性分析模块
│   ├── __init__.py
│   └── correlation_analyzer.py
├── ensemble/             # 因子融合模块
│   ├── __init__.py
│   └── factor_combiner.py
├── storage/              # 存储模块
│   ├── __init__.py
│   └── factor_storage.py
├── validation/           # 验证模块
│   ├── __init__.py
│   └── adversarial_validator.py
├── references/           # 参考文档
│   ├── factor_description.md
│   ├── ic_analysis_guide.md
│   ├── ensemble_methods.md
│   └── config_guide.md
└── tests/                # 测试
    ├── __init__.py
    └── test_engine.py
```

## 文档

- [因子说明](./references/factor_description.md)
- [IC 分析指南](./references/ic_analysis_guide.md)
- [多因子融合方法](./references/ensemble_methods.md)
- [配置指南](./references/config_guide.md)

## 测试

```bash
cd tests
python test_engine.py
```

## 许可证

MIT License
