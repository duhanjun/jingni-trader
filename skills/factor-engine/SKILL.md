---
name: factor-engine
version: 1.0.0
description: A股阿尔法因子研究与构建引擎。支持定义和计算A股专属的Alpha因子（动量反转、市值、换手率、资金流、事件驱动等），提供因子IC分析（含行业中性化处理）、分层回测、相关性去冗余、多因子融合等功能。底层技术指标计算支持TA-Lib和pandas_ta双后端切换（默认pandas_ta）。支持表达式引擎声明式因子定义、Alpha158扩展因子库、提前期偏差检测、IC衰减分析。可选接入惊泥因子库(jingni-datafeed)获取已沉淀因子数据。
author: quant-team
license: MIT
tags:
  - quant-trading
  - A股
  - factor-engine
  - alpha-factor
  - talib
  - pandas-ta
  - jingni-datafeed
dependencies:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - scikit-learn>=1.3.0
  - statsmodels>=0.14.0
  - alphalens>=0.4.0
  - ta-lib (可选，C 语言依赖)
  - pandas-ta (可选，纯 Python 替代)
environment_variables:
  - name: FACTOR_BACKEND
    description: 因子计算后端（pandas_ta / talib），默认 pandas_ta（纯 Python）
    required: false
    default: "pandas_ta"
  - name: QUANT_WORK_DIR
    description: 工作目录根路径
    required: false
    default: "./workspace"
  - name: IC_TYPE
    description: IC计算方式（normal / spearman / rank）
    required: false
    default: "normal"
  - name: JINGNI_URL
    description: 惊泥因子库服务地址（启用 jingni-datafeed 时需要）
    required: false
  - name: JINGNI_TOKEN
    description: 惊泥因子库 API Token（启用 jingni-datafeed 时需要）
    required: false
language: python
python_version: "3.9+"
entry_point: engine.py
trigger_keywords:
  - 因子
  - 因子研究
  - Alpha因子
  - IC分析
  - 因子挖掘
  - 因子中性化
  - 因子相关性
  - 多因子
  - 表达式引擎
  - 因子DSL
  - 提前期偏差
  - 惊泥因子库
---

# factor-engine

## 概述

factor-engine 是 A 股量化投研的**因子研究与构建引擎**，提供：

1. **A股专用Alpha因子**：动量反转、市值、换手率、资金流、波动率等
2. **行业中性化处理**：市值+行业中性回归
3. **因子IC分析**：Spearman Rank IC / Pearson IC / 向量化IC分析
4. **因子相关性分析**：去冗余处理
5. **多因子融合**：等权/IC加权融合（含 NaN 隔离）
6. **表达式引擎**：声明式因子定义（如 `RANK(DELTA($close, 5))`）
7. **Alpha158 扩展因子库**：47 个因子，覆盖 6 大类
8. **提前期偏差检测**：自动检测因子计算中的未来数据泄露
9. **惊泥因子库集成**：可选从 jingni-datafeed 获取已沉淀因子数据

## 技术指标后端

| 后端 | 实现 | 特点 |
|------|------|------|
| `pandas_ta`（默认） | pandas_ta_calculator.py | 纯 Python，无需 C 编译，150+ 指标完整对齐 |
| `talib` | talib_calculator.py | 需 C 依赖（TA-Lib），性能更优 |

通过 `FACTOR_BACKEND` 环境变量切换，代码使用 try/except 自动 fallback。

## 因子定义体系

### 内置因子

- **动量因子**：`momentum_20d`、`momentum_60d`、`reversal_5d`、`reversal_20d`
- **规模因子**：`lncap`（对数市值）、`estimated_mv`
- **交易因子**：`turnover_20d`、`turnover_5d`、`turnover_change`、`volume_ratio`
- **波动率因子**：`volatility_20d`
- **资金流因子**：`money_flow_20d`（20日累计资金流）

### 表达式引擎因子

支持声明式定义，预设因子包括 KDJ、RSI、MACD、布林带等，例如：
```python
engine.compute_expression_factors(data, {
    "rsi_14": "RSI($close, 14)",
    "macd_diff": "MACD($close, 12, 26, 9)",
})
```

### Alpha158 扩展因子库

借鉴 Qlib Alpha158，合计 47 个因子，覆盖：
- 动量/反转类（12 个）
- 波动率类（8 个）
- 成交量/换手率类（8 个）
- 技术指标类（10 个）
- 资金流向类（5 个）
- 其他（4 个）

## 惊泥因子库集成

当 `JINGNI_URL` 和 `JINGNI_TOKEN` 均已配置，且用户明确要求从因子库取数时（如"从惊泥因子库加载因子"），factor-engine 优先从 jingni-datafeed 获取已沉淀因子数据，跳过本地计算。

- `ctx.metadata["factor_source"]` 控制：
  - `"jingni"`：强制走 jingni-datafeed 路径
  - `"auto"`：先尝试 jingni，失败回退本地计算
  - `"local"`：始终本地计算（默认）

## 使用示例

### Python API

```python
from engine import run
from context import Context

ctx = Context(
    task_id="task_001",
    user_intent="计算因子",
    current_stage="IDLE"
)
ctx.stock_pool = ["000001.SZ"]
ctx.start_date = "2021-01-01"
ctx.end_date = "2024-01-01"

result = run(ctx)

# 使用表达式引擎
from engine import FactorEngine
engine = FactorEngine()
expr_factors = engine.compute_expression_factors(data)

# 使用 Alpha158 扩展因子库
extended = engine.compute_extended_factors(data)
```

### CLI 运行

```bash
python engine.py -i "计算反转因子"
```

## 优化模块

可通过 `from engine import optimizations` 访问以下优化模块：

- **因子DSL**：`FactorEngine`、`AlphaEngine`、`AlphaRegistry`
- **表达式DSL**：`FactorDSLEvaluator`、`AstNode`、`FieldNode`
- **IC分析**：`ic_analysis_batch`、`ic_summary`、`ICDecayAnalyzer`、`calc_ic_series`、`calc_ic_stats`
- **因子验证**：`validate_factor`、`FactorVerdict`、`lookahead_detector`
- **因子注册**：`FactorRegistryV2`、`NeutralizerV2`、`neutralize_factor`、`neutralize_factors_batch`

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)