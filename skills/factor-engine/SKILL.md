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
  - alphalens-reloaded>=0.4.5  # 维护版本，兼容 pandas 2.0（替代已停止维护的 alphalens）
  - polars>=0.20.0  # 可选，高性能 DataFrame 后端（IC/中性化等热路径加速 5-15×）
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
  - name: QUANT_ALPHALENS_REPORT
    description: 是否生成 alphalens 完整因子分析报告（0/1），默认 0；启用后每个因子输出 4 PNG + 1 HTML + 1 JSON
    required: false
    default: "0"
  - name: QUANT_FACTOR_BACKEND
    description: DataFrame 后端（pandas / polars / auto），作用于 IC 计算/中性化/IC Decay/相关性分析等热路径。默认 pandas，polars 缺失时自动回退
    required: false
    default: "pandas"
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

## 高性能 DataFrame 后端（可选）

IC 计算 / 中性化 / IC Decay / 相关性分析等热路径支持 **pandas / polars 双后端**，通过 `QUANT_FACTOR_BACKEND` 环境变量切换。

| 后端 | 适用场景 | 性能（5000 股 × 1000 日） |
|------|---------|------------------------|
| `pandas`（默认） | 兼容性最广，零额外依赖 | 基线 |
| `polars` | 大规模截面计算（5000+ 股票） | IC 提速 5-15×，中性化提速 3-5× |
| `auto` | 自动检测 polars 可用性 | polars 可用时自动启用 |

### 覆盖模块

| 模块 | 文件 | 提速来源 |
|------|------|---------|
| Pearson IC | `scripts/optimizations/ic_vectorized.py` | polars 多线程 Rust 引擎 |
| Spearman IC | `scripts/optimizations/ic_vectorized.py` | polars 窗口函数 + rank |
| 中性化 | `scripts/optimizations/vectorized_neutralize.py` | polars group_by 批量截面求解 |
| IC Decay | `scripts/optimizations/ic_decay.py` | polars `over(code/date)` 一次性扫描 |
| 相关性分析 | `scripts/optimizations/vectorized_correlation.py` | polars `DataFrame.corr()` |

### 使用方式

```bash
# 显式启用 polars
export QUANT_FACTOR_BACKEND=polars

# 自动检测（推荐生产环境）
export QUANT_FACTOR_BACKEND=auto

# 临时禁用（强制 pandas）
export QUANT_FACTOR_BACKEND=pandas
```

### 双后端一致性

所有支持 polars 后端的模块均通过 L2 单元测试验证：双后端输出最大绝对偏差 < 1e-10（IC Decay 因 rank 实现细节差异放宽到 1e-6）。polars 缺失时自动回退 pandas 并输出 warning 日志。

## Processor Pipeline 架构（方向一）

借鉴 Microsoft Qlib 的 Processor + Recorder 设计，将因子处理流程抽象为**可插拔的工序链 + 实验可重放记录器**。通过 `pipeline.yaml` 声明式配置工序组合。

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `Processor`（抽象基类） | `scripts/processors/base.py` | 声明 `requires` / `__call__` / `describe`；子类实现具体工序 |
| `ProcessContext` | `scripts/processors/base.py` | 工序间状态传递载体（IC 结果、selected_factors、forward_returns 等） |
| `ProcessorChain` | `scripts/processors/chain.py` | 调度器，按序执行 Processor + 依赖检查 + 异常隔离 |
| `ExperimentRecorder` | `scripts/recorder.py` | 实验可重放记录器，输出 manifest.json（含 7 字段） |
| `load_pipeline_config` | `scripts/processors/loader.py` | YAML 加载器（work_dir 优先 → skill 默认 → 兜底默认链） |

### 7 个内置 Processor

| Processor | 功能 | 默认启用 |
|-----------|------|---------|
| `NeutralizeProcessor` | 行业 + 市值中性化（截面回归取残差） | 否 |
| `WinsorizeProcessor` | 去极值（MAD 法 / 分位数法） | 否 |
| `FillnaProcessor` | 缺失值填充（rank_pct / zero / mean / ffill） | 否 |
| `StandardizeProcessor` | 标准化（zscore / minmax） | 否 |
| `ICAnalysisProcessor` | IC 分析（Pearson / Spearman），写入 ctx.ic_results | 是 |
| `CorrelationFilterProcessor` | 相关性去冗余，写入 ctx.selected_factors | 是 |
| `FusionProcessor` | 多因子融合（ic_weighted / equal_weighted），输出 alpha_score | 是 |

### 声明式 YAML 配置

默认配置文件：`scripts/processors/pipeline.yaml`。在 `QUANT_WORK_DIR` 下放置 `pipeline.yaml` 可覆盖默认配置。

```yaml
pipeline:
  - processor: NeutralizeProcessor
    enabled: false              # 默认禁用，保持与 v1.x run() 一致
    params:
      neutralize_mcap: true
      neutralize_industry: true
      min_count: 30

  - processor: ICAnalysisProcessor
    enabled: true
    params:
      ic_type: normal
      forward_periods: [1, 5, 20]
      min_count: 10

  - processor: CorrelationFilterProcessor
    enabled: true
    params:
      max_correlation: 0.7

  - processor: FusionProcessor
    enabled: true
    params:
      method: ic_weighted
      forward_period_for_weight: ret_forward_5d
```

### ExperimentRecorder manifest

每次 pipeline 运行自动在 `<QUANT_WORK_DIR>/archives/factor_engine/run_YYYYMMDD_HHMMSS/` 下生成 `manifest.json`，含 7 字段：

| 字段 | 说明 |
|------|------|
| `run_id` | uuid4().hex，唯一标识本次运行 |
| `start_time` / `end_time` | 运行起止时间（ISO 格式） |
| `pipeline_config` | 各 Processor 的 `describe()` 输出列表 |
| `input_data_hash` | 输入文件 sha256（接入 P1-3 artifact_store） |
| `steps` | 每步执行后状态（processor 名 / 参数 / 行列数 / nan_ratio） |
| `output_artifacts` | 输出产物路径列表 |
| `env` | 环境快照（Python/pandas/polars 版本 + QUANT_ 环境变量） |

manifest.json 自身 sha256 纳入 P1-3 sha256 Manifest 覆盖范围，支持实验可重放。

## 因子定义体系

### 内置因子

- **动量因子**：`momentum_20d`、`momentum_60d`、`reversal_5d`、`reversal_20d`
- **规模因子**：`lncap`（对数市值，`estimated_mv` 仅作为内部中间变量不输出）
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

## Alphalens 因子分析报告（可选）

设置环境变量 `QUANT_ALPHALENS_REPORT=1` 后，FACTOR 阶段会为每个入选因子自动生成完整的 alphalens 因子分析报告，输出到 `<QUANT_WORK_DIR>/reports/alphalens/<task_id>/`。

### 产物清单（每个因子）

| 文件 | 说明 |
|------|------|
| `<factor>_returns.png` | 分层累积收益曲线 |
| `<factor>_ic.png` | IC 时序图与分布 |
| `<factor>_turnover.png` | 分层换手率 |
| `<factor>_summary.png` | 综合统计摘要 |
| `<factor>_report.html` | 单因子 HTML 报告（含上述图表） |
| `<factor>_metrics.json` | 8 项关键指标（机器可读） |

### metrics.json 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `factor` | str | 因子名 |
| `top_quantile_return` | float | 最高分层平均收益 |
| `bottom_quantile_return` | float | 最低分层平均收益 |
| `long_short_return` | float | 多空收益（Top - Bot） |
| `long_short_sharpe` | float | 多空收益年化夏普 |
| `ic_mean` | float | IC 均值 |
| `ic_ir` | float | IC 信息比（IC 均值 / IC 标准差） |
| `avg_turnover_top_quantile` | float | Top 分层平均换手率 |
| `suggested_verdict` | str | 建议结论（ACCEPT / REVIEW / REJECT） |

### 降级方案

当 `alphalens-reloaded` 未安装或运行报错时，自动降级到方案 C（自研轻量分层回测）：
- 仅输出 `<factor>_metrics.json` + `<factor>_report.html`，不生成 PNG
- metrics.json 额外含 `_backend: "fallback_lite"` 标识
- 主流程不阻塞，仅记录 warning 日志

### 下游聚合

REPORT 阶段会自动聚合所有因子的 `metrics.json`，生成 `factor_analysis_report.html` 汇总报告（含各因子指标卡片与 ACCEPT/REVIEW/REJECT 结论），归档到 REPORT 阶段 artifacts。

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

以下优化模块位于 `scripts/optimizations/`，按需直接 import，例如 `from scripts.optimizations.ic_vectorized import ic_analysis_batch`：

- **Alpha158 库**：`alpha158_lib.AlphaEngine`、`alpha158_lib.AlphaRegistry`
- **IC分析**：`ic_vectorized.ic_analysis_batch`、`ic_vectorized.ic_summary`、`ic_decay.ICDecayAnalyzer`、`ic_analysis_v2.calc_ic_series`、`ic_analysis_v2.calc_ic_stats`
- **因子验证**：`factor_validator.validate_factor`、`factor_validator.FactorVerdict`、`lookahead_detector`
- **因子注册**：`factor_registry_v2.FactorRegistry`、`factor_registry_v2.Neutralizer`、`vectorized_neutralize.neutralize_factor`
- **相关性分析**：`vectorized_correlation.correlation_analysis`（支持 polars 后端）

## Processor Pipeline 模块

可通过 `from engine import processors` 访问所有 Processor 相关类与工具函数：

- **基类与异常**：`Processor`、`ProcessContext`、`ProcessorRequirementError`、`ChainValidationError`
- **调度器与加载器**：`ProcessorChain`、`load_pipeline_config`、`parse_yaml_to_processors`、`register_processor`、`PROCESSOR_REGISTRY`
- **7 个内置 Processor**：`NeutralizeProcessor`、`WinsorizeProcessor`、`FillnaProcessor`、`StandardizeProcessor`、`ICAnalysisProcessor`、`CorrelationFilterProcessor`、`FusionProcessor`
- **实验记录器**：`ExperimentRecorder`
- **工具方法**：`get_all_processors()`、`create_default_chain()`

## 配置说明

详见 [references/config_guide.md](references/config_guide.md)

## API 文档

详见 [references/api_reference.md](references/api_reference.md)