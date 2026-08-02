# API 参考文档

本文档提供 a-share-factor-engine 的完整 API 参考。

## 核心类

### FactorEngine

因子引擎类，负责因子计算和分析。

#### 方法

##### `__init__()`

初始化因子引擎，自动加载配置的因子计算器。

```python
engine = FactorEngine()
```

##### `compute_a_share_factors(data: pd.DataFrame) -> pd.DataFrame`

计算A股专用Alpha因子。

**参数：**
- `data` (pd.DataFrame): 清洗后的日线数据

**返回：**
- `pd.DataFrame`: 包含因子的DataFrame

**示例：**

```python
factors = engine.compute_a_share_factors(data)
```

##### `neutralize(factor_df, industry_df, neutralize_mcap, neutralize_industry) -> pd.DataFrame`

因子中性化处理。

##### `ic_analysis(factor_df, forward_returns, factor_names) -> Dict`

计算因子IC分析。

##### `correlation_analysis(factor_df, factor_names, max_correlation) -> Dict`

因子相关性分析。

##### `factor_fusion(factor_df, ic_results, selected_factors, fusion_method) -> pd.DataFrame`

多因子融合。

## 标准入口函数

### `run(ctx) -> Dict[str, Any]`

Skill 标准入口函数。

**参数：**
- `ctx` (Context): 上下文对象

**返回：**
```python
{
    "success": bool,
    "artifact_path": str,      # 因子数据文件路径
    "metadata": {
        "factor_names": [...],
        "selected_factors": [...],
        "ic_results": {...}
    },
    "error": str
}
```

**示例：**

```python
from engine import run

result = run(ctx)
if result['success']:
    print(f"因子数据已保存至: {result['artifact_path']}")
```

## 因子列表

| 因子名 | 描述 | 类型 |
|--------|------|------|
| reversal_5d | 5日反转 | 动量 |
| reversal_20d | 20日反转 | 动量 |
| lncap | 对数市值 | 规模 |
| turnover_20d | 20日换手率 | 交易 |
| turnover_change | 换手率变化 | 交易 |
| volatility_20d | 20日波动率 | 风险 |
| volume_ratio | 量比 | 量价 |
| money_flow_20d | 20日资金流 | 资金流 |

## CLI 使用

```bash
python engine.py context.json
```

## Alphalens 因子分析报告

### 模块

`scripts/alphalens_adapter.py`

### 模块级函数

#### `is_alphalens_enabled() -> bool`

检查环境变量 `QUANT_ALPHALENS_REPORT` 是否启用（返回 `True` 当值为 `"1"`）。

```python
from scripts.alphalens_adapter import is_alphalens_enabled
if is_alphalens_enabled():
    # 启用 alphalens 报告生成
```

#### `_alphalens_available() -> bool`

检查 `alphalens-reloaded` 是否可导入。内部函数，供降级判断使用。

### AlphalensAdapter 类

将 jingni-trader 内部数据结构（`code`/`date`/`factor` 列）适配为 alphalens 期望的 MultiIndex Series + price pivot，并生成完整的因子分析报告。

#### `to_alphalens_format(factor_df, price_df, factor_name, forward_periods=(1,5,20), quantiles=5, max_loss=0.25)`

将内部数据转换为 alphalens 标准格式（调用 `alphalens.utils.get_clean_factor_and_forward_returns`）。

**参数：**
- `factor_df` (pd.DataFrame): 内部因子数据，需含 `code`、`date`、`<factor_name>` 列
- `price_df` (pd.DataFrame): 价格数据，需含 `code`、`date`、`close` 列
- `factor_name` (str): 要分析的因子列名
- `forward_periods` (Tuple[int, ...]): 前瞻期，默认 `(1, 5, 20)`
- `quantiles` (int): 分层数，默认 5
- `max_loss` (float): 允许的最大数据丢失率（0-1），默认 0.25

**返回：** alphalens `factor_data`（MultiIndex DataFrame）

#### `generate_full_report(factor_data, output_dir, factor_name) -> Optional[Dict[str, str]]`

生成完整 alphalens 报告（4 PNG + 1 HTML + 1 JSON）。需 `alphalens-reloaded` 已安装。

**参数：**
- `factor_data`: `to_alphalens_format` 返回的 MultiIndex DataFrame
- `output_dir` (Path): 输出目录
- `factor_name` (str): 因子名（用作文件前缀）

**返回：** 生成的文件路径字典，含 `returns_png`、`ic_png`、`turnover_png`、`summary_png`、`html`、`metrics_json` 键；失败返回 `None`

#### `generate_for_factor(factor_df, price_df, factor_name, output_dir, forward_periods=(1,5,20), quantiles=5) -> Optional[Dict[str, str]]`

端到端单因子报告生成（推荐入口）。自动选择 alphalens 路径或降级到方案 C。

**参数：**
- `factor_df` (pd.DataFrame): 内部因子数据
- `price_df` (pd.DataFrame): 价格数据
- `factor_name` (str): 因子列名
- `output_dir` (Path): 输出目录
- `forward_periods` (Tuple[int, ...]): 前瞻期
- `quantiles` (int): 分层数

**返回：** 文件路径字典；两条路径都失败时返回 `None`

**示例：**

```python
from scripts.alphalens_adapter import AlphalensAdapter

result = AlphalensAdapter.generate_for_factor(
    factor_df=factor_df,
    price_df=price_df,
    factor_name="reversal_5d",
    output_dir="./workspace/reports/alphalens/task_001",
)
# result = {"metrics_json": "...", "html": "...", ...}
```

#### `_extract_metrics(factor_data, factor_name) -> Dict[str, Any]`

从 alphalens `factor_data` 提取 8 项关键指标。内部方法。

**返回字段：** `factor`、`top_quantile_return`、`bottom_quantile_return`、`long_short_return`、`long_short_sharpe`、`ic_mean`、`ic_ir`、`avg_turnover_top_quantile`、`suggested_verdict`

**verdict 阈值：**
- `ACCEPT`: `ic_ir >= 0.5` 且 `long_short_sharpe >= 0.8`
- `REVIEW`: 不满足 ACCEPT 条件

### 降级方案 C

当 `alphalens-reloaded` 不可用时，`generate_for_factor` 自动调用 `_fallback_layered_backtest`：

- 仅输出 `<factor>_metrics.json` + `<factor>_report.html`（无 PNG）
- metrics.json 额外含 `_backend: "fallback_lite"` 标识
- 使用 pandas `qcut` 按截面分层，自研计算 IC（Spearman）、多空收益、换手率
- std 容差判断 `> 1e-10` 避免浮点精度导致除零

### engine.py 集成函数

#### `_maybe_generate_alphalens_reports(factor_df, price_df, factor_names, task_id) -> str`

FACTOR 阶段可选报告生成入口。环境变量 `QUANT_ALPHALENS_REPORT=1` 时启用。

**返回：** 报告目录路径；未启用时返回空字符串。路径写入 `metadata["alphalens_report_dir"]` 供主调度器归档。

