# API 参考文档

本文档提供 jingnitrader 的完整 API 参考。

## 核心类

### MasterEngine

主调度引擎类，负责管理整个量化投研流程。

#### 方法

##### `__init__()`

初始化主调度引擎。

```python
engine = MasterEngine()
```

##### `parse_intent(user_input: str) -> Context`

解析用户自然语言输入，生成 Context 对象。

**参数：**
- `user_input` (str): 用户自然语言描述

**返回：**
- `Context`: 初始化后的上下文对象

**示例：**

```python
ctx = engine.parse_intent("帮我用近3年A股数据做一个20日反转因子选股回测")
```

##### `execute_stage(stage: str) -> bool`

执行单个阶段，调用对应的子 Skill。

**参数：**
- `stage` (str): 阶段名称（DATA/FACTOR/MODEL/BACKTEST/PORTFOLIO/EXECUTION/REPORT）

**返回：**
- `bool`: 执行是否成功

**示例：**

```python
success = engine.execute_stage("DATA")
```

##### `run_pipeline(user_input: str = None, ctx: Context = None) -> dict`

执行全流程管道。

**参数：**
- `user_input` (str, optional): 用户自然语言输入
- `ctx` (Context, optional): 已有的上下文对象

**返回：**
- `dict`: 执行结果，包含以下字段：
  - `success` (bool): 是否成功
  - `completed_stages` (List[str]): 已完成的阶段列表
  - `failed_stages` (List[str]): 失败的阶段列表
  - `summary` (str): 执行摘要
  - `context` (dict): 上下文对象

**示例：**

```python
result = engine.run_pipeline(user_input="帮我做一个回测")
print(result['success'])
print(result['summary'])
```

## 标准入口函数

### `run(ctx: Context = None, user_input: str = None) -> dict`

Skill 标准入口函数，所有 Skill 都应该实现此接口。

**参数：**
- `ctx` (Context, optional): 上下文对象
- `user_input` (str, optional): 用户自然语言

**返回：**
- `dict`: 执行结果

**示例：**

```python
from engine import run

# 使用自然语言
result = run(user_input="帮我做一个回测")

# 使用 Context
ctx = Context(...)
result = run(ctx=ctx)
```

## Context 类

上下文对象，用于在各个阶段之间传递状态。

### 属性

| 属性名 | 类型 | 描述 |
|--------|------|------|
| task_id | str | 任务ID |
| user_intent | str | 用户原始意图 |
| current_stage | str | 当前阶段 |
| target_stages | List[str] | 目标阶段列表 |
| stock_pool | List[str] | 股票池 |
| start_date | str | 开始日期 |
| end_date | str | 结束日期 |
| artifacts | Dict[str, str] | 各阶段产物路径 |
| metadata | Dict[str, Any] | 各阶段元数据 |
| errors | List[str] | 错误记录 |

### 方法

#### `update_artifact(stage: str, path: str)`

更新阶段产物路径。

```python
ctx.update_artifact("DATA", "/path/to/data.parquet")
```

#### `add_error(error: str)`

添加错误记录。

```python
ctx.add_error("数据获取失败：网络错误")
```

#### `to_dict() -> dict`

转换为字典格式。

```python
data = ctx.to_dict()
```

#### `from_json(json_str: str) -> Context`

从 JSON 字符串创建 Context 对象。

```python
ctx = Context.from_json(json_str)
```

## 常量

### STAGES

阶段列表：`["IDLE", "DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"]`

### STAGE_ORDER

阶段顺序映射：
```python
{
    "DATA": 1,
    "FACTOR": 2,
    "MODEL": 3,
    "BACKTEST": 4,
    "PORTFOLIO": 5,
    "EXECUTION": 6,
    "REPORT": 7,
}
```

### SKILL_MODULES

子 Skill 模块映射：
```python
{
    "DATA": "a_share_data_engine.scripts",
    "FACTOR": "a_share_factor_engine.scripts",
    "MODEL": "strategy_model_engine.scripts",
    "BACKTEST": "backtest_engine.scripts",
    "PORTFOLIO": "portfolio_risk_engine.scripts",
    "EXECUTION": "execution_monitor_engine.scripts",
    "REPORT": "reports_engine.scripts",
}
```

### EXPECTED_ARTIFACTS

各阶段预期产物文件：
```python
{
    "DATA": "cleaned_data.parquet",
    "FACTOR": "factor_data.parquet",
    "MODEL": "model.pkl",
    "BACKTEST": "backtest_result.json",
    "PORTFOLIO": "portfolio_weights.json",
    "EXECUTION": "trade_log.json",
    "REPORT": "report.html",
}
```

## 异常处理

所有方法都会捕获异常并记录到 `Context.errors` 中。如果执行失败，请检查：

1. `result['errors']` - 错误列表
2. `ctx.errors` - Context 中的错误记录
3. 日志文件 - 位于 `workspace/logs/` 目录

## 示例代码

### 完整流程

```python
from engine import run, MasterEngine
from context import Context

# 方式1：使用自然语言
result = run(user_input="帮我用近3年A股数据做一个20日反转因子选股回测")

# 方式2：使用 Context
ctx = Context(
    task_id="task_001",
    user_intent="帮我做一个回测",
    current_stage="IDLE",
    target_stages=["DATA", "BACKTEST", "REPORT"]
)
result = run(ctx=ctx)

# 检查结果
if result['success']:
    print("执行成功！")
    print(result['summary'])
else:
    print("执行失败：", result.get('errors'))
```

### CLI 使用

```bash
# 基础使用
python engine.py -i "帮我做一个回测"

# 保存结果
python engine.py -i "帮我做一个回测" -o result.json

# 使用已有的 Context
python engine.py -c context.json -i "继续执行"
```

## factor-engine API

### FactorEngine

因子引擎主类，位于 `skills/factor-engine/engine.py`。

#### `correlation_analysis(factor_df, factor_names=None, max_correlation=0.7, backend=None) -> dict`

因子相关性分析（支持 polars 后端）。

**参数：**
- `factor_df` (pd.DataFrame): 含 date, code, factor 列的面板数据
- `factor_names` (List[str], optional): 待分析的因子列表；默认分析所有非 date/code/industry 列
- `max_correlation` (float): 最大允许相关性阈值，默认 0.7
- `backend` (str, optional): `"pandas"` / `"polars"` / `"auto"` / `None`；`None` 时使用环境变量 `QUANT_FACTOR_BACKEND`

**返回：**
- `dict`: 包含 `correlation_matrix`、`selected_factors`、`removed_factors` 三个字段

**示例：**

```python
from engine import FactorEngine

engine = FactorEngine()
result = engine.correlation_analysis(
    df,
    factor_names=["momentum_20d", "lncap", "turnover_5d"],
    backend="polars",  # 强制使用 polars
)
print(result["removed_factors"])  # 被剔除的高相关因子
```

### 优化模块（`scripts/optimizations/`）

factor-engine 的热路径计算模块均位于 `scripts/optimizations/`，支持 pandas/polars 双后端。

#### IC 计算 - `ic_vectorized.py`

##### `ic_series_pearson(factor, fwd_ret, dates, min_obs=20, backend=None) -> pd.Series`

计算 Pearson IC 时间序列。

**参数：**
- `factor` (pd.Series): 因子值序列
- `fwd_ret` (pd.Series): 前瞻收益序列
- `dates` (pd.Series): 日期序列（与 factor/fwd_ret 等长对齐）
- `min_obs` (int): 单日截面最小样本数，默认 20
- `backend` (str, optional): `"pandas"` / `"polars"` / `"auto"` / `None`

**返回：**
- `pd.Series`: 以 date 为索引的 IC 时间序列

##### `ic_series_spearman(factor, fwd_ret, dates, min_obs=20, backend=None) -> pd.Series`

计算 Spearman Rank IC 时间序列（参数与返回值同 `ic_series_pearson`）。

#### 中性化 - `vectorized_neutralize.py`

##### `neutralize_factor(factor_df, factor_names, neutralize_mcap=True, neutralize_industry=True, mcap_col="lncap", industry_col="industry", min_count=30, backend=None) -> pd.DataFrame`

向量化因子中性化（行业 + 市值）。

**参数：**
- `factor_df` (pd.DataFrame): 含 date, code, factor, lncap, industry 列的面板数据
- `factor_names` (List[str]): 待中性化的因子列表
- `neutralize_mcap` (bool): 是否市值中性化，默认 True
- `neutralize_industry` (bool): 是否行业中性化，默认 True
- `mcap_col` (str): 对数市值列名，默认 `"lncap"`
- `industry_col` (str): 行业列名，默认 `"industry"`
- `min_count` (int): 截面最少样本数，默认 30
- `backend` (str, optional): `"pandas"` / `"polars"` / `"auto"` / `None`

**返回：**
- `pd.DataFrame`: 新增 `{factor}_neutral` 列的 DataFrame

##### `neutralize_factors_batch(...)` - 别名

与 `neutralize_factor` 等价，保留用于批量调用语义。

#### IC Decay - `ic_decay.py`

##### `class ICDecayAnalyzer`

IC 衰减分析器。

##### `ICDecayAnalyzer.calc_ic_decay(df, factor_col, backend=None) -> List[ICLagResult]`

计算不同前瞻期下的 IC 衰减情况。

**参数：**
- `df` (pd.DataFrame): 含 date, code, factor_col, close 列的面板数据
- `factor_col` (str): 因子列名
- `backend` (str, optional): `"pandas"` / `"polars"` / `"auto"` / `None`

**返回：**
- `List[ICLagResult]`: 每个 lag 的 IC 统计量列表

#### 相关性分析 - `vectorized_correlation.py`

##### `correlation_analysis(factor_df, factor_names=None, max_correlation=0.7, backend=None) -> dict`

独立模块版相关性分析（参数与 `FactorEngine.correlation_analysis` 一致）。

### 后端选择 - `scripts/optimizations/__init__.py`

##### `resolve_backend(backend: Optional[str] = None) -> str`

解析实际使用的后端。

**逻辑：**
1. `backend` 参数非 `None` 时优先使用
2. 否则读取环境变量 `QUANT_FACTOR_BACKEND`
3. `"auto"` 时检测 polars 可用性，可用返回 `"polars"`，不可用返回 `"pandas"`
4. `"polars"` 时检测 polars 可用性，不可用回退 `"pandas"` 并输出 warning

**返回：**
- `str`: 实际后端名称（`"pandas"` 或 `"polars"`）

### Alphalens 适配器 - `scripts/alphalens_adapter.py`

##### `class AlphalensAdapter`

Alphalens 因子分析报告生成器，详见 [SKILL.md - Alphalens 因子分析报告](../skills/factor-engine/SKILL.md#alphalens-因子分析报告可选)。

##### `AlphalensAdapter.generate_for_factor(factor_df, price_df, factor_name, output_dir, forward_periods=(1,5,10), quantiles=5) -> dict`

为单个因子生成完整的 Alphalens 分析报告。

**参数：**
- `factor_df` (pd.DataFrame): 含 date, code, {factor_name} 列
- `price_df` (pd.DataFrame): 含 date, code, close 列
- `factor_name` (str): 因子列名
- `output_dir` (str): 输出目录
- `forward_periods` (tuple): 前瞻期，默认 `(1, 5, 10)`
- `quantiles` (int): 分层数，默认 5

**返回：**
- `dict`: 文件路径字典，含 `returns_png`、`ic_png`、`turnover_png`、`summary_png`、`html`、`metrics_json`

##### `is_alphalens_enabled() -> bool`

检查环境变量 `QUANT_ALPHALENS_REPORT` 是否启用（`"1"` 时为 True）。

### Processor Pipeline - `scripts/processors/`

因子处理流程的可插拔工序链架构（方向一）。所有组件可通过 `from engine import processors` 访问。

#### 基类 - `scripts/processors/base.py`

##### `class Processor(ABC)`

Processor 抽象基类。子类需实现 `__call__` 与 `describe`，可声明 `requires` / `provides` 列表。

**类属性：**
- `requires: List[str]`：依赖的列名（执行前自动校验是否存在）
- `provides: List[str]`：提供的列名（仅文档化）

**实例属性：**
- `name: str`：默认返回类名
- `params: Dict[str, Any]`：构造参数

**方法：**
- `__call__(df: pd.DataFrame, ctx: ProcessContext) -> pd.DataFrame`：处理 DataFrame，返回新 DataFrame
- `describe() -> Dict[str, Any]`：返回工序元数据（用于 Recorder 落盘）
- `check_requirements(df: pd.DataFrame) -> None`：检查 df 是否包含 `requires` 声明的列，缺失时抛 `ProcessorRequirementError`

##### `@dataclass class ProcessContext`

工序间状态传递载体（轻量元数据）。

**字段：**
- `industry_df: Optional[pd.DataFrame]`：行业数据（中性化用）
- `recorder: Optional[ExperimentRecorder]`：实验记录器
- `ic_results: Dict[str, Any]`：IC 分析结果（ICAnalysisProcessor 写入）
- `selected_factors: List[str]`：选中的因子列表（CorrelationFilterProcessor 写入）
- `forward_returns: Optional[pd.DataFrame]`：前瞻收益 DataFrame
- `factor_names: List[str]`：待处理的因子名列表
- `task_id: str`：任务 ID
- `work_dir: Optional[Path]`：工作目录
- `backend: Optional[str]`：DataFrame 后端
- `metadata: Dict[str, Any]`：扩展元数据

##### `class ProcessorRequirementError(RuntimeError)`

依赖列缺失时抛出。

#### 调度器 - `scripts/processors/chain.py`

##### `class ProcessorChain(processors: List[Processor], fail_fast: bool = True)`

Processor 调度器，按序执行 Processor + 依赖检查 + 异常隔离。

**参数：**
- `processors`：有序的 Processor 列表
- `fail_fast`：`True` 时首个 Processor 失败立即抛异常；`False` 时跳过失败工序继续

**方法：**
- `run(df: pd.DataFrame, ctx: ProcessContext) -> pd.DataFrame`：顺序执行所有 Processor，返回处理后的 DataFrame
- `describe_chain() -> List[Dict[str, Any]]`：返回整条链的描述（供 Recorder 落盘）

##### `class ChainValidationError(RuntimeError)`

ProcessorChain 初始化时校验失败。

#### 加载器 - `scripts/processors/loader.py`

##### `load_pipeline_config(work_dir: Optional[Path] = None, config_filename: str = "pipeline.yaml") -> List[Processor]`

加载 pipeline 配置，返回 Processor 实例列表。加载顺序：`work_dir/config_filename` → skill 默认配置 → 兜底默认链。

##### `parse_yaml_to_processors(config: Dict[str, Any]) -> List[Processor]`

将已解析的 YAML 字典转为 Processor 实例列表（`enabled: false` 的工序被跳过）。

##### `register_processor(name: str, cls: type) -> None`

注册自定义 Processor（cls 必须继承 Processor）。

##### `PROCESSOR_REGISTRY: Dict[str, type]`

Processor 类名 → 类对象的注册表，预注册 7 个内置 Processor。

#### 7 个内置 Processor

| 类名 | 文件 | 关键参数 |
|------|------|---------|
| `NeutralizeProcessor` | `scripts/processors/neutralize.py` | `neutralize_mcap` / `neutralize_industry` / `min_count` |
| `WinsorizeProcessor` | `scripts/processors/winsorize.py` | `method`（mad/quantile）/ `threshold` / `quantile_range` |
| `FillnaProcessor` | `scripts/processors/fillna.py` | `method`（rank_pct/zero/mean/ffill）/ `fill_value` |
| `StandardizeProcessor` | `scripts/processors/standardize.py` | `method`（zscore/minmax） |
| `ICAnalysisProcessor` | `scripts/processors/ic_analysis.py` | `ic_type`（normal/spearman）/ `forward_periods` / `min_count` |
| `CorrelationFilterProcessor` | `scripts/processors/correlation_filter.py` | `max_correlation` |
| `FusionProcessor` | `scripts/processors/fusion.py` | `method`（ic_weighted/equal_weighted）/ `forward_period_for_weight` |

每个 Processor 实现 `__call__(df, ctx) -> DataFrame` 与 `describe() -> dict`，详见各文件 docstring。

#### 实验记录器 - `scripts/recorder.py`

##### `class ExperimentRecorder(archive_dir: Path, pipeline_config: Optional[List[Dict]] = None, input_data_paths: Optional[List[str]] = None)`

实验可重放记录器。`__init__` 时创建 `<archive_dir>/run_YYYYMMDD_HHMMSS/` 子目录。

**方法：**
- `set_pipeline_config(config: List[Dict]) -> None`：设置 pipeline 配置
- `log_step(processor, df_after, before_rows=None, before_cols=None, after_rows=None, after_cols=None) -> None`：记录单个 Processor 执行后的状态
- `log_output_artifact(name: str, path: str) -> None`：记录输出产物
- `finalize() -> Optional[str]`：写入 `manifest.json`，返回其 sha256 指纹；写盘失败不阻塞主流程（降级为内存模式）

**属性：**
- `manifest_path: Optional[Path]`：manifest.json 完整路径（内存模式下为 None）

##### manifest.json 7 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `run_id` | str | uuid4().hex |
| `start_time` / `end_time` | str | ISO 格式时间戳 |
| `pipeline_config` | List[Dict] | 各 Processor 的 describe() 输出 |
| `input_data_hash` | Dict[str, str] | 输入文件名 → sha256 |
| `steps` | List[Dict] | 每步执行状态 |
| `output_artifacts` | List[Dict] | 输出产物列表 |
| `env` | Dict | 环境快照（Python/pandas/polars 版本 + QUANT_ 环境变量） |
