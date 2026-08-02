# 配置指南

本文档说明 jingnitrader 的配置选项和使用方法。

## 环境变量

### 必需的环境变量

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| TUSHARE_TOKEN | Tushare Pro API Token | 否 | 空 |
| GM_TOKEN | 掘金量化API Token | 否 | 空 |
| DATA_DIR | 数据和工作目录 | 否 | "./workspace" |
| LOG_LEVEL | 日志级别 | 否 | "INFO" |

### 可选的环境变量

| 变量名 | 描述 | 可选值 | 默认值 |
|--------|------|--------|--------|
| DATA_BACKEND | 数据源后端 | tushare/baostock/akshare/xtquant/gm | "tushare" |
| BACKTEST_BACKEND | 回测框架 | rqalpha/backtrader/gm | "rqalpha" |
| TRADE_BACKEND | 交易接口 | xtquant/gm | "xtquant" |
| FACTOR_BACKEND | 因子计算库（技术指标） | talib/pandas_ta | "talib" |
| QUANT_FACTOR_BACKEND | DataFrame 后端（IC/中性化/相关性热路径） | pandas/polars/auto | "pandas" |
| QUANT_ALPHALENS_REPORT | 是否生成 Alphalens 因子分析报告 | 0/1 | "0" |
| QUANT_LEGACY_PIPELINE | 强制走旧 4 步硬编码因子处理路径（兼容回滚） | 0/1 | "0" |
| QUANT_WORK_DIR | 工作目录根路径 | 任意路径 | "./workspace" |

## factor-engine 高性能 DataFrame 后端

IC 计算 / 中性化 / IC Decay / 相关性分析等热路径支持 pandas / polars 双后端，通过 `QUANT_FACTOR_BACKEND` 环境变量切换。

### 取值说明

| 取值 | 行为 | 适用场景 |
|------|------|---------|
| `pandas` | 强制使用 pandas 后端（默认） | 兼容性最广，零额外依赖 |
| `polars` | 强制使用 polars 后端 | 大规模截面计算（5000+ 股票）；polars 缺失时报错 |
| `auto` | 自动检测 polars 可用性 | 生产环境推荐；polars 可用时自动启用，否则回退 pandas |

### 性能对比（5000 股 × 1000 日面板）

| 模块 | pandas 基线 | polars 后端 | 加速比 |
|------|------------|-------------|--------|
| Pearson IC | ~12s | ~1.0s | 10-15× |
| Spearman IC | ~15s | ~1.5s | 8-12× |
| 中性化（行业+市值） | ~10s | ~2s | 3-5× |
| IC Decay（lag 1-20） | ~30s | ~3s | 8-10× |
| 相关性分析（10 因子） | ~0.5s | ~0.4s | 1-1.5×（小矩阵无优势） |

### 覆盖模块

| 模块 | 文件 | 提速来源 |
|------|------|---------|
| Pearson IC | `skills/factor-engine/scripts/optimizations/ic_vectorized.py` | polars 多线程 Rust 引擎 |
| Spearman IC | `skills/factor-engine/scripts/optimizations/ic_vectorized.py` | polars 窗口函数 + rank |
| 中性化 | `skills/factor-engine/scripts/optimizations/vectorized_neutralize.py` | polars group_by 批量截面求解 |
| IC Decay | `skills/factor-engine/scripts/optimizations/ic_decay.py` | polars `over(code/date)` 一次性扫描 |
| 相关性分析 | `skills/factor-engine/scripts/optimizations/vectorized_correlation.py` | polars `DataFrame.corr()` |

### 使用示例

```bash
# 显式启用 polars（高性能场景）
export QUANT_FACTOR_BACKEND=polars

# 自动检测（推荐生产环境）
export QUANT_FACTOR_BACKEND=auto

# 临时禁用（强制 pandas，用于排查问题）
export QUANT_FACTOR_BACKEND=pandas
```

```python
import os
os.environ["QUANT_FACTOR_BACKEND"] = "polars"

from engine import FactorEngine
engine = FactorEngine()
# IC/中性化/IC Decay/相关性分析将自动使用 polars 后端
```

### 双后端一致性保证

- 所有支持 polars 后端的模块均通过 L2 单元测试验证
- 双后端输出最大绝对偏差 < 1e-10（IC Decay 因 rank 实现细节差异放宽到 1e-6）
- polars 缺失或运行异常时自动回退 pandas 并输出 warning 日志
- 通过 `backend` 参数可临时覆盖环境变量：
  ```python
  engine.correlation_analysis(df, factor_names, backend="pandas")  # 临时强制 pandas
  ```

### 依赖安装

```bash
# polars 为可选依赖，需手动安装
pip install "polars>=0.20.0"

# 验证可用性
python -c "import polars; print(polars.__version__)"
```

## factor-engine Alphalens 因子分析报告

通过 `QUANT_ALPHALENS_REPORT` 环境变量控制是否在 FACTOR 阶段自动生成 Alphalens 因子分析报告。

| 取值 | 行为 |
|------|------|
| `0` | 不生成报告（默认） |
| `1` | 为每个入选因子生成 4 PNG + 1 HTML + 1 metrics.json |

详见 [SKILL.md - Alphalens 因子分析报告](../skills/factor-engine/SKILL.md#alphalens-因子分析报告可选)。

## factor-engine Processor Pipeline（方向一）

因子处理流程抽象为可插拔的工序链 + 实验可重放记录器，通过 `pipeline.yaml` 声明式配置。

### QUANT_LEGACY_PIPELINE 环境变量

| 取值 | 行为 |
|------|------|
| `0`（默认） | 走 ProcessorChain 新路径，加载 `pipeline.yaml`，输出 manifest.json |
| `1` | 走 v1.x 的 4 步硬编码路径（IC → 相关性 → 选因子 → 融合），绕过 pipeline.yaml 与 ExperimentRecorder |

### pipeline.yaml 加载顺序

1. `<QUANT_WORK_DIR>/pipeline.yaml`（用户覆盖，优先级最高）
2. `<skill>/scripts/processors/pipeline.yaml`（默认配置）
3. 兜底默认链（`_default_processors()`，仅 IC + Correlation + Fusion）

### YAML 配置示例

```yaml
pipeline:
  - processor: NeutralizeProcessor
    enabled: true                  # 启用行业+市值中性化
    params:
      neutralize_mcap: true
      neutralize_industry: true
      min_count: 30

  - processor: WinsorizeProcessor
    enabled: false                  # 禁用去极值
    params:
      method: mad
      threshold: 3.0

  - processor: ICAnalysisProcessor
    enabled: true
    params:
      ic_type: normal               # 或 spearman
      forward_periods: [1, 5, 20]
      min_count: 10

  - processor: CorrelationFilterProcessor
    enabled: true
    params:
      max_correlation: 0.7

  - processor: FusionProcessor
    enabled: true
    params:
      method: ic_weighted           # 或 equal_weighted
      forward_period_for_weight: ret_forward_5d
```

### ExperimentRecorder 归档路径

每次 pipeline 运行自动在 `<QUANT_WORK_DIR>/archives/factor_engine/run_YYYYMMDD_HHMMSS/manifest.json` 输出 7 字段 manifest（run_id / start_time / pipeline_config / input_data_hash / steps / output_artifacts / env），manifest 自身 sha256 纳入 P1-3 artifact_store 覆盖范围，支持实验可重放。

详见 [SKILL.md - Processor Pipeline 架构](../skills/factor-engine/SKILL.md#processor-pipeline-架构方向一)。

## 配置文件

配置文件位于 `scripts/config.py`，包含以下配置项：

### 工作目录

```python
WORK_DIR = "./workspace"
DATA_DIR = os.path.join(WORK_DIR, "data")
FACTOR_DIR = os.path.join(WORK_DIR, "factors")
MODEL_DIR = os.path.join(WORK_DIR, "models")
BACKTEST_DIR = os.path.join(WORK_DIR, "backtest_results")
PORTFOLIO_DIR = os.path.join(WORK_DIR, "portfolio")
REPORT_DIR = os.path.join(WORK_DIR, "reports")
LOG_DIR = os.path.join(WORK_DIR, "logs")
```

### A股市场配置

```python
A_SHARE_COMMISSION_RATE = 0.00025    # 佣金 万2.5
A_SHARE_STAMP_TAX = 0.001           # 印花税 千1（卖出）
A_SHARE_MIN_COMMISSION = 5.0        # 最低佣金 5元
A_SHARE_MIN_LOT = 100               # 最小交易单位
A_SHARE_T_PLUS_1 = True             # T+1 交易
```

### 风控阈值

```python
MAX_DAILY_LOSS_RATIO = 0.03         # 单日最大亏损 3%
MAX_SINGLE_STOCK_WEIGHT = 0.10      # 单票最大持仓 10%
MAX_INDUSTRY_DEVIATION = 0.05       # 行业偏离基准 ±5%
NEW_STOCK_EXCLUDE_DAYS = 60         # 新股保护期
```

## 使用示例

### 1. 设置环境变量

```bash
export TUSHARE_TOKEN="your_token_here"
export GM_TOKEN="your_gm_token_here"
export DATA_BACKEND="tushare"
export LOG_LEVEL="DEBUG"
```

### 2. 在 Python 中使用

```python
import os
os.environ['TUSHARE_TOKEN'] = 'your_token_here'

from engine import run, MasterEngine
from context import Context

# 创建 Context
ctx = Context(
    task_id="task_001",
    user_intent="帮我用近3年A股数据做一个20日反转因子选股回测",
    current_stage="IDLE"
)

# 运行
result = run(ctx=ctx)
```

### 3. CLI 使用

```bash
# 设置环境变量
export TUSHARE_TOKEN="your_token_here"

# 运行
python engine.py -i "帮我用近3年A股数据做一个20日反转因子选股回测"
```

## 故障排除

### 常见问题

1. **Token 未设置**
   - 确保已设置 `TUSHARE_TOKEN` 环境变量
   - 或在代码中设置 `os.environ['TUSHARE_TOKEN'] = 'your_token'`

2. **目录权限错误**
   - 确保 `DATA_DIR` 目录存在且有写入权限
   - 或使用默认目录 `./workspace`

3. **后端加载失败**
   - 检查相应的包是否已安装
   - 如使用 tushare 后端，确保 `pip install tushare`
