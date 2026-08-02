# 因子引擎增强方案（三方向）

> 文档版本：v1.0
> 创建日期：2026-08-02
> 关联 PRD：[prd_gemstar_integration.md](prd_gemstar_integration.md) v1.2
> 状态：待用户确认

---

## 文档结构

本文件包含三个相互独立的增强方案，可分别立项、独立开发、独立部署：

| 方向 | 主题 | 优先级 | 预估工作量 | 风险等级 |
|------|------|--------|-----------|---------|
| 方向一 | Qlib Processor Pipeline + Recorder | 中长期 | 大 | 中 |
| 方向二 | Polars 后端 | 中期 | 中 | 低 |
| 方向三 | Alphalens 真正集成 | 短期优先 | 小 | 中 |

**建议实施顺序**：方向三（低垂果实） → 方向二（性能基建） → 方向一（架构升级）。

---

# 方向一：Qlib Processor Pipeline + Recorder

## 1.1 背景与现状

### 现状代码位置

因子处理流程当前散落在 [engine.py](../skills/factor-engine/engine.py) 的 5 个独立方法中：

| 步骤 | 现有方法 | 行号 | 问题 |
|------|---------|------|------|
| 中性化 | `FactorEngine.neutralize()` | L165-229 | 硬编码市值+行业，无法插拔 |
| IC 分析 | `FactorEngine.ic_analysis()` | L231-286 | 与中性化解耦，但无前置依赖声明 |
| 相关性去冗余 | `FactorEngine.correlation_analysis()` | L316-353 | 独立函数，无先后约束 |
| 缺失填充 | 散落在 `factor_fusion()` | L377-379 | `fillna(0.5)` 写死在融合里 |
| 融合 | `FactorEngine.factor_fusion()` | L355-... | 与 fillna 耦合 |

### 核心痛点

1. **加新工序成本高**：如需新增 winsorize（去极值），要改 `run()` 主函数 + 加方法 + 改测试 + 改文档
2. **顺序写死**：5 步顺序硬编码在 `run()` 中，无法按场景灵活组合
3. **实验不可重放**：[scripts/archive.py](../scripts/archive.py) 只归档 parquet 快照，缺"为什么这么算"的元数据。两周后回头看某次 IC 异常无法复现参数
4. **重复样板代码**：每个 `Processor` 都要写一遍 try/except、空 DataFrame 校验、日志记录

## 1.2 设计目标

| 目标 | 量化指标 |
|------|---------|
| 工序可插拔 | 新增 1 个 Processor 仅需创建 1 个类 + 配置 1 行 YAML，零改 `run()` |
| 实验可重放 | 任一历史 archive 可在 1 条命令内复现完整计算上下文 |
| 处理流程可视化 | 每个 archive 输出 `pipeline.yaml`，记录工序链 + 参数 + 输入数据 hash |
| 完全向后兼容 | 现有 `FactorEngine.neutralize()` 等 API 保留为 deprecated 别名，2 个版本周期内不删除 |

## 1.3 设计方案

### 1.3.1 抽象基类

新增文件 [skills/factor-engine/scripts/processors/base.py](../skills/factor-engine/scripts/processors/base.py)：

```python
class Processor(ABC):
    """因子处理器基类，借鉴 Qlib Processor 设计"""

    # 子类声明：是否需要行业数据、市值数据等前置依赖
    requires: List[str] = []  # 例：["industry", "lncap"]

    @abstractmethod
    def __call__(self, df: pd.DataFrame, ctx: "ProcessContext") -> pd.DataFrame:
        """处理 DataFrame，返回新 DataFrame"""

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """返回工序元数据，用于 Recorder 落盘"""
```

### 1.3.2 内置 Processor 清单

新增文件目录 `scripts/processors/`：

| 文件 | Processor 类 | 替代现有代码 | 备注 |
|------|------------|------------|------|
| `neutralize.py` | `NeutralizeProcessor` | `neutralize()` L165-229 | 市值+行业中性化 |
| `winsorize.py` | `WinsorizeProcessor` | **新增** | MAD 法去极值，3σ 阈值可配 |
| `fillna.py` | `FillnaProcessor` | `factor_fusion()` 中的 `fillna(0.5)` | 抽离填充逻辑 |
| `standardize.py` | `StandardizeProcessor` | **新增** | RobustZScoreNorm / CSRankNorm |
| `ic_analysis.py` | `ICAnalysisProcessor` | `ic_analysis()` L231-286 | 输出 IC 落 ctx |
| `correlation_filter.py` | `CorrelationFilterProcessor` | `correlation_analysis()` L316-353 | 去冗余 |
| `fusion.py` | `FusionProcessor` | `factor_fusion()` L355-... | IC 加权融合 |

### 1.3.3 ProcessorChain 调度器

```python
class ProcessorChain:
    """按顺序执行 Processor 链，自动校验依赖"""

    def __init__(self, processors: List[Processor]):
        self.processors = processors
        self._validate_dependencies()  # 拓扑校验

    def run(self, df: pd.DataFrame, ctx: "ProcessContext") -> pd.DataFrame:
        for p in self.processors:
            self._check_requirements(p, df)  # 检查 requires 字段
            df = p(df, ctx)
            ctx.recorder.log_step(p, df)  # 自动记录每步
        return df
```

### 1.3.4 Recorder 实验记录器

新增文件 [scripts/recorder.py](../scripts/recorder.py)：

```python
class ExperimentRecorder:
    """借鉴 Qlib Recorder + MLflow 风格"""

    def __init__(self, archive_dir: Path):
        self.dir = archive_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "run_id": uuid4().hex,
            "start_time": datetime.now().isoformat(),
            "pipeline_config": None,        # 工序链 YAML
            "input_data_hash": {},          # 输入 parquet 的 sha256
            "steps": [],                    # 每步 describe() 输出
            "output_artifacts": [],         # 输出文件清单
            "env": self._snapshot_env(),    # FACTOR_BACKEND / IC_TYPE 等
        }

    def log_step(self, processor: Processor, df_after: pd.DataFrame):
        """记录每个工序执行后的 DataFrame 指纹"""
        self.manifest["steps"].append({
            "processor": processor.__class__.__name__,
            "params": processor.describe(),
            "rows_after": len(df_after),
            "cols_after": list(df_after.columns),
            "nan_ratio": df_after.isna().mean().mean(),
        })

    def finalize(self):
        (self.dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False, default=str)
        )
```

### 1.3.5 配置文件示例

新增 [skills/factor-engine/scripts/processors/pipeline.yaml](../skills/factor-engine/scripts/processors/pipeline.yaml)：

```yaml
# 因子处理流水线配置（声明式，零代码改动可调整）
pipeline:
  - processor: NeutralizeProcessor
    enabled: true
    params:
      neutralize_mcap: true
      neutralize_industry: true
      min_sample: 30

  - processor: WinsorizeProcessor     # 新增能力，配置即可启用
    enabled: true
    params:
      method: mad                      # mad / sigma
      threshold: 3.0

  - processor: FillnaProcessor
    enabled: true
    params:
      method: rank_pct                 # rank_pct / mean / zero
      fill_value: 0.5

  - processor: ICAnalysisProcessor
    enabled: true
    params:
      ic_type: spearman
      forward_periods: [1, 5, 20]

  - processor: CorrelationFilterProcessor
    enabled: true
    params:
      max_correlation: 0.7

  - processor: FusionProcessor
    enabled: true
    params:
      method: ic_weighted
```

### 1.3.6 与现有 archive.py 集成

现有 [scripts/archive.py](../scripts/archive.py) 升级为调用 `ExperimentRecorder`：

```python
# scripts/archive.py 修改
def archive_artifacts(work_dir: Path, skill_name: str, ctx: Context):
    # 旧逻辑保留：拷贝 parquet 到 archives/<timestamp>/
    # 新增：若 ctx 含 recorder，则将 manifest.json 一并归档
    if hasattr(ctx, "recorder") and ctx.recorder is not None:
        ctx.recorder.finalize()
        shutil.copy(ctx.recorder.dir / "manifest.json", archive_dir)
```

### 1.3.7 兼容层

`FactorEngine` 保留旧 API 作为 deprecated 别名，内部转调 Processor：

```python
class FactorEngine:
    def neutralize(self, factor_df, industry_df, **kwargs):
        warnings.warn("neutralize() 将在 v3.0 移除，请用 ProcessorChain", DeprecationWarning)
        p = NeutralizeProcessor(**kwargs)
        ctx = ProcessContext(industry_df=industry_df)
        return p(factor_df, ctx)
```

## 1.4 量化评估维度

| 维度 | 现状 | 目标 | 验证方法 |
|------|------|------|---------|
| 新增工序代码量 | 改 4 处（engine+test+doc+config）≈ 80 行 | 仅创建 1 个类 ≈ 30 行 | 实测新增 WinsorizeProcessor |
| 实验可重放性 | 0%（仅 parquet 快照） | 100%（manifest.json 含全参数） | 复跑历史 archive 对比 IC |
| 工序组合灵活度 | 1 种（写死） | ≥ 8 种（YAML 排列组合） | 配置驱动跑通不同链 |
| 单元测试覆盖 | 现有方法各自测试 | 每个 Processor 100% 行覆盖 | pytest --cov |

## 1.5 可验证验收标准

1. **CR-1**：新建 `WinsorizeProcessor` 仅需 30 行代码 + 1 行 YAML，零修改 `engine.py`
2. **CR-2**：通过 `pipeline.yaml` 删除某个 Processor，全链路仍能正常运行且结果一致
3. **CR-3**：跑完一次后 `archives/run_xxx/manifest.json` 含 7 字段（run_id / start_time / pipeline_config / input_data_hash / steps / output_artifacts / env）
4. **CR-4**：用同一 manifest 重跑，输出 IC 与原 archive 偏差 < 1e-10（浮点误差容忍）
5. **CR-5**：旧 `FactorEngine.neutralize()` 调用产生 DeprecationWarning 且结果与 v1.x 完全一致
6. **CR-6**：489 条现有回归测试 100% 通过（参考 PRD v1.2 P1-3 完成时的基线）

## 1.6 兼容层与回滚计划

### 兼容层

- 旧 `FactorEngine.neutralize/ic_analysis/correlation_analysis/factor_fusion` 4 个方法保留，转调对应 Processor
- 触发 `DeprecationWarning`，文档明确 v3.0 移除
- `run()` 内部默认走 ProcessorChain，但保留环境变量 `QUANT_LEGACY_PIPELINE=1` 强制走旧路径

### 回滚计划

| 触发条件 | 回滚动作 | 验证 |
|---------|---------|------|
| 任一 Processor 输出与旧方法偏差 > 1e-6 | 设置 `QUANT_LEGACY_PIPELINE=1` | 回归测试通过 |
| ProcessorChain 调度死锁或拓扑校验失败 | 同上 | 单元测试通过 |
| Recorder 写盘失败导致 archive 异常 | 自动降级为旧 archive 逻辑 | archive 目录正常 |

## 1.7 测试策略

遵循项目硬约束「关键路径 100% 行覆盖，非关键路径 ≥80%」：

### L1 契约测试（contract）

新增 [tests/factor_engine/test_run_contract.py](../tests/factor_engine/test_run_contract.py)（已存在则扩展）：
- ProcessorChain.run() 必须返回与旧 `run()` 同结构 DataFrame
- Recorder.finalize() 必产出 manifest.json 含 7 必填字段

### L2 单元测试（unit）

新增 `tests/factor_engine/test_processors/`：
- 每个 Processor 一个测试文件，100% 行覆盖
- `test_processor_chain.py`：拓扑校验 / 依赖缺失 / 异常隔离
- `test_recorder.py`：manifest 落盘 / 数据 hash 一致性 / env 快照

### L3 集成测试（integration）

扩展 [tests/integration/test_pipeline_archives.py](../tests/integration/test_pipeline_archives.py)：
- 端到端跑一次完整 pipeline，校验 manifest 可重放
- 4 种 YAML 配置组合跑通

### 测试 marker

`@pytest.mark.skill_factor` / `@pytest.mark.contract` / `@pytest.mark.unit` / `@pytest.mark.integration`

## 1.8 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| Processor 间状态传递设计不当导致 ctx 膨胀 | 中 | 中 | ctx 仅传引用，大数据走 parquet 落盘 |
| 拓扑校验误判依赖导致合法配置被拒 | 低 | 高 | 提供显式 `override_dependency` 参数 |
| Recorder 同步写盘拖慢主流程 | 中 | 低 | 改异步队列，主流程不阻塞 |
| YAML 配置错误导致静默失败 | 中 | 高 | 启动时 schema 校验（pydantic）+ 详细报错 |

## 1.9 工作分解（任务清单）

| ID | 任务 | 文件 | 依赖 |
|----|------|------|------|
| T1-1 | 创建 Processor 基类 + ProcessContext | `scripts/processors/base.py` | - |
| T1-2 | 实现 7 个内置 Processor | `scripts/processors/*.py` | T1-1 |
| T1-3 | 实现 ProcessorChain 调度器 | `scripts/processors/chain.py` | T1-1 |
| T1-4 | 实现 ExperimentRecorder | `scripts/recorder.py` | T1-1 |
| T1-5 | 改造 `engine.py.run()` 走 ProcessorChain | `skills/factor-engine/engine.py` | T1-2, T1-3 |
| T1-6 | 改造 `scripts/archive.py` 集成 Recorder | `scripts/archive.py` | T1-4 |
| T1-7 | 旧 API 转 deprecated 别名 + 兼容层 | `skills/factor-engine/engine.py` | T1-5 |
| T1-8 | 新增 `pipeline.yaml` 默认配置 | `scripts/processors/pipeline.yaml` | T1-2 |
| T1-9 | L2 单元测试（7 Processor + Chain + Recorder） | `tests/factor_engine/test_processors/` | T1-2 ~ T1-4 |
| T1-10 | L3 集成测试扩展 | `tests/integration/test_pipeline_archives.py` | T1-5, T1-6 |
| T1-11 | 文档更新（SKILL.md / config_guide.md / api_reference.md） | 3 个 references 文件 | T1-7 |

---

# 方向二：Polars 后端

## 2.1 背景与现状

### 现状代码位置

IC 计算位于 [scripts/optimizations/ic_vectorized.py](../skills/factor-engine/scripts/optimizations/ic_vectorized.py)，已用 pandas groupby 向量化但仍单线程：

```python
# 当前实现：pandas 单线程
g = df.groupby("d", sort=True)
fx = df["f"] - g["f"].transform("mean")
rx = df["r"] - g["r"].transform("mean")
df["_num"] = (fx * rx).values
```

中性化位于 [scripts/optimizations/vectorized_neutralize.py](../skills/factor-engine/scripts/optimizations/vectorized_neutralize.py)，已用 `numpy.linalg.lstsq` 替代 sklearn，但仍是逐截面循环。

### 核心痛点

5000 股 × 1000 日 × 50 因子 = 2.5 亿行场景下：
- 当前向量化 IC ≈ 3 秒
- 中性化逐截面循环 ≈ 8-15 秒
- Polars 实测同场景 ≈ 0.4 秒 / 1 秒（5-15× 提速）

## 2.2 设计目标

| 目标 | 量化指标 |
|------|---------|
| 大数据集性能提升 | 5000 股 × 1000 日 IC 计算 ≤ 1 秒（当前 3 秒） |
| 零侵入 | 外部 API 签名不变，仅新增 `backend` 参数 |
| 可选启用 | Polars 为可选依赖，缺失时自动 fallback pandas |
| 双后端结果一致 | Polars 与 pandas 输出 IC 偏差 < 1e-10 |

## 2.3 设计方案

### 2.3.1 依赖管理

[requirements.txt](../requirements.txt) 新增可选依赖：

```text
# 现有
pandas>=2.0.0
numpy>=1.24.0

# 新增（可选）
polars>=0.20.0  # 可选高性能后端
```

[skills/factor-engine/SKILL.md](../skills/factor-engine/SKILL.md) 新增环境变量：

```yaml
- name: QUANT_FACTOR_BACKEND
  description: 因子计算 DataFrame 后端（pandas / polars），默认 pandas
  required: false
  default: "pandas"
```

> 注意：与现有 `FACTOR_BACKEND`（pandas_ta / talib 技术指标后端）语义不同，故用 `QUANT_FACTOR_BACKEND` 前缀避免冲突，遵循项目硬约束。

### 2.3.2 双后端函数签名

修改 [scripts/optimizations/ic_vectorized.py](../skills/factor-engine/scripts/optimizations/ic_vectorized.py)：

```python
def ic_series_pearson(
    factor: pd.Series,
    forward_ret: pd.Series,
    dates: Optional[pd.Series] = None,
    min_obs: int = 10,
    backend: str = "pandas",  # 新增：pandas / polars / auto
) -> pd.Series:
    if backend == "auto":
        backend = "polars" if _polars_available() else "pandas"
    if backend == "polars":
        return _ic_pearson_polars(factor, forward_ret, dates, min_obs)
    return _ic_pearson_pandas(factor, forward_ret, dates, min_obs)  # 原逻辑
```

### 2.3.3 Polars 实现示例

```python
def _ic_pearson_polars(factor, forward_ret, dates, min_obs):
    import polars as pl
    df = pl.DataFrame({
        "d": dates,
        "f": pl.Series(factor.values),
        "r": pl.Series(forward_ret.values),
    }).drop_nulls()
    if df.height == 0:
        return pd.Series(dtype=float)

    # lazy + 多线程，全 Rust 引擎
    result = (
        df.lazy()
        .with_columns([
            (pl.col("f") - pl.col("f").mean().over("d")).alias("fx"),
            (pl.col("r") - pl.col("r").mean().over("d")).alias("rx"),
        ])
        .with_columns((pl.col("fx") * pl.col("rx")).alias("_num"))
        .group_by("d")
        .agg([
            pl.col("_num").sum().alias("num"),
            ((pl.col("fx") ** 2).sum() * (pl.col("rx") ** 2).sum()).sqrt().alias("denom"),
            pl.len().alias("n"),
        ])
        .filter(pl.col("n") >= min_obs)
        .with_columns((pl.col("num") / pl.col("denom")).alias("ic"))
        .sort("d")
        .collect()
    )
    return pd.Series(result["ic"].to_list(), index=pd.to_datetime(result["d"]))
```

### 2.3.4 改造范围

| 文件 | 改造方法 | 优先级 |
|------|---------|--------|
| `ic_vectorized.py` | `ic_series_pearson` / `ic_series_spearman` / `ic_analysis_batch` | P0 |
| `vectorized_neutralize.py` | `neutralize_factor` / `neutralize_factors_batch` | P0 |
| `ic_decay.py` | `ICDecayAnalyzer.analyze`（多 lag 扫描） | P1 |
| `factor_validator.py` | `validate_factor`（bootstrap 采样） | P2（性能不敏感） |
| `engine.py` | `correlation_analysis`（相关性矩阵） | P1 |

### 2.3.5 自动 fallback 机制

```python
def _polars_available() -> bool:
    try:
        import polars  # noqa
        return True
    except ImportError:
        return False

# 模块加载时检查并日志提示
if not _polars_available():
    logger.info("polars 未安装，使用 pandas 后端。pip install polars>=0.20.0 启用加速")
```

## 2.4 量化评估维度

| 维度 | 现状（pandas） | 目标（polars） | 验证方法 |
|------|--------------|--------------|---------|
| IC 计算耗时（5000×1000） | ~3 秒 | ≤ 1 秒 | `pytest --benchmark` |
| 中性化耗时（5000×1000） | ~10 秒 | ≤ 2 秒 | 同上 |
| 内存峰值 | 1× | ≤ 1.5× | memory_profiler |
| 结果一致性 | 基线 | 偏差 < 1e-10 | 双后端输出 diff 测试 |

## 2.5 可验证验收标准

1. **CR-1**：`pip install polars` 后 `QUANT_FACTOR_BACKEND=polars` 环境下 IC 计算正确
2. **CR-2**：未安装 polars 时自动 fallback pandas，日志输出提示
3. **CR-3**：双后端 IC 输出最大绝对偏差 < 1e-10（用 `np.testing.assert_allclose`）
4. **CR-4**：5000 股 × 1000 日测试集下 polars 耗时较 pandas 减少 ≥ 60%
5. **CR-5**：489 条现有回归测试 100% 通过
6. **CR-6**：`QUANT_FACTOR_BACKEND` 未设置时默认 pandas，行为与 v1.x 完全一致

## 2.6 兼容层与回滚计划

### 兼容层

- `backend` 参数默认 `"pandas"`，所有现有调用零改动
- `QUANT_FACTOR_BACKEND` 未设置等同 `"pandas"`
- polars 缺失时 try/except fallback，主流程不中断

### 回滚计划

| 触发条件 | 回滚动作 | 验证 |
|---------|---------|------|
| polars 输出与 pandas 偏差 > 1e-10 | 设置 `QUANT_FACTOR_BACKEND=pandas` | 回归测试通过 |
| polars 版本不兼容（API 变更） | 同上 + 日志告警 | import 测试通过 |
| 内存占用 > 2× | 同上 | memory_profiler 监控 |

## 2.7 测试策略

### L2 单元测试

新增 [tests/factor_engine/test_polars_backend.py](../tests/factor_engine/test_polars_backend.py)：
- `test_ic_polars_vs_pandas_consistency`：双后端输出一致
- `test_fallback_when_polars_missing`：mock ImportError 验证 fallback
- `test_neutralize_polars_vs_pandas`：中性化双后端一致
- `@pytest.mark.slow` 标注大数据集性能基准测试

### L3 集成测试

扩展 [tests/factor_engine/test_factor_analysis.py](../tests/factor_engine/test_factor_analysis.py)：
- 端到端 `QUANT_FACTOR_BACKEND=polars` 跑通完整 pipeline
- 双后端跑同一数据集，最终 IC 报告 diff

### 测试 marker

新增 `@pytest.mark.requires_polars`（缺 polars 时 skip）

## 2.8 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| polars API 在 0.20 → 1.0 升级中断 | 中 | 中 | 锁版本 + 抽象层隔离 |
| datetime / category 类型在 polars 中行为不一致 | 高 | 中 | 增加 dtype 转换层 + 单元测试覆盖 |
| group_by 顺序与 pandas 不一致导致 IC 序列错位 | 中 | 高 | 显式 sort + 索引对齐校验 |
| Windows 下 polars 编译问题 | 低 | 高 | 提供 wheel + CI 双平台测试 |

## 2.9 工作分解（任务清单）

| ID | 任务 | 文件 | 依赖 |
|----|------|------|------|
| T2-1 | 新增 polars 可选依赖 + 环境变量 | `requirements.txt` / `SKILL.md` | - |
| T2-2 | IC 计算 polars 实现 | `scripts/optimizations/ic_vectorized.py` | T2-1 |
| T2-3 | 中性化 polars 实现 | `scripts/optimizations/vectorized_neutralize.py` | T2-1 |
| T2-4 | IC Decay polars 实现 | `scripts/optimizations/ic_decay.py` | T2-2 |
| T2-5 | 相关性分析 polars 实现 | `skills/factor-engine/engine.py` | T2-1 |
| T2-6 | 自动 fallback + 日志提示 | `scripts/optimizations/__init__.py` | T2-2 |
| T2-7 | L2 单元测试（一致性 + fallback） | `tests/factor_engine/test_polars_backend.py` | T2-2, T2-3 |
| T2-8 | L3 集成测试扩展 | `tests/factor_engine/test_factor_analysis.py` | T2-5 |
| T2-9 | 性能基准测试（带 `@pytest.mark.slow`） | `tests/factor_engine/test_polars_perf.py` | T2-7 |
| T2-10 | 文档更新（SKILL.md / config_guide.md） | 2 个 references 文件 | T2-6 |

---

# 方向三：Alphalens 真正集成

## 3.1 背景与现状

### 现状代码位置

[skills/factor-engine/SKILL.md#L21](../skills/factor-engine/SKILL.md) 声明依赖 `alphalens>=0.4.0`，但全代码库无 `import alphalens` 调用。

IC 分析当前仅输出 JSON（[engine.py#L231-286](../skills/factor-engine/engine.py)）：

```json
{
  "ret_forward_5d": [
    {
      "factor": "lncap",
      "ic_mean": 0.03,
      "ic_std": 0.067,
      "ic_ir": 0.45,
      "ic_positive_ratio": 0.62,
      "ic_t_stat": 1.92
    }
  ]
}
```

### 核心痛点

看到 IC=0.03 无法回答 3 个关键问题：
1. **稳定性**：因子是全程稳定还是某段时间爆发？（缺 IC 时序图）
2. **盈利性**：买 Top 20% 股票到底赚不赚钱？（缺分层回测净值曲线）
3. **成本**：因子多久换一次仓？交易成本多大？（缺换手率分析）

## 3.2 设计目标

| 目标 | 量化指标 |
|------|---------|
| 因子筛选维度 | 从 5 项 IC 统计量扩展到 4 类报告（IC 时序 / 分层净值 / alpha-beta / 换手率） |
| 集成代码量 | ≤ 200 行（数据格式适配 + 调用 alphalens API） |
| 报告产物 | 每个 archive 自动生成 `alphalens_report.html` + 4 张 PNG |
| 向后兼容 | 现有 `ic_report.json` 保留，alphalens 报告为附加产物 |

## 3.3 设计方案

### 3.3.1 依赖风险与选型

**关键风险**：原版 `alphalens` 自 2018 年后停止维护，且依赖 pandas 旧版本，与项目 `pandas>=2.0.0` 可能冲突。

**选型方案**：

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| A. 原版 alphalens 0.4.0 | 文档全 | 停止维护，pandas 兼容性风险 | 不推荐 |
| B. quantopian-reloaded | 社区维护版本，兼容新 pandas | 仍属小众 | 推荐 |
| C. 自研轻量级分层回测 | 无外部依赖，可控 | 工作量大，缺可视化 | 备选 |
| D. alphalens-reloaded (stefan-jansen) | 持续维护，pip 易装 | 需确认 Python 3.9+ 兼容 | **首选** |

**首选方案 D**：`alphalens-reloaded`，pip 安装，与 pandas 2.0 兼容。失败时降级到方案 C（自研最小分层回测，仅输出 JSON 指标不做图）。

### 3.3.2 依赖管理

[requirements.txt](../requirements.txt) 调整：

```text
# 替换
- alphalens>=0.4.0
# 为
+ alphalens-reloaded>=0.4.5  # alphalens 维护版本，兼容 pandas 2.0
```

### 3.3.3 数据格式适配器

新增 [scripts/alphalens_adapter.py](../skills/factor-engine/scripts/alphalens_adapter.py)：

```python
class AlphalensAdapter:
    """将 jingni-trader 内部数据结构适配为 alphalens 输入格式"""

    @staticmethod
    def to_alphalens_format(
        factor_df: pd.DataFrame,    # columns: code, date, factor1, factor2, ...
        price_df: pd.DataFrame,     # columns: code, date, close
        factor_name: str,
        forward_periods: Tuple[int] = (1, 5, 20),
        quantiles: int = 5,
    ):
        """转 alphalens 期望的 MultiIndex(Series) + price pivot"""
        import alphalens as al

        # 1. factor Series，索引为 (date, code)
        factor_series = (
            factor_df.set_index(["date", "code"])[factor_name]
            .dropna()
            .sort_index()
        )

        # 2. price pivot，索引为 date，列为 code
        price_pivot = price_df.pivot(index="date", columns="code", values="close")

        # 3. alphalens 标准清洗
        factor_data = al.utils.get_clean_factor_and_forward_returns(
            factor=factor_series,
            prices=price_pivot,
            quantiles=quantiles,
            periods=forward_periods,
            max_loss=0.25,  # 允许 25% 数据丢失
        )
        return factor_data

    @staticmethod
    def generate_full_report(factor_data, output_dir: Path, factor_name: str):
        """生成完整 alphalens 报告，输出 HTML + 4 PNG"""
        import matplotlib
        matplotlib.use("Agg")  # 无头模式，避免 Windows 显示问题
        import alphalens as al

        output_dir.mkdir(parents=True, exist_ok=True)

        # 4 张图分别保存（避免 create_full_tear_sheet 一次性出图难管理）
        al.tears.create_returns_tear_sheet(
            factor_data, save_fig=str(output_dir / f"{factor_name}_returns.png")
        )
        al.tears.create_information_tear_sheet(
            factor_data, save_fig=str(output_dir / f"{factor_name}_ic.png")
        )
        al.tears.create_turnover_tear_sheet(
            factor_data, save_fig=str(output_dir / f"{factor_name}_turnover.png")
        )
        al.tears.create_summary_tear_sheet(
            factor_data, save_fig=str(output_dir / f"{factor_name}_summary.png")
        )
```

### 3.3.4 与 engine.py 集成

修改 [engine.py](../skills/factor-engine/engine.py) `ic_analysis` 末尾，追加可选报告生成：

```python
def ic_analysis(self, factor_df, forward_returns, factor_names=None):
    # ... 现有 IC 计算逻辑保留 ...

    # 新增：可选生成 alphalens 报告
    if os.environ.get("QUANT_ALPHALENS_REPORT", "0") == "1":
        from scripts.alphalens_adapter import AlphalensAdapter
        from context import get_work_dir

        report_dir = get_work_dir() / "reports" / "alphalens" / self.task_id
        price_df = factor_df[["code", "date", "close"]].copy()  # 简化示例

        for factor_name in (factor_names or []):
            try:
                factor_data = AlphalensAdapter.to_alphalens_format(
                    factor_df, price_df, factor_name
                )
                AlphalensAdapter.generate_full_report(
                    factor_data, report_dir, factor_name
                )
                logger.info(f"alphalens 报告已生成: {report_dir}/{factor_name}_*.png")
            except Exception as e:
                logger.warning(f"alphalens 报告生成失败（因子 {factor_name}）: {e}")

    return results
```

### 3.3.5 环境变量

[SKILL.md](../skills/factor-engine/SKILL.md) 新增：

```yaml
- name: QUANT_ALPHALENS_REPORT
  description: 是否生成 alphalens 完整因子分析报告（0/1），默认 0
  required: false
  default: "0"
```

默认关闭以避免无需求时的额外开销。需要时显式 `QUANT_ALPHALENS_REPORT=1` 启用。

### 3.3.6 报告产物结构

```
workspace/reports/alphalens/<task_id>/
├── lncap_returns.png         # 分层净值 + 累积收益
├── lncap_ic.png              # IC 时序 + 累积 IC + IC 热力图
├── lncap_turnover.png        # 分层换手率
├── lncap_summary.png         # 综合统计表
├── lncap_report.html         # 全报告 HTML
└── lncap_metrics.json        # 关键指标 JSON（供下游 reports-engine 引用）
```

`lncap_metrics.json` 示例（供 reports-engine 自动引用）：

```json
{
  "factor": "lncap",
  "top_quantile_return": 0.082,
  "bottom_quantile_return": 0.013,
  "long_short_return": 0.069,
  "long_short_sharpe": 1.23,
  "ic_mean": 0.031,
  "ic_ir": 0.45,
  "avg_turnover_top_quantile": 0.18,
  "suggested_verdict": "ACCEPT"
}
```

## 3.4 量化评估维度

| 维度 | 现状 | 目标 | 验证方法 |
|------|------|------|---------|
| 因子分析报告数 | 0 | 4 类 PNG + 1 HTML + 1 JSON | 文件存在性检查 |
| 因子筛选维度 | 5 项 IC 统计量 | 5 项 + 4 类图 + 8 项分层指标 | metrics.json 字段数 |
| 单因子报告生成耗时 | N/A | ≤ 3 秒 | pytest 计时 |
| 报告自动归档 | 无 | archive 自动包含 | archive 目录检查 |

## 3.5 可验证验收标准

1. **CR-1**：`pip install alphalens-reloaded` + `QUANT_ALPHALENS_REPORT=1` 后跑因子引擎，每个因子生成 6 个文件（4 PNG + 1 HTML + 1 JSON）
2. **CR-2**：未安装 alphalens-reloaded 时 try/except 静默跳过，主流程不报错
3. **CR-3**：`QUANT_ALPHALENS_REPORT=0`（默认）时不生成任何额外文件
4. **CR-4**：`lncap_metrics.json` 含 8 个必填字段（top_quantile_return / bottom_quantile_return / long_short_return / long_short_sharpe / ic_mean / ic_ir / avg_turnover_top_quantile / suggested_verdict）
5. **CR-5**：archive 自动包含 `reports/alphalens/<task_id>/` 目录
6. **CR-6**：489 条现有回归测试 100% 通过

## 3.6 兼容层与回滚计划

### 兼容层

- 默认 `QUANT_ALPHALENS_REPORT=0`，行为与 v1.x 完全一致
- 现有 `ic_report.json` 输出保留不变，alphalens 报告为附加产物
- alphalens-reloaded 缺失时静默 fallback，日志 warning

### 回滚计划

| 触发条件 | 回滚动作 | 验证 |
|---------|---------|------|
| alphalens-reloaded 安装失败 | 设置 `QUANT_ALPHALENS_REPORT=0` | 主流程正常 |
| matplotlib 无头模式在 Windows 异常 | 同上 + 日志告警 | 不影响 IC 计算 |
| 报告生成耗时 > 10 秒/因子 | 同上 | 性能测试通过 |
| 报告产物与 reports-engine 集成失败 | 仅生成 PNG，不输出 JSON | 文件存在性检查 |

## 3.7 测试策略

### L2 单元测试

新增 [tests/factor_engine/test_alphalens_adapter.py](../tests/factor_engine/test_alphalens_adapter.py)：
- `test_to_alphalens_format_basic`：数据格式转换正确
- `test_generate_full_report_outputs_6_files`：6 个文件全部生成
- `test_fallback_when_alphalens_missing`：mock ImportError 验证静默跳过
- `test_metrics_json_contains_8_fields`：JSON 字段完整性
- `test_disabled_by_default`：默认环境变量下不生成文件

### L3 集成测试

扩展 [tests/factor_engine/test_factor_analysis.py](../tests/factor_engine/test_factor_analysis.py)：
- 端到端 `QUANT_ALPHALENS_REPORT=1` 跑通完整 pipeline，验证 archive 含报告目录
- 报告引用：reports-engine 能读取 `metrics.json` 并渲染到分析报告

### 测试 marker

新增 `@pytest.mark.requires_alphalens`（缺 alphalens-reloaded 时 skip）

## 3.8 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| alphalens-reloaded 与 pandas 2.0 兼容性问题 | 中 | 高 | CI 双版本 pandas 测试 + 方案 C 备选 |
| matplotlib 无头模式在 Windows 报错 | 中 | 中 | `matplotlib.use("Agg")` + try/except |
| 大数据集（5000 股 × 1000 日）报告生成慢 | 高 | 低 | 分因子并行 + 超时跳过 |
| alphalens 数据丢失率超 25% 报错 | 中 | 中 | `max_loss=0.25` + 日志记录丢失率 |

## 3.9 工作分解（任务清单）

| ID | 任务 | 文件 | 依赖 |
|----|------|------|------|
| T3-1 | 替换 alphalens → alphalens-reloaded 依赖 | `requirements.txt` / `SKILL.md` | - |
| T3-2 | 新增 QUANT_ALPHALENS_REPORT 环境变量 | `SKILL.md` | T3-1 |
| T3-3 | 实现 AlphalensAdapter 数据格式适配 | `scripts/alphalens_adapter.py` | T3-1 |
| T3-4 | 实现报告生成（4 PNG + 1 HTML + 1 JSON） | `scripts/alphalens_adapter.py` | T3-3 |
| T3-5 | 集成到 engine.py ic_analysis 末尾 | `skills/factor-engine/engine.py` | T3-4 |
| T3-6 | 集成到 archive.py 自动归档 | `scripts/archive.py` | T3-5 |
| T3-7 | 集成到 reports-engine 渲染 metrics.json | `skills/reports-engine/scripts/templates/` | T3-4 |
| T3-8 | L2 单元测试（5 项） | `tests/factor_engine/test_alphalens_adapter.py` | T3-4 |
| T3-9 | L3 集成测试扩展 | `tests/factor_engine/test_factor_analysis.py` | T3-6, T3-7 |
| T3-10 | 文档更新（SKILL.md / config_guide.md / api_reference.md） | 3 个 references 文件 | T3-7 |

---

# 附录 A：三方向横向对比

## A.1 实施顺序建议

```
方向三（短期，1-2 周）
  │  低垂果实，立即可见收益
  │  风险可控（fallback 机制完善）
  ▼
方向二（中期，2-3 周）
  │  性能基建，方向三报告生成受益
  │  低侵入，双后端可并行验证
  ▼
方向一（长期，3-4 周）
  │  架构升级，需重构处理流程
  │  建议在方向二/三稳定后启动
  │  与 PRD v1.2 已交付的 P0/P1 阶段不冲突
```

## A.2 共同遵循的工程约束

1. **环境变量**：所有新增环境变量统一 `QUANT_` 前缀（项目硬约束）
2. **测试目录**：所有测试归入 `tests/factor_engine/`，遵循 L1/L2/L3 三级结构
3. **测试覆盖率**：关键路径 100% 行覆盖，非关键路径 ≥ 80%（项目硬约束）
4. **测试 marker**：复用现有 `skill_factor` / `contract` / `unit` / `integration` / `slow`，新增 `requires_polars` / `requires_alphalens`
5. **分支策略**：三个方向各自独立 feature 分支，按 `factor-enhance-3 → factor-enhance-2 → factor-enhance-1` 顺序合并（遵循 P0-1→P0-2 顺序合并约定）
6. **Frozen Core 保护**：三方向均不触碰 PRD v1.2 定义的 6 项 Frozen Core 路径
7. **sha256 Manifest**：所有新增产物（manifest.json / alphalens_report / polars 中间结果）纳入 P1-3 已建立的 artifact_store 覆盖范围

## A.3 三方向联合后的预期收益

| 维度 | 现状 | 三方向完成后 |
|------|------|------------|
| 因子筛选质量 | 5 项 IC 统计量 | 5 项 + 4 类图 + 8 项分层指标 |
| 大数据集性能 | 3-10 秒 | ≤ 1 秒 |
| 工程可维护性 | 5 个散落方法 | 7 个可插拔 Processor + 实验可重放 |
| 实验可重放 | 0% | 100% |
| 因子库扩展成本 | 改 4 处 ≈ 80 行 | 1 个类 ≈ 30 行 |

---

# 附录 B：待用户确认的开放问题

## B.1 方向一相关

| 编号 | 问题 | 默认建议 |
|------|------|---------|
| Q1-1 | Processor 间状态传递用 ctx 还是 parquet 落盘？ | ctx 传引用 + 大数据落盘 |
| Q1-2 | Recorder 是否接入 PRD v1.2 P1-3 的 artifact_store？ | 是，复用 sha256 机制 |
| Q1-3 | pipeline.yaml 放在 skill 目录还是 QUANT_WORK_DIR？ | skill 目录（默认配置）+ work_dir（用户覆盖） |

## B.2 方向二相关

| 编号 | 问题 | 默认建议 |
|------|------|---------|
| Q2-1 | polars 版本下限设 0.20 还是 1.0？ | 0.20（兼容性更广） |
| Q2-2 | 是否对因子计算（不仅是 IC）也提供 polars 后端？ | 暂不，仅 IC/中性化等热路径 |

## B.3 方向三相关

| 编号 | 问题 | 默认建议 |
|------|------|---------|
| Q3-1 | 默认是否开启 alphalens 报告？ | 否，需 `QUANT_ALPHALENS_REPORT=1` 显式启用 |
| Q3-2 | 报告 HTML 是否嵌入 PNG，还是分文件？ | 分文件，便于 reports-engine 单独引用 |
| Q3-3 | 是否集成到 reports-engine 的现有报告模板？ | 是，metrics.json 作为分析要素自动渲染 |

---

**文档结束。请确认三方向方案及附录 B 的开放问题，确认后开始出具对应 PRD 与实际开发。**
