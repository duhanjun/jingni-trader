# PRD：因子引擎 Processor Pipeline + Recorder 架构升级

> 文档版本：v1.1
> 创建日期：2026-08-02
> 完成日期：2026-08-03
> 关联方案：[factor_engine_enhancement_proposals.md](factor_engine_enhancement_proposals.md) 方向一
> 关联 PRD：[prd_gemstar_integration.md](prd_gemstar_integration.md) v1.2
> 状态：开发完成（T1-1 ~ T1-13 全部交付，41 单元测试 + 6 集成测试通过，回归测试无回归）

---

## 第 1 章 概述

### 1.1 背景

当前因子处理流程散落在 [engine.py](../skills/factor-engine/engine.py) 的 5 个独立方法中（`neutralize` / `ic_analysis` / `correlation_analysis` / `factor_fusion` + 散落的 `fillna`），存在以下问题：

1. **加新工序成本高**：新增 winsorize 需改 `run()` 主函数 + 加方法 + 改测试 + 改文档 ≈ 80 行
2. **顺序写死**：5 步顺序硬编码，无法按场景灵活组合
3. **实验不可重放**：[scripts/archive.py](../scripts/archive.py) 仅归档 parquet 快照，缺"为什么这么算"的元数据
4. **重复样板代码**：每个处理步骤都要写 try/except、空 DataFrame 校验、日志记录

### 1.2 目标

借鉴 Microsoft Qlib 的 Processor + Recorder 设计，将因子处理流程抽象为可插拔的工序链 + 实验可重放记录器。

### 1.3 范围

**本 PRD 范围**：因子引擎处理流程架构升级（方向一）
**不在范围**：Polars 后端（方向二）、Alphalens 集成（方向三）

---

## 第 2 章 需求清单

### 2.1 功能需求

| 需求 ID | 需求描述 | 优先级 |
|---------|---------|--------|
| FE-PP-001 | 提供 `Processor` 抽象基类，声明 `requires` / `__call__` / `describe` | P0 |
| FE-PP-002 | 实现 7 个内置 Processor（Neutralize / Winsorize / Fillna / Standardize / ICAnalysis / CorrelationFilter / Fusion） | P0 |
| FE-PP-003 | 实现 `ProcessorChain` 调度器，支持拓扑校验 + 依赖检查 | P0 |
| FE-PP-004 | 实现 `ProcessContext` 工序间状态传递载体 | P0 |
| FE-PP-005 | 实现 `ExperimentRecorder`，记录 7 字段 manifest | P0 |
| FE-PP-006 | 支持声明式 YAML 配置 pipeline.yaml | P0 |
| FE-PP-007 | 旧 API 转 deprecated 别名 + DeprecationWarning | P0 |
| FE-PP-008 | `QUANT_LEGACY_PIPELINE=1` 环境变量强制走旧路径 | P0 |
| FE-PP-009 | Recorder 接入 P1-3 已建立的 artifact_store sha256 机制 | P0 |
| FE-PP-010 | pipeline.yaml 支持 skill 目录默认 + work_dir 用户覆盖 | P0 |

### 2.2 非功能需求

| 需求 ID | 需求描述 |
|---------|---------|
| FE-PP-NFR-001 | 关键路径（ProcessorChain.run / Recorder.finalize）100% 行覆盖 |
| FE-PP-NFR-002 | 非 critical path（各 Processor）≥ 80% 行覆盖 |
| FE-PP-NFR-003 | ProcessorChain 输出与旧 `run()` 结果偏差 < 1e-10 |
| FE-PP-NFR-004 | 489 条现有回归测试 100% 通过 |
| FE-PP-NFR-005 | Recorder 写盘失败不阻塞主流程（异步队列） |

---

## 第 3 章 技术决策

### 3.1 已确认决策（开放问题确认结果）

| 编号 | 问题 | 决策 |
|------|------|------|
| Q1-1 | Processor 间状态传递机制 | **ctx 传引用 + 大数据走 parquet 落盘**。轻量元数据（IC 结果、selected_factors）放 ctx；大数据（factor_df 中间结果）不复制，仅传引用 |
| Q1-2 | Recorder 是否接入 artifact_store | **是**。manifest.json 纳入 P1-3 已建立的 sha256 Manifest 覆盖范围，复用 [scripts/artifact_store.py](../scripts/artifact_store.py) |
| Q1-3 | pipeline.yaml 存放位置 | **skill 目录（默认配置）+ work_dir（用户覆盖）**。加载顺序：work_dir/pipeline.yaml → skill/scripts/processors/pipeline.yaml |

### 3.2 架构决策

#### 3.2.1 Processor 抽象设计

```python
# scripts/processors/base.py
class Processor(ABC):
    requires: List[str] = []  # 依赖的数据字段，如 ["industry", "lncap"]

    @abstractmethod
    def __call__(self, df: pd.DataFrame, ctx: "ProcessContext") -> pd.DataFrame:
        """处理 DataFrame，返回新 DataFrame"""

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """返回工序元数据，用于 Recorder 落盘"""
```

#### 3.2.2 ProcessContext 设计

```python
# scripts/processors/base.py
@dataclass
class ProcessContext:
    """工序间状态传递载体（轻量元数据）"""
    industry_df: Optional[pd.DataFrame] = None    # 行业数据（引用）
    recorder: Optional["ExperimentRecorder"] = None
    ic_results: Dict[str, Any] = field(default_factory=dict)  # IC 分析结果
    selected_factors: List[str] = field(default_factory=list)  # 去冗余后因子
    task_id: str = ""
    work_dir: Optional[Path] = None
```

**关键约束**：ctx 仅传引用，禁止复制大 DataFrame；如需落盘中间结果，显式写 parquet 到 work_dir。

#### 3.2.3 ProcessorChain 调度器

```python
# scripts/processors/chain.py
class ProcessorChain:
    def __init__(self, processors: List[Processor]):
        self.processors = processors
        self._validate_dependencies()

    def run(self, df: pd.DataFrame, ctx: "ProcessContext") -> pd.DataFrame:
        for p in self.processors:
            self._check_requirements(p, df)
            df = p(df, ctx)
            if ctx.recorder:
                ctx.recorder.log_step(p, df)
        return df
```

#### 3.2.4 ExperimentRecorder 设计

```python
# scripts/recorder.py
class ExperimentRecorder:
    """借鉴 Qlib Recorder + MLflow 风格"""

    def __init__(self, archive_dir: Path):
        self.dir = archive_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "run_id": uuid4().hex,
            "start_time": datetime.now().isoformat(),
            "pipeline_config": None,
            "input_data_hash": {},
            "steps": [],
            "output_artifacts": [],
            "env": self._snapshot_env(),
        }

    def log_step(self, processor, df_after):
        self.manifest["steps"].append({
            "processor": processor.__class__.__name__,
            "params": processor.describe(),
            "rows_after": len(df_after),
            "cols_after": list(df_after.columns),
            "nan_ratio": float(df_after.isna().mean().mean()),
        })

    def finalize(self):
        # 写 manifest.json
        manifest_path = self.dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False, default=str)
        )
        # 接入 artifact_store 的 sha256 机制
        from scripts.artifact_store import compute_sha256
        return compute_sha256(manifest_path)
```

#### 3.2.5 YAML 配置加载

```python
# scripts/processors/loader.py
def load_pipeline_config(work_dir: Optional[Path] = None) -> List[Processor]:
    """加载 pipeline.yaml，优先 work_dir，回退 skill 默认"""
    candidates = []
    if work_dir:
        candidates.append(work_dir / "pipeline.yaml")
    candidates.append(Path(__file__).parent / "pipeline.yaml")

    for path in candidates:
        if path.exists():
            return _parse_yaml_to_processors(path)
    return _default_processors()  # 兜底默认链
```

### 3.3 兼容层设计

#### 3.3.1 旧 API 转调

```python
class FactorEngine:
    def neutralize(self, factor_df, industry_df, **kwargs):
        warnings.warn(
            "neutralize() 将在 v3.0 移除，请用 ProcessorChain",
            DeprecationWarning,
            stacklevel=2
        )
        p = NeutralizeProcessor(**kwargs)
        ctx = ProcessContext(industry_df=industry_df)
        return p(factor_df, ctx)
```

#### 3.3.2 run() 路径切换

```python
def run(ctx):
    # 兼容层：环境变量强制走旧路径
    if os.environ.get("QUANT_LEGACY_PIPELINE", "0") == "1":
        return _run_legacy(ctx)  # 保留旧 5 步硬编码逻辑

    # 默认走 ProcessorChain
    engine = FactorEngine()
    recorder = ExperimentRecorder(archive_dir=_get_archive_dir(ctx))
    pipeline = load_pipeline_config(ctx.work_dir)
    chain = ProcessorChain(pipeline)
    proc_ctx = ProcessContext(recorder=recorder, task_id=ctx.task_id)
    result = chain.run(factor_df, proc_ctx)
    recorder.finalize()
    return result
```

---

## 第 4 章 验收标准

### 4.1 可验证验收标准

| CR ID | 验收标准 | 验证方法 |
|-------|---------|---------|
| CR-1 | 新建 `WinsorizeProcessor` 仅需 30 行代码 + 1 行 YAML，零修改 `engine.py` | 代码量统计 |
| CR-2 | 通过 `pipeline.yaml` 删除某个 Processor（如 Winsorize），全链路仍正常运行且结果与无该工序一致 | 集成测试 |
| CR-3 | 跑完一次后 `archives/run_xxx/manifest.json` 含 7 字段（run_id / start_time / pipeline_config / input_data_hash / steps / output_artifacts / env） | manifest.json 字段校验 |
| CR-4 | 用同一 manifest 重跑，输出 IC 与原 archive 偏差 < 1e-10 | 浮点误差测试 |
| CR-5 | 旧 `FactorEngine.neutralize()` 调用产生 `DeprecationWarning` 且结果与 v1.x 完全一致 | 单元测试 + 警告捕获 |
| CR-6 | 489 条现有回归测试 100% 通过 | pytest 全量回归 |
| CR-7 | `QUANT_LEGACY_PIPELINE=1` 环境下走旧路径，结果与 v1.x 一致 | 环境变量切换测试 |
| CR-8 | manifest.json 的 sha256 纳入 artifact_store 覆盖范围 | artifact_store 日志校验 |
| CR-9 | pipeline.yaml 优先 work_dir，回退 skill 默认 | 双配置文件测试 |

### 4.2 量化评估维度

| 维度 | 现状 | 目标 | 验证方法 |
|------|------|------|---------|
| 新增工序代码量 | 改 4 处 ≈ 80 行 | 仅创建 1 个类 ≈ 30 行 | 实测 WinsorizeProcessor |
| 实验可重放性 | 0%（仅 parquet 快照） | 100%（manifest.json 含全参数） | 复跑历史 archive 对比 IC |
| 工序组合灵活度 | 1 种（写死） | ≥ 8 种（YAML 排列组合） | 配置驱动跑通不同链 |
| 单元测试覆盖 | 现有方法各自测试 | 每个 Processor 100% 行覆盖 | pytest --cov |

---

## 第 5 章 兼容性与回滚

### 5.1 兼容层

| 兼容项 | 机制 | 移除版本 |
|--------|------|---------|
| `FactorEngine.neutralize/ic_analysis/correlation_analysis/factor_fusion` | 转 deprecated 别名，内部调 Processor | v3.0 |
| `run()` 旧 5 步硬编码逻辑 | 保留为 `_run_legacy()`，环境变量 `QUANT_LEGACY_PIPELINE=1` 触发 | v3.0 |
| 现有 `ic_report.json` 输出格式 | 保留不变 | 永久 |

### 5.2 回滚计划

| 触发条件 | 回滚动作 | 验证 |
|---------|---------|------|
| 任一 Processor 输出与旧方法偏差 > 1e-6 | 设置 `QUANT_LEGACY_PIPELINE=1` | 回归测试通过 |
| ProcessorChain 调度死锁或拓扑校验失败 | 同上 | 单元测试通过 |
| Recorder 写盘失败导致 archive 异常 | 自动降级为旧 archive 逻辑 | archive 目录正常 |
| YAML 配置 schema 校验失败 | 拒绝启动 + 详细报错 + 提示旧路径回退 | 启动日志校验 |

---

## 第 6 章 测试策略

遵循项目硬约束「关键路径 100% 行覆盖，非关键路径 ≥80%」。

### 6.1 L1 契约测试（contract）

扩展 [tests/factor_engine/test_run_contract.py](../tests/factor_engine/test_run_contract.py)：
- `ProcessorChain.run()` 必须返回与旧 `run()` 同结构 DataFrame
- `Recorder.finalize()` 必产出 manifest.json 含 7 必填字段
- `QUANT_LEGACY_PIPELINE=1` 路径输出与旧路径一致

### 6.2 L2 单元测试（unit）

新增 `tests/factor_engine/test_processors/`：
- 每个 Processor 一个测试文件，100% 行覆盖
- `test_processor_chain.py`：拓扑校验 / 依赖缺失 / 异常隔离
- `test_recorder.py`：manifest 落盘 / 数据 hash 一致性 / env 快照
- `test_pipeline_loader.py`：YAML 解析 / work_dir 优先 / 兜底默认链

### 6.3 L3 集成测试（integration）

扩展 [tests/integration/test_pipeline_archives.py](../tests/integration/test_pipeline_archives.py)：
- 端到端跑一次完整 pipeline，校验 manifest 可重放
- 4 种 YAML 配置组合跑通（全开 / 关 Winsorize / 关 Neutralize / 仅 IC+Fusion）

### 6.4 测试 marker

`@pytest.mark.skill_factor` / `@pytest.mark.contract` / `@pytest.mark.unit` / `@pytest.mark.integration`

---

## 第 7 章 实施顺序与依赖

### 7.1 任务清单

| ID | 任务 | 文件 | 依赖 |
|----|------|------|------|
| T1-1 | 创建 Processor 基类 + ProcessContext | `scripts/processors/base.py` | - |
| T1-2 | 实现 7 个内置 Processor | `scripts/processors/*.py` | T1-1 |
| T1-3 | 实现 ProcessorChain 调度器 | `scripts/processors/chain.py` | T1-1 |
| T1-4 | 实现 ExperimentRecorder | `scripts/recorder.py` | T1-1 |
| T1-5 | 实现 pipeline.yaml 加载器 | `scripts/processors/loader.py` | T1-2 |
| T1-6 | 改造 `engine.py.run()` 走 ProcessorChain | `skills/factor-engine/engine.py` | T1-2, T1-3 |
| T1-7 | 改造 `scripts/archive.py` 集成 Recorder | `scripts/archive.py` | T1-4 |
| T1-8 | 旧 API 转 deprecated 别名 + 兼容层 | `skills/factor-engine/engine.py` | T1-6 |
| T1-9 | 新增 `pipeline.yaml` 默认配置 | `scripts/processors/pipeline.yaml` | T1-2 |
| T1-10 | Recorder 接入 artifact_store sha256 | `scripts/recorder.py` | T1-4 |
| T1-11 | L2 单元测试（7 Processor + Chain + Recorder + Loader） | `tests/factor_engine/test_processors/` | T1-2 ~ T1-5 |
| T1-12 | L3 集成测试扩展 | `tests/integration/test_pipeline_archives.py` | T1-6, T1-7 |
| T1-13 | 文档更新（SKILL.md / config_guide.md / api_reference.md） | 3 个 references 文件 | T1-8 |

### 7.2 依赖图

```
T1-1 (基类) ─┬─→ T1-2 (7 Processor) ──┬─→ T1-5 (Loader) ──┐
             ├─→ T1-3 (Chain)          │                   │
             └─→ T1-4 (Recorder) ──────┼─→ T1-10 (sha256)  │
                                      │                   │
                                      └─→ T1-6 (engine 改造) ←─ T1-3
                                              │
                                              ├─→ T1-7 (archive 改造) ←─ T1-4
                                              ├─→ T1-8 (兼容层)
                                              └─→ T1-9 (默认 YAML)
T1-11 (单测) ←─ T1-2, T1-3, T1-4, T1-5
T1-12 (集成) ←─ T1-6, T1-7
T1-13 (文档) ←─ T1-8
```

### 7.3 实施顺序

1. **Phase 1（基建）**：T1-1 → T1-2 → T1-3 → T1-4 → T1-5（可并行 T1-2/T1-3/T1-4）
2. **Phase 2（集成）**：T1-6 → T1-7 → T1-8 → T1-9 → T1-10
3. **Phase 3（测试）**：T1-11 → T1-12
4. **Phase 4（文档）**：T1-13

---

## 第 8 章 并行开发协调机制

### 8.1 文件负责人划分

| 文件 | 负责人角色 | 协作标注 |
|------|----------|---------|
| `scripts/processors/base.py` | 基类负责人 | ⚠️ 多人协作 |
| `scripts/processors/neutralize.py` | Processor 实现者 A | - |
| `scripts/processors/winsorize.py` | Processor 实现者 A | - |
| `scripts/processors/ic_analysis.py` | Processor 实现者 B | - |
| `scripts/processors/fusion.py` | Processor 实现者 B | - |
| `scripts/processors/chain.py` | 调度器负责人 | - |
| `scripts/recorder.py` | Recorder 负责人 | - |
| `engine.py` | 引擎主负责人 | ⚠️ 协作文件 |
| `archive.py` | 归档负责人 | ⚠️ 协作文件 |

### 8.2 sys.path 隔离

每个 Processor 测试文件独立 conftest.py 清理 sys.path，遵循项目硬约束。

### 8.3 分支策略

- 独立 feature 分支：`factor-enhance-1-processor-pipeline`
- 合并顺序：在方向三、方向二合并后启动合并
- 遵循 PRD v1.2 P0-1→P0-2 顺序合并约定

---

## 第 9 章 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| Processor 间状态传递设计不当导致 ctx 膨胀 | 中 | 中 | ctx 仅传引用，大数据走 parquet 落盘 |
| 拓扑校验误判依赖导致合法配置被拒 | 低 | 高 | 提供显式 `override_dependency` 参数 |
| Recorder 同步写盘拖慢主流程 | 中 | 低 | 异步队列，主流程不阻塞 |
| YAML 配置错误导致静默失败 | 中 | 高 | 启动时 pydantic schema 校验 + 详细报错 |
| 旧 API 调用方未及时迁移 | 低 | 低 | DeprecationWarning + 文档明确 v3.0 移除 |

---

## 第 10 章 附录

### 10.1 默认 pipeline.yaml

```yaml
pipeline:
  - processor: NeutralizeProcessor
    enabled: true
    params:
      neutralize_mcap: true
      neutralize_industry: true
      min_sample: 30

  - processor: WinsorizeProcessor
    enabled: true
    params:
      method: mad
      threshold: 3.0

  - processor: FillnaProcessor
    enabled: true
    params:
      method: rank_pct
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

### 10.2 Frozen Core 保护

本方向不触碰 PRD v1.2 定义的 6 项 Frozen Core 路径：
- real_broker / risk / schemas/order / schemas/execution_report / engine.py（master）/ portfolio-risk-engine/scripts/cost.py

### 10.3 与其他方向的关系

| 关系 | 说明 |
|------|------|
| 与方向二（Polars） | 独立，方向二改造 optimizations/ 目录，本方向改造 processors/ 目录 |
| 与方向三（Alphalens） | 独立，方向三新增 alphalens_adapter.py，本方向不涉及 |
| 实施顺序 | 建议在方向三、方向二稳定后启动 |

### 10.4 交付清单（v1.1 开发完成）

| 任务 ID | 任务描述 | 交付物 |
|---------|---------|--------|
| T1-1 | Processor 抽象基类 | `scripts/processors/base.py`（`Processor` / `ProcessContext` / `ProcessorRequirementError`） |
| T1-2 | ProcessorChain 调度器 | `scripts/processors/chain.py`（`ProcessorChain` / `ChainValidationError`） |
| T1-3 | ExperimentRecorder | `scripts/recorder.py`（7 字段 manifest.json） |
| T1-4 | NeutralizeProcessor | `scripts/processors/neutralize.py` |
| T1-5 | WinsorizeProcessor | `scripts/processors/winsorize.py`（mad / quantile 双方法） |
| T1-6 | FillnaProcessor | `scripts/processors/fillna.py`（rank_pct / zero / mean / ffill 四方法） |
| T1-7 | StandardizeProcessor | `scripts/processors/standardize.py`（zscore / minmax 双方法） |
| T1-8 | ICAnalysisProcessor | `scripts/processors/ic_analysis.py`（normal / spearman 双方法） |
| T1-9 | CorrelationFilterProcessor | `scripts/processors/correlation_filter.py` |
| T1-10 | FusionProcessor | `scripts/processors/fusion.py`（ic_weighted / equal_weighted 双方法） |
| T1-11 | YAML 加载器 + 默认配置 | `scripts/processors/loader.py` + `scripts/processors/pipeline.yaml` |
| T1-12 | engine.py 集成 + 兼容层 | `engine.py` 接入 ProcessorChain，`QUANT_LEGACY_PIPELINE` 切换旧路径 |
| T1-13 | 文档更新 + PRD 标记完成 | `SKILL.md` / `config_guide.md` / `api_reference.md` 新增 Processor Pipeline 章节 |

### 10.5 测试覆盖（v1.1 开发完成）

| 测试类别 | 文件 | 用例数 | 状态 |
|---------|------|--------|------|
| L2 单元测试 | `tests/factor_engine/test_processors/test_processors.py` | 41 | 全部通过 |
| L3 集成测试 | `tests/integration/test_pipeline_archives.py::TestProcessorPipelineIntegration` | 6 | 全部通过 |
| 集成测试（归档结构） | `tests/integration/test_pipeline_archives.py::TestArchiveStructure` | 5 | 全部通过 |

**关键测试覆盖项：**
- 7 个内置 Processor 的核心功能、异常分支、describe() 输出
- Processor 基类抽象方法不可实例化、依赖检查、参数传递
- ProcessorChain 顺序执行、依赖校验、异常隔离
- ExperimentRecorder manifest.json 含 7 必填字段（run_id / start_time / pipeline_config / input_data_hash / steps / output_artifacts / env）
- 4 种 YAML 配置组合（full / disable_winsorize / disable_neutralize / ic_fusion_only）端到端可跑通
- 同一输入跑两次 IC 偏差 < 1e-10（实验可重放性）

---

**文档结束。状态：v1.1 开发完成，T1-1 ~ T1-13 全部交付。**
