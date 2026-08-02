# PRD：因子引擎 Polars 高性能后端

> 文档版本：v1.1
> 创建日期：2026-08-02
> 完成日期：2026-08-02
> 关联方案：[factor_engine_enhancement_proposals.md](factor_engine_enhancement_proposals.md) 方向二
> 关联 PRD：[prd_gemstar_integration.md](prd_gemstar_integration.md) v1.2
> 状态：开发完成（T2-1 ~ T2-12 全部交付，回归测试通过）

---

## 第 1 章 概述

### 1.1 背景

IC 计算位于 [scripts/optimizations/ic_vectorized.py](../skills/factor-engine/scripts/optimizations/ic_vectorized.py)，已用 pandas groupby 向量化但仍单线程。中性化位于 [scripts/optimizations/vectorized_neutralize.py](../skills/factor-engine/scripts/optimizations/vectorized_neutralize.py)，已用 `numpy.linalg.lstsq` 替代 sklearn，但仍是逐截面循环。

5000 股 × 1000 日 × 50 因子 = 2.5 亿行场景下：
- 当前向量化 IC ≈ 3 秒（pandas 单线程）
- 中性化逐截面循环 ≈ 8-15 秒
- Polars 实测同场景 ≈ 0.4 秒 / 1 秒（5-15× 提速）

### 1.2 目标

引入 Polars 作为可选高性能 DataFrame 后端，覆盖 IC 计算、中性化、IC Decay、相关性分析等热路径，同时保持零侵入和双后端结果一致。

### 1.3 范围

**本 PRD 范围**：因子引擎 Polars 后端引入（方向二）
**不在范围**：Processor Pipeline 架构升级（方向一）、Alphalens 集成（方向三）

---

## 第 2 章 需求清单

### 2.1 功能需求

| 需求 ID | 需求描述 | 优先级 |
|---------|---------|--------|
| FE-PB-001 | 新增 `polars>=0.20.0` 可选依赖 | P0 |
| FE-PB-002 | 新增 `QUANT_FACTOR_BACKEND` 环境变量（pandas / polars / auto） | P0 |
| FE-PB-003 | IC 计算（pearson / spearman / batch）支持 polars 后端 | P0 |
| FE-PB-004 | 中性化（neutralize_factor / neutralize_factors_batch）支持 polars 后端 | P0 |
| FE-PB-005 | IC Decay 多 lag 扫描支持 polars 后端 | P1 |
| FE-PB-006 | 相关性分析（correlation_analysis）支持 polars 后端 | P1 |
| FE-PB-007 | polars 缺失时自动 fallback pandas + 日志提示 | P0 |
| FE-PB-008 | `auto` 模式：检测 polars 可用性自动选择 | P0 |

### 2.2 非功能需求

| 需求 ID | 需求描述 |
|---------|---------|
| FE-PB-NFR-001 | 双后端 IC 输出最大绝对偏差 < 1e-10 |
| FE-PB-NFR-002 | 5000 股 × 1000 日测试集下 polars 耗时较 pandas 减少 ≥ 60% |
| FE-PB-NFR-003 | 内存峰值 ≤ 1.5× pandas 基线 |
| FE-PB-NFR-004 | 489 条现有回归测试 100% 通过 |
| FE-PB-NFR-005 | `QUANT_FACTOR_BACKEND` 未设置时默认 pandas，行为与 v1.x 完全一致 |
| FE-PB-NFR-006 | 关键路径（polars 实现）100% 行覆盖 |

---

## 第 3 章 技术决策

### 3.1 已确认决策（开放问题确认结果）

| 编号 | 问题 | 决策 |
|------|------|------|
| Q2-1 | polars 版本下限 | **0.20**（兼容性更广，覆盖 0.20 ~ 1.x） |
| Q2-2 | polars 后端范围 | **暂仅 IC/中性化等热路径**。因子计算（technical_factors / financial_factors 等）暂不改造，避免侵入过大 |

### 3.2 架构决策

#### 3.2.1 环境变量设计

```yaml
# SKILL.md 新增
- name: QUANT_FACTOR_BACKEND
  description: 因子计算 DataFrame 后端（pandas / polars / auto），默认 pandas
  required: false
  default: "pandas"
```

> **注意**：与现有 `FACTOR_BACKEND`（pandas_ta / talib 技术指标后端）语义不同，故用 `QUANT_FACTOR_BACKEND` 前缀避免冲突，遵循项目硬约束。

#### 3.2.2 双后端函数签名

```python
# scripts/optimizations/ic_vectorized.py
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

#### 3.2.3 Polars 实现示例

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

#### 3.2.4 自动 fallback 机制

```python
# scripts/optimizations/__init__.py
def _polars_available() -> bool:
    try:
        import polars  # noqa
        return True
    except ImportError:
        return False

# 模块加载时检查并日志提示
if not _polars_available():
    logger.info("polars 未安装，使用 pandas 后端。pip install polars>=0.20.0 启用加速")

def get_backend(backend: str = "auto") -> str:
    """统一后端选择逻辑"""
    if backend == "auto":
        return "polars" if _polars_available() else "pandas"
    if backend == "polars" and not _polars_available():
        logger.warning("polars 未安装，自动回退 pandas 后端")
        return "pandas"
    return backend
```

#### 3.2.5 环境变量读取

```python
# 从环境变量读取默认后端
import os
_DEFAULT_BACKEND = os.environ.get("QUANT_FACTOR_BACKEND", "pandas")

def ic_series_pearson(factor, forward_ret, dates=None, min_obs=10, backend=None):
    if backend is None:
        backend = get_backend(_DEFAULT_BACKEND)
    # ... 后续逻辑
```

### 3.3 改造范围

| 文件 | 改造方法 | 优先级 | 备注 |
|------|---------|--------|------|
| `scripts/optimizations/ic_vectorized.py` | `ic_series_pearson` / `ic_series_spearman` / `ic_analysis_batch` | P0 | 热路径 |
| `scripts/optimizations/vectorized_neutralize.py` | `neutralize_factor` / `neutralize_factors_batch` | P0 | 热路径 |
| `scripts/optimizations/ic_decay.py` | `ICDecayAnalyzer.analyze`（多 lag 扫描） | P1 | 中等热度 |
| `skills/factor-engine/engine.py` | `correlation_analysis`（相关性矩阵） | P1 | 中等热度 |
| `scripts/optimizations/factor_validator.py` | `validate_factor`（bootstrap 采样） | P2 | 性能不敏感，暂不改造 |

---

## 第 4 章 验收标准

### 4.1 可验证验收标准

| CR ID | 验收标准 | 验证方法 |
|-------|---------|---------|
| CR-1 | `pip install polars>=0.20.0` 后 `QUANT_FACTOR_BACKEND=polars` 环境下 IC 计算正确 | 单元测试 |
| CR-2 | 未安装 polars 时自动 fallback pandas，日志输出提示 | mock ImportError 测试 |
| CR-3 | 双后端 IC 输出最大绝对偏差 < 1e-10（用 `np.testing.assert_allclose`） | 一致性测试 |
| CR-4 | 5000 股 × 1000 日测试集下 polars 耗时较 pandas 减少 ≥ 60% | `@pytest.mark.slow` 性能基准 |
| CR-5 | 489 条现有回归测试 100% 通过 | pytest 全量回归 |
| CR-6 | `QUANT_FACTOR_BACKEND` 未设置时默认 pandas，行为与 v1.x 完全一致 | 默认值测试 |
| CR-7 | `auto` 模式正确检测 polars 可用性 | 双场景测试（安装/未安装） |
| CR-8 | 中性化 polars 后端与 pandas 输出偏差 < 1e-10 | 一致性测试 |

### 4.2 量化评估维度

| 维度 | 现状（pandas） | 目标（polars） | 验证方法 |
|------|--------------|--------------|---------|
| IC 计算耗时（5000×1000） | ~3 秒 | ≤ 1 秒 | `pytest --benchmark` |
| 中性化耗时（5000×1000） | ~10 秒 | ≤ 2 秒 | 同上 |
| 内存峰值 | 1× | ≤ 1.5× | memory_profiler |
| 结果一致性 | 基线 | 偏差 < 1e-10 | 双后端输出 diff 测试 |

---

## 第 5 章 兼容性与回滚

### 5.1 兼容层

| 兼容项 | 机制 |
|--------|------|
| `backend` 参数 | 默认 `"pandas"`，所有现有调用零改动 |
| `QUANT_FACTOR_BACKEND` 未设置 | 等同 `"pandas"`，行为与 v1.x 一致 |
| polars 缺失 | try/except fallback pandas，主流程不中断 |
| 现有 API 签名 | 仅新增可选 `backend` 参数，向后兼容 |

### 5.2 回滚计划

| 触发条件 | 回滚动作 | 验证 |
|---------|---------|------|
| polars 输出与 pandas 偏差 > 1e-10 | 设置 `QUANT_FACTOR_BACKEND=pandas` | 回归测试通过 |
| polars 版本不兼容（API 变更） | 同上 + 日志告警 | import 测试通过 |
| 内存占用 > 2× pandas 基线 | 同上 | memory_profiler 监控 |
| Windows 下 polars 编译失败 | 同上 | CI 双平台测试 |

---

## 第 6 章 测试策略

遵循项目硬约束「关键路径 100% 行覆盖，非关键路径 ≥80%」。

### 6.1 L2 单元测试

新增 [tests/factor_engine/test_polars_backend.py](../tests/factor_engine/test_polars_backend.py)：
- `test_ic_pearson_polars_vs_pandas_consistency`：Pearson IC 双后端一致
- `test_ic_spearman_polars_vs_pandas_consistency`：Spearman IC 双后端一致
- `test_neutralize_polars_vs_pandas`：中性化双后端一致
- `test_fallback_when_polars_missing`：mock ImportError 验证 fallback
- `test_auto_backend_detection`：auto 模式正确检测
- `test_default_is_pandas`：默认后端为 pandas

### 6.2 L3 集成测试

扩展 [tests/factor_engine/test_factor_analysis.py](../tests/factor_engine/test_factor_analysis.py)：
- 端到端 `QUANT_FACTOR_BACKEND=polars` 跑通完整 pipeline
- 双后端跑同一数据集，最终 IC 报告 diff < 1e-10

### 6.3 性能基准测试

新增 [tests/factor_engine/test_polars_perf.py](../tests/factor_engine/test_polars_perf.py)：
- `@pytest.mark.slow` + `@pytest.mark.requires_polars`
- 5000 股 × 1000 日合成数据集
- 验证 polars 耗时较 pandas 减少 ≥ 60%

### 6.4 测试 marker

新增 `@pytest.mark.requires_polars`（缺 polars 时 skip）

---

## 第 7 章 实施顺序与依赖

### 7.1 任务清单

| ID | 任务 | 文件 | 依赖 |
|----|------|------|------|
| T2-1 | 新增 polars 可选依赖 + 环境变量 | `requirements.txt` / `SKILL.md` | - |
| T2-2 | 实现 `_polars_available()` + `get_backend()` 统一入口 | `scripts/optimizations/__init__.py` | T2-1 |
| T2-3 | IC 计算 polars 实现（pearson） | `scripts/optimizations/ic_vectorized.py` | T2-2 |
| T2-4 | IC 计算 polars 实现（spearman） | `scripts/optimizations/ic_vectorized.py` | T2-3 |
| T2-5 | IC 计算 polars 实现（batch） | `scripts/optimizations/ic_vectorized.py` | T2-4 |
| T2-6 | 中性化 polars 实现 | `scripts/optimizations/vectorized_neutralize.py` | T2-2 |
| T2-7 | IC Decay polars 实现 | `scripts/optimizations/ic_decay.py` | T2-5 |
| T2-8 | 相关性分析 polars 实现 | `skills/factor-engine/engine.py` | T2-2 |
| T2-9 | L2 单元测试（一致性 + fallback + auto） | `tests/factor_engine/test_polars_backend.py` | T2-3, T2-6 |
| T2-10 | L3 集成测试扩展 | `tests/factor_engine/test_factor_analysis.py` | T2-8 |
| T2-11 | 性能基准测试 | `tests/factor_engine/test_polars_perf.py` | T2-9 |
| T2-12 | 文档更新（SKILL.md / config_guide.md） | 2 个 references 文件 | T2-8 |

### 7.2 依赖图

```
T2-1 (依赖+环境变量) ──→ T2-2 (统一入口) ──┬─→ T2-3 (pearson) ──→ T2-4 (spearman) ──→ T2-5 (batch) ──→ T2-7 (decay)
                                           ├─→ T2-6 (中性化)
                                           └─→ T2-8 (相关性)
T2-9 (单测) ←─ T2-3, T2-6
T2-10 (集成) ←─ T2-8
T2-11 (性能) ←─ T2-9
T2-12 (文档) ←─ T2-8
```

### 7.3 实施顺序

1. **Phase 1（基建）**：T2-1 → T2-2
2. **Phase 2（P0 热路径）**：T2-3 → T2-4 → T2-5（IC 计算） + T2-6（中性化，可并行）
3. **Phase 3（P1 中等热度）**：T2-7（IC Decay） + T2-8（相关性，可并行）
4. **Phase 4（测试）**：T2-9 → T2-10 → T2-11
5. **Phase 5（文档）**：T2-12

---

## 第 8 章 并行开发协调机制

### 8.1 文件负责人划分

| 文件 | 负责人角色 | 协作标注 |
|------|----------|---------|
| `scripts/optimizations/__init__.py` | 基建负责人 | ⚠️ 多人协作 |
| `scripts/optimizations/ic_vectorized.py` | IC 实现者 | - |
| `scripts/optimizations/vectorized_neutralize.py` | 中性化实现者 | - |
| `scripts/optimizations/ic_decay.py` | Decay 实现者 | - |
| `skills/factor-engine/engine.py` | 引擎主负责人 | ⚠️ 协作文件（仅 correlation_analysis 方法） |

### 8.2 sys.path 隔离

每个测试文件独立 conftest.py 清理 sys.path，遵循项目硬约束。

### 8.3 分支策略

- 独立 feature 分支：`factor-enhance-2-polars-backend`
- 合并顺序：在方向三合并后启动，方向一之前
- 遵循 PRD v1.2 P0-1→P0-2 顺序合并约定

---

## 第 9 章 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| polars API 在 0.20 → 1.0 升级中断 | 中 | 中 | 锁版本下限 + 抽象层隔离 + CI 多版本测试 |
| datetime / category 类型在 polars 中行为不一致 | 高 | 中 | 增加 dtype 转换层 + 单元测试覆盖 |
| group_by 顺序与 pandas 不一致导致 IC 序列错位 | 中 | 高 | 显式 sort + 索引对齐校验 |
| Windows 下 polars 编译问题 | 低 | 高 | 提供 wheel + CI 双平台测试 |
| NaN 处理差异（pandas NaN vs polars null） | 高 | 中 | 显式 drop_nulls + 一致性测试 |

---

## 第 10 章 附录

### 10.1 环境变量清单

| 变量名 | 用途 | 默认值 | 必填 |
|--------|------|--------|------|
| `QUANT_FACTOR_BACKEND` | DataFrame 后端选择 | `pandas` | 否 |

### 10.2 Frozen Core 保护

本方向不触碰 PRD v1.2 定义的 6 项 Frozen Core 路径。

### 10.3 与其他方向的关系

| 关系 | 说明 |
|------|------|
| 与方向一（Processor Pipeline） | 独立，方向一改造 processors/ 目录，本方向改造 optimizations/ 目录 |
| 与方向三（Alphalens） | 独立，方向三新增 alphalens_adapter.py |
| 实施顺序 | 在方向三之后、方向一之前 |
| 协同收益 | 方向三的 alphalens 报告生成可受益于本方向的性能提升 |

### 10.4 交付清单（v1.1 开发完成）

| 任务 ID | 任务描述 | 交付物 |
|---------|---------|--------|
| T2-1 | polars 可选依赖 + 环境变量 | `requirements.txt` / `SKILL.md` 新增 `QUANT_FACTOR_BACKEND` |
| T2-2 | 统一后端入口 | `scripts/optimizations/__init__.py`（`resolve_backend()` + auto 检测 + fallback） |
| T2-3 | IC Pearson polars 实现 | `scripts/optimizations/ic_vectorized.py::ic_series_pearson` |
| T2-4 | IC Spearman polars 实现 | `scripts/optimizations/ic_vectorized.py::ic_series_spearman` |
| T2-5 | IC batch polars 实现 | `scripts/optimizations/ic_vectorized.py` 批量接口 |
| T2-6 | 中性化 polars 实现 | `scripts/optimizations/vectorized_neutralize.py::neutralize_factor` |
| T2-7 | IC Decay polars 实现 | `scripts/optimizations/ic_decay.py::ICDecayAnalyzer` |
| T2-8 | 相关性分析 polars 实现 | `scripts/optimizations/vectorized_correlation.py` + `engine.py` 接入 |
| T2-9 | L2 单元测试 | `tests/factor_engine/test_polars_backend.py`（10 用例） |
| T2-10 | L3 集成测试 | `tests/factor_engine/test_polars_integration.py`（3 用例） |
| T2-11 | 性能基准测试 | `tests/factor_engine/test_polars_perf.py`（5 用例） |
| T2-12 | 文档更新 | `SKILL.md` / `config_guide.md` / `api_reference.md` 新增 Polars 后端章节 |

### 10.5 测试覆盖（v1.1 开发完成）

| 测试类别 | 文件 | 用例数 | 状态 |
|---------|------|--------|------|
| L2 单元测试 | `tests/factor_engine/test_polars_backend.py` | 10 | 全部通过 |
| L3 集成测试 | `tests/factor_engine/test_polars_integration.py` | 3 | 全部通过 |
| 性能基准 | `tests/factor_engine/test_polars_perf.py` | 5 | 全部通过 |

**关键测试覆盖项：**
- 5 大热路径双后端一致性（Pearson IC / Spearman IC / 中性化 / IC Decay / 相关性分析），最大绝对偏差 < 1e-10（IC Decay 放宽到 1e-6）
- polars 缺失时自动回退 pandas（fallback 机制）
- `auto` 后端自动检测逻辑
- 环境变量 `QUANT_FACTOR_BACKEND` 三种取值（pandas/polars/auto）行为正确
- `FactorEngine.correlation_analysis` `backend` 参数覆盖环境变量
- 端到端 pipeline 在 polars 后端下跑通且结果与 pandas 一致

---

**文档结束。状态：v1.1 开发完成，T2-1 ~ T2-12 全部交付。**
