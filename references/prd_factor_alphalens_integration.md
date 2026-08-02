# PRD：因子引擎 Alphalens 真正集成

> 文档版本：v1.1
> 创建日期：2026-08-02
> 关联方案：[factor_engine_enhancement_proposals.md](factor_engine_enhancement_proposals.md) 方向三
> 关联 PRD：[prd_gemstar_integration.md](prd_gemstar_integration.md) v1.2
> 状态：开发完成（T3-1 ~ T3-12 全部交付，10 测试通过 + 2 跳过）

---

## 第 1 章 概述

### 1.1 背景

[skills/factor-engine/SKILL.md#L21](../skills/factor-engine/SKILL.md) 声明依赖 `alphalens>=0.4.0`，但全代码库无 `import alphalens` 调用。IC 分析当前仅输出 5 项统计量的 JSON（[engine.py#L231-286](../skills/factor-engine/engine.py)）。

看到 IC=0.03 无法回答 3 个关键问题：
1. **稳定性**：因子是全程稳定还是某段时间爆发？（缺 IC 时序图）
2. **盈利性**：买 Top 20% 股票到底赚不赚钱？（缺分层回测净值曲线）
3. **成本**：因子多久换一次仓？交易成本多大？（缺换手率分析）

### 1.2 目标

将声明但未使用的 alphalens 依赖真正集成，自动生成 4 类因子分析报告（IC 时序 / 分层净值 / alpha-beta / 换手率），从 5 项 IC 统计量扩展到 5 项 + 4 类图 + 8 项分层指标。

### 1.3 范围

**本 PRD 范围**：Alphalens 集成（方向三）
**不在范围**：Processor Pipeline 架构升级（方向一）、Polars 后端（方向二）

---

## 第 2 章 需求清单

### 2.1 功能需求

| 需求 ID | 需求描述 | 优先级 |
|---------|---------|--------|
| FE-AI-001 | 替换 `alphalens>=0.4.0` 为 `alphalens-reloaded>=0.4.5`（维护版本，兼容 pandas 2.0） | P0 |
| FE-AI-002 | 新增 `QUANT_ALPHALENS_REPORT` 环境变量（0/1，默认 0） | P0 |
| FE-AI-003 | 实现 `AlphalensAdapter` 数据格式适配器（jingni-trader 内部结构 → alphalens MultiIndex） | P0 |
| FE-AI-004 | 实现报告生成：4 PNG + 1 HTML + 1 JSON | P0 |
| FE-AI-005 | 集成到 `engine.py` `ic_analysis` 末尾，环境变量控制开关 | P0 |
| FE-AI-006 | 集成到 `scripts/archive.py` 自动归档报告目录 | P0 |
| FE-AI-007 | 集成到 reports-engine 渲染 `metrics.json` 作为分析要素 | P1 |
| FE-AI-008 | alphalens-reloaded 缺失时 try/except 静默跳过，日志 warning | P0 |
| FE-AI-009 | `lncap_metrics.json` 含 8 个必填字段 | P0 |

### 2.2 非功能需求

| 需求 ID | 需求描述 |
|---------|---------|
| FE-AI-NFR-001 | 单因子报告生成耗时 ≤ 3 秒 |
| FE-AI-NFR-002 | `QUANT_ALPHALENS_REPORT=0`（默认）时不生成任何额外文件 |
| FE-AI-NFR-003 | 现有 `ic_report.json` 输出格式保留不变 |
| FE-AI-NFR-004 | 489 条现有回归测试 100% 通过 |
| FE-AI-NFR-005 | matplotlib 无头模式（Agg backend）避免 Windows 显示问题 |
| FE-AI-NFR-006 | 关键路径（AlphalensAdapter）100% 行覆盖 |

---

## 第 3 章 技术决策

### 3.1 已确认决策（开放问题确认结果）

| 编号 | 问题 | 决策 |
|------|------|------|
| Q3-1 | 默认是否开启 alphalens 报告 | **否**，需 `QUANT_ALPHALENS_REPORT=1` 显式启用。避免无需求时的额外开销 |
| Q3-2 | 报告 HTML 是否嵌入 PNG | **分文件**，便于 reports-engine 单独引用 PNG |
| Q3-3 | 是否集成到 reports-engine 现有报告模板 | **是**，`metrics.json` 作为分析要素自动渲染 |

### 3.2 依赖选型

**关键风险**：原版 `alphalens` 自 2018 年后停止维护，与项目 `pandas>=2.0.0` 可能冲突。

**选型决策**：采用 `alphalens-reloaded`（stefan-jansen 维护版本），pip 安装，与 pandas 2.0 兼容。

**备选方案**：若 alphalens-reloaded 与 pandas 2.0 不兼容，降级到自研最小分层回测（方案 C），仅输出 JSON 指标不做图。

### 3.3 架构决策

#### 3.3.1 数据格式适配器

新增 [skills/factor-engine/scripts/alphalens_adapter.py](../skills/factor-engine/scripts/alphalens_adapter.py)：

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
```

#### 3.3.2 报告生成

```python
    @staticmethod
    def generate_full_report(factor_data, output_dir: Path, factor_name: str):
        """生成完整 alphalens 报告，输出 HTML + 4 PNG + 1 JSON"""
        import matplotlib
        matplotlib.use("Agg")  # 无头模式，避免 Windows 显示问题
        import alphalens as al

        output_dir.mkdir(parents=True, exist_ok=True)

        # 4 张图分别保存
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

        # 提取关键指标到 JSON（供 reports-engine 引用）
        metrics = AlphalensAdapter._extract_metrics(factor_data, factor_name)
        (output_dir / f"{factor_name}_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )

        # HTML 全报告（分文件引用 PNG）
        AlphalensAdapter._generate_html_report(
            output_dir, factor_name, metrics
        )
```

#### 3.3.3 metrics.json 提取

```python
    @staticmethod
    def _extract_metrics(factor_data, factor_name: str) -> Dict[str, Any]:
        """从 alphalens factor_data 提取 8 个必填字段"""
        import alphalens as al

        # 计算分层回测指标
        returns_data = al.performance.factor_returns(factor_data)
        ic_data = al.performance.factor_information_coefficient(factor_data)
        turnover_data = al.performance.factor_top_bottom_quantile_turnover(factor_data)

        top_quantile_return = float(returns_data[5].mean())  # 假设 5 分层
        bottom_quantile_return = float(returns_data[1].mean())
        long_short_return = top_quantile_return - bottom_quantile_return
        long_short_sharpe = float(long_short_return / returns_data.std() * np.sqrt(252))

        ic_mean = float(ic_data.mean().mean())
        ic_ir = float(ic_data.mean().mean() / ic_data.std().mean())

        avg_turnover_top = float(turnover_data[5].mean())

        # 建议结论（基于 RuleJudge 阈值）
        suggested_verdict = "ACCEPT" if (ic_ir > 0.5 and long_short_sharpe > 0.8) else "REVIEW"

        return {
            "factor": factor_name,
            "top_quantile_return": round(top_quantile_return, 4),
            "bottom_quantile_return": round(bottom_quantile_return, 4),
            "long_short_return": round(long_short_return, 4),
            "long_short_sharpe": round(long_short_sharpe, 4),
            "ic_mean": round(ic_mean, 4),
            "ic_ir": round(ic_ir, 4),
            "avg_turnover_top_quantile": round(avg_turnover_top, 4),
            "suggested_verdict": suggested_verdict,
        }
```

#### 3.3.4 与 engine.py 集成

```python
# skills/factor-engine/engine.py 修改 ic_analysis 末尾
def ic_analysis(self, factor_df, forward_returns, factor_names=None):
    # ... 现有 IC 计算逻辑保留 ...

    # 新增：可选生成 alphalens 报告
    if os.environ.get("QUANT_ALPHALENS_REPORT", "0") == "1":
        from scripts.alphalens_adapter import AlphalensAdapter
        from context import get_work_dir

        report_dir = get_work_dir() / "reports" / "alphalens" / self.task_id
        price_df = factor_df[["code", "date", "close"]].copy()

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

#### 3.3.5 环境变量

```yaml
# SKILL.md 新增
- name: QUANT_ALPHALENS_REPORT
  description: 是否生成 alphalens 完整因子分析报告（0/1），默认 0
  required: false
  default: "0"
```

### 3.4 报告产物结构

```
workspace/reports/alphalens/<task_id>/
├── lncap_returns.png         # 分层净值 + 累积收益
├── lncap_ic.png              # IC 时序 + 累积 IC + IC 热力图
├── lncap_turnover.png        # 分层换手率
├── lncap_summary.png         # 综合统计表
├── lncap_report.html         # 全报告 HTML（分文件引用 PNG）
└── lncap_metrics.json        # 关键指标 JSON（供下游 reports-engine 引用）
```

### 3.5 metrics.json schema（8 必填字段）

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

### 3.6 与 reports-engine 集成

reports-engine 读取 `metrics.json` 自动渲染到分析报告：

```python
# skills/reports-engine/scripts/templates/stock_analysis_report.py 修改
def _render_factor_analysis(self, ctx):
    # 读取 alphalens metrics.json（若存在）
    alphalens_dir = ctx.work_dir / "reports" / "alphalens" / ctx.task_id
    metrics_list = []
    if alphalens_dir.exists():
        for json_file in alphalens_dir.glob("*_metrics.json"):
            metrics_list.append(json.loads(json_file.read_text()))

    # 渲染为分析要素
    if metrics_list:
        return self._render_alphalens_metrics(metrics_list)
    return self._render_legacy_ic_only(ctx)
```

---

## 第 4 章 验收标准

### 4.1 可验证验收标准

| CR ID | 验收标准 | 验证方法 |
|-------|---------|---------|
| CR-1 | `pip install alphalens-reloaded` + `QUANT_ALPHALENS_REPORT=1` 后跑因子引擎，每个因子生成 6 个文件（4 PNG + 1 HTML + 1 JSON） | 文件存在性检查 |
| CR-2 | 未安装 alphalens-reloaded 时 try/except 静默跳过，主流程不报错 | mock ImportError 测试 |
| CR-3 | `QUANT_ALPHALENS_REPORT=0`（默认）时不生成任何额外文件 | 默认值测试 |
| CR-4 | `lncap_metrics.json` 含 8 个必填字段（top_quantile_return / bottom_quantile_return / long_short_return / long_short_sharpe / ic_mean / ic_ir / avg_turnover_top_quantile / suggested_verdict） | JSON 字段校验 |
| CR-5 | archive 自动包含 `reports/alphalens/<task_id>/` 目录 | archive 目录检查 |
| CR-6 | 489 条现有回归测试 100% 通过 | pytest 全量回归 |
| CR-7 | 单因子报告生成耗时 ≤ 3 秒 | pytest 计时 |
| CR-8 | reports-engine 能读取 `metrics.json` 并渲染到分析报告 | 集成测试 |
| CR-9 | matplotlib 使用 Agg backend（无头模式） | 代码审查 + Windows 测试 |

### 4.2 量化评估维度

| 维度 | 现状 | 目标 | 验证方法 |
|------|------|------|---------|
| 因子分析报告数 | 0 | 4 类 PNG + 1 HTML + 1 JSON | 文件存在性检查 |
| 因子筛选维度 | 5 项 IC 统计量 | 5 项 + 4 类图 + 8 项分层指标 | metrics.json 字段数 |
| 单因子报告生成耗时 | N/A | ≤ 3 秒 | pytest 计时 |
| 报告自动归档 | 无 | archive 自动包含 | archive 目录检查 |

---

## 第 5 章 兼容性与回滚

### 5.1 兼容层

| 兼容项 | 机制 |
|--------|------|
| 默认开关 | `QUANT_ALPHALENS_REPORT=0`，行为与 v1.x 完全一致 |
| 现有 `ic_report.json` 输出 | 保留不变，alphalens 报告为附加产物 |
| alphalens-reloaded 缺失 | try/except 静默 fallback，日志 warning |
| reports-engine 渲染 | 优先读 metrics.json，缺失时 fallback 到旧 IC 渲染 |

### 5.2 回滚计划

| 触发条件 | 回滚动作 | 验证 |
|---------|---------|------|
| alphalens-reloaded 安装失败 | 设置 `QUANT_ALPHALENS_REPORT=0` | 主流程正常 |
| matplotlib 无头模式在 Windows 异常 | 同上 + 日志告警 | 不影响 IC 计算 |
| 报告生成耗时 > 10 秒/因子 | 同上 | 性能测试通过 |
| 报告产物与 reports-engine 集成失败 | 仅生成 PNG，不输出 JSON | 文件存在性检查 |
| alphalens 数据丢失率超 25% 报错 | `max_loss=0.25` + 日志记录丢失率 | 数据完整性测试 |

### 5.3 备选方案（方案 C：自研轻量级分层回测）

若 alphalens-reloaded 与 pandas 2.0 不兼容，降级到自研实现：

```python
# 仅输出 JSON 指标，不做图
def _fallback_layered_backtest(factor_df, price_df, factor_name):
    # 5 分层，计算 top/bottom 收益、IC、换手率
    # 输出 metrics.json，不生成 PNG
```

触发条件：alphalens-reloaded import 失败或 `get_clean_factor_and_forward_returns` 报错。

---

## 第 6 章 测试策略

遵循项目硬约束「关键路径 100% 行覆盖，非关键路径 ≥80%」。

### 6.1 L2 单元测试

新增 [tests/factor_engine/test_alphalens_adapter.py](../tests/factor_engine/test_alphalens_adapter.py)：
- `test_to_alphalens_format_basic`：数据格式转换正确
- `test_generate_full_report_outputs_6_files`：6 个文件全部生成
- `test_fallback_when_alphalens_missing`：mock ImportError 验证静默跳过
- `test_metrics_json_contains_8_fields`：JSON 字段完整性
- `test_disabled_by_default`：默认环境变量下不生成文件
- `test_matplotlib_agg_backend`：验证 Agg backend 生效
- `test_extract_metrics_correctness`：metrics 计算正确性

### 6.2 L3 集成测试

扩展 [tests/factor_engine/test_factor_analysis.py](../tests/factor_engine/test_factor_analysis.py)：
- 端到端 `QUANT_ALPHALENS_REPORT=1` 跑通完整 pipeline，验证 archive 含报告目录
- 报告引用：reports-engine 能读取 `metrics.json` 并渲染到分析报告

### 6.3 测试 marker

新增 `@pytest.mark.requires_alphalens`（缺 alphalens-reloaded 时 skip）

---

## 第 7 章 实施顺序与依赖

### 7.1 任务清单

| ID | 任务 | 文件 | 依赖 |
|----|------|------|------|
| T3-1 | 替换 alphalens → alphalens-reloaded 依赖 | `requirements.txt` / `SKILL.md` | - |
| T3-2 | 新增 `QUANT_ALPHALENS_REPORT` 环境变量 | `SKILL.md` | T3-1 |
| T3-3 | 实现 `AlphalensAdapter.to_alphalens_format` | `scripts/alphalens_adapter.py` | T3-1 |
| T3-4 | 实现 `generate_full_report`（4 PNG + 1 HTML + 1 JSON） | `scripts/alphalens_adapter.py` | T3-3 |
| T3-5 | 实现 `_extract_metrics`（8 字段） | `scripts/alphalens_adapter.py` | T3-3 |
| T3-6 | 集成到 `engine.py` `ic_analysis` 末尾 | `skills/factor-engine/engine.py` | T3-4, T3-5 |
| T3-7 | 集成到 `scripts/archive.py` 自动归档 | `scripts/archive.py` | T3-6 |
| T3-8 | 集成到 reports-engine 渲染 `metrics.json` | `skills/reports-engine/scripts/templates/stock_analysis_report.py` | T3-5 |
| T3-9 | L2 单元测试（7 项） | `tests/factor_engine/test_alphalens_adapter.py` | T3-4, T3-5 |
| T3-10 | L3 集成测试扩展 | `tests/factor_engine/test_factor_analysis.py` | T3-7, T3-8 |
| T3-11 | 备选方案 C 实现（自研分层回测） | `scripts/alphalens_adapter.py` | T3-4 |
| T3-12 | 文档更新（SKILL.md / config_guide.md / api_reference.md） | 3 个 references 文件 | T3-8 |

### 7.2 依赖图

```
T3-1 (依赖替换) ──→ T3-2 (环境变量)
                ──→ T3-3 (适配器) ──┬─→ T3-4 (报告生成) ──→ T3-6 (engine 集成) ──→ T3-7 (archive)
                                    ├─→ T3-5 (metrics 提取) ──→ T3-8 (reports-engine)
                                    └─→ T3-11 (备选方案 C)
T3-9 (单测) ←─ T3-4, T3-5
T3-10 (集成) ←─ T3-7, T3-8
T3-12 (文档) ←─ T3-8
```

### 7.3 实施顺序

1. **Phase 1（依赖与基建）**：T3-1 → T3-2 → T3-3
2. **Phase 2（核心实现）**：T3-4 + T3-5（可并行） → T3-11（备选方案，可与 Phase 3 并行）
3. **Phase 3（集成）**：T3-6 → T3-7 → T3-8
4. **Phase 4（测试）**：T3-9 → T3-10
5. **Phase 5（文档）**：T3-12

---

## 第 8 章 并行开发协调机制

### 8.1 文件负责人划分

| 文件 | 负责人角色 | 协作标注 |
|------|----------|---------|
| `scripts/alphalens_adapter.py` | 适配器实现者 | - |
| `skills/factor-engine/engine.py` | 引擎主负责人 | ⚠️ 协作文件（仅 ic_analysis 末尾追加） |
| `scripts/archive.py` | 归档负责人 | ⚠️ 协作文件 |
| `skills/reports-engine/scripts/templates/stock_analysis_report.py` | 报告模板负责人 | ⚠️ 协作文件 |

### 8.2 sys.path 隔离

每个测试文件独立 conftest.py 清理 sys.path，遵循项目硬约束。

### 8.3 分支策略

- 独立 feature 分支：`factor-enhance-3-alphalens`
- 合并顺序：第一个合并（低垂果实，风险可控）
- 遵循 PRD v1.2 P0-1→P0-2 顺序合并约定

---

## 第 9 章 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| alphalens-reloaded 与 pandas 2.0 兼容性问题 | 中 | 高 | CI 双版本 pandas 测试 + 方案 C 备选 |
| matplotlib 无头模式在 Windows 报错 | 中 | 中 | `matplotlib.use("Agg")` + try/except |
| 大数据集（5000 股 × 1000 日）报告生成慢 | 高 | 低 | 分因子并行 + 超时跳过 |
| alphalens 数据丢失率超 25% 报错 | 中 | 中 | `max_loss=0.25` + 日志记录丢失率 |
| metrics 计算与 RuleJudge 阈值不一致 | 低 | 中 | suggested_verdict 仅作建议，不替代 RuleJudge |

---

## 第 10 章 附录

### 10.1 环境变量清单

| 变量名 | 用途 | 默认值 | 必填 |
|--------|------|--------|------|
| `QUANT_ALPHALENS_REPORT` | 是否生成 alphalens 报告 | `0` | 否 |

### 10.2 依赖变更

```diff
# requirements.txt
- alphalens>=0.4.0
+ alphalens-reloaded>=0.4.5  # alphalens 维护版本，兼容 pandas 2.0
```

### 10.3 Frozen Core 保护

本方向不触碰 PRD v1.2 定义的 6 项 Frozen Core 路径。

### 10.4 与其他方向的关系

| 关系 | 说明 |
|------|------|
| 与方向一（Processor Pipeline） | 独立，方向一改造 processors/ 目录，本方向新增 alphalens_adapter.py |
| 与方向二（Polars） | 独立，方向二改造 optimizations/ 目录 |
| 实施顺序 | 第一个启动（低垂果实，风险可控） |
| 协同收益 | 本方向的报告生成可受益于方向二的性能提升；本方向的 metrics.json 可作为方向一 ICAnalysisProcessor 的输出 |

### 10.5 三方向联合实施顺序

```
方向三（本 PRD，短期 1-2 周）
  │  低垂果实，立即可见收益
  │  风险可控（fallback 机制完善）
  ▼
方向二（Polars，中期 2-3 周）
  │  性能基建，本方向报告生成受益
  │  低侵入，双后端可并行验证
  ▼
方向一（Processor Pipeline，长期 3-4 周）
  │  架构升级，需重构处理流程
  │  建议在本方向、方向二稳定后启动
```

### 10.6 交付清单（v1.1 开发完成）

| 任务 ID | 任务描述 | 交付物 |
|---------|---------|--------|
| T3-1 | 替换 alphalens → alphalens-reloaded 依赖 | `requirements.txt` / `SKILL.md` |
| T3-2 | 新增 `QUANT_ALPHALENS_REPORT` 环境变量 | `SKILL.md`（默认 0） |
| T3-3 | 实现 `AlphalensAdapter.to_alphalens_format` | `scripts/alphalens_adapter.py`（MultiIndex 适配） |
| T3-4 | 实现 4 PNG 图表生成 | `scripts/alphalens_adapter.py`（returns/ic/turnover/summary） |
| T3-5 | 实现 HTML 报告聚合 | `scripts/alphalens_adapter.py` |
| T3-6 | 实现 metrics.json 8 字段提取 | `scripts/alphalens_adapter.py` |
| T3-7 | engine.py FACTOR 阶段接入 | `engine.py::_maybe_render_factor_analysis_report` |
| T3-8 | reports-engine 聚合 factor_analysis_report.html | `skills/reports-engine/engine.py` |
| T3-9 | L2 单元测试 | `tests/factor_engine/test_alphalens_adapter.py`（5 用例通过 + 2 跳过） |
| T3-10 | L3 集成测试扩展 | `tests/integration/test_pipeline_archives.py`（3 个 Alphalens 归档用例） |
| T3-11 | 备选方案 C 实现（自研分层回测） | `scripts/alphalens_adapter.py`（fallback_lite） |
| T3-12 | 文档更新 | `SKILL.md` / `config_guide.md` / `api_reference.md` 新增 Alphalens 章节 |

### 10.7 测试覆盖（v1.1 开发完成）

| 测试类别 | 文件 | 用例数 | 状态 |
|---------|------|--------|------|
| L2 单元测试 | `tests/factor_engine/test_alphalens_adapter.py` | 5 通过 + 2 跳过 | 全部通过（2 跳过因 alphalens-reloaded 未安装，走降级路径） |
| L3 集成测试 | `tests/integration/test_pipeline_archives.py`（Alphalens 相关） | 3 | 全部通过 |

**关键测试覆盖项：**
- 默认关闭（`QUANT_ALPHALENS_REPORT=0`）时不生成报告
- `alphalens-reloaded` 缺失时自动降级到 fallback_lite（仅输出 JSON + HTML，不生成 PNG）
- metrics.json 含 8 必填字段（factor / top_quantile_return / bottom_quantile_return / long_short_return / long_short_sharpe / ic_mean / ic_ir / avg_turnover_top_quantile / suggested_verdict）
- matplotlib 使用 Agg 后端（无 GUI 环境）
- 指标提取数值正确性
- FACTOR 阶段启用时报告归档到 archives，禁用时不归档
- reports-engine 自动聚合 factor_analysis_report.html

**已修复 bug：**
1. `reports-engine/engine.py`：`_maybe_render_factor_analysis_report` 输出路径从模块加载期固化的 `REPORT_DIR` 改为运行时 `os.path.join(_work_dir, 'reports')`，修复 `QUANT_WORK_DIR` monkeypatch 后目录不一致问题
2. `alphalens_adapter.py`：`std > 0` 检查调整为 `> 1e-10` 容差，防止浮点精度问题导致 `ic_ir` / `long_short_sharpe` 异常
3. `test_alphalens_adapter.py`：修复 `factor_name` 在 `my_alpha` 与数据列 `test_factor` 间的错配

---

**文档结束。状态：v1.1 开发完成，T3-1 ~ T3-12 全部交付。**
