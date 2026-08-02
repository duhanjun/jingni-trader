# 产品需求文档（PRD）：jingni-trader 工程化升级（借鉴 GemStar P0/P1）

> **文档版本**：v1.1（已确认决策，进入实施阶段）
> **创建日期**：2026-08-02
> **确认日期**：2026-08-02
> **需求来源**：[gemstar_evaluation_report.md](file:///d:/codebuddy/jingni-trader/references/gemstar_evaluation_report.md) P0/P1 优先级借鉴点
> **目标读者**：项目负责人、开发工程师、测试工程师
> **状态**：✅ 用户已确认，进入实际开发环节

---

## 一、概述

### 1.1 背景

jingni-trader 已具备完整的主调度器 + 7 子 Skill 架构（DATA/FACTOR/MODEL/BACKTEST/PORTFOLIO/EXECUTION/REPORT），保留多数据源降级链、xtquant/gm 真实 broker、自然语言意图解析等优势。但在工程化深度上存在以下短板：

| 短板 | 风险 |
|------|------|
| 财务数据未强制 PIT 检查 | 量化回测 look-ahead bias，所有指标失真 |
| 数据缺失即失败 | 非核心表缺失拖垮整条流水线，过度保守 |
| 回测无硬门评审 | 过拟合策略进入生产，资金风险 |
| 真实下单代码无防护 | LLM agent 误改下单核心导致资金安全事故 |
| Paper Trading 用覆盖式 JSON | 持仓跨日跟踪不可重放，事务日志缺失 |
| STAGES 线性推进无转移校验 | 非法跳转无法捕获，故障状态不可追溯 |
| 工件无 sha256/版本 | 调试无血缘，无法判定两次 run 是否同输入 |
| 阶段间用 dict 传递 | 字段拼写错误、类型 drift，无 schema 校验 |

### 1.2 目标

借鉴 GemStar P0/P1 设计，补齐工程化短板，**不破坏现有架构优势**。具体目标：

- **P0（防底线风险）**：PIT 强制契约 + 三态数据质量门 + RuleJudge 五硬门 + frozen core 路径保护
- **P1（提升可追溯/可重放）**：JSONL Paper Trading 账本 + 显式 FSM + 工件 sha256 manifest + Pydantic Schema 全链路

### 1.3 非目标

- 不引入 14 状态 FSM（保持 9-11 状态）
- 不替换多数据源降级链（保留 8 adapter 优势）
- 不放弃 xtquant/gm 真实 broker（保留实盘能力）
- 不引入 LLM Role 三件套（属 P2）
- 不引入因子 DSL/AST 沙箱（属 P2，待 LLM 因子生成需求出现再做）
- 不改自然语言意图解析机制（保留 `strategy_required` 路由）

### 1.4 整合原则

1. **保留 Context 对象**，但增强为 Pydantic V2 模型，dataclass 接口保持向后兼容
2. **保留归档目录结构**（`step_N_<阶段>/summary.md + artifacts/`），新增 sha256 sidecar manifest
3. **保留 MasterEngine 主调度**，FSM 作为内部状态机校验层嵌入，不替换 STAGES
4. **保留 execution-monitor-engine 的 BaseExecutor 抽象**，JSONL 账本作为 PaperExecutor 内部实现
5. **保留 data-engine 的多 adapter 降级链**，PIT 与质量门作为出口校验层，不污染 adapter
6. **新增模块一律放在各 Skill 的 `scripts/` 包内**，遵循"engineering scripts 放 scripts/"约定

---

## 二、需求清单（按优先级分组）

### P0-1：PIT 强制契约

**所属 Skill**：data-engine

**问题**：
- [scripts/adapters/akshare_adapter.py](file:///d:/codebuddy/jingni-trader/skills/data-engine/scripts/adapters/akshare_adapter.py) 财务数据列定义为 `['code', 'report_date', 'pe_ttm', 'pb', ...]`，**缺 `disclosure_date` 字段**
- 下游（factor-engine / backtest-engine）使用财务数据时无 PIT 校验，可能误用未来披露日数据

**需求**：

| ID | 需求 | 验收标准 |
|---|---|---|
| P0-1.1 | 新增 `scripts/pit.py`，提供 `pit_filter(df, asof)` 哨兵函数 | 函数签名 `(df: pd.DataFrame, asof: str) -> pd.DataFrame`；输入 df 无 `disclosure_date` 列时 `raise ValueError`；返回 `df[df["disclosure_date"] <= asof].copy()` |
| P0-1.2 | 财务数据 adapter 出口新增 `disclosure_date` 字段 | akshare/tushare/baostock adapter 在 `_FINANCIAL_STANDARD_COLS` 末尾追加 `disclosure_date`；缺值时回填为 `report_date`（保守降级） |
| P0-1.3 | data-engine 主流程出口增加 `_check_pit` 扫描 | 扫描财务 DataFrame，若存在 `disclosure_date > asof` 的行 → 记 warning + 过滤；扫描结果写入 `ctx.metadata["pit_warnings"]` |
| P0-1.4 | factor-engine / backtest-engine 使用财务数据时强制走 `pit_filter` | 凡读 `ctx.artifacts["DATA"]` 中的 fina 表，必须 `pit_filter(df, asof=trade_date)`；未调用直接 raise |
| P0-1.5 | 单元测试覆盖 | `tests/data-engine/test_pit.py`：覆盖正常过滤、缺列 raise、空 df、asof 边界、disclosure_date 缺失回填 5 个用例 |

**兼容性**：
- `disclosure_date` 字段对下游未知代码无副作用（pandas 会忽略未读列）
- 缺 `disclosure_date` 时回填为 `report_date`，保证不阻断既有降级链

---

### P0-2：三态数据质量门

**所属 Skill**：data-engine

**问题**：
- 当前 data-engine 是"成功/失败"二元状态，非核心数据缺失即整体失败
- 缺乏对核心表 vs 非核心表的分级

**需求**：

| ID | 需求 | 验收标准 |
|---|---|---|
| P0-2.1 | 新增 `scripts/quality_gate.py`，定义 `DataQualityGate` 类 | 纯函数式设计，不发起数据源调用，只检查已拉取的 DataFrame 字典 |
| P0-2.2 | 定义核心表清单 `CORE_TABLES` 与非核心表清单 `OPTIONAL_TABLES` | CORE: `stock_basic / daily / daily_basic / adj_factor / fina_indicator / trade_cal`；OPTIONAL: `forecast / news / top_list / moneyflow / margin / hk_hold / index_weight / limit_list / concept / announcement / express / div / fina_audit / hold_ctrl` |
| P0-2.3 | 实现 `gate.check(tables: Dict[str, pd.DataFrame], asof: str) -> QualityVerdict` | 返回 `QualityVerdict(mode: Literal["normal","degraded","abort"], missing_core: list, missing_optional: list, freshness_days: int, pit_warnings: list)` |
| P0-2.4 | 三态判定规则 | **abort**：任一 CORE 表缺失 或 freshness > 10 交易日；**degraded**：freshness > 5 交易日 或存在 PIT warning；**normal**：全部通过 |
| P0-2.5 | data-engine 出口调用 `gate.check`，结果写入 `ctx.metadata["data_quality"]` | 主流程在返回前调用；`mode="abort"` 时 `run()` 返回 `{"success": False, "error": "data_quality_abort"}`；`mode="degraded"` 时仅记录，继续后续流程 |
| P0-2.6 | 下游 Skill 读取 `ctx.metadata["data_quality"]["mode"]` 决策 | factor-engine / strategy-model-engine / reports-engine 检测 `degraded` 时跳过策略 promote（若有）；reports-engine 在报告中标注数据质量降级 |
| P0-2.7 | 单元测试覆盖 | `tests/data-engine/test_quality_gate.py`：normal/degraded/abort 三态各 2 个用例 + freshness 边界 + PIT warning 触发 degraded |

**兼容性**：
- data-engine 失败语义保持：`abort` → `success=False`，与现有错误处理兼容
- `degraded` 是新增态，不阻断流程，对现有调用方零影响

---

### P0-3：RuleJudge 五硬门 + 分段一致性

**所属 Skill**：backtest-engine

**问题**：
- [skills/backtest-engine/engine.py:87-110](file:///d:/codebuddy/jingni-trader/skills/backtest-engine/engine.py#L87-110) 仅计算 sharpe/calmar/max_drawdown/win_rate，无硬门评审
- 无分段一致性检查，靠单段暴涨拉高指标的过拟合策略可进入生产

**需求**：

| ID | 需求 | 验收标准 |
|---|---|---|
| P0-3.1 | 新增 `scripts/rule_judge.py`，定义 `RuleJudge` 类 | 纯 Python 实现，无 LLM 依赖；构造函数接收 `config: dict` 可覆盖默认阈值 |
| P0-3.2 | 实现五硬门 | 1) `sharpe >= 0.8`（默认放宽档，适合小样本阶段） 2) `calmar >= 0.5` 3) `max_drawdown <= 0.35` 4) `completed_trades >= 50` 5) `segment_sharpe_ir_std <= 0.5` |
| P0-3.3 | 分段一致性算法 | 把回测期按 ~252 交易日切分（不足一段单独成段），计算各段 Sharpe IR 的标准差；段数 < 2 时第 5 门跳过（记 warning 不阻塞） |
| P0-3.4 | 实现 `judge(metrics: Dict, equity_curve: pd.DataFrame, trade_count: int) -> Verdict` | 返回 `Verdict(recommended_state: Literal["candidate","rejected"], passed_gates: List[str], failed_gates: List[str], segment_stats: Dict)` |
| P0-3.5 | backtest-engine 主流程出口调用 `RuleJudge.judge` | 在 `BacktestEngine.run()` 返回前调用；结果写入 `result["verdict"]` 字段；`rejected` 时仍返回 `success=True`（回测本身成功，只是策略未通过评审） |
| P0-3.6 | `verdict.recommended_state` 写入 `ctx.metadata["strategy_verdict"]` | 下游（portfolio-risk-engine / execution-monitor-engine）检测 `rejected` 时拒绝进入生产 |
| P0-3.7 | 阈值通过环境变量可调（默认放宽档，可调至严格档） | `QUANT_RULE_JUDGE_SHARPE_MIN`（默认 0.8，严格档 1.0）、`QUANT_RULE_JUDGE_CALMAR_MIN`（默认 0.5，严格档 0.8）、`QUANT_RULE_JUDGE_MDD_MAX`（默认 0.35，严格档 0.30）、`QUANT_RULE_JUDGE_TRADES_MIN`（默认 50，严格档 100）、`QUANT_RULE_JUDGE_SEG_IR_STD_MAX`（默认 0.5，严格档 0.5）；严格档预设值写入 `scripts/rule_judge.py` 注释供参考 |
| P0-3.8 | 单元测试覆盖 | `tests/backtest-engine/test_rule_judge.py`：五门各自 pass/fail 用例 + 分段边界（1 段/2 段/N 段）+ 阈值环境变量覆盖 |

**兼容性**：
- 现有 `result["metrics"]` 字段保留不变
- `result["verdict"]` 为新增字段，对未读取的下游无副作用
- `rejected` 不阻断回测成功返回，保证调用方兼容

---

### P0-4：Frozen Core 路径策略保护

**所属 Skill**：execution-monitor-engine（核心防护对象）+ 跨 Skill 通用机制

**问题**：
- jingni-trader 涉及真实下单（xtquant/gm），但缺少对自身代码改动的防护
- LLM agent（reports-engine 内嵌）若误改下单核心代码，可能导致资金安全事故

**需求**：

| ID | 需求 | 验收标准 |
|---|---|---|
| P0-4.1 | 新增 `scripts/path_policy.py`，定义 `validate_changed_paths(changed: List[str], allowed: List[str], forbidden: List[str]) -> List[str]` | 返回违规路径列表；空列表表示合规。规则：1) 空路径违规 2) `../` 或绝对路径违规 3) **forbidden 优先于 allowed** 4) 不在 allowed 内违规 |
| P0-4.2 | 定义 frozen core 清单（写入 `scripts/path_policy.py` 常量，共 6 项） | `FROZEN_PATHS = ["scripts/real_broker/**", "scripts/risk/**", "schemas/order.py", "schemas/execution_report.py", "engine.py", "skills/portfolio-risk-engine/scripts/cost.py"]`（执行下单/风控/订单 schema/主调度入口/成本模型） |
| P0-4.3 | 新增 `GitChangeTracker` 类 | `pre_snapshot()` 调用 `git status --porcelain --untracked-files=all`（10s 超时）；`post_diff()` 返回本次运行新增/修改的路径；要求 pre 时 worktree clean（保护用户改动） |
| P0-4.4 | **双重校验机制**（前置 + 退出兜底） | **前置**：reports-engine LLM 调用 write/edit 工具时立即调用 `validate_changed_paths`，触碰 frozen core → raise + 记录审计日志；**退出兜底**：MasterEngine.run_pipeline 注册 `atexit` 钩子，运行结束时调用 `GitChangeTracker.post_diff()`，对全量变更路径再次校验，发现 frozen core 改动 → 记录 critical 审计日志（此时已无法 raise，仅告警） |
| P0-4.5 | 新增审计日志 `audit/path_violations.jsonl` | 每行一条 `{"timestamp": "...", "violator": "reports-engine", "violations": [...], "rejected": true}` |
| P0-4.6 | 单元测试覆盖 | `tests/execution-monitor-engine/test_path_policy.py`：合规/违规/forbidden 优先级/绝对路径/git status 解析 5 个用例 |

**兼容性**：
- 路径策略仅在 LLM 主动修改文件时触发，对纯读取流程零影响
- frozen core 清单可通过 `scripts/config.py` 环境变量 `FROZEN_PATHS_EXTRA` 追加（不覆盖默认）

---

### P1-1：追加式 JSONL Paper Trading 账本

**所属 Skill**：execution-monitor-engine

**问题**：
- [skills/execution-monitor-engine/engine.py:298-308](file:///d:/codebuddy/jingni-trader/skills/execution-monitor-engine/engine.py#L298-308) `PaperExecutor.save_state()` 用 `"w"` 覆盖模式写 `account_state.json`
- 每次状态覆盖前序历史，无法回放事务，跨日持仓跟踪不可追溯

**需求**：

| ID | 需求 | 验收标准 |
|---|---|---|
| P1-1.1 | 新增 `scripts/paper_ledger.py`，定义 `PaperTradeRecordV1`（Pydantic V2） | 字段：`execution_id: str`（唯一）、`trade_date: str`、`code: str`、`side: Literal["buy","sell"]`、`shares: int`（≥0，multiple_of=100）、`price: float`（>0）、`commission: float`（≥0）、`stamp_tax: float`（≥0）、`slippage_cost: float`、`position_after_shares: int`（≥0）、`cash_after: float`、`nav_after: float`、`confirmed: bool`（必填=True）、`created_at: datetime` |
| P1-1.2 | 实现 `append_paper_trade(path: Path, record: PaperTradeRecordV1)` | 写入前扫整文件去重（`execution_id` 重复 raise）；`with path.open("a")` 追加；`record.model_dump_json() + "\n"` |
| P1-1.3 | 实现 `read_paper_trades(path: Path) -> List[PaperTradeRecordV1]` | 逐行 `PaperTradeRecordV1.model_validate_json(line)`；损坏行 skip + 记 warning |
| P1-1.4 | 实现 `replay_ledger(path: Path) -> AccountSnapshot` | 顺序读 JSONL 重建 `AccountSnapshot(positions: Dict[str, PositionState], cash: float, nav: float, bought_today: Set[str])`；T+1 强制（sell/reduce 当日买入的标的 raise） |
| P1-1.5 | `PaperExecutor` 改造：`save_state` 改为 append 单条 record 到 `ledger.jsonl` | 移除 `"w"` 覆盖式 `account_state.json`；保留 `account_state.json` 作为快照（由 `replay_ledger` 重建生成），方便快速启动 |
| P1-1.6 | `PaperExecutor.__init__` 启动时调用 `replay_ledger` 重建状态 | ledger 不存在时使用初始资金；存在时按事务顺序重建；重建失败 raise（fail fast） |
| P1-1.6a | **旧状态自动迁移流程**（检测到 `account_state.json` 存在但 `ledger.jsonl` 不存在时） | 1) 备份 `account_state.json` → `account_state.json.bak.<timestamp>`；2) 读取旧状态，校验字段完整性（nav/cash/positions 是否齐全）；3) 校验通过 → 生成一条 `execution_id="opening_balance_<timestamp>"`、`side="buy"`、`shares=0`、`position_after_shares=当前持仓`、`price=0.0`（占位）、`confirmed=True` 的 record；4) 校验失败 → raise + 提示用户用 `INIT_CAPITAL` 重新启动 |
| P1-1.7 | `confirmed=True` 必填硬约束 | 任何 `confirmed=False` 的 record 写入时 raise；防止 LLM 输出直接落账 |
| P1-1.8 | 路径配置 | `PAPER_LEDGER_PATH = os.path.join(EXECUTION_DIR, "ledger.jsonl")`；通过环境变量 `PAPER_LEDGER_PATH` 可覆盖 |
| P1-1.9 | 单元测试覆盖 | `tests/execution-monitor-engine/test_paper_ledger.py`：追加/去重/重放/T+1 强制/confirmed=False raise/损坏行 skip/position_after_shares 重建/旧状态迁移成功/旧状态迁移字段缺失 raise 8 个用例 |

**兼容性**：
- 旧 `account_state.json` 仍生成（作为快照），现有读取代码无影响
- 旧文件可手动迁移：检测到 `account_state.json` 存在但 `ledger.jsonl` 不存在时，生成一条 `opening_balance` record 作为初始状态

---

### P1-2：显式 FSM + 状态转移校验

**所属 Skill**：master（engine.py）

**问题**：
- [engine.py:85-96](file:///d:/codebuddy/jingni-trader/engine.py#L85-96) `STAGES` 是线性列表，无转移白名单
- 非法跳转（如 DATA → BACKTEST）无法捕获
- 无 `DEGRADED` 中间态，数据降级时只能 fail
- 无 `MANUAL_ATTENTION` 终态供人工介入

**需求**：

| ID | 需求 | 验收标准 |
|---|---|---|
| P1-2.1 | 新增 `scripts/fsm.py`，定义 `DailyFSM` 类 | 显式状态枚举：`INITIALIZED / DATA / FACTOR / MODEL / BACKTEST / PORTFOLIO / EXECUTION / REPORT / DEGRADED / FAILED / MANUAL_ATTENTION`（共 **11 状态**，REPORT 为终态，run_manifest 在 REPORT 阶段内落盘） |
| P1-2.2 | 定义显式转移白名单 `_ALLOWED_TRANSITIONS: Dict[str, List[str]]` | 主路径：`INITIALIZED→DATA→FACTOR→MODEL→BACKTEST→PORTFOLIO→EXECUTION→REPORT`（REPORT 即终态，run_manifest 在此阶段落盘后流程结束）；分析路径：`FACTOR→REPORT`；任何非终态可 →`DEGRADED`；`DEGRADED→REPORT/FAILED/前置阶段`；`任何→FAILED/MANUAL_ATTENTION` |
| P1-2.3 | 实现 `transition(current: str, target: str) -> str` | 非法跳转 `raise ValueError(f"illegal transition: {current}→{target}")`；合法跳转返回新状态 |
| P1-2.4 | `DEGRADED` 是非终态 | 可恢复到 `REPORT`（跳过失败阶段）或转入 `FAILED`；不阻塞流程 |
| P1-2.5 | `MANUAL_ATTENTION` 是终态 | 进入此状态后任何 transition 都 raise；必须人工清理后重启 |
| P1-2.6 | MasterEngine.run_pipeline 内嵌 FSM 校验 | 每次进入下一阶段前调用 `fsm.transition(current, target)`；非法跳转记录到 `ctx.errors` 并 fail-fast |
| P1-2.7 | data-engine 返回 `data_quality.mode == "abort"` 时 → `MANUAL_ATTENTION` | MasterEngine 检测到 abort 不再继续，转入 `MANUAL_ATTENTION` 终态 |
| P1-2.8 | 新增 `IncidentFSM`（7 状态故障自愈，纯内存） | 状态：`DETECTED → CLASSIFIED → {RETRYING / DEGRADED / MANUAL_ATTENTION / RESOLVED}`；`RETRYING ↔ CLASSIFIED` 重试回路；不落库（持久化由后续 P2 state.db 负责） |
| P1-2.9 | MasterEngine 捕获阶段异常时创建 `IncidentFSM` 实例 | 重试 1 次（仅对可重试异常：网络/数据源降级）；仍失败转 `DEGRADED` 或 `MANUAL_ATTENTION` |
| P1-2.10 | 单元测试覆盖 | `tests/master/test_fsm.py`：合法/非法转移 × 主路径/分析路径/降级路径/MANUAL_ATTENTION 终态 + IncidentFSM 重试回路 |

**兼容性**：
- `STAGES` 和 `STAGE_ORDER` 常量保留，FSM 是内部校验层
- `target_stages` 字段语义不变（仍由 `parse_intent` 设置），FSM 只校验转移合法性
- `run_pipeline` 返回结构不变，仅增加 `ctx.metadata["fsm_transitions"]` 调试字段

---

### P1-3：工件版本化 + sha256 sidecar manifest

**所属 Skill**：master（scripts/archive.py 增强）+ 各 Skill 出口

**问题**：
- [scripts/archive.py:49-62](file:///d:/codebuddy/jingni-trader/scripts/archive.py#L49-62) 仅 `shutil.copy2` 复制产物，无完整性校验、无血缘
- 同输入两次 run 是否同输出无法判定，调试困难

**需求**：

| ID | 需求 | 验收标准 |
|---|---|---|
| P1-3.1 | 新增 `scripts/artifact_store.py`，提供 `compute_sha256(path: str) -> str` | 对任意文件计算 sha256（按 1MB chunk 流式读取，支持大文件） |
| P1-3.2 | 提供 `write_artifact(name: str, data: Any, output_dir: str, inputs: List[str] = None) -> str` | 写入 `<name>.json` + `<name>.manifest.json`；manifest 含 `{"name":..., "version":"V1", "sha256":..., "created_at":..., "inputs":[{"name":..., "sha256":...}]}` |
| P1-3.3 | 提供 `read_artifact(name: str, output_dir: str) -> Tuple[Any, Dict]` | 读数据 + manifest；校验 sha256 不匹配时 raise |
| P1-3.4 | 各 Skill 出口产物带 `version: Literal["...V1"]` 字段 | data-engine: `CleanedDataV1`；factor-engine: `FactorDataV1`；backtest-engine: `BacktestResultV1`；execution-monitor-engine: `ExecutionReportV1`；reports-engine: `ReportV1` |
| P1-3.5 | `RunArchiver.save_artifact_copy` 增强：复制产物时同时生成 manifest | manifest 落到 `<step_dir>/artifacts/<name>.manifest.json`；inputs 字段从 `ctx.artifacts` 推断上游依赖 |
| P1-3.6 | 新增 `run_manifest.json` | 每次 run 结束时写到 `<run_dir>/run_manifest.json`；内容：`{"run_id":..., "start_at":..., "end_at":..., "stages":[{"name":..., "status":..., "latency_sec":..., "artifacts":[{"name":..., "sha256":...}]}], "inputs_sha256": {...}}` |
| P1-3.7 | 同输入回放校验工具 `scripts/replay_check.py` | 对比两次 run 的 `run_manifest.json`，输出"输入 sha256 相同 / 输出 sha256 差异"报告；用于调试确定性 |
| P1-3.8 | 单元测试覆盖 | `tests/master/test_artifact_store.py`：sha256 计算/manifest 读写/inputs 血缘/校验失败 raise/run_manifest 生成/replay_check 6 个用例 |

**兼容性**：
- 现有 `ctx.artifacts` 字典结构保留，仅新增 sidecar manifest 文件
- 旧产物（无 manifest）读取时不校验，记 warning（向后兼容）

---

### P1-4：Pydantic V2 Schema 全链路强校验

**所属 Skill**：master（scripts/context.py）+ 各 Skill 出口 schema

**问题**：
- [scripts/context.py:11-53](file:///d:/codebuddy/jingni-trader/scripts/context.py#L11-53) Context 是 dataclass，`artifacts: Dict[str, str]` 无类型约束
- 阶段间用 dict 传递结果，字段拼写错误、类型 drift 无校验

**需求**：

| ID | 需求 | 验收标准 |
|---|---|---|
| P1-4.1 | 新增 `scripts/schemas.py`，定义全链路 Pydantic V2 模型 | `OrderIntentV1`、`ExecutionReportV1`、`PositionSnapshotV1`、`RiskLimitV1`、`BacktestResultV1`、`CleanedDataV1`、`FactorDataV1`、`ReportV1` |
| P1-4.2 | 所有模型 `ConfigDict(extra="forbid")` | 防止字段拼写错误；多余字段 raise |
| P1-4.3 | 字段约束 | `shares: int = Field(ge=0, multiple_of=100)`；`price: float = Field(gt=0)`；`code: str = Field(min_length=8, max_length=10, pattern=r"^\d{6}\.(SH|SZ|BJ)$")`；`side: Literal["buy","sell"]` |
| P1-4.4 | Context 增强为 Pydantic V2 `BaseModel`，保留 dataclass 兼容接口 | 实现 `update_artifact` / `get_artifact` / `add_error` / `to_dict` / `to_json` / `from_dict` / `from_json` 同名方法；`from_dict` 内部用 `model_validate` |
| P1-4.5 | 各 Skill 出口返回值用 schema 校验 | data-engine 返回 `CleanedDataV1`；factor-engine 返回 `FactorDataV1`；backtest-engine 返回 `BacktestResultV1`（含 `verdict: VerdictV1`）；execution-monitor-engine 返回 `ExecutionReportV1`；reports-engine 返回 `ReportV1` |
| P1-4.6 | `MasterEngine.run_pipeline` 在阶段间传递时调用 `model_validate` | 上游产物 schema 不匹配 → 记录错误 + 转入 `FAILED` 状态 |
| P1-4.7 | 单元测试覆盖 | `tests/master/test_schemas.py`：每个 schema 的合法/非法用例（extra 字段 raise / 字段约束 violation / code 格式校验 / Literal 校验） |

**兼容性**：
- Context 同名方法保留，现有调用代码（`ctx.update_artifact`、`ctx.get_artifact`）零改动
- 各 Skill 的 `run()` 函数签名不变，仅返回值内部用 schema 校验
- 现有用 dict 读取 `result["metrics"]` 的代码改为读取 `BacktestResultV1.metrics` 字段（保持 dict 风格访问兼容）

---

## 三、技术设计要点

### 3.1 模块依赖关系

```
engine.py (master)
├── scripts/fsm.py (P1-2 新增)
├── scripts/artifact_store.py (P1-3 新增)
├── scripts/path_policy.py (P0-4 新增，跨 Skill)
├── scripts/schemas.py (P1-4 新增)
└── scripts/context.py (P1-4 改造为 Pydantic)

skills/data-engine/
├── scripts/pit.py (P0-1 新增)
├── scripts/quality_gate.py (P0-2 新增)
└── engine.py (出口调用 pit/quality_gate)

skills/backtest-engine/
├── scripts/rule_judge.py (P0-3 新增)
└── engine.py (出口调用 RuleJudge)

skills/execution-monitor-engine/
├── scripts/paper_ledger.py (P1-1 新增)
└── engine.py (PaperExecutor 改造)
```

### 3.2 数据流（含新增校验点）

```
用户意图
  ↓
parse_intent → ctx (Pydantic Context, P1-4)
  ↓
[FSM: INITIALIZED → DATA] (P1-2)
  ↓
data-engine.run(ctx)
  ├── adapter 降级链 (保留)
  ├── 出口: pit_filter (P0-1)
  └── 出口: DataQualityGate.check (P0-2)
       ├── mode=abort  → FSM 转 MANUAL_ATTENTION (P1-2)
       ├── mode=degraded → FSM 转 DEGRADED (P1-2)
       └── mode=normal  → 继续
  ↓
write_artifact + sha256 manifest (P1-3)
  ↓
[FSM: DATA → FACTOR]
  ↓
factor-engine.run(ctx) (使用财务数据时强制 pit_filter, P0-1)
  ↓
[分析路径] FSM: FACTOR → REPORT
[策略路径] FSM: FACTOR → MODEL → BACKTEST
  ↓
backtest-engine.run(ctx)
  └── 出口: RuleJudge.judge (P0-3)
       └── verdict.recommended_state=rejected → 跳过 PORTFOLIO/EXECUTION
  ↓
[FSM: BACKTEST → PORTFOLIO → EXECUTION → REPORT → COMPLETED]
  ↓
execution-monitor-engine.run(ctx)
  └── PaperExecutor (P1-1)
       ├── 启动: replay_ledger 重建状态
       ├── 成交: append_paper_trade (含 position_after_shares)
       └── confirmed=True 必填
  ↓
reports-engine.run(ctx) (LLM 注入路径校验 path_policy, P0-4)
  ↓
write run_manifest.json (P1-3)
```

### 3.3 状态机转移图

```
                    INITIALIZED
                        ↓
                      DATA ←──────┐
                        ↓         │
                      FACTOR      │ (恢复)
                        ↓         │
            ┌─────── MODEL ←──────┘
            │           ↓
            │       BACKTEST
            │           ↓
            │       PORTFOLIO
            │           ↓
            │       EXECUTION
            │           ↓
            │        REPORT (终态，run_manifest 落盘后流程结束)
            │
            └──→ DEGRADED (非终态)
                   ↓
              ┌────┴────┐
              ↓         ↓
            REPORT    FAILED (终态)
           (终态)

任何状态 → MANUAL_ATTENTION (终态，需人工清理)
任何状态 → FAILED (终态)
```

### 3.4 Pydantic 模型示例（P1-4）

```python
from pydantic import BaseModel, ConfigDict, Field, Literal
from datetime import datetime
from typing import Dict, List, Optional

class OrderIntentV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(pattern=r"^\d{6}\.(SH|SZ|BJ)$")
    side: Literal["buy", "sell"]
    shares: int = Field(ge=0, multiple_of=100)
    price: float = Field(gt=0)
    order_type: Literal["limit", "market"] = "limit"

class ExecutionReportV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["ExecutionReportV1"] = "ExecutionReportV1"
    execution_id: str
    trade_date: str
    orders: List[OrderIntentV1]
    fills: List[Dict[str, float]]
    nav_after: float
    cash_after: float
    positions_after: Dict[str, int]
    verdict: Literal["confirmed", "rejected"]
    created_at: datetime
```

### 3.5 Paper Ledger Record 示例（P1-1）

```json
{"execution_id":"20260802_001","trade_date":"2026-08-02","code":"000001.SZ","side":"buy","shares":200,"price":12.34,"commission":5.0,"stamp_tax":0.0,"slippage_cost":1.23,"position_after_shares":200,"cash_after":99753.20,"nav_after":100000.00,"confirmed":true,"created_at":"2026-08-02T10:30:00"}
```

### 3.6 sha256 Manifest 示例（P1-3）

```json
{
  "name": "cleaned_data.parquet",
  "version": "CleanedDataV1",
  "sha256": "abc123...",
  "size_bytes": 1234567,
  "created_at": "2026-08-02T10:25:00",
  "inputs": [
    {"name": "daily_raw.parquet", "sha256": "def456..."}
  ]
}
```

---

## 四、测试策略

### 4.1 测试金字塔

| 层级 | 范围 | 标记 | 工具 |
|---|---|---|---|
| L1 契约测试 | 模块接口契约 | `@pytest.mark.contract` | pytest |
| L2 单元测试 | 各新增模块内部逻辑 | `@pytest.mark.unit` | pytest |
| L3 集成测试 | 跨 Skill 数据流 | `@pytest.mark.integration` | pytest + 合成数据 |
| 回归测试 | 现有 98 个测试不破坏 | 全量 | pytest |

### 4.2 测试目录结构（遵循工程约定）

```
tests/
├── data-engine/
│   ├── test_pit.py (P0-1 新增)
│   └── test_quality_gate.py (P0-2 新增)
├── backtest-engine/
│   └── test_rule_judge.py (P0-3 新增)
├── execution-monitor-engine/
│   ├── test_path_policy.py (P0-4 新增)
│   └── test_paper_ledger.py (P1-1 新增)
├── master/
│   ├── test_fsm.py (P1-2 新增)
│   ├── test_artifact_store.py (P1-3 新增)
│   └── test_schemas.py (P1-4 新增)
└── integration/
    └── test_p0_p1_integration.py (新增，端到端)
```

### 4.3 验收标准（全量）

- [ ] 新增测试用例 ≥ 50 个，全部通过
- [ ] 现有 98 个测试 100% 通过（无回归）
- [ ] **测试覆盖率门槛**：关键路径 100% 行覆盖 + 非关键路径 ≥ 80% 行覆盖
- [ ] PIT 强制契约：财务数据缺 disclosure_date 时下游 raise
- [ ] 数据质量门：核心表缺失 abort，非核心缺失 degraded
- [ ] RuleJudge：过拟合策略（单段暴涨）被 rejected
- [ ] Frozen Core：LLM 改 `engine.py` 触发违规日志
- [ ] Paper Ledger：ledger.jsonl 追加 + 重放重建状态正确
- [ ] FSM：非法跳转（DATA→BACKTEST）raise
- [ ] sha256：两次同输入 run 输出 sha256 一致
- [ ] Pydantic：extra 字段 raise，code 格式违规 raise

### 4.4 关键路径清单（必须 100% 行覆盖）

| 模块 | 关键函数 |
|---|---|
| P0-1 | `pit_filter` |
| P0-2 | `DataQualityGate.check` |
| P0-3 | `RuleJudge.judge` + 分段一致性算法 |
| P0-4 | `validate_changed_paths` + `GitChangeTracker.pre_snapshot/post_diff` |
| P1-1 | `append_paper_trade` + `replay_ledger` + 旧状态迁移 |
| P1-2 | `DailyFSM.transition` + `IncidentFSM.transition` |
| P1-3 | `compute_sha256` + `write_artifact` + `read_artifact` |
| P1-4 | `Context.model_validate` + 各 V1 模型的 `extra="forbid"` 校验 |

非关键路径（≥ 80%）：各模块的辅助函数、错误处理分支、日志记录、环境变量读取。

---

## 五、实施路线图

### 5.1 阶段划分（P0 四项并行 + P1 串行）

| 阶段 | 内容 | 依赖 | 并行度 |
|---|---|---|---|
| 阶段一（P0） | P0-1 PIT + P0-2 质量门 + P0-3 RuleJudge + P0-4 Frozen Core | 无 | **4 项并行**（分属 data-engine/backtest-engine/execution-monitor-engine，代码无交叉） |
| 阶段二（P1-基础） | P1-4 Pydantic Schema（先行，为后续提供类型基础） | 阶段一 | 串行 |
| 阶段三（P1-账本） | P1-1 Paper Ledger | P1-4 | 串行 |
| 阶段四（P1-状态机） | P1-2 FSM | P1-4 | 串行 |
| 阶段五（P1-追溯） | P1-3 sha256 Manifest | P1-4 | 串行 |

> **并行开发协调机制详见第十章**。

### 5.2 每阶段交付物

- 新增源码文件（按上文需求清单）
- 新增测试文件（按上文测试目录）
- 更新 SKILL.md（各 Skill 文档补充新模块说明）
- 更新 [references/prd_gemstar_integration.md](file:///d:/codebuddy/jingni-trader/references/prd_gemstar_integration.md) 状态字段（标记已完成阶段）

### 5.3 回滚预案

| 风险 | 回滚方案 |
|---|---|
| Pydantic Context 不兼容旧调用方 | 保留 dataclass 同名方法，降级路径走 dataclass |
| FSM 转移校验过严阻塞正常流程 | 通过 `QUANT_FSM_STRICT_MODE=false` 环境变量降级为 warning |
| RuleJudge 阈值过严误杀正常策略 | 阈值通过环境变量可调，可临时调低（默认已为放宽档） |
| Paper Ledger 损坏导致启动失败 | 保留 `account_state.json` 作为兜底快照，提供 `--rebuild-ledger` 命令 |
| sha256 计算性能影响大文件 | 流式 1MB chunk，测试样本 > 100MB 文件验证 < 1s |
| 并行开发导致 sys.path 冲突 | 各 Skill 测试用独立 conftest.py 清理 sys.path（详见第十章） |

---

## 六、配置与命令

### 6.1 新增环境变量（统一 `QUANT_` 前缀，遵循工程约定）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QUANT_PIT_STRICT` | `true` | PIT 强制契约开关，`false` 时降级为 warning |
| `QUANT_QUALITY_GATE_FRESHNESS_ABORT_DAYS` | `10` | 核心 freshness abort 阈值 |
| `QUANT_QUALITY_GATE_FRESHNESS_DEGRADED_DAYS` | `5` | 降级 freshness 阈值 |
| `QUANT_RULE_JUDGE_SHARPE_MIN` | `0.8`（放宽档） | RuleJudge Sharpe 下限；严格档设为 `1.0` |
| `QUANT_RULE_JUDGE_CALMAR_MIN` | `0.5`（放宽档） | RuleJudge Calmar 下限；严格档设为 `0.8` |
| `QUANT_RULE_JUDGE_MDD_MAX` | `0.35`（放宽档） | RuleJudge 最大回撤上限；严格档设为 `0.30` |
| `QUANT_RULE_JUDGE_TRADES_MIN` | `50`（放宽档） | RuleJudge 完成交易数下限；严格档设为 `100` |
| `QUANT_RULE_JUDGE_SEG_IR_STD_MAX` | `0.5` | 分段 Sharpe IR Std 上限（放宽档与严格档一致） |
| `QUANT_FROZEN_PATHS_EXTRA` | （空） | 追加 frozen core 路径（不覆盖默认 6 项） |
| `QUANT_PAPER_LEDGER_PATH` | `<EXECUTION_DIR>/ledger.jsonl` | Paper Trading 账本路径 |
| `QUANT_FSM_STRICT_MODE` | `true` | FSM 严格模式，`false` 时非法转移仅 warning |

### 6.2 不新增命令

本次升级全部为内部模块增强，不新增 CLI 命令，不改变 `MasterEngine.run()` 调用方式。

---

## 七、文档更新清单

| 文档 | 更新内容 |
|---|---|
| `SKILL.md`（master） | 新增"工程化设计"章节，引用本 PRD |
| `skills/data-engine/SKILL.md` | 新增 PIT 强制契约 + DataQualityGate 章节 |
| `skills/backtest-engine/SKILL.md` | 新增 RuleJudge 五硬门章节 |
| `skills/execution-monitor-engine/SKILL.md` | 新增 Paper Ledger + Frozen Core 章节 |
| `references/gemstar_evaluation_report.md` | 在文末追加"实施进度"小节，链接本 PRD |
| `references/prd_gemstar_integration.md` | 实施过程中更新阶段状态 |

---

## 八、验收检查表（用户确认后实施时使用）

- [ ] P0-1 PIT 强制契约：5 个测试通过
- [ ] P0-2 三态数据质量门：7 个测试通过
- [ ] P0-3 RuleJudge 五硬门：6 个测试通过
- [ ] P0-4 Frozen Core 路径策略：5 个测试通过
- [ ] P1-1 Paper Trading JSONL 账本：7 个测试通过
- [ ] P1-2 显式 FSM + IncidentFSM：8 个测试通过
- [ ] P1-3 工件 sha256 Manifest：6 个测试通过
- [ ] P1-4 Pydantic V2 Schema 全链路：7 个测试通过
- [ ] 现有 98 个测试 100% 通过（无回归）
- [ ] 端到端集成测试通过（含 PIT/质量门/RuleJudge/Ledger/FSM 全链路）
- [ ] 所有新增环境变量文档化
- [ ] 所有 SKILL.md 更新完成

---

## 九、决策记录（用户已确认）

| # | 决策项 | 确认方案 | 影响章节 |
|---|---|---|---|
| 1 | 状态机状态数 | 合并 `COMPLETED → REPORT` 终态，共 **11 状态** | P1-2.1 / 3.3 |
| 2 | RuleJudge 默认阈值 | **默认放宽档**：Sharpe≥0.8 / Calmar≥0.5 / MDD≤0.35 / Trades≥50 / SegIR≤0.5；环境变量可调至严格档 | P0-3.2 / P0-3.7 / 6.1 |
| 3 | Frozen Core 清单 | 6 项（追加 `portfolio-risk-engine/scripts/cost.py`） | P0-4.2 |
| 4 | LLM 路径策略触发时机 | **双重校验**（前置 + 退出兜底） | P0-4.4 |
| 5 | Paper Ledger 旧状态迁移 | 自动迁移 + 备份原文件 + 字段校验 | P1-1.6a |
| 6 | sha256 Manifest 范围 | 所有 Skill 产物全覆盖 | P1-3.5 |
| 7 | Pydantic Context 改造深度 | 保留 dataclass 兼容接口（零回归） | P1-4.4 |
| 8 | 实施顺序 | P0 四项并行 + P1 串行 | 5.1 / 第十章 |
| 9 | 测试覆盖率门槛 | 关键路径 100% + 非关键 80% | 4.3 / 4.4 |
| 10 | state.db 引入时机 | 推迟到 P2，P1-2 不落库 | P1-2.8 |

---

## 十、并行开发协调机制（P0 四项并行）

### 10.1 并行可行性分析

P0 四项分属不同 Skill，代码物理隔离：

| P0 项 | 所属 Skill | 新增文件 | 修改文件 |
|---|---|---|---|
| P0-1 PIT | data-engine | `scripts/pit.py` | `scripts/adapters/*.py`（追加 disclosure_date）、`engine.py`（出口调用） |
| P0-2 质量门 | data-engine | `scripts/quality_gate.py` | `engine.py`（出口调用） |
| P0-3 RuleJudge | backtest-engine | `scripts/rule_judge.py` | `engine.py`（出口调用） |
| P0-4 Frozen Core | execution-monitor-engine | `scripts/path_policy.py` | `engine.py`（双重校验注入点） |

**冲突点**：
- P0-1 与 P0-2 同时修改 `data-engine/engine.py`（前者加 PIT 出口，后者加质量门出口）→ **需合并出口逻辑**
- P0-4 的 `path_policy.py` 需被 `reports-engine` 调用，但 P0-4 主开发在 `execution-monitor-engine` → **跨 Skill 依赖**
- 所有 P0 项的测试若同时运行，`sys.path` 注入会冲突（项目记忆中已记录的已知问题）

### 10.2 协调机制一：文件分工边界

**原则**：每个文件只有一个负责人，并行开发期间禁止交叉修改。

| 文件 | P0-1 负责 | P0-2 负责 | P0-3 负责 | P0-4 负责 |
|---|---|---|---|---|
| `data-engine/scripts/pit.py` | ✅ 主开发 | 只读 | 只读 | 只读 |
| `data-engine/scripts/quality_gate.py` | 只读 | ✅ 主开发 | 只读 | 只读 |
| `data-engine/scripts/adapters/*.py` | ✅ 主开发（追加 disclosure_date） | 只读 | 只读 | 只读 |
| `data-engine/engine.py` | ⚠️ 协作 | ⚠️ 协作 | 不涉及 | 不涉及 |
| `backtest-engine/scripts/rule_judge.py` | 只读 | 只读 | ✅ 主开发 | 只读 |
| `backtest-engine/engine.py` | 不涉及 | 不涉及 | ✅ 主开发 | 不涉及 |
| `execution-monitor-engine/scripts/path_policy.py` | 只读 | 只读 | 只读 | ✅ 主开发 |
| `execution-monitor-engine/engine.py` | 不涉及 | 不涉及 | 不涉及 | ✅ 主开发（atexit 注入） |
| `master/engine.py` | 不涉及（P0 阶段） | 不涉及 | 不涉及 | ⚠️ 协作（atexit 注册） |

**协作文件处理**：
- `data-engine/engine.py`：P0-1 与 P0-2 的出口逻辑合并到一个 `_validate_output()` 方法中，P0-1 先实现 PIT 出口，P0-2 在同一方法内追加质量门调用
- `master/engine.py`：P0-4 的 `atexit` 注册由 P0-4 负责人在 `MasterEngine.__init__` 中注入，与 P0-1/2/3 无冲突

### 10.3 协调机制二：sys.path 隔离

**问题**：项目记忆已记录"sub-skill engine.py causes module import conflicts; clean sys.path and reload root engine.py explicitly before each test run"。

**解决方案**：每个 Skill 的测试目录提供独立 `conftest.py`，统一模板：

```python
# tests/data-engine/conftest.py
"""data-engine 测试专用 conftest。

隔离原则：
1. 每个 Skill 测试启动前清理 sys.path 中其他 Skill 的注入
2. 重新加载当前 Skill 的 scripts 包
3. 避免 import 残留导致跨 Skill 测试污染
"""
import sys
import importlib.util as _ilu
import os
from pathlib import Path

_SKILL_ROOT = Path(__file__).parent.parent.parent / "skills" / "data-engine"
_SCRIPTS_DIR = _SKILL_ROOT / "scripts"


def _clean_stale_paths():
    """清理 sys.path 中其他 Skill 的 scripts 路径"""
    sys.path = [p for p in sys.path if "skills" not in p or "data-engine" in p]


def _reload_scripts_pkg():
    """重新加载当前 Skill 的 scripts 包"""
    init_py = _SCRIPTS_DIR / "__init__.py"
    if init_py.exists():
        spec = _ilu.spec_from_file_location(
            "scripts", str(init_py),
            submodule_search_locations=[str(_SCRIPTS_DIR)],
        )
        pkg = _ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)


def pytest_configure(config):
    """pytest 启动时统一初始化"""
    _clean_stale_paths()
    _reload_scripts_pkg()
```

**其他 Skill 的 conftest.py** 同构，只需修改 `_SKILL_ROOT` 中的 Skill 名称。

### 10.4 协调机制三：测试运行隔离

**问题**：并行开发时多个 Skill 测试同时跑会相互干扰。

**解决方案**：
1. **开发期间单 Skill 测试**：`pytest tests/data-engine/ -m "unit or contract"`（只跑单 Skill）
2. **集成测试隔离**：`pytest tests/integration/ -m integration`（集成测试单独运行，不与单元测试混跑）
3. **回归测试在 P0 阶段结束统一跑**：`pytest tests/ -m "not slow"`（P0 四项全部完成后统一回归）

**pytest marker 约定**（遵循现有工程约定）：
- `@pytest.mark.skill_data` / `skill_backtest` / `skill_execution` / `skill_master`
- `@pytest.mark.unit` / `contract` / `integration` / `slow` / `requires_network`

### 10.5 协调机制四：Git 分支策略

**分支模型**：
```
main
 ├── feature/p0-1-pit          (P0-1 开发分支)
 ├── feature/p0-2-quality-gate (P0-2 开发分支)
 ├── feature/p0-3-rule-judge   (P0-3 开发分支)
 └── feature/p0-4-frozen-core  (P0-4 开发分支)
```

**合并顺序**（P0 阶段结束）：
1. P0-1 先合并（最底层，无依赖）
2. P0-2 合并（依赖 P0-1 的 disclosure_date 字段做 PIT warning）
3. P0-3 合并（独立，无依赖）
4. P0-4 合并（依赖 master/engine.py 的 atexit 注入点稳定）

**冲突预防**：协作文件（`data-engine/engine.py`、`master/engine.py`）每天至少 rebase 一次 main 分支。

### 10.6 协调机制五：进度同步

**每日同步清单**（P0 阶段）：
- [ ] 各 P0 项新增文件清单（避免命名冲突）
- [ ] 协作文件（`data-engine/engine.py`、`master/engine.py`）当日 diff
- [ ] 新增环境变量清单（避免命名重复）
- [ ] 测试用例数 + 通过率

**P0 阶段完成标准**：
- 4 项全部合并到 main
- `pytest tests/ -m "not slow"` 100% 通过
- `pytest tests/ -m "skill_data or skill_backtest or skill_execution"` 覆盖率达标（关键路径 100%）

### 10.7 协调机制六：跨 Skill 依赖处理

**P0-4 跨 Skill 调用 reports-engine 的处理**：
- P0-4 在 `execution-monitor-engine/scripts/path_policy.py` 定义 `validate_changed_paths` 与 `GitChangeTracker`
- reports-engine 的 LLM 注入流程调用 `validate_changed_paths`，通过 `importlib.util.spec_from_file_location` 动态加载（避免 sys.path 污染）
- 调用方式封装为 `reports-engine/scripts/path_guard.py` 的薄包装：

```python
# reports-engine/scripts/path_guard.py
"""reports-engine 调用 path_policy 的薄包装。

避免直接 import execution-monitor-engine 的模块（会污染 sys.path），
改为动态加载并缓存模块引用。
"""
import importlib.util as _ilu
import os
from pathlib import Path

_CACHED_MODULE = None
_PATH_POLICY_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "execution-monitor-engine" / "scripts" / "path_policy.py"
)


def _load_path_policy():
    global _CACHED_MODULE
    if _CACHED_MODULE is not None:
        return _CACHED_MODULE
    spec = _ilu.spec_from_file_location("_path_policy", str(_PATH_POLICY_PATH))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHED_MODULE = mod
    return mod


def guard_llm_write(changed_paths: list) -> list:
    """LLM 写文件前调用，返回违规路径列表（空=合规）。"""
    mod = _load_path_policy()
    return mod.validate_changed_paths(
        changed=changed_paths,
        allowed=mod.ALLOWED_PATHS,
        forbidden=mod.FROZEN_PATHS,
    )
```

### 10.8 协调机制七：开发顺序细化

P0 四项虽并行，但内部仍有微调顺序：

```
Day 1-2:
  P0-1: 实现 pit_filter + adapter disclosure_date 追加
  P0-2: 实现 DataQualityGate 类（不接 data-engine 出口）
  P0-3: 实现 RuleJudge 类（不接 backtest-engine 出口）
  P0-4: 实现 path_policy + GitChangeTracker（不接 master/engine.py）

Day 3:
  P0-1: data-engine/engine.py 出口调用 pit_filter（先合并出口逻辑）
  P0-2: data-engine/engine.py 出口追加 DataQualityGate 调用（与 P0-1 协作）
  P0-3: backtest-engine/engine.py 出口调用 RuleJudge
  P0-4: master/engine.py 注入 atexit 钩子 + reports-engine 薄包装

Day 4:
  全部测试通过 + 回归测试
  合并到 main（顺序：P0-1 → P0-2 → P0-3 → P0-4）
```

---

## 十一、实施启动检查表

进入开发前最终确认：

- [x] PRD v1.1 已用户确认（2026-08-02）
- [x] 第九章 10 项决策已全部确认并记录
- [x] 第十章并行开发协调机制已细化
- [ ] 创建 4 个 feature 分支（feature/p0-1-pit 等）
- [ ] 各 Skill 的 conftest.py 模板就位
- [ ] 协作文件分工边界已通知各开发负责人
- [ ] 开发环境 Python 依赖已就位（pydantic V2、pytest-cov）

---

**PRD 确认完成，进入实际开发环节。**
