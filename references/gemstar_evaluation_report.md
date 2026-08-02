# GemStar 项目深度评估报告

> **评估对象**：[JustHappyLab/GemStar](https://github.com/JustHappyLab/GemStar)（2026-08-02 克隆版本）
> **评估目的**：分析 GemStar 项目可借鉴的代码、功能、流程、系统结构等设计，为 jingni-trader 演进提供决策参考
> **评估时间**：2026-08-02
> **评估方法**：源码精读 + 文档对照 + 与 jingni-trader 现状对比

---

## 一、项目概览

### 1.1 定位

GemStar 是一套面向 A 股的**量化研究 + 交易雷达**框架，按两条线组织：

- **生产链路**（确定性）：`gemstar trade` 只运行正式策略，强制关闭 LLM 策略生成，复用当日 production run，产出交易目标、状态快照和飞书中文提醒
- **研究链路**（探索性）：`gemstar research` 手动启用 LLM 策略生成/评审，产 draft/candidate，不自动进入生产

**核心设计哲学**（源自 `docs/gemstar-v2-opus-plan.md`）：

1. **PIT 锁死优先于性能**——任何 look-ahead 都会让其他指标失真
2. **状态变更只属于 Orchestrator（Python）**——LLM 与 Agent 只产 draft，由 Python 网关决定是否接受
3. **日常流水线不写代码**——Engineer/Bugfix Agent 移出 daily run，需人工批准
4. **抗过拟合是 RuleJudge 的硬门**——分段一致性、coverage、最低样本数都是 Python 检查
5. **可重放**——每个工件带 sha256，每次 run 写 `run_manifest.json`
6. **降级 > 崩溃**——核心数据缺失阻塞，非核心缺失降级
7. **预算先于野心**——LLM/回测/Tushare 调用上限全部写入配置

### 1.2 技术栈

| 依赖 | 用途 |
|------|------|
| Python 3.13 + uv | 运行时与包管理 |
| PyTorch | LSTM 模型训练 |
| pandas / numpy | 数据处理 |
| tushare | A 股数据源（单一数据源） |
| scikit-learn | 预处理 |
| Pydantic v2 | Schema 校验（全链路） |
| Claude Code CLI | LLM provider（唯一） |
| typer / rich | CLI |
| pyarrow | Parquet 读写 |
| matplotlib | 回测图表 |
| swanlab | 实验追踪 |

### 1.3 与 jingni-trader 的定位差异

| 维度 | GemStar | jingni-trader |
|------|---------|---------------|
| 定位 | 研究 + 实盘雷达双循环 | 主调度器 + 7 子 Skill |
| 数据源 | Tushare 单一 | 8 个 adapter 降级链 |
| LLM 角色 | 研究侧（手动启用） | reports-engine 内嵌 |
| 实盘能力 | 纯 paper trading | xtquant + gm 真实下单 |
| 状态机 | 14 状态 FSM + 7 状态 incident FSM | 7 段线性 STAGES |
| 工件存储 | SQLite + JSON + sha256 sidecar | workspace 目录归档 |
| 意图解析 | CLI 子命令 | 自然语言 + strategy_required 路由 |

---

## 二、核心架构亮点

### 2.1 双 FSM 编排（src/orchestrator/）

#### DailyFSM：14 状态主流程状态机

```
INITIALIZED → COLLECTING → QUALITY_CHECKING → FACTOR_MONITORING →
STRATEGY_IDEATION → STRATEGY_VALIDATION → BACKTESTING → JUDGING →
LEADERBOARD_BUILDING → REPORTING → COMPLETED
```

**关键设计**：

- **显式白名单转移表** `_ALLOWED_TRANSITIONS: dict[str, list[str]]`，非法跳转立即 `ValueError`
- **`DEGRADED` 是非终态**——有 3 条出路（恢复到 FACTOR_MONITORING / 提前到 REPORTING / 失败到 FAILED）
- **`MANUAL_ATTENTION` 是终态**——需人工介入
- 每次 transition 调用 `record_step(run_id, step_id, role, status)` 落库到 SQLite `steps` 表

#### IncidentFSM：7 状态故障自愈 FSM（独立于 DailyFSM）

```
DETECTED → CLASSIFIED → {RETRYING | DEGRADED | MANUAL_ATTENTION | ENGINEERING_TASK_CREATED | RESOLVED}
```

- **RETRYING ↔ CLASSIFIED 重试回路**——重试失败可重新分类
- 纯内存状态机（持久化由 `state_db.incidents` 表负责）
- 短生命周期，一个 incident 一个实例

### 2.2 三态数据质量门（src/data_quality/gate.py）

| Mode | 触发条件 | 流水线行为 |
|------|---------|---------|
| **abort** | 核心表缺失或 freshness > 10 交易日 | 转入 `manual_attention`，立即返回 |
| **degraded** | freshness > 5 交易日 或 PIT warning | 继续出报告，但禁止 promote 任何策略/因子 |
| **normal** | 全部通过 | 正常流程 |

**数据分级**：

- `CORE_TABLES`：trade_cal / stock_basic / daily / daily_basic / adj_factor / fina_indicator——缺失即 abort
- `OPTIONAL_TABLES`：forecast / express / news 等 14 张——缺失仅 warning，不影响 mode

**纯函数无副作用**：模块顶部声明 "Pure function"，不发起 Tushare 调用，只检查已拉取 DataFrame。

### 2.3 PIT 强制契约（src/data/pit.py）

```python
def pit_filter(df: pd.DataFrame, asof: str) -> pd.DataFrame:
    if "disclosure_date" not in df.columns:
        raise ValueError(...)
    return df[df["disclosure_date"] <= asof].copy()
```

**哨兵函数设计**——不复杂，但它的存在让任何使用 fina 数据的下游代码必须显式声明 asof 日期，否则报错。把 PIT 检查从"约定"提升为"强制契约"。DataQualityGate 中的 `_check_pit` 是运行时补充检查。

### 2.4 因子 DSL + AST 沙箱（src/factors/engine.py）

**这是整个项目最具技术含量的模块**。

#### 安全模型

通过 AST 白名单实现沙箱：

- `_ALLOWED_NODES` 限定为 Expression / Call / Name / Load / Constant / BinOp / UnaryOp / Compare 等基础节点
- **明确禁止**：属性访问（`ast.Attribute`）、下标（`ast.Subscript`）、推导式（`ast.ListComp`）、关键字参数、赋值
- `validate_expression` 在执行前遍历 AST，遇非白名单节点立即 `raise ValueError`

直接对应 Opus plan §2.3 "Prompt 注入隔离层"——LLM 生成的因子表达式必须经过这道关卡。

#### 三类算子

- **时序算子**（per ts_code group）：`ts_mean / ts_std / ts_max / ts_min / ts_sum / ts_delta / ts_pct_change / ts_delay / ts_rank / ts_zscore`
- **二元时序算子**：`ts_corr(a, b, window)`
- **截面算子**（per trade_date group）：`cs_rank` / `cs_zscore`
- **元素算子**：`abs / log / sign / sqrt / clip / where`

#### 递归求值器

`_Evaluator` 类递归遍历 AST，窗口参数必须是常量，最终结果替换 `±inf` 为 `nan`。

### 2.5 模板驱动因子挖掘（src/factors/miner.py）

**关键取舍**：函数签名保留 `llm_client: LLMGenerate` 参数，但函数体内 `_ = llm_client` 明确忽略它——架构上预留 LLM 入口，但实际挖掘完全由确定性模板驱动，避免"因子代码突变等价于在测试集上调参"的过拟合风险。

#### 三步评估流水线

1. `mine_factors` → 生成 `FactorProposal` 列表（~12 个模板：价格类/波动类/动量类/估值类/规模类/流动性类）
2. `evaluate_proposals` → 计算因子值 + IC 分析 + 三道准入门
3. `register_accepted` → 写入 `pool.candidates`（不是 active）

**三道准入门**：

- `min_ic_ir=0.3`（方向感知 IR：negative → `-IC_IR`，neutral → `|IC_IR|`，positive → `IC_IR`）
- `min_coverage=0.6`（非 NaN 比例）
- `max_redundancy=0.85`（与现有因子最大 |Pearson 相关|，样本数 ≥30）

### 2.6 五硬门 RuleJudge（src/judge/rules.py）

| 门 | 阈值 | 方向 |
|---|---|---|
| Sharpe | ≥ 1.0 | ≥ |
| Calmar | ≥ 0.8 | ≥ |
| Max Drawdown | ≤ 0.30 | ≤ |
| Completed Trades | ≥ 100 | ≥ |
| **Segment Sharpe IR Std** | ≤ 0.5 | ≤ |

第 5 门（分段一致性）是抗过拟合核心——把回测期切成 ~1 年段，要求各段 Sharpe 标准差 ≤ 0.5，防止靠某一年暴涨拉高整体指标。

**与 Reviewer 的职责分离**：

- `RuleJudge`（纯 Python）：输出 `VerdictV1(recommended_state="candidate" | "rejected")`
- `Reviewer`（LLM）：只产 `ReviewNotesV1`（explanation + risks），**不修改 verdict**

### 2.7 回测引擎的工程细节（src/engine/backtest.py + src/portfolio/）

#### A 股交易约束

| 约束 | 实现 |
|---|---|
| T+1 | `apply_t_plus_1` 通过 `bought_today` 集合追踪 |
| 涨跌停 | `check_limit_up_down` 用 open 价（非 close），20% ± 0.5% 容差 |
| 最小交易单位 | `int(x // 100) * 100` 向下取整 |
| 成交量限制 | `_cap_shares_by_volume` 单笔不超过当日成交量 25% |
| 现金约束 | `_fit_buy_shares_to_cash` 逐步减 100 股直到买得起 |

#### 分批成本核算（Lot Accounting，FIFO）

- 每次买入记录 lot：`{shares, cost_per_share}`
- 卖出时按 FIFO 扣减，计算 `net_sell_proceeds - basis_cost` 作为本次 PnL
- 一个完整"开仓-平仓"周期才计入 `trade_pnls`（用于胜率、盈亏比）

#### 交易成本模型（src/portfolio/cost.py）

- 佣金：`max(turnover * 0.00025, 5.0)`
- 印花税：仅卖出，**含政策变更分界点** `STAMP_TAX_CUTOFF = "20230828"`（前 0.001，后 0.0005）
- 滑点：买入价 `*1.0005`，卖出价 `*0.9995`

#### 执行顺序

`open 价估值 → 涨跌停检查 → 卖出（先）→ 买入（后）→ close 价 mark-to-market`——先卖后买保证现金充足，符合 A 股资金 T+0 可用规则。

### 2.8 策略 YAML + 注册表治理（src/strategies/）

#### 三维元数据管理

`StrategyRegistryV1` 用 `scope × lifecycle × source` 三维管理：

- `scope`：production / research
- `lifecycle`：draft → candidate → paper → active → retired/rejected
- `source`：manual / llm / promoted / imported

#### LLM 策略起草的边界（src/strategies/architect.py）

`_normalize_strategy_draft` 强制约束：

- `timer.mode` 强制设为 `"full"`（timing-policy 硬约束：AI 不能自由发明 LSTM 参数）
- universe 别名归一化
- 因子过滤：只保留 active ∪ candidate 中带 expression 的因子，权重必须为正
- 权重归一化到 sum=1

### 2.9 受控择时模板（src/timer/ + docs/timing-policy.md）

`TIMER_REGISTRY = {"full": build_full_signals, "ma": build_ma_signals, "lstm": build_lstm_signals}`

**白名单注册制**——只有三种 mode，未注册的 mode 直接 raise。LLM 无法通过 YAML 引入新 mode。

LSTM 关键设计：

- **walk-forward 训练**：按 `retrain_months` 分段，每段只用 `label_end_dates < retrain_date` 的样本（`<` 而非 `<=` 严格防泄漏）
- `MIN_LSTM_TRAIN_SAMPLES = 100`：样本不足直接返回 0% 仓位（fail closed）
- 模型架构写死（LSTM 64 → LSTM 32 → FC 16 → FC 3），LLM 完全不能改参数

### 2.10 Paper Trading 账本（src/live/ledger.py）

**最值得 jingni-trader 借鉴的设计之一**。

#### 追加式 JSONL

```python
def append_paper_trade(path, record):
    existing_ids = {r.execution_id for r in read_paper_trades(path)}
    if record.execution_id in existing_ids:
        raise ValueError(f"duplicate execution_id: {record.execution_id}")
    with p.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
```

#### 关键约束

- **append-only**：永不 in-place 修改
- **execution_id 唯一性**：写入前扫整文件去重
- **T+1 强制**：sell/reduce 当日买入的标的直接 `raise ValueError("T+1 restriction")`
- **position_after_shares 字段**：每次成交后持仓，跨日重放只需顺序读 JSONL 即可重建状态
- **confirmed=True 必填**：LLM 输出永远不能直接落账

`docs/state-storage.md` 明确：`alerts/ledger.jsonl` 是 paper trading 的事实底稿，`trade_status.json/md` 只是便利快照。

### 2.11 LLM 抽象层（src/llm/）

#### 三层抽象

```
LLMGenerate (Protocol, 2 方法) ← RoleLLMAdapter ← RoleRegistry
                                          ↓
                                    AgentProvider (ABC)
                                          ↓
                                    BaseCliProvider
                                          ↓
                                 ClaudeCodeProvider
```

- `LLMGenerate`：极简 Protocol，所有业务模块只依赖它，可被任何 mock 替换
- `BaseCliProvider`：支持 `json_schema` 约束 + `json_schema_unwrap_key`（Claude Code 要求 top-level object）
- `json_utils.py:loads_llm_json`：容错 JSON 解析，处理 LLM"我会输出 JSON 但忍不住加点解释"问题

### 2.12 工程自愈（src/engineering/）

#### 三件套

- `tasks.py`：异常/校验失败 → `EngineeringTaskV1`
- `executor.py`：执行 task，调用 LLM agent，校验 changed_paths
- `policy.py`：路径白名单/黑名单校验

#### 路径策略（核心安全机制）

`validate_changed_paths` 规则（顺序重要）：

1. 空路径 → 违规
2. `../` 或绝对路径 → 违规
3. **forbidden 优先于 allowed**——先检查 forbidden patterns
4. 不在 allowed 内 → 违规

**frozen core** 两层防护：

- prompt 层（软约束）：`role_skills/write_code/prompt.txt` 明确列出 frozen 文件
- policy.py forbidden_paths（硬约束）：`src/engine/**, src/judge/**, src/portfolio/cost.py, src/schemas/metrics.py, src/schemas/verdict.py`

#### GitChangeTracker

`subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"])`，10s 超时。要求 clean worktree（保护用户改动），post-run diff = after - preexisting。

### 2.13 双 sink 通知（src/notify/）

- `LocalFileNotificationSink`：默认兜底，append-only JSONL，无网络
- `FeishuNotificationSink`：可选增强，HMAC-SHA256 签名，`opener: UrlOpen = urlopen` 可注入做单测

`NotificationMessageV1`（`extra="forbid"`）：severity 三档（info/warning/critical），可选 decision_id / action / symbols。

### 2.14 Role Skill 三件套

```
role_skills/<role>/
├── prompt.txt    # 系统提示词（参与运行时）
├── sop.md        # 人类可读 SOP（不参与运行时）
└── schema.json   # 输出 JSON schema 约束
```

`schema.json` 三种格式：

- `object + schema_ref`：动态加载 Pydantic 模型生成 JSON Schema，传给 Claude Code `--json-schema`
- `file`：文件输出不走 schema 约束，由后续 import/test 验证
- `object + items_schema_ref`：array 输出自动 wrap 成 `{"items": [...]}`

### 2.15 状态持久化三件套（src/orchestrator/）

#### SQLite state_db（WAL 模式，7 张表）

| 表 | 用途 |
|---|---|
| runs | 运行生命周期 |
| steps | 每次状态转移的步骤记录（latency_sec / error） |
| artifacts | 产物索引（sha256 / schema_name） |
| strategies | 策略状态机 |
| factors | 因子状态机（ic_mean / ic_ir / coverage） |
| incidents | 故障记录 |
| **costs** | **LLM token/USD 成本追踪**（LLM 时代特征） |

#### artifact_store（sidecar manifest 模式）

- 数据文件 `<name>.json` + 元数据文件 `<name>.manifest.json`
- 元数据含 sha256 用于完整性校验和血缘追踪
- `inputs` 参数支持声明产物上游依赖，构建 DAG 血缘图

#### run_manifest

- `start_run`：`INSERT OR IGNORE` 一条 running 行（幂等）
- `record_step`：INSERT 而非 UPSERT，保留同一 step_id 多次状态变更轨迹
- `finalize_run`：UPDATE runs 行 + 聚合 step_statuses 到 `run_manifest.json`

### 2.16 panic finalize 三件套

注册 `SIGTERM`/`SIGINT` 信号处理器和 `atexit` 钩子，确保进程被杀时也能把 `running` 状态的 run 行 finalize 为 `failed`。`_finalized` 标志位防止重复 finalize。

### 2.17 Universe 预设 + auto 路由（src/orchestrator/universe.py）

4 基础 universe（a_share / chinext / star / main_board）× 2 后缀（_core / _liquid）+ auto 关键词路由。

`auto` 模式根据策略 name + hypothesis + source_idea + factor_ids 文本关键词匹配自动选池，降低配置心智。PIT 过滤按 trade_date 动态判断上市/退市状态，避免幸存者偏差。

### 2.18 跨策略因子框缓存

`ranking_factor_cache: dict[tuple, pd.DataFrame]`，键为 `(resolution.resolved, tuple(trade_dates), expression_factors)`。同一 universe+日期窗+表达式因子的策略共享一份 factor_df，避免重复计算。

---

## 三、与 jingni-trader 的详细对比

### 3.1 架构对比

| 维度 | jingni-trader | GemStar | 差异说明 |
|------|--------------|---------|---------|
| 调度模型 | 7 段线性 STAGES + MasterEngine | 14 状态 FSM + IncidentFSM | GemStar 状态更细，有降级回路 |
| 子模块组织 | 7 子 Skill（data/factor/strategy-model/backtest/portfolio-risk/execution-monitor/reports） | 23 个 src/ 子模块扁平组织 | jingni 更符合主调度器抽象层级 |
| 数据源 | 8 adapter 降级链 + 自动 pip install | Tushare 单一 | jingni 更鲁棒 |
| 实盘能力 | xtquant + gm 真实下单 + paper | 纯 paper trading | jingni 更进一步 |
| LLM 集成 | reports-engine 内嵌 + HTML placeholder 注入 | 研究侧独立 Role + Pydantic schema 强约束 | GemStar 边界更清晰 |
| 工件存储 | workspace 目录归档 | SQLite + JSON + sha256 sidecar | GemStar 可追溯性更强 |
| 意图解析 | 自然语言 + strategy_required 路由 | CLI 子命令 | jingni 更友好 |
| Schema 校验 | 弱（dict 传递） | 全链路 Pydantic V2 | GemStar 类型安全 |

### 3.2 jingni-trader 已有但 GemStar 缺失的能力

- **真实 broker 执行器**：xtquant/gm adapter
- **多数据源降级链**：8 adapter + 精准异常降级
- **LLM 内容 HTML 注入**：placeholder 机制，可后期回填
- **意图解析 + 单一工作流路由**：`strategy_required` 布尔驱动
- **运行归档目录结构**：`step_N_<阶段名>/summary.md + artifacts/`，对人工审计友好

---

## 四、可借鉴设计模式（按优先级排序）

### P0：强烈建议借鉴（防数据污染、防过拟合、防资金安全）

#### 4.1 PIT 强制契约

**问题**：jingni-trader 的 data-engine 输出未强制 PIT 检查，下游可能误用未来信息。

**借鉴方案**：

- 在 data-engine 增加 `pit.py`，提供 `pit_filter(df, asof)` 工具函数
- 把 `disclosure_date` 作为 fina 数据必填字段
- data-engine 出口加 `_check_pit` 扫描未来披露日
- 下游使用 fina 数据必须显式声明 asof

**收益**：从源头消除 look-ahead bias，这是量化系统底线。

#### 4.2 数据质量门三态机制

**问题**：jingni-trader 当前是"成功/失败"二元状态，非核心数据缺失会拖垮整条流水线。

**借鉴方案**：

- 在 data-engine 出口增加 `DataQualityGate`
- `normal` / `degraded` / `abort` 三态
- 核心表缺失 → abort + manual_attention
- 非核心表缺失 → degraded（继续出报告，但禁止 promote 策略）

**收益**：避免"缺一条新闻就崩溃"的过度保守，同时保留对核心数据缺失的硬阻塞。

#### 4.3 RuleJudge 硬门 + 分段一致性

**问题**：jingni-trader 的 backtest-engine 当前可能只看整体指标，过拟合风险高。

**借鉴方案**：

- 在 backtest-engine 增加 `RuleJudge`（纯 Python）
- 五硬门：Sharpe ≥ 1.0 / Calmar ≥ 0.8 / MDD ≤ 0.30 / 完成交易数 ≥ 100 / 分段 Sharpe IR Std ≤ 0.5
- 第 5 门（分段一致性）是抗过拟合核心——把回测期切成 ~1 年段，要求各段稳定

**收益**：低成本但有效的过拟合检测。

#### 4.4 工程自愈的 frozen core 保护

**问题**：jingni-trader 涉及真实下单，但缺少对自身代码改动的防护。

**借鉴方案**：

- 引入 `engineering/policy.py` 风格的路径策略
- `forbidden_paths` 明确列出下单核心：`src/execution/real_broker/**`、`src/risk/**`、`src/schemas/order.py`
- `GitChangeTracker` 用 `git status --porcelain` 做 pre/post diff 校验
- bugfix agent 只能改非核心文件

**收益**：防止 LLM agent 误改下单核心代码导致资金安全事故。

### P1：建议借鉴（提升可追溯性、可重放性、状态机鲁棒性）

#### 4.5 追加式 JSONL Paper Trading 账本

**问题**：jingni-trader 的 execution-monitor-engine 缺少强 schema 的事实底稿。

**借鉴方案**：

- 在 execution-monitor-engine 增加 `paper_trade_ledger.py`
- 定义 `PaperTradeRecordV1` Pydantic 模型：
  - `execution_id` 唯一
  - `confirmed=True` 必填
  - T+1 强制
  - `position_after_shares` 字段
- append-only 写 `workspace/paper_trades.jsonl`
- 每次启动重放 JSONL 重建持仓/现金/avg_cost/bought_today

**收益**：完整可重放的事务日志，跨日持仓跟踪可靠。

#### 4.6 双 FSM（DailyFSM + IncidentFSM）

**问题**：jingni-trader 用 `STAGES` 列表 + 简单 if/else 控制流程，无 transition 校验。

**借鉴方案**：

- 引入 `DailyFSM` 类，定义 `INITIALIZED → COLLECTING → ... → COMPLETED/FAILED/DEGRADED`
- 显式白名单转移表，非法跳转 raise
- `DEGRADED` 非终态，可恢复
- `IncidentFSM` 单独管理故障 lifecycle（含 RETRYING ↔ CLASSIFIED 重试回路）
- 状态变化写 `state.db`

**收益**：状态机更鲁棒，故障可追溯。

**注意**：jingni-trader 的 7 段已够用，不必照搬 14 状态。建议借鉴"降级 + 人工介入 + 故障 FSM"思路，核心状态数保持在 9-11 个。

#### 4.7 工件版本化 + sha256 sidecar manifest

**问题**：jingni-trader 的 Context.artifacts 是路径字典，缺少血缘和完整性校验。

**借鉴方案**：

- 每个工件带 `version: Literal["...V1"]` 字段
- `run_manifest.json` 记录输入工件 URI + sha256
- 同输入应得同输出（确定性回放）

**收益**：调试和审计极有帮助。

#### 4.8 Pydantic V1 Schema 全链路强校验

**问题**：jingni-trader 当前用 dict 在阶段间传递，少 schema。

**借鉴方案**：

- 定义 `OrderIntentV1`、`ExecutionReportV1`、`PositionSnapshotV1`、`RiskLimitV1`
- 所有子 Skill 间传递必须 validate
- `ConfigDict(extra="forbid")` + Field 约束（min_length/ge/le/multiple_of/pattern）

**收益**：避免字段拼写错误和类型 drift。

### P2：可选借鉴（提升开发体验、测试性、扩展性）

#### 4.9 因子 DSL + AST 沙箱

**问题**：若 jingni-trader 的 factor-engine 未来要支持 LLM 生成因子，必须防 prompt 注入。

**借鉴方案**：

- AST 白名单沙箱
- 三类算子（时序/截面/元素）
- 窗口参数必须是常量

**收益**：安全的因子表达式语言，支持 LLM 生成因子。

#### 4.10 模板驱动因子挖掘 + 三道准入门

**借鉴方案**：

- 确定性模板（非 LLM 自由生成）产出候选
- 三道准入门：min_ic_ir + min_coverage + max_redundancy
- 方向感知 IR
- 接受的因子进 candidates 桶（不是 active），需人工 promote

#### 4.11 双 sink 通知 + opener 注入测试

**借鉴方案**：

- `LocalFileNotificationSink` 默认兜底
- `FeishuNotificationSink` 可选增强，`opener: UrlOpen = urlopen` 可注入做单测
- 本地 JSONL 先写成功才允许飞书发送，飞书失败不阻断主流程

#### 4.12 受控择时模板（避免 LLM 自由生成模型参数）

**借鉴方案**：

- 定义 `ModelTemplateRegistry`，只允许命名模板
- LLM 只能推荐模板名，不能直接产 `epochs/lr/hidden_size`
- 模板参数写死，需人工 review + 回测证据才能新增

#### 4.13 Role Skill 三件套

**借鉴方案**：

- 为 factor-engine / strategy-model-engine / reports-engine 各定义一个 role
- prompt.txt + sop.md + schema.json 强约束输出
- reports-engine 的 LLM 输出可改为 `ReviewNotesV1` 风格 Pydantic 模型，再渲染到 HTML

#### 4.14 Universe 预设 + auto 路由 + PIT 过滤

**借鉴方案**：

- 4 基础 + 2 后缀 + auto 关键词路由
- PIT 过滤按 trade_date 动态判断上市/退市
- `describe_resolution` 生成人类可读描述直接进报告

#### 4.15 Skill Manifest（第三方集成）

**借鉴方案**：

- 增加 `integrations/jingni-skill/SKILL.md`
- 给外部宿主查询当前账户、持仓、回测结果、最新信号
- **只读 + 单次触发**，禁止实盘下单

#### 4.16 Engineering Task 工厂

**借鉴方案**：

- 引入 `task_from_exception`，根据异常类型分类成 `engineer / unsupported_capability` 或 `bugfix / code_bug`
- 自动构造 bounded engineering task，交由 LLM agent 修复
- `max_attempts` 限制防死循环

---

## 五、不建议照搬的部分

### 5.1 14 状态 FSM 过细

jingni-trader 的 7 段已够用，过度细分会增加转移表维护成本。建议借鉴"降级 + 人工介入 + 故障 FSM"思路，但保持核心状态数在 9-11 个。

### 5.2 result 大字典作为流程上下文

GemStar 用一个大字典（17+ 键）作为运行上下文容器，类型不安全，键渐进填充易出错。jingni-trader 若重构建议用 dataclass 或 pydantic Model 做运行上下文容器（jingni 已有 Context 对象，应保留并增强）。

### 5.3 record_step 每次开关连接

GemStar 每次 `record_step` 都 `connect(db_path)` 然后关闭，高频调用时连接开销大。jingni-trader 若引入应改用连接池。

### 5.4 IncidentFSM 不落库

GemStar 把 incident 持久化放在 `pipeline.py` 和 `state_db.incidents`，FSM 本身是内存的。若 incident 跨进程恢复需额外机制。

### 5.5 _strategy_inputs 吞掉 YAML 解析异常

GemStar 返回 explicit 值掩盖配置错误，应改为显式上报。jingni-trader 不应复制此模式。

### 5.6 单数据源

GemStar 只有 Tushare，jingni-trader 的 8 adapter 降级链是明显优势，应保留。

---

## 六、整合建议路线图

### 阶段一（P0，立即）：防数据污染 + 防过拟合 + 防资金安全

1. data-engine 增加 PIT 强制契约（`pit_filter` 哨兵函数 + `_check_pit` 扫描）
2. data-engine 增加三态数据质量门
3. backtest-engine 增加 RuleJudge 五硬门（含分段一致性）
4. execution-monitor-engine 增加 frozen core 路径策略保护

### 阶段二（P1，短期）：提升可追溯性 + 状态机鲁棒性

5. execution-monitor-engine 增加 paper trading JSONL 账本
6. 引入 DailyFSM + IncidentFSM（保持 9-11 状态）
7. 工件版本化 + sha256 sidecar manifest
8. Pydantic V1 Schema 全链路强校验（OrderIntentV1 / ExecutionReportV1 / PositionSnapshotV1 / RiskLimitV1）

### 阶段三（P2，中期）：提升开发体验 + 扩展性

9. 因子 DSL + AST 沙箱（若引入 LLM 因子生成）
10. 模板驱动因子挖掘 + 三道准入门
11. 双 sink 通知 + opener 注入测试
12. 受控择时模板
13. Role Skill 三件套
14. Universe 预设 + auto 路由

### 阶段四（P3，长期）：生态扩展

15. Engineering Task 工厂
16. Skill Manifest（第三方集成）

---

## 七、关键文件清单

### GemStar 源码（d:\codebuddy\GemStar\）

**编排层**：

- [src/orchestrator/pipeline.py](file:///d:/codebuddy/GemStar/src/orchestrator/pipeline.py) — 主流程编排器（~1057 行）
- [src/orchestrator/fsm_daily.py](file:///d:/codebuddy/GemStar/src/orchestrator/fsm_daily.py) — 14 状态日级 FSM
- [src/orchestrator/fsm_incident.py](file:///d:/codebuddy/GemStar/src/orchestrator/fsm_incident.py) — 7 状态故障自愈 FSM
- [src/orchestrator/state_db.py](file:///d:/codebuddy/GemStar/src/orchestrator/state_db.py) — SQLite 7 表 schema
- [src/orchestrator/artifact_store.py](file:///d:/codebuddy/GemStar/src/orchestrator/artifact_store.py) — 产物存储 + sha256 manifest
- [src/orchestrator/run_manifest.py](file:///d:/codebuddy/GemStar/src/orchestrator/run_manifest.py) — 运行清单
- [src/orchestrator/universe.py](file:///d:/codebuddy/GemStar/src/orchestrator/universe.py) — 股票池预设 + PIT 过滤

**数据层**：

- [src/data/fetcher.py](file:///d:/codebuddy/GemStar/src/data/fetcher.py) — Tushare 数据拉取
- [src/data/pit.py](file:///d:/codebuddy/GemStar/src/data/pit.py) — PIT 过滤哨兵
- [src/data_quality/gate.py](file:///d:/codebuddy/GemStar/src/data_quality/gate.py) — 三态数据质量门

**因子层**：

- [src/factors/engine.py](file:///d:/codebuddy/GemStar/src/factors/engine.py) — 因子 DSL + AST 沙箱
- [src/factors/miner.py](file:///d:/codebuddy/GemStar/src/factors/miner.py) — 模板驱动因子挖掘
- [src/factors/monitor.py](file:///d:/codebuddy/GemStar/src/factors/monitor.py) — 因子健康度监控

**回测层**：

- [src/engine/backtest.py](file:///d:/codebuddy/GemStar/src/engine/backtest.py) — 回测引擎
- [src/engine/metrics.py](file:///d:/codebuddy/GemStar/src/engine/metrics.py) — 绩效指标
- [src/portfolio/cost.py](file:///d:/codebuddy/GemStar/src/portfolio/cost.py) — 交易成本（含印花税分界点）

**策略层**：

- [src/strategies/architect.py](file:///d:/codebuddy/GemStar/src/strategies/architect.py) — LLM 策略起草 + 强制归一化
- [src/strategies/validator.py](file:///d:/codebuddy/GemStar/src/strategies/validator.py) — 策略验证门
- [src/strategies/registry.py](file:///d:/codebuddy/GemStar/src/strategies/registry.py) — 三维元数据管理

**评审层**：

- [src/judge/rules.py](file:///d:/codebuddy/GemStar/src/judge/rules.py) — 五硬门 RuleJudge

**实盘层**：

- [src/live/ledger.py](file:///d:/codebuddy/GemStar/src/live/ledger.py) — Paper Trading JSONL 账本
- [src/live/loop.py](file:///d:/codebuddy/GemStar/src/live/loop.py) — 实盘循环
- [src/live/signal_engine.py](file:///d:/codebuddy/GemStar/src/live/signal_engine.py) — 信号引擎（纯函数）

**LLM 层**：

- [src/llm/adapter.py](file:///d:/codebuddy/GemStar/src/llm/adapter.py) — RoleRegistry → LLMGenerate 桥接
- [src/llm/providers/base.py](file:///d:/codebuddy/GemStar/src/llm/providers/base.py) — AgentProvider 接口
- [src/llm/json_utils.py](file:///d:/codebuddy/GemStar/src/llm/json_utils.py) — 容错 JSON 解析

**工程自愈**：

- [src/engineering/policy.py](file:///d:/codebuddy/GemStar/src/engineering/policy.py) — 路径策略
- [src/engineering/executor.py](file:///d:/codebuddy/GemStar/src/engineering/executor.py) — 执行器
- [src/engineering/tasks.py](file:///d:/codebuddy/GemStar/src/engineering/tasks.py) — 任务定义

**通知层**：

- [src/notify/feishu.py](file:///d:/codebuddy/GemStar/src/notify/feishu.py) — 飞书通知（HMAC-SHA256 签名）
- [src/notify/local_file.py](file:///d:/codebuddy/GemStar/src/notify/local_file.py) — 本地文件 sink

**文档**：

- [docs/gemstar-v2-opus-plan.md](file:///d:/codebuddy/GemStar/docs/gemstar-v2-opus-plan.md) — V2 设计原则
- [docs/timing-policy.md](file:///d:/codebuddy/GemStar/docs/timing-policy.md) — 择时治理政策
- [docs/state-storage.md](file:///d:/codebuddy/GemStar/docs/state-storage.md) — 状态存储约定
- [docs/live-trading-roadmap.md](file:///d:/codebuddy/GemStar/docs/live-trading-roadmap.md) — 实盘路线图

### jingni-trader 现有实现（对比参考）

- [engine.py](file:///d:/codebuddy/jingni-trader/engine.py) — MasterEngine 主调度器
- [SKILL.md](file:///d:/codebuddy/jingni-trader/SKILL.md) — 项目说明
- [skills/data-engine/](file:///d:/codebuddy/jingni-trader/skills/data-engine) — 数据引擎（8 adapter）
- [skills/backtest-engine/](file:///d:/codebuddy/jingni-trader/skills/backtest-engine) — 回测引擎
- [skills/execution-monitor-engine/](file:///d:/codebuddy/jingni-trader/skills/execution-monitor-engine) — 执行监控（xtquant + gm + paper）
- [skills/reports-engine/](file:///d:/codebuddy/jingni-trader/skills/reports-engine) — 报告引擎（LLM 注入）

---

## 八、核心结论

### 8.1 GemStar 的核心价值

GemStar 是一个"**LLM 推理与 Python 确定性计算严格分离**"的量化研究流水线范例。其设计哲学可概括为：

1. **确定性优先，LLM 辅助**——实盘决策完全确定性，LLM 只用于研究阶段
2. **PIT 锁死是底线**——`pit_filter` 哨兵函数把 PIT 检查从约定提升为强制契约
3. **降级优于崩溃**——三态数据质量门 + DEGRADED 非终态 FSM
4. **LLM 不拥有状态变更权**——RuleJudge 是 Python 硬门，Reviewer 是 LLM 软解释
5. **可重放**——每个工件带 sha256，每次 run 写 manifest
6. **预算先于野心**——LLM/回测/数据调用上限全部配置化

### 8.2 对 jingni-trader 的核心建议

jingni-trader 应**保留自身优势**（多数据源降级链、真实 broker 执行器、自然语言意图解析、单一工作流路由、运行归档目录结构），**吸收 GemStar 的工程化设计**：

| 优先级 | 借鉴点 | 核心收益 |
|--------|--------|---------|
| P0 | PIT 强制契约 | 防数据污染（量化底线） |
| P0 | 三态数据质量门 | 避免过度保守崩溃 |
| P0 | RuleJudge 五硬门 + 分段一致性 | 抗过拟合 |
| P0 | frozen core 路径策略 | 防资金安全事故 |
| P1 | Paper Trading JSONL 账本 | 可重放事务日志 |
| P1 | 双 FSM + IncidentFSM | 状态机鲁棒性 |
| P1 | 工件版本化 + sha256 manifest | 可追溯性 |
| P1 | Pydantic Schema 全链路 | 类型安全 |
| P2 | 因子 DSL + AST 沙箱 | 安全的 LLM 因子生成 |
| P2 | 模板驱动因子挖掘 | 自动因子发现 |
| P2 | 双 sink 通知 | 通知可靠性 |
| P2 | 受控择时模板 | 防过拟合 |
| P2 | Role Skill 三件套 | LLM 输出强约束 |

### 8.3 整合原则

- **不照搬 14 状态 FSM**——jingni 的 7 段已够用，借鉴"降级 + 人工介入 + 故障 FSM"思路，保持 9-11 状态
- **不替换多数据源**——jingni 的 8 adapter 降级链是明显优势
- **不放弃真实 broker**——jingni 的 xtquant/gm 执行器比 GemStar 的纯 paper 更进一步
- **不复制 result 大字典**——保留 jingni 的 Context 对象，增强为 dataclass 或 pydantic Model
- **不复制单 LLM provider**——jingni 的 LLM 集成应保持灵活性

---

**报告完成。**

本报告基于对 GemStar 项目全部核心模块（src/ 下 23 个子模块、roles/、role_skills/、docs/）的源码精读，以及与 jingni-trader 现有实现的逐维度对比。所有可借鉴设计模式已按优先级排序并给出具体整合方案，可直接作为 jingni-trader 演进规划的输入。
