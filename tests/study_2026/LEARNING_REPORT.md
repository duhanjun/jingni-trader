# Jingni-Trader 量化交易学习报告

---

## 报告编号: 2026-001
**日期**: 2026-06-14  
**分支**: feature/quant-stream-inspired  
**研究员**: AI Agent

---

## 一、学习项目清单及核心亮点

### 项目 1: Microsoft Qlib (⭐ 42k+)
- **仓库**: https://github.com/microsoft/qlib
- **核心定位**: AI 驱动的量化投资平台，面向因子挖掘、模型训练、回测评估的全流程
- **核心亮点**:
  1. **Expression Engine (表达式引擎)**: 使用 DSL 语法定义因子，如 `Ref($close, 20)`、`Mean($high - $low, 5)`。因子定义从"如何计算"转变为"声明要计算什么"，实现因子与代码解耦，也对 LLM 生成因子非常友好。
  2. **Point-in-Time (PIT) 数据库**: 确保在任何历史时间点只使用该时间点实际可用的数据。通过记录发布日期和报告期，支持财务数据的修订链管理，从根本上杜绝未来函数。
  3. **Alpha158/Alpha360 因子库**: 预设 158 个经过验证的 Alpha 因子，覆盖量价、基本面、技术指标等维度，可作为因子库的起点。
  4. **RD-Agent**: LLM 驱动的因子自动挖掘框架，通过 LLM 生成和迭代优化因子表达式，大幅降低因子挖掘的人力成本。
  5. **Columnar Binary Storage**: 列式二进制存储，极大提升数据处理速度。

### 项目 2: NautilusTrader (⭐ 20.7k)
- **仓库**: https://github.com/nautechsystems/nautilus_trader
- **核心定位**: 高性能事件驱动型量化交易平台，统一回测和实盘执行
- **核心亮点**:
  1. **Rust 核心 + Python 接口**: 用 Rust 实现核心执行引擎，Python 提供策略编写接口，兼顾性能与易用性。
  2. **统一回测与实盘**: 同一套代码无需修改即可在回测和实盘之间切换，消除"回测好但实盘差"的偏差。
  3. **事件驱动架构**: 完整的事件总线体系，支持 BarEvent、OrderEvent、FillEvent 等，便于扩展和监控。
  4. **组件化设计**: 策略、组合、风控、执行各自独立为组件，通过消息总线通信，实现高内聚低耦合。

### 项目 3: Backtesting.py (⭐ 16k+)
- **仓库**: https://github.com/kernc/backtesting.py
- **核心定位**: 轻量级交互式回测框架，适合快速原型验证
- **核心亮点**:
  1. **Interactive HTML 可视化**: 回测结果以交互式 HTML 呈现，支持缩放、平移、标注。
  2. **极简 API**: 简单策略只需几十行代码即可完成回测，学习成本极低。
  3. **内置统计指标**: Sharpe Ratio、最大回撤、胜率等一键计算。

### 参考论文: Chain-of-Alpha (arxiv 2025)
- **论文**: "Chain-of-Alpha: An LLM-based Alpha Mining Framework"
- **核心思路**: 双链架构（Search Chain + Evaluation Chain），LLM 自主生成和筛选 Alpha 因子
- **可借鉴**: 结合 jingni-trader 的因子表达式引擎，可实现 LLM 驱动的因子自动挖掘

---

## 二、可借鉴方向列表

| 序号 | 优化方向 | 借鉴来源 | 涉及模块 | 影响等级 | 可行性 |
|------|---------|---------|---------|---------|-------|
| 1 | 因子表达式引擎 (DSL 因子定义) | Qlib Expression Engine | factor-engine | 高 | 已验证 |
| 2 | Point-in-Time 数据系统 | Qlib PIT Database | data-engine | 高 | 已验证 |
| 3 | Purged K-Fold 交叉验证 + Embargo | Qlib/Lopez de Prado | strategy-model-engine | 中 | 已验证 |
| 4 | 列式二进制存储 | Qlib D.features | data-engine | 中 | 待验证 |
| 5 | 事件驱动回测架构 | NautilusTrader | backtest-engine | 高 | 待验证 |
| 6 | LLM 驱动因子挖掘 | Chain-of-Alpha | factor-engine | 中 | 待验证 |
| 7 | 交互式回测可视化 | Backtesting.py | backtest-engine | 低 | 可选 |
| 8 | 统一回测/实盘接口 | NautilusTrader | execution-monitor-engine | 中 | 待验证 |

---

## 三、已完成的验证测试及结论

### 测试 1: 因子表达式引擎 (test_factor_expression_engine.py)

**优化点**: 将现有的硬编码因子计算改为声明式 DSL 表达式定义

**测试结果**: 17 项测试全部通过
- 基础表达式: Field, Ref, Return, Mean - 计算结果与硬编码版本一致
- 组合表达式: 算术二元运算、嵌套表达式 - 正确
- 复杂因子: `(close - Mean(close, 20)) / Std(close, 20)` - 与硬编码一致
- 引擎集成: 注册、批量计算、依赖分析、回溯窗口计算 - 正常
- 边界条件: 空数据、单股票、NaN 处理 - 正常
- 可扩展性: 无需修改代码即可添加新因子 (示例: Alpha#1 简化版)

**性能对比** (25,000 行数据):
- 表达式引擎: 0.1301s (计算 7 个因子)
- 硬编码版本: 0.0483s (计算 5 个因子)
- 比值: 2.69x (表达式引擎约慢 2.7 倍，但换来了可配置性和可扩展性)

**结论**: 表达式引擎方案可行，性能损失可接受（可通过缓存和批量优化来弥补），核心价值在于因子的可配置性和 LLM 友好性。

### 测试 2: Point-in-Time 数据验证 (test_point_in_time_validation.py)

**优化点**: 建立 PIT 数据系统，防止回测中的前视偏差

**测试结果**: 10 项测试全部通过
- 基本插入和查询: 在发布日期后才能查询到数据
- 修订链处理: 修订前后返回不同版本的值
- 多股票/多字段管理: 正确隔离不同股票和字段
- 跨报告期查询: 返回观测时间前最新的可用数据
- 前视偏差检测: 能够自动检测因子与未来价格的相关性
- 财务数据时间线检查: 能够发现发布日晚于观测日的违规数据

**关键发现**: 在演示场景中，PIT 查询在 20250321 正确返回 600001.SH 的 ROE 为 None（数据尚未发布），而简单合并方式会错误地使用该数据——这就是前视偏差的典型来源。

**结论**: PIT 数据系统是消除前视偏差的关键基础设施，建议在 data-engine 中优先实现。

### 测试 3: Purged K-Fold 交叉验证 (test_purged_cross_validation.py)

**优化点**: 增强交叉验证，引入 Purge Gap 和 Embargo 机制

**测试结果**: 9 项测试全部通过
- 基本分割: 正确生成 5 个 fold
- 无日期重叠: 所有 fold 训练集和验证集日期完全分离
- Purge Gap 效果: 有 purge 的 fold 间隔 (11 天) vs 无 purge (4 天)，显著降低泄漏风险
- 分割信息表: 正确输出每个 fold 的起止日期、样本数、间隔天数
- Embargo 机制: 验证集之后的数据被排除出训练集
- Walk-forward 验证: 各 fold IC 在 -0.04 到 +0.03 之间波动，符合随机因子预期

**结论**: Purged K-Fold 是金融时序数据的标准做法，jingni-trader 现有的 `purged_group_ts_split` 方法可在此基础上增强 embargo 机制和更严格的 group split。

---

## 四、待用户确认的优化建议

### 优先级 P0 (建议立即实施)

**1. 引入因子表达式引擎到 factor-engine**
- 在 `skills/factor-engine/` 中新增 `expression_engine.py` 模块
- 保持向后兼容，现有硬编码因子可逐步迁移
- 新增因子配置文件（JSON/YAML），支持无代码添加因子
- 预计改动: 新增 1 个文件，修改 `engine.py`（增加表达式引擎适配层）

**2. 实现 PIT 数据系统到 data-engine**
- 在 `skills/data-engine/` 中新增 `pit_database.py` 模块
- 财务数据存储增加 publish_date 字段
- 因子计算时集成 PIT 查询，杜绝前视偏差
- 预计改动: 新增 1 个文件，修改数据获取和因子计算流程

### 优先级 P1 (建议近期实施)

**3. 增强 Purged K-Fold 的 Embargo 机制**
- 在现有 `purged_group_ts_split` 方法中增加 embargo 参数
- 增加更严格的 group split（按 code 分组）
- 预计改动: 修改 `strategy-model-engine` 中的交叉验证方法

**4. 研究 LLM 驱动的因子挖掘**
- 结合表达式引擎，让 LLM 生成因子表达式
- 自动回测评估因子有效性
- 预计改动: 新增 `factor-engine/scripts/llm_factor_miner.py`

### 优先级 P2 (可后续评估)

**5. 事件驱动回测架构重构**
- 借鉴 NautilusTrader 的事件驱动架构
- 评估是否值得重构现有回测引擎
- 预计改动: 较大，需谨慎评估

**6. 交互式回测结果可视化**
- 借鉴 Backtesting.py 的 HTML 可视化
- 为 reports-engine 增加交互式图表输出
- 预计改动: 新增可视化模块

---

## 五、验证文件清单

| 文件 | 测试数 | 状态 | 说明 |
|------|--------|------|------|
| `tests/study_2026/test_factor_expression_engine.py` | 17 | 全部通过 | 因子表达式引擎原型与验证 |
| `tests/study_2026/test_point_in_time_validation.py` | 10 | 全部通过 | PIT 数据系统与前视偏差检测 |
| `tests/study_2026/test_purged_cross_validation.py` | 9 | 全部通过 | Purged K-Fold + Embargo + Walk-forward |

---

## 六、重要约束提醒

1. **所有优化代码在用户确认前禁止执行 git commit/push/merge**
2. 验证代码位于独立测试文件中，未修改主代码
3. 当前工作在 `feature/quant-stream-inspired` 分支上
4. 确认后由用户主动告知可执行 git 操作