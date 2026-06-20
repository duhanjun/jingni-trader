# jingni-trader 量化优化验证报告

| 项目 | 内容 |
|------|------|
| 执行日期 | 2026-06-20 |
| 分支 | `feat/quant-opt-20260620` |
| 远程仓库 | https://github.com/duhanjun/jingni-trader |
| 验证代码目录 | `quant_opt_20260620/` |
| 测试结果 | **43/43 通过** |
| 是否合并 main | **否**（待用户确认） |

---

## 一、学习项目清单及核心亮点

通过联网搜索 GitHub Trending、Awesome Quant、arXiv、QuantConnect 等平台，筛选出以下 3 个最有借鉴价值的开源项目：

### 1. Microsoft Qlib（GitHub Star 36.5k+）

- **定位**：AI 驱动的量化投资全流程平台
- **核心亮点**：
  - **Point-in-Time (PIT) 数据库**：通过修订链（revision chain）结构存储财务数据，确保回测中任意时间点只使用当时已公开的数据版本，从根本上杜绝未来数据泄漏（look-ahead bias）
  - **全链路覆盖**：数据 → 因子 → 模型 → 回测 → 组合优化 → 执行，单框架串联
  - **Model Zoo**：内置 20+ SOTA 量化模型（LightGBM/Transformer/GATs/TFT 等）
  - **RD-Agent**：LLM 驱动的自动因子挖掘与模型优化
- **借鉴价值**：**PIT 数据库设计**对 jingni-trader 的数据引擎有直接借鉴意义

### 2. akquant（GitHub Star 1.5k+，2026 年活跃）

- **定位**：基于 Rust + Python 的高性能量化框架
- **核心亮点**：
  - **因子表达式引擎**：基于 Polars Lazy API，用字符串公式（如 `Rank(Ts_Mean(Close, 5))`）声明式定义因子，自动处理并行计算与数据对齐
  - **Alpha101 算子集**：支持 `Ts_Mean/Ts_Rank/Rank/Correlation/Delta/Delay` 等标准算子
  - **Walk-forward Validation**：内置滚动训练框架
  - **Zero-Copy 架构**：Rust 内核降低 Python 层开销
- **借鉴价值**：**因子表达式引擎**对 jingni-trader 的因子引擎有直接借鉴意义

### 3. WorldQuant Alpha101 因子体系

- **定位**：101 个公式化量价因子的标准实现
- **核心亮点**：
  - 用统一算子（rank/ts_rank/correlation/delta/sum 等）组合表达 101 个因子
  - 覆盖反转(45%)/动量(25%)/复合(20%)/自适应(5%)/行业中性(5%)五类逻辑
  - 证明了"因子 = 公式"的声明式范式的可行性
- **借鉴价值**：为因子表达式引擎提供标准算子集与验证基准

---

## 二、可借鉴的方向列表

对照 jingni-trader 现有代码结构（7 个子引擎），分析出以下改进方向：

| # | 借鉴来源 | jingni-trader 现状 | 优化方向 | 优先级 |
|---|---------|-------------------|---------|--------|
| 1 | akquant + Alpha101 | [factor-engine/engine.py](file:///workspace/skills/factor-engine/engine.py) 中 `compute_a_share_factors()` 硬编码每个因子 | **因子表达式引擎**：用字符串公式声明式定义因子，新增因子无需改代码 | 高 |
| 2 | Qlib PIT | [data-engine](file:///workspace/skills/data-engine/engine.py) 直接 merge 财务数据，无版本概念 | **PIT 数据层**：修订链结构，查询时只返回观测时点可得版本，防泄漏 | 高 |
| 3 | Qlib Model Zoo | [strategy-model-engine](file:///workspace/skills/strategy-model-engine/engine.py) 仅支持 LightGBM 等 | 扩展模型库（Transformer/GATs），统一模型接口 | 中 |
| 4 | akquant | [backtest-engine](file:///workspace/skills/backtest-engine/engine.py) 依赖外部框架适配器 | 向量化回测内核，减少外部依赖 | 中 |
| 5 | Qlib RD-Agent | 主调度器 [engine.py](file:///workspace/engine.py) 关键词匹配意图 | LLM 辅助因子挖掘与意图解析 | 低 |

本次验证聚焦于**优先级最高的方向 1 和方向 2**。

---

## 三、已完成的验证测试及结论

### 3.1 验证代码结构

```
quant_opt_20260620/
├── factor_expression_engine.py   # 因子表达式引擎（借鉴 akquant/Alpha101）
├── pit_data_layer.py             # PIT 数据层（借鉴 Qlib）
├── test_optimizations.py         # 测试套件（43 项）
└── test_results.json             # 测试结果
```

### 3.2 优化点 1：因子表达式引擎

**借鉴来源**：akquant 因子表达式引擎 + WorldQuant Alpha101 算子集

**设计要点**：
- 递归下降解析器将公式字符串解析为 AST
- 算子分三类：时序算子（`Ts_*`，按 code 分组）、横截面算子（`Rank/Scale`，按 date 分组）、逐元素算子（`Abs/Sign/Log`）
- 支持 16 个算子：`Ts_Mean/Ts_Sum/Ts_StdDev/Ts_Max/Ts_Min/Ts_Rank/Ts_ArgMax/Delay/Delta/Correlation/Covariance/Rank/Scale/Abs/Sign/Log/SignedPower`
- 内置 9 个 Alpha101 公式用于验证

**与现有代码对比**：
```python
# 现有 factor-engine: 硬编码
result['reversal_5d'] = -result['ret_5d']
result['turnover_20d'] = df.groupby('code')['turnover_rate'].transform(
    lambda x: x.rolling(20, min_periods=5).mean()
)

# 优化后: 声明式公式
engine.evaluate("-1 * Rank(Ts_Mean(Close, 20))", panel_df)
engine.evaluate("Rank(Correlation(Close, Volume, 20))", panel_df)
```

**测试结果**（22 项全通过）：

| 测试类别 | 测试数 | 通过 | 关键结论 |
|---------|--------|------|---------|
| 公式解析器 | 6 | 6 | AST 结构正确，支持嵌套/负号/二元运算 |
| 求值正确性 | 7 | 7 | 与手动 pandas 计算结果一致（`Ts_Mean/Rank/Delta/Delay/Correlation` 及复合公式） |
| Alpha101 公式集 | 9 | 9 | 9 个公式均能正常求值，非空率 91%-100% |
| 性能对比 | 1 | 1 | 引擎 54ms vs 手动 57ms（12500 行），开销可忽略 |
| 边界条件 | 6 | 6 | 空数据/单股票/大窗口/NaN/未知字段/语法错误均正确处理 |

### 3.3 优化点 2：PIT 数据层

**借鉴来源**：Microsoft Qlib PIT Database（`qlib/data/pit.py`）

**设计要点**：
- `PITStorage`：内存版文件存储模拟，按 `field → code → List[PITRecord]` 组织
- 每条记录含 `date`(发布日)/`period`(报告期)/`value`/`next_idx`(修订链下一跳)
- `query(field, code, observe_date)`：沿修订链找到发布日 ≤ 观测日的最新版本，多报告期取最近期
- `detect_lookahead_bias()`：对比 PIT 查询值与"直接用最新值"，统计泄漏次数

**与现有代码对比**：
```python
# 现有 data-engine: 直接 merge，存在泄漏风险
df = df.merge(financial_df, on=['code', 'date'], how='left')

# 优化后: PIT 查询，防泄漏
provider = PITProvider(storage)
panel = provider.get_feature_panel('revenue', codes, dates)
# 在 2024-09-15 查询 Q2 报告 → 只返回 8-30 发布的原始值，不含 10-15 修订值
```

**测试结果**（14 项全通过）：

| 测试类别 | 测试数 | 通过 | 关键结论 |
|---------|--------|------|---------|
| 防泄漏 | 4 | 4 | 修订前查询=原始值(100)，修订后=修订值(120)，发布前=None，泄漏检测命中 24.24% |
| 修订链 | 4 | 4 | 三次修订（快报→年报→修订）链表遍历正确 |
| 多报告期 | 2 | 2 | Q2 发布前取 Q1(300)，发布后取 Q2(350) |
| 面板查询 | 3 | 3 | 批量查询形状正确，存储统计正确 |
| 性能 | 1 | 1 | 50 股票 × 522 天 × 400 记录，耗时 51ms |

### 3.4 综合结论

| 维度 | 结论 |
|------|------|
| 正确性 | 43/43 测试通过，因子引擎与手动 pandas 完全一致，PIT 层防泄漏逻辑正确 |
| 性能 | 因子引擎开销可忽略（~1x），PIT 面板查询 51ms/2.6万行可接受 |
| 边界鲁棒性 | 空数据/单股票/大窗口/NaN/未知字段/语法错误均正确处理 |
| 可扩展性 | 新增因子仅需写公式字符串，无需改引擎代码；新增算子仅需注册到 FUNC_TABLE |
| 防泄漏 | PIT 层成功检测并阻止 24.24% 的未来数据泄漏 |

---

## 四、待用户确认的优化建议

以下优化方案已在 `feat/quant-opt-20260620` 分支验证通过，**尚未合并 main**，待用户确认：

### 建议 1：将因子表达式引擎集成到 factor-engine（推荐）

- **改动范围**：[skills/factor-engine/engine.py](file:///workspace/skills/factor-engine/engine.py)
- **改动方式**：在 `FactorEngine` 中新增 `compute_by_formula(formula)` 方法，保留现有 `compute_a_share_factors()` 兼容旧调用
- **收益**：因子库可由配置/LLM 生成，无需改代码；支持 Alpha101 全量因子
- **风险**：低（新增方法，不破坏现有接口）

### 建议 2：将 PIT 数据层集成到 data-engine（推荐）

- **改动范围**：[skills/data-engine](file:///workspace/skills/data-engine)
- **改动方式**：新增 `PITProvider` 适配器，财务数据查询走 PIT 路径，行情数据保持不变
- **收益**：消除财务因子的未来数据泄漏风险，提升回测可信度
- **风险**：中（需改造财务数据获取与存储流程，建议分阶段实施）

### 建议 3：扩展算子集与模型库（后续迭代）

- 补充 Alpha101 剩余因子（条件表达式 `If`、`Decay_Linear` 等）
- 参考 Qlib Model Zoo 集成 Transformer/GATs 时序模型
- 建议作为下一轮迭代

---

## 五、复现方式

```bash
# 切换到验证分支
git checkout feat/quant-opt-20260620

# 运行测试
python3 quant_opt_20260620/test_optimizations.py

# 预期输出: 汇总: 43/43 通过
```

---

## 六、约束遵守说明

- 所有新代码位于 `quant_opt_20260620/` 独立目录，**未修改 main 分支任何代码**
- 分支已推送到 GitHub 远程（`git push`），**未执行 merge / PR 合入**
- 待用户明确确认后，方可执行合并操作
