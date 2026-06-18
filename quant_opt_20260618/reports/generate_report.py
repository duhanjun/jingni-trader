"""
jingni-trader 量化优化验证报告
=============================

执行日期: 2026-06-18
执行分支: feat/quant-opt-20260618
报告生成: 自动化流程

本报告汇总：
  1. 联网学习成果
  2. jingni-trader 现状分析
  3. 借鉴点 → 优化方向的映射
  4. 验证测试结果
  5. 待用户确认的优化建议
"""
import json
import os
import sys
from datetime import datetime


REPORT_DATE = "2026-06-18"
BRANCH_NAME = "feat/quant-opt-20260618"


REPORT_MD = r"""
# jingni-trader 量化交易优化验证报告

> **执行日期**: 2026-06-18
> **执行分支**: `feat/quant-opt-20260618`
> **报告版本**: v1.0

---

## 1. 联网学习成果

### 1.1 调研对象

| 项目 | Stars | 核心亮点 | 与 jingni-trader 相关性 |
|------|-------|---------|---------------------|
| [Microsoft Qlib](https://github.com/microsoft/qlib) | ~30K | AI 量化平台；Point-in-Time 数据；Alpha158 因子库；Expression Engine；Rolling Training | ★★★★★ |
| [VeighNa/VNPY](https://github.com/vnpy/vnpy) | ~28K | A 股生态最完整；事件驱动；40+ 交易接口；CTA / 价差 / 期权策略 | ★★★ |
| [AKQuant](https://github.com/akfamily/akquant) | 新生 | Rust+Python 高性能；Polars 因子引擎；ML 集成；回测→实盘一键切换 | ★★★★ |
| [FinHack](https://github.com/FinHackCN/FinHack) | ~2K | A 股全栈；因子管理；回测；自动机器学习 | ★★★ |
| [Abu](https://github.com/bbfamily/abu) | ~12K | 多市场支持；ML 集成；可视化 | ★★ |
| [backtrader](https://github.com/mementum/backtrader) | ~12K | 经典回测框架；事件驱动；多数据源 | ★★ |
| [backtesting.py](https://github.com/kernc/backtesting.py) | ~5K | 轻量；Pythonic；高 star；在线回测 | ★★ |
| [Lean (QuantConnect)](https://github.com/QuantConnect/Lean) | ~10K | 云端量化；机构级；海量内置数据 | ★★ |
| [WorldQuant 101 Alphas](https://arxiv.org/abs/1501.00991) | 学术 | 101 个公式化 alpha 公式 | ★★★ |

### 1.2 三大重点学习项目

#### ① Qlib (Microsoft)
**核心设计**：
- **Point-in-Time (PIT) 数据系统**：所有基本面/公告类数据带 `announce_date`，join 时严格按发布日过滤
- **Expression Engine**：研究员用公式声明因子（`Rank(Mean($close, 5))`），引擎解析并计算
- **Rolling/Expanding 训练**：把全样本训练拆为多个 (train, test) 窗口，模拟真实"样本外"表现
- **Alpha158 因子库**：158 个公式化 alpha，覆盖动量/价值/质量/波动等
- **Handler + DataLoader 解耦**：数据源切换不影响上层策略
- **多层缓存**：原始数据 → 特征 → 标签 → 模型

**可借鉴之处**：
1. PIT 数据校验是 A 股研究的基础防线
2. 因子 DSL 显著提升可扩展性
3. WFA 评估比"全样本回测"更接近真实业绩

#### ② AKQuant
**核心设计**：
- **Polars 因子引擎**：相比 pandas 性能提升 5-20×
- **表达式因子声明**：字符串公式 → 高效执行
- **统一的 ML/回测/实盘框架**：策略写一次，回测和实盘共享代码

**可借鉴之处**：
- 表达式因子（与 Qlib 思路一致）
- 回测→实盘接口同构

#### ③ WorldQuant 101 Alphas
**核心设计**：
- 101 个公式化 alpha，每个都通过 PIT 校验
- 全部用表达式声明，不含硬编码逻辑

**可借鉴之处**：
- 公式化 alpha 可作为我们 DSL 的"内置库"

---

## 2. jingni-trader 现状分析

通过阅读源码（`engine.py`, `scripts/`, `skills/` 目录），识别出以下可优化点：

### 2.1 数据层
- **P0：缺乏 PIT 校验**：`data-engine` 直接读取原始数据，没有校验 `announce_date`/`publish_date` 类的发布时间列，存在未来泄露风险
- **P1：缓存粒度粗**：未见有精细化的多层缓存机制

### 2.2 因子层
- **P0：因子硬编码**：`skills/factor-engine/engine.py` 中的 `compute_a_share_factors()` 是 Python 硬编码的 if-else 列表，新增/修改因子必须改源码
- **P1：不可序列化**：因子逻辑无法保存为 YAML/JSON 配置
- **P1：难以做因子挖掘/遗传编程**

### 2.3 策略层
- **P1：API 较为底层**：研究员需要直接写 `data["mom_5"]` 这种列名

### 2.4 回测层
- **P0：全样本回测**：未实现 WFA，无法识别过拟合
- **P1：缺乏样本外评估框架**

### 2.5 风险层
- **P1：风险因子与策略因子耦合度低**：没有统一的因子评估管线

### 2.6 代码架构
- **P2：模块间通过 dict 传递 context**：类型不易追踪
- **P2：缺少集成测试**：现有 `test_engine_v3.py` 是端到端冒烟测试

---

## 3. 借鉴点 → 优化方向映射

| 借鉴来源 | 借鉴设计 | jingni-trader 优化方向 | 优先级 |
|---------|---------|--------------------|-------|
| Qlib PIT | Point-in-Time 校验 | 新增 PIT Checker 模块 | P0 |
| Qlib Expression Engine | 因子公式声明 | 新增 Factor DSL 模块 | P0 |
| Qlib Rolling Training | Walk-Forward 验证 | 新增 WFA Validator 模块 | P0 |
| Qlib Alpha158 | 内置因子库 | DSL 内置 alpha_expressions | P1 |
| AKQuant Polars | 高性能引擎 | (本轮仅做 PoC，未引入 Polars) | P3 |
| WQ 101 Alphas | 公式化 alpha | 作为 DSL 内置库的扩展 | P2 |

**本轮实现的 3 个 P0 优化方向**：
1. **PIT Checker** (借鉴 Qlib)
2. **Factor DSL** (借鉴 Qlib + AKQuant)
3. **Walk-Forward Validator** (借鉴 Qlib Rolling Training)

---

## 4. 验证测试结果

### 4.1 测试执行汇总

| 模块 | 测试数 | 通过 | 失败 | 用时 |
|------|------|-----|-----|-----|
| PIT Checker | 10 | 10 | 0 | 0.81s |
| Factor DSL | 15 | 15 | 0 | 0.88s |
| WFA Validator | 11 | 11 | 0 | 18.31s |
| **合计** | **36** | **36** | **0** | **~20s** |

✅ **全部通过**

### 4.2 性能基准测试

详见 `quant_opt_20260618/tests/benchmark_results.json`。

#### PIT Checker
```
  50×250 =  12,500 rows:  0.059s   (212,638 rows/s)   violations=1000
 100×500 =  50,000 rows:  0.055s   (904,510 rows/s)   violations=1000
 200×750 = 150,000 rows:  0.057s (2,630,042 rows/s)   violations=1000
```
**结论**：单线程即可达到 260 万行/秒，15 万行数据集只需 57ms。性能完全可用。

#### Factor DSL vs 硬编码
```
Hardcoded pandas: 0.135s
DSL engine:       0.143s
Ratio (DSL/Hard): 1.06x  ← 仅 6% 性能损失
Output shape: hard=(50000, 8), dsl=(50000, 8)
✓ mom_5 matches
✓ mom_20 matches
✓ vol_20 matches
✓ alpha_mom_rank matches
```
**结论**：
- DSL 性能损失仅 6%，可接受
- DSL 输出与硬编码输出**完全一致**（正确性验证）
- 获得的能力：可序列化、可动态扩展、支持嵌套与依赖图

#### WFA Validator
```
Generated 18 folds for 4-year span
WFA evaluation: 3.420s
IC mean avg:    -0.0033
IC IR:          -0.2382
Rank IC avg:    -0.0031
Consistency:    44.44%
Long-short total: -0.1225
Long-only total:  -0.0762
Avg turnover:     77.75%
```
**结论**：
- 4 年数据、18 个 fold、3.4s 完成，性能可接受
- 在 mock 数据上 IC 接近 0（合理，mock 数据 alpha 弱）
- Consistency 仅 44%（弱 alpha 的预期表现）

### 4.3 关键测试场景

#### PIT Checker 覆盖
- ✅ 干净数据全过
- ✅ 脏数据 100% 检测（`make_dirty_data` 中 30+ 条违规全部捕获）
- ✅ 同日公告（合法）边界
- ✅ NaT 公告日（忽略）边界
- ✅ 多 PIT 列同时检查
- ✅ PIT-safe merge 过滤未来
- ✅ 1 万行性能 < 1s
- ✅ 空 DataFrame 安全
- ✅ 自动检测 PIT 列
- ✅ 缺乏日期列的容错

#### Factor DSL 覆盖
- ✅ 基础算子：Ref / Mean / Std / Delta / Sum
- ✅ 横截面算子：Rank / Zscore
- ✅ 字段引用：`$close`
- ✅ 算术运算：`$close / Ref($close, 1) - 1`
- ✅ 嵌套表达式：`Rank(Mean($close, 5))`
- ✅ 因子间引用：`Rank(mom_20) - Rank(vol_20)`（含依赖图）
- ✅ 拓扑排序正确性
- ✅ 多因子注册（Alpha158 风格）
- ✅ 边界：空数据 / 未知字段 / 单只股票
- ✅ DSL vs 硬编码正确性比对

#### WFA Validator 覆盖
- ✅ Rolling 切分数量正确
- ✅ Anchored 切分数量正确
- ✅ Train/Test 区间不重叠
- ✅ Min train days 约束
- ✅ 空数据安全
- ✅ 强 alpha → 高 IC（> 0.01）
- ✅ 随机 alpha → IC ≈ 0
- ✅ 强 alpha → long-short 正收益
- ✅ 空 fold 安全
- ✅ **WFA vs In-sample 对比**：WFA 评估的 IC 显著低于 in-sample（证明 WFA 能识别过拟合）
- ✅ 1 万行 × 多个 fold 性能 < 5s

---

## 5. 对比分析：WFA vs 现有回测方式

### 5.1 现有方式
jingni-trader 当前回测链路：
```
[数据] → [因子] → [模型] → [回测（单次）] → [绩效]
                          ↑ 全部 in-sample
```

### 5.2 改进后
```
[数据] → [因子(PIT 校验)] → [模型] → [WFA 多次样本外] → [绩效]
                                      ↑ 更接近真实业绩
```

### 5.3 关键差异

| 维度 | 现有方式 | WFA 改进 |
|------|---------|---------|
| 评估类型 | in-sample (全样本) | 多次 OOS (样本外) |
| 过拟合识别 | 难 | 易（可观察 IC 一致性） |
| 实现复杂度 | 低 | 中等 |
| 性能开销 | 1× | 5-10×（多次评估） |
| 业绩可信度 | 低 | 中（更接近真实） |
| 与实盘差距 | 不可控 | 可控（可估算 degradation） |

---

## 6. 待用户确认的优化建议

### 6.1 立即可落地（已通过验证）

#### 建议 1：将 PIT Checker 集成到 `data-engine`
**改动**：
- 在 `data-engine` 读取数据后增加 `check_pit()` 调用
- 默认对所有 `announce_date`/`publish_date` 列启用 PIT 校验
- 校验失败时返回警告日志，但不阻塞（可配置）

**收益**：
- 防止未来数据泄露
- 一行代码即可启用

**风险**：低（向后兼容）

#### 建议 2：将 Factor DSL 作为 factor-engine 的可选后端
**改动**：
- 在 `factor-engine` 内部封装 `FactorEngine`
- 支持从 YAML/JSON 加载因子定义
- 保留 `compute_a_share_factors()` 作为 backward-compat 入口

**收益**：
- 因子可外部配置，无需改代码
- 支持因子挖掘 / 遗传编程
- 性能损失 6% 可接受

**风险**：中（API 扩展，不是破坏）

#### 建议 3：新增 WFA 验证流程
**改动**：
- 在 `backtest-engine` 中增加 WFA 模式
- 用户可通过配置 `evaluation_mode: "wfa"` 切换
- 报告输出 WFA 风格的 IC 一致性 / 多段收益

**收益**：
- 显著降低过拟合风险
- 业绩评估更接近真实

**风险**：中（新增模块，不影响旧流程）

### 6.2 中期可考虑（需要更多工作）

#### 建议 4：引入 Polars
- 在数据量大（>1000 万行）时启用
- 当前 pandas 方案已足够 50 万行以下规模

#### 建议 5：DSL 内置 101 Alphas
- 把 WorldQuant 101 Alphas 实现为 DSL 公式
- 提供 alpha101_expressions() 入口
- 便于用户快速体验 DSL

#### 建议 6：PIT-aware 数据集 API
- 改造 `data-engine` 暴露 `get_pit_safe_panel()`
- 自动处理财务数据发布日

### 6.3 长期规划

- 集成 ML 因子挖掘（XGBoost / LightGBM）
- 实时风险归因
- 多策略组合优化器

---

## 7. 文件清单

```
quant_opt_20260618/
├── README.md                       (本报告)
├── __init__.py
├── pit_checker/
│   ├── __init__.py
│   └── checker.py                  PIT 检查器实现（225 行）
├── factor_dsl/
│   ├── __init__.py
│   └── engine.py                   因子 DSL 实现（470 行）
├── wf_validator/
│   ├── __init__.py
│   └── splitter.py                 WFA 验证器实现（320 行）
└── tests/
    ├── __init__.py
    ├── test_pit_checker.py         PIT 单元测试（10 用例）
    ├── test_factor_dsl.py          DSL 单元测试（15 用例）
    ├── test_wf_validator.py        WFA 单元测试（11 用例）
    ├── run_all.py                  测试运行器
    ├── benchmark.py                性能基准
    └── benchmark_results.json      基准结果
```

**代码量统计**：
- 实现代码：~1015 行
- 测试代码：~700 行
- 文档/报告：本文件
- 合计：~1700 行

---

## 8. 分支与 Git 操作

- **分支名**：`feat/quant-opt-20260618`（遵守 YYYYMMDD 命名规范）
- **已提交**：✅ 全部 3 个模块 + 测试
- **已推送**：✅ 推送到 GitHub
- **合并到 main**：❌ 未合并（等待用户确认）
- **GitHub 链接**：见推送日志

---

## 9. 结论

### 9.1 已完成
- ✅ 3 个有借鉴价值的项目（Qlib / AKQuant / WQ101）调研
- ✅ 3 个 P0 优化方向实现（基于学习成果）
- ✅ 36 个测试用例，全部通过
- ✅ 性能基准达标（PIT 260 万行/秒，DSL 1.06× 慢）
- ✅ 完整验证报告

### 9.2 关键价值
1. **PIT Checker**：直接解决 A 股最常见的过拟合来源
2. **Factor DSL**：让因子库从"硬编码"升级到"可配置"，是后续因子挖掘的基础
3. **WFA Validator**：将回测从"in-sample"升级到"OOS"，业绩可信度显著提升

### 9.3 下一步
- 等待用户对建议 1-3 的确认
- 确认后即可合并到 main 分支
- 后续按建议 4-6 推进

---

**报告生成时间**：`2026-06-18`
**报告作者**：jingni-trader 自动学习流程
**联系方式**：GitHub Issues
"""


def main():
    out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(out_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"optimization_report_{REPORT_DATE}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(REPORT_MD)

    print(f"Report saved: {out_path}")
    print(f"Length: {len(REPORT_MD)} chars, {REPORT_MD.count(chr(10))} lines")


if __name__ == "__main__":
    main()
