# jingni-trader 量化优化验证报告

**执行日期**: 2026-06-23
**分支**: `feat/quant-opt-20260623`
**测试结果**: 12/12 通过 (100%)

---

## 一、联网学习成果

### 学习项目清单

通过搜索 GitHub Trending、Awesome Quant、arXiv、量化社区等，筛选出以下 3 个最具借鉴价值的项目：

| 项目 | Star | 类型 | 借鉴价值 |
|------|------|------|----------|
| **Microsoft Qlib** | 15k+ | AI 量化研究平台 | ★★★★★ 因子表达式引擎、Alpha158 因子库 |
| **VectorBT** | - | 向量化回测库 | ★★★★★ 向量化回测、参数扫描、防前视偏差 |
| **FactorEngine** (arXiv:2603.16365) | 论文 | LLM 因子挖掘 | ★★★★ LLM 驱动因子自动生成 |

### 核心亮点

#### 1. Microsoft Qlib
- **表达式引擎**: 用声明式表达式定义因子（如 `Mean(Ref($close, 1), 20)`），而非硬编码 Python 函数。因子可序列化、可被 LLM 自动生成。
- **Alpha158/Alpha360**: 系统化因子库，按价格/成交量/波动率等分类，带方向元信息。
- **Point-in-Time (PIT) 数据**: 严格防前视偏差。
- **分层架构**: Interface → Learning Framework → Execution → Data，松耦合。

#### 2. VectorBT
- **向量化回测**: 10,000 组参数回测 ~15 秒（事件驱动需数小时）。
- **信号对齐纪律**: 信号生成与执行严格分离，防前视偏差。
- **参数扫描原生支持**: 单次矩阵运算完成多参数组合评估。

#### 3. FactorEngine (论文)
- **程序级因子挖掘**: 将因子视为可执行、可审计的代码。
- **逻辑修订与参数优化分离**: LLM 负责方向性搜索，贝叶斯优化负责参数调优。
- **经验知识库**: 从失败中学习，轨迹感知优化。

---

## 二、jingni-trader 现有代码改进空间分析

对照 jingni-trader 现有代码，识别出以下改进点：

| 模块 | 现状问题 | 改进方向 | 借鉴来源 |
|------|----------|----------|----------|
| factor-engine | `compute_a_share_factors()` 硬编码 ~12 个因子，无法动态扩展 | 表达式引擎驱动，声明式因子定义 | Qlib |
| factor-engine | `_calc_ic()` 逐日 `for dt in dates` 循环，性能瓶颈 | groupby 向量化 IC 计算 | VectorBT |
| backtest-engine | `native_adapter.py` 逐日 `for dt` + iterrows() 事件循环 | 权重制向量化回测 | VectorBT |
| factor-engine | 因子无元信息（方向、类别），不利于管理与中性化 | FactorMeta 注册表 | Qlib |
| backtest-engine | 无参数扫描支持，单次回测 | 矩阵化批量参数评估 | VectorBT |

---

## 三、已完成的验证测试

### 3.1 因子表达式引擎（借鉴 Qlib）

**文件**: [engine.py](file:///workspace/optimizations/opt_20260623/factor_expression/engine.py)

**设计**:
- 递归下降解析器，支持 `Ref/Mean/Std/Sum/Max/Min/Rank/Delta/Ret/Corr/Scale` 算子
- `FactorMeta` 携带方向、类别、说明元信息
- 注册时即校验语法与算子合法性

**测试结果**:
- 表达式解析器基础语法: 4 个用例通过
- 与硬编码因子数值一致性: 12 个因子全部计算，`ret_5d`/`reversal_20d`/`volatility_20d` 数值差异 < 1e-6
- 因子可扩展性: 动态注册自定义因子 `Ret($close, 10) / Std(Ret($close, 1), 10)` 成功，非空值 500

### 3.2 向量化 IC 分析（借鉴 VectorBT）

**文件**: [engine.py](file:///workspace/optimizations/opt_20260623/vectorized_ic/engine.py)

**设计**:
- 用 `groupby(level='date').rank()` 预计算截面排名
- 用 `groupby(level='date').apply(lambda g: g['x'].corr(g['y']))` 一次性算出每日 IC 序列
- 替代原 `for dt in dates` 逐日循环

**性能对比** (100 股票 × 300 日 × 6 因子):

| 方法 | 耗时 | 加速比 |
|------|------|--------|
| 循环版 (基线) | 7.6016s | 1.0x |
| **向量化版** | **1.6735s** | **4.54x** |

**正确性**: ic_mean 最大差异 0.000000（完全一致）

### 3.3 向量化回测引擎（借鉴 VectorBT）

**文件**: [engine.py](file:///workspace/optimizations/opt_20260623/vectorized_backtest/engine.py)

**设计**:
- 信号转目标权重矩阵（date × code），截面选 top_k 等权
- T+1 信号对齐（`exec_weights = target_weights.shift(1)`）防前视偏差
- 权重制净值跟踪: `equity *= (1 + sum(holdings × daily_returns)) - costs`
- 涨跌停限制: 涨停日不买入、跌停日不卖出

**性能对比** (50 股票 × 250 日, top_k=15):

| 方法 | 耗时 | 加速比 |
|------|------|--------|
| 事件驱动版 (基线) | 0.5963s | 1.0x |
| **向量化版** | **0.3468s** | **1.72x** |

**产出合理性**: 总收益 17.58%，夏普 2.42，最大回撤 -4.59%

### 3.4 边界条件测试

全部 5 项通过:
- 空数据: 不报错，返回空结果
- 单只股票: 不报错
- 极短周期 (5 日): 因子含 NaN 但不报错
- 全涨停日: 涨跌停限制生效
- 非法表达式: 4 个非法表达式均被正确拒绝

---

## 四、待用户确认的优化建议

以下优化方案已通过验证测试，**尚未合并到 main 分支**，等待用户确认：

### 建议 1: 用因子表达式引擎替换硬编码因子（高优先级）

**现状**: [factor-engine/engine.py](file:///workspace/skills/factor-engine/engine.py) 的 `compute_a_share_factors()` 硬编码 12 个因子，新增因子需修改引擎代码。

**方案**: 采用表达式引擎，因子通过字符串配置定义，支持动态注册。便于 LLM 自动生成因子（对接 FactorEngine 论文思路）。

**收益**: 因子库可扩展性大幅提升，因子可序列化持久化。

### 建议 2: 向量化 IC 分析替换逐日循环（高优先级）

**现状**: [factor-engine/engine.py](file:///workspace/skills/factor-engine/engine.py#L242-L268) 的 `_calc_ic()` 逐日循环。

**方案**: 用 groupby 向量化实现，性能提升 4.54x。

**收益**: 大规模因子筛选（如 Alpha158 级别）耗时从分钟级降至秒级。

### 建议 3: 向量化回测作为研究阶段加速器（中优先级）

**现状**: [backtest-engine/scripts/adapters/native_adapter.py](file:///workspace/skills/backtest-engine/scripts/adapters/native_adapter.py) 事件驱动回测。

**方案**: 新增向量化回测适配器，作为研究阶段的快速验证工具。复杂路径依赖策略仍用事件驱动。

**收益**: 参数扫描场景性能提升 1.72x+，规模越大优势越明显。

### 建议 4: 引入因子元信息注册表（中优先级）

**现状**: 因子仅为 DataFrame 列名，无方向/类别元信息。

**方案**: 采用 `FactorMeta` 注册表，中性化与融合时利用方向信息。

**收益**: 因子管理规范化，支持自动方向调整。

---

## 五、文件清单

```
optimizations/opt_20260623/
├── factor_expression/
│   └── engine.py              # 因子表达式引擎
├── vectorized_ic/
│   └── engine.py              # 向量化 IC 分析
├── vectorized_backtest/
│   └── engine.py              # 向量化回测引擎
├── tests/
│   ├── data_generator.py      # 测试数据生成器
│   └── run_verification.py    # 验证测试套件
└── reports/
    ├── test_results.json      # 测试结果（机器可读）
    └── verification_report.md # 本报告
```

---

## 六、复现方式

```bash
cd /workspace
git checkout feat/quant-opt-20260623
pip install numpy pandas scipy
python -m optimizations.opt_20260623.tests.run_verification
```

---

## 七、约束遵守说明

- 所有优化代码位于 `feat/quant-opt-20260623` 分支，**未修改 main 分支任何代码**
- **未执行 git merge**，等待用户确认后方可合并
- 分支已推送到 GitHub 远程仓库（仅 push，不合并）
