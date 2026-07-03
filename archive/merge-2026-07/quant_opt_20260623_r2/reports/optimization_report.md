# jingni-trader 量化优化报告

> **执行日期**: 2026-06-23
> **分支**: `feat/quant-opt-20260623-r2` (远程已存在 `feat/quant-opt-20260623`，按仓库既有命名惯例加 `-r2` 后缀避免冲突，未 force-push)
> **状态**: 验证完成，待用户确认是否合并到 main

---

## 一、学习项目清单及核心亮点

本次联网调研覆盖 GitHub Trending、Awesome Quant、QuantConnect、社区博客等渠道，
重点研读了以下 3 个对 jingni-trader 最有借鉴价值的项目：

### 1. Microsoft Qlib (15k+ Stars)
- **定位**: AI 驱动的量化研究平台，聚焦 A 股与美股因子研究
- **核心亮点**:
  - **表达式引擎 (Expression Engine)**: 用字符串声明式定义因子，如 `Ref($close, 60) / $close`，
    支持 Ref/Mean/Std/Sum/Corr/Cov/Rank 等算子，零代码扩展因子库
  - **DataHandler / Processor 流水线**: 数据加载、清洗、标准化、CSRankNorm 等处理器解耦
  - **配置驱动工作流**: YAML 配置文件定义完整 qrun 流程
  - **Recorder 实验管理**: MLflow 风格的实验追踪
  - **Alpha158 / Alpha360 预置因子库**: 158/360 个开箱即用因子
- **文档**: https://qlib.readthedocs.io/

### 2. VectorBT (开源版 + PRO)
- **定位**: 超高速向量化回测引擎，秒级完成数千策略参数扫描
- **核心亮点**:
  - **矩阵化思维**: 将策略表示为 NumPy 多维数组，避免 Python 逐元素循环
  - **Numba JIT + Rust 内核**: 路径依赖逻辑用编译后端加速
  - **Portfolio.from_signals / from_orders**: 分层 API，从信号到组合模拟
  - **参数扫描**: `MA.run_combs` 一次测试所有参数组合，热力图可视化
  - **性能**: 比 Backtrader 快数百倍，数万模式秒级完成
- **文档**: https://vectorbt.dev/

### 3. Riskfolio-Lib
- **定位**: 组合优化与风险建模库
- **核心亮点**: 风险平价、Black-Litterman、层次风险平价 (HRP) 等多种优化器
- **对本项目启示**: portfolio-risk-engine 可借鉴其多优化器注册表设计

### 其他参考项目
| 项目 | Stars | 借鉴点 |
|------|-------|--------|
| vn.py | 23k+ | 实盘接口设计、多交易所适配 |
| RQAlpha | 7k+ | A 股回测 API 优雅设计 |
| Backtrader | 10k+ | 事件驱动回测架构 (已停止维护) |
| QUANTAXIS | 9k+ | 全栈中文量化平台 |

---

## 二、jingni-trader 现状分析与改进空间

通过逐行阅读 main 分支代码，识别出以下可优化点 (均有代码行号佐证):

### 2.1 回测引擎 (backtest-engine) — 性能与正确性问题

**问题 1: T+1 参数是死代码 (正确性 bug)**
- 位置: [native_adapter.py](file:///workspace/skills/backtest-engine/scripts/adapters/native_adapter.py#L17-L28)
- `run_backtest` 接受 `t_plus_1: bool = True` 参数，但函数体内**从未使用**该参数
- 验证: 测试证明 legacy 引擎 `t_plus_1=True` 与 `t_plus_1=False` 产生**完全相同**的净值曲线
- 影响: 用户以为启用了 T+1，实际未生效

**问题 2: O(N) 全表扫描 (性能)**
- 位置: [native_adapter.py#L46](file:///workspace/skills/backtest-engine/scripts/adapters/native_adapter.py#L46)
- `day_data = data[data['date'] == dt]` 每个交易日对全表做一次布尔扫描
- 对于 100 股 × 500 天 = 5 万行数据，回测循环 500 次，每次扫描 5 万行 → 2500 万次比较

**问题 3: iterrows 逐行迭代 (性能)**
- 位置: [native_adapter.py#L55](file:///workspace/skills/backtest-engine/scripts/adapters/native_adapter.py#L55)
- `for _, row in day_signal.iterrows()` 逐行处理信号，无法向量化

**问题 4: pnl 字段语义错误 (正确性)**
- 位置: [native_adapter.py#L115](file:///workspace/skills/backtest-engine/scripts/adapters/native_adapter.py#L115)
- 买入记录 `pnl = -buy_amount - commission`，这其实是"现金流出"而非"盈亏"
- 导致 `BaseBacktestMetrics.calc_win_rate` 基于 pnl 计算的胜率失真

### 2.2 因子引擎 (factor-engine) — 可扩展性问题

**问题 5: 因子硬编码，无扩展机制**
- 位置: [factor-engine/engine.py#L48-L117](file:///workspace/skills/factor-engine/engine.py#L48-L117)
- `compute_a_share_factors` 方法硬编码了 ~12 个因子，新增因子需修改源码
- 对比 Qlib: 因子用表达式字符串定义，零代码扩展

**问题 6: IC 分析逐日 Python 循环 (性能)**
- 位置: [factor-engine/engine.py#L250](file:///workspace/skills/factor-engine/engine.py#L250)
- `for dt in dates: spearmanr(cross)` 逐日循环计算 IC
- 对比: 可用 `groupby('date').apply` 一次向量化完成

**问题 7: 中性化逐日 Python 循环 (性能)**
- 位置: [factor-engine/engine.py#L148](file:///workspace/skills/factor-engine/engine.py#L148)
- `for dt in dates: LinearRegression().fit` 逐日循环做 OLS
- 对比: 可用 `groupby('date').resid` 向量化

### 2.3 架构层面

**问题 8: 模块重载 hack**
- 位置: [engine.py#L170-L172](file:///workspace/engine.py#L170-L172)
- `for key in list(sys.modules.keys()): if key == 'scripts'...: del sys.modules[key]`
- 通过删除 sys.modules 强制重载子 Skill，有副作用风险

**问题 9: 意图解析基于关键词 (脆弱)**
- 位置: [engine.py#L88-L101](file:///workspace/engine.py#L88-L101)
- `parse_intent` 用 `if any(kw in input_lower for kw in [...])` 匹配阶段
- 无实验追踪/Recorder 机制 (Qlib 有)

---

## 三、本次完成的验证测试

### 3.1 优化代码结构

```
quant_opt/
├── __init__.py
├── core/
│   ├── vectorized_backtest.py    # 向量化回测引擎 (借鉴 VectorBT)
│   ├── factor_expression.py      # 因子表达式引擎 (借鉴 Qlib)
│   └── vectorized_ic.py          # 向量化 IC 分析与中性化
├── tests/
│   ├── synthetic_data.py         # 合成数据生成器
│   ├── legacy_backtest.py        # legacy 参考实现 (复刻 main 分支)
│   └── test_optimization.py      # 完整测试套件
└── reports/
    ├── test_report.json          # 测试结果 (机器可读)
    └── optimization_report.md    # 本报告
```

### 3.2 测试结果汇总

**总计 21 项测试，全部通过 (21/21)**

| 类别 | 通过/总计 | 说明 |
|------|-----------|------|
| 正确性 (correctness) | 8/8 | 等价性、T+1、因子表达式、IC、中性化 |
| 性能 (performance) | 5/5 | 回测/IC/因子 三维度加速比 |
| 边界 (boundary) | 8/8 | 空数据、单股、涨停、语法错误等 |

### 3.3 关键测试结论

#### ① 向量化回测 vs legacy 等价性 (T+1 关闭)
- **结论**: 最大相对误差 < 2% (源于整手取整与资金分配细节)，**逻辑等价**
- legacy 0.042s → 向量化 0.026s

#### ② T+1 交割约束 (核心 bug 修复验证)
- **新引擎 T+1=on**: 同日买入+卖出 → 卖出被阻止 (0 笔同日卖出) ✓
- **新引擎 T+1=on**: Day1 买 + Day2 卖 → 卖出成功 (不过度阻止) ✓
- **legacy 引擎**: `t_plus_1=True` 与 `=False` 净值曲线**完全相同** → 证实参数是死代码 ✓

#### ③ 回测引擎性能加速 (规模越大加速越明显)

| 规模 | legacy | 向量化 | 加速比 |
|------|--------|--------|--------|
| 20股×120天 | 0.044s | 0.023s | **1.96x** |
| 50股×250天 | 0.156s | 0.045s | **3.48x** |
| 100股×500天 | 0.628s | 0.101s | **6.24x** |

> 规模越大，O(N) 扫描与 iterrows 的开销越显著，向量化优势越明显。
> 预计全 A 股 (5000+ 股) 回测可获 10x+ 加速。

#### ④ IC 分析向量化加速
- 逐日 scipy 循环 0.186s → 向量化 0.031s → **6.07x 加速**
- 数值精度: 与 scipy spearmanr 最大误差 < 1e-10

#### ⑤ 因子表达式引擎
- 14 个预置因子，表达式引擎 0.225s vs 手写 pandas 0.239s → **开销 0.94x** (几乎无额外开销)
- 数值精度: momentum_20 / vol_20 与手写公式最大误差 < 1e-10
- **关键收益**: 因子定义从硬编码改为配置字符串，零代码扩展

#### ⑥ 向量化中性化
- 与 sklearn LinearRegression 逐日结果最大误差 < 1e-6 ✓

#### ⑦ 边界条件全部通过
- 空数据、单只股票、全部涨停、表达式语法错误、未知算子、缺失字段、IC 空数据

---

## 四、可借鉴方向列表 (已完成 + 待确认)

### 已完成验证 (本分支)
| # | 方向 | 借鉴来源 | 验证状态 | 收益 |
|---|------|----------|----------|------|
| 1 | 向量化回测引擎 | VectorBT | ✓ 21/21 测试通过 | 2-6x 加速 + T+1 修复 |
| 2 | 因子表达式引擎 | Qlib | ✓ 正确性+性能通过 | 零代码扩展因子库 |
| 3 | 向量化 IC/中性化 | Qlib Processor | ✓ 正确性+性能通过 | 6x 加速 |

### 待用户确认的后续优化建议
| # | 方向 | 借鉴来源 | 预期收益 | 工作量 |
|---|------|----------|----------|--------|
| 4 | 修复 pnl 字段语义 | 通用最佳实践 | 胜率指标正确 | 小 |
| 5 | DataHandler/Processor 流水线 | Qlib | 数据处理可组合 | 中 |
| 6 | Recorder 实验追踪 | Qlib + MLflow | 实验可复现 | 中 |
| 7 | YAML 配置驱动工作流 | Qlib qrun | 减少硬编码 | 中 |
| 8 | 组合优化器注册表 | Riskfolio-Lib | 多优化器可插拔 | 中 |
| 9 | 移除 sys.modules hack | 通用最佳实践 | 消除副作用 | 小 |
| 10 | Numba JIT 加速路径依赖 | VectorBT | 进一步 10x+ 加速 | 大 |

---

## 五、约束遵守说明

- ✅ 所有新代码位于 `feat/quant-opt-20260623-r2` 分支的 `quant_opt/` 目录
- ✅ **未修改 main 分支任何代码** (legacy_backtest.py 是 main 逻辑的复刻，用于对比)
- ✅ 仅执行 `git push`，**未执行 git merge / PR**
- ✅ 等待用户明确确认后方可合并到 main

---

## 六、复现方式

```bash
# 切换到优化分支
git checkout feat/quant-opt-20260623-r2

# 安装依赖 (若未安装)
pip install numpy pandas scipy scikit-learn

# 运行完整测试套件
python -m quant_opt.tests.test_optimization

# 查看机器可读报告
cat quant_opt/reports/test_report.json
```

---

## 七、参考来源

- Qlib 文档: https://qlib.readthedocs.io/
- Qlib 论文: https://arxiv.org/abs/2009.11189
- VectorBT: https://vectorbt.dev/
- VectorBT Portfolio API: https://vectorbt.dev/api/portfolio/base/
- Awesome Quant: https://github.com/wilsonfreitas/awesome-quant
- 20+ Algo Trading Frameworks Reviewed: https://autotradelab.com/blog/nautilus-vs-vectorbt-vs-freqtrade-20-python-quant-trading-frameworks-compared
- 10 GitHub Repositories to Master Quant Trading: https://www.kdnuggets.com/10-github-repositories-to-master-quant-trading