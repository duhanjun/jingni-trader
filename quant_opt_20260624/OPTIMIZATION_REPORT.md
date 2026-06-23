# jingni-trader 量化优化验证报告

> **执行日期**: 2026-06-24
> **分支**: `feat/quant-opt-20260624` (基于 main, 仅 push 未合并)
> **测试结果**: 47/47 通过
> **约束遵守**: 未执行任何 git merge 到 main 的操作

---

## 一、学习项目清单及核心亮点

通过 GitHub、量化社区(QuantConnect/BigQuant)、技术博客检索,筛选出近期活跃且高 Star 的量化交易开源项目,重点深入研究了 **2 个最具借鉴价值** 的项目:

### 1. Microsoft Qlib (⭐ 15k+, AI 量化研究平台)

**核心亮点**:
- **表达式 DSL + 算子注册表**: 因子以字符串表达式定义(`Mean($close, 20)`、`Ref($close,-2)/Ref($close,-1)-1`),解析为算子树,而非手写 if/elif 函数。算子通过 `OpsWrapper` 注册,用户可扩展。
- **三级缓存 (MemCache / ExpressionCache / DatasetCache)**: 按 `hash(instrument, expr, freq)` 落盘缓存。官方基准: 14 因子 × 800 股 × 2007-2020,从 HDF5 的 184s 降至 **7.4s**(约 25× 加速)。
- **DataHandlerLP 三视图设计**: `DK_R`(原始)/`DK_I`(推断)/`DK_L`(训练),配合 `infer_processors` vs `learn_processors` 的 `fit()/process()` 分离,**在训练段 fit、冻结后应用到测试段,消除前视偏差**。
- **Alpha158 T+1 感知标签**: 标签用 `Ref($close,-2)/Ref($close,-1)-1`(T+1 买、T+2 卖),而非 `Ref($close,-1)/$close-1`(假设 T 日收盘可买,不符合 A 股 T+1)。
- **YAML 工作流配置**: `qrun config.yaml` 端到端可复现运行。

### 2. RQAlpha (⭐ 7k+, A 股事件驱动回测/实盘引擎)

**核心亮点**:
- **EventBus + PRE/POST 自动包装**: 每个策略钩子(`before_trading`/`handle_bar`/`after_trading`/`settlement`)自动发布 PRE/POST 事件,风控/日志/进度 Mod 可拦截,无需改策略代码。
- **Generator 事件源**: `events()` 生成器 yield 出 `BEFORE_TRADING → BAR → AFTER_TRADING → SETTLEMENT`,**换生成器即换回测↔实盘**。
- **可替换抽象接口**: `AbstractEventSource`/`AbstractBroker`/`AbstractDataSource`/`AbstractPriceBoard`/`AbstractPersistProvider` 通过 `env.set_*` 启动时替换,实现回测-实盘一致性。
- **Position.today_closable vs closable**: 显式区分"今日新买(T+1 不可卖)"与"总持仓",是 T+1 最干净的建模方式。
- **FrontendValidator**: `validate_submission(order, account) → None|reason`,返回 None 通过、返回字符串则取消订单。可组合多个校验器,是可插拔盘前风控的最小 API。
- **结果契约**: pickle dict 含 `summary`(alpha/beta/sharpe/sortino/max_drawdown/IR/volatility/downside_risk/tracking_error) + `portfolios`/`positions`/`trades`/`plots`。

### 其他参考项目(未深入,仅作背景)
- vnpy (23k+, 国产实盘框架)、QuantConnect/Lean (15.5k, Alpha Streams 架构)、NautilusTrader (Rust/Cython, 5M rows/sec, 回测-实盘一致性标杆)、VectorBT (向量化参数扫描)、WonderTrader (C++ 高性能)。

---

## 二、可借鉴的方向列表

对照 jingni-trader 现有代码,识别出以下可借鉴方向:

| # | 借鉴来源 | jingni-trader 现状 | 可借鉴方向 | 优先级 |
|---|---------|-------------------|-----------|-------|
| 1 | RQAlpha | native_adapter 热循环内 `signals[signals['date']==dt]` 重复过滤 O(days×rows) | 预分组 dict 一次 O(rows),循环内 O(1) 查找 | 高 |
| 2 | RQAlpha | `pnl = sell_amount - cost` 是净现金流而非盈亏,胜率失真 | 引入 cost_basis 成本基准,真实 PnL = 卖出净额 − 成本基准×股数 | 高 |
| 3 | RQAlpha | T+1 靠"先卖后买"隐式保证,脆弱 | Position.today_closable 显式分离 | 高 |
| 4 | RQAlpha | benchmark 参数从未使用,无 alpha/beta/IR | 跟踪基准净值,计算相对指标 | 高 |
| 5 | RQAlpha | 卖出无滑点,停牌股静默跳过持仓残留 | 买卖双向滑点 + 停牌显式标记 | 中 |
| 6 | RQAlpha | 无盘前风控 | FrontendValidator 可插拔校验(涨跌停/单票上限) | 中 |
| 7 | Qlib | pandas_ta_calculator 逐股 `for code in unique()` 循环 | groupby().transform() 向量化 | 高 |
| 8 | Qlib | 因子 if/elif 硬编码,新增需改源码 | 表达式 DSL + 算子注册表,声明式定义 | 高 |
| 9 | Qlib | 每次全量重算因子 | 按 hash(expr, 股票池, 频率) 落盘缓存 | 高 |
| 10 | Qlib | 前向收益 `close[T+n]/close[T]-1` 假设 T 收盘可买(前视偏差) | T+1 标签 `close[T+2]/close[T+1]-1` | 高 |
| 11 | RQAlpha | portfolio-risk HRP `returns=pd.DataFrame()` 空表致优化无意义 | 传入真实收益率矩阵 | 高 |
| 12 | RQAlpha | CVaR/Barra 为占位 stub | 接入 EfficientCVaR 真实优化 | 中 |

---

## 三、已完成的验证测试及结论

### 验证代码位置
所有新代码位于 `quant_opt_20260624/` 独立目录,**未修改 main 分支任何文件**:
- [optimized_backtest.py](file:///workspace/quant_opt_20260624/optimized_backtest.py) — 优化回测引擎
- [factor_expression_dsl.py](file:///workspace/quant_opt_20260624/factor_expression_dsl.py) — 因子表达式 DSL
- [risk_fixes.py](file:///workspace/quant_opt_20260624/risk_fixes.py) — 风险引擎修复
- [test_verification.py](file:///workspace/quant_opt_20260624/test_verification.py) — 验证测试套件
- [test_results.json](file:///workspace/quant_opt_20260624/test_results.json) — 测试结果

### 测试汇总: 47/47 通过, 0 失败

| 测试组 | 测试数 | 结果 | 关键结论 |
|-------|-------|------|---------|
| 1. 回测正确性(优化版 vs 独立参考实现) | 5 | ✅ 全通过 | 终值、交易笔数、净值曲线长度一致;PnL 基于成本基准计算正确 |
| 2. 回测性能(预分组 vs 重复过滤) | 2 | ✅ 全通过 | **1.85× 加速**(8.83s vs 16.34s, 200股×500日×每日信号) |
| 3. 回测边界条件 | 12 | ✅ 全通过 | 空数据/单股/停牌/T+1/涨跌停/成本基准/加权成本均正确 |
| 4. 基准跟踪(alpha/beta/IR) | 6 | ✅ 全通过 | 新增 alpha/beta/IR/tracking_error/benchmark_return 指标 |
| 5. 因子 DSL 正确性(vs 手工 pandas) | 5 | ✅ 全通过 | Mean/Ref/复合表达式/RSI 与手工计算数值一致 |
| 6. 因子向量化性能(vs 逐股循环) | 1 | ✅ 通过 | **1.22× 加速**(0.59s vs 0.72s, 200股×300日) |
| 7. 因子缓存命中 | 2 | ✅ 全通过 | **7.98× 加速**(0.087s vs 0.011s),缓存文件正确生成 |
| 8. T+1 感知标签 | 3 | ✅ 全通过 | T+1 标签 = close[T+2]/close[T+1]-1,与 naive 标签不同(避免前视偏差) |
| 9. 风险引擎修复(HRP/CVaR) | 8 | ✅ 全通过 | HRP 权重和≈1、非负;CVaR 权重有效;VaR<CVaR 关系正确 |
| 10. 盘前风控校验器组合 | 3 | ✅ 全通过 | 涨停/超仓位被拒,正常订单通过,可组合 |

### 性能对比数据(实测)

```
回测引擎:  优化版 8.832s  vs  重复过滤 16.344s   →  1.85× 加速
因子计算:  向量化 0.591s  vs  逐股循环 0.723s    →  1.22× 加速
因子缓存:  首次 0.0868s  vs  命中 0.0109s       →  7.98× 加速
```

### 关键正确性验证

1. **PnL 修复**: 原实现 `pnl = sell_amount - cost`(cost=佣金+印花税)是净现金流,不是盈亏。优化版引入 `cost_basis`,真实 `PnL = sell_amount - commission - tax - cost_basis × shares`,胜率指标恢复意义。
2. **T+1 修复**: 原实现靠"先卖后买"顺序隐式保证,若同日同股出现买卖信号会失效。优化版 `Position.today_bought` 显式追踪,`closable = shares - today_bought`,日终 `settle_day()` 转换。测试验证: 同日买入+卖出信号,T+1 下卖出被阻止。
3. **T+1 标签修复**: 原前向收益 `close[T+1]/close[T]-1` 假设 T 日收盘可买,存在前视偏差。采用 Qlib Alpha158 标签 `close[T+2]/close[T+1]-1`(T 日出信号→T+1 买→T+2 卖),测试验证与 naive 标签不同。
4. **HRP 修复**: 原实现 `returns = pd.DataFrame()` 空表传入 `HRPOpt`,优化无意义。修复版传入真实历史收益率矩阵,测试验证权重和≈1、非负。

---

## 四、待用户确认的优化建议

以下优化已通过验证测试,**但尚未合并到 main**(遵守约束: 用户确认前禁止 merge)。建议合并优先级:

### 建议立即合并(高价值、低风险、已验证)

1. **回测 PnL 计算修复** ([optimized_backtest.py L220-221](file:///workspace/quant_opt_20260624/optimized_backtest.py))
   - 影响: 当前胜率指标完全失真,修复后才有意义
   - 风险: 极低,纯计算修正

2. **T+1 显式建模** ([optimized_backtest.py Position 类](file:///workspace/quant_opt_20260624/optimized_backtest.py))
   - 影响: 避免同日买卖信号下的 T+1 失效
   - 风险: 低,行为更正确

3. **HRP 空表 bug 修复** ([risk_fixes.py](file:///workspace/quant_opt_20260624/risk_fixes.py))
   - 影响: 当前 HRP 优化完全无效
   - 风险: 极低,修复明显 bug

4. **T+1 感知标签** ([factor_expression_dsl.py t_plus_1_label](file:///workspace/quant_opt_20260624/factor_expression_dsl.py))
   - 影响: 消除因子 IC 分析的前视偏差
   - 风险: 低,标签更严谨

### 建议评估后合并(中价值,需适配)

5. **回测预分组性能优化** (1.85× 加速)
   - 需替换 native_adapter.py 内部实现,接口不变
   - 建议作为 NativeAdapter 的增强版逐步切换

6. **因子表达式 DSL** (声明式 + 缓存 7.98× 加速)
   - 需评估是否替换现有 pandas_ta_calculator,或作为并行方案
   - DSL 更易扩展,但需补充算子覆盖面(目前 12 个算子)

7. **基准跟踪指标** (alpha/beta/IR/tracking_error)
   - 新增指标,需确认 reports-engine 是否消费

8. **FrontendValidator 盘前风控**
   - 可插拔设计,建议接入单票仓位上限校验

### 暂不建议合并(需更多设计)

9. **RQAlpha EventBus 架构** — 改动面大,建议单独立项评估回测-实盘一致性改造
10. **Qlib 三级缓存完整版** — 当前仅实现表达式级缓存,Memory/Dataset 级缓存需更大改造

---

## 五、合并操作说明

当前状态: 分支 `feat/quant-opt-20260624` 已推送到 GitHub 远程,**未合并到 main**。

如确认合并,请告知,届时将执行:
```bash
git checkout main
git merge feat/quant-opt-20260624
```

或通过 GitHub PR 流程合入。

---

## 六、附录: 文件清单

```
quant_opt_20260624/
├── optimized_backtest.py      # 优化回测引擎 (Position/FrontendValidator/OptimizedBacktestEngine)
├── factor_expression_dsl.py   # 因子表达式 DSL (算子注册表/解析器/缓存/T+1标签)
├── risk_fixes.py              # 风险修复 (HRP/CVaR/VaR)
├── test_verification.py       # 验证测试套件 (47 用例)
├── test_results.json          # 测试结果
└── OPTIMIZATION_REPORT.md     # 本报告
```
