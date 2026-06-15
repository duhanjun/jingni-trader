# quant_opt — 量化交易优化研究

> 分支: `feat/quant-opt-20260615`
> 创建日期: 2026-06-15
> 状态: 等待用户确认合并到 main

## 这是什么

从 4 个高 Star 量化开源项目 (Qlib 42.5K / backtesting.py 6.1K / QUANTAXIS 3.5K / vnpy 30K+) 中
提炼的可立即落地的 4 个优化模块。所有代码位于独立目录,**不修改 main 分支的现有代码**。

## 模块清单

| 模块 | 借鉴来源 | 行数 | 关键能力 |
|---|---|---|---|
| `expression_engine/` | Qlib 表达式引擎 | 450 | 一行代码注册 Qlib 风格因子 |
| `walk_forward/` | Qlib TrainerRM | 200 | 滚动前向验证 + 多维 IC 评估 |
| `look_ahead_detector/` | backtesting.py progressive | 260 | AST + 表达式 + IC 三层前视偏差检测 |
| `ic_optimizer/` | Qlib D.features + vectorbt | 180 | 向量化 IC 计算,8.5x 加速 |

## 快速验证

```bash
cd /data/user/skills/jingni-trader
python3 quant_opt/tests/run_all_tests.py
```

预期输出: 4 个测试全部 PASS,关键指标
- T1: 23 因子 25,000 行 = 0.165s,与硬编码结果误差 0.00e+00
- T3: IC 计算 8.5x 加速
- T4: 15 fold, IC IR=0.73, win rate 80%

## 详细报告

见 [reports/verification_report_20260615.md](reports/verification_report_20260615.md)

## 约束

- 本分支代码**不**合并到 main,等待用户明确确认
- 编译、测试、git push 等操作不受限制

## 作者

jingni-trader 学习 Agent
