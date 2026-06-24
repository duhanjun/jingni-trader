# jingni-trader 整合版本目录索引

本分支整合了 15 个 feat/quant-opt-* 分支的优化代码。
每个分支的代码位于独立的目录中，互不冲突。

## 目录结构

| 目录 | 来源分支 | 主要优化方向 |
|------|---------|------------|
| `quant_opt_experiments/` | feat/quant-opt-20260615 | 因子表达式引擎+矢量化回测(48.9x)+IC稳定性+Walk-Forward |
| `quant_opt_20260615_trae/` | feat/quant-opt-20260615-trae | 因子表达式引擎(AST沙箱)+向量化回测(3.2x)+Brinson归因 |
| `research/quant-opt-20260616/` | feat/quant-opt-20260616-trae | 因子表达式(32算子)+TopK Dropout策略+Walk-Forward |
| `quant_opt_20260617/` | feat/quant-opt-20260617 | 向量化IC(HAC 17.97x)+回测(Numba JIT)+因子表达式(17算子) |
| `quant_opt_20260617_r2/` | feat/quant-opt-20260617-r2 | 向量化回测(19.7-32.9x)+WFO+Alpha158(44因子)+PIT |
| `quant_opt_20260618_r3/` | feat/quant-opt-20260618-r3 | Walk-Forward(过拟合检测)+因子DSL+前视偏差检测器(4类) |
| `quant_opt_20260619_m3/` | feat/quant-opt-20260619-m3 | 扩展绩效指标(14个)+因子表达式+A股T+1回测 |
| `optimizations_20260621_r2/` | feat/quant-opt-20260621-r2 | IC向量化(9.92x)+回测向量化(2.37x)+Bug复现 |
| `optimizations_20260622_v2/` | feat/quant-opt-20260622-v2 | IC(6.2x)/中性化(15.7x)/回测(12.7x)+22扩展指标 |
| `quant_opt_20260623_r2/` | feat/quant-opt-20260623-r2 | 向量化回测(T+1修复)+因子表达式+IC/中性化 |
| `optimizations_20260616/` | feat/quant-opt-20260616 | 事件驱动回测+因子表达式+Walk-Forward |
| `quant_opt_20260616_core/` | feat/quant-opt-20260616 | 动态加权IC-IR+向量化回测(7.4x)+PIT适配器 |
| `quant_opt_20260616/` | feat/quant-opt-20260616 | 因子表达式+绩效指标+Walk-Forward |
| `research_20260617/` | feat/quant-opt-20260617-agent-m3 | 向量化回测(5.0x)+因子IC/IR+Walk-Forward |
| `reports_20260617_agent_m3/` | feat/quant-opt-20260617-agent-m3 | 验证报告 |
| `quant_opt_20260618/` | feat/quant-opt-20260618 | 因子DSL(13算子+ALPHA158)+IC Decay+分位组合+bootstrap |
| `skills_quant_opt_20260618/` | feat/quant-opt-20260618 | IC Decay+分位分析+向量化回测+Walk-Forward |
| `reports_20260618/` | feat/quant-opt-20260618 | 验证报告 |
| `quant_opt_20260619/` | feat/quant-opt-20260619 | PIT+CPCV+记录器+YAML验证 |
| `quant_opt_20260619_extra/` | feat/quant-opt-20260619 | 多层风控引擎+意图解析器+向量化回测 |
| `docs_20260624/` | feat/quant-opt-20260624 | 优化报告 |
| `optimizations_20260624/` | feat/quant-opt-20260624 | 回测v2+风险v2+因子v2+Walk-Forward |
| `quant_opt_20260624/` | feat/quant-opt-20260624 | 因子筛选+交易所模拟+验证 |
| `tests_20260624/` | feat/quant-opt-20260624 | 测试 |
| `skills_backtest_opt_20260624/` | feat/quant-opt-20260624 | backtest-engine优化扩展 |

## 测试方法

每个目录通常包含独立的测试套件，可通过以下方式运行：

```bash
# 运行单个目录的测试
python -m pytest <目录名>/tests/ -v

# 或使用目录自带的运行器
python <目录名>/tests/run_all.py
```

## 注意事项

- 各目录代码独立，存在功能重复（如多个因子表达式引擎实现），这是有意为之
- 部分目录的 import 路径已从原始分支调整（如 `quant_opt` → `quant_opt_20260617`）
- 0624 分支的 `quant_opt_20260624/tests/` 存在已知坏代码（API 不匹配），运行测试会失败
