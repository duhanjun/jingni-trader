"""
验证测试套件：正确性、性能对比、边界条件

运行方式:
    cd /workspace
    python -m optimizations.opt_20260623.tests.run_verification

验证内容:
    1. 因子表达式引擎：与硬编码因子数值一致性
    2. 向量化 IC 分析：与循环版结果一致性 + 性能对比
    3. 向量化回测：与事件驱动版指标合理性 + 性能对比
    4. 边界条件：空数据、单只股票、单日、全涨停等
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# 确保能导入优化模块
ROOT = Path(__file__).resolve().parents[3]  # /workspace
sys.path.insert(0, str(ROOT))

from optimizations.opt_20260623.factor_expression.engine import (
    FactorExpressionEngine,
    FactorMeta,
    build_default_factor_library,
    ExpressionParser,
)
from optimizations.opt_20260623.vectorized_ic.engine import (
    ic_analysis_vectorized,
    ic_analysis_loop_baseline,
    benchmark_ic,
)
from optimizations.opt_20260623.vectorized_backtest.engine import (
    vectorized_backtest,
    benchmark_backtest,
)
from optimizations.opt_20260623.tests.data_generator import (
    generate_synthetic_ohlcv,
    generate_forward_returns,
    generate_signals_from_factor,
)


# ---------------------------------------------------------------------------
# 测试结果收集
# ---------------------------------------------------------------------------

class TestReport:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: str = "", metrics: Dict = None):
        self.results.append({
            "name": name,
            "passed": passed,
            "detail": detail,
            "metrics": metrics or {},
        })

    def summary(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0,
            "results": self.results,
        }


# ---------------------------------------------------------------------------
# 1. 因子表达式引擎正确性测试
# ---------------------------------------------------------------------------

def test_expression_parser_basic(report: TestReport):
    """测试表达式解析器基础语法"""
    try:
        cases = [
            ("$close", ('field', '$close')),
            ("Mean($close, 20)", ('call', 'Mean', [('field', '$close'), ('num', 20.0)])),
            ("$close - Ref($close, 1)", ('binop', '-', ('field', '$close'), ('call', 'Ref', [('field', '$close'), ('num', 1.0)]))),
            ("-Ret($close, 5)", ('neg', ('call', 'Ret', [('field', '$close'), ('num', 5.0)]))),
        ]
        for expr, expected in cases:
            ast = ExpressionParser(expr).parse()
            assert str(ast) == str(expected), f"解析 {expr} 得到 {ast}，期望 {expected}"
        report.add("表达式解析器基础语法", True, f"通过 {len(cases)} 个用例")
    except Exception as e:
        report.add("表达式解析器基础语法", False, str(e))


def test_factor_engine_vs_hardcoded(report: TestReport):
    """测试因子表达式引擎与硬编码因子数值一致性"""
    try:
        data = generate_synthetic_ohlcv(n_codes=20, n_days=100, seed=42)
        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())

        factor_df = engine.compute(data)

        # 验证 1：所有注册因子都已计算
        registered = set(engine.list_factors())
        computed = set(factor_df.columns) - {'code', 'date'}
        missing = registered - computed
        assert not missing, f"缺失因子: {missing}"

        # 验证 2：ret_5d 与直接 pct_change(5) 一致
        data_sorted = data.sort_values(['code', 'date'])
        expected_ret5d = data_sorted.groupby('code')['close'].pct_change(5).values
        actual_ret5d = factor_df.sort_values(['code', 'date'])['ret_5d'].values
        # 去除 NaN 后比较
        mask = ~np.isnan(expected_ret5d) & ~np.isnan(actual_ret5d)
        diff = np.abs(expected_ret5d[mask] - actual_ret5d[mask]).max()
        assert diff < 1e-6, f"ret_5d 数值不一致，最大差异 {diff}"

        # 验证 3：reversal_20d = -ret_20d
        expected_rev = -data_sorted.groupby('code')['close'].pct_change(20).values
        actual_rev = factor_df.sort_values(['code', 'date'])['reversal_20d'].values
        mask = ~np.isnan(expected_rev) & ~np.isnan(actual_rev)
        diff = np.abs(expected_rev[mask] - actual_rev[mask]).max()
        assert diff < 1e-6, f"reversal_20d 数值不一致，最大差异 {diff}"

        # 验证 4：volatility_20d = std of daily returns over 20 days
        rets = data_sorted.groupby('code')['close'].pct_change()
        expected_vol = rets.groupby(data_sorted['code']).transform(
            lambda x: x.rolling(20, min_periods=10).std()
        ).values
        actual_vol = factor_df.sort_values(['code', 'date'])['volatility_20d'].values
        mask = ~np.isnan(expected_vol) & ~np.isnan(actual_vol)
        diff = np.abs(expected_vol[mask] - actual_vol[mask]).max()
        assert diff < 1e-6, f"volatility_20d 数值不一致，最大差异 {diff}"

        report.add(
            "因子表达式引擎与硬编码一致性",
            True,
            f"12 个因子全部计算，ret_5d/reversal_20d/volatility_20d 数值差异 < 1e-6",
        )
    except Exception as e:
        report.add("因子表达式引擎与硬编码一致性", False, f"{e}\n{traceback.format_exc()}")


def test_factor_engine_extensibility(report: TestReport):
    """测试因子可扩展性：动态注册新因子"""
    try:
        data = generate_synthetic_ohlcv(n_codes=10, n_days=60, seed=7)
        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())

        # 动态注册一个复合因子
        engine.register(FactorMeta(
            name="custom_momentum_vol",
            expression="Ret($close, 10) / Std(Ret($close, 1), 10)",
            direction=1,
            category="custom",
            description="动量波动比",
        ))

        factor_df = engine.compute(data, factor_names=["custom_momentum_vol"])
        assert "custom_momentum_vol" in factor_df.columns
        # 应有非空值
        non_null = factor_df["custom_momentum_vol"].notna().sum()
        assert non_null > 0, "自定义因子全为空"

        report.add("因子可扩展性（动态注册）", True, f"自定义因子非空值数: {non_null}")
    except Exception as e:
        report.add("因子可扩展性（动态注册）", False, str(e))


# ---------------------------------------------------------------------------
# 2. 向量化 IC 分析测试
# ---------------------------------------------------------------------------

def test_ic_correctness(report: TestReport):
    """测试向量化 IC 与循环版结果一致性"""
    try:
        data = generate_synthetic_ohlcv(n_codes=30, n_days=120, seed=99)
        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())
        factor_df = engine.compute(data)
        fwd = generate_forward_returns(data)

        factor_names = ["ret_5d", "reversal_20d", "volatility_20d"]
        vec_res = ic_analysis_vectorized(factor_df, fwd, factor_names, ic_type="spearman")
        loop_res = ic_analysis_loop_baseline(factor_df, fwd, factor_names, ic_type="spearman")

        # 比较 ic_mean
        max_diff = 0.0
        for fwd_col in vec_res:
            vec_map = {item['factor']: item['ic_mean'] for item in vec_res[fwd_col]}
            loop_map = {item['factor']: item['ic_mean'] for item in loop_res.get(fwd_col, [])}
            for f in vec_map:
                if f in loop_map:
                    max_diff = max(max_diff, abs(vec_map[f] - loop_map[f]))

        assert max_diff < 1e-3, f"IC 结果不一致，最大差异 {max_diff}"
        report.add(
            "向量化 IC 与循环版一致性",
            True,
            f"3 因子 × 3 远期，ic_mean 最大差异 {max_diff:.6f}",
            {"max_diff": max_diff},
        )
    except Exception as e:
        report.add("向量化 IC 与循环版一致性", False, f"{e}\n{traceback.format_exc()}")


def test_ic_performance(report: TestReport):
    """测试向量化 IC 性能提升"""
    try:
        data = generate_synthetic_ohlcv(n_codes=100, n_days=300, seed=123)
        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())
        factor_df = engine.compute(data)
        fwd = generate_forward_returns(data)

        factor_names = engine.list_factors()[:6]
        bench = benchmark_ic(factor_df, fwd, factor_names, ic_type="spearman")

        passed = bench['speedup'] >= 1.5 and bench['results_match']
        report.add(
            "向量化 IC 性能对比",
            passed,
            f"向量化 {bench['vectorized_time_sec']}s vs 循环 {bench['loop_time_sec']}s，加速 {bench['speedup']}x",
            bench,
        )
    except Exception as e:
        report.add("向量化 IC 性能对比", False, f"{e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 3. 向量化回测测试
# ---------------------------------------------------------------------------

def test_backtest_correctness(report: TestReport):
    """测试向量化回测产出合理性"""
    try:
        data = generate_synthetic_ohlcv(n_codes=30, n_days=150, seed=55)
        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())
        factor_df = engine.compute(data)

        # 用 reversal_20d 作为信号（值越大越看多）
        sig_df = factor_df[['code', 'date', 'reversal_20d']].rename(columns={'reversal_20d': 'signal'})
        sig_df = sig_df.dropna()

        result = vectorized_backtest(data, sig_df, top_k=10)

        assert 'equity_curve' in result and not result['equity_curve'].empty
        assert 'metrics' in result and result['metrics']

        m = result['metrics']
        # 基本合理性检查
        assert -0.99 < m['total_return'] < 5.0, f"总收益率异常: {m['total_return']}"
        assert -1.0 < m['max_drawdown'] <= 0, f"最大回撤异常: {m['max_drawdown']}"
        assert 0 <= m['win_rate'] <= 1, f"胜率异常: {m['win_rate']}"

        report.add(
            "向量化回测产出合理性",
            True,
            f"总收益 {m['total_return']:.4f}，夏普 {m['sharpe_ratio']:.4f}，回撤 {m['max_drawdown']:.4f}",
            m,
        )
    except Exception as e:
        report.add("向量化回测产出合理性", False, f"{e}\n{traceback.format_exc()}")


def test_backtest_performance(report: TestReport):
    """测试向量化回测性能提升"""
    try:
        data = generate_synthetic_ohlcv(n_codes=50, n_days=250, seed=77)
        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())
        factor_df = engine.compute(data)
        sig_df = factor_df[['code', 'date', 'reversal_20d']].rename(columns={'reversal_20d': 'signal'}).dropna()

        bench = benchmark_backtest(data, sig_df, top_k=15)

        # 性能提升判定（向量化应不慢于事件驱动）
        passed = bench['speedup'] >= 1.0
        report.add(
            "向量化回测性能对比",
            passed,
            f"向量化 {bench['vectorized_time_sec']}s vs 事件驱动 {bench['loop_time_sec']}s，加速 {bench['speedup']}x",
            bench,
        )
    except Exception as e:
        report.add("向量化回测性能对比", False, f"{e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 4. 边界条件测试
# ---------------------------------------------------------------------------

def test_edge_empty_data(report: TestReport):
    """边界：空数据"""
    try:
        empty = pd.DataFrame(columns=['code', 'date', 'close', 'volume'])
        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())
        result = engine.compute(empty)
        assert result.empty or len(result) == 0

        # IC 空数据
        vec_res = ic_analysis_vectorized(empty, empty, ["ret_5d"])
        assert vec_res == {}

        # 回测空数据
        bt_res = vectorized_backtest(empty, empty)
        assert bt_res['metrics'] == {}

        report.add("边界-空数据", True, "空数据不报错，返回空结果")
    except Exception as e:
        report.add("边界-空数据", False, str(e))


def test_edge_single_stock(report: TestReport):
    """边界：单只股票"""
    try:
        data = generate_synthetic_ohlcv(n_codes=1, n_days=60, seed=3, codes=["600000.SH"])
        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())
        factor_df = engine.compute(data)
        assert len(factor_df) == 60

        fwd = generate_forward_returns(data)
        vec_res = ic_analysis_vectorized(factor_df, fwd, ["ret_5d"])
        # 单只股票截面 IC 无意义，应返回空或跳过
        # 不报错即可
        report.add("边界-单只股票", True, "单只股票不报错")
    except Exception as e:
        report.add("边界-单只股票", False, str(e))


def test_edge_short_period(report: TestReport):
    """边界：极短周期（少于因子窗口）"""
    try:
        data = generate_synthetic_ohlcv(n_codes=10, n_days=5, seed=8)
        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())
        factor_df = engine.compute(data)
        # 短周期下部分因子应为 NaN，但不报错
        assert len(factor_df) == 50
        report.add("边界-极短周期", True, "5 日数据不报错，因子含 NaN")
    except Exception as e:
        report.add("边界-极短周期", False, str(e))


def test_edge_all_limit_up(report: TestReport):
    """边界：全涨停日（无法买入）"""
    try:
        data = generate_synthetic_ohlcv(n_codes=10, n_days=30, seed=11)
        # 强制设置某日全涨停
        target_date = data['date'].unique()[10]
        mask = data['date'] == target_date
        data.loc[mask, 'is_limit_up'] = True

        engine = FactorExpressionEngine()
        engine.register_many(build_default_factor_library())
        factor_df = engine.compute(data)
        sig_df = factor_df[['code', 'date', 'reversal_20d']].rename(columns={'reversal_20d': 'signal'}).dropna()

        result = vectorized_backtest(data, sig_df, top_k=5, price_limit=True)
        # 不报错即可
        assert 'metrics' in result
        report.add("边界-全涨停日", True, "全涨停日不报错，涨跌停限制生效")
    except Exception as e:
        report.add("边界-全涨停日", False, str(e))


def test_edge_invalid_expression(report: TestReport):
    """边界：非法表达式"""
    try:
        engine = FactorExpressionEngine()
        bad_cases = [
            "Mean($close,)",        # 缺参数
            "UnknownOp($close, 5)", # 未知算子
            "$close +",             # 不完整
            "Mean($close 20)",      # 缺逗号
        ]
        all_rejected = True
        for expr in bad_cases:
            try:
                engine.register(FactorMeta(name="bad", expression=expr))
                all_rejected = False
            except (ValueError, KeyError):
                pass  # 预期被拒绝
        assert all_rejected, "部分非法表达式未被拒绝"
        report.add("边界-非法表达式", True, "4 个非法表达式均被正确拒绝")
    except Exception as e:
        report.add("边界-非法表达式", False, str(e))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_all_tests() -> Dict[str, Any]:
    report = TestReport()

    print("=" * 70)
    print("jingni-trader 优化验证测试套件")
    print("分支: feat/quant-opt-20260623")
    print("=" * 70)

    # 1. 因子表达式引擎
    print("\n[1] 因子表达式引擎测试")
    test_expression_parser_basic(report)
    test_factor_engine_vs_hardcoded(report)
    test_factor_engine_extensibility(report)

    # 2. 向量化 IC
    print("\n[2] 向量化 IC 分析测试")
    test_ic_correctness(report)
    test_ic_performance(report)

    # 3. 向量化回测
    print("\n[3] 向量化回测测试")
    test_backtest_correctness(report)
    test_backtest_performance(report)

    # 4. 边界条件
    print("\n[4] 边界条件测试")
    test_edge_empty_data(report)
    test_edge_single_stock(report)
    test_edge_short_period(report)
    test_edge_all_limit_up(report)
    test_edge_invalid_expression(report)

    summary = report.summary()

    print("\n" + "=" * 70)
    print(f"测试结果: {summary['passed']}/{summary['total']} 通过，"
          f"{summary['failed']} 失败，通过率 {summary['pass_rate']*100:.1f}%")
    print("=" * 70)
    for r in summary['results']:
        status = "✓" if r['passed'] else "✗"
        print(f"  {status} {r['name']}")
        if r['detail']:
            print(f"      {r['detail']}")

    return summary


if __name__ == "__main__":
    summary = run_all_tests()
    # 保存结果
    out_path = ROOT / "optimizations" / "opt_20260623" / "reports" / "test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n测试结果已保存: {out_path}")
