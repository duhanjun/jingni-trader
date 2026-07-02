"""
优化验证测试套件

测试内容：
1. 正确性测试：向量化实现 vs 循环实现的结果一致性
2. 性能对比测试：向量化 vs 循环的执行时间
3. 边界条件测试：空数据、单股票、单日期等
4. 增强指标测试：验证新增指标的合理性

运行方式：
    python -m optimizations.test_optimizations
    或
    python optimizations/test_optimizations.py
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 确保能导入 optimizations 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizations.vectorized_backtest import (
    VectorizedBacktester, LoopBacktester, benchmark_backtest
)
from optimizations.enhanced_metrics import calc_enhanced_metrics, calc_basic_metrics
from optimizations.vectorized_ic import vectorized_ic, loop_ic, benchmark_ic


# ------------------------------------------------------------------
# 测试数据生成
# ------------------------------------------------------------------
def generate_market_data(
    n_stocks: int = 50,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """生成模拟 A 股日线数据"""
    np.random.seed(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    rows = []
    for code in codes:
        price = np.random.uniform(10, 50)
        for dt in dates:
            ret = np.random.normal(0.0005, 0.02)
            price = max(price * (1 + ret), 1.0)
            change_pct = ret * 100
            rows.append({
                'date': dt,
                'code': code,
                'open': price * (1 + np.random.normal(0, 0.003)),
                'high': price * (1 + abs(np.random.normal(0, 0.008))),
                'low': price * (1 - abs(np.random.normal(0, 0.008))),
                'close': price,
                'volume': int(np.random.lognormal(15, 0.5)),
                'change_pct': change_pct,
                'is_st': False,
                'is_limit_up': change_pct >= 9.9,
                'is_limit_down': change_pct <= -9.9,
            })

    return pd.DataFrame(rows)


def generate_signals(
    data: pd.DataFrame,
    rebalance_days: int = 5,
    top_pct: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """基于动量生成买卖信号"""
    np.random.seed(seed)
    data = data.sort_values(['code', 'date']).copy()
    # 计算 20 日动量
    data['mom_20d'] = data.groupby('code')['close'].pct_change(20)
    data = data.dropna(subset=['mom_20d'])

    signals = []
    dates = sorted(data['date'].unique())
    # 每 rebalance_days 天调仓
    for i, dt in enumerate(dates):
        if i % rebalance_days != 0:
            continue
        day_data = data[data['date'] == dt]
        if day_data.empty:
            continue
        # 选前 top_pct 动量的股票买入
        threshold = day_data['mom_20d'].quantile(1 - top_pct)
        buy_codes = day_data[day_data['mom_20d'] >= threshold]['code'].tolist()
        # 其余股票中，动量最低的 10% 卖出
        sell_threshold = day_data['mom_20d'].quantile(0.1)
        sell_codes = day_data[day_data['mom_20d'] <= sell_threshold]['code'].tolist()

        for code in buy_codes:
            signals.append({'date': dt, 'code': code, 'signal': 1.0})
        for code in sell_codes:
            signals.append({'date': dt, 'code': code, 'signal': -1.0})

    return pd.DataFrame(signals)


def generate_factor_data(
    n_stocks: int = 100,
    n_dates: int = 200,
    n_factors: int = 5,
    seed: int = 42,
) -> tuple:
    """生成因子数据和前向收益数据"""
    np.random.seed(seed)
    dates = pd.bdate_range(start="2022-01-01", periods=n_dates)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    factor_rows = []
    return_rows = []
    for code in codes:
        for dt in dates:
            row = {'date': dt, 'code': code}
            for k in range(n_factors):
                row[f'factor_{k}'] = np.random.normal(0, 1)
            factor_rows.append(row)

            # 前向收益与 factor_0 有一定相关性（让 IC 非零）
            ret_1d = 0.01 * row['factor_0'] + np.random.normal(0, 0.02)
            ret_5d = 0.02 * row['factor_0'] + np.random.normal(0, 0.04)
            ret_20d = 0.03 * row['factor_0'] + np.random.normal(0, 0.08)
            return_rows.append({
                'date': dt, 'code': code,
                'ret_forward_1d': ret_1d,
                'ret_forward_5d': ret_5d,
                'ret_forward_20d': ret_20d,
            })

    factor_df = pd.DataFrame(factor_rows)
    forward_returns = pd.DataFrame(return_rows)
    return factor_df, forward_returns


# ------------------------------------------------------------------
# 测试用例
# ------------------------------------------------------------------
class TestResults:
    """收集测试结果"""
    def __init__(self):
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {},
        }

    def add(self, name: str, passed: bool, details: dict):
        self.results["tests"].append({
            "name": name,
            "passed": passed,
            "details": details,
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        for k, v in details.items():
            if isinstance(v, float):
                print(f"         {k}: {v:.6f}")
            else:
                print(f"         {k}: {v}")

    def summary_str(self):
        n_pass = sum(1 for t in self.results["tests"] if t["passed"])
        n_fail = sum(1 for t in self.results["tests"] if not t["passed"])
        return f"\n==== 测试汇总: {n_pass} 通过 / {n_fail} 失败 / 共 {len(self.results['tests'])} 项 ====\n"


def test_backtest_correctness(tr: TestResults, data, signals):
    """测试1: 向量化回测 vs 循环回测的正确性"""
    print("\n=== 测试1: 回测正确性验证 ===")
    vb = VectorizedBacktester()
    lb = LoopBacktester()

    vb_result = vb.run(data, signals)
    lb_result = lb.run(data, signals)

    vb_eq = vb_result['equity_curve']['equity'].iloc[-1] if not vb_result['equity_curve'].empty else 0
    lb_eq = lb_result['equity_curve']['equity'].iloc[-1] if not lb_result['equity_curve'].empty else 0

    diff = abs(vb_eq - lb_eq)
    # 由于买卖顺序在多股票场景下可能略有差异（资金分配顺序），
    # 这里允许小偏差，但要求相对误差 < 1%
    rel_diff = diff / max(abs(lb_eq), 1e-6)
    passed = rel_diff < 0.01

    tr.add("回测终值一致性", passed, {
        "vectorized_final_equity": float(vb_eq),
        "loop_final_equity": float(lb_eq),
        "abs_diff": float(diff),
        "rel_diff": float(rel_diff),
        "vectorized_n_trades": len(vb_result['trades']),
        "loop_n_trades": len(lb_result['trades']),
    })


def test_backtest_performance(tr: TestResults, data, signals):
    """测试2: 回测性能对比"""
    print("\n=== 测试2: 回测性能对比 ===")
    result = benchmark_backtest(data, signals, runs=3)

    passed = result['speedup'] >= 1.0  # 至少不慢于循环版
    tr.add("回测性能提升", passed, {
        "vectorized_median_time_s": result['vectorized']['median_time'],
        "loop_median_time_s": result['loop']['median_time'],
        "speedup_x": result['speedup'],
        "vectorized_n_trades": result['vectorized']['n_trades'],
        "loop_n_trades": result['loop']['n_trades'],
    })


def test_backtest_edge_cases(tr: TestResults):
    """测试3: 边界条件测试"""
    print("\n=== 测试3: 边界条件测试 ===")
    vb = VectorizedBacktester()
    lb = LoopBacktester()

    # 3.1 空数据
    empty = pd.DataFrame(columns=['date', 'code', 'close', 'is_limit_up', 'is_limit_down'])
    empty_sig = pd.DataFrame(columns=['date', 'code', 'signal'])
    vb_r = vb.run(empty, empty_sig)
    lb_r = lb.run(empty, empty_sig)
    tr.add("空数据处理", vb_r['equity_curve'].empty and lb_r['equity_curve'].empty, {
        "vectorized_empty": vb_r['equity_curve'].empty,
        "loop_empty": lb_r['equity_curve'].empty,
    })

    # 3.2 单股票单日
    single_data = pd.DataFrame([{
        'date': pd.Timestamp('2024-01-01'),
        'code': '600000.SH',
        'close': 10.0,
        'is_limit_up': False,
        'is_limit_down': False,
    }])
    single_sig = pd.DataFrame([{
        'date': pd.Timestamp('2024-01-01'),
        'code': '600000.SH',
        'signal': 1.0,
    }])
    vb_r = vb.run(single_data, single_sig)
    lb_r = lb.run(single_data, single_sig)
    tr.add("单股票单日", not vb_r['equity_curve'].empty and not lb_r['equity_curve'].empty, {
        "vectorized_records": len(vb_r['equity_curve']),
        "loop_records": len(lb_r['equity_curve']),
    })

    # 3.3 涨跌停限制
    limit_data = pd.DataFrame([{
        'date': pd.Timestamp('2024-01-01'),
        'code': '600000.SH',
        'close': 11.0,
        'is_limit_up': True,   # 涨停，无法买入
        'is_limit_down': False,
    }])
    limit_sig = pd.DataFrame([{
        'date': pd.Timestamp('2024-01-01'),
        'code': '600000.SH',
        'signal': 1.0,
    }])
    vb_r = vb.run(limit_data, limit_sig)
    # 涨停时应无交易
    tr.add("涨停限制买入", len(vb_r['trades']) == 0, {
        "n_trades_when_limit_up": len(vb_r['trades']),
    })


def test_enhanced_metrics(tr: TestResults, data, signals):
    """测试4: 增强绩效指标"""
    print("\n=== 测试4: 增强绩效指标验证 ===")
    vb = VectorizedBacktester()
    result = vb.run(data, signals)

    if result['equity_curve'].empty:
        tr.add("增强指标计算", False, {"error": "equity_curve 为空"})
        return

    enhanced = calc_enhanced_metrics(
        result['equity_curve'],
        result['trades'],
    )
    basic = calc_basic_metrics(result['equity_curve'])

    # 验证：增强指标应包含所有基础指标 + 新增指标
    basic_keys = set(basic.keys())
    enhanced_keys = set(enhanced.keys())
    new_keys = enhanced_keys - basic_keys

    # 验证基础指标一致性
    consistent = True
    for k in ['total_return', 'sharpe_ratio', 'max_drawdown']:
        if k in basic and k in enhanced:
            if abs(basic[k] - enhanced[k]) > 1e-6:
                consistent = False
                break

    passed = len(new_keys) >= 10 and consistent
    tr.add("增强指标完整性", passed, {
        "basic_metrics_count": len(basic),
        "enhanced_metrics_count": len(enhanced),
        "new_metrics_count": len(new_keys),
        "new_metrics": sorted(list(new_keys)),
        "basic_consistent": consistent,
    })

    # 验证 Sortino >= Sharpe（下行风险 <= 总波动，故 Sortino 通常 >= Sharpe）
    if 'sortino_ratio' in enhanced and 'sharpe_ratio' in enhanced:
        # 注意：当 returns 全为正时，downside_deviation 可能为 0，导致 sortino=inf
        sortino_ok = (
            enhanced['sortino_ratio'] >= enhanced['sharpe_ratio']
            or enhanced['downside_deviation'] == 0
        )
        tr.add("Sortino >= Sharpe 关系", sortino_ok, {
            "sharpe": enhanced['sharpe_ratio'],
            "sortino": enhanced['sortino_ratio'],
            "downside_dev": enhanced['downside_deviation'],
            "volatility": enhanced['volatility'],
        })

    # 验证 Beta/Alpha（用等权基准收益率作为 benchmark）
    # 构造一个简单的基准收益序列（全市场等权日收益）
    eq_series = result['equity_curve'].set_index('date')['equity'].sort_index()
    strat_returns = eq_series.pct_change().dropna()
    # 构造一个与策略收益有一定相关性的基准（模拟沪深300）
    np.random.seed(123)
    benchmark_returns = 0.6 * strat_returns + np.random.normal(0, 0.01, len(strat_returns))
    benchmark_returns = pd.Series(
        benchmark_returns, index=strat_returns.index
    )

    enhanced_with_bench = calc_enhanced_metrics(
        result['equity_curve'],
        result['trades'],
        benchmark=benchmark_returns,
    )
    tr.add("Beta/Alpha 计算", enhanced_with_bench['beta'] != 0, {
        "beta": enhanced_with_bench['beta'],
        "alpha": enhanced_with_bench['alpha'],
        "information_ratio": enhanced_with_bench['information_ratio'],
        "tracking_error": enhanced_with_bench['tracking_error'],
    })


def test_ic_correctness(tr: TestResults):
    """测试5: IC 分析正确性"""
    print("\n=== 测试5: IC 分析正确性验证 ===")
    factor_df, forward_returns = generate_factor_data(
        n_stocks=80, n_dates=150, n_factors=4
    )
    factor_names = [f'factor_{i}' for i in range(4)]

    vec_result = vectorized_ic(factor_df, forward_returns, factor_names)
    loop_result = loop_ic(factor_df, forward_returns, factor_names)

    # 比较 IC 均值
    max_diff = 0.0
    mean_diff = 0.0
    n_compared = 0
    for period in vec_result:
        for v_item, l_item in zip(
            sorted(vec_result[period], key=lambda x: x['factor']),
            sorted(loop_result[period], key=lambda x: x['factor']),
        ):
            diff = abs(v_item['ic_mean'] - l_item['ic_mean'])
            max_diff = max(max_diff, diff)
            mean_diff += diff
            n_compared += 1

    mean_diff = mean_diff / n_compared if n_compared > 0 else 0
    # IC 均值差异应 < 1e-4（rank 计算的微小数值差异）
    passed = max_diff < 1e-4
    tr.add("IC 均值一致性", passed, {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "n_compared": n_compared,
    })

    # 验证 factor_0 的 IC 应显著非零（因为生成数据时 factor_0 与收益相关）
    f0_ic_5d = None
    for item in vec_result.get('ret_forward_5d', []):
        if item['factor'] == 'factor_0':
            f0_ic_5d = item
            break
    if f0_ic_5d:
        passed = abs(f0_ic_5d['ic_mean']) > 0.05 and f0_ic_5d['ic_t_stat'] > 2
        tr.add("factor_0 IC 显著性", passed, {
            "ic_mean": f0_ic_5d['ic_mean'],
            "ic_t_stat": f0_ic_5d['ic_t_stat'],
            "ic_ir": f0_ic_5d['ic_ir'],
        })


def test_ic_performance(tr: TestResults):
    """测试6: IC 分析性能对比"""
    print("\n=== 测试6: IC 分析性能对比 ===")
    # 用较大的数据集测试性能
    factor_df, forward_returns = generate_factor_data(
        n_stocks=300, n_dates=250, n_factors=5
    )
    factor_names = [f'factor_{i}' for i in range(5)]

    result = benchmark_ic(factor_df, forward_returns, factor_names, runs=3)

    passed = result['speedup'] >= 1.0
    tr.add("IC 性能提升", passed, {
        "vectorized_median_time_s": result['vectorized']['median_time'],
        "loop_median_time_s": result['loop']['median_time'],
        "speedup_x": result['speedup'],
        "max_ic_diff": result['max_ic_diff'],
        "n_factors": result['n_factors'],
        "n_stocks": 300,
        "n_dates": 250,
    })


def test_ic_edge_cases(tr: TestResults):
    """测试7: IC 边界条件"""
    print("\n=== 测试7: IC 边界条件测试 ===")

    # 7.1 空数据
    empty = pd.DataFrame(columns=['date', 'code', 'factor_0'])
    empty_ret = pd.DataFrame(columns=['date', 'code', 'ret_forward_1d'])
    r1 = vectorized_ic(empty, empty_ret)
    r2 = loop_ic(empty, empty_ret)
    tr.add("IC 空数据", r1 == {} and r2 == {}, {
        "vectorized_empty": r1 == {},
        "loop_empty": r2 == {},
    })

    # 7.2 单日期（样本不足）
    single = pd.DataFrame([
        {'date': pd.Timestamp('2024-01-01'), 'code': 'A', 'factor_0': 1.0},
        {'date': pd.Timestamp('2024-01-01'), 'code': 'B', 'factor_0': 2.0},
    ])
    single_ret = pd.DataFrame([
        {'date': pd.Timestamp('2024-01-01'), 'code': 'A', 'ret_forward_1d': 0.01},
        {'date': pd.Timestamp('2024-01-01'), 'code': 'B', 'ret_forward_1d': -0.01},
    ])
    # min_samples=10，应返回空
    r1 = vectorized_ic(single, single_ret, ['factor_0'], min_samples=10)
    tr.add("IC 样本不足保护", len(r1.get('ret_forward_1d', [])) == 0, {
        "result_keys": list(r1.keys()),
    })


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def run_all_tests():
    """运行全部测试"""
    print("=" * 70)
    print("jingni-trader 优化验证测试套件")
    print(f"执行时间: {datetime.now().isoformat()}")
    print(f"分支: feat/quant-opt-20260620")
    print("=" * 70)

    tr = TestResults()

    # 生成测试数据
    print("\n生成测试数据...")
    t0 = time.perf_counter()
    data = generate_market_data(n_stocks=50, start_date="2022-01-01", end_date="2024-12-31")
    signals = generate_signals(data, rebalance_days=5, top_pct=0.2)
    print(f"  行情数据: {len(data)} 行, {data['code'].nunique()} 只股票")
    print(f"  信号数据: {len(signals)} 条")
    print(f"  数据生成耗时: {time.perf_counter() - t0:.3f}s")

    # 运行测试
    test_backtest_correctness(tr, data, signals)
    test_backtest_performance(tr, data, signals)
    test_backtest_edge_cases(tr)
    test_enhanced_metrics(tr, data, signals)
    test_ic_correctness(tr)
    test_ic_performance(tr)
    test_ic_edge_cases(tr)

    # 汇总
    print(tr.summary_str())

    n_pass = sum(1 for t in tr.results["tests"] if t["passed"])
    n_fail = sum(1 for t in tr.results["tests"] if not t["passed"])
    tr.results["summary"] = {
        "total": len(tr.results["tests"]),
        "passed": n_pass,
        "failed": n_fail,
        "pass_rate": n_pass / len(tr.results["tests"]) if tr.results["tests"] else 0,
    }

    return tr.results


if __name__ == "__main__":
    results = run_all_tests()

    # 保存测试结果
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_results.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n测试结果已保存至: {output_path}")

    sys.exit(0 if results["summary"]["failed"] == 0 else 1)
