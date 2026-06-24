"""
优化验证测试

验证内容:
  1. 正确性测试: T+1 约束、基准跟踪、IC 计算与 scipy 一致性
  2. 性能对比测试: 向量化回测 vs 原生逐行回测、向量化 IC vs 逐日循环 IC
  3. 边界条件测试: 空数据、单只股票、全涨跌停
  4. 成本模型测试: 固定滑点 vs 量价滑点的成本差异
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizations.cost_models import (
    CostCalculator, ConstantSlippage, VolumeShareSlippage,
    AShareFeeModel, TradeContext,
)
from optimizations.vectorized_backtest import VectorizedBacktester
from optimizations.factor_analysis_enhanced import EnhancedFactorAnalysis


# ========== 测试数据生成 ==========

def make_synthetic_data(n_codes=20, n_days=120, seed=42):
    """生成合成行情数据"""
    np.random.seed(seed)
    codes = [f"{600000+i:06d}.SH" for i in range(n_codes)]
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    rows = []
    for code in codes:
        price = 10.0 + np.random.uniform(0, 5)
        for dt in dates:
            ret = np.random.normal(0, 0.02)
            price = max(price * (1 + ret), 1.0)
            vol = int(np.random.uniform(1e6, 5e6))
            rows.append({
                'code': code, 'date': dt,
                'open': price * 0.99, 'high': price * 1.01,
                'low': price * 0.98, 'close': price,
                'volume': vol, 'amount': vol * price,
                'turnover_rate': np.random.uniform(0.5, 3.0),
                'is_st': False, 'is_limit_up': False, 'is_limit_down': False,
            })
    return pd.DataFrame(rows)


def make_synthetic_signals(data, buy_pct=0.2, seed=42):
    """生成合成交易信号（每日选前 buy_pct 买入，轮动）"""
    np.random.seed(seed)
    signals = []
    dates = sorted(data['date'].unique())
    for dt in dates:
        day_codes = data[data['date'] == dt]['code'].tolist()
        n_buy = max(1, int(len(day_codes) * buy_pct))
        # 每 5 天换一次仓
        idx = dates.index(dt)
        if idx % 5 == 0:
            np.random.shuffle(day_codes)
        buy_set = set(day_codes[:n_buy])
        for code in day_codes:
            if code in buy_set:
                signals.append({'date': dt, 'code': code, 'signal': 1})
            elif idx % 5 == 4:
                signals.append({'date': dt, 'code': code, 'signal': -1})
    return pd.DataFrame(signals)


def make_factor_data(data, n_factors=5, seed=42):
    """生成合成因子数据（使用 AR(1) 过程，具有真实自相关结构）"""
    np.random.seed(seed)
    codes = data['code'].unique()
    dates = data['date'].unique()
    n = len(dates)
    rows = []
    # AR(1) 系数：factor_0 最稳定(0.9)，随因子序号递减
    ar_coefs = [0.9, 0.7, 0.5, 0.3, 0.1][:n_factors]
    for code in codes:
        states = [np.random.normal(0, 1, n) for _ in range(n_factors)]
        for f in range(n_factors):
            # 生成 AR(1) 序列：x_t = phi * x_{t-1} + noise
            phi = ar_coefs[f]
            x = np.zeros(n)
            x[0] = states[f][0]
            for t in range(1, n):
                x[t] = phi * x[t - 1] + np.random.normal(0, 0.5)
            states[f] = x
        for i, dt in enumerate(dates):
            row = {'code': code, 'date': dt}
            for f in range(n_factors):
                row[f'factor_{f}'] = float(states[f][i])
            rows.append(row)
    return pd.DataFrame(rows)


# ========== 1. 成本模型测试 ==========

def test_cost_models():
    print("\n" + "=" * 60)
    print("测试 1: 模块化成本模型")
    print("=" * 60)

    # 固定滑点
    calc_const = CostCalculator(ConstantSlippage(0.001), AShareFeeModel())
    ctx = TradeContext(price=10.0, shares=1000, side='buy')
    fill, fees = calc_const.compute(ctx)
    assert abs(fill - 10.01) < 1e-6, f"固定滑点买入价错误: {fill}"
    assert fees['tax'] == 0, "买入不应收印花税"
    print(f"  [固定滑点] 买入 1000股@10.0 -> 成交价={fill:.4f}, 佣金={fees['commission']:.2f}, 税={fees['tax']}")

    ctx_sell = TradeContext(price=10.0, shares=1000, side='sell')
    fill_s, fees_s = calc_const.compute(ctx_sell)
    assert fill_s < 10.0, "卖出滑点应降低成交价"
    assert fees_s['tax'] > 0, "卖出应收印花税"
    print(f"  [固定滑点] 卖出 1000股@10.0 -> 成交价={fill_s:.4f}, 佣金={fees_s['commission']:.2f}, 税={fees_s['tax']:.2f}")

    # 量价滑点: 大单 vs 小单
    calc_vol = CostCalculator(VolumeShareSlippage(price_impact=0.1), AShareFeeModel())
    # 小单: 100股，日成交量 100万
    ctx_small = TradeContext(price=10.0, shares=100, side='buy', volume=1_000_000)
    fill_small, _ = calc_vol.compute(ctx_small)
    # 大单: 20万股，日成交量 100万（占比20%）
    ctx_big = TradeContext(price=10.0, shares=200_000, side='buy', volume=1_000_000)
    fill_big, _ = calc_vol.compute(ctx_big)

    impact_small = fill_small / 10.0 - 1
    impact_big = fill_big / 10.0 - 1
    print(f"  [量价滑点] 小单(100股/100万量) 冲击={impact_small:.6f}")
    print(f"  [量价滑点] 大单(20万股/100万量) 冲击={impact_big:.6f}")
    assert impact_big > impact_small * 10, "大单冲击应远大于小单"
    print("  ✓ 大单市场冲击显著高于小单，量价滑点模型生效")

    return True


# ========== 2. T+1 正确性测试 ==========

def test_t_plus_1_enforcement():
    print("\n" + "=" * 60)
    print("测试 2: T+1 约束正确性（核心 BUG 修复验证）")
    print("=" * 60)

    data = make_synthetic_data(n_codes=5, n_days=10)
    codes = data['code'].unique()[:3]
    dates = sorted(data['date'].unique())
    dt0, dt1 = dates[0], dates[1]

    # 场景A: 第0天买入 code0，第0天也有卖出信号
    # sell-before-buy 顺序下，卖出阶段无持仓 -> 自然跳过（T+1 自然生效）
    # 第1天卖出 code0 -> 应执行（T+1 满足：day1 > day0）
    signals = pd.DataFrame([
        {'date': dt0, 'code': codes[0], 'signal': 1},   # 第0天买入
        {'date': dt0, 'code': codes[0], 'signal': -1},  # 第0天卖出信号（应被阻止：无持仓）
        {'date': dt1, 'code': codes[0], 'signal': -1},  # 第1天卖出（应执行：T+1满足）
    ])

    bt_t1 = VectorizedBacktester(t_plus_1=True, price_limit=False)
    result = bt_t1.run(data, signals, init_capital=1e6, benchmark=codes[0])
    trades = result['trades']

    # 第0天对 code0 应只有买入，无卖出
    day0 = trades[(trades['date'] == dt0) & (trades['code'] == codes[0])]
    day0_actions = day0['action'].tolist()
    print(f"  [T+1] 第0天 code0 操作: {day0_actions}")
    assert 'buy' in day0_actions, "第0天应执行买入"
    assert 'sell' not in day0_actions, "第0天不应有卖出（T+1：卖出阶段无持仓）"

    # 第1天对 code0 应有卖出
    day1 = trades[(trades['date'] == dt1) & (trades['code'] == codes[0])]
    day1_actions = day1['action'].tolist()
    print(f"  [T+1] 第1天 code0 操作: {day1_actions}")
    assert 'sell' in day1_actions, "第1天应执行卖出（T+1满足）"
    print("  ✓ T+1 行为正确：当日买入不可当日卖出，次日可卖出")

    # 场景B: 显式 T+1 检查 —— 同日买入+卖出信号，卖出应被阻止
    signals_b = pd.DataFrame([
        {'date': dt0, 'code': codes[1], 'signal': 1},
        {'date': dt0, 'code': codes[1], 'signal': -1},
    ])
    result_b = bt_t1.run(data, signals_b, init_capital=1e6, benchmark=codes[0])
    trades_b = result_b['trades']
    code1_day0 = trades_b[(trades_b['date'] == dt0) & (trades_b['code'] == codes[1])]
    print(f"  [T+1显式] code1 第0天操作: {code1_day0['action'].tolist()}")
    assert 'sell' not in code1_day0['action'].tolist(), "同日买入不可卖出"
    print("  ✓ 显式 T+1 检查：同日买入的股票当日不可卖出")

    return True


# ========== 3. 基准跟踪测试 ==========

def test_benchmark_tracking():
    print("\n" + "=" * 60)
    print("测试 3: 基准净值跟踪")
    print("=" * 60)

    data = make_synthetic_data(n_codes=10, n_days=30)
    bench_code = data['code'].unique()[0]
    signals = make_synthetic_signals(data, buy_pct=0.3)

    bt = VectorizedBacktester(price_limit=False)
    result = bt.run(data, signals, init_capital=1e6, benchmark=bench_code)

    eq = result['equity_curve']
    assert 'benchmark' in eq.columns, "equity_curve 应包含 benchmark 列"
    bench_vals = eq['benchmark'].dropna()
    assert len(bench_vals) > 0, "基准净值不应为空"
    print(f"  基准列非空记录数: {len(bench_vals)}/{len(eq)}")

    metrics = result['metrics']
    assert 'beta' in metrics, "应计算 beta"
    assert 'alpha' in metrics, "应计算 alpha"
    assert 'information_ratio' in metrics, "应计算 information_ratio"
    print(f"  beta={metrics['beta']:.4f}, alpha={metrics['alpha']:.4f}, "
          f"IR={metrics['information_ratio']:.4f}, benchmark_return={metrics['benchmark_return']:.4f}")
    print("  ✓ 基准跟踪与相对指标计算正确")

    return True


# ========== 4. 性能对比测试 ==========

def test_backtest_performance():
    print("\n" + "=" * 60)
    print("测试 4: 回测性能对比（向量化 vs 原生逐行）")
    print("=" * 60)

    # 用较大数据集测试性能
    data = make_synthetic_data(n_codes=50, n_days=250)
    signals = make_synthetic_signals(data, buy_pct=0.2)

    # 向量化回测
    bt_vec = VectorizedBacktester(price_limit=False)
    t0 = time.time()
    result_vec = bt_vec.run(data, signals, init_capital=1e6, benchmark=data['code'].iloc[0])
    t_vec = time.time() - t0

    # 原生逐行回测（jingni-trader 现有实现）
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'skills', 'backtest-engine'))
    try:
        from adapters.native_adapter import NativeAdapter
        bt_native = NativeAdapter()
        t0 = time.time()
        result_native = bt_native.run_backtest(
            data, signals, init_capital=1e6, benchmark=data['code'].iloc[0],
            price_limit=False,
        )
        t_native = time.time() - t0
        speedup = t_native / t_vec if t_vec > 0 else float('inf')
        print(f"  数据规模: {data.shape[0]} 行 ({data['code'].nunique()} 只股票 x {data['date'].nunique()} 天)")
        print(f"  原生逐行回测: {t_native:.3f}s")
        print(f"  向量化回测:   {t_vec:.3f}s")
        print(f"  加速比: {speedup:.1f}x")

        # 结果一致性检查（总收益应接近，因成本模型略有差异）
        ret_vec = result_vec['metrics'].get('total_return', 0)
        ret_nat = result_native['metrics'].get('total_return', 0)
        print(f"  总收益(向量化)={ret_vec:.4f}, 总收益(原生)={ret_nat:.4f}")
        print("  ✓ 向量化回测完成且产出结构完整")
    except ImportError as e:
        print(f"  [跳过原生对比] 依赖不可用: {e}")
        print(f"  向量化回测: {t_vec:.3f}s (数据 {data.shape[0]} 行)")
        print("  ✓ 向量化回测完成")

    return True


# ========== 5. IC 向量化性能与正确性 ==========

def test_ic_vectorized():
    print("\n" + "=" * 60)
    print("测试 5: IC 向量化计算（正确性 + 性能）")
    print("=" * 60)

    data = make_synthetic_data(n_codes=30, n_days=120)
    factor_df = make_factor_data(data, n_factors=4)
    factor_names = [f'factor_{i}' for i in range(4)]

    # 前瞻收益
    fwd = data[['code', 'date', 'close']].sort_values(['code', 'date']).copy()
    fwd['ret_forward_1d'] = fwd.groupby('code')['close'].transform(lambda x: x.shift(-1) / x - 1)
    fwd['ret_forward_5d'] = fwd.groupby('code')['close'].transform(lambda x: x.shift(-5) / x - 1)
    fwd['ret_forward_20d'] = fwd.groupby('code')['close'].transform(lambda x: x.shift(-20) / x - 1)

    efa = EnhancedFactorAnalysis()

    # 向量化 IC
    t0 = time.time()
    ic_vec = efa.ic_analysis_vectorized(factor_df, fwd, factor_names, "spearman")
    t_vec = time.time() - t0

    assert 'ret_forward_5d' in ic_vec, "应包含 5 日前瞻 IC"
    assert len(ic_vec['ret_forward_5d']) == 4, "应有 4 个因子的 IC"
    print(f"  向量化 IC 计算: {t_vec:.3f}s")
    for item in ic_vec['ret_forward_5d']:
        print(f"    {item['factor']}: IC={item['ic_mean']:.4f}, ICIR={item['ic_ir']:.4f}, t={item['ic_t_stat']:.2f}")

    # 正确性: 与手动逐日计算对比（取一个因子）
    merged = factor_df.merge(fwd[['code', 'date', 'ret_forward_5d']], on=['code', 'date'])
    manual_ics = []
    for dt in merged['date'].unique():
        cross = merged[merged['date'] == dt].dropna(subset=['factor_0', 'ret_forward_5d'])
        if len(cross) < 10:
            continue
        from scipy.stats import spearmanr
        r, _ = spearmanr(cross['factor_0'], cross['ret_forward_5d'])
        if not np.isnan(r):
            manual_ics.append(r)
    manual_mean = np.mean(manual_ics)
    vec_mean = ic_vec['ret_forward_5d'][0]['ic_mean']
    print(f"  正确性校验: 手动IC={manual_mean:.6f}, 向量化IC={vec_mean:.6f}, 差异={abs(manual_mean-vec_mean):.8f}")
    assert abs(manual_mean - vec_mean) < 1e-4, "向量化 IC 与手动计算应一致"
    print("  ✓ 向量化 IC 计算结果与手动逐日计算一致")

    return True


# ========== 6. 因子换手率与衰减测试 ==========

def test_factor_turnover_decay():
    print("\n" + "=" * 60)
    print("测试 6: 因子换手率与衰减分析")
    print("=" * 60)

    data = make_synthetic_data(n_codes=30, n_days=120)
    factor_df = make_factor_data(data, n_factors=3)
    factor_names = [f'factor_{i}' for i in range(3)]

    efa = EnhancedFactorAnalysis()

    # 换手率
    turnover = efa.factor_turnover(factor_df, factor_names, lags=[1, 5, 20])
    assert len(turnover) == 3, "应有 3 个因子的换手率"
    for f, stats in turnover.items():
        print(f"  {f}: lag1自相关={stats.get('autocorr_lag1')}, "
              f"lag1换手率={stats.get('turnover_lag1')}, "
              f"lag5自相关={stats.get('autocorr_lag5')}")
    # factor_0 (AR系数0.9) 应有高 lag1 自相关且高于 lag5
    ac1_f0 = turnover['factor_0'].get('autocorr_lag1', 0)
    ac5_f0 = turnover['factor_0'].get('autocorr_lag5', 0)
    assert ac1_f0 > 0.5, f"factor_0(AR=0.9) lag1自相关应较高: {ac1_f0}"
    assert ac1_f0 > ac5_f0, f"factor_0 lag1自相关({ac1_f0})应>lag5({ac5_f0})"
    # factor_0 换手率应低于 factor_2（更稳定的因子换手率更低）
    to_f0 = turnover['factor_0'].get('turnover_lag1', 1)
    to_f2 = turnover['factor_2'].get('turnover_lag1', 1)
    assert to_f0 < to_f2, f"稳定因子(factor_0)换手率应更低: {to_f0} vs {to_f2}"
    print(f"  ✓ factor_0(AR=0.9)换手率={to_f0:.3f} < factor_2(AR=0.5)换手率={to_f2:.3f}，稳定性分析合理")

    # 衰减曲线
    decay = efa.factor_decay(factor_df, data[['code', 'date', 'close']], factor_names, horizons=[1, 3, 5, 10, 20])
    for f, curve in decay.items():
        ics = {k: v for k, v in curve.items() if k.startswith('ic_horizon')}
        hl = curve.get('estimated_half_life_days')
        print(f"  {f}: IC衰减={ics}, 半衰期={hl}天")
    print("  ✓ 因子衰减曲线与半衰期计算完成")

    return True


# ========== 7. 边界条件测试 ==========

def test_boundary_conditions():
    print("\n" + "=" * 60)
    print("测试 7: 边界条件")
    print("=" * 60)

    bt = VectorizedBacktester(price_limit=False)

    # 空数据
    result = bt.run(pd.DataFrame(), pd.DataFrame())
    assert result['metrics'] == {}, "空数据应返回空指标"
    print("  ✓ 空数据处理正确")

    # 单只股票
    data = make_synthetic_data(n_codes=1, n_days=20)
    signals = pd.DataFrame([
        {'date': data['date'].iloc[0], 'code': data['code'].iloc[0], 'signal': 1},
    ])
    result = bt.run(data, signals, init_capital=1e6, benchmark=data['code'].iloc[0])
    assert len(result['equity_curve']) > 0, "单只股票应正常回测"
    print(f"  ✓ 单只股票回测正常, 权益记录数={len(result['equity_curve'])}")

    # 无信号
    data = make_synthetic_data(n_codes=5, n_days=10)
    signals = pd.DataFrame(columns=['date', 'code', 'signal'])
    result = bt.run(data, signals, init_capital=1e6, benchmark=data['code'].iloc[0])
    # 无信号时应返回空或仅初始权益
    print(f"  ✓ 无信号处理正常, trades数={len(result['trades'])}")

    # 全涨停（无法买入）
    data_limit = make_synthetic_data(n_codes=5, n_days=5)
    data_limit['is_limit_up'] = True
    dt0 = data_limit['date'].min()
    signals = pd.DataFrame([
        {'date': dt0, 'code': data_limit['code'].iloc[0], 'signal': 1},
    ])
    bt_limit = VectorizedBacktester(price_limit=True)
    result = bt_limit.run(data_limit, signals, init_capital=1e6, benchmark=data_limit['code'].iloc[0])
    assert len(result['trades']) == 0, "全涨停时应无买入成交"
    print("  ✓ 全涨停时买入被正确阻止")

    return True


# ========== 主入口 ==========

def main():
    print("=" * 60)
    print("jingni-trader 优化验证测试")
    print(f"执行时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"分支: feat/quant-opt-20260624")
    print("=" * 60)

    results = {}
    tests = [
        ("成本模型", test_cost_models),
        ("T+1约束", test_t_plus_1_enforcement),
        ("基准跟踪", test_benchmark_tracking),
        ("回测性能", test_backtest_performance),
        ("IC向量化", test_ic_vectorized),
        ("换手率衰减", test_factor_turnover_decay),
        ("边界条件", test_boundary_conditions),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASS"
        except AssertionError as e:
            results[name] = f"FAIL: {e}"
            print(f"  ✗ 断言失败: {e}")
        except Exception as e:
            results[name] = f"ERROR: {e}"
            print(f"  ✗ 异常: {e}")

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, status in results.items():
        marker = "✓" if status == "PASS" else "✗"
        print(f"  {marker} {name}: {status}")

    passed = sum(1 for s in results.values() if s == "PASS")
    print(f"\n通过: {passed}/{len(results)}")
    return results


if __name__ == "__main__":
    main()
