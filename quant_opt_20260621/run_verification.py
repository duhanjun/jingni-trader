"""
验证测试：向量化回测 + 因子表达式引擎 + 向量化 IC

测试内容：
1. 正确性测试：向量化回测 vs 原生回测结果一致性
2. 性能对比测试：向量化 vs 原生在不同数据规模下的耗时
3. 边界条件测试：空数据、单只股票、无信号、全涨停等
4. 因子表达式引擎：解析正确性、计算正确性、与手写实现一致性
5. 向量化 IC：与原逐日循环实现结果一致性 + 性能对比

运行: python -m quant_opt_20260621.run_verification
"""
from __future__ import annotations
import os
import sys
import time
import json
import traceback
from typing import Callable, Dict, Any, Tuple

import numpy as np
import pandas as pd

# 让 quant_opt_20260621 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_opt_20260621.vectorized_backtest import VectorizedBacktest
from quant_opt_20260621.factor_expression_engine import (
    FactorExpressionEngine, ExpressionParser, FieldNode, ConstNode,
    TsOpNode, CrossOpNode,
)
from quant_opt_20260621.vectorized_ic import VectorizedIC

# 导入原实现用于对比
# 注意：skills/backtest-engine 目录名含连字符，无法用普通 import，
# 用 importlib 按文件路径加载
import importlib.util
def _load_module(name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    # 注入相对导入所需的父包
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_BACKTEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'skills', 'backtest-engine', 'scripts'
)
# 先加载 base 包，使相对导入可用
for pkg in ['base_pkg', 'base_pkg.base_backtest', 'base_pkg.base_backtest_engine']:
    pass
# 用一个临时包名承载相对导入
_BASE_DIR = os.path.join(_BACKTEST_DIR, 'base')
sys.path.insert(0, _BACKTEST_DIR)
sys.path.insert(0, _BASE_DIR)

# 直接加载 base 模块
_base_bt = _load_module('base_backtest', os.path.join(_BASE_DIR, 'base_backtest.py'))
_base_bt_engine = _load_module('base_backtest_engine', os.path.join(_BASE_DIR, 'base_backtest_engine.py'))
# native_adapter 使用 `from ..base.base_backtest_engine import ...` 相对导入，
# 这里改用直接加载并手动注入依赖
_ADAPTERS_DIR = os.path.join(_BACKTEST_DIR, 'adapters')
sys.path.insert(0, _ADAPTERS_DIR)

# 构造一个可被相对导入解析的包结构
import types
_pkg_root = types.ModuleType('bt_scripts')
_pkg_root.__path__ = [_BACKTEST_DIR]
sys.modules['bt_scripts'] = _pkg_root
_pkg_base = types.ModuleType('bt_scripts.base')
_pkg_base.__path__ = [_BASE_DIR]
sys.modules['bt_scripts.base'] = _pkg_base
sys.modules['bt_scripts.base.base_backtest'] = _base_bt
sys.modules['bt_scripts.base.base_backtest_engine'] = _base_bt_engine
_pkg_adapters = types.ModuleType('bt_scripts.adapters')
_pkg_adapters.__path__ = [_ADAPTERS_DIR]
sys.modules['bt_scripts.adapters'] = _pkg_adapters

# 读取 native_adapter 源码，将相对导入改为绝对导入后加载
_native_path = os.path.join(_ADAPTERS_DIR, 'native_adapter.py')
with open(_native_path, 'r', encoding='utf-8') as f:
    _native_src = f.read()
_native_src_fixed = _native_src.replace(
    'from ..base.base_backtest_engine import',
    'from bt_scripts.base.base_backtest_engine import'
).replace(
    'from ..base.base_backtest import',
    'from bt_scripts.base.base_backtest import'
)
_native_mod = types.ModuleType('bt_scripts.adapters.native_adapter')
exec(compile(_native_src_fixed, _native_path, 'exec'), _native_mod.__dict__)
sys.modules['bt_scripts.adapters.native_adapter'] = _native_mod
NativeAdapter = _native_mod.NativeAdapter


# ---------------- 测试数据生成 ----------------

def make_synthetic_data(n_codes: int = 50, n_days: int = 100,
                        start_price: float = 10.0, seed: int = 42) -> pd.DataFrame:
    """生成合成日线数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=n_days)
    codes = [f'{600000 + i:06d}.SH' for i in range(n_codes)]

    rows = []
    for code in codes:
        price = start_price
        for dt in dates:
            ret = rng.normal(0, 0.02)
            price *= (1 + ret)
            open_p = price * (1 + rng.normal(0, 0.005))
            high = max(price, open_p) * (1 + abs(rng.normal(0, 0.005)))
            low = min(price, open_p) * (1 - abs(rng.normal(0, 0.005)))
            volume = int(rng.integers(1_000_000, 10_000_000))
            amount = volume * price
            # 涨跌停（约 2% 概率）
            is_limit_up = bool(rng.random() < 0.02)
            is_limit_down = bool(rng.random() < 0.01)
            rows.append({
                'code': code, 'date': dt,
                'open': round(open_p, 2), 'high': round(high, 2),
                'low': round(low, 2), 'close': round(price, 2),
                'volume': volume, 'amount': round(amount, 2),
                'turnover_rate': round(float(rng.uniform(0.5, 5.0)), 4),
                'is_limit_up': is_limit_up,
                'is_limit_down': is_limit_down,
            })
    return pd.DataFrame(rows)


def make_signals(data: pd.DataFrame, strategy: str = 'reversal_5d',
                 hold_days: int = 5) -> pd.DataFrame:
    """基于 5 日反转生成信号：每 hold_days 天调仓一次"""
    df = data.sort_values(['code', 'date']).copy()
    df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    # 反转：过去 5 日跌得多的买入，涨得多的卖出
    df['signal'] = 0
    df.loc[df['ret_5d'] < -0.03, 'signal'] = 1
    df.loc[df['ret_5d'] > 0.03, 'signal'] = -1

    # 每 hold_days 天调仓
    dates = sorted(df['date'].unique())
    rebalance_dates = set(dates[i] for i in range(0, len(dates), hold_days))
    df.loc[~df['date'].isin(rebalance_dates), 'signal'] = 0

    return df[['code', 'date', 'signal']].reset_index(drop=True)


# ---------------- 测试工具 ----------------

class TestRunner:
    def __init__(self):
        self.results = []

    def run(self, name: str, fn: Callable[[], Dict[str, Any]]):
        print(f"\n[TEST] {name}")
        t0 = time.perf_counter()
        try:
            result = fn()
            elapsed = time.perf_counter() - t0
            result['name'] = name
            result['elapsed'] = elapsed
            result['status'] = 'PASS' if result.get('passed', True) else 'FAIL'
            print(f"  -> {result['status']} ({elapsed:.3f}s)")
            if 'detail' in result:
                print(f"     {result['detail']}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            result = {
                'name': name, 'elapsed': elapsed, 'status': 'ERROR',
                'error': str(e), 'traceback': traceback.format_exc(),
            }
            print(f"  -> ERROR ({elapsed:.3f}s): {e}")
        self.results.append(result)
        return result


# ---------------- 测试用例 ----------------

def test_backtest_correctness(tr: TestRunner):
    """正确性：向量化回测 vs 原生回测，关键指标应接近"""
    def fn():
        data = make_synthetic_data(n_codes=20, n_days=60, seed=1)
        signals = make_signals(data, hold_days=5)

        native = NativeAdapter()
        vec = VectorizedBacktest()

        r_native = native.run_backtest(data, signals, init_capital=1e6)
        r_vec = vec.run_backtest(data, signals, init_capital=1e6)

        m_n = r_native['metrics']
        m_v = r_vec['metrics']

        # 两者交易笔数应相同（业务规则一致）
        n_trades_n = len(r_native['trades'])
        n_trades_v = len(r_vec['trades'])

        # 终值应非常接近（允许小数误差，因买卖顺序在 ties 上可能略有差异）
        eq_n = float(r_native['equity_curve']['equity'].iloc[-1])
        eq_v = float(r_vec['equity_curve']['equity'].iloc[-1])
        rel_diff = abs(eq_n - eq_v) / max(abs(eq_n), 1.0)

        passed = rel_diff < 0.02  # 2% 内认为一致
        return {
            'passed': passed,
            'detail': (f"trades native={n_trades_n} vec={n_trades_v}, "
                       f"final_equity native={eq_n:.2f} vec={eq_v:.2f}, "
                       f"rel_diff={rel_diff:.4%}"),
            'native_trades': n_trades_n, 'vec_trades': n_trades_v,
            'native_equity': eq_n, 'vec_equity': eq_v, 'rel_diff': rel_diff,
            'native_metrics': m_n, 'vec_metrics': m_v,
        }
    tr.run("backtest_correctness", fn)


def test_backtest_performance(tr: TestRunner):
    """性能对比：不同数据规模"""
    def fn():
        scenarios = [
            (10, 60),    # 小规模
            (50, 120),   # 中规模
            (100, 200),  # 大规模
        ]
        rows = []
        for n_codes, n_days in scenarios:
            data = make_synthetic_data(n_codes=n_codes, n_days=n_days, seed=2)
            signals = make_signals(data, hold_days=5)

            native = NativeAdapter()
            vec = VectorizedBacktest()

            t0 = time.perf_counter()
            native.run_backtest(data, signals, init_capital=1e6)
            t_native = time.perf_counter() - t0

            t0 = time.perf_counter()
            vec.run_backtest(data, signals, init_capital=1e6)
            t_vec = time.perf_counter() - t0

            speedup = t_native / t_vec if t_vec > 0 else float('inf')
            rows.append({
                'n_codes': n_codes, 'n_days': n_days,
                'n_rows': len(data),
                't_native': round(t_native, 4),
                't_vectorized': round(t_vec, 4),
                'speedup': round(speedup, 2),
            })
            print(f"     [{n_codes}x{n_days}] native={t_native:.3f}s vec={t_vec:.3f}s speedup={speedup:.2f}x")

        passed = all(r['speedup'] >= 1.0 for r in rows)
        return {
            'passed': passed,
            'detail': f"{len(rows)} scenarios, "
                      f"avg speedup={np.mean([r['speedup'] for r in rows]):.2f}x",
            'scenarios': rows,
        }
    tr.run("backtest_performance", fn)


def test_backtest_edge_cases(tr: TestRunner):
    """边界条件"""
    def fn():
        vec = VectorizedBacktest()
        results = {}

        # 1. 空数据
        try:
            r = vec.run_backtest(pd.DataFrame(), pd.DataFrame())
            results['empty_data'] = (r['metrics'] == {})
        except Exception as e:
            results['empty_data'] = False
            results['empty_data_error'] = str(e)

        # 2. 单只股票
        data = make_synthetic_data(n_codes=1, n_days=30, seed=3)
        signals = make_signals(data, hold_days=5)
        try:
            r = vec.run_backtest(data, signals)
            results['single_stock'] = not r['equity_curve'].empty
        except Exception as e:
            results['single_stock'] = False
            results['single_stock_error'] = str(e)

        # 3. 无信号（全 0）
        data = make_synthetic_data(n_codes=10, n_days=30, seed=4)
        signals = data[['code', 'date']].copy()
        signals['signal'] = 0
        try:
            r = vec.run_backtest(data, signals)
            # 无交易，净值应等于初始资金
            final_eq = float(r['equity_curve']['equity'].iloc[-1])
            results['no_signal'] = (abs(final_eq - 1e6) < 1.0) and (len(r['trades']) == 0)
        except Exception as e:
            results['no_signal'] = False
            results['no_signal_error'] = str(e)

        # 4. 全涨停日（无法买入）
        data = make_synthetic_data(n_codes=5, n_days=20, seed=5)
        data['is_limit_up'] = True  # 全涨停
        signals = make_signals(data, hold_days=5)
        try:
            r = vec.run_backtest(data, signals, price_limit=True)
            # 涨停日无法买入，交易数应为 0
            results['all_limit_up'] = (len(r['trades']) == 0)
        except Exception as e:
            results['all_limit_up'] = False
            results['all_limit_up_error'] = str(e)

        passed = all(v for k, v in results.items() if not k.endswith('_error'))
        return {
            'passed': passed,
            'detail': str(results),
            'results': results,
        }
    tr.run("backtest_edge_cases", fn)


def test_expression_parser(tr: TestRunner):
    """表达式解析正确性"""
    def fn():
        cases = [
            ("Close", FieldNode),
            ("Rank(Close)", CrossOpNode),
            ("Ts_Mean(Close, 5)", TsOpNode),
            ("Rank(Ts_Mean(Close, 5))", CrossOpNode),
            ("Add(Close, Open)", type(None)),  # BinaryOpNode
            ("Close + Open", type(None)),
            ("Close * 2", type(None)),
            ("(Close + Open) / 2", type(None)),
        ]
        results = {}
        for expr, _ in cases:
            try:
                ast = ExpressionParser(expr).parse()
                results[expr] = str(ast)
            except Exception as e:
                results[expr] = f"ERROR: {e}"

        # 验证嵌套解析
        ast = ExpressionParser("Rank(Ts_Mean(Close, 5))").parse()
        is_nested = (isinstance(ast, CrossOpNode) and
                     isinstance(ast.child, TsOpNode) and
                     isinstance(ast.child.child, FieldNode) and
                     ast.child.n == 5)

        passed = is_nested
        return {
            'passed': passed,
            'detail': f"nested parse ok={is_nested}",
            'parsed': results,
        }
    tr.run("expression_parser", fn)


def test_expression_calculation(tr: TestRunner):
    """因子表达式计算正确性：与手写 pandas 实现对比"""
    def fn():
        data = make_synthetic_data(n_codes=10, n_days=60, seed=6)
        engine = FactorExpressionEngine()

        # 1. Ts_Mean(Close, 5) 应等于手写 rolling mean
        df = data.sort_values(['code', 'date']).reset_index(drop=True)
        manual_ma5 = df.groupby('code')['close'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        expr_result = engine.calculate_expression(data, "Ts_Mean(Close, 5)", 'ma5')
        # 对齐比较（去掉 NaN）
        a = manual_ma5.values
        b = expr_result['ma5'].values
        mask = np.isfinite(a) & np.isfinite(b)
        max_diff = float(np.max(np.abs(a[mask] - b[mask]))) if mask.sum() > 0 else 0.0
        ma5_ok = max_diff < 1e-9

        # 2. Rank(-Returns(Close, 5)) 横截面排名
        manual_ret5 = df.groupby('code')['close'].pct_change(5)
        manual_rank = (-manual_ret5).groupby(df['date']).rank(pct=True)
        expr_result2 = engine.calculate_expression(data, "Rank(-Returns(Close, 5))", 'rank_rev5')
        a2 = manual_rank.values
        b2 = expr_result2['rank_rev5'].values
        mask2 = np.isfinite(a2) & np.isfinite(b2)
        max_diff2 = float(np.max(np.abs(a2[mask2] - b2[mask2]))) if mask2.sum() > 0 else 0.0
        rank_ok = max_diff2 < 1e-9

        # 3. 预置因子批量计算
        preset_factors = ['rev_5', 'mom_20', 'vol_20', 'vol_ratio', 'rank_rev_20']
        batch = engine.calculate(data, preset_factors)
        batch_ok = all(f in batch.columns for f in preset_factors)

        passed = ma5_ok and rank_ok and batch_ok
        return {
            'passed': passed,
            'detail': f"ma5_diff={max_diff2:.2e} rank_diff={max_diff2:.2e} batch_ok={batch_ok}",
            'ma5_max_diff': max_diff,
            'rank_max_diff': max_diff2,
            'batch_columns': list(batch.columns),
            'batch_shape': list(batch.shape),
        }
    tr.run("expression_calculation", fn)


def test_vectorized_ic(tr: TestRunner):
    """向量化 IC vs 原逐日循环实现"""
    def fn():
        data = make_synthetic_data(n_codes=50, n_days=120, seed=7)
        df = data.sort_values(['code', 'date']).reset_index(drop=True)

        # 构造因子与远期收益
        df['factor'] = -df.groupby('code')['close'].pct_change(5)  # 5日反转
        df['ret_forward_5d'] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-5) / x - 1
        )

        # 原实现（逐日循环，复刻 engine.py _calc_ic 逻辑）
        from scipy import stats as scs
        t0 = time.perf_counter()
        ic_list_orig = []
        for dt in df['date'].unique():
            cross = df[df['date'] == dt].dropna(subset=['factor', 'ret_forward_5d'])
            if len(cross) < 10:
                continue
            ic, _ = scs.spearmanr(cross['factor'], cross['ret_forward_5d'], nan_policy='omit')
            if not np.isnan(ic):
                ic_list_orig.append({'date': dt, 'ic': ic})
        t_orig = time.perf_counter() - t0
        ic_orig = pd.DataFrame(ic_list_orig).set_index('date')['ic']

        # 向量化实现
        t0 = time.perf_counter()
        ic_vec = VectorizedIC.calc_ic_series(df, 'factor', 'ret_forward_5d', 'spearman')
        t_vec = time.perf_counter() - t0

        # 对齐比较
        common = ic_orig.index.intersection(ic_vec.index)
        diff = (ic_orig.loc[common] - ic_vec.loc[common]).abs()
        max_diff = float(diff.max()) if len(diff) > 0 else 0.0

        speedup = t_orig / t_vec if t_vec > 0 else float('inf')
        passed = max_diff < 1e-9 and speedup >= 1.0

        return {
            'passed': passed,
            'detail': f"ic_max_diff={max_diff:.2e} speedup={speedup:.2f}x "
                      f"(orig {len(ic_orig)} periods, vec {len(ic_vec)} periods)",
            'ic_max_diff': max_diff,
            't_original': round(t_orig, 4),
            't_vectorized': round(t_vec, 4),
            'speedup': round(speedup, 2),
            'n_periods_orig': len(ic_orig),
            'n_periods_vec': len(ic_vec),
        }
    tr.run("vectorized_ic", fn)


# ---------------- 主入口 ----------------

def main():
    print("=" * 70)
    print("jingni-trader 量化优化验证 (feat/quant-opt-20260621)")
    print("=" * 70)

    tr = TestRunner()

    # 1. 回测正确性
    test_backtest_correctness(tr)
    # 2. 回测性能
    test_backtest_performance(tr)
    # 3. 回测边界
    test_backtest_edge_cases(tr)
    # 4. 表达式解析
    test_expression_parser(tr)
    # 5. 表达式计算
    test_expression_calculation(tr)
    # 6. 向量化 IC
    test_vectorized_ic(tr)

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    n_pass = sum(1 for r in tr.results if r['status'] == 'PASS')
    n_fail = sum(1 for r in tr.results if r['status'] == 'FAIL')
    n_err = sum(1 for r in tr.results if r['status'] == 'ERROR')
    for r in tr.results:
        print(f"  [{r['status']:5s}] {r['name']:30s} {r['elapsed']:.3f}s")
    print(f"\nPASS={n_pass} FAIL={n_fail} ERROR={n_err}")

    # 保存结果 JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'verification_results.json')
    serializable = []
    for r in tr.results:
        s = {}
        for k, v in r.items():
            try:
                json.dumps(v)
                s[k] = v
            except (TypeError, ValueError):
                s[k] = str(v)
        serializable.append(s)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存: {out_path}")

    return 0 if (n_fail == 0 and n_err == 0) else 1


if __name__ == '__main__':
    sys.exit(main())
