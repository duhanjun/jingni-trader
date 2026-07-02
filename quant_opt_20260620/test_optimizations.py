"""
优化验证测试套件

验证内容:
    1. 因子表达式引擎 - 正确性 / 性能 / 边界
    2. PIT 数据层 - 防泄漏 / 修订链 / 性能

运行: python3 quant_opt_20260620/test_optimizations.py
"""
from __future__ import annotations

import os
import sys
import time
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factor_expression_engine import (
    FactorExpressionEngine, FormulaParser, ALPHA101_FORMULAS, Operators,
)
from pit_data_layer import (
    PITStorage, PITProvider, build_pit_storage_from_records, detect_lookahead_bias,
)


# ============================================================
# 测试辅助
# ============================================================

def make_panel(n_codes: int = 20, n_days: int = 60, seed: int = 42) -> pd.DataFrame:
    """生成模拟面板数据"""
    rng = np.random.default_rng(seed)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    dates = pd.bdate_range(end='2024-12-31', periods=n_days)

    rows = []
    for code in codes:
        price = 10.0
        for d in dates:
            ret = rng.normal(0, 0.02)
            price = max(1.0, price * (1 + ret))
            open_ = price * (1 + rng.normal(0, 0.005))
            high = max(price, open_) * (1 + abs(rng.normal(0, 0.005)))
            low = min(price, open_) * (1 - abs(rng.normal(0, 0.005)))
            volume = int(rng.lognormal(15, 0.5))
            rows.append({
                'code': code, 'date': d,
                'Open': open_, 'High': high, 'Low': low, 'Close': price,
                'Volume': volume,
            })
    df = pd.DataFrame(rows).sort_values(['code', 'date']).reset_index(drop=True)
    return df


TEST_RESULTS: List[Dict] = []


def record(name: str, passed: bool, detail: str = '', extra: Dict = None):
    TEST_RESULTS.append({
        'name': name, 'passed': bool(passed), 'detail': detail,
        'extra': extra or {},
    })
    status = 'PASS' if passed else 'FAIL'
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ''))


# ============================================================
# 1. 因子表达式引擎 - 正确性测试
# ============================================================

def test_parser_basic():
    """测试公式解析器"""
    print("\n=== 1.1 公式解析器 ===")
    cases = [
        ("Close", ('ident', 'Close')),
        ("123", ('num', 123.0)),
        ("-1", ('neg', ('num', 1.0))),
        ("Rank(Close)", ('func', 'Rank', [('ident', 'Close')])),
        ("Ts_Mean(Close, 20)", ('func', 'Ts_Mean', [('ident', 'Close'), ('num', 20.0)])),
        ("-1 * Correlation(Open, Volume, 10)",
         ('binop', '*', ('neg', ('num', 1.0)),
          ('func', 'Correlation', [('ident', 'Open'), ('ident', 'Volume'), ('num', 10.0)]))),
    ]
    for formula, expected in cases:
        try:
            ast = FormulaParser(formula).parse()
            record(f"解析 '{formula}'", ast == expected, f"ast={ast}")
        except Exception as e:
            record(f"解析 '{formula}'", False, str(e))


def test_engine_correctness():
    """测试引擎求值正确性 - 与手动 pandas 计算对比"""
    print("\n=== 1.2 引擎求值正确性 ===")
    df = make_panel(n_codes=10, n_days=80)
    engine = FactorExpressionEngine()

    # 测试 1: Ts_Mean(Close, 20) == rolling(20).mean
    try:
        got = engine.evaluate("Ts_Mean(Close, 20)", df)
        exp = df.groupby('code')['Close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
        # 对齐索引比较非 NaN 部分
        mask = ~got.isna() & ~exp.isna()
        match = np.allclose(got[mask], exp[mask])
        record("Ts_Mean(Close,20) vs rolling.mean", match,
               f"匹配点数={mask.sum()}")
    except Exception as e:
        record("Ts_Mean(Close,20) vs rolling.mean", False, str(e))

    # 测试 2: Rank(Close) == groupby(date).rank(pct=True)
    try:
        got = engine.evaluate("Rank(Close)", df)
        exp = df.groupby('date')['Close'].transform(lambda x: x.rank(pct=True))
        match = np.allclose(got, exp, equal_nan=True)
        record("Rank(Close) vs groupby.rank(pct)", match)
    except Exception as e:
        record("Rank(Close) vs groupby.rank(pct)", False, str(e))

    # 测试 3: Delta(Close, 1) == diff(1)
    try:
        got = engine.evaluate("Delta(Close, 1)", df)
        exp = df.groupby('code')['Close'].transform(lambda x: x.diff(1))
        mask = ~got.isna() & ~exp.isna()
        match = np.allclose(got[mask], exp[mask])
        record("Delta(Close,1) vs diff(1)", match)
    except Exception as e:
        record("Delta(Close,1) vs diff(1)", False, str(e))

    # 测试 4: Delay(Close, 5) == shift(5)
    try:
        got = engine.evaluate("Delay(Close, 5)", df)
        exp = df.groupby('code')['Close'].transform(lambda x: x.shift(5))
        mask = ~got.isna() & ~exp.isna()
        match = np.allclose(got[mask], exp[mask])
        record("Delay(Close,5) vs shift(5)", match)
    except Exception as e:
        record("Delay(Close,5) vs shift(5)", False, str(e))

    # 测试 5: 复合公式 -1 * Correlation(Open, Volume, 10)
    try:
        got = engine.evaluate("-1 * Correlation(Open, Volume, 10)", df)
        exp = df.groupby('code', group_keys=False).apply(
            lambda g: -g['Open'].rolling(10, min_periods=5).corr(g['Volume'])
        )
        mask = ~got.isna() & ~exp.isna()
        match = np.allclose(got[mask], exp[mask], atol=1e-8)
        record("复合公式 -1*Correlation(Open,Volume,10)", match,
               f"匹配点数={mask.sum()}")
    except Exception as e:
        record("复合公式 -1*Correlation(Open,Volume,10)", False, str(e))

    # 测试 6: 算术运算 Open / Close
    try:
        got = engine.evaluate("Open / Close", df)
        exp = df['Open'] / df['Close']
        match = np.allclose(got, exp, equal_nan=True)
        record("Open / Close 逐元素除法", match)
    except Exception as e:
        record("Open / Close 逐元素除法", False, str(e))

    # 测试 7: 嵌套 Rank(Ts_Mean(Close, 20))
    try:
        got = engine.evaluate("Rank(Ts_Mean(Close, 20))", df)
        inner = df.groupby('code')['Close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
        df_tmp = df.copy()
        df_tmp['__inner'] = inner
        exp = df_tmp.groupby('date')['__inner'].transform(lambda x: x.rank(pct=True))
        mask = ~got.isna() & ~exp.isna()
        match = np.allclose(got[mask], exp[mask])
        record("嵌套 Rank(Ts_Mean(Close,20))", match)
    except Exception as e:
        record("嵌套 Rank(Ts_Mean(Close,20))", False, str(e))


def test_alpha101_formulas():
    """测试 Alpha101 公式集能否正常求值"""
    print("\n=== 1.3 Alpha101 公式集 ===")
    df = make_panel(n_codes=15, n_days=100)
    engine = FactorExpressionEngine()

    for name, formula in ALPHA101_FORMULAS.items():
        try:
            result = engine.evaluate(formula, df)
            non_null = result.notna().sum()
            ok = non_null > 0 and len(result) == len(df)
            record(f"Alpha101 {name}", ok,
                   f"非空值={non_null}/{len(result)}")
        except Exception as e:
            record(f"Alpha101 {name}", False, str(e))


# ============================================================
# 2. 因子表达式引擎 - 性能测试
# ============================================================

def test_engine_performance():
    """性能对比: 表达式引擎 vs 手动 pandas 循环"""
    print("\n=== 1.4 性能对比 ===")
    df = make_panel(n_codes=50, n_days=250)
    engine = FactorExpressionEngine()

    formula = "Rank(Ts_Mean(Close, 20))"

    # 方式 A: 表达式引擎
    t0 = time.perf_counter()
    for _ in range(3):
        r_engine = engine.evaluate(formula, df)
    t_engine = (time.perf_counter() - t0) / 3

    # 方式 B: 手动 pandas (模拟现有 factor-engine 的写法)
    t0 = time.perf_counter()
    for _ in range(3):
        df_tmp = df.copy()
        df_tmp['_m'] = df_tmp.groupby('code')['Close'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        r_manual = df_tmp.groupby('date')['_m'].transform(lambda x: x.rank(pct=True))
    t_manual = (time.perf_counter() - t0) / 3

    # 正确性
    mask = ~r_engine.isna() & ~r_manual.isna()
    correct = np.allclose(r_engine[mask], r_manual[mask])

    speedup = t_manual / t_engine if t_engine > 0 else float('inf')
    record("性能 Rank(Ts_Mean(Close,20))", correct,
           f"引擎={t_engine*1000:.1f}ms, 手动={t_manual*1000:.1f}ms, 加速比={speedup:.2f}x",
           extra={'engine_ms': t_engine * 1000, 'manual_ms': t_manual * 1000,
                  'speedup': speedup, 'rows': len(df)})


# ============================================================
# 3. 因子表达式引擎 - 边界测试
# ============================================================

def test_engine_edge_cases():
    """边界条件测试"""
    print("\n=== 1.5 边界条件 ===")
    engine = FactorExpressionEngine()

    # 边界 1: 空数据
    try:
        empty = pd.DataFrame(columns=['code', 'date', 'Close'])
        r = engine.evaluate("Rank(Close)", empty)
        record("空数据 Rank(Close)", len(r) == 0)
    except Exception as e:
        record("空数据 Rank(Close)", False, str(e))

    # 边界 2: 单只股票
    try:
        df = make_panel(n_codes=1, n_days=30)
        r = engine.evaluate("Ts_Mean(Close, 10)", df)
        record("单只股票 Ts_Mean", r.notna().sum() > 0)
    except Exception as e:
        record("单只股票 Ts_Mean", False, str(e))

    # 边界 3: 窗口大于数据长度
    try:
        df = make_panel(n_codes=5, n_days=15)
        r = engine.evaluate("Ts_Mean(Close, 100)", df)
        # 应全部为 NaN (min_periods=50 > 15)
        record("窗口>数据长度", r.isna().all(),
               f"非空数={r.notna().sum()}")
    except Exception as e:
        record("窗口>数据长度", False, str(e))

    # 边界 4: 含 NaN 的数据
    try:
        df = make_panel(n_codes=5, n_days=30)
        df.loc[df.sample(frac=0.1).index, 'Close'] = np.nan
        r = engine.evaluate("Ts_Mean(Close, 10)", df)
        record("含NaN数据 Ts_Mean", not r.isna().all())
    except Exception as e:
        record("含NaN数据 Ts_Mean", False, str(e))

    # 边界 5: 未知字段
    try:
        df = make_panel(n_codes=3, n_days=20)
        engine.evaluate("Rank(NonExistent)", df)
        record("未知字段应报错", False, "未抛出异常")
    except KeyError:
        record("未知字段应报错", True, "正确抛出 KeyError")
    except Exception as e:
        record("未知字段应报错", False, f"异常类型错误: {type(e).__name__}")

    # 边界 6: 语法错误
    try:
        engine.evaluate("Rank(Close", df)
        record("语法错误应报错", False, "未抛出异常")
    except SyntaxError:
        record("语法错误应报错", True, "正确抛出 SyntaxError")
    except Exception as e:
        record("语法错误应报错", False, f"异常类型错误: {type(e).__name__}")


# ============================================================
# 4. PIT 数据层 - 防泄漏测试
# ============================================================

def test_pit_lookahead_prevention():
    """测试 PIT 层防止未来数据泄漏"""
    print("\n=== 2.1 PIT 防泄漏 ===")
    storage = PITStorage()

    # 模拟: 某公司 2024Q2 营收, 原始值 100 (2024-08-30 发布), 修订为 120 (2024-10-15 发布)
    code = "600000.SH"
    field = "revenue"
    # period 编码: 202402 = 2024 Q2
    storage.append(field, code, date=20240830, period=202402, value=100.0)
    storage.append(field, code, date=20241015, period=202402, value=120.0)

    # 在 2024-09-15 查询: 只能看到原始值 100
    v_sep = storage.query(field, code, observe_date=20240915)
    record("修订前查询=原始值", v_sep == 100.0, f"got={v_sep}")

    # 在 2024-10-20 查询: 能看到修订值 120
    v_oct = storage.query(field, code, observe_date=20241020)
    record("修订后查询=修订值", v_oct == 120.0, f"got={v_oct}")

    # 在 2024-07-01 查询: 报告尚未发布, 应为 None
    v_jul = storage.query(field, code, observe_date=20240701)
    record("发布前查询=None", v_jul is None, f"got={v_jul}")

    # 泄漏检测: 直接用最新值 vs PIT 查询
    dates = pd.bdate_range('2024-07-01', '2024-12-31')
    pit_vals = []
    latest_vals = []
    pub_dates = []
    for d in dates:
        od = int(d.strftime('%Y%m%d'))
        pit_vals.append(storage.query(field, code, od))
        latest_vals.append(120.0)  # 直接用最新值
        pub_dates.append(pd.Timestamp('2024-10-15'))

    result = detect_lookahead_bias(
        pd.Series(pit_vals, index=dates),
        pd.Series(latest_vals, index=dates),
        pd.Series(pub_dates, index=dates),
    )
    # 在 2024-10-15 之前, latest=120 而 pit=100 或 None, 应检测到泄漏
    record("泄漏检测生效", result['leakage_count'] > 0,
           f"泄漏次数={result['leakage_count']}, 总数={result['total']}, 比例={result['leakage_ratio']:.2%}")


def test_pit_revision_chain():
    """测试多次修订链"""
    print("\n=== 2.2 PIT 修订链 ===")
    storage = PITStorage()
    code = "000001.SZ"
    field = "eps"
    period = 202404  # 2024 Q4

    # 三次修订
    storage.append(field, code, date=20250130, period=period, value=0.50)  # 业绩快报
    storage.append(field, code, date=20250315, period=period, value=0.55)  # 年报
    storage.append(field, code, date=20250420, period=period, value=0.58)  # 修订

    cases = [
        (20250101, None,  "发布前"),
        (20250201, 0.50,  "快报后"),
        (20250320, 0.55,  "年报后"),
        (20250425, 0.58,  "修订后"),
    ]
    all_ok = True
    for od, expected, label in cases:
        v = storage.query(field, code, od)
        ok = (v is None and expected is None) or (v == expected)
        if not ok:
            all_ok = False
        record(f"修订链 {label}", ok, f"expect={expected}, got={v}")


def test_pit_multi_period():
    """测试多报告期查询 - 取最近报告期"""
    print("\n=== 2.3 PIT 多报告期 ===")
    storage = PITStorage()
    code = "600519.SH"
    field = "revenue"

    # Q1 (202401) 发布于 2024-04-30, 值 300
    storage.append(field, code, date=20240430, period=202401, value=300.0)
    # Q2 (202402) 发布于 2024-08-30, 值 350
    storage.append(field, code, date=20240830, period=202402, value=350.0)

    # 2024-06-01: 只能看到 Q1 = 300
    v = storage.query(field, code, 20240601)
    record("Q2发布前取Q1", v == 300.0, f"got={v}")

    # 2024-09-01: 能看到 Q2 = 350
    v = storage.query(field, code, 20240901)
    record("Q2发布后取Q2", v == 350.0, f"got={v}")


def test_pit_panel_query():
    """测试面板批量查询"""
    print("\n=== 2.4 PIT 面板查询 ===")
    records = []
    for code in ["600000.SH", "000001.SZ", "600519.SH"]:
        for q, (pub, val) in enumerate(
            [(20240430, 100.0), (20240830, 110.0), (20241030, 120.0)], start=1
        ):
            records.append({
                'field': 'revenue', 'code': code,
                'pub_date': pub, 'period': 202400 + q, 'value': val,
            })
    df = pd.DataFrame(records)
    storage = build_pit_storage_from_records(df)

    dates = pd.bdate_range('2024-04-01', '2024-12-31')
    provider = PITProvider(storage)
    panel = provider.get_feature_panel('revenue', ["600000.SH", "000001.SZ"], dates)

    record("面板查询形状", len(panel) == 2 * len(dates),
           f"shape={panel.shape}")
    record("面板有非空值", panel['revenue'].notna().sum() > 0)
    record("存储统计", storage.stats()['total_records'] == 9,
           f"stats={storage.stats()}")


def test_pit_performance():
    """PIT 查询性能测试"""
    print("\n=== 2.5 PIT 性能 ===")
    storage = PITStorage()
    # 构建较大数据集
    n_codes = 50
    n_periods = 8  # 8 个季度
    for i in range(n_codes):
        code = f"{600000 + i:06d}.SH"
        for q in range(n_periods):
            period = 202300 + q + 1
            pub_date = 20231000 + q * 100 + 15
            storage.append('revenue', code, pub_date, period, float(100 + i + q))

    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    dates = pd.bdate_range('2023-01-01', '2024-12-31')

    t0 = time.perf_counter()
    provider = PITProvider(storage)
    panel = provider.get_feature_panel('revenue', codes, dates)
    t_elapsed = time.perf_counter() - t0

    record("PIT 面板查询性能", len(panel) == n_codes * len(dates),
           f"{n_codes}股票 x {len(dates)}天, 耗时={t_elapsed*1000:.1f}ms, "
           f"记录数={storage.stats()['total_records']}",
           extra={'elapsed_ms': t_elapsed * 1000, 'rows': len(panel),
                  'records': storage.stats()['total_records']})


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("jingni-trader 优化验证测试")
    print(f"日期: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"分支: feat/quant-opt-20260620")
    print("=" * 60)

    test_parser_basic()
    test_engine_correctness()
    test_alpha101_formulas()
    test_engine_performance()
    test_engine_edge_cases()

    test_pit_lookahead_prevention()
    test_pit_revision_chain()
    test_pit_multi_period()
    test_pit_panel_query()
    test_pit_performance()

    # 汇总
    print("\n" + "=" * 60)
    total = len(TEST_RESULTS)
    passed = sum(1 for r in TEST_RESULTS if r['passed'])
    print(f"汇总: {passed}/{total} 通过")
    print("=" * 60)

    # 保存结果
    out_path = os.path.join(os.path.dirname(__file__), 'test_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total': total, 'passed': passed, 'failed': total - passed,
            'results': TEST_RESULTS,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
