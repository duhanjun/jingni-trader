"""
优化点 1：向量化因子中性化（性能优化）

借鉴来源：
- Microsoft Qlib 的高效数据处理管道（Expression Cache + 向量化计算）
- vectorbt 的向量化回测哲学（用 Numpy 批量运算替代 Python 循环）

问题分析（对照 jingni-trader 现有实现）：
skills/factor-engine/engine.py 的 FactorEngine.neutralize() 方法使用
Python for-loop 逐日遍历：
    for dt in dates:
        cross = result[result['date'] == dt].copy()
        ...
        model = LinearRegression()
        model.fit(X, y)
        residual = y - model.predict(X)
每个交易日都重新创建 LinearRegression 对象并做布尔索引过滤，
复杂度 O(N_dates × N_factors)，在 3 年全 A 股（约 730 交易日 × 5000 股 × 15 因子）
场景下极慢。

本模块提供两种优化实现并与原实现做正确性 + 性能对比：
1. VectorizedNeutralizer：用 groupby + numpy.linalg.lstsq 替代 for-loop
2. BatchNeutralizer：进一步用矩阵分块批处理，减少 Python 层开销
"""
from __future__ import annotations

import time
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# 基线实现：复刻 jingni-trader 原始 neutralize 逻辑（用于对比）
# ---------------------------------------------------------------------------
def neutralize_baseline(
    factor_df: pd.DataFrame,
    neutralize_mcap: bool = True,
    neutralize_industry: bool = True,
) -> pd.DataFrame:
    """复刻 skills/factor-engine/engine.py 的原始 neutralize 实现。"""
    if not neutralize_industry and not neutralize_mcap:
        return factor_df
    if factor_df.empty:
        return factor_df

    result = factor_df.copy()
    factor_cols = [c for c in factor_df.columns if c not in ['code', 'date', 'industry']]

    for factor in factor_cols:
        if factor not in result.columns:
            continue
        dates = result['date'].unique()
        neutralized_values = pd.Series(index=result.index, dtype=float)

        for dt in dates:
            cross = result[result['date'] == dt].copy()
            if len(cross) < 30:
                neutralized_values.loc[cross.index] = cross[factor]
                continue

            X_vars = []
            if neutralize_mcap and 'lncap' in cross.columns:
                X_vars.append('lncap')
            if neutralize_industry and 'industry' in cross.columns:
                industry_dummies = pd.get_dummies(cross['industry'], prefix='ind')
                for col in industry_dummies.columns:
                    cross[col] = industry_dummies[col].values
                    X_vars.append(col)

            if not X_vars:
                neutralized_values.loc[cross.index] = cross[factor]
                continue

            X = cross[X_vars].fillna(0).values
            y = cross[factor].fillna(0).values
            try:
                model = LinearRegression()
                model.fit(X, y)
                residual = y - model.predict(X)
                neutralized_values.loc[cross.index] = residual
            except Exception:
                neutralized_values.loc[cross.index] = cross[factor]

        result[f"{factor}_neutral"] = neutralized_values

    return result


# ---------------------------------------------------------------------------
# 优化实现 1：groupby + numpy.linalg.lstsq
# ---------------------------------------------------------------------------
def neutralize_vectorized(
    factor_df: pd.DataFrame,
    neutralize_mcap: bool = True,
    neutralize_industry: bool = True,
) -> pd.DataFrame:
    """
    向量化中性化：用 groupby('date') 替代 for-loop，
    用 numpy.linalg.lstsq 替代 sklearn.LinearRegression。

    关键优化点：
    1. 一次性构建行业哑变量，避免每日重复 get_dummies
    2. groupby 后用 apply 批量处理，减少布尔索引开销
    3. numpy.lstsq 比 sklearn.LinearRegression 少了对象创建与参数校验开销
    """
    if not neutralize_industry and not neutralize_mcap:
        return factor_df
    if factor_df.empty:
        return factor_df

    result = factor_df.copy()
    factor_cols = [c for c in factor_df.columns if c not in ['code', 'date', 'industry']]

    # 一次性构建行业哑变量矩阵（全期复用）
    x_cols: List[str] = []
    if neutralize_mcap and 'lncap' in result.columns:
        x_cols.append('lncap')
    if neutralize_industry and 'industry' in result.columns:
        dummies = pd.get_dummies(result['industry'], prefix='ind', dtype=float)
        result = pd.concat([result, dummies], axis=1)
        x_cols.extend(dummies.columns.tolist())

    if not x_cols:
        return result

    # 预填充 X 矩阵的 NaN
    result[x_cols] = result[x_cols].fillna(0.0)

    def _neutralize_group(group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        if n < 30:
            for factor in factor_cols:
                if factor in group.columns:
                    group[f"{factor}_neutral"] = group[factor]
            return group

        X = group[x_cols].to_numpy(dtype=float)
        # 加截距项
        X_with_const = np.column_stack([np.ones(n), X])
        for factor in factor_cols:
            if factor not in group.columns:
                continue
            y = group[factor].fillna(0.0).to_numpy(dtype=float)
            try:
                # lstsq 返回最小二乘解，比 sklearn 快且无对象开销
                coef, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
                residual = y - X_with_const @ coef
                group[f"{factor}_neutral"] = residual
            except Exception:
                group[f"{factor}_neutral"] = group[factor]
        return group

    result = result.groupby('date', group_keys=False, sort=False).apply(_neutralize_group)

    # 清理临时哑变量列
    dummy_cols = [c for c in result.columns if c.startswith('ind_')]
    result = result.drop(columns=dummy_cols, errors='ignore')
    return result


# ---------------------------------------------------------------------------
# 优化实现 2：纯 Numpy 批处理（无 groupby.apply 开销）
# ---------------------------------------------------------------------------
def neutralize_batch_numpy(
    factor_df: pd.DataFrame,
    neutralize_mcap: bool = True,
    neutralize_industry: bool = True,
) -> pd.DataFrame:
    """
    进一步优化：完全用 Numpy 矩阵运算，避免 pandas groupby.apply 的开销。
    按日期分组后，对每组直接做矩阵运算，结果写回预分配的数组。
    """
    if not neutralize_industry and not neutralize_mcap:
        return factor_df
    if factor_df.empty:
        return factor_df

    result = factor_df.copy()
    factor_cols = [c for c in factor_df.columns if c not in ['code', 'date', 'industry']]

    # 构建设计矩阵列
    x_cols: List[str] = []
    if neutralize_mcap and 'lncap' in result.columns:
        x_cols.append('lncap')
    if neutralize_industry and 'industry' in result.columns:
        dummies = pd.get_dummies(result['industry'], prefix='ind', dtype=float)
        result = pd.concat([result, dummies], axis=1)
        x_cols.extend(dummies.columns.tolist())

    if not x_cols:
        return result

    result[x_cols] = result[x_cols].fillna(0.0)

    # 按日期分组索引（用 dict 替代 groupby，更快）
    date_groups: dict = {}
    for idx, dt in enumerate(result['date'].values):
        date_groups.setdefault(dt, []).append(idx)

    y_full_map = {f: result[f].fillna(0.0).to_numpy(dtype=float) for f in factor_cols if f in result.columns}
    X_full = result[x_cols].to_numpy(dtype=float)

    for factor in factor_cols:
        if factor not in result.columns:
            continue
        neutral_col = f"{factor}_neutral"
        # 用可写副本（pandas 3.0 下 to_numpy() 可能返回只读视图）
        out = np.array(y_full_map[factor], dtype=float, copy=True)

        for dt, indices in date_groups.items():
            idx_arr = np.array(indices, dtype=int)
            n = len(idx_arr)
            if n < 30:
                continue
            X = X_full[idx_arr]
            y = y_full_map[factor][idx_arr]
            X_with_const = np.column_stack([np.ones(n), X])
            try:
                coef, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
                out[idx_arr] = y - X_with_const @ coef
            except Exception:
                out[idx_arr] = y
        result[neutral_col] = out

    dummy_cols = [c for c in result.columns if c.startswith('ind_')]
    result = result.drop(columns=dummy_cols, errors='ignore')
    return result


# ---------------------------------------------------------------------------
# 测试数据生成
# ---------------------------------------------------------------------------
def make_synthetic_factor_data(
    n_dates: int = 100,
    n_stocks: int = 200,
    n_industries: int = 8,
    n_factors: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """生成与真实 A 股因子数据结构一致的合成数据。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    industries = [f"IND_{i % n_industries}" for i in range(n_stocks)]

    rows = []
    for dt in dates:
        for i, code in enumerate(codes):
            row = {
                'code': code,
                'date': dt,
                'industry': industries[i],
                'lncap': rng.normal(22, 1.5),
            }
            for f in range(n_factors):
                row[f"factor_{f}"] = rng.normal(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 验证测试
# ---------------------------------------------------------------------------
def run_tests() -> dict:
    """运行正确性、性能、边界测试，返回结构化结果。"""
    results = {
        "optimization": "向量化因子中性化",
        "borrowed_from": "Microsoft Qlib 数据处理管道 + vectorbt 向量化哲学",
        "correctness": {},
        "performance": {},
        "boundary": {},
    }

    # ---- 正确性测试：三种实现结果应近似一致 ----
    print("[1/3] 正确性测试：对比 baseline / vectorized / batch_numpy 结果...")
    df = make_synthetic_factor_data(n_dates=40, n_stocks=200, n_factors=3, seed=1)
    base = neutralize_baseline(df)
    vec = neutralize_vectorized(df)
    bat = neutralize_batch_numpy(df)

    factor_cols = [c for c in df.columns if c not in ['code', 'date', 'industry', 'lncap']]
    max_diff_vec = 0.0
    max_diff_bat = 0.0
    for f in factor_cols:
        col = f"{f}_neutral"
        if col in base.columns and col in vec.columns:
            diff = (base[col].fillna(0) - vec[col].fillna(0)).abs().max()
            max_diff_vec = max(max_diff_vec, float(diff))
        if col in base.columns and col in bat.columns:
            diff = (base[col].fillna(0) - bat[col].fillna(0)).abs().max()
            max_diff_bat = max(max_diff_bat, float(diff))

    results["correctness"] = {
        "max_abs_diff_vectorized_vs_baseline": round(max_diff_vec, 8),
        "max_abs_diff_batch_vs_baseline": round(max_diff_bat, 8),
        "tolerance": 1e-6,
        "passed": max_diff_vec < 1e-6 and max_diff_bat < 1e-6,
    }
    print(f"  vectorized vs baseline 最大绝对差: {max_diff_vec:.2e}")
    print(f"  batch_numpy vs baseline 最大绝对差: {max_diff_bat:.2e}")
    print(f"  通过: {results['correctness']['passed']}")

    # ---- 性能测试：不同数据规模下的耗时对比 ----
    print("[2/3] 性能测试：不同数据规模耗时对比...")
    scenarios = [
        ("small", 50, 100, 3),
        ("medium", 120, 300, 5),
        ("large", 250, 500, 8),
    ]
    perf = {}
    for name, n_dates, n_stocks, n_factors in scenarios:
        data = make_synthetic_factor_data(n_dates, n_stocks, n_factors, seed=7)

        t0 = time.perf_counter()
        neutralize_baseline(data)
        t_base = time.perf_counter() - t0

        t0 = time.perf_counter()
        neutralize_vectorized(data)
        t_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        neutralize_batch_numpy(data)
        t_bat = time.perf_counter() - t0

        speedup_vec = t_base / t_vec if t_vec > 0 else float('inf')
        speedup_bat = t_base / t_bat if t_bat > 0 else float('inf')
        perf[name] = {
            "scale": f"{n_dates}日 x {n_stocks}股 x {n_factors}因子",
            "baseline_sec": round(t_base, 4),
            "vectorized_sec": round(t_vec, 4),
            "batch_numpy_sec": round(t_bat, 4),
            "speedup_vectorized": round(speedup_vec, 2),
            "speedup_batch_numpy": round(speedup_bat, 2),
        }
        print(f"  [{name}] {n_dates}日x{n_stocks}股x{n_factors}因子: "
              f"baseline={t_base:.3f}s vec={t_vec:.3f}s({speedup_vec:.1f}x) "
              f"batch={t_bat:.3f}s({speedup_bat:.1f}x)")
    results["performance"] = perf

    # ---- 边界条件测试 ----
    print("[3/3] 边界条件测试...")
    boundary = {}

    # 空数据
    empty = pd.DataFrame(columns=['code', 'date', 'industry', 'lncap', 'factor_0'])
    try:
        r = neutralize_vectorized(empty)
        boundary["empty_data"] = {"passed": r.empty, "note": "空 DataFrame 不报错"}
    except Exception as e:
        boundary["empty_data"] = {"passed": False, "error": str(e)}

    # 单日数据（不足 30 只，应原样返回）
    small = make_synthetic_factor_data(n_dates=1, n_stocks=20, n_factors=2, seed=3)
    try:
        r = neutralize_vectorized(small)
        col = "factor_0_neutral"
        if col in r.columns:
            unchanged = np.allclose(
                r[col].fillna(0).to_numpy(),
                small['factor_0'].fillna(0).to_numpy(),
            )
            boundary["small_group_skip"] = {"passed": unchanged, "note": "样本<30 时原样返回"}
        else:
            boundary["small_group_skip"] = {"passed": False, "note": "未生成 neutral 列"}
    except Exception as e:
        boundary["small_group_skip"] = {"passed": False, "error": str(e)}

    # 缺失行业字段
    no_ind = make_synthetic_factor_data(n_dates=10, n_stocks=100, n_factors=2, seed=4)
    no_ind = no_ind.drop(columns=['industry'])
    try:
        r = neutralize_vectorized(no_ind, neutralize_industry=False, neutralize_mcap=True)
        boundary["missing_industry"] = {
            "passed": "factor_0_neutral" in r.columns,
            "note": "仅市值中性化时正常工作",
        }
    except Exception as e:
        boundary["missing_industry"] = {"passed": False, "error": str(e)}

    # 含 NaN 的因子值
    nan_df = make_synthetic_factor_data(n_dates=10, n_stocks=100, n_factors=2, seed=5)
    nan_df.loc[nan_df.index[:50], 'factor_0'] = np.nan
    try:
        r = neutralize_vectorized(nan_df)
        boundary["nan_values"] = {
            "passed": "factor_0_neutral" in r.columns and not r["factor_0_neutral"].isna().all(),
            "note": "因子含 NaN 时不崩溃",
        }
    except Exception as e:
        boundary["nan_values"] = {"passed": False, "error": str(e)}

    all_passed = all(v.get("passed", False) for v in boundary.values())
    boundary["all_passed"] = all_passed
    results["boundary"] = boundary
    for k, v in boundary.items():
        if k == "all_passed":
            continue
        print(f"  [{k}] 通过: {v.get('passed')} - {v.get('note', v.get('error', ''))}")

    return results


if __name__ == "__main__":
    import json
    res = run_tests()
    print("\n=== 测试结果汇总 ===")
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
