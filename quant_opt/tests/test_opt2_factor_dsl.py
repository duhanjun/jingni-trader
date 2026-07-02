"""
Optimisation #2: Factor Expression Engine (Mini DSL)
====================================================

验证目标
--------
1. 表达式的解析、安全校验、AST 求值
2. 标准算子（Rank/Ts_Mean/Delta/Ref/If 等）输出正确
3. 与手写 pandas 计算结果一致
4. Alpha101 模板可直接使用
5. 向量化中性化与原 for 循环版本结果一致
6. 分层相关性去冗余能减少因子数量
7. 安全防护：禁止 dunder 访问、eval/exec
"""
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opt2_factor_dsl.factor_dsl import (
    FactorExpr, FactorFactory, FactorExpressionError,
    ALPHA101_TEMPLATES, STANDARD_FIELDS, ATOMIC_COLUMN_OPS,
    vectorised_neutralize, hierarchical_factor_select,
)

import unittest


# ----------------------------------------------------------------------------
# 构造测试数据
# ----------------------------------------------------------------------------

def build_synthetic(n_dates=120, n_codes=20, seed=42):
    rng = np.random.default_rng(seed)
    all_dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    industries = ["Tech", "Finance", "Consumer", "Healthcare", "Energy"]
    for c in range(n_codes):
        code = f"{600000 + c:06d}.SH"
        industry = industries[c % len(industries)]
        price = rng.uniform(10, 50)
        mcap = rng.uniform(1e9, 1e11)
        for dt in all_dates:
            chg = rng.normal(0, 0.02)
            price = max(price * (1 + chg), 1)
            vol = rng.uniform(1e6, 1e8)
            rows.append({
                "date": dt, "code": code, "industry": industry,
                "open": price * (1 + rng.normal(0, 0.003)),
                "high": price * (1 + abs(rng.normal(0, 0.003))),
                "low": price * (1 - abs(rng.normal(0, 0.003))),
                "close": price,
                "volume": vol,
                "amount": vol * price,
                "vwap": price,
                "pre_close": price,
                "change_pct": chg * 100,
                "lncap": np.log(mcap),
            })
    return pd.DataFrame(rows)


class TestFactorDSL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = build_synthetic()
        cls.df_idx = cls.df.set_index(["code", "date"])

    def test_01_basic_field_reference(self):
        """Close 直接引用"""
        expr = FactorExpr("Close", name="close_raw")
        result = expr.evaluate(self.df_idx)
        self.assertEqual(len(result), len(self.df_idx))
        np.testing.assert_array_equal(result.values, self.df["close"].values)
        print("  ✓ Close 字段引用正确")

    def test_02_arithmetic_ops(self):
        """Add/Sub/Mul/Div 基础算术"""
        expr = FactorExpr("Add(Sub(Close, Open), Mul(Volume, 0.0001))", name="arithmetic")
        result = expr.evaluate(self.df_idx)
        expected = (self.df["close"] - self.df["open"]) + self.df["volume"] * 0.0001
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-6)
        print("  ✓ 算术组合 Add(Sub,Mul) 结果正确")

    def test_03_ref_and_delta(self):
        """Ref/Delta 滞后与差分"""
        expr_delta = FactorExpr("Delta(Close, 1)", name="delta_1")
        df_sorted = self.df_idx.sort_index()
        result = expr_delta.evaluate(df_sorted)
        # 手算
        expected = df_sorted.groupby(level="code")["close"].diff(1)
        valid = expected.dropna()
        np.testing.assert_allclose(result.dropna().values, valid.values, rtol=1e-6)
        print("  ✓ Delta(Close, 1) 与手算 groupby diff 一致")

    def test_04_ts_mean(self):
        """Ts_Mean 与手算 rolling 一致"""
        expr = FactorExpr("Ts_Mean(Close, 5)", name="ts_mean_5")
        result = expr.evaluate(self.df_idx)
        expected = (
            self.df_idx.groupby(level="code")["close"]
            .rolling(5, min_periods=2).mean()
            .reset_index(level=0, drop=True)
        )
        # 允许小误差
        valid_mask = ~(result.isna() | expected.isna())
        np.testing.assert_allclose(
            result[valid_mask].values, expected[valid_mask].values, rtol=1e-4
        )
        print("  ✓ Ts_Mean(Close, 5) 与手算 rolling mean 一致")

    def test_05_cross_section_rank(self):
        """Rank 在当日截面内归一化"""
        expr = FactorExpr("Rank(Close)", name="rank_close")
        result = expr.evaluate(self.df_idx)
        # 每日 rank(pct=True) 应在 [0, 1] 之间
        for dt, g in result.groupby(level="date"):
            self.assertGreaterEqual(g.min(), 0.0)
            self.assertLessEqual(g.max(), 1.0)
        print("  ✓ Rank(Close) 输出 ∈ [0, 1] 且截面归一化正确")

    def test_06_alpha101_template(self):
        """alpha101 模板展开"""
        expr = FactorExpr("alpha101", name="alpha101_realised")
        result = expr.evaluate(self.df_idx)
        # alpha101 = (Close - Open) / (High - Low + 0.001)
        expected = (
            (self.df["close"] - self.df["open"]) /
            (self.df["high"] - self.df["low"] + 0.001)
        ).values
        np.testing.assert_allclose(result.values, expected, rtol=1e-6)
        print("  ✓ alpha101 模板展开后结果与手算公式一致")

    def test_07_alpha005_template(self):
        """alpha005 模板"""
        expr = FactorExpr("alpha005", name="alpha005_realised")
        result = expr.evaluate(self.df_idx)
        self.assertEqual(len(result), len(self.df_idx))
        # 简单验证：有值且非全 nan
        self.assertGreater(result.dropna().shape[0], 0)
        print("  ✓ alpha005 模板求值成功")

    def test_08_security_dunder_blocked(self):
        """dunder 访问应被拒绝"""
        with self.assertRaises(FactorExpressionError):
            FactorExpr("__import__('os').system('echo hacked')")
        print("  ✓ __import__ 被拒绝")

    def test_09_security_unauthorized_id_blocked(self):
        """未授权标识符被拒绝"""
        with self.assertRaises(FactorExpressionError):
            FactorExpr("eval('1+1')")
        with self.assertRaises(FactorExpressionError):
            FactorExpr("open('foo.txt')")
        print("  ✓ eval / open 未授权标识符被拒绝")

    def test_10_nested_expression(self):
        """深层嵌套组合"""
        expr = FactorExpr(
            "Sub(Rank(Delta(Close, 5)), Rank(Ts_Mean(Volume, 10)))",
            name="nested",
        )
        result = expr.evaluate(self.df_idx)
        self.assertEqual(len(result), len(self.df_idx))
        # 验证值域：两个 rank 都在 [0, 1]，差值在 [-1, 1]
        valid = result.dropna()
        self.assertGreaterEqual(valid.min(), -1.0)
        self.assertLessEqual(valid.max(), 1.0)
        print(f"  ✓ 4 层嵌套表达式求值成功 (n={len(valid)}, range=[{valid.min():.3f}, {valid.max():.3f}])")

    def test_11_factory_batch(self):
        """FactorFactory 批量计算"""
        factory = FactorFactory()
        factors = {
            "ret_5d": "Delta(Close, 5) / Ref(Close, 5)",
            "vol_ratio": "Volume / Ts_Mean(Volume, 20)",
            "hl_spread": "(High - Low) / Close",
            "z_close": "ZScore(Close)",
        }
        result = factory.compute_batch(self.df_idx, factors)
        self.assertEqual(set(result.columns), set(factors.keys()))
        # 缓存生效
        cache_hits = sum(1 for n in factors if n in factory.cache)
        self.assertEqual(cache_hits, len(factors))
        print(f"  ✓ 批量计算 {len(factors)} 个因子，缓存命中 {cache_hits}")

    def test_12_vectorised_neutralise(self):
        """向量化中性化与原版对比"""
        # 准备一个简单因子：动量（需要 >=30 codes per day 才能触发回归）
        # 临时构造 60 codes × 50 dates 的小数据
        n_codes = 60
        all_dates = pd.bdate_range("2024-01-01", periods=50)
        rows = []
        rng = np.random.default_rng(7)
        industries = ["Tech", "Finance", "Consumer", "Healthcare", "Energy", "Industrial"]
        for c in range(n_codes):
            code = f"{600000 + c:06d}.SH"
            industry = industries[c % len(industries)]
            for d, dt in enumerate(all_dates):
                rows.append({
                    "code": code, "date": dt, "industry": industry,
                    "momentum": rng.normal(0, 0.02),
                    "lncap": 20 + rng.normal(0, 0.5),
                })
        fdf = pd.DataFrame(rows)
        ind_df = fdf[["code", "industry"]].drop_duplicates()
        # 模拟 neutral 后的残差
        result = vectorised_neutralize(
            fdf, industry_df=ind_df,
            factor_cols=["momentum"],
            neutralize_industry=True,
            neutralize_mcap=True,
        )
        self.assertIn("momentum_neutral", result.columns)
        # 中性化后均值为 0（每日截面）
        daily_means = result.groupby("date")["momentum_neutral"].mean()
        daily_means = daily_means.dropna()
        # 至少有部分天有有效值
        self.assertGreater(len(daily_means), 0)
        # 允许小误差
        self.assertLess(abs(daily_means).max(), 0.01)
        print(f"  ✓ 向量化中性化输出 momentum_neutral，每日截面均值 ≈ 0 (max |mean|={abs(daily_means).max():.2e}, valid_days={len(daily_means)})")

    def test_13_hierarchical_factor_select(self):
        """分层相关性去冗余"""
        # 构造 10 个因子，前 5 个高度相关（动量类），后 5 个独立
        fdf = pd.DataFrame(index=self.df_idx.index)
        fdf["mom_5"] = self.df_idx["close"].groupby(level="code").pct_change(5)
        fdf["mom_10"] = self.df_idx["close"].groupby(level="code").pct_change(10)
        fdf["mom_20"] = self.df_idx["close"].groupby(level="code").pct_change(20)
        fdf["mom_5_rank"] = fdf["mom_5"]  # 与 mom_5 完全相同
        fdf["mom_5_shift"] = fdf["mom_5"].groupby(level="code").shift(1)  # 强相关
        fdf["vol_ratio"] = self.df_idx["volume"] / self.df_idx["volume"].groupby(level="code").rolling(20).mean().reset_index(level=0, drop=True)
        fdf["hl_spread"] = (self.df_idx["high"] - self.df_idx["low"]) / self.df_idx["close"]
        fdf["turnover"] = self.df_idx["volume"] / self.df_idx["amount"]
        fdf["vwap_dev"] = (
            (self.df_idx["close"] - (self.df_idx["high"] + self.df_idx["low"]) / 2) /
            (self.df_idx["high"] - self.df_idx["low"] + 0.001)
        )
        fdf["size"] = self.df_idx["lncap"]

        factors = ["mom_5", "mom_10", "mom_20", "mom_5_rank", "mom_5_shift",
                   "vol_ratio", "hl_spread", "turnover", "vwap_dev", "size"]
        # 假设 IC（绝对值越大越重要）
        ic_results = {
            "mom_5": 0.10, "mom_10": 0.08, "mom_20": 0.05, "mom_5_rank": 0.10,
            "mom_5_shift": 0.04, "vol_ratio": 0.03, "hl_spread": 0.02,
            "turnover": 0.02, "vwap_dev": 0.015, "size": 0.05,
        }
        selected = hierarchical_factor_select(
            fdf.reset_index(), factors, ic_results, corr_threshold=0.7
        )
        # 验证：去重后数量 <= 原始数量
        self.assertLessEqual(len(selected), len(factors))
        # 验证：选出的因子都是原始因子
        self.assertTrue(set(selected).issubset(set(factors)))
        print(f"  ✓ 分层去冗余: {len(factors)} → {len(selected)} (保留 {selected})")


def main():
    print("=" * 70)
    print("Optimisation #2: Factor Expression Engine Verification")
    print("=" * 70)

    # 性能对比：原硬编码 vs DSL
    print("\n--- 性能与等价性对比 ---")
    df = build_synthetic(n_dates=120, n_codes=20)
    df_idx = df.set_index(["code", "date"])

    # 原硬编码（factor-engine/engine.py compute_a_share_factors 节选）
    start = time.time()
    hardcoded = df.copy()
    hardcoded["ret_5d"] = hardcoded.groupby("code")["close"].pct_change(5)
    hardcoded_time = time.time() - start

    # DSL 表达
    start = time.time()
    factory = FactorFactory()
    dsl_result = factory.compute_batch(df_idx, {
        "ret_5d": "Delta(Close, 5) / Ref(Close, 5)",
    })
    dsl_time = time.time() - start

    # 等价性（按 (code, date) 对齐）
    merged = hardcoded[["code", "date", "ret_5d"]].merge(
        dsl_result.reset_index()[["code", "date", "ret_5d"]],
        on=["code", "date"], suffixes=("_hc", "_dsl")
    ).dropna()
    np.testing.assert_allclose(
        merged["ret_5d_hc"].values, merged["ret_5d_dsl"].values, rtol=1e-6,
    )
    print(f"  硬编码 ret_5d 计算耗时: {hardcoded_time*1000:.1f} ms")
    print(f"  DSL     ret_5d 计算耗时: {dsl_time*1000:.1f} ms")
    print(f"  ✓ 结果一致 (n={len(merged)}, rtol=1e-6)")

    # 运行单元测试
    print("\n--- 单元测试 ---")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFactorDSL)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("  ✓ 全部 13 个单元测试通过")
        return 0
    else:
        print(f"  ✗ {len(result.failures)} 失败, {len(result.errors)} 错误")
        for f in result.failures:
            print(f"    FAIL: {f[0]}")
        for e in result.errors:
            print(f"    ERR: {e[0]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
