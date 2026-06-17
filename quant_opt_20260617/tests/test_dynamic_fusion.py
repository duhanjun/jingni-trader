"""
动态因子融合测试
"""
import unittest
import numpy as np
import pandas as pd

from quant_opt_20260617.dynamic_factor_fusion import (
    DynamicFactorFusion, FusionConfig, FusionMethod
)
from quant_opt_20260617.tests._synthetic_data import (
    generate_synthetic_a_share_data, compute_forward_returns
)


class TestDynamicFactorFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 30 只股票 × 1000 天
        cls.data = generate_synthetic_a_share_data(n_stocks=30, n_days=1000, seed=42)
        cls.factor_cols = [
            'ret_5d', 'ret_20d', 'turnover_5d', 'turnover_change',
            'momentum_volume', 'noise_factor'
        ]
        # 准备前向收益
        cls.fwd = compute_forward_returns(cls.data, forward_periods=[5, 20])
        # 截取部分因子数据用于测试
        cls.factor_df = cls.data[['code', 'date'] + cls.factor_cols].dropna().copy()

    def test_01_static_weights(self):
        """静态 IC 加权：所有权重和为 1"""
        fuser = DynamicFactorFusion(FusionConfig(method=FusionMethod.STATIC_IC_WEIGHTED))
        ic_results = {
            "ret_forward_5d": [
                {"factor": "ret_5d", "ic_ir": 1.2},
                {"factor": "ret_20d", "ic_ir": -0.8},
                {"factor": "noise_factor", "ic_ir": 0.05},
            ]
        }
        w = fuser._static_ic_weights(["ret_5d", "ret_20d", "noise_factor"], ic_results)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=5)
        # ret_5d IC_IR 绝对值最大，应获得最大权重
        self.assertGreater(w["ret_5d"], w["ret_20d"])
        self.assertGreater(w["ret_5d"], w["noise_factor"])

    def test_02_static_empty_ic_fallback(self):
        """空 IC 结果应降级为等权"""
        fuser = DynamicFactorFusion(FusionConfig(method=FusionMethod.STATIC_IC_WEIGHTED))
        w = fuser._static_ic_weights(["a", "b", "c"], {})
        self.assertAlmostEqual(sum(w.values()), 1.0, places=5)
        for v in w.values():
            self.assertAlmostEqual(v, 1.0 / 3, places=5)

    def test_03_ema_weights_sum_to_one(self):
        """EMA IC 加权：权重和为 1"""
        fuser = DynamicFactorFusion(FusionConfig(
            method=FusionMethod.EMA_IC_WEIGHTED,
            ema_halflife_days=60,
            lookback_days=252,
        ))
        w, _ = fuser._ema_ic_weights(
            self.factor_df, self.factor_cols, self.fwd
        )
        self.assertAlmostEqual(sum(w.values()), 1.0, places=5)
        # 所有权重 >= 0
        for v in w.values():
            self.assertGreaterEqual(v, 0.0)

    def test_04_ema_dead_factor_filtered(self):
        """无效因子（噪声）应被赋 0 权重"""
        fuser = DynamicFactorFusion(FusionConfig(
            method=FusionMethod.EMA_IC_WEIGHTED,
            ema_halflife_days=60,
            lookback_days=252,
            ic_floor=0.05,  # 较高门槛
        ))
        w, _ = fuser._ema_ic_weights(
            self.factor_df, self.factor_cols, self.fwd
        )
        # noise_factor 应被识别为死因子
        self.assertLess(w.get("noise_factor", 0), 0.05,
                        f"noise_factor 未被有效过滤: w={w.get('noise_factor')}")

    def test_05_adaptive_topk(self):
        """Adaptive topK：top_k 个因子被选中"""
        fuser = DynamicFactorFusion(FusionConfig(
            method=FusionMethod.ADAPTIVE_TOPK,
            top_k=2,
            ic_floor=0.01,
            ema_halflife_days=30,
        ))
        w, _ = fuser._adaptive_topk_weights(
            self.factor_df, self.factor_cols, self.fwd
        )
        # 每个有非零权重的因子应该是 1/top_k
        nonzero = [v for v in w.values() if v > 0]
        if nonzero:
            for v in nonzero:
                self.assertAlmostEqual(v, 1.0 / 2, places=5)

    def test_06_fuse_returns_valid_alpha(self):
        """完整 fuse 流程：返回的 alpha_score 应有非空值"""
        fuser = DynamicFactorFusion(FusionConfig(
            method=FusionMethod.EMA_IC_WEIGHTED,
            ema_halflife_days=60,
        ))
        result = fuser.fuse(self.factor_df, forward_returns=self.fwd)
        self.assertIn('code', result.columns)
        self.assertIn('date', result.columns)
        self.assertIn('alpha_score', result.columns)
        self.assertGreater(result['alpha_score'].notna().sum(), 0)
        # alpha_score 应该是截面百分位的加权求和，范围大致在 [0, 1]
        self.assertGreaterEqual(result['alpha_score'].min(), 0.0)
        self.assertLessEqual(result['alpha_score'].max(), 1.0 + 1e-6)

    def test_07_compare_methods(self):
        """compare_methods：所有方法都应返回 OK 状态"""
        fuser = DynamicFactorFusion(FusionConfig(
            ema_halflife_days=60, top_k=3, ic_floor=0.01,
        ))
        comparison = fuser.compare_methods(self.factor_df, self.fwd)
        # 四种方法都应成功
        self.assertEqual(len(comparison), 4)
        statuses = comparison['status'].tolist()
        self.assertTrue(all(s == "OK" for s in statuses),
                        f"存在失败: {statuses}")

    def test_08_alpha158_subset_fusion(self):
        """类 Alpha158 因子集的融合结果应非空"""
        factor_cols_small = ['ret_5d', 'ret_20d', 'turnover_5d']
        sub = self.factor_df[['code', 'date'] + factor_cols_small].dropna()
        fuser = DynamicFactorFusion(FusionConfig(
            method=FusionMethod.EMA_IC_WEIGHTED,
            ema_halflife_days=30,
        ))
        result = fuser.fuse(sub, forward_returns=self.fwd)
        self.assertGreater(len(result), 0)
        self.assertGreater(result['alpha_score'].notna().sum(), 0)

    def test_09_smooth_blends_with_uniform(self):
        """smooth 启用时，权重应该略向均匀靠拢"""
        fuser_smooth = DynamicFactorFusion(FusionConfig(
            method=FusionMethod.EMA_IC_WEIGHTED,
            ema_halflife_days=60,
            weight_smooth=True, smooth_eps=0.3,
        ))
        fuser_no_smooth = DynamicFactorFusion(FusionConfig(
            method=FusionMethod.EMA_IC_WEIGHTED,
            ema_halflife_days=60,
            weight_smooth=False, smooth_eps=0.0,
        ))
        w_s, _ = fuser_smooth._ema_ic_weights(
            self.factor_df, self.factor_cols, self.fwd
        )
        w_n, _ = fuser_no_smooth._ema_ic_weights(
            self.factor_df, self.factor_cols, self.fwd
        )
        # 平滑后的权重应该比无平滑的更接近均匀（1/n）
        n = len(self.factor_cols)
        uniform = 1.0 / n
        for f in self.factor_cols:
            if w_n.get(f, 0) > 0:
                d_smooth = abs(w_s[f] - uniform)
                d_no = abs(w_n[f] - uniform)
                self.assertLessEqual(d_smooth, d_no + 1e-9,
                                     f"因子 {f} 平滑后未更接近均匀")

    def test_10_no_forward_returns(self):
        """无 forward_returns 时 EMA 方法应降级为静态"""
        fuser = DynamicFactorFusion(FusionConfig(method=FusionMethod.EMA_IC_WEIGHTED))
        w, _ = fuser._ema_ic_weights(self.factor_df, self.factor_cols, None)
        # 降级后权重和仍为 1
        self.assertAlmostEqual(sum(w.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
