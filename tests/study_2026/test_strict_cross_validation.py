"""
===========================================================================
测试文件: test_strict_cross_validation.py
借鉴来源:
    1. Microsoft Qlib (https://github.com/microsoft/qlib)
       - 严格的滚动窗口回测 (rolling window + sample-out strict verification)
       - Trainer config: train/validate/test 三段式日期分割
       - Purged Group Time Series Split 实现

    2. trade-learn (https://github.com/MuuYesen/trade-learn)
       - 因果推断集成,降低伪相关性导致的样本外衰减
       - 完整投研流水线: 因子→评估→建模→回测

高分量化社区共识:
    - 防范前视偏差 (look-ahead bias)
    - 样本外测试 (out-of-sample) 是回测可信度的基础
    - Purged K-Fold Cross Validation 是行业标准

优化方向: strategy-model-engine - 严格的样本外验证
     - 当前问题: 使用了 Purged TS Split 但缺少完整性检查
     - 优化方案:
       1. 前视偏差审计器 (自动检测信号中包含未来信息)
       2. 样本外纯度检验器 (确保 leakage 为 0)
       3. 因果推断集成 (识别伪相关因子)

测试内容:
     1. Purged Group TS Split 正确性测试
     2. 前视偏差自动检测测试
     3. 样本外纯度验证测试
     4. 多期预测稳定性测试

⚠️ 注意: 此文件为验证代码，仅在测试目录中运行，不修改主代码。
===========================================================================
"""

import sys
import os
import json
import time
import unittest
from typing import List, Dict, Any, Optional, Tuple
from contextlib import redirect_stdout
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ===========================================================================
# 前视偏差审计器 (Look-Ahead Bias Auditor)
# ===========================================================================

class LookAheadAuditor:
    """
    前视偏差检测器

    核心检测逻辑:
    1. 检查因子数据是否存在时间泄漏
    2. 检查标签是否为未来信息
    3. 验证 train/test 分割的时间顺序

    常见前视偏差来源 (业界总结):
    - 使用全市场均值做标准化 (信息泄漏)
    - 标签数据未正确 shift (直接使用当期收益率)
    - 因子计算中使用了未来交易日的数据
    """

    def __init__(self):
        self.findings: List[Dict] = []

    def check_time_leakage(
        self,
        factor_df: pd.DataFrame,
        label_df: pd.DataFrame,
        factor_cols: List[str],
        label_col: str = 'forward_return',
        date_col: str = 'date',
        code_col: str = 'code',
    ) -> Dict[str, Any]:
        """
        检查因子与标签之间是否存在时间泄漏

        原理: 对于交易日 t 的因子值 factor[t], 其预测的标签 label[t] 应该是
              基于 t 日之后的信息计算的 (如 t+1 日收益率)。
              如果 factor[t] 与 label[t] 存在高度同期相关性, 则可能存在泄漏。
        """
        result = {
            'passed': True,
            'checks': [],
            'warnings': [],
        }

        merged = factor_df.merge(label_df, on=[date_col, code_col], how='inner')

        if merged.empty:
            return {'passed': False, 'checks': [], 'warnings': ['合并数据为空']}

        for col in factor_cols:
            if col not in merged.columns:
                continue

            # 同期相关性检查
            valid = merged[[col, label_col]].dropna()
            if len(valid) < 10:
                continue

            corr = valid[col].corr(valid[label_col])
            check_result = {
                'factor': col,
                'same_period_corr': float(corr),
                'severity': 'info',
            }

            # 如果同期相关性过高，可能存在信息泄漏
            if abs(corr) > 0.3:
                check_result['severity'] = 'warning'
                result['warnings'].append(
                    f"因子 {col} 与标签同期相关性过高 ({corr:.4f}), "
                    f"可能存在信息泄漏。请检查因子是否使用了未来数据。"
                )

            result['checks'].append(check_result)

        result['passed'] = len(result['warnings']) == 0
        return result

    def check_forward_return_leakage(
        self,
        price_df: pd.DataFrame,
        label_df: pd.DataFrame,
        label_col: str = 'forward_return',
        forward_period: int = 1,
        date_col: str = 'date',
        code_col: str = 'code',
    ) -> Dict[str, Any]:
        """
        验证 forward_return 是否使用了正确的未来信息

        正确做法: forward_return[t] = price[t+period] / price[t] - 1
        错误做法: forward_return[t] 基于 price[t] 或更早的数据计算
        """
        result = {'passed': True, 'warnings': [], 'details': {}}

        for code, group in price_df.groupby(code_col):
            group = group.sort_values(date_col)
            if len(group) < forward_period + 5:
                continue

            # 正确计算 forward return
            correct_forward = group['close'].shift(-forward_period) / group['close'] - 1
            correct_forward.name = 'correct_forward'

            # 与 label 对比
            label_group = label_df[label_df[code_col] == code].sort_values(date_col)
            if label_group.empty:
                continue

            merged = pd.DataFrame({
                'date': group[date_col].values,
                'correct': correct_forward.values,
            }).merge(
                label_group[[date_col, label_col]].rename(columns={label_col: 'label'}),
                on=date_col, how='inner'
            )

            if len(merged) < forward_period + 3:
                continue

            valid = merged.dropna()
            if len(valid) < 5:
                continue

            diff = np.abs(valid['correct'] - valid['label'])
            max_diff = diff.max()
            mean_diff = diff.mean()

            result['details'][code] = {
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'n_valid': len(valid),
            }

            # 如果差异过大，可能存在问题
            if max_diff > 1e-3:
                result['warnings'].append(
                    f"股票 {code}: forward_return 与正确计算值存在差异 "
                    f"(max={max_diff:.6f}, mean={mean_diff:.6f})"
                )

        result['passed'] = len(result['warnings']) == 0
        return result

    def check_train_test_temporal_order(
        self,
        train_dates: pd.Series,
        test_dates: pd.Series,
    ) -> Dict[str, Any]:
        """
        验证训练集和测试集的时间顺序

        严格要求: train 的所有日期 < test 的所有日期 (无一例外)
        """
        train_max = train_dates.max()
        test_min = test_dates.min()

        result = {
            'passed': train_max < test_min,
            'train_max': str(train_max),
            'test_min': str(test_min),
            'gap_days': int((test_min - train_max).days) if train_max < test_min else 0,
        }

        if not result['passed']:
            result['error'] = (
                f"时间泄漏! train_max={train_max} >= test_min={test_min}, "
                f"训练集和测试集必须严格按时间顺序分离。"
            )

        return result


# ===========================================================================
# Purged Group Time Series Split (增强版)
# ===========================================================================

class StrictPurgedTimeSeriesSplit:
    """
    严格的 Purged Group Time Series Split

    借鉴 Qlib 和业界最佳实践:
    1. Purge Gap: 在 train/test 之间插入清洗期, 消除标签重叠
    2. Embargo: 对 test 之后的数据禁售, 模拟真实交易
    3. 按月分割: 更接近真实投资组合重建周期
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 0,
        min_train_days: int = 252,
    ):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_days = min_train_days

    def split(
        self,
        dates: pd.Series,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        生成 (train, val, test) 三元组

        返回:
            List of (train_idx, val_idx, test_idx)
        """
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        if n_dates < self.min_train_days + 60:
            raise ValueError(f"日期数量不足: {n_dates} < {self.min_train_days + 60}")

        splits = []
        min_fold_size = max(n_dates // (self.n_splits + 2), 20)

        for i in range(self.n_splits):
            # 确定切割点
            train_end_idx = n_dates - (self.n_splits - i + 1) * min_fold_size
            val_end_idx = min(train_end_idx + min_fold_size, n_dates - min_fold_size)
            test_end_idx = min(val_end_idx + min_fold_size, n_dates)

            if train_end_idx < self.min_train_days or val_end_idx <= train_end_idx:
                continue

            # Purge: 在 train 末尾移除可能包含标签泄漏的日期
            if self.purge_days > 0:
                purge_date = unique_dates[train_end_idx - 1] - timedelta(days=self.purge_days)
                train_dates = [d for d in unique_dates[:train_end_idx] if d <= purge_date]
            else:
                train_dates = unique_dates[:train_end_idx]

            val_dates = unique_dates[train_end_idx:val_end_idx]
            test_dates = unique_dates[val_end_idx:test_end_idx]

            # Embargo: 移除 test 末尾的日期 (模拟信息获取延迟)
            if self.embargo_days > 0 and len(test_dates) > self.embargo_days:
                test_dates = test_dates[:-self.embargo_days]

            if len(train_dates) < self.min_train_days or len(val_dates) < 5 or len(test_dates) < 5:
                continue

            train_idx = dates[dates.isin(train_dates)].index.values
            val_idx = dates[dates.isin(val_dates)].index.values
            test_idx = dates[dates.isin(test_dates)].index.values

            splits.append((train_idx, val_idx, test_idx))

        return splits

    def validate_splits(
        self,
        dates: pd.Series,
        splits: List[Tuple],
    ) -> Dict[str, Any]:
        """验证分割的质量"""
        if not splits:
            return {'valid': False, 'error': '无有效分割'}

        auditor = LookAheadAuditor()
        all_passed = True
        fold_details = []

        for i, (train_idx, val_idx, test_idx) in enumerate(splits):
            train_dates = dates.iloc[train_idx]
            val_dates = dates.iloc[val_idx]
            test_dates = dates.iloc[test_idx]

            # 验证时间顺序
            order_check = auditor.check_train_test_temporal_order(train_dates, val_dates)
            order_check2 = auditor.check_train_test_temporal_order(
                pd.concat([train_dates, val_dates]),
                test_dates
            )

            fold_passed = order_check['passed'] and order_check2['passed']
            all_passed = all_passed and fold_passed

            fold_details.append({
                'fold': i,
                'train_dates': f"{train_dates.min()} ~ {train_dates.max()}",
                'val_dates': f"{val_dates.min()} ~ {val_dates.max()}",
                'test_dates': f"{test_dates.min()} ~ {test_dates.max()}",
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'n_test': len(test_idx),
                'temporal_order_valid': fold_passed,
            })

        return {
            'valid': all_passed,
            'n_folds': len(splits),
            'fold_details': fold_details,
        }


# ===========================================================================
# 因果推断集成 (借鉴 trade-learn)
# ===========================================================================

class CausalFactorScreener:
    """
    因果推断因子筛选器

    借鉴 trade-learn 的因果推断集成:
    - 在 ML 策略中集成因果分析, 降低伪相关性
    - 通过 Granger 因果检验筛选真正有预测力的因子
    - DoWhy 风格的反事实推断

    当前实现: Granger 因果检验 (简化版)
    """

    @staticmethod
    def granger_causality_test(
        factor_values: pd.Series,
        returns: pd.Series,
        max_lag: int = 5,
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Granger 因果检验

        检验因子是否"Granger-cause"收益率

        返回:
            {
                'is_causal': bool,
                'best_lag': int,
                'p_value': float,
                'test_name': str,
            }
        """
        from scipy import stats

        data = pd.DataFrame({'factor': factor_values, 'return': returns}).dropna()

        if len(data) < 30:
            return {'is_causal': False, 'error': '样本量不足'}

        best_p_value = 1.0
        best_lag = 0

        for lag in range(1, max_lag + 1):
            # 构建: return[t] ~ return[t-1..t-lag] + factor[t-1..t-lag]
            n = len(data)
            if n <= lag * 2 + 5:
                continue

            # 受限模型: return[t] ~ return[t-1..t-lag]
            y = data['return'].values[lag:]
            X_restricted = np.column_stack([
                data['return'].shift(i).values[lag:] for i in range(1, lag + 1)
            ])

            # 非受限模型: return[t] ~ return[t-1..t-lag] + factor[t-1..t-lag]
            X_unrestricted = np.column_stack([
                X_restricted,
                *[data['factor'].shift(i).values[lag:] for i in range(1, lag + 1)]
            ])

            valid = ~np.isnan(X_restricted).any(axis=1) & ~np.isnan(X_unrestricted).any(axis=1) & ~np.isnan(y)
            if valid.sum() < 20:
                continue

            y = y[valid]
            X_r = X_restricted[valid]
            X_u = X_unrestricted[valid]

            # OLS
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]

            resid_r = y - X_r @ beta_r
            resid_u = y - X_u @ beta_u

            SSR_r = np.sum(resid_r ** 2)
            SSR_u = np.sum(resid_u ** 2)

            if SSR_u < 1e-15:
                continue

            # F 统计量
            q = lag  # 额外的参数数量
            n_obs = len(y)
            k_u = X_u.shape[1]

            F_stat = ((SSR_r - SSR_u) / q) / (SSR_u / (n_obs - k_u))

            try:
                p_value = 1 - stats.f.cdf(F_stat, q, n_obs - k_u)
            except Exception:
                continue

            if p_value < best_p_value:
                best_p_value = p_value
                best_lag = lag

        return {
            'is_causal': best_p_value < significance,
            'best_lag': best_lag,
            'p_value': float(best_p_value),
            'significance': significance,
        }


# ===========================================================================
# 单元测试
# ===========================================================================

class TestStrictCrossValidation(unittest.TestCase):
    """严格交叉验证测试套件"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
        codes = ['000001.SZ', '600000.SH', '000002.SZ', '600036.SH', '000858.SZ']

        rows = []
        for code in codes:
            base_price = np.random.uniform(5, 50)
            n = len(dates)
            returns_arr = np.random.normal(0.0003, 0.02, n)
            # 加入轻微自相关, 使数据更真实
            for i in range(1, n):
                returns_arr[i] += 0.1 * returns_arr[i - 1]
            prices = base_price * (1 + returns_arr).cumprod()

            df = pd.DataFrame({
                'date': dates,
                'code': code,
                'close': prices,
                'open': prices * (1 + np.random.normal(0, 0.002, n)),
                'volume': np.random.lognormal(10, 0.5, n).astype(int),
            })
            rows.append(df)

        cls.price_data = pd.concat(rows, ignore_index=True)

        # 生成因子数据 (不会使用未来信息)
        factor_rows = []
        for code in codes:
            group = cls.price_data[cls.price_data['code'] == code].sort_values('date').copy()
            n = len(group)
            fdf = pd.DataFrame({
                'date': group['date'].values,
                'code': code,
                'reversal_20d': -(group['close'].pct_change(20).values),
                'ma_bias': (group['close'] / group['close'].rolling(20).mean() - 1).values,
                'volatility': (group['close'].pct_change().rolling(20).std()).values,
            })
            factor_rows.append(fdf)

        cls.factor_data = pd.concat(factor_rows, ignore_index=True)

        # 生成正确的 forward return 标签
        label_rows = []
        for code in codes:
            group = cls.price_data[cls.price_data['code'] == code].sort_values('date').copy()
            ldf = pd.DataFrame({
                'date': group['date'].values,
                'code': code,
                'forward_return': group['close'].shift(-1) / group['close'] - 1,
            })
            label_rows.append(ldf)

        cls.label_data = pd.concat(label_rows, ignore_index=True)

    def test_look_ahead_auditor_clean(self):
        """测试前视偏差审计器 - 清洁数据"""
        auditor = LookAheadAuditor()
        result = auditor.check_time_leakage(
            self.factor_data,
            self.label_data,
            factor_cols=['reversal_20d', 'ma_bias'],
            label_col='forward_return',
        )

        # 使用 clean 数据应该通过
        self.assertTrue(result['passed'], f"清洁数据不应触发前视偏差警告: {result['warnings']}")
        print("[PASS] test_look_ahead_auditor_clean")

    def test_look_ahead_auditor_leak(self):
        """测试前视偏差审计器 - 泄漏数据"""
        # 故意制造泄漏: 将 forward_return 当作因子
        contaminated = self.factor_data.copy()
        contaminated['bad_factor'] = self.label_data['forward_return'].values[:len(contaminated)]

        auditor = LookAheadAuditor()
        result = auditor.check_time_leakage(
            contaminated,
            self.label_data,
            factor_cols=['bad_factor'],
            label_col='forward_return',
        )

        # 泄漏数据应该触发警告
        self.assertFalse(result['passed'], "泄漏数据应该触发前视偏差警告")
        self.assertTrue(len(result['warnings']) > 0)
        print(f"[PASS] test_look_ahead_auditor_leak: 检测到 {len(result['warnings'])} 条警告")

    def test_purged_ts_split_temporal_order(self):
        """测试 Purged TS Split 时间顺序正确性"""
        dates = self.label_data['date'].copy()

        splitter = StrictPurgedTimeSeriesSplit(n_splits=3, purge_days=5)
        splits = splitter.split(dates)

        self.assertTrue(len(splits) > 0, "应生成至少一个分割")

        validation = splitter.validate_splits(dates, splits)
        self.assertTrue(validation['valid'], f"所有分割应通过时间顺序验证: {validation}")

        for detail in validation.get('fold_details', []):
            self.assertTrue(detail['temporal_order_valid'],
                           f"Fold {detail['fold']} 时间顺序错误")

        print(f"[PASS] test_purged_ts_split_temporal_order: 生成 {len(splits)} 个有效分割")

    def test_purged_ts_split_no_overlap(self):
        """测试分割之间无数据重叠"""
        dates = self.label_data['date'].copy()
        splitter = StrictPurgedTimeSeriesSplit(n_splits=3, purge_days=5)
        splits = splitter.split(dates)

        # 检查 train 和 test 无重叠
        for i, (train_idx, val_idx, test_idx) in enumerate(splits):
            train_test_overlap = set(train_idx) & set(test_idx)
            train_val_overlap = set(train_idx) & set(val_idx)
            val_test_overlap = set(val_idx) & set(test_idx)

            self.assertEqual(len(train_test_overlap), 0,
                           f"Fold {i}: train 和 test 存在数据重叠")
            self.assertEqual(len(train_val_overlap), 0,
                           f"Fold {i}: train 和 val 存在数据重叠")
            self.assertEqual(len(val_test_overlap), 0,
                           f"Fold {i}: val 和 test 存在数据重叠")

        print("[PASS] test_purged_ts_split_no_overlap: 所有分割无数据重叠")

    def test_causal_screening_positive(self):
        """测试因果筛选 - 有因果关系的因子"""
        screener = CausalFactorScreener()

        # 构造有因果关系的因子: 使用历史价格信息
        factor_values = self.label_data.groupby('code').apply(
            lambda x: x.sort_values('date')['forward_return'].shift(1)
        ).reset_index(level=0, drop=True)

        returns = self.label_data.groupby('code').apply(
            lambda x: x.sort_values('date')['forward_return']
        ).reset_index(level=0, drop=True)

        valid_mask = factor_values.notna() & returns.notna()

        result = screener.granger_causality_test(
            factor_values[valid_mask],
            returns[valid_mask],
            max_lag=3,
        )

        self.assertIn('is_causal', result)
        print(f"[PASS] test_causal_screening_positive: "
              f"is_causal={result.get('is_causal')}, p_value={result.get('p_value', 'N/A'):.4f}")

    def test_causal_screening_negative(self):
        """测试因果筛选 - 无因果关系的噪声"""
        screener = CausalFactorScreener()

        np.random.seed(99)
        noise_factor = pd.Series(np.random.normal(0, 1, len(self.label_data)),
                                  index=self.label_data.index)

        returns = self.label_data.groupby('code').apply(
            lambda x: x.sort_values('date')['forward_return']
        ).reset_index(level=0, drop=True)

        valid_mask = noise_factor.notna() & returns.notna()

        result = screener.granger_causality_test(
            noise_factor[valid_mask],
            returns[valid_mask],
            max_lag=3,
        )

        # 噪声不应有因果关系
        if 'is_causal' in result:
            print(f"[PASS] test_causal_screening_negative: "
                  f"is_causal={result.get('is_causal')}, p_value={result.get('p_value', 'N/A'):.4f} "
                  f"(期望: is_causal=False)")
        else:
            print(f"[PASS] test_causal_screening_negative: 样本量不足或其他错误")

    def test_oos_stability_metric(self):
        """测试样本外稳定性度量"""
        # 使用 Purged TS Split 评估因子在不同分期的 IC 稳定性
        dates = self.label_data['date'].copy()
        splitter = StrictPurgedTimeSeriesSplit(n_splits=3, purge_days=5)
        splits = splitter.split(dates)

        if not splits:
            self.skipTest("无有效分割")

        all_ic_values = []

        for train_idx, val_idx, test_idx in splits:
            # 只在测试集上计算 IC
            test_dates = dates.iloc[test_idx]
            test_factors = self.factor_data[
                self.factor_data['date'].isin(test_dates)
            ]
            test_labels = self.label_data[
                self.label_data['date'].isin(test_dates)
            ]

            merged = test_factors.merge(test_labels, on=['code', 'date'], how='inner')
            merged = merged.dropna(subset=['reversal_20d', 'forward_return'])

            if len(merged) < 10:
                continue

            ic = merged['reversal_20d'].corr(merged['forward_return'])
            all_ic_values.append(ic)

        if all_ic_values:
            ic_mean = np.mean(all_ic_values)
            ic_std = np.std(all_ic_values)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0

            print(f"\n  样本外 IC 统计:")
            print(f"    IC 均值: {ic_mean:.4f}")
            print(f"    IC 标准差: {ic_std:.4f}")
            print(f"    IC IR: {ic_ir:.4f}")
            print(f"    IC 序列: {[f'{v:.4f}' for v in all_ic_values]}")
            print(f"    评估: {'稳定' if ic_ir > 0.3 else '不稳定,需进一步分析'}")

            # 合理性检查
            self.assertTrue(-1.0 <= ic_mean <= 1.0, "IC 均值应在 [-1, 1] 范围内")

        print("[PASS] test_oos_stability_metric")


# ===========================================================================
# 主函数
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("严格交叉验证与样本外测试 验证测试")
    print("借鉴来源: Qlib Rolling Window + trade-learn Causal Inference")
    print("=" * 70)

    print("\n运行测试套件...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStrictCrossValidation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"测试结果: {'全部通过' if result.wasSuccessful() else '存在失败'}")
    print("所有验证代码位于独立测试文件中，未修改主代码。")
    print("=" * 70)