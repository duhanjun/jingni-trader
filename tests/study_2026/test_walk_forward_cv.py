"""
验证测试：Walk-Forward 交叉验证框架
借鉴来源：AKQuant (github.com/akfamily/akquant) - Walk-forward Validation
         Microsoft Qlib (github.com/microsoft/qlib) - Purged Group TimeSeries Split
优化方向：strategy-model-engine - 模型训练与评估的样本外验证

当前 jingni-trader 的 strategy-model-engine 使用简单的 Purged Group TS Split，
但缺少完整的 Walk-Forward 交叉验证框架。AKQuant 和 Qlib 都提供了更完善的方案：

Qlib 的 Purged Group TS Split:
  - 训练集/验证集/测试集三段划分
  - Purge Gap 避免数据泄露

AKQuant 的 Walk-forward Validation:
  - 滚动窗口训练
  - 固定步长前进
  - 多轮评估取平均

本测试实现增强版 Walk-Forward 框架，包括：
  1. 滚动窗口 + 固定步长
  2. 锚定窗口 + 滚动测试
  3. 组合 Purge Gap 的扩展切分
  4. 因子稳定性评估（跨窗口 IC 一致性）
  5. 过拟合检测（样本内 vs 样本外表现对比）
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. Walk-Forward 交叉验证框架
# ============================================================================

class WalkForwardValidator:
    """
    Walk-Forward 交叉验证框架。

    支持三种模式：
    1. rolling: 滚动窗口（训练窗口固定，每次向前滚动一步）
    2. expanding: 锚定窗口（训练窗口不断扩展，测试窗口向前滚动）
    3. anchored: 固定训练窗口（训练窗口固定，测试窗口不断向前）
    """

    def __init__(
        self,
        train_window: int = 252,       # 训练窗口长度（交易日）
        test_window: int = 63,         # 测试窗口长度（交易日）
        step_size: int = 21,           # 步长（交易日）
        purge_gap: int = 5,            # 清除间隔（交易日）
        min_train_size: int = 126,     # 最小训练集大小
        mode: str = "rolling",         # rolling / expanding / anchored
        n_splits: Optional[int] = None,  # 最大切分次数
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.purge_gap = purge_gap
        self.min_train_size = min_train_size
        self.mode = mode
        self.n_splits = n_splits

    def split(
        self,
        dates: pd.Series,
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        生成 Walk-Forward 切分。

        参数:
          dates: 包含所有日期（去重排序）的 Series

        返回:
          [(train_dates, test_dates, purge_dates), ...]
        """
        unique_dates = pd.DatetimeIndex(sorted(dates.unique()))
        n_dates = len(unique_dates)

        if n_dates < self.min_train_size + self.test_window:
            return []

        splits = []
        start_idx = 0

        while True:
            if self.mode == "rolling":
                train_end = start_idx + self.train_window
                test_start = train_end + self.purge_gap
                test_end = test_start + self.test_window
            elif self.mode == "expanding":
                train_end = start_idx + self.train_window + len(splits) * self.step_size
                test_start = train_end + self.purge_gap
                test_end = test_start + self.test_window
            elif self.mode == "anchored":
                train_end = self.train_window
                test_start = train_end + self.purge_gap + len(splits) * self.step_size
                test_end = test_start + self.test_window

            if test_end > n_dates:
                break

            if train_end < self.min_train_size:
                if self.mode == "rolling":
                    start_idx += self.step_size
                    continue
                else:
                    break

            train_dates = unique_dates[start_idx:train_end]
            test_dates = unique_dates[test_start:test_end]
            purge_dates = unique_dates[train_end:test_start]

            splits.append((train_dates, test_dates, purge_dates))

            if self.n_splits is not None and len(splits) >= self.n_splits:
                break

            if self.mode == "rolling":
                start_idx += self.step_size
            elif self.mode == "expanding":
                start_idx = 0  # 训练窗口始终从 0 开始
            elif self.mode == "anchored":
                start_idx = 0  # 训练窗口固定

        return splits

    def validate(
        self,
        data: pd.DataFrame,
        train_fn: Callable[[pd.DataFrame], Any],
        predict_fn: Callable[[Any, pd.DataFrame], np.ndarray],
        eval_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
        feature_cols: List[str],
        label_col: str,
        date_col: str = 'date',
        code_col: str = 'code',
    ) -> Dict[str, Any]:
        """
        执行 Walk-Forward 验证。

        参数:
          data: 包含特征、标签和日期的 DataFrame
          train_fn: 训练函数 (train_data) -> model
          predict_fn: 预测函数 (model, test_data) -> predictions
          eval_fn: 评估函数 (y_true, y_pred) -> metrics_dict
          feature_cols: 特征列名
          label_col: 标签列名
          date_col: 日期列名
          code_col: 股票代码列名

        返回:
          {
            "splits": int,
            "per_split_metrics": [{...}, ...],
            "aggregate_metrics": {...},
            "stability_metrics": {...},
            "overfit_check": {...},
          }
        """
        dates = data[date_col]
        splits = self.split(dates)

        if not splits:
            return {"splits": 0, "per_split_metrics": [], "error": "无法生成切分"}

        all_train_metrics = []
        all_test_metrics = []

        for i, (train_dates, test_dates, purge_dates) in enumerate(splits):
            # 准备训练集
            train_mask = dates.isin(train_dates)
            train_data = data[train_mask]

            # 准备测试集
            test_mask = dates.isin(test_dates)
            test_data = data[test_mask]

            if len(train_data) < self.min_train_size or len(test_data) < 10:
                continue

            X_train = train_data[feature_cols].values
            y_train = train_data[label_col].values
            X_test = test_data[feature_cols].values
            y_test = test_data[label_col].values

            # 训练
            model = train_fn(train_data)

            # 预测
            y_pred_train = predict_fn(model, train_data)
            y_pred_test = predict_fn(model, test_data)

            # 评估
            train_metrics = eval_fn(y_train, y_pred_train)
            test_metrics = eval_fn(y_test, y_pred_test)

            all_train_metrics.append(train_metrics)
            all_test_metrics.append({
                "split": i,
                "train_start": str(train_dates[0].date()),
                "train_end": str(train_dates[-1].date()),
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
                "train_size": len(train_data),
                "test_size": len(test_data),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            })

        # 聚合指标
        aggregate = self._aggregate_metrics(all_test_metrics)

        # 稳定性评估
        stability = self._evaluate_stability(all_test_metrics)

        # 过拟合检测
        overfit = self._detect_overfit(all_train_metrics, all_test_metrics)

        return {
            "splits": len(splits),
            "per_split_metrics": all_test_metrics,
            "aggregate_metrics": aggregate,
            "stability_metrics": stability,
            "overfit_check": overfit,
        }

    def _aggregate_metrics(self, per_split: List[Dict]) -> Dict[str, float]:
        """聚合各切分的指标"""
        if not per_split:
            return {}

        metric_names = list(per_split[0]['test_metrics'].keys())
        aggregate = {}

        for name in metric_names:
            values = [s['test_metrics'][name] for s in per_split if name in s['test_metrics']]
            if values:
                aggregate[f"{name}_mean"] = float(np.mean(values))
                aggregate[f"{name}_std"] = float(np.std(values))
                aggregate[f"{name}_min"] = float(np.min(values))
                aggregate[f"{name}_max"] = float(np.max(values))

        return aggregate

    def _evaluate_stability(self, per_split: List[Dict]) -> Dict[str, float]:
        """
        评估因子/策略的稳定性。

        指标：
        - IC 的标准差：越小越稳定
        - IC 的正向比例：越高越好
        - 收益的标准差/均值比：变异系数
        """
        if not per_split:
            return {}

        stability = {}
        metric_names = list(per_split[0]['test_metrics'].keys())

        for name in metric_names:
            values = [s['test_metrics'][name] for s in per_split if name in s['test_metrics']]
            if values:
                mean_val = np.mean(values)
                stability[f"{name}_cv"] = float(np.std(values) / abs(mean_val)) if mean_val != 0 else float('inf')
                stability[f"{name}_positive_ratio"] = float(np.mean([v > 0 for v in values]))

        return stability

    def _detect_overfit(
        self,
        train_metrics: List[Dict],
        test_metrics: List[Dict],
    ) -> Dict[str, Any]:
        """
        检测过拟合。

        指标：
        - train-test gap: 样本内外的指标差距
        - 衰减率: 指标随时间的衰减趋势
        - 过拟合指数: gap / (train + test) 的比值
        """
        if not train_metrics or not test_metrics:
            return {"overfit_detected": False, "reason": "数据不足"}

        overfit_info = {}
        metric_names = list(test_metrics[0]['test_metrics'].keys())

        for name in metric_names:
            train_vals = [m.get(name, 0) for m in train_metrics]
            test_vals = [s['test_metrics'].get(name, 0) for s in test_metrics]

            if len(train_vals) < 2 or len(test_vals) < 2:
                continue

            train_mean = np.mean(train_vals)
            test_mean = np.mean(test_vals)

            gap = train_mean - test_mean
            denominator = abs(train_mean) + abs(test_mean)
            overfit_ratio = gap / denominator if denominator > 0 else 0

            # 测试集衰减趋势
            if len(test_vals) >= 3:
                x = np.arange(len(test_vals))
                slope = np.polyfit(x, test_vals, 1)[0]
                decay = slope / (abs(test_mean) + 1e-8)
            else:
                decay = 0

            overfit_info[name] = {
                "train_mean": float(train_mean),
                "test_mean": float(test_mean),
                "gap": float(gap),
                "overfit_ratio": float(overfit_ratio),
                "decay_trend": float(decay),
            }

        # 综合判断
        max_overfit = max(
            (info['overfit_ratio'] for info in overfit_info.values()),
            default=0
        )

        overfit_info["overall"] = {
            "max_overfit_ratio": float(max_overfit),
            "overfit_detected": max_overfit > 0.5,
            "severity": "high" if max_overfit > 0.7 else ("medium" if max_overfit > 0.5 else "low"),
        }

        return overfit_info


# ============================================================================
# 2. 因子稳定性评估（借鉴 Qlib IC 分析）
# ============================================================================

class FactorStabilityAnalyzer:
    """
    因子稳定性分析器。

    借鉴 Qlib 的 IC 分析框架，在 Walk-Forward 验证中增加因子稳定性评估：
    - 跨窗口 IC 均值和标准差
    - IC 衰减分析
    - 因子拥挤度（多窗口 IC 正相关检测）
    """

    def __init__(self):
        self._ic_history: List[Dict] = []

    def analyze(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_name: str,
        dates: pd.DatetimeIndex,
        window: int = 63,
    ) -> Dict[str, Any]:
        """
        滚动窗口 IC 分析。

        参数:
          factor_df: 因子数据
          forward_returns: 未来收益
          factor_name: 因子名称
          dates: 分析日期
          window: 滚动窗口大小
        """
        data = factor_df.merge(
            forward_returns[['code', 'date', 'ret_forward_5d']],
            on=['code', 'date'],
            how='inner'
        )

        unique_dates = sorted(dates.unique())
        ic_series = []

        for i in range(window, len(unique_dates)):
            window_dates = unique_dates[i - window:i]
            cross = data[data['date'].isin(window_dates)]
            cross = cross.dropna(subset=[factor_name, 'ret_forward_5d'])

            if len(cross) < 10:
                continue

            ic = cross[factor_name].corr(cross['ret_forward_5d'])
            if not np.isnan(ic):
                ic_series.append({
                    "end_date": unique_dates[i],
                    "ic": ic,
                    "n_stocks": len(cross),
                })

        if not ic_series:
            return {"error": "无有效 IC 数据"}

        ic_values = [item['ic'] for item in ic_series]

        return {
            "factor_name": factor_name,
            "ic_mean": float(np.mean(ic_values)),
            "ic_std": float(np.std(ic_values)),
            "ic_ir": float(np.mean(ic_values) / np.std(ic_values)) if np.std(ic_values) > 0 else 0,
            "ic_positive_ratio": float(np.mean([v > 0 for v in ic_values])),
            "ic_t_stat": float(np.mean(ic_values) / (np.std(ic_values) / np.sqrt(len(ic_values)))),
            "ic_decay": self._calc_decay(ic_values),
            "ic_series": ic_series,
        }

    def _calc_decay(self, ic_values: List[float]) -> float:
        """计算 IC 衰减趋势"""
        if len(ic_values) < 10:
            return 0

        x = np.arange(len(ic_values))
        slope = np.polyfit(x, ic_values, 1)[0]
        return float(slope)


# ============================================================================
# 3. 测试代码
# ============================================================================

def generate_test_data(n_stocks: int = 50, n_days: int = 500) -> pd.DataFrame:
    """生成模拟的多因子测试数据"""
    np.random.seed(42)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')

    rows = []
    for code in codes:
        # 构造有预测能力的因子
        true_alpha = np.random.normal(0, 0.01, n_days)

        prices = [10.0]
        for r in true_alpha:
            prices.append(prices[-1] * (1 + r + np.random.normal(0, 0.005)))
        prices = np.array(prices[1:])

        df_one = pd.DataFrame({
            'date': dates,
            'code': code,
            'close': prices,
            'factor_1': true_alpha + np.random.normal(0, 0.01, n_days),  # 有预测能力
            'factor_2': np.random.normal(0, 0.02, n_days),                # 噪音因子
            'factor_3': 0.5 * true_alpha + np.random.normal(0, 0.015, n_days),  # 部分预测能力
        })

        # 构造标签：未来 5 日收益
        df_one['forward_return'] = df_one.groupby('code')['close'].transform(
            lambda x: x.shift(-5) / x - 1
        )
        rows.append(df_one)

    df = pd.concat(rows, ignore_index=True)
    return df.sort_values(['code', 'date']).reset_index(drop=True)


def _simple_train(train_data: pd.DataFrame) -> Dict:
    """简单训练函数：线性回归"""
    from sklearn.linear_model import LinearRegression
    feature_cols = ['factor_1', 'factor_2', 'factor_3']
    X = train_data[feature_cols].dropna().values
    y = train_data['forward_return'].dropna().values

    # 确保 X 和 y 长度一致
    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]

    model = LinearRegression()
    model.fit(X, y)
    return model


def _simple_predict(model, test_data: pd.DataFrame) -> np.ndarray:
    """简单预测函数"""
    feature_cols = ['factor_1', 'factor_2', 'factor_3']
    X = test_data[feature_cols].fillna(0).values
    return model.predict(X)


def _simple_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """简单评估函数"""
    from sklearn.metrics import mean_squared_error, r2_score
    # 确保长度一致
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    ic = np.corrcoef(y_true, y_pred)[0, 1] if min_len > 1 else 0
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "ic": float(ic),
    }


def test_split_modes():
    """测试三种切分模式"""
    print("=" * 60)
    print("测试 1: Walk-Forward 切分模式")
    print("=" * 60)

    dates = pd.Series(pd.date_range('2020-01-01', periods=600, freq='B'))

    for mode in ['rolling', 'expanding', 'anchored']:
        validator = WalkForwardValidator(
            train_window=252,
            test_window=63,
            step_size=21,
            purge_gap=5,
            mode=mode,
        )
        splits = validator.split(dates)
        print(f"  {mode:12s}: {len(splits)} 个切分")

        if splits:
            first = splits[0]
            last = splits[-1]
            print(f"    首次: train={first[0][0].date()}~{first[0][-1].date()}, "
                  f"test={first[1][0].date()}~{first[1][-1].date()}")
            print(f"    末次: train={last[0][0].date()}~{last[0][-1].date()}, "
                  f"test={last[1][0].date()}~{last[1][-1].date()}")

    print()


def test_full_validation():
    """测试完整 Walk-Forward 验证流程"""
    print("=" * 60)
    print("测试 2: 完整 Walk-Forward 验证")
    print("=" * 60)

    df = generate_test_data(n_stocks=30, n_days=500)
    df = df.dropna()

    validator = WalkForwardValidator(
        train_window=200,
        test_window=50,
        step_size=40,
        purge_gap=5,
        mode="rolling",
        n_splits=5,
    )

    start = time.time()
    result = validator.validate(
        data=df,
        train_fn=_simple_train,
        predict_fn=_simple_predict,
        eval_fn=_simple_eval,
        feature_cols=['factor_1', 'factor_2', 'factor_3'],
        label_col='forward_return',
    )
    elapsed = time.time() - start

    print(f"  耗时: {elapsed:.3f}s")
    print(f"  切分数: {result['splits']}")

    # 聚合指标
    print(f"\n  聚合指标:")
    agg = result['aggregate_metrics']
    for k, v in sorted(agg.items()):
        print(f"    {k}: {v:.6f}")

    # 稳定性
    print(f"\n  稳定性指标:")
    stability = result['stability_metrics']
    for k, v in sorted(stability.items()):
        print(f"    {k}: {v:.6f}")

    # 过拟合检查
    print(f"\n  过拟合检查:")
    overfit = result['overfit_check']
    if 'overall' in overfit:
        o = overfit['overall']
        print(f"    最大过拟合比例: {o['max_overfit_ratio']:.4f}")
        print(f"    检测到过拟合: {o['overfit_detected']}")
        print(f"    严重程度: {o['severity']}")

    for metric_name, info in overfit.items():
        if metric_name != 'overall':
            print(f"    {metric_name}: "
                  f"train={info['train_mean']:.4f}, "
                  f"test={info['test_mean']:.4f}, "
                  f"gap={info['gap']:.4f}, "
                  f"decay={info['decay_trend']:.4f}")

    print()


def test_factor_stability():
    """测试因子稳定性分析"""
    print("=" * 60)
    print("测试 3: 因子稳定性分析")
    print("=" * 60)

    df = generate_test_data(n_stocks=30, n_days=500)
    df = df.dropna()

    forward_returns = df[['code', 'date']].copy()
    forward_returns['ret_forward_5d'] = df.groupby('code')['close'].transform(
        lambda x: x.shift(-5) / x - 1
    )

    analyzer = FactorStabilityAnalyzer()

    for factor_name in ['factor_1', 'factor_2', 'factor_3']:
        result = analyzer.analyze(
            factor_df=df,
            forward_returns=forward_returns,
            factor_name=factor_name,
            dates=pd.DatetimeIndex(sorted(df['date'].unique())),
            window=63,
        )

        if 'error' in result:
            print(f"  {factor_name}: {result['error']}")
            continue

        print(f"  {factor_name}: "
              f"IC_mean={result['ic_mean']:.4f}, "
              f"IC_std={result['ic_std']:.4f}, "
              f"IC_IR={result['ic_ir']:.4f}, "
              f"positive_ratio={result['ic_positive_ratio']:.2%}, "
              f"decay={result['ic_decay']:.6f}")

    print()


def test_edge_cases():
    """测试边界条件"""
    print("=" * 60)
    print("测试 4: 边界条件")
    print("=" * 60)

    # 数据不足
    dates = pd.Series(pd.date_range('2024-01-01', periods=100, freq='B'))
    validator = WalkForwardValidator(train_window=252, test_window=63)
    splits = validator.split(dates)
    assert len(splits) == 0, "数据不足应返回空"
    print("  PASS: 数据不足 (100天 < 252天训练窗口)")

    # 刚好够一个切分
    dates = pd.Series(pd.date_range('2024-01-01', periods=400, freq='B'))
    validator = WalkForwardValidator(train_window=252, test_window=63, step_size=21, purge_gap=5)
    splits = validator.split(dates)
    assert len(splits) > 0, "应该有至少一个切分"
    print(f"  PASS: 刚好够切分 (400天 -> {len(splits)} 个切分)")

    # 大量数据
    dates = pd.Series(pd.date_range('2015-01-01', periods=2500, freq='B'))
    validator = WalkForwardValidator(train_window=252, test_window=63, step_size=21, purge_gap=5)
    splits = validator.split(dates)
    assert len(splits) > 0, "应该有多个切分"
    print(f"  PASS: 大量数据 (2500天 -> {len(splits)} 个切分)")

    # 测试无 purge_gap
    validator = WalkForwardValidator(train_window=252, test_window=63, step_size=21, purge_gap=0)
    splits = validator.split(dates)
    assert len(splits) > 0, "无 purge gap 也应有效"
    print(f"  PASS: 无 purge_gap ({len(splits)} 个切分)")

    print()


def test_data_leak_prevention():
    """测试数据泄露防护"""
    print("=" * 60)
    print("测试 5: 数据泄露防护")
    print("=" * 60)

    dates = pd.Series(pd.date_range('2022-01-01', periods=500, freq='B'))
    validator = WalkForwardValidator(
        train_window=200,
        test_window=50,
        step_size=40,
        purge_gap=10,
        mode="rolling",
    )
    splits = validator.split(dates)

    for i, (train_dates, test_dates, purge_dates) in enumerate(splits):
        # 确保训练集和测试集不重叠
        train_set = set(train_dates)
        test_set = set(test_dates)

        overlap = train_set & test_set
        assert len(overlap) == 0, f"切分 {i}: 训练集和测试集有重叠: {overlap}"

        # 确保 purge gap 存在
        if len(purge_dates) > 0:
            assert train_dates[-1] < purge_dates[0], "Purge gap 顺序错误"
            assert purge_dates[-1] < test_dates[0], "Purge gap 与测试集顺序错误"

        print(f"  切分 {i}: train={train_dates[0].date()}~{train_dates[-1].date()} "
              f"| purge={len(purge_dates)}天 "
              f"| test={test_dates[0].date()}~{test_dates[-1].date()} "
              f"| 无泄露=PASS")

    print()


def test_performance():
    """测试性能"""
    print("=" * 60)
    print("测试 6: 性能测试")
    print("=" * 60)

    df = generate_test_data(n_stocks=100, n_days=600)
    df = df.dropna()

    # 多次切分性能
    for train_w, test_w in [(252, 63), (126, 21), (504, 126)]:
        validator = WalkForwardValidator(
            train_window=train_w,
            test_window=test_w,
            step_size=21,
            purge_gap=5,
            mode="rolling",
        )

        start = time.time()
        result = validator.validate(
            data=df,
            train_fn=_simple_train,
            predict_fn=_simple_predict,
            eval_fn=_simple_eval,
            feature_cols=['factor_1', 'factor_2', 'factor_3'],
            label_col='forward_return',
        )
        elapsed = time.time() - start

        print(f"  train={train_w} test={test_w}: "
              f"{result['splits']} 切分, {elapsed:.3f}s")

    print()


# ============================================================================
# 主测试入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Walk-Forward 交叉验证框架验证测试")
    print("借鉴来源: AKQuant Walk-forward Validation + Qlib PurgedGroupTS")
    print("优化方向: strategy-model-engine - 模型评估与过拟合检测")
    print("=" * 60 + "\n")

    test_split_modes()
    test_full_validation()
    test_factor_stability()
    test_edge_cases()
    test_data_leak_prevention()
    test_performance()

    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)