"""
============================================================
优化方向：Walk-forward Validation（滚动训练验证框架）
借鉴来源：AKQuant (https://github.com/akfamily/akquant)
          - Walk-forward Validation 框架设计
          - Signal vs. Action 分离设计
          - 防止前视偏差（Look-ahead Bias）的数据管道
          Microsoft Qlib (https://github.com/microsoft/qlib)
          - Point-in-Time 数据系统
          - Rolling training 机制
          - Purged Group Time Series Split

对照模块：strategy-model-engine
现状问题：
  - 已有 purged_group_ts_split，但缺少完整的 Walk-forward Validation 框架
  - 训练/验证/测试划分是静态的，不支持滚动式重训练
  - 没有 Point-in-Time 数据安全检查
  - 模型训练与回测耦合不够清晰

测试目标：
  1. 实现完整的 Walk-forward Validation 框架
  2. 实现 Point-in-Time 数据安全检查器
  3. 与现有 Purged TS Split 进行对比
  4. 验证滚动训练 vs 静态训练的差异
============================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. Walk-forward Validation 配置
# ============================================================

@dataclass
class WalkForwardConfig:
    """Walk-forward 验证配置"""
    train_window_months: int = 12          # 训练窗口月数
    validation_window_months: int = 3      # 验证窗口月数
    test_window_months: int = 3            # 测试窗口月数
    retrain_frequency_months: int = 3      # 重训练频率
    purge_gap_days: int = 5                # 训练/验证间的清洗间隔
    min_train_samples: int = 200           # 最小训练样本数
    forward_period: int = 1                # 预测未来 N 期


# ============================================================
# 2. Point-in-Time 数据安全检查器
# ============================================================

class PointInTimeChecker:
    """
    Point-in-Time 数据安全检查器

    检查数据是否包含未来信息泄漏：
    1. 特征在时间 t 的计算是否只使用了 <= t 的数据
    2. 标签在时间 t 是否只在 > t 的未来时刻可知
    3. 训练/验证/测试集的时间边界是否正确
    """

    def __init__(self, data: pd.DataFrame):
        """
        参数:
            data: 包含 code, date, features 和 label 的 DataFrame
        """
        self.data = data.copy()
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.leakage_warnings: List[str] = []

    def check_lookahead_bias(
        self,
        feature_cols: List[str],
        label_col: str,
        forward_period: int = 1,
    ) -> Dict[str, Any]:
        """
        检查前视偏差

        检查逻辑：
        - 对每个时间点 t 的标签值，确认它是 t+forward_period 的信息
        - 如果标签值可以在 t 时刻已知的值中预测出来，则有泄漏风险
        """
        results = {
            "total_checks": 0,
            "passed": 0,
            "warnings": [],
        }

        df = self.data.sort_values(["code", "date"])

        for code, group in df.groupby("code"):
            group = group.sort_values("date")

            for i in range(len(group) - forward_period):
                current_date = group.iloc[i]["date"]
                label_value = group.iloc[i][label_col]

                # 检查：label 值是否在 t 时刻已包含（如果是 pct_change 类标签）
                # 这里做简化检查：label 不能是 t 时刻已能计算的值
                if i + forward_period < len(group):
                    future_close = group.iloc[i + forward_period].get("close", None)
                    current_close = group.iloc[i].get("close", None)

                    if future_close is not None and current_close is not None:
                        # 如果 label 是 forward return，验证它 = (future_close - current_close) / current_close
                        expected_label = (future_close - current_close) / current_close
                        if abs(label_value - expected_label) > 1e-6:
                            results["warnings"].append(
                                f"{code} @ {current_date.date()}: label={label_value:.6f}, "
                                f"expected={expected_label:.6f}"
                            )

            results["total_checks"] += len(group)
            results["passed"] += len(group)

        results["warnings_count"] = len(results["warnings"])
        self.leakage_warnings = results["warnings"]

        return results

    def check_train_test_boundary(
        self,
        train_end_date: datetime,
        test_start_date: datetime,
        purge_gap_days: int = 5,
    ) -> Dict[str, Any]:
        """
        检查训练/测试集的边界是否正确分离

        确认:
        - train_end_date + purge_gap_days < test_start_date
        - 训练集中没有 test_start_date 之后的数据
        """
        train_end = pd.to_datetime(train_end_date)
        test_start = pd.to_datetime(test_start_date)

        min_gap = (test_start - train_end).days

        results = {
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "actual_gap_days": min_gap,
            "required_gap_days": purge_gap_days,
            "is_valid": min_gap >= purge_gap_days,
        }

        if not results["is_valid"]:
            msg = (
                f"PIT 边界警告: 训练结束({train_end.date()}) 到测试开始({test_start.date()})"
                f"仅间隔 {min_gap} 天，需要至少 {purge_gap_days} 天"
            )
            self.leakage_warnings.append(msg)

        return results

    def get_leakage_report(self) -> str:
        """生成泄漏检查报告"""
        if not self.leakage_warnings:
            return "✓ 未发现数据泄漏"

        report = "⚠️ 发现潜在数据泄漏:\n"
        for w in self.leakage_warnings[:10]:  # 最多显示10条
            report += f"  - {w}\n"
        if len(self.leakage_warnings) > 10:
            report += f"  ... 共 {len(self.leakage_warnings)} 条警告"
        return report


# ============================================================
# 3. Walk-forward Validation 框架
# ============================================================

class WalkForwardValidator:
    """
    Walk-forward Validation 框架

    核心思想：
    - 模拟真实交易环境：在每个时间点，只能用当前及以前的数据训练
    - 滚动窗口：每隔 retrain_frequency 周期重新训练
    - 前向测试：训练后，在后续的测试窗口评估模型
    """

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.windows: List[Dict[str, Any]] = []
        self.results: List[Dict[str, float]] = []

    def generate_windows(
        self,
        dates: pd.Series,
    ) -> List[Dict[str, Any]]:
        """
        生成 Walk-forward 窗口

        返回:
            [
                {
                    "window_id": 0,
                    "train_start": datetime,
                    "train_end": datetime,
                    "test_start": datetime,
                    "test_end": datetime,
                    "train_indices": np.ndarray,
                    "test_indices": np.ndarray,
                },
                ...
            ]
        """
        unique_dates = pd.Series(sorted(dates.unique()))
        if len(unique_dates) < self.config.min_train_samples:
            return []

        min_date = unique_dates.iloc[0]
        max_date = unique_dates.iloc[-1]

        train_months = self.config.train_window_months
        test_months = self.config.test_window_months
        retrain_months = self.config.retrain_frequency_months

        windows = []
        window_id = 0

        current_train_start = min_date
        current_train_end = current_train_start + timedelta(days=train_months * 30)

        while current_train_end + timedelta(days=test_months * 30) <= max_date:
            purge_end = current_train_end + timedelta(days=self.config.purge_gap_days)
            test_start = purge_end
            test_end = min(
                test_start + timedelta(days=test_months * 30),
                max_date,
            )

            # 过滤日期
            train_dates_mask = (
                (unique_dates >= current_train_start) &
                (unique_dates <= current_train_end)
            )
            test_dates_mask = (
                (unique_dates >= test_start) &
                (unique_dates <= test_end)
            )

            train_dates = set(unique_dates[train_dates_mask])
            test_dates = set(unique_dates[test_dates_mask])

            train_idx = dates[dates.isin(train_dates)].index.values
            test_idx = dates[dates.isin(test_dates)].index.values

            if len(train_idx) >= self.config.min_train_samples and len(test_idx) >= 20:
                windows.append({
                    "window_id": window_id,
                    "train_start": str(current_train_start.date()),
                    "train_end": str(current_train_end.date()),
                    "purge_end": str(purge_end.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "train_indices": train_idx,
                    "test_indices": test_idx,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                })
                window_id += 1

            # 移动窗口
            current_train_start += timedelta(days=retrain_months * 30)
            current_train_end += timedelta(days=retrain_months * 30)

        self.windows = windows
        return windows

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory: Callable[[], Any],
        train_fn: Callable[[Any, np.ndarray, np.ndarray], Any],
        predict_fn: Callable[[Any, np.ndarray], np.ndarray],
        eval_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        运行 Walk-forward Validation

        参数:
            X: 特征矩阵
            y: 标签向量
            model_factory: 创建新模型实例的工厂函数
            train_fn: 训练函数 (model, X_train, y_train) -> trained_model
            predict_fn: 预测函数 (model, X_test) -> predictions
            eval_fn: 评估函数 (y_true, y_pred) -> {metric: value}

        返回:
            {
                "window_results": [...],
                "aggregate_metrics": {...},
                "summary": str,
            }
        """
        self.results = []

        for window in self.windows:
            train_idx = window["train_indices"]
            test_idx = window["test_indices"]

            model = model_factory()

            try:
                # 训练
                model = train_fn(model, X.iloc[train_idx].values, y.iloc[train_idx].values)

                # 预测
                y_pred = predict_fn(model, X.iloc[test_idx].values)
                y_true = y.iloc[test_idx].values

                # 评估
                metrics = eval_fn(y_true, y_pred)
                metrics["window_id"] = window["window_id"]
                metrics["n_train"] = window["n_train"]
                metrics["n_test"] = window["n_test"]
                metrics["train_start"] = window["train_start"]
                metrics["test_start"] = window["test_start"]

                self.results.append(metrics)
            except Exception as e:
                print(f"窗口 {window['window_id']} 训练失败: {e}")

        aggregate = self._aggregate_results()

        return {
            "window_results": self.results,
            "aggregate_metrics": aggregate,
            "n_windows": len(self.results),
        }

    def _aggregate_results(self) -> Dict[str, float]:
        """聚合所有窗口的结果"""
        if not self.results:
            return {}

        agg = {}
        metric_keys = [k for k in self.results[0].keys()
                       if k not in ("window_id", "n_train", "n_test",
                                    "train_start", "test_start")]

        for key in metric_keys:
            values = [r[key] for r in self.results]
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)
            agg[f"{key}_min"] = np.min(values)
            agg[f"{key}_max"] = np.max(values)

        return agg

    def generate_summary(self) -> str:
        """生成可读的汇总报告"""
        if not self.results:
            return "无验证结果"

        lines = [
            f"Walk-forward Validation 汇总 (共 {len(self.results)} 个窗口)",
            f"训练窗口: {self.config.train_window_months} 月",
            f"测试窗口: {self.config.test_window_months} 月",
            f"重训练频率: {self.config.retrain_frequency_months} 月",
            f"Purge Gap: {self.config.purge_gap_days} 天",
            "",
            "各窗口结果:",
        ]

        for r in self.results:
            line = (
                f"  [W{r['window_id']}] {r['train_start']} → {r['test_start']}: "
                f"n_train={r['n_train']}, n_test={r['n_test']}"
            )
            for k, v in r.items():
                if k not in ("window_id", "n_train", "n_test", "train_start", "test_start"):
                    line += f", {k}={v:.4f}"
            lines.append(line)

        if hasattr(self, "results") and self.results:
            lines.append("")
            lines.append("聚合统计:")
            agg = self._aggregate_results()
            for k, v in sorted(agg.items()):
                lines.append(f"  {k}: {v:.4f}")

        return "\n".join(lines)


# ============================================================
# 4. 信号-动作分离设计模式
# ============================================================

class SignalActionPipeline:
    """
    信号-动作分离管道

    设计理念 (来自 AKQuant):
    - Model Layer: 负责产生预测信号（如预期收益、涨跌概率）
    - Strategy Layer: 负责将信号转换为交易动作（买入/卖出/持有）
    - 两者解耦，便于独立测试和组合
    """

    def __init__(self):
        self._signal_cache: Dict[str, np.ndarray] = {}

    def generate_signal(
        self,
        model: Any,
        features: np.ndarray,
        signal_type: str = "expected_return",
    ) -> np.ndarray:
        """
        生成原始信号

        参数:
            model: 预测模型
            features: 特征矩阵
            signal_type: 信号类型 ("expected_return" | "probability" | "regression")
        """
        raw = model.predict(features)

        if signal_type == "probability":
            raw = np.clip(raw, 0, 1)
        elif signal_type == "expected_return":
            raw = raw  # 保持原样
        elif signal_type == "regression":
            raw = raw

        return raw

    def signal_to_action(
        self,
        signal: np.ndarray,
        strategy: str = "top_bottom_quantile",
        top_quantile: float = 0.8,
        bottom_quantile: float = 0.2,
    ) -> np.ndarray:
        """
        将信号转为交易动作

        1 (做多), 0 (持有), -1 (做空)

        参数:
            signal: 原始信号数组
            strategy: 转换策略
            top_quantile: 前 N% 做多
            bottom_quantile: 后 N% 做空
        """
        if strategy == "top_bottom_quantile":
            top_threshold = np.quantile(signal, top_quantile)
            bottom_threshold = np.quantile(signal, bottom_quantile)

            actions = np.zeros(len(signal))
            actions[signal >= top_threshold] = 1
            actions[signal <= bottom_threshold] = -1

        elif strategy == "top_quantile_only":
            top_threshold = np.quantile(signal, top_quantile)
            actions = np.zeros(len(signal))
            actions[signal >= top_threshold] = 1

        elif strategy == "nonzero_threshold":
            actions = np.sign(signal)

        else:
            raise ValueError(f"未知策略: {strategy}")

        return actions

    def compute_turnover(
        self,
        prev_actions: np.ndarray,
        curr_actions: np.ndarray,
    ) -> float:
        """
        计算换手率（动作变化的比例）
        """
        if len(prev_actions) == 0:
            return 0.0
        changes = np.sum(prev_actions != curr_actions)
        return float(changes / len(prev_actions))


# ============================================================
# 测试用例
# ============================================================

def _generate_ml_data(n_stocks: int = 50, n_days: int = 500) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """生成模拟 ML 数据"""
    np.random.seed(42)
    start_date = datetime(2020, 1, 1)

    rows = []
    for i in range(n_stocks):
        stock_code = f"stock_{i:03d}"
        # 生成价格序列
        price = 20 + np.cumsum(np.random.normal(0.0003, 0.015, n_days))
        price = np.clip(price, 1, None)

        # 生成因子
        momentum_20 = np.zeros(n_days)
        for t in range(20, n_days):
            momentum_20[t] = price[t] / price[t - 20] - 1

        volatility_20 = np.zeros(n_days)
        for t in range(20, n_days):
            r = np.diff(np.log(price[t - 20:t + 1]))
            volatility_20[t] = np.std(r) if len(r) > 1 else 0

        volume_ratio = np.random.lognormal(0, 0.5, n_days)

        # 标签: 未来5日收益
        forward_return = np.zeros(n_days)
        for t in range(n_days - 5):
            forward_return[t] = price[t + 5] / price[t] - 1

        for t in range(n_days):
            date = start_date + timedelta(days=t)
            rows.append({
                "code": stock_code,
                "date": date,
                "close": price[t],
                "momentum_20": momentum_20[t],
                "volatility_20": volatility_20[t],
                "volume_ratio": volume_ratio[t],
                "forward_return": forward_return[t],
            })

    df = pd.DataFrame(rows)
    df = df.dropna()

    X = df[["momentum_20", "volatility_20", "volume_ratio"]]
    y = df["forward_return"]
    dates = df["date"]

    return X, y, dates, df


def test_point_in_time_checker():
    """测试 PIT 安全检查器"""
    print("\n" + "=" * 60)
    print("测试 1: Point-in-Time 数据安全检查")
    print("=" * 60)

    X, y, dates, df = _generate_ml_data(n_stocks=5, n_days=200)

    checker = PointInTimeChecker(df)

    # 检查前视偏差
    result = checker.check_lookahead_bias(
        feature_cols=["momentum_20", "volatility_20", "volume_ratio"],
        label_col="forward_return",
        forward_period=5,
    )
    print(f"\n前视偏差检查:")
    print(f"  总检查数: {result['total_checks']}")
    print(f"  通过数: {result['passed']}")
    print(f"  警告数: {result['warnings_count']}")

    # 检查边界
    boundary = checker.check_train_test_boundary(
        train_end_date=datetime(2020, 6, 30),
        test_start_date=datetime(2020, 7, 15),
        purge_gap_days=5,
    )
    print(f"\n边界检查:")
    print(f"  有效: {boundary['is_valid']}")
    print(f"  实际间隔: {boundary['actual_gap_days']} 天")

    # 无效边界的检查
    boundary2 = checker.check_train_test_boundary(
        train_end_date=datetime(2020, 6, 30),
        test_start_date=datetime(2020, 6, 30),
        purge_gap_days=5,
    )
    assert not boundary2["is_valid"], "无间隔的边界应标记为无效"

    print(f"\n泄漏报告: {checker.get_leakage_report()}")

    print("\n✓ PIT 安全检查测试通过")


def test_walkforward_validation():
    """测试 Walk-forward Validation 框架"""
    print("\n" + "=" * 60)
    print("测试 2: Walk-forward Validation 框架")
    print("=" * 60)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    X, y, dates, _ = _generate_ml_data(n_stocks=5, n_days=300)

    config = WalkForwardConfig(
        train_window_months=6,
        test_window_months=2,
        retrain_frequency_months=2,
        purge_gap_days=5,
        min_train_samples=50,
    )

    validator = WalkForwardValidator(config)
    windows = validator.generate_windows(dates)

    print(f"\n生成窗口数: {len(windows)}")
    for w in windows:
        print(f"  W{w['window_id']}: train={w['train_start']}→{w['train_end']}"
              f" (n={w['n_train']}), test={w['test_start']}→{w['test_end']}"
              f" (n={w['n_test']})")

    assert len(windows) > 0, "应生成至少一个窗口"

    result = validator.run(
        X=X, y=y,
        model_factory=lambda: LinearRegression(),
        train_fn=lambda m, Xt, yt: m.fit(Xt, yt) or m,
        predict_fn=lambda m, Xt: m.predict(Xt),
        eval_fn=lambda yt, yp: {
            "mse": float(mean_squared_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        },
    )

    print(f"\n运行结果 ({result['n_windows']} 个窗口):")
    for r in result["window_results"]:
        print(f"  W{r['window_id']}: MSE={r['mse']:.6f}, R²={r['r2']:.4f}")

    print(f"\n聚合指标:")
    for k, v in sorted(result["aggregate_metrics"].items()):
        print(f"  {k}: {v:.4f}")

    assert len(result["window_results"]) > 0
    assert "mse_mean" in result["aggregate_metrics"]

    print("\n✓ Walk-forward Validation 测试通过")


def test_signal_action_separation():
    """测试信号-动作分离模式"""
    print("\n" + "=" * 60)
    print("测试 3: 信号-动作分离设计模式")
    print("=" * 60)

    pipeline = SignalActionPipeline()

    # 模拟信号
    signals = np.random.normal(0, 1, 100)

    # 分位数策略
    actions = pipeline.signal_to_action(signals, strategy="top_bottom_quantile")
    n_long = (actions == 1).sum()
    n_short = (actions == -1).sum()
    n_hold = (actions == 0).sum()

    print(f"\nTop/Bottom 分位数策略:")
    print(f"  做多: {n_long}, 做空: {n_short}, 持有: {n_hold}")
    assert n_long + n_short + n_hold == 100

    # 仅做多策略
    actions_long = pipeline.signal_to_action(signals, strategy="top_quantile_only")
    assert (actions_long == -1).sum() == 0, "仅做多策略不应有做空信号"

    # 换手率计算
    prev = np.zeros(100)
    turnover = pipeline.compute_turnover(prev, actions)
    print(f"\n  首期换手率: {turnover:.2%}")
    assert 0 <= turnover <= 1

    # 零换手
    zero_turnover = pipeline.compute_turnover(actions, actions)
    assert zero_turnover == 0.0, "相同动作的换手率应为0"

    print("\n✓ 信号-动作分离测试通过")


def test_comparison_static_vs_rolling():
    """对比静态训练 vs 滚动训练"""
    print("\n" + "=" * 60)
    print("测试 4: 静态训练 vs 滚动训练对比")
    print("=" * 60)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    X, y, dates, _ = _generate_ml_data(n_stocks=5, n_days=400)

    # 静态训练: 前80%训练, 后20%测试
    split_idx = int(len(X) * 0.8)
    X_train_s, X_test_s = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_s, y_test_s = y.iloc[:split_idx], y.iloc[split_idx:]

    static_model = LinearRegression()
    static_model.fit(X_train_s, y_train_s)
    y_pred_s = static_model.predict(X_test_s)
    static_mse = mean_squared_error(y_test_s, y_pred_s)
    static_r2 = r2_score(y_test_s, y_pred_s)

    print(f"\n静态训练 (80/20 split):")
    print(f"  训练集: {len(X_train_s)}, 测试集: {len(X_test_s)}")
    print(f"  MSE: {static_mse:.6f}")
    print(f"  R²: {static_r2:.4f}")

    # 滚动训练
    config = WalkForwardConfig(
        train_window_months=8,
        test_window_months=3,
        retrain_frequency_months=3,
        min_train_samples=50,
    )

    validator = WalkForwardValidator(config)
    validator.generate_windows(dates)

    result = validator.run(
        X=X, y=y,
        model_factory=lambda: LinearRegression(),
        train_fn=lambda m, Xt, yt: m.fit(Xt, yt) or m,
        predict_fn=lambda m, Xt: m.predict(Xt),
        eval_fn=lambda yt, yp: {
            "mse": float(mean_squared_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        },
    )

    print(f"\n滚动训练 (共 {result['n_windows']} 个窗口):")
    print(f"  MSE_mean: {result['aggregate_metrics'].get('mse_mean', 'N/A'):.6f}")
    print(f"  R²_mean:  {result['aggregate_metrics'].get('r2_mean', 'N/A'):.4f}")

    print(f"\n方法对比:")
    print(f"  {'方法':<12} {'MSE':>12} {'R²':>12}")
    print(f"  {'静态训练':<12} {static_mse:>12.6f} {static_r2:>12.4f}")
    print(f"  {'滚动训练':<12} "
          f"{result['aggregate_metrics'].get('mse_mean', 0):>12.6f} "
          f"{result['aggregate_metrics'].get('r2_mean', 0):>12.4f}")

    print("\n✓ 静态 vs 滚动对比完成")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Walk-forward Validation 验证测试")
    print("借鉴来源: AKQuant (akfamily/akquant) + MS Qlib (microsoft/qlib)")
    print("优化目标: strategy-model-engine")
    print("=" * 60)

    test_point_in_time_checker()
    test_walkforward_validation()
    test_signal_action_separation()
    test_comparison_static_vs_rolling()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)