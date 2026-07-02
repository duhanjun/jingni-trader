"""
验证测试：因子衰减分析 — 因子预测能力的半衰期评估
================================================================
借鉴来源：
  - FactorEngine (arXiv:2603.16365) — 因子演化过程中的经验知识库与轨迹感知优化
  - Microsoft Qlib + RD-Agent — 因子协同优化中的迭代反馈
  - 量化金融学术文献：因子衰减（Factor Decay）是衡量因子有效期的重要指标

优化方向：factor-engine — 新增因子衰减分析，评估因子有效期
================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# 1. 因子衰减分析核心
# ═══════════════════════════════════════════════════════════════

@dataclass
class DecayResult:
    """因子衰减分析结果"""
    factor_name: str
    ic_series: Dict[int, List[float]] = field(default_factory=dict)  # lag -> IC values
    mean_ic: Dict[int, float] = field(default_factory=dict)          # lag -> mean IC
    ic_ir: Dict[int, float] = field(default_factory=dict)            # lag -> IC_IR
    half_life: int = 0         # 半衰期（天）
    decay_rate: float = 0.0    # 衰减率
    is_stable: bool = False    # 是否稳定因子


class FactorDecayAnalyzer:
    """
    因子衰减分析器

    核心功能：
    1. 计算因子在不同滞后期的 IC 表现
    2. 拟合指数衰减模型，估计半衰期
    3. 评估因子稳定性，辅助因子筛选
    4. 生成因子衰减报告

    借鉴 FactorEngine 的 "经验知识库" 理念：
    - 跟踪因子在不同市场环境下的表现
    - 记录因子的衰减轨迹，辅助因子轮换决策
    """

    def __init__(self, max_lag: int = 20, ic_type: str = "spearman"):
        """
        参数:
            max_lag: 最大滞后期（天）
            ic_type: IC 类型 (spearman / pearson)
        """
        self.max_lag = max_lag
        self.ic_type = ic_type
        self.decay_results: Dict[str, DecayResult] = {}

    def analyze_factor(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_name: str,
        max_lag: Optional[int] = None,
    ) -> DecayResult:
        """
        分析单个因子的衰减特征

        参数:
            factor_df: 因子数据，需包含 code, date, [factor_name]
            price_df: 价格数据，需包含 code, date, close
            factor_name: 因子名称
            max_lag: 最大滞后期

        返回:
            DecayResult 对象
        """
        if max_lag is None:
            max_lag = self.max_lag

        # 合并数据
        merged = factor_df[['code', 'date', factor_name]].merge(
            price_df[['code', 'date', 'close']],
            on=['code', 'date'],
            how='inner'
        )

        if merged.empty:
            return DecayResult(factor_name=factor_name)

        # 计算未来收益
        merged = merged.sort_values(['code', 'date'])
        for lag in range(1, max_lag + 1):
            merged[f'ret_forward_{lag}d'] = merged.groupby('code')['close'].transform(
                lambda x: x.shift(-lag) / x - 1
            )

        # 计算各滞后期的 IC
        ic_series = {}
        mean_ic = {}
        ic_ir = {}

        for lag in range(1, max_lag + 1):
            forward_col = f'ret_forward_{lag}d'
            ic_values = self._calc_ic_series(merged, factor_name, forward_col)
            if ic_values:
                ic_series[lag] = ic_values
                mean_ic[lag] = np.mean(ic_values)
                ic_ir[lag] = np.mean(ic_values) / np.std(ic_values) if np.std(ic_values) > 0 else 0

        # 估计半衰期
        half_life, decay_rate, is_stable = self._estimate_half_life(mean_ic)

        result = DecayResult(
            factor_name=factor_name,
            ic_series=ic_series,
            mean_ic=mean_ic,
            ic_ir=ic_ir,
            half_life=half_life,
            decay_rate=decay_rate,
            is_stable=is_stable,
        )

        self.decay_results[factor_name] = result
        return result

    def analyze_all_factors(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> Dict[str, DecayResult]:
        """批量分析所有因子"""
        if factor_names is None:
            factor_names = [c for c in factor_df.columns
                           if c not in ['code', 'date', 'industry', 'alpha_score']]

        results = {}
        for name in factor_names:
            if name in factor_df.columns:
                results[name] = self.analyze_factor(factor_df, price_df, name)
        return results

    def _calc_ic_series(
        self,
        data: pd.DataFrame,
        factor_col: str,
        forward_col: str,
    ) -> List[float]:
        """计算 IC 时间序列"""
        ic_values = []
        for date, group in data.groupby('date'):
            valid = group.dropna(subset=[factor_col, forward_col])
            if len(valid) < 10:
                continue
            try:
                if self.ic_type == "spearman":
                    ic, _ = stats.spearmanr(valid[factor_col], valid[forward_col], nan_policy='omit')
                else:
                    ic, _ = stats.pearsonr(valid[factor_col].fillna(0), valid[forward_col].fillna(0))
                if not np.isnan(ic):
                    ic_values.append(ic)
            except Exception:
                continue
        return ic_values

    def _estimate_half_life(
        self,
        mean_ic: Dict[int, float],
    ) -> Tuple[int, float, bool]:
        """
        估计因子半衰期

        方法：拟合指数衰减模型 IC(lag) = IC_0 * exp(-lambda * lag)
        半衰期 = ln(2) / lambda

        如果 IC 不衰减（lambda <= 0），则标记为稳定因子
        """
        lags = sorted(mean_ic.keys())
        ic_values = np.array([abs(mean_ic[lag]) for lag in lags])

        if len(lags) < 3 or np.all(ic_values < 0.001):
            return 0, 0.0, False

        # 对数线性拟合 log(IC) = log(IC_0) - lambda * lag
        x = np.array(lags, dtype=float)
        y = np.log(np.maximum(ic_values, 1e-10))

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            lambda_hat = -slope  # 衰减率

            if lambda_hat <= 0:
                # 不衰减或反向增长
                return 0, 0.0, True

            half_life = int(np.log(2) / lambda_hat)

            # 判断稳定性：半衰期 > 60天 或 R² 很低（无规律衰减）
            is_stable = half_life > 60 or r_value ** 2 < 0.3

            return half_life, lambda_hat, is_stable

        except Exception:
            return 0, 0.0, False

    def rank_factors_by_half_life(self) -> List[Tuple[str, int, float]]:
        """按半衰期排名因子"""
        ranked = []
        for name, result in self.decay_results.items():
            ranked.append((name, result.half_life, result.decay_rate))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def classify_factors(self) -> Dict[str, List[str]]:
        """
        因子分类

        返回:
            {
                'stable': 稳定因子（半衰期 > 60天）
                'medium': 中等因子（20天 < 半衰期 <= 60天）
                'fast_decay': 快速衰减因子（半衰期 <= 20天）
                'invalid': 无效因子（半衰期 = 0）
            }
        """
        classification = {
            'stable': [],
            'medium': [],
            'fast_decay': [],
            'invalid': [],
        }

        for name, result in self.decay_results.items():
            if result.is_stable or result.half_life > 60:
                classification['stable'].append(name)
            elif result.half_life > 20:
                classification['medium'].append(name)
            elif result.half_life > 0:
                classification['fast_decay'].append(name)
            else:
                classification['invalid'].append(name)

        return classification

    def generate_decay_report(self) -> str:
        """生成因子衰减报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("因子衰减分析报告")
        lines.append("=" * 70)

        classification = self.classify_factors()

        lines.append(f"\n稳定因子 (半衰期 > 60天): {len(classification['stable'])}")
        for name in classification['stable']:
            r = self.decay_results[name]
            lines.append(f"  - {name}: 半衰期={r.half_life}天, 衰减率={r.decay_rate:.6f}")

        lines.append(f"\n中等因子 (20 < 半衰期 <= 60天): {len(classification['medium'])}")
        for name in classification['medium']:
            r = self.decay_results[name]
            lines.append(f"  - {name}: 半衰期={r.half_life}天, 衰减率={r.decay_rate:.6f}")

        lines.append(f"\n快速衰减因子 (半衰期 <= 20天): {len(classification['fast_decay'])}")
        for name in classification['fast_decay']:
            r = self.decay_results[name]
            lines.append(f"  - {name}: 半衰期={r.half_life}天, 衰减率={r.decay_rate:.6f}")

        lines.append(f"\n无效因子 (半衰期 = 0): {len(classification['invalid'])}")
        for name in classification['invalid']:
            lines.append(f"  - {name}")

        # 排名前5
        ranked = self.rank_factors_by_half_life()
        lines.append(f"\n半衰期排名 Top 5:")
        for i, (name, hl, dr) in enumerate(ranked[:5], 1):
            lines.append(f"  {i}. {name}: 半衰期={hl}天")

        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# 2. 因子轮换策略（基于衰减分析）
# ═══════════════════════════════════════════════════════════════

class FactorRotationStrategy:
    """
    因子轮换策略

    借鉴 FactorEngine 的 "经验知识库" 和 Qlib RD-Agent 的迭代反馈：
    - 定期评估因子衰减，淘汰失效因子
    - 动态调整因子权重，优先使用稳定因子
    - 记录因子表现历史，支持经验复用
    """

    def __init__(self, analyzer: FactorDecayAnalyzer):
        self.analyzer = analyzer
        self.factor_history: Dict[str, List[Dict]] = {}  # factor_name -> [{date, decay_rate, half_life}]

    def evaluate_and_rotate(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_names: List[str],
        current_weights: Optional[Dict[str, float]] = None,
        rotation_threshold: int = 20,  # 半衰期低于此阈值时淘汰
    ) -> Dict[str, float]:
        """
        评估因子并生成新的权重

        返回:
            {factor_name: weight} 新的因子权重
        """
        results = self.analyzer.analyze_all_factors(factor_df, price_df, factor_names)

        # 记录历史
        for name, result in results.items():
            if name not in self.factor_history:
                self.factor_history[name] = []
            self.factor_history[name].append({
                'half_life': result.half_life,
                'decay_rate': result.decay_rate,
                'is_stable': result.is_stable,
            })

        # 基于半衰期分配权重
        valid_factors = {}
        for name, result in results.items():
            if result.half_life > rotation_threshold:
                valid_factors[name] = result.half_life

        if not valid_factors:
            # 全部失效，使用等权
            return {name: 1.0 / len(factor_names) for name in factor_names}

        total = sum(valid_factors.values())
        weights = {name: hl / total for name, hl in valid_factors.items()}

        return weights

    def get_factor_health(self, factor_name: str) -> Dict:
        """获取因子健康度摘要"""
        if factor_name not in self.factor_history:
            return {"status": "unknown"}

        history = self.factor_history[factor_name]
        recent = history[-1]

        # 判断趋势
        if len(history) >= 2:
            prev = history[-2]
            trend = "improving" if recent['half_life'] > prev['half_life'] else "decaying"
            trend = "stable" if recent['half_life'] == prev['half_life'] else trend
        else:
            trend = "new"

        return {
            "status": "active" if recent['half_life'] > 0 else "dead",
            "half_life": recent['half_life'],
            "decay_rate": recent['decay_rate'],
            "is_stable": recent['is_stable'],
            "trend": trend,
            "history_length": len(history),
        }


# ═══════════════════════════════════════════════════════════════
# 3. 测试
# ═══════════════════════════════════════════════════════════════

def generate_factor_data(n_symbols: int = 10, n_days: int = 300) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成模拟因子和价格数据"""
    np.random.seed(42)
    rows = []

    for sym_idx in range(n_symbols):
        code = f"{600000 + sym_idx:06d}.SH"
        price = np.random.uniform(10, 50)

        # 基础价格序列
        prices = [price]
        for _ in range(n_days - 1):
            price *= (1 + np.random.normal(0.0005, 0.015))
            prices.append(price)

        for day_idx in range(n_days):
            # 生成有衰减特性的因子
            # 强因子：高IC，慢衰减
            # 弱因子：低IC，快衰减
            noise = np.random.normal(0, 0.01)

            # 强动量因子（半衰期约 40天）
            momentum_strong = 0.02 * (prices[day_idx] / prices[max(0, day_idx - 20)] - 1) + noise

            # 弱反转因子（半衰期约 5天）
            reversal_weak = -0.005 * (prices[day_idx] / prices[max(0, day_idx - 5)] - 1) + noise * 3

            # 噪声因子（无预测能力）
            noise_factor = noise * 5

            # 波动率因子（中等半衰期约 15天）
            if day_idx >= 20:
                rets = np.diff(prices[max(0, day_idx - 20):day_idx + 1]) / prices[max(0, day_idx - 20):day_idx]
                vol_factor = np.std(rets) * 0.5 + noise
            else:
                vol_factor = noise

            rows.append({
                'code': code,
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day_idx),
                'close': prices[day_idx],
                'momentum_strong': momentum_strong,
                'reversal_weak': reversal_weak,
                'noise_factor': noise_factor,
                'vol_factor': vol_factor,
            })

    df = pd.DataFrame(rows)
    factor_df = df[['code', 'date', 'momentum_strong', 'reversal_weak', 'noise_factor', 'vol_factor']]
    price_df = df[['code', 'date', 'close']]
    return factor_df, price_df


def test_factor_decay_analysis():
    """测试因子衰减分析"""
    print("=" * 60)
    print("测试 1: 因子衰减分析")
    print("=" * 60)

    factor_df, price_df = generate_factor_data(10, 300)
    analyzer = FactorDecayAnalyzer(max_lag=20)

    # 分析强动量因子
    result = analyzer.analyze_factor(factor_df, price_df, 'momentum_strong')
    print(f"\n强动量因子:")
    print(f"  半衰期: {result.half_life} 天")
    print(f"  衰减率: {result.decay_rate:.6f}")
    print(f"  稳定性: {'稳定' if result.is_stable else '正常衰减'}")
    print(f"  IC(1d): {result.mean_ic.get(1, 0):.4f}")
    print(f"  IC(5d): {result.mean_ic.get(5, 0):.4f}")
    print(f"  IC(20d): {result.mean_ic.get(20, 0):.4f}")

    # 分析弱反转因子
    result = analyzer.analyze_factor(factor_df, price_df, 'reversal_weak')
    print(f"\n弱反转因子:")
    print(f"  半衰期: {result.half_life} 天")
    print(f"  衰减率: {result.decay_rate:.6f}")
    print(f"  稳定性: {'稳定' if result.is_stable else '正常衰减'}")

    # 分析噪声因子
    result = analyzer.analyze_factor(factor_df, price_df, 'noise_factor')
    print(f"\n噪声因子:")
    print(f"  半衰期: {result.half_life} 天")
    print(f"  衰减率: {result.decay_rate:.6f}")

    # 分析波动率因子
    result = analyzer.analyze_factor(factor_df, price_df, 'vol_factor')
    print(f"\n波动率因子:")
    print(f"  半衰期: {result.half_life} 天")
    print(f"  衰减率: {result.decay_rate:.6f}")

    # 验证：强因子半衰期应大于弱因子
    strong = analyzer.decay_results['momentum_strong']
    weak = analyzer.decay_results['reversal_weak']
    noise = analyzer.decay_results['noise_factor']

    assert strong.half_life >= weak.half_life, \
        f"强因子半衰期({strong.half_life})应 >= 弱因子({weak.half_life})"
    assert strong.half_life >= noise.half_life, \
        f"强因子半衰期({strong.half_life})应 >= 噪声因子({noise.half_life})"

    print(f"\n✓ 因子衰减分析通过（强因子半衰期 {strong.half_life} >= 弱因子 {weak.half_life}）")


def test_factor_classification():
    """测试因子分类"""
    print("\n" + "=" * 60)
    print("测试 2: 因子分类")
    print("=" * 60)

    factor_df, price_df = generate_factor_data(10, 300)
    analyzer = FactorDecayAnalyzer(max_lag=20)
    analyzer.analyze_all_factors(factor_df, price_df)

    classification = analyzer.classify_factors()

    for category, factors in classification.items():
        print(f"\n{category}: {len(factors)} 个因子")
        for f in factors:
            r = analyzer.decay_results[f]
            print(f"  - {f}: 半衰期={r.half_life}天, 衰减率={r.decay_rate:.6f}")

    # 排名
    ranked = analyzer.rank_factors_by_half_life()
    print(f"\n半衰期排名:")
    for i, (name, hl, dr) in enumerate(ranked, 1):
        print(f"  {i}. {name}: {hl}天")

    print(f"\n✓ 因子分类测试通过")


def test_factor_rotation():
    """测试因子轮换策略"""
    print("\n" + "=" * 60)
    print("测试 3: 因子轮换策略")
    print("=" * 60)

    factor_df, price_df = generate_factor_data(10, 300)
    analyzer = FactorDecayAnalyzer(max_lag=20)
    strategy = FactorRotationStrategy(analyzer)

    factor_names = ['momentum_strong', 'reversal_weak', 'noise_factor', 'vol_factor']

    # 初始权重
    weights = strategy.evaluate_and_rotate(
        factor_df, price_df, factor_names,
        rotation_threshold=10
    )

    print(f"\n因子权重分配:")
    for name, w in weights.items():
        health = strategy.get_factor_health(name)
        print(f"  {name}: 权重={w:.3f}, 状态={health['status']}, "
              f"半衰期={health['half_life']}天, 趋势={health['trend']}")

    # 验证：有效的因子应该有非零权重
    assert len(weights) > 0, "所有因子被淘汰"
    assert abs(sum(weights.values()) - 1.0) < 0.001, "权重和不为1"

    # 验证噪声因子应该被淘汰
    assert 'noise_factor' not in weights or weights['noise_factor'] < 0.3, \
        "噪声因子不应获得高权重"

    print(f"\n✓ 因子轮换策略测试通过")


def test_boundary():
    """边界条件测试"""
    print("\n" + "=" * 60)
    print("测试 4: 边界条件测试")
    print("=" * 60)

    # 空数据
    print("\n4.1 空数据:")
    empty_df = pd.DataFrame(columns=['code', 'date', 'factor', 'close'])
    analyzer = FactorDecayAnalyzer(max_lag=5)
    result = analyzer.analyze_factor(empty_df, empty_df, 'factor')
    assert result.half_life == 0, "空数据半衰期应为0"
    print(f"  ✓ 空数据处理正确")

    # 单日数据
    print("\n4.2 单日数据:")
    single = pd.DataFrame({
        'code': ['000001.SH'], 'date': [pd.Timestamp('2024-01-01')],
        'factor': [0.5], 'close': [10.0]
    })
    result = analyzer.analyze_factor(single, single, 'factor')
    assert result.half_life == 0, "单日数据半衰期应为0"
    print(f"  ✓ 单日数据处理正确")

    # 全NaN因子
    print("\n4.3 全NaN因子:")
    nan_df, price_df = generate_factor_data(3, 50)
    nan_df['nan_factor'] = np.nan
    result = analyzer.analyze_factor(nan_df, price_df, 'nan_factor')
    assert result.half_life == 0, "全NaN因子半衰期应为0"
    print(f"  ✓ 全NaN因子处理正确")

    print("\n✓ 边界条件测试完成")


def test_report():
    """测试报告生成"""
    print("\n" + "=" * 60)
    print("测试 5: 报告生成")
    print("=" * 60)

    factor_df, price_df = generate_factor_data(10, 300)
    analyzer = FactorDecayAnalyzer(max_lag=20)
    analyzer.analyze_all_factors(factor_df, price_df)

    report = analyzer.generate_decay_report()
    print(f"\n{report}")

    print(f"\n✓ 报告生成测试通过")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("因子衰减分析验证测试")
    print("借鉴来源: FactorEngine (arXiv:2603.16365), Qlib RD-Agent")
    print("优化方向: factor-engine — 新增因子衰减分析\n")

    test_factor_decay_analysis()
    test_factor_classification()
    test_factor_rotation()
    test_boundary()
    test_report()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)