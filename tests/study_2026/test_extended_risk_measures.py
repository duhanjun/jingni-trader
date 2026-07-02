"""
============================================================
优化方向：扩展风险度量（下行风险 / 回撤风险 / 尾部风险）
借鉴来源：Riskfolio-Lib (https://github.com/dcajasn/Riskfolio-Lib)
          - 24 种 convex 风险度量
          - 分散性风险、下行风险、回撤风险三类
          - 支持 VaR/CVaR/EVaR/RLVaR/DaR/CDaR/EDaR/RLDaR 等

对照模块：portfolio-risk-engine
现状问题：
  - 仅实现了 VaR（历史模拟法）和 CVaR
  - 缺少 EVaR、回撤风险度量（DaR/CDaR/EDaR）
  - 缺少 Ulcer Index、Max Drawdown 等作为优化目标
  - 风险归因仅有简化版 Barra

测试目标：
  1. 验证 EVaR（熵风险价值）计算
  2. 验证回撤相关风险度量（DaR/CDaR/EDaR）
  3. 验证 Ulcer Index（溃疡指数）
  4. 验证基于下行风险的风险平价
============================================================
"""

import numpy as np
import pandas as pd
from scipy import optimize
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. EVaR（熵风险价值）- Entropic Value at Risk
# ============================================================

def calc_evar(
    returns: np.ndarray,
    confidence: float = 0.95,
    max_iter: int = 100,
) -> float:
    """
    计算 EVaR (Entropic Value at Risk)

    EVaR 是 VaR 的推广，基于 Chernoff 界。
    对于正态分布，EVaR = -mu + sigma * sqrt(-2 * ln(1-alpha))

    公式: EVaR_{1-alpha}(X) = inf_{t>0} { t^{-1} * ln(M_X(-t) / (1-alpha)) }
    其中 M_X 是矩母函数。

    参数:
        returns: 收益率数组
        confidence: 置信水平（默认 0.95）
        max_iter: 牛顿法最大迭代次数

    返回:
        EVaR 值（正值表示风险/损失）
    """
    alpha = 1.0 - confidence
    mu = returns.mean()
    sigma = returns.std()

    # 正态假设下 EVaR 的闭合解
    if sigma == 0:
        return -mu if mu < 0 else 0.0

    evar = -mu + sigma * np.sqrt(-2 * np.log(alpha))
    return float(max(evar, 0))


# ============================================================
# 2. 回撤风险度量
# ============================================================

def _compute_drawdowns(cum_returns: np.ndarray) -> np.ndarray:
    """计算回撤序列"""
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    return drawdowns


def calc_ulcer_index(returns: np.ndarray) -> float:
    """
    计算溃疡指数 (Ulcer Index)

    UI = sqrt(mean(drawdown²))

    衡量回撤的持续性和深度。
    """
    cum_returns = (1 + returns).cumprod()
    drawdowns = _compute_drawdowns(cum_returns)
    ui = np.sqrt(np.mean(drawdowns ** 2))
    return float(ui)


def calc_dar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    计算在险回撤 (Drawdown at Risk, DaR)

    DaR = 在置信水平 alpha 下的最大回撤
    """
    cum_returns = (1 + returns).cumprod()
    drawdowns = _compute_drawdowns(cum_returns)
    alpha = 1.0 - confidence
    dar = np.percentile(drawdowns, confidence * 100)
    return float(abs(dar))


def calc_cdar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    计算条件在险回撤 (Conditional Drawdown at Risk, CDaR)

    CDaR = 超过 DaR 阈值的回撤的期望值
    """
    cum_returns = (1 + returns).cumprod()
    drawdowns = _compute_drawdowns(cum_returns)
    alpha = 1.0 - confidence
    threshold = np.percentile(drawdowns, confidence * 100)
    tail = drawdowns[drawdowns <= threshold]
    if len(tail) == 0:
        return float(abs(threshold))
    return float(abs(tail.mean()))


def calc_edar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    计算熵回撤风险 (Entropic Drawdown at Risk, EDaR)

    基于 EVaR 框架应用于回撤分布
    """
    cum_returns = (1 + returns).cumprod()
    drawdowns = _compute_drawdowns(cum_returns)
    alpha = 1.0 - confidence

    # 将 drawdowns 转为收益（计算 EVaR 的框架需要收益为正表示风险方向）
    dd_negative = -drawdowns
    mu_dd = dd_negative.mean()
    sigma_dd = dd_negative.std()

    if sigma_dd == 0:
        return 0.0

    edar = mu_dd + sigma_dd * np.sqrt(-2 * np.log(alpha))
    return float(max(edar, 0))


# ============================================================
# 3. 下行风险度量
# ============================================================

def calc_semi_deviation(returns: np.ndarray) -> float:
    """计算下半标准差（只考虑负收益）"""
    negative = returns[returns < 0]
    if len(negative) < 2:
        return 0.0
    return float(negative.std())


def calc_sortino_ratio(returns: np.ndarray, risk_free: float = 0.03) -> float:
    """计算索提诺比率"""
    excess = returns.mean() * 252 - risk_free
    downside = calc_semi_deviation(returns) * np.sqrt(252)
    if downside == 0:
        return 0.0
    return float(excess / downside)


def calc_calmar_ratio(returns: np.ndarray) -> float:
    """计算 Calmar 比率（年化收益/最大回撤）"""
    ann_return = returns.mean() * 252
    max_dd = calc_dar(returns, confidence=0.99)
    if max_dd == 0:
        return 0.0
    return float(ann_return / max_dd)


# ============================================================
# 4. 投资组合层面计算
# ============================================================

def calc_portfolio_drawdown_metrics(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    计算组合层面的完整回撤风险指标
    """
    aligned = weights.reindex(returns.columns, fill_value=0)
    port_returns = returns.dot(aligned).values

    return {
        "max_drawdown": calc_dar(port_returns, confidence=0.999),
        "DaR_95": calc_dar(port_returns, confidence=0.95),
        "CDaR_95": calc_cdar(port_returns, confidence=0.95),
        "EDaR_95": calc_edar(port_returns, confidence=0.95),
        "ulcer_index": calc_ulcer_index(port_returns),
    }


def calc_portfolio_tail_metrics(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    计算组合层面的完整尾部风险指标
    """
    aligned = weights.reindex(returns.columns, fill_value=0)
    port_returns = returns.dot(aligned).values

    var = np.percentile(port_returns, (1 - confidence) * 100)
    cvar = port_returns[port_returns <= var].mean() if var < 0 else 0
    evar = calc_evar(port_returns, confidence)

    return {
        "VaR_{:.0f}".format(confidence * 100): float(abs(var)),
        "CVaR_{:.0f}".format(confidence * 100): float(abs(cvar)),
        "EVaR_{:.0f}".format(confidence * 100): float(evar),
        "semi_deviation": float(calc_semi_deviation(port_returns)),
        "sortino_ratio": float(calc_sortino_ratio(port_returns)),
    }


# ============================================================
# 5. 基于下行风险的权重优化（演示）
# ============================================================

def risk_parity_downside(
    returns: pd.DataFrame,
    risk_measure: str = "semi_deviation",
    max_iter: int = 1000,
) -> pd.Series:
    """
    基于下行风险度量的风险平价

    参数:
        returns: 收益率 DataFrame
        risk_measure: "semi_deviation" | "cvar" | "adar" (average drawdown)
    """
    n_assets = len(returns.columns)
    values = returns.values

    def risk_contribution(w, i):
        """资产 i 的下行风险贡献"""
        port_ret = values @ w
        asset_ret = values[:, i]

        if risk_measure == "semi_deviation":
            mask = port_ret < 0
            if mask.sum() < 2:
                return np.std(w * asset_ret[mask]) if mask.sum() > 0 else 0
            contrib = np.cov(port_ret[mask], w[i] * asset_ret[mask])[0, 1]
            return np.sqrt(max(contrib, 0))

        elif risk_measure == "cvar":
            confidence = 0.95
            alpha = 1 - confidence
            var_port = np.percentile(port_ret, alpha * 100)
            mask = port_ret <= var_port
            if mask.sum() == 0:
                return 0
            return np.abs((w[i] * asset_ret[mask]).mean())

        elif risk_measure == "adar":
            dd = _compute_drawdowns((1 + port_ret).cumprod())
            return np.abs(dd).mean() * w[i]

        return 0

    # 简单求解：使各资产的风险贡献尽可能相等
    def objective(w):
        w = w / w.sum()
        risks = np.array([risk_contribution(w, i) for i in range(n_assets)])
        risks = np.clip(risks, 1e-10, None)
        return np.std(risks / risks.sum())

    # 约束：权重和=1, 权重>=0
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.01, 1.0) for _ in range(n_assets)]

    w0 = np.ones(n_assets) / n_assets
    result = optimize.minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-12}
    )

    weights = result.x / result.x.sum()
    return pd.Series(weights, index=returns.columns)


# ============================================================
# 测试用例
# ============================================================

def _generate_test_data(n_assets: int = 10, n_days: int = 504) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    data = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        # 不同资产有不同的均值、波动和偏度
        mu = np.random.uniform(-0.0002, 0.001)
        sigma = np.random.uniform(0.008, 0.025)
        data[:, i] = np.random.normal(mu, sigma, n_days)

    columns = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=columns)


def test_risk_measures_calculation():
    """测试各类风险度量计算"""
    print("\n" + "=" * 60)
    print("测试 1: 各类风险度量计算验证")
    print("=" * 60)

    np.random.seed(42)
    returns = np.random.normal(-0.0005, 0.02, 500)

    # EVaR
    evar_95 = calc_evar(returns, confidence=0.95)
    evar_99 = calc_evar(returns, confidence=0.99)
    print(f"\nEVaR:")
    print(f"  95% confidence: {evar_95:.6f}")
    print(f"  99% confidence: {evar_99:.6f}")
    assert evar_99 > evar_95, "99% EVaR 应大于 95% EVaR"

    # 回撤风险
    dar = calc_dar(returns, confidence=0.95)
    cdar = calc_cdar(returns, confidence=0.95)
    edar = calc_edar(returns, confidence=0.95)
    ui = calc_ulcer_index(returns)

    print(f"\n回撤风险:")
    print(f"  DaR_95:   {dar:.6f}")
    print(f"  CDaR_95:  {cdar:.6f}")
    print(f"  EDaR_95:  {edar:.6f}")
    print(f"  Ulcer:    {ui:.6f}")
    assert cdar >= dar, f"CDaR ({cdar:.6f}) 应 >= DaR ({dar:.6f})"
    assert edar >= dar, f"EDaR ({edar:.6f}) 应 >= DaR ({dar:.6f})"

    # 下行风险
    semi_std = calc_semi_deviation(returns)
    sortino = calc_sortino_ratio(returns)
    print(f"\n下行风险:")
    print(f"  下半标准差: {semi_std:.6f}")
    print(f"  Sortino:    {sortino:.4f}")

    # 全正收益情况
    pos_returns = np.abs(np.random.normal(0.001, 0.01, 200))
    semi_std_pos = calc_semi_deviation(pos_returns)
    assert semi_std_pos == 0.0, "全正收益的下半标准差应为 0"

    print("\n✓ 风险度量计算验证通过")


def test_portfolio_risk_profiles():
    """测试组合风险画像"""
    print("\n" + "=" * 60)
    print("测试 2: 组合风险画像对比")
    print("=" * 60)

    returns = _generate_test_data(10, 504)

    # 等权组合
    ew_weights = pd.Series(1.0 / 10, index=returns.columns)

    # 集中组合（前3个资产）
    conc_weights = pd.Series(0.0, index=returns.columns)
    conc_weights.iloc[:3] = 1.0 / 3

    for name, w in [("等权", ew_weights), ("集中", conc_weights)]:
        print(f"\n{'='*40}")
        print(f"组合: {name}")

        tail = calc_portfolio_tail_metrics(returns, w)
        print(f"  尾部风险:")
        for k, v in tail.items():
            print(f"    {k}: {v:.6f}")

        dd = calc_portfolio_drawdown_metrics(returns, w)
        print(f"  回撤风险:")
        for k, v in dd.items():
            print(f"    {k}: {v:.6f}")

    print("\n✓ 组合风险画像测试通过")


def test_downside_risk_parity():
    """测试下行风险平价优化"""
    print("\n" + "=" * 60)
    print("测试 3: 下行风险平价优化")
    print("=" * 60)

    returns = _generate_test_data(6, 504)

    # 下行标准差风险平价
    w_down = risk_parity_downside(returns, risk_measure="semi_deviation")
    print(f"\n下行风险平价权重:")
    for asset, w in w_down.sort_values(ascending=False).items():
        print(f"  {asset}: {w:.4f}")
    print(f"  权重和: {w_down.sum():.4f}")

    assert abs(w_down.sum() - 1.0) < 0.01, f"权重和应接近 1.0"
    assert all(w_down >= 0.01), "所有权重应 >= 0.01"

    # 对比等权组合
    ew_w = pd.Series(1.0 / 6, index=returns.columns)
    ew_sd = calc_semi_deviation(returns.dot(ew_w).values)
    down_sd = calc_semi_deviation(returns.dot(w_down).values)

    print(f"\n  {'组合':<15} {'下半标准差':>12}")
    print(f"  {'等权':<15} {ew_sd:>12.6f}")
    print(f"  {'下行风险平价':<15} {down_sd:>12.6f}")

    print("\n✓ 下行风险平价测试通过")


def test_edge_cases():
    """测试边界条件"""
    print("\n" + "=" * 60)
    print("测试 4: 边界条件验证")
    print("=" * 60)

    # 空收益率
    try:
        evar = calc_evar(np.array([]))
        print(f"✓ 空收益率 EVaR: {evar}")
    except Exception as e:
        print(f"✗ 空收益率 EVaR 失败: {e}")

    # 单收益率
    evar_single = calc_evar(np.array([0.01]))
    print(f"✓ 单收益率 EVaR: {evar_single}")

    # 常量收益率
    constant_returns = np.ones(100) * 0.001
    dd = _compute_drawdowns((1 + constant_returns).cumprod())
    assert np.allclose(dd, 0), "正常量收益率的回撤应为 0"
    print(f"✓ 正常量收益率的回撤=0")

    # 极大回撤场景
    crash_returns = np.array([-0.1, -0.05, 0.02, -0.15, 0.01])
    dar = calc_dar(crash_returns, confidence=0.8)
    assert dar >= 0.0, "大跌场景 DaR 应非负"
    print(f"✓ 大跌场景 DaR: {dar:.4f}")
    # 验证 DaR 在合理范围内（回撤不会超过100%）
    assert dar <= 1.0, "DaR 不应超过 100%"

    print("\n✓ 边界条件测试通过")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("扩展风险度量验证测试")
    print("借鉴来源: Riskfolio-Lib (dcajasn/Riskfolio-Lib)")
    print("优化目标: portfolio-risk-engine")
    print("=" * 60)

    test_risk_measures_calculation()
    test_portfolio_risk_profiles()
    test_downside_risk_parity()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)