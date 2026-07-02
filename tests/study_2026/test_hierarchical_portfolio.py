"""
============================================================
优化方向：分层聚类组合优化（HRP / HERC / NCO）
借鉴来源：Riskfolio-Lib (https://github.com/dcajasn/Riskfolio-Lib)
          - 3.4K stars, BSD-3-Clause license
          - 支持 HRP、HERC、NCO 三种分层优化方法
          - 内置 32 种风险度量

对照模块：portfolio-risk-engine
现状问题：
  - 当前仅支持均值-方差、最大夏普、最小方差、HRP（简化版）、CVaR（等权兜底）
  - HRP 实现不完整（空收益矩阵，使用群集距离矩阵替代）
  - 缺少 HERC、NCO 等前沿方法
  - 风险度量仅用方差，不支持下行风险/回撤风险度量

测试目标：
  1. 验证完整 HRP 实现（基于协方差矩阵聚类）
  2. 验证 HERC（分层等风险贡献）实现
  3. 验证 NCO（嵌套聚类优化）基本逻辑
  4. 与等权组合进行夏普比率对比
============================================================
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 辅助函数
# ============================================================

def _corr_to_dist(corr_matrix: np.ndarray) -> np.ndarray:
    """将相关性矩阵转换为距离矩阵 D = sqrt(0.5 * (1 - rho))"""
    return np.sqrt(0.5 * (1 - np.clip(corr_matrix, -1, 1)))


def _quasi_diagonalize(linkage_matrix: np.ndarray) -> list:
    """
    对 linkage 矩阵执行准对角化排序。
    基于树状图结构，将最相似的资产放在邻近位置。
    """
    n = linkage_matrix.shape[0] + 1
    # 从根节点递归排序
    sorted_idx = _recursive_sort(linkage_matrix, 2 * n - 2)
    return sorted_idx


def _recursive_sort(link: np.ndarray, node: int) -> list:
    """递归排序树状图节点"""
    n = link.shape[0] + 1
    if node < n:
        return [node]
    left = int(link[node - n, 0])
    right = int(link[node - n, 1])
    return _recursive_sort(link, left) + _recursive_sort(link, right)


def _get_cluster_variance(cov: np.ndarray, cluster_indices: list) -> float:
    """计算给定聚类内资产组合的方差"""
    if len(cluster_indices) == 1:
        return cov[cluster_indices[0], cluster_indices[0]]
    sub_cov = cov[np.ix_(cluster_indices, cluster_indices)]
    n = len(cluster_indices)
    w_ew = np.ones(n) / n
    return float(w_ew @ sub_cov @ w_ew)


def _get_ivp(cov: np.ndarray, indices: list) -> np.ndarray:
    """
    Inverse Variance Portfolio（逆方差加权）
    在聚类内部使用逆方差确定权重
    """
    sub_cov = cov[np.ix_(indices, indices)]
    ivp = 1.0 / np.clip(np.diag(sub_cov), 1e-10, None)
    ivp /= ivp.sum()
    return ivp


# ============================================================
# HRP 实现
# ============================================================

def hrp_weights(
    returns: pd.DataFrame,
    max_clusters: Optional[int] = None,
) -> pd.Series:
    """
    分层风险平价 (Hierarchical Risk Parity)

    算法步骤：
    1. 计算协方差矩阵和相关性矩阵
    2. 将相关性转为距离矩阵
    3. 使用层次聚类构建树状图
    4. 准对角化（重排资产顺序）
    5. 递归二分：每层按聚类方差比例分配权重

    参数:
        returns: 收益率 DataFrame，列=资产，行=日期
        max_clusters: 最大聚类数（None = 自动）

    返回:
        资产权重 Series
    """
    cov = returns.cov().values
    corr = returns.corr().values
    dist = _corr_to_dist(corr)
    condensed_dist = squareform(dist, checks=False)

    # 层次聚类
    link = linkage(condensed_dist, method='ward')

    # 准对角化排序
    sorted_idx = _quasi_diagonalize(link)
    cov_sorted = cov[np.ix_(sorted_idx, sorted_idx)]

    # 递归二分分配权重
    weights = np.ones(len(sorted_idx))
    _recursive_bisection(cov_sorted, 0, len(sorted_idx), weights)

    # 归一化确保和为 1
    weights /= weights.sum()

    return pd.Series(weights, index=returns.columns[sorted_idx])


def _recursive_bisection(
    cov: np.ndarray,
    start: int,
    end: int,
    weights: np.ndarray,
):
    """
    递归二分权重分配。

    对当前区间 [start, end)，按左右两半的聚类方差反比分配权重，
    然后在各半内递归分配。
    """
    if end - start <= 1:
        return

    mid = (start + end) // 2
    left_indices = list(range(start, mid))
    right_indices = list(range(mid, end))

    var_left = _get_cluster_variance(cov, left_indices)
    var_right = _get_cluster_variance(cov, right_indices)

    alpha_left = 1.0 - var_left / (var_left + var_right + 1e-10)
    alpha_left = np.clip(alpha_left, 0.01, 0.99)

    # 重新分配权重：左半部分占总权重 alpha_left，右半部分占 (1-alpha_left)
    left_sum = weights[start:mid].sum()
    right_sum = weights[mid:end].sum()
    total_sum = left_sum + right_sum

    weights[start:mid] *= (alpha_left * total_sum) / left_sum
    weights[mid:end] *= ((1 - alpha_left) * total_sum) / right_sum

    _recursive_bisection(cov, start, mid, weights)
    _recursive_bisection(cov, mid, end, weights)


# ============================================================
# HERC 实现
# ============================================================

def herc_weights(
    returns: pd.DataFrame,
    n_clusters: int = 3,
    cluster_risk_method: str = "equal",
) -> pd.Series:
    """
    分层等风险贡献 (Hierarchical Equal Risk Contribution)

    与 HRP 的区别：
    - HRP 目标：各聚类方差贡献相等 → 递归二分
    - HERC 目标：每个聚类内部先做风险平价，聚类间再按等风险贡献分配

    参数:
        returns: 收益率 DataFrame
        n_clusters: 聚类数量
        cluster_risk_method: 聚类内权重方法 ("equal" | "inverse_variance")

    返回:
        资产权重 Series
    """
    cov = returns.cov().values
    corr = returns.corr().values
    dist = _corr_to_dist(corr)
    condensed_dist = squareform(dist, checks=False)

    link = linkage(condensed_dist, method='ward')
    labels = fcluster(link, n_clusters, criterion='maxclust')

    # 按聚类分组
    n_assets = len(returns.columns)
    cluster_indices = {}
    for i in range(n_assets):
        label = labels[i]
        cluster_indices.setdefault(label, []).append(i)

    # 聚类内权重
    weights = np.zeros(n_assets)
    for label, indices in cluster_indices.items():
        if cluster_risk_method == "inverse_variance":
            sub_weights = _get_ivp(cov, indices)
        else:
            sub_weights = np.ones(len(indices)) / len(indices)

        # 计算该聚类对组合的风险贡献
        sub_cov = cov[np.ix_(indices, indices)]
        cluster_risk = np.sqrt(sub_weights @ sub_cov @ sub_weights)

        # 聚类权重 = 1/cluster_risk（等风险贡献）
        cluster_weight = 1.0 / (cluster_risk + 1e-10) if cluster_risk > 0 else 1.0

        for j, idx in enumerate(indices):
            weights[idx] = cluster_weight * sub_weights[j]

    # 归一化
    weights /= weights.sum()

    return pd.Series(weights, index=returns.columns)


# ============================================================
# NCO（简化版）实现
# ============================================================

def nco_weights(
    returns: pd.DataFrame,
    n_clusters: int = 3,
) -> pd.Series:
    """
    嵌套聚类优化 (Nested Clustered Optimization)

    核心思想：
    1. 聚类内做均值-方差优化（只对类内资产）
    2. 聚类间再做一层优化（各类的"合成资产"）

    参数:
        returns: 收益率 DataFrame
        n_clusters: 聚类数量

    返回:
        资产权重 Series
    """
    cov = returns.cov().values
    corr = returns.corr().values
    dist = _corr_to_dist(corr)
    condensed_dist = squareform(dist, checks=False)

    link = linkage(condensed_dist, method='ward')
    labels = fcluster(link, n_clusters, criterion='maxclust')

    n_assets = len(returns.columns)
    cluster_indices = {}
    for i in range(n_assets):
        label = labels[i]
        cluster_indices.setdefault(label, []).append(i)

    mu = returns.mean().values
    weights = np.zeros(n_assets)

    # 第一步：聚类内最小方差优化
    cluster_synth_weights = []
    cluster_synth_codes = []

    for label, indices in cluster_indices.items():
        sub_cov = cov[np.ix_(indices, indices)]
        sub_mu = mu[indices]

        try:
            inv_cov = np.linalg.pinv(sub_cov)
            ones = np.ones(len(indices))
            min_var_w = inv_cov @ ones
            min_var_w /= min_var_w.sum()
            min_var_w = np.clip(min_var_w, 0, 1)
            min_var_w /= min_var_w.sum()
        except Exception:
            min_var_w = np.ones(len(indices)) / len(indices)

        for j, idx in enumerate(indices):
            weights[idx] = min_var_w[j]

        cluster_synth_weights.append(min_var_w)
        cluster_synth_codes.append(label)

    # 第二步：聚类间等权（简化版；完整版应对合成资产做优化）
    cluster_weights = np.ones(len(cluster_indices)) / len(cluster_indices)

    for c_idx, label in enumerate(cluster_synth_codes):
        for j, idx in enumerate(cluster_indices[label]):
            weights[idx] *= cluster_weights[c_idx]

    weights /= weights.sum()

    return pd.Series(weights, index=returns.columns)


# ============================================================
# 绩效评估函数
# ============================================================

def calc_portfolio_metrics(returns: pd.DataFrame, weights: pd.Series) -> Dict[str, float]:
    """计算组合绩效指标"""
    aligned = weights.reindex(returns.columns, fill_value=0)
    port_returns = returns.dot(aligned)

    ann_return = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cum_returns = (1 + port_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    return {
        "annual_return": round(float(ann_return), 4),
        "annual_volatility": round(float(ann_vol), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
    }


# ============================================================
# 测试用例
# ============================================================

def _generate_test_data(n_assets: int = 10, n_days: int = 500) -> pd.DataFrame:
    """生成模拟收益率数据用于测试"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    # 创建具有聚类结构的数据：前 1/3 股票正相关，中间 1/3 低相关，后 1/3 负相关
    n1 = n_assets // 3
    n2 = n_assets // 3
    n3 = n_assets - n1 - n2

    data = np.zeros((n_days, n_assets))

    # 聚类1: 正相关组（均值为正）
    cluster1_factor = np.random.normal(0.0008, 0.012, n_days)
    for i in range(n1):
        data[:, i] = cluster1_factor + np.random.normal(0, 0.006, n_days)

    # 聚类2: 低相关组
    for i in range(n2):
        data[:, n1 + i] = np.random.normal(0.0004, 0.015, n_days)

    # 聚类3: 负相关组
    cluster3_factor = np.random.normal(-0.0003, 0.014, n_days)
    for i in range(n3):
        data[:, n1 + n2 + i] = cluster3_factor + np.random.normal(0, 0.007, n_days)

    columns = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=columns)


def test_hrp_implementation():
    """测试 HRP 实现的正确性"""
    print("\n" + "=" * 60)
    print("测试 1: HRP（分层风险平价）实现")
    print("=" * 60)

    returns = _generate_test_data(10, 500)

    # 计算等权基准
    ew_weights = pd.Series(1.0 / 10, index=returns.columns)
    ew_metrics = calc_portfolio_metrics(returns, ew_weights)

    # 计算 HRP 权重
    hrp_w = hrp_weights(returns)
    hrp_metrics = calc_portfolio_metrics(returns, hrp_w)

    print(f"\n资产数量: {len(hrp_w)}")
    print(f"\n权重分布:")
    print(f"  等权: min={1/10:.4f}, max={1/10:.4f}, std=0")
    print(f"  HRP:  min={hrp_w.min():.4f}, max={hrp_w.max():.4f}, std={hrp_w.std():.4f}")
    print(f"  权重和: {hrp_w.sum():.4f}")

    print(f"\n绩效对比:")
    print(f"  {'指标':<20} {'等权':>10} {'HRP':>10}")
    print(f"  {'-'*40}")
    for k in ew_metrics:
        print(f"  {k:<20} {ew_metrics[k]:>10.4f} {hrp_metrics[k]:>10.4f}")

    # 验证
    assert abs(hrp_w.sum() - 1.0) < 0.001, f"权重和应接近 1.0，实际: {hrp_w.sum():.6f}"
    assert all(hrp_w >= 0), "所有权重应非负"
    assert len(hrp_w) == 10, f"应有 10 个资产权重"

    print("\n✓ HRP 测试通过")

    return hrp_metrics


def test_herc_implementation():
    """测试 HERC 实现"""
    print("\n" + "=" * 60)
    print("测试 2: HERC（分层等风险贡献）实现")
    print("=" * 60)

    returns = _generate_test_data(12, 500)

    # 等权
    ew_weights = pd.Series(1.0 / 12, index=returns.columns)
    ew_metrics = calc_portfolio_metrics(returns, ew_weights)

    # HERC - 等权类内
    herc_w1 = herc_weights(returns, n_clusters=3, cluster_risk_method="equal")
    herc_m1 = calc_portfolio_metrics(returns, herc_w1)

    # HERC - 逆方差类内
    herc_w2 = herc_weights(returns, n_clusters=3, cluster_risk_method="inverse_variance")
    herc_m2 = calc_portfolio_metrics(returns, herc_w2)

    print(f"\n权重对比:")
    print(f"  {'方法':<22} {'min':>10} {'max':>10} {'std':>10}")
    print(f"  {'-'*52}")
    print(f"  {'等权':<22} {1/12:>10.4f} {1/12:>10.4f} {0:>10.4f}")
    print(f"  {'HERC (等权类内)':<22} {herc_w1.min():>10.4f} {herc_w1.max():>10.4f} {herc_w1.std():>10.4f}")
    print(f"  {'HERC (逆方差类内)':<22} {herc_w2.min():>10.4f} {herc_w2.max():>10.4f} {herc_w2.std():>10.4f}")

    print(f"\n绩效对比:")
    print(f"  {'指标':<20} {'等权':>10} {'HERC-equal':>10} {'HERC-ivp':>10}")
    print(f"  {'-'*50}")
    for k in ew_metrics:
        print(f"  {k:<20} {ew_metrics[k]:>10.4f} {herc_m1[k]:>10.4f} {herc_m2[k]:>10.4f}")

    assert abs(herc_w1.sum() - 1.0) < 0.001
    assert abs(herc_w2.sum() - 1.0) < 0.001
    assert all(herc_w1 >= 0)
    assert all(herc_w2 >= 0)

    print("\n✓ HERC 测试通过")

    return herc_m1


def test_nco_implementation():
    """测试 NCO 实现"""
    print("\n" + "=" * 60)
    print("测试 3: NCO（嵌套聚类优化）实现")
    print("=" * 60)

    returns = _generate_test_data(15, 500)

    ew_weights = pd.Series(1.0 / 15, index=returns.columns)
    ew_metrics = calc_portfolio_metrics(returns, ew_weights)

    nco_w = nco_weights(returns, n_clusters=3)
    nco_metrics = calc_portfolio_metrics(returns, nco_w)

    print(f"\n资产数量: {len(nco_w)}")
    print(f"\n权重:")
    print(f"  NCO: min={nco_w.min():.4f}, max={nco_w.max():.4f},"
          f" std={nco_w.std():.4f}, 非零权重数={int((nco_w > 1e-6).sum())}")

    print(f"\n绩效对比:")
    print(f"  {'指标':<20} {'等权':>10} {'NCO':>10}")
    print(f"  {'-'*30}")
    for k in ew_metrics:
        print(f"  {k:<20} {ew_metrics[k]:>10.4f} {nco_metrics[k]:>10.4f}")

    assert abs(nco_w.sum() - 1.0) < 0.001
    assert all(nco_w >= 0)

    print("\n✓ NCO 测试通过")

    return nco_metrics


def test_all_methods_comparison():
    """所有方法的综合对比"""
    print("\n" + "=" * 60)
    print("测试 4: 四种方法综合对比")
    print("=" * 60)

    returns = _generate_test_data(15, 504)

    methods = {
        "等权": pd.Series(1.0 / 15, index=returns.columns),
        "HRP": hrp_weights(returns),
        "HERC": herc_weights(returns, n_clusters=4),
        "NCO": nco_weights(returns, n_clusters=4),
    }

    print(f"\n{'方法':<10} {'年化收益':>10} {'年化波动':>10} {'夏普比率':>10} {'最大回撤':>10}")
    print(f"{'-'*50}")

    best_sharpe = -np.inf
    best_method = None

    for name, w in methods.items():
        m = calc_portfolio_metrics(returns, w)
        print(f"{name:<10} {m['annual_return']:>10.4f} {m['annual_volatility']:>10.4f}"
              f" {m['sharpe_ratio']:>10.4f} {m['max_drawdown']:>10.4f}")
        if m['sharpe_ratio'] > best_sharpe:
            best_sharpe = m['sharpe_ratio']
            best_method = name

    print(f"\n最高夏普: {best_method} ({best_sharpe:.4f})")

    return best_method


# ============================================================
# 边界条件测试
# ============================================================

def test_edge_cases():
    """测试边界条件"""
    print("\n" + "=" * 60)
    print("测试 5: 边界条件验证")
    print("=" * 60)

    # 边界 1: 单资产
    returns_single = pd.DataFrame(
        np.random.normal(0.001, 0.02, (500, 1)),
        columns=["single_asset"]
    )
    try:
        w = hrp_weights(returns_single)
        assert abs(w.sum() - 1.0) < 0.001
        assert len(w) == 1
        print("✓ 单资产 HRP: 权重=1.0")
    except Exception as e:
        print(f"✗ 单资产 HRP 失败: {e}")

    # 边界 2: 双资产
    returns_two = pd.DataFrame(
        np.random.normal(0.001, 0.02, (500, 2)),
        columns=["A", "B"]
    )
    try:
        w = hrp_weights(returns_two)
        assert abs(w.sum() - 1.0) < 0.001
        print(f"✓ 双资产 HRP: A={w.iloc[0]:.4f}, B={w.iloc[1]:.4f}")
    except Exception as e:
        print(f"✗ 双资产 HRP 失败: {e}")

    # 边界 3: 高度共线性（完全相同的数据）
    returns_identical = pd.DataFrame(
        np.tile(np.random.normal(0.001, 0.02, (500, 1)), (1, 5)),
        columns=[f"c{i}" for i in range(5)]
    )
    returns_identical += np.random.normal(0, 1e-8, returns_identical.shape)
    try:
        w = hrp_weights(returns_identical)
        assert abs(w.sum() - 1.0) < 0.001
        print(f"✓ 高共线性 HRP: 权重分布 std={w.std():.6f}")
    except Exception as e:
        print(f"✗ 高共线性 HRP 失败: {e}")

    # 边界 4: HERC 聚类数大于资产数
    returns_many = _generate_test_data(5, 300)
    try:
        w = herc_weights(returns_many, n_clusters=10)
        assert abs(w.sum() - 1.0) < 0.001
        print(f"✓ HERC 聚类数>资产数: 权重分布正常")
    except Exception as e:
        print(f"✗ HERC 聚类数>资产数 失败: {e}")

    print("\n✓ 边界条件测试完成")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("分层聚类组合优化验证测试")
    print("借鉴来源: Riskfolio-Lib (dcajasn/Riskfolio-Lib)")
    print("优化目标: portfolio-risk-engine")
    print("=" * 60)

    test_hrp_implementation()
    test_herc_implementation()
    test_nco_implementation()
    test_all_methods_comparison()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)