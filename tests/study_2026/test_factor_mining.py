"""
测试文件：自动化因子挖掘验证
优化方向：引入遗传编程(GP)基础的因子自动发现机制
借鉴来源：AlphaGen (RL-MLDM/alphagen, KDD 2023)
           https://github.com/RL-MLDM/alphagen
           tsfresh (时间序列特征自动提取)
           https://github.com/blue-yonder/tsfresh

AlphaGen 核心思路：
  - 使用强化学习(RL)自动生成公式化 Alpha 因子
  - 通过表达式树表示因子公式
  - 以 IC (Information Coefficient) 为目标函数
  - 同时考虑因子之间的协同性(synergy)

本验证实现：
  - 轻量级遗传编程(GP)因子挖掘器
  - 表达式树随机生成与交叉变异
  - IC 导向的适应度函数
  - 不依赖 GPU/RL，可在 CPU 上运行
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import copy
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================================
# 表达式树 (与 AlphaGen 核心数据结构对齐)
# ============================================================================

# 操作符定义
UNARY_OPS = {
    'neg':   (lambda x: -x,           1),
    'abs':   (lambda x: np.abs(x),    1),
    'log':   (lambda x: np.log(np.maximum(x, 1e-8)), 1),
    'sign':  (lambda x: np.sign(x),   1),
    'inv':   (lambda x: 1.0 / (x + 1e-8), 1),
    'sqrt':  (lambda x: np.sqrt(np.maximum(x, 0)), 1),
    'square':(lambda x: x ** 2,       1),
}

BINARY_OPS = {
    'add':  (lambda x, y: x + y,       2),
    'sub':  (lambda x, y: x - y,       2),
    'mul':  (lambda x, y: x * y,       2),
    'div':  (lambda x, y: x / (y + 1e-8), 2),
    'max':  (lambda x, y: np.maximum(x, y), 2),
    'min':  (lambda x, y: np.minimum(x, y), 2),
}

# 时序操作符 (窗口函数)
TS_OPS = {
    'ts_mean':    (lambda x, w: pd.Series(x).rolling(w, min_periods=max(3, w//2)).mean().values, 1),
    'ts_std':     (lambda x, w: pd.Series(x).rolling(w, min_periods=max(3, w//2)).std().values, 1),
    'ts_max':     (lambda x, w: pd.Series(x).rolling(w, min_periods=max(3, w//2)).max().values, 1),
    'ts_min':     (lambda x, w: pd.Series(x).rolling(w, min_periods=max(3, w//2)).min().values, 1),
    'ts_delta':   (lambda x, w: pd.Series(x).diff(w).values, 1),
    'ts_roc':     (lambda x, w: pd.Series(x).pct_change(w).values, 1),
    'ts_ema':     (lambda x, w: pd.Series(x).ewm(span=w, adjust=False).mean().values, 1),
    'ts_rank':    (lambda x, w: pd.Series(x).rolling(w, min_periods=5).rank(pct=True).values, 1),
    'ts_corr_v':  (lambda x, y, w: _rolling_corr(x, y, w), 2),
    'ts_delay':   (lambda x, w: pd.Series(x).shift(w).fillna(0).values, 1),
}

WINDOW_SIZES = [5, 10, 20, 60]


def _rolling_corr(x, y, w):
    """两个序列的滚动相关性"""
    s1 = pd.Series(x)
    s2 = pd.Series(y)
    return s1.rolling(w, min_periods=5).corr(s2).fillna(0).values


@dataclass
class ExprNode:
    """表达式树节点"""
    op: str                          # 操作符名称
    children: List['ExprNode'] = field(default_factory=list)
    value: Optional[float] = None    # 常量值 (仅叶节点)
    field: Optional[str] = None      # 数据字段 (仅叶节点)
    window: Optional[int] = None     # 窗口大小 (时序操作符)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_string(self) -> str:
        """转为可读字符串"""
        if self.is_leaf():
            if self.field:
                return f"${self.field}"
            if self.value is not None:
                return f"{self.value:.4f}"
            return str(self.op)

        args = [c.to_string() for c in self.children]
        if self.window:
            args.append(str(self.window))
        return f"{self.op}({', '.join(args)})"

    def clone(self) -> 'ExprNode':
        return copy.deepcopy(self)

    def count_nodes(self) -> int:
        return 1 + sum(c.count_nodes() for c in self.children)


# ============================================================================
# 表达式树评估器
# ============================================================================

class ExpressionEvaluator:
    """表达式树评估器：将树结构转为 numpy 数组"""

    # 叶子节点可选字段 (与 data-engine 输出对齐)
    LEAF_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount',
                   'turnover_rate', 'change_pct']

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._field_cache: Dict[str, np.ndarray] = {}

    def _get_field(self, name: str) -> np.ndarray:
        if name not in self._field_cache:
            self._field_cache[name] = self.data[name].values.copy()
        return self._field_cache[name]

    def evaluate(self, node: ExprNode) -> np.ndarray:
        """递归评估表达式树"""
        if node.is_leaf():
            if node.field:
                return self._get_field(node.field)
            if node.value is not None:
                return np.full(len(self.data), node.value)
            raise ValueError(f"无效叶子节点: {node}")

        # 评估子节点
        child_vals = [self.evaluate(c) for c in node.children]

        # 时序操作符
        if node.op in TS_OPS:
            op_func, n_children = TS_OPS[node.op]
            if n_children == 1:
                return op_func(child_vals[0], node.window)
            elif n_children == 2:
                return op_func(child_vals[0], child_vals[1], node.window)

        # 一元操作符
        if node.op in UNARY_OPS:
            op_func, _ = UNARY_OPS[node.op]
            return op_func(child_vals[0])

        # 二元操作符
        if node.op in BINARY_OPS:
            op_func, _ = BINARY_OPS[node.op]
            return op_func(child_vals[0], child_vals[1])

        raise ValueError(f"未知操作符: {node.op}")


# ============================================================================
# 遗传编程因子挖掘器
# ============================================================================

class GPMiner:
    """
    轻量级遗传编程因子挖掘器。

    流程:
      1. 随机生成初始种群 (N棵表达式树)
      2. 计算每棵树的 IC (适应度)
      3. 选择、交叉、变异 → 下一代
      4. 重复 M 代
      5. 返回最优因子
    """

    def __init__(
        self,
        data: pd.DataFrame,
        future_returns: np.ndarray,       # 未来收益率 (目标变量)
        population_size: int = 50,
        generations: int = 10,
        max_depth: int = 4,
        tournament_size: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        random_state: int = 42,
    ):
        self.data = data
        self.future_returns = future_returns
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.evaluator = ExpressionEvaluator(data)
        random.seed(random_state)
        np.random.seed(random_state)

    def run(self) -> Dict:
        """
        运行 GP 因子挖掘。

        返回:
            {
                'best_factor': ExprNode,       # 最优因子表达式
                'best_ic': float,               # 最优 IC
                'all_results': List[Dict],      # 每代结果
                'factors_found': List[Dict],    # 所有不重复因子
            }
        """
        population = self._init_population()
        all_results = []
        best_overall = None
        best_ic_overall = -1

        for gen in range(self.generations):
            # 评估适应度
            fitness = []
            for node in population:
                ic = self._calc_fitness(node)
                fitness.append(ic)

            # 记录最优
            best_idx = np.argmax([abs(f) for f in fitness])
            best_node = population[best_idx]
            best_ic = fitness[best_idx]

            if abs(best_ic) > abs(best_ic_overall):
                best_ic_overall = best_ic
                best_overall = best_node.clone()

            all_results.append({
                'generation': gen,
                'best_ic': best_ic,
                'best_expr': best_node.to_string(),
                'avg_ic': np.mean([abs(f) for f in fitness]),
                'max_ic': best_ic,
            })

            # 生成下一代
            next_population = []

            # 精英保留 (前 10%)
            elite_count = max(1, self.population_size // 10)
            elite_indices = np.argsort([abs(f) for f in fitness])[-elite_count:]
            for idx in elite_indices:
                next_population.append(population[idx].clone())

            # 交叉和变异
            while len(next_population) < self.population_size:
                if random.random() < self.crossover_rate and len(population) >= 2:
                    # 锦标赛选择两个父代
                    p1 = self._tournament_select(population, fitness)
                    p2 = self._tournament_select(population, fitness)
                    child = self._crossover(p1, p2)
                else:
                    child = random.choice(population).clone()

                # 变异
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                # 控制深度
                if child.count_nodes() <= 20:
                    next_population.append(child)

            population = next_population

        # 收集所有不重复因子
        seen = set()
        factors_found = []
        for gen_result in all_results:
            expr = gen_result['best_expr']
            if expr not in seen:
                seen.add(expr)
                factors_found.append(gen_result)

        return {
            'best_factor': best_overall,
            'best_ic': best_ic_overall,
            'all_results': all_results,
            'factors_found': factors_found,
        }

    def _calc_fitness(self, node: ExprNode) -> float:
        """计算 IC (Spearman Rank Correlation) 作为适应度"""
        try:
            values = self.evaluator.evaluate(node)
            # 移除 NaN 和 Inf
            mask = ~(np.isnan(values) | np.isinf(values) | np.isnan(self.future_returns) | np.isinf(self.future_returns))
            if mask.sum() < 50:
                return -999.0  # 惩罚无效因子
            from scipy import stats
            ic, _ = stats.spearmanr(values[mask], self.future_returns[mask])
            return float(ic) if not np.isnan(ic) else -999.0
        except Exception:
            return -999.0

    def _tournament_select(self, population, fitness):
        """锦标赛选择"""
        candidates_idx = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(candidates_idx, key=lambda i: abs(fitness[i]))
        return population[best_idx].clone()

    def _crossover(self, p1: ExprNode, p2: ExprNode) -> ExprNode:
        """子树交叉"""
        if p1.is_leaf() or p2.is_leaf():
            return p1.clone()

        # 随机选择 p1 的非根节点子树位置
        subtrees1 = self._get_subtree_list(p1)
        if not subtrees1:
            return p1.clone()

        # 随机选择 p2 的一个子树
        subtrees2 = self._get_subtree_list(p2)
        if not subtrees2:
            return p1.clone()

        new_p1 = p1.clone()
        _, target_path = random.choice(subtrees1)
        source_subtree, _ = random.choice(subtrees2)

        # 替换
        self._replace_subtree_at_path(new_p1, target_path, source_subtree.clone())
        return new_p1

    def _mutate(self, node: ExprNode) -> ExprNode:
        """随机变异"""
        # 随机选择一个子树替换为一个新的随机子树
        subtrees = self._get_subtree_list(node)
        if not subtrees:
            return node

        _, target_path = random.choice(subtrees)
        new_subtree = self._random_node(depth=random.randint(0, 2))
        self._replace_subtree_at_path(node, target_path, new_subtree)
        return node

    def _get_subtree_list(self, node: ExprNode, path: tuple = ()) -> List[
        Tuple[ExprNode, tuple]
    ]:
        """获取所有子树的列表 [(子树, 路径), ...]"""
        result = []
        for i, child in enumerate(node.children):
            child_path = path + (i,)
            result.append((child, child_path))
            result.extend(self._get_subtree_list(child, child_path))
        return result

    def _replace_subtree_at_path(self, root: ExprNode, path: tuple,
                                  new_node: ExprNode):
        """按路径替换子树"""
        if len(path) == 0:
            root.__dict__.update(new_node.__dict__)
            return
        current = root
        for idx in path[:-1]:
            current = current.children[idx]
        current.children[path[-1]] = new_node

    def _init_population(self) -> List[ExprNode]:
        """初始化随机种群"""
        population = []
        for _ in range(self.population_size):
            depth = random.randint(2, self.max_depth)
            population.append(self._random_node(depth=depth))
        return population

    def _random_node(self, depth: int = 0) -> ExprNode:
        """随机生成一个表达式树节点"""
        if depth <= 0:
            # 叶子节点
            if random.random() < 0.8:
                field = random.choice(self.LEAF_FIELDS)  # type: ignore
                return ExprNode(op='field', field=field)
            else:
                return ExprNode(op='const', value=random.uniform(-1, 1))

        # 随机选择操作符类别
        op_category = random.choice(['unary', 'binary', 'ts'])

        if op_category == 'unary':
            op_name = random.choice(list(UNARY_OPS.keys()))
            child = self._random_node(depth - 1)
            return ExprNode(op=op_name, children=[child])

        elif op_category == 'binary':
            op_name = random.choice(list(BINARY_OPS.keys()))
            left = self._random_node(depth - 1)
            right = self._random_node(depth - 1)
            return ExprNode(op=op_name, children=[left, right])

        else:  # ts
            op_name = random.choice(list(TS_OPS.keys()))
            n_children = TS_OPS[op_name][1]
            w = random.choice(WINDOW_SIZES)
            children = [self._random_node(depth - 1) for _ in range(n_children)]
            return ExprNode(op=op_name, children=children, window=w)

    # 全局变量 (与 GPMiner 实例共享)
    LEAF_FIELDS = ExpressionEvaluator.LEAF_FIELDS


# ============================================================================
# 测试
# ============================================================================

def test_gp_miner_basic():
    """测试：GP 因子挖掘基本功能"""
    print("=" * 60)
    print("测试 1: 遗传编程因子挖掘基础功能")
    print("=" * 60)

    np.random.seed(42)
    random.seed(42)

    # 生成测试数据
    n_stocks, n_days = 10, 100
    data_rows = []
    returns_rows = []

    for stock_idx in range(n_stocks):
        base = np.random.uniform(10, 50)
        rets = np.random.normal(0.0002, 0.02, n_days)
        prices = base * np.cumprod(1 + rets)

        for day in range(n_days):
            data_rows.append({
                'open': prices[day] * np.random.uniform(0.99, 1.01),
                'high': prices[day] * np.random.uniform(1.01, 1.03),
                'low': prices[day] * np.random.uniform(0.97, 0.99),
                'close': prices[day],
                'volume': np.random.lognormal(13, 0.5),
                'amount': np.random.lognormal(18, 0.5),
                'turnover_rate': np.random.uniform(0.5, 5),
                'change_pct': rets[day] * 100,
            })
            # 未来5日收益作为目标
            future_ret = prices[min(day + 5, n_days - 1)] / prices[day] - 1 if day < n_days - 5 else 0
            returns_rows.append(future_ret)

    df = pd.DataFrame(data_rows)
    future_returns = np.array(returns_rows)

    # 运行 GP
    miner = GPMiner(
        data=df,
        future_returns=future_returns,
        population_size=30,
        generations=5,
        max_depth=3,
        tournament_size=3,
        mutation_rate=0.3,
        crossover_rate=0.7,
        random_state=42,
    )

    result = miner.run()

    print(f"\n挖掘结果:")
    print(f"  最优因子: {result['best_factor'].to_string()}")
    print(f"  最优 IC:  {result['best_ic']:.6f}")

    print(f"\n各代表现:")
    for r in result['all_results']:
        print(f"  世代 {r['generation']}: IC={r['best_ic']:.6f}, "
              f"Avg|IC|={r['avg_ic']:.6f}, 表达式={r['best_expr'][:60]}...")

    print(f"\n发现的不重复因子数: {len(result['factors_found'])}")

    # 验证
    assert result['best_factor'] is not None, "应该发现至少一个因子"
    assert abs(result['best_ic']) >= 0.001, f"最优因子 IC 应有一定显著度: {result['best_ic']}"
    print("\n✅ 测试通过：GP 因子挖掘基本功能正常")


def test_expression_tree_evaluation():
    """测试：表达式树评估功能"""
    print("\n" + "=" * 60)
    print("测试 2: 表达式树构建与评估")
    print("=" * 60)

    np.random.seed(1)
    n = 500
    df = pd.DataFrame({
        'open': np.random.uniform(9, 51, n),
        'high': np.random.uniform(10, 52, n),
        'low': np.random.uniform(8, 50, n),
        'close': np.random.uniform(10, 50, n),
        'volume': np.random.lognormal(13, 0.5, n),
        'amount': np.random.lognormal(18, 0.5, n),
        'turnover_rate': np.random.uniform(0.1, 10, n),
        'change_pct': np.random.uniform(-10, 10, n),
    })

    evaluator = ExpressionEvaluator(df)

    # 测试各种表达式树
    test_cases = [
        # 简单因子: close - open
        ExprNode(op='sub', children=[
            ExprNode(op='field', field='close'),
            ExprNode(op='field', field='open'),
        ]),
        # 时序因子: ts_mean(close, 5)
        ExprNode(op='ts_mean', children=[
            ExprNode(op='field', field='close'),
        ], window=5),
        # 复合: ts_std(ts_delta(close, 1), 20)
        ExprNode(op='ts_std', children=[
            ExprNode(op='ts_delta', children=[
                ExprNode(op='field', field='close'),
            ], window=1),
        ], window=20),
    ]

    results = []
    for case in test_cases:
        val = evaluator.evaluate(case)
        nan_rate = np.isnan(val).mean()
        expr_str = case.to_string()
        results.append({
            'expression': expr_str,
            'nan_rate': nan_rate,
            'mean': np.nanmean(val),
            'std': np.nanstd(val),
        })

    print(f"{'表达式':<45s} {'NaN%':>8s} {'均值':>10s} {'标准差':>10s}")
    print("-" * 75)
    for r in results:
        print(f"{r['expression']:<45s} {r['nan_rate']:>7.1%} "
              f"{r['mean']:>10.4f} {r['std']:>10.4f}")

    # 验证所有表达式都能正常计算
    all_valid = all(r['nan_rate'] < 0.5 for r in results)
    assert all_valid, "存在表达式计算结果过多 NaN"
    print(f"\n所有表达式评估正常: {all_valid}")
    print("✅ 测试通过：表达式树评估功能正常")


def test_multiple_runs_stability():
    """测试：多次运行的稳定性"""
    print("\n" + "=" * 60)
    print("测试 3: GP 挖掘器稳定性验证")
    print("=" * 60)

    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'open': np.random.uniform(9, 51, n),
        'high': np.random.uniform(10, 52, n),
        'low': np.random.uniform(8, 50, n),
        'close': np.random.uniform(10, 50, n),
        'volume': np.random.lognormal(13, 0.5, n),
        'amount': np.random.lognormal(18, 0.5, n),
        'turnover_rate': np.random.uniform(0.1, 10, n),
        'change_pct': np.random.uniform(-10, 10, n),
    })
    # 人为构造一个有预测力的因子: close 的变化率
    future_rets = df['close'].pct_change(5).shift(-5).fillna(0).values

    results = []
    for run_i in range(3):
        miner = GPMiner(
            data=df,
            future_returns=future_rets,
            population_size=20,
            generations=3,
            max_depth=2,
            tournament_size=3,
            random_state=42 + run_i,
        )
        result = miner.run()
        results.append(result)

    print(f"3 次运行结果:")
    for i, r in enumerate(results):
        print(f"  运行 {i + 1}: 最优 IC={r['best_ic']:.6f}, "
              f"因子={r['best_factor'].to_string()[:60]}...")

    # 验证每次运行都有输出
    all_success = all(r['best_factor'] is not None for r in results)
    assert all_success, "部分运行未产生有效因子"
    print(f"\n所有运行均产生有效因子: {all_success}")
    print("✅ 测试通过：GP 挖掘器稳定运行")


def main():
    print("\n" + "=" * 60)
    print("自动化因子挖掘验证测试套件")
    print("借鉴来源: AlphaGen (KDD 2023), tsfresh")
    print("=" * 60)

    test_expression_tree_evaluation()
    test_gp_miner_basic()
    test_multiple_runs_stability()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
    print("\n总结:")
    print("- 实现轻量级遗传编程(GP)因子挖掘器")
    print("- 支持 6 种一元操作符、6 种二元操作符、10 种时序操作符")
    print("- 以 Spearman Rank IC 为适应度函数")
    print("- 支持锦标赛选择、子树交叉、随机变异")
    print("- 可在 CPU 上运行，不需 GPU")
    print("- 相比 AlphaGen 的 RL 方案更轻量、易集成")


if __name__ == "__main__":
    main()