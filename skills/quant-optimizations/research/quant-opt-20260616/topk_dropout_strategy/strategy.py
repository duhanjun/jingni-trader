"""
Top-K Dropout 组合策略

借鉴自：
- Microsoft Qlib: qlib.contrib.strategy.TopkDropoutStrategy
- 优点：相比普通 TopK 策略，能够换手更平稳，避免频繁调仓的摩擦成本。
        当调仓日重新打分时，只在「新增上榜 + 跌出榜」两端发生换手。

设计要点：
- 已知当前持仓权重与目标得分，输出调仓后的目标权重。
- 不依赖回测引擎，纯函数式，便于单元测试和组合到任意回测框架。
- 与 jingni-trader portfolio-risk-engine 的现有优化器无依赖，可独立验证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["TopKDropoutStrategy"]


class TopKDropoutStrategy:
    """Top-K 换出策略 (TopKDropout)。

    每一调仓日，先按 alpha_score 排序：
        - 取分数最高的前 k 个作为目标持仓。
        - 在已有的 N_dropout 个持仓中剔除分数最低的（让出位置给新入选的）。
        - 目标权重 = 1/k 的等权分配（可改为 score-weighted）。

    Parameters
    ----------
    top_k : int
        期望持仓数量。
    n_dropout : int
        每次调仓强制换出的旧持仓数。
    score_col : str
        评分列名。
    weight_method : str
        - "equal" : 等权
        - "score" : 归一化分数加权
    long_only : bool
        是否仅做多（暂保留接口位）。
    """

    def __init__(
        self,
        top_k: int = 50,
        n_dropout: int = 5,
        score_col: str = "alpha_score",
        weight_method: str = "equal",
        long_only: bool = True,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        if n_dropout < 0 or n_dropout > top_k:
            raise ValueError("n_dropout 必须在 [0, top_k] 之间")
        if weight_method not in ("equal", "score"):
            raise ValueError("weight_method 仅支持 equal / score")
        self.top_k = top_k
        self.n_dropout = n_dropout
        self.score_col = score_col
        self.weight_method = weight_method
        self.long_only = long_only

    # ------------------------------------------------------------------
    def rebalance(
        self,
        current_holdings: list[str],
        scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """生成调仓后的目标持仓。

        Parameters
        ----------
        current_holdings : list[str]
            当前已持有的股票代码列表。
        scores : pd.DataFrame
            必须含 code 和 score_col 两列；同时可以有其他列作为输出保留。

        Returns
        -------
        pd.DataFrame
            含 code / weight 两列的目标持仓。
        """
        if self.score_col not in scores.columns:
            raise KeyError(f"scores 缺少列 {self.score_col}")
        if "code" not in scores.columns:
            raise KeyError("scores 缺少 code 列")

        # 1) 排序：分数从高到低
        sorted_df = (
            scores[["code", self.score_col]]
            .dropna(subset=[self.score_col])
            .sort_values(self.score_col, ascending=False)
            .reset_index(drop=True)
        )
        if sorted_df.empty:
            return pd.DataFrame({"code": [], "weight": []})

        # 2) 选出 top_k
        top_set = set(sorted_df.head(self.top_k)["code"].tolist())

        # 3) 在旧持仓中淘汰分数最低的 n_dropout 个
        to_drop: list[str] = []
        if self.n_dropout > 0 and current_holdings:
            old_in_scores = sorted_df[
                sorted_df["code"].isin(current_holdings)
            ].copy()
            # 分数最低的 n_dropout 个要淘汰
            to_drop = old_in_scores.tail(self.n_dropout)["code"].tolist()
            top_set -= set(to_drop)

        # 4) 补足到 top_k：从 top_k 之后的位置取前 n_dropout 个分数最高的
        #    （保证补入的股票是「榜外」新股票，而不是刚被淘汰的旧持仓）
        if len(top_set) < self.top_k:
            drop_set = set(to_drop)
            for code in sorted_df["code"].iloc[self.top_k:]:
                if code in top_set or code in drop_set:
                    continue
                top_set.add(code)
                if len(top_set) >= self.top_k:
                    break

        target_codes = list(top_set)
        target_df = pd.DataFrame({"code": target_codes})

        # 5) 计算权重
        if self.weight_method == "equal":
            w = 1.0 / max(len(target_codes), 1)
            target_df["weight"] = w
        else:  # score
            # 保证 weights 与 target_codes 顺序一一对应
            score_map = dict(
                zip(scores["code"], scores[self.score_col].clip(lower=0))
            )
            sc = np.array(
                [score_map.get(c, 0.0) for c in target_codes], dtype=float
            )
            if sc.sum() <= 0:
                w = 1.0 / max(len(target_codes), 1)
                target_df["weight"] = w
            else:
                weights = sc / sc.sum()
                target_df["weight"] = weights

        return target_df[["code", "weight"]].sort_values(
            "weight", ascending=False
        ).reset_index(drop=True)