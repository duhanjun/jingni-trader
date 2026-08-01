"""
行业对比分析模块
将个股的基本面指标与同行业公司进行对比
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("industry_compare")


# 指标配置: metric -> (higher_is_better, 中文标签, 类别)
# 类别: "valuation" 估值类 / "fundamental" 基本面类
# higher_is_better=True 表示该指标数值越大对公司越有利（如 ROE）
#                        False 表示越小越有利（如 PE、资产负债率）
_METRIC_CONFIG: Dict[str, Tuple[bool, str, str]] = {
    "pe_ttm":         (False, "PE(市盈率)",   "valuation"),
    "pb":             (False, "PB(市净率)",   "valuation"),
    "ps_ttm":         (False, "PS(市销率)",   "valuation"),
    "dv_ratio":       (True,  "股息率",        "valuation"),
    "roe":            (True,  "ROE",          "fundamental"),
    "roa":            (True,  "ROA",          "fundamental"),
    "gross_margin":   (True,  "毛利率",        "fundamental"),
    "net_margin":     (True,  "净利率",        "fundamental"),
    "revenue_growth": (True,  "营收增速",      "fundamental"),
    "profit_growth":  (True,  "利润增速",      "fundamental"),
    "debt_ratio":     (False, "资产负债率",    "fundamental"),
    "current_ratio":  (True,  "流动比率",      "fundamental"),
    "quick_ratio":    (True,  "速动比率",      "fundamental"),
}

_DEFAULT_METRICS: List[str] = list(_METRIC_CONFIG.keys())


class IndustryComparator:
    """个股行业对比分析"""

    def compare(self, stock_code: str, financial_data: pd.DataFrame,
                industry: Optional[str] = None) -> Dict:
        """
        将个股与同行业公司进行多维度对比

        参数:
            stock_code: 股票代码
            financial_data: 全市场财务数据，含 code, industry, pe_ttm, pb, roe等列
            industry: 指定行业（若None则自动从financial_data中查找）

        返回:
            {
                "stock": "000001.SZ",
                "industry": "银行",
                "industry_avg": {"pe_ttm": 8.5, "pb": 0.9, "roe": 12.0, ...},
                "stock_values": {"pe_ttm": 12.5, "pb": 1.8, "roe": 15.2, ...},
                "rankings": {
                    "pe_ttm": {"rank": 15, "total": 30, "percentile": 50.0},
                    "roe": {"rank": 8, "total": 30, "percentile": 73.3},
                },
                "advantages": ["ROE高于行业平均", "营收增速领先"],
                "disadvantages": ["PE高于行业平均"],
                "summary": "盈利能力优于行业平均，但估值偏高"
            }
        """
        empty = {
            "stock": stock_code,
            "industry": industry,
            "industry_avg": {},
            "stock_values": {},
            "rankings": {},
            "advantages": [],
            "disadvantages": [],
            "summary": "无数据",
        }

        if financial_data is None or len(financial_data) == 0:
            logger.warning("financial_data 为空: %s", stock_code)
            return empty
        if "code" not in financial_data.columns:
            logger.error("financial_data 缺少 code 列")
            return empty
        if "industry" not in financial_data.columns:
            logger.error("financial_data 缺少 industry 列")
            return empty

        # 定位个股
        stock_rows = financial_data[financial_data["code"] == stock_code]
        if stock_rows.empty:
            logger.warning("financial_data 中未找到股票 %s", stock_code)
            return empty
        stock_row = stock_rows.iloc[0]

        # 行业判定：显式传入优先，否则从个股数据中读取
        if industry is None:
            industry = stock_row.get("industry")
        if industry is None or (isinstance(industry, float) and np.isnan(industry)) \
                or (isinstance(industry, str) and not industry.strip()):
            logger.warning("股票 %s 缺少行业信息", stock_code)
            return empty

        # 行业内全部公司（含自身，用于排名）
        industry_df = financial_data[financial_data["industry"] == industry].copy()
        if industry_df.empty:
            logger.warning("行业 %s 无成员公司", industry)
            return empty
        if len(industry_df) == 1 and industry_df["code"].iloc[0] == stock_code:
            logger.warning("行业 %s 仅含目标股票，无法对比", industry)
            return {**empty, "industry": industry}

        # 待对比指标：默认列表与数据列的交集
        metrics = [m for m in _DEFAULT_METRICS if m in industry_df.columns]

        # 个股指标值
        stock_values: Dict[str, Optional[float]] = {}
        for m in metrics:
            v = stock_row.get(m)
            try:
                v = float(v)
                if not np.isfinite(v):
                    v = None
            except (TypeError, ValueError):
                v = None
            stock_values[m] = v

        # 行业均值：剔除自身后计算（更公允的同行对比）
        peers_df = industry_df[industry_df["code"] != stock_code]
        industry_avg: Dict[str, Optional[float]] = {}
        for m in metrics:
            s = pd.to_numeric(peers_df[m], errors="coerce") \
                       .replace([np.inf, -np.inf], np.nan).dropna()
            industry_avg[m] = None if s.empty else round(float(s.mean()), 4)

        # 行业排名
        rankings = self._calc_rankings(stock_values, industry_df, metrics)

        # 优势/劣势洞察
        advantages, disadvantages = self._generate_insights(
            stock_values, industry_avg, rankings
        )

        # 汇总结论
        summary = self._build_summary(rankings)

        return {
            "stock": stock_code,
            "industry": industry,
            "industry_avg": industry_avg,
            "stock_values": {
                m: (round(v, 4) if v is not None else None)
                for m, v in stock_values.items()
            },
            "rankings": rankings,
            "advantages": advantages,
            "disadvantages": disadvantages,
            "summary": summary,
        }

    def _calc_rankings(self, stock_values: Dict, industry_df: pd.DataFrame,
                       metrics: List[str]) -> Dict:
        """计算个股在各指标中的行业排名

        rank 为 1 基，rank=1 表示该指标最优（PE 最低 或 ROE 最高）。
        percentile = (total - rank) / total * 100，代表个股优于行业内同行的比例。
        """
        rankings: Dict[str, Dict] = {}
        for m in metrics:
            sv = stock_values.get(m)
            if sv is None or m not in industry_df.columns:
                continue

            higher_is_better, label, _cat = _METRIC_CONFIG.get(
                m, (True, m, "fundamental")
            )
            col = pd.to_numeric(industry_df[m], errors="coerce") \
                         .replace([np.inf, -np.inf], np.nan).dropna()
            total = len(col)
            if total == 0:
                continue

            # 优于个股的数量（不含自身：自身等于 sv，不参与严格比较）
            if higher_is_better:
                better = int((col > sv).sum())
            else:
                better = int((col < sv).sum())

            rank = better + 1  # 1-based，最优为 1
            percentile = (total - rank) / total * 100.0
            percentile = max(0.0, min(100.0, percentile))

            rankings[m] = {
                "rank": rank,
                "total": int(total),
                "percentile": round(percentile, 2),
                "label": label,
            }
        return rankings

    def _generate_insights(self, stock_values: Dict, industry_avg: Dict,
                           rankings: Dict) -> Tuple[List[str], List[str]]:
        """生成优势/劣势洞察

        分位 >= 60% 视为优势，<= 40% 视为劣势；
        中段(40%-60%) 则与行业均值做对比补充判断。
        """
        advantages: List[str] = []
        disadvantages: List[str] = []

        for m, info in rankings.items():
            label = info.get("label", m)
            pct = info["percentile"]
            sv = stock_values.get(m)
            av = industry_avg.get(m)

            if pct >= 60:
                advantages.append(f"{label}处于行业{pct:.0f}%分位，优于行业平均")
            elif pct <= 40:
                disadvantages.append(f"{label}处于行业{pct:.0f}%分位，低于行业平均")
            else:
                # 中段：以与行业均值的对比做补充
                if sv is not None and av is not None:
                    higher_is_better = _METRIC_CONFIG.get(m, (True,))[0]
                    if (higher_is_better and sv > av) or \
                       (not higher_is_better and sv < av):
                        advantages.append(f"{label}优于行业平均")
                    elif (higher_is_better and sv < av) or \
                         (not higher_is_better and sv > av):
                        disadvantages.append(f"{label}弱于行业平均")

        return advantages, disadvantages

    def _build_summary(self, rankings: Dict) -> str:
        """汇总基本面与估值两类的对比结论"""
        val_adv = val_dis = fund_adv = fund_dis = 0
        for m, info in rankings.items():
            cat = _METRIC_CONFIG.get(m, (True, m, "fundamental"))[2]
            pct = info["percentile"]
            if cat == "valuation":
                if pct >= 60:
                    val_adv += 1
                elif pct <= 40:
                    val_dis += 1
            else:
                if pct >= 60:
                    fund_adv += 1
                elif pct <= 40:
                    fund_dis += 1

        parts: List[str] = []
        if fund_adv > fund_dis:
            parts.append("盈利能力优于行业平均")
        elif fund_dis > fund_adv:
            parts.append("盈利能力弱于行业平均")

        if val_dis > val_adv:
            parts.append("估值偏高")
        elif val_adv > val_dis:
            parts.append("估值偏低")

        if not parts:
            return "整体与行业平均接近"

        # 基本面优于但估值偏高时，用转折语气拼接
        if len(parts) == 2 and "优于" in parts[0] and "偏高" in parts[1]:
            return f"{parts[0]}，但{parts[1]}"
        return "，".join(parts)
