"""
优化方向: Point-in-Time (PIT) 数据基础设施
借鉴来源: Microsoft Qlib (github.com/microsoft/qlib, 36.5k+ stars)
         Qlib 的 Point-in-Time 数据系统

核心问题:
  jingni-trader 当前的数据处理流程不区分"数据发布时间"和"数据观测时间"。
  对于财务报表数据（季报/年报），这会导致 look-ahead bias（前视偏差）:
  - 例如，2024年Q1的季报实际在4月底才发布
  - 如果在3月31日的回测中使用了Q1的财务数据，就是错误的
  - Qlib 通过 PIT 数据系统确保每个时间点只能使用当时已发布的数据

验证目标:
  1. 构造一个含发布延迟的财务数据样例
  2. 对比 PIT 处理和 non-PIT 处理的结果差异
  3. 量化 look-ahead bias 的影响

关于 PIT 的详细阐释:
  - 财务报表的"截止日期"(report_date) 和 "公告日期"(ann_date) 不同
  - A股季报公告截止日: Q1在4/30, 半年报在8/31, Q3在10/31, 年报在次年4/30
  - 回测中使用 PIT 数据意味着: 在 ann_date 之后才能使用该报告的数据
  - 常见的错误做法: 直接用 report_date 对齐，导致使用了"未来信息"
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("pit-data-test")


# ============================================================================
# 1. 模拟数据生成
# ============================================================================


def generate_pit_test_data(n_stocks: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成含报告发布日期的财务数据。

    A股季报公告时间规则:
      Q1 (截止 3/31):  公告截止日 4/30
      Q2 (截止 6/30):  公告截止日 8/31
      Q3 (截止 9/30):  公告截止日 10/31
      Q4 (截止 12/31): 公告截止日 次年 4/30

    实际公告日期通常在截止日前 1-30 天随机分布。

    返回:
        price_data: 日线行情数据
        fin_data: 财务数据 (含 report_date 和 ann_date)
    """
    np.random.seed(42)

    dates = pd.bdate_range(start='2022-01-01', end='2024-12-31')
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    # 生成价格数据
    price_rows = []
    for code in codes:
        start_price = np.random.uniform(5, 50)
        trend = np.random.uniform(-0.0001, 0.0004)
        returns = np.random.normal(trend, 0.02, len(dates))
        returns[0] = 0
        prices = start_price * np.cumprod(1 + returns)
        price_rows.append(pd.DataFrame({
            'date': dates,
            'code': code,
            'close': prices,
        }))
    price_data = pd.concat(price_rows, ignore_index=True)

    # 生成财务数据（PE, PB, ROE 等）
    def _get_ann_deadline(report_date: pd.Timestamp) -> pd.Timestamp:
        """获取某季度报告的公告截止日"""
        month = report_date.month
        year = report_date.year
        if month == 3:
            return pd.Timestamp(f"{year}-04-30")
        elif month == 6:
            return pd.Timestamp(f"{year}-08-31")
        elif month == 9:
            return pd.Timestamp(f"{year}-10-31")
        else:
            return pd.Timestamp(f"{year + 1}-04-30")

    fin_rows = []
    for code in codes:
        # 每季度一份报告
        report_dates = []
        for year in [2022, 2023, 2024]:
            for month in [3, 6, 9, 12]:
                # 月末日期：各月天数不同
                if month in [1, 3, 5, 7, 8, 10, 12]:
                    day = '31'
                elif month == 2:
                    day = '28'  # 简化处理，忽略闰年
                else:
                    day = '30'
                rd = pd.Timestamp(f"{year}-{month:02d}-{day}")
                if rd <= pd.Timestamp('2024-12-31'):
                    report_dates.append(rd)

        base_roe = np.random.uniform(0.05, 0.20)
        base_pe = np.random.uniform(10, 40)

        for rd in report_dates:
            deadline = _get_ann_deadline(rd)
            # 公告日: deadline 前 1-30 天随机（模拟提前公告）
            days_before = np.random.randint(1, min(31, (deadline - rd).days + 1))
            ann_date = deadline - pd.Timedelta(days=days_before)

            # 模拟因子值随时间的 drift
            drift = np.random.normal(0, 0.02)
            roe = max(base_roe * (1 + drift), 0.01)
            pe = max(base_pe * (1 + drift * 3), 5)

            fin_rows.append({
                'code': code,
                'report_date': rd,
                'ann_date': ann_date,
                'roe': round(roe, 4),
                'pe': round(pe, 2),
                'pb': round(pe * roe, 2),
                'revenue_yoy': round(np.random.normal(0.05, 0.15), 4),
                'profit_yoy': round(np.random.normal(0.03, 0.20), 4),
            })

    fin_data = pd.DataFrame(fin_rows)
    return price_data, fin_data


# ============================================================================
# 2. PIT 数据处理 vs. 非 PIT 处理
# ============================================================================


def process_non_pit(price_data: pd.DataFrame, fin_data: pd.DataFrame) -> pd.DataFrame:
    """
    非 PIT 处理（当前 jingni-trader 的潜在问题）。

    直接用 report_date 对齐行情日期，意味着在 3月31日就可能使用了
    尚未公告的 Q1 季报数据 → 这就是 look-ahead bias。
    """
    result = price_data.copy()

    # 按季度和股票匹配最近的财务数据
    # 错误做法: 用 report_date <= date 来匹配（不考虑公告延迟）
    fin_data_sorted = fin_data.sort_values(['code', 'report_date'])

    # 构建一个字典: (code, date) -> 最新财务数据
    fin_lookup = {}
    for code in fin_data['code'].unique():
        code_fin = fin_data_sorted[fin_data_sorted['code'] == code]
        for _, prow in price_data[price_data['code'] == code].iterrows():
            dt = prow['date']
            # 错误: 找到 report_date <= dt 的最新报告
            available = code_fin[code_fin['report_date'] <= dt]
            if not available.empty:
                latest = available.iloc[-1]
                fin_lookup[(code, dt)] = latest

    # 填充因子值
    for col in ['roe', 'pe', 'pb', 'revenue_yoy', 'profit_yoy']:
        result[col] = np.nan

    for (code, dt), fin_row in fin_lookup.items():
        mask = (result['code'] == code) & (result['date'] == dt)
        for col in ['roe', 'pe', 'pb', 'revenue_yoy', 'profit_yoy']:
            result.loc[mask, col] = fin_row[col]

    result['method'] = 'non_pit'
    return result


def process_pit(price_data: pd.DataFrame, fin_data: pd.DataFrame) -> pd.DataFrame:
    """
    PIT 处理（推荐方式）。

    使用 ann_date (公告日期) 来对齐，确保回测中只使用已公告的数据。
    """
    result = price_data.copy()

    fin_data_sorted = fin_data.sort_values(['code', 'ann_date'])

    # 构建 PIT 查找表
    fin_lookup = {}
    for code in fin_data['code'].unique():
        code_fin = fin_data_sorted[fin_data_sorted['code'] == code]
        for _, prow in price_data[price_data['code'] == code].iterrows():
            dt = prow['date']
            # 正确: 找到 ann_date <= dt 的最新报告
            available = code_fin[code_fin['ann_date'] <= dt]
            if not available.empty:
                latest = available.iloc[-1]
                fin_lookup[(code, dt)] = latest

    for col in ['roe', 'pe', 'pb', 'revenue_yoy', 'profit_yoy']:
        result[col] = np.nan

    for (code, dt), fin_row in fin_lookup.items():
        mask = (result['code'] == code) & (result['date'] == dt)
        for col in ['roe', 'pe', 'pb', 'revenue_yoy', 'profit_yoy']:
            result.loc[mask, col] = fin_row[col]

    result['method'] = 'pit'
    return result


# ============================================================================
# 3. Look-ahead Bias 量化分析
# ============================================================================


def quantify_lookahead_bias(
    price_data: pd.DataFrame,
    pit_data: pd.DataFrame,
    non_pit_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    量化 look-ahead bias 的影响。

    分析维度:
    1. 有多少天的数据受到了"未来信息"污染
    2. 哪些财务指标受影响最大
    3. 在构建因子信号时，差异会如何放大
    """
    # 合并 PIT 和 non-PIT 数据
    combined = pit_data[['date', 'code', 'close', 'roe', 'pe', 'pb', 'revenue_yoy', 'profit_yoy']].copy()
    combined.columns = ['date', 'code', 'close',
                        'roe_pit', 'pe_pit', 'pb_pit',
                        'revenue_yoy_pit', 'profit_yoy_pit']

    non_pit_cols = non_pit_data[['date', 'code', 'roe', 'pe', 'pb', 'revenue_yoy', 'profit_yoy']].copy()
    non_pit_cols.columns = ['date', 'code',
                            'roe_nonpit', 'pe_nonpit', 'pb_nonpit',
                            'revenue_yoy_nonpit', 'profit_yoy_nonpit']

    combined = combined.merge(non_pit_cols, on=['date', 'code'], how='inner')

    # 1. 统计差异天数
    for metric in ['roe', 'pe', 'pb']:
        pit_col = f'{metric}_pit'
        nonpit_col = f'{metric}_nonpit'
        # 找出值不同的记录
        diff_mask = (
            (combined[pit_col].notna()) &
            (combined[nonpit_col].notna()) &
            (np.abs(combined[pit_col] - combined[nonpit_col]) > 1e-10)
        )
        n_diff = diff_mask.sum()
        n_total = combined[pit_col].notna().sum()

        if n_total > 0:
            logger.info(f"  {metric}: {n_diff}/{n_total} 天数据有差异 "
                       f"({n_diff/n_total*100:.1f}%), "
                       f"平均差异: {np.abs(combined.loc[diff_mask, pit_col] - combined.loc[diff_mask, nonpit_col]).mean():.4f}")

    # 2. 分析公告延迟窗口内的数据差异
    # 对于每份报告，找出 report_date 到 ann_date 之间的天数
    # non-PIT 在这些天里使用了"未来"数据
    combined['year_month'] = combined['date'].dt.to_period('M')
    has_pit = combined['roe_pit'].notna()
    has_nonpit = combined['roe_nonpit'].notna()
    early_access = has_nonpit & (~has_pit)
    n_early = early_access.sum()

    logger.info(f"\n  提前获取数据的天数（look-ahead 风险天数）: {n_early}")
    if n_early > 0:
        logger.info(f"  占总有效数据比例: {n_early/has_nonpit.sum()*100:.1f}%")

    # 3. 因子构建中的差异放大
    # 假设使用 PE 构建反转因子: rank(-PE)
    for dt in sorted(combined['date'].unique()):
        row_mask = combined['date'] == dt
        if row_mask.sum() < 5:
            continue
        sub = combined[row_mask].copy()
        sub['rank_pit'] = sub['pe_pit'].rank(pct=True)
        sub['rank_nonpit'] = sub['pe_nonpit'].rank(pct=True)
        combined.loc[row_mask, 'rank_pit'] = sub['rank_pit']
        combined.loc[row_mask, 'rank_nonpit'] = sub['rank_nonpit']

    rank_diff = (combined['rank_pit'] - combined['rank_nonpit']).abs()
    logger.info(f"\n  因子排名差异（PE 反转因子）:")
    logger.info(f"    平均排名差异: {rank_diff.mean():.4f} (最大 1.0)")
    logger.info(f"    Top/Bottom 不一致比例: "
               f"{(rank_diff > 0.2).sum()/rank_diff.notna().sum()*100:.1f}% (差异 > 0.2)")

    return {
        'n_early_access_days': int(n_early),
        'early_access_pct': round(n_early / has_nonpit.sum() * 100, 2) if has_nonpit.sum() > 0 else 0,
        'mean_rank_diff': round(rank_diff.mean(), 4),
        'large_rank_diff_pct': round((rank_diff > 0.2).sum() / rank_diff.notna().sum() * 100, 2) if rank_diff.notna().sum() > 0 else 0,
    }


# ============================================================================
# 4. 兼容层设计 (PIT 数据标准化接口)
# ============================================================================


class PITDataProvider:
    """
    PIT 数据提供者 - 为 jingni-trader 设计的兼容接口。

    设计思路:
    1. 对于技术面因子（量价）: 不需要 PIT 处理，直接返回
    2. 对于基本面因子（PE/ROE等）: 按 ann_date 过滤
    3. 提供与现有 data-engine 一致的接口，便于无缝替换
    """

    def __init__(self, price_data: pd.DataFrame, fin_data: pd.DataFrame):
        self.price_data = price_data
        self.fin_data = fin_data
        self._build_index()

    def _build_index(self):
        """构建 PIT 索引，加速查询"""
        self.fin_sorted = self.fin_data.sort_values(['code', 'ann_date'])
        self.fin_by_code = {
            code: group for code, group in self.fin_sorted.groupby('code')
        }

    def get_fundamentals(self, code: str, date: pd.Timestamp, fields: list = None) -> dict:
        """
        PIT 方式获取某股票在某个交易日的基本面数据。

        确保只返回该日期前已公告的数据。
        """
        if code not in self.fin_by_code:
            return {}

        code_fin = self.fin_by_code[code]
        available = code_fin[code_fin['ann_date'] <= date]

        if available.empty:
            return {}

        latest = available.iloc[-1]
        if fields is None:
            fields = ['roe', 'pe', 'pb', 'revenue_yoy', 'profit_yoy']

        return {f: latest[f] for f in fields if f in latest.index}

    def enrich_price_data(self) -> pd.DataFrame:
        """
        为行情数据添加 PIT 基本面因子。

        返回: 增加了 PIT 基本面列的行情 DataFrame
        """
        result = self.price_data.copy()

        fin_cols = ['roe', 'pe', 'pb', 'revenue_yoy', 'profit_yoy']
        for col in fin_cols:
            result[col] = np.nan

        for code in result['code'].unique():
            code_mask = result['code'] == code
            code_data = result[code_mask]

            for idx in code_data.index:
                dt = code_data.loc[idx, 'date']
                fundamentals = self.get_fundamentals(code, dt, fin_cols)
                for col, val in fundamentals.items():
                    result.loc[idx, col] = val

        return result


# ============================================================================
# 5. 主测试
# ============================================================================


def run_all_tests():
    """运行所有 PIT 验证测试"""
    logger.info("=" * 60)
    logger.info("jingni-trader Point-in-Time 数据验证")
    logger.info(f"测试时间: {datetime.now().isoformat()}")
    logger.info("借鉴来源: Microsoft Qlib PIT Data System")
    logger.info("=" * 60)

    # 生成测试数据
    logger.info("\n生成测试数据: 50只股票, 2022-2024 年...")
    price_data, fin_data = generate_pit_test_data(n_stocks=50)

    # PIT vs non-PIT 处理
    logger.info("\n处理中: PIT 方法...")
    pit_result = process_pit(price_data, fin_data)

    logger.info("处理中: non-PIT 方法...")
    non_pit_result = process_non_pit(price_data, fin_data)

    # 量化 bias
    logger.info("\n" + "=" * 60)
    logger.info("Look-ahead Bias 量化分析")
    logger.info("=" * 60)
    bias_analysis = quantify_lookahead_bias(price_data, pit_result, non_pit_result)

    # PIT 兼容层验证
    logger.info("\n" + "=" * 60)
    logger.info("PITDataProvider 兼容层验证")
    logger.info("=" * 60)
    provider = PITDataProvider(price_data, fin_data)
    enriched = provider.enrich_price_data()

    logger.info(f"  原始行情数据: {len(price_data)} 行")
    logger.info(f"  增强后数据:   {len(enriched)} 行")
    logger.info(f"  基本面覆盖:   {enriched['roe'].notna().sum()}/{len(enriched)} 行 "
               f"({enriched['roe'].notna().sum()/len(enriched)*100:.1f}%)")

    # 验证接口: 查询某股票某日基本面
    test_code = fin_data['code'].iloc[0]
    test_date = pd.Timestamp('2023-06-15')
    fundamentals = provider.get_fundamentals(test_code, test_date)
    logger.info(f"\n  PIT 查询验证: {test_code} @ {test_date.date()}")
    logger.info(f"    基本面数据: {fundamentals}")

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("验证结论")
    logger.info("=" * 60)
    logger.info(f"""
    1. Look-ahead Bias 确实存在:
       - 提前获取数据天数: {bias_analysis['n_early_access_days']} 天
       - 占有效数据比例: {bias_analysis['early_access_pct']}%
       - 因子排名平均差异: {bias_analysis['mean_rank_diff']}
       - 排名差异 > 0.2 的比例: {bias_analysis['large_rank_diff_pct']}%

    2. 影响评估:
       - 对于纯量价因子（如动量、反转）: 无影响
       - 对于基本面因子（PE/ROE等）: 有显著影响，尤其在季报窗口期
       - 在因子组合中，若基本面因子占比较大，整体 ranking 会显著偏离

    3. 对 jingni-trader 的建议:
       - data-engine: 增加 PIT 数据处理选项
       - factor-engine: 为基本面因子增加 ann_date 过滤
       - 配置文件: 增加 PIT_ENABLED 开关
       - 优先实施: PITDataProvider 兼容层（见上方实现）

    4. 实施优先级:
       - 高: data-engine 的 fetch_and_clean 增加 PIT 模式
       - 高: factor-engine 区分技术面/基本面因子的数据对齐策略
       - 中: 配置文件增加 PIT 开关
       - 低: 完整的 ann_date 数据维护管道
    """)


if __name__ == "__main__":
    run_all_tests()