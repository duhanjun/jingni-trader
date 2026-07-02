"""
基准对比 + 因子热力图 HTML 报告
================================

借鉴来源：
- AkQuant 的 `result.report(..., benchmark=...)` HTML 报告
  （含基准对比区块：跟踪误差、IR、Alpha、Beta、累计超额）
- Qlib 的图表主题 + Plotly 嵌入
- jingni-trader 现有 reports-engine

本报告在保持轻量（仅 stdlib + 简单 HTML/CSS）的前提下，
提供 AkQuant 报告中最具价值的"基准对比"区块，
便于开发者直观判断策略相对收益。
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime
import html as html_lib
import os


def _fmt_pct(x: float, digits: int = 2) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.{digits}f}%"


def _fmt_num(x: float, digits: int = 4) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}"


def _sparkline_svg(values, width: int = 200, height: int = 40) -> str:
    """用内嵌 SVG 画净值曲线（避免外部依赖）"""
    if values is None or len(values) < 2:
        return ""
    import numpy as np
    v = np.asarray(values, dtype=float)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    n = len(v)
    step = width / (n - 1)
    points = " ".join(f"{i * step:.1f},{height - v_i * height:.1f}" for i, v_i in enumerate(v))
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"><polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="1.5"/></svg>'


def _drawdown_svg(values, width: int = 200, height: int = 40) -> str:
    """回撤曲线（红色）"""
    if values is None or len(values) < 2:
        return ""
    import numpy as np
    v = np.asarray(values, dtype=float)
    cm = np.maximum.accumulate(v)
    dd = (v - cm) / cm
    dmin, dmax = dd.min(), 0.0
    rng = (dmax - dmin) or 1e-12
    norm = (dd - dmin) / rng
    n = len(norm)
    step = width / (n - 1)
    points = " ".join(f"{i * step:.1f},{height - n_i * height:.1f}" for i, n_i in enumerate(norm))
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"><polyline points="{points}" fill="none" stroke="#dc2626" stroke-width="1.5"/></svg>'


def render_metrics_html(
    metrics: Dict[str, Any],
    equity_curve: Optional[Any] = None,
    benchmark: Optional[Any] = None,
    title: str = "策略回测报告",
) -> str:
    """
    渲染单策略绩效报告 HTML

    参数:
        metrics:       calc_all_metrics() 输出
        equity_curve:  净值 Series（index=date, name=equity）
        benchmark:     基准净值 Series
        title:         报告标题
    """
    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html_lib.escape(title)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#f9fafb;color:#111827;margin:0;padding:24px;}}
.container{{max-width:1100px;margin:0 auto;background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.06);padding:24px;}}
h1{{margin-top:0;font-size:22px;color:#1e3a8a;}}
h2{{font-size:16px;border-left:4px solid #2563eb;padding-left:8px;margin-top:28px;}}
table{{width:100%;border-collapse:collapse;margin:8px 0;}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #e5e7eb;font-size:13px;}}
th{{background:#f3f4f6;color:#374151;font-weight:600;}}
tr:hover td{{background:#f9fafb;}}
.kpi{{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0;}}
.kpi-card{{flex:1 1 180px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:12px;}}
.kpi-card .label{{font-size:12px;color:#6b7280;}}
.kpi-card .value{{font-size:20px;font-weight:600;color:#1e3a8a;margin-top:4px;}}
.kpi-card.neg .value{{color:#b91c1c;}}
.note{{font-size:12px;color:#6b7280;margin-top:24px;}}
</style>
</head><body><div class="container">""")
    parts.append(f"<h1>{html_lib.escape(title)}</h1>")
    parts.append(f"<div style='font-size:12px;color:#6b7280;'>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>")

    # ── 关键指标卡片 ──
    cards = [
        ("年化收益", _fmt_pct(metrics.get("annual_return", 0)), metrics.get("annual_return", 0) >= 0),
        ("夏普比率", _fmt_num(metrics.get("sharpe_ratio", 0)), metrics.get("sharpe_ratio", 0) >= 0),
        ("最大回撤", _fmt_pct(metrics.get("max_drawdown", 0)), False),
        ("胜率", _fmt_pct(metrics.get("win_rate", 0), 1), True),
    ]
    parts.append('<div class="kpi">')
    for label, val, pos in cards:
        cls = "kpi-card" + ("" if pos else " neg")
        parts.append(f'<div class="{cls}"><div class="label">{label}</div><div class="value">{val}</div></div>')
    parts.append("</div>")

    # ── 净值曲线 ──
    if equity_curve is not None and len(equity_curve) >= 2:
        eq_values = list(equity_curve.values)
        parts.append("<h2>净值曲线</h2>")
        parts.append(_sparkline_svg(eq_values, width=1040, height=80))
        # 同步画基准
        if benchmark is not None and len(benchmark) >= 2:
            parts.append(f"<div style='font-size:12px;color:#6b7280;'>蓝色: 策略; 灰色虚线: 基准</div>")
            bench_svg = _sparkline_svg(list(benchmark.reindex(equity_curve.index).ffill().dropna().values), width=1040, height=80)
            parts.append(bench_svg.replace('stroke="#2563eb"', 'stroke="#9ca3af" stroke-dasharray="3,3"'))

        # 回撤
        parts.append("<h2>回撤曲线</h2>")
        parts.append(_drawdown_svg(eq_values, width=1040, height=60))

    # ── 详细指标 ──
    parts.append("<h2>详细绩效指标</h2>")
    parts.append("<table><tr><th>指标</th><th>数值</th></tr>")
    detail_keys = [
        ("total_return", "累计收益", _fmt_pct),
        ("annual_return", "年化收益", _fmt_pct),
        ("volatility", "年化波动率", _fmt_pct),
        ("sharpe_ratio", "夏普比率", _fmt_num),
        ("sortino_ratio", "索提诺比率", _fmt_num),
        ("max_drawdown", "最大回撤", _fmt_pct),
        ("max_drawdown_duration_days", "最长回撤持续期（天）", lambda x: str(int(x)) if x is not None else "N/A"),
        ("calmar_ratio", "Calmar 比率", _fmt_num),
        ("win_rate", "胜率", lambda x: _fmt_pct(x, 1)),
        ("total_trades", "总交易笔数", lambda x: str(int(x)) if x is not None else "N/A"),
    ]
    for key, label, fmt in detail_keys:
        if key in metrics:
            parts.append(f"<tr><td>{label}</td><td>{fmt(metrics[key])}</td></tr>")
    parts.append("</table>")

    # ── 基准对比（核心：借鉴 AkQuant）──
    if "benchmark" in metrics and "excess" in metrics:
        parts.append("<h2>基准对比（vs 基准）</h2>")
        bench = metrics["benchmark"]
        exc = metrics["excess"]
        parts.append("<table><tr><th>指标</th><th>数值</th></tr>")
        rows = [
            ("基准累计收益", _fmt_pct(bench.get("total_return", 0))),
            ("基准年化收益", _fmt_pct(bench.get("annual_return", 0))),
            ("累计超额收益", _fmt_pct(exc.get("cumulative_excess_return", 0))),
            ("跟踪误差 (Tracking Error)", _fmt_pct(exc.get("tracking_error", 0))),
            ("信息比率 (Information Ratio)", _fmt_num(exc.get("information_ratio", 0))),
            ("Jensen's Alpha (年化)", _fmt_pct(exc.get("alpha", 0))),
            ("Beta (vs 基准)", _fmt_num(exc.get("beta", 0))),
        ]
        for label, val in rows:
            parts.append(f"<tr><td>{label}</td><td>{val}</td></tr>")
        parts.append("</table>")

    parts.append("""<div class="note">本报告由 jingni-trader 量化优化模块 (quant_opt) 自动生成。
借鉴 AkQuant 0.2.43 报告结构（基准对比区块）+ Qlib 净值曲线风格。</div>""")
    parts.append("</div></body></html>")
    return "".join(parts)


def save_report(html: str, output_path: str) -> str:
    """保存报告到文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
