"""资金流向表渲染器"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.flow_table")


def render_flow_table(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染资金流向表"""
    try:
        from scripts.factors.registry import get_factor_meta
    except Exception:
        get_factor_meta = None

    rows = []
    for name, val in factor_values.items():
        label = name
        if get_factor_meta:
            meta = get_factor_meta(name)
            if meta:
                label = meta.description

        formatted = "—"
        cls = ""
        if val is not None:
            try:
                v = float(val)
                if abs(v) >= 1e8:
                    formatted = f"{v/1e8:.2f} 亿"
                elif abs(v) >= 1e4:
                    formatted = f"{v/1e4:.2f} 万"
                else:
                    formatted = f"{v:.2f}"
                cls = "flow-in" if v > 0 else ("flow-out" if v < 0 else "")
            except (ValueError, TypeError):
                formatted = str(val)

        rows.append(f'<tr><td>{label}</td><td class="{cls}">{formatted}</td></tr>')

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    table_html = f'<table class="flow-table"><thead><tr><th>指标</th><th>金额</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    return f'{hint_html}{table_html}'
