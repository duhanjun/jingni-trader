"""股东表格渲染器"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.holder_table")


def render_holder_table(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染股东结构表格"""
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
                if 'change' in name:
                    if v > 0:
                        formatted = f"增持 (+{int(v)})"
                        cls = "holder-increase"
                    elif v < 0:
                        formatted = f"减持 ({int(v)})"
                        cls = "holder-decrease"
                    else:
                        formatted = "不变"
                        cls = "holder-neutral"
                elif 'concentration' in name:
                    formatted = f"{v:.2f}%"
                else:
                    formatted = str(val)
            except (ValueError, TypeError):
                formatted = str(val)

        rows.append(f'<tr><td>{label}</td><td class="{cls}">{formatted}</td></tr>')

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    table_html = f'<table class="holder-table"><thead><tr><th>指标</th><th>状态</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    return f'{hint_html}{table_html}'
