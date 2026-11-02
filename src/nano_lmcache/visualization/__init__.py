"""Self-contained HTML rendering for cache replay reports."""

from nano_lmcache.visualization.html import render_html, write_html
from nano_lmcache.visualization.trace import load_trace_json, report_from_events

__all__ = ["load_trace_json", "render_html", "report_from_events", "write_html"]
