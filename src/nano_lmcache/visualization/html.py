"""Generate an offline HTML trace viewer using only the Python standard library."""

# Embedded CSS and JavaScript are intentionally readable in the generated
# template; their long lines are not Python source formatting concerns.
# ruff: noqa: E501

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nano_lmcache.replay.runner import ReplayReport


def render_html(report: ReplayReport | Mapping[str, Any], *, title: str = "nano-lmcache Trace") -> str:
    """Return a standalone HTML visualizer for a replay report."""
    payload = report.to_dict() if isinstance(report, ReplayReport) else dict(report)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Prevent a JSON value from closing the script element when requests come
    # from an untrusted dataset.
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e")
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a2e; --line:#2d3b5f; --text:#e6edf7; --muted:#9baacc; --hit:#36d399; --miss:#fb7185; --pending:#fbbf24; --incomplete:#a78bfa; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }}
header {{ padding:20px 24px 12px; border-bottom:1px solid var(--line); }} h1 {{ margin:0 0 6px; font-size:22px; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(280px,360px); gap:14px; padding:14px; max-width:1440px; margin:auto; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; min-width:0; }} .wide {{ grid-column:1 / -1; }}
.controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }} button {{ border:1px solid var(--line); background:#1c2948; color:var(--text); border-radius:7px; padding:7px 12px; cursor:pointer; }} button:hover {{ background:#2b3b64; }}
#step-label,.muted {{ color:var(--muted); }} .summary {{ display:flex; gap:20px; flex-wrap:wrap; margin-top:10px; }} .metric b {{ display:block; font-size:20px; }}
.request {{ border-top:1px solid var(--line); padding:12px 0 4px; }} .request:first-child {{ border-top:0; }} .request-title {{ display:flex; justify-content:space-between; gap:12px; }}
.chunks {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }} .chunk {{ padding:7px 9px; border-radius:7px; border:1px solid var(--line); min-width:80px; }}
.chunk.hit {{ border-color:var(--hit); color:var(--hit); }} .chunk.miss {{ border-color:var(--miss); color:var(--miss); }} .chunk.not_checked {{ border-color:var(--pending); color:var(--pending); }} .chunk.incomplete {{ border-color:var(--incomplete); color:var(--incomplete); }} .chunk.active {{ box-shadow:0 0 0 2px white; }}
#timeline {{ display:flex; flex-direction:column; gap:6px; max-height:380px; overflow:auto; }} .event {{ padding:8px 10px; border-left:3px solid var(--line); background:#0e1628; border-radius:5px; cursor:pointer; }} .event.current {{ border-left-color:#60a5fa; background:#182643; }} .event-name {{ font-weight:650; }} .event-detail {{ color:var(--muted); font-size:12px; word-break:break-word; }}
#tree {{ width:100%; min-height:230px; overflow:auto; }} #tree svg {{ min-width:500px; height:230px; }} .tree-node {{ fill:#172443; stroke:#7083af; stroke-width:1.5; }} .tree-node.current {{ stroke:#fff; stroke-width:3; }} .tree-edge {{ stroke:#53658c; stroke-width:1.5; }} .tree-label {{ fill:var(--text); font-size:10px; text-anchor:middle; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} .wide {{ grid-column:auto; }} }}
</style>
</head>
<body>
<header><h1>{safe_title}</h1><div class="muted">Offline replay viewer · no external assets</div></header>
<main>
<section class="panel wide"><div class="controls"><button id="reset">Reset</button><button id="prev">Previous</button><button id="next">Next</button><button id="play">Play</button><button id="pause">Pause</button><span id="step-label"></span></div><div id="summary" class="summary"></div></section>
<section class="panel"><h2>Chunks</h2><div id="requests"></div></section>
<section class="panel"><h2>Prefix Tree</h2><div id="tree"></div><h2>Event Detail</h2><pre id="detail" class="muted"></pre></section>
<section class="panel wide"><h2>Request Timeline</h2><div id="timeline"></div></section>
</main>
<script>
const DATA = {encoded};
let step = -1, timer = null;
const $ = id => document.getElementById(id);
const events = DATA.events || [], requests = DATA.requests || [];
function statusClass(value) {{ return String(value).toLowerCase(); }}
function renderSummary() {{
  const s = DATA.summary || {{}};
  const storage = events.filter(e=>e.event_type.startsWith("storage_")).length;
  const evictions = events.filter(e=>e.event_type === "eviction").length;
  const gpu = events.filter(e=>["gpu_export","cpu_offload","gpu_import","l2_restore","l1_hit","l2_hit"].includes(e.event_type)).length;
  const promotions = events.filter(e=>e.event_type === "promotion").length;
  const metrics = [["Requests",s.requests||0],["Chunks",s.total_chunks||0],["Reused",s.matched_chunks||0],["Hit rate",((s.hit_rate||0)*100).toFixed(1)+"%"],["Storage events",storage],["Evictions",evictions],["Tier promotions",promotions],["GPU events",gpu]];
  $("summary").innerHTML = metrics.map(x=>`<div class="metric"><span class="muted">${{x[0]}}</span><b>${{x[1]}}</b></div>`).join("");
}}
function currentEvent() {{ return step >= 0 ? events[step] : null; }}
function escapeText(value) {{
  return String(value ?? "").replace(/[&<>\"']/g, function(char) {{
    if (char === "&") return "&amp;";
    if (char === "<") return "&lt;";
    if (char === ">") return "&gt;";
    if (char === '\"') return "&quot;";
    return "&#39;";
  }});
}}
function renderRequests() {{
  const e = currentEvent();
  $("requests").innerHTML = requests.map(r => {{
    const active = e && e.request_id === r.request_id;
    const chunks = (r.statuses || []).map((s,i)=>`<span class="chunk ${{escapeText(statusClass(s))}} ${{active && e.chunk_index === i ? "active":""}}"><b>${{i}}</b> · ${{escapeText(s)}}</span>`).join("");
    return `<div class="request"><div class="request-title"><b>${{escapeText(r.request_id)}}</b><span class="muted">${{r.matched_chunks}}/${{r.total_chunks}} chunks</span></div><div class="chunks">${{chunks || '<span class="muted">no chunks</span>'}}</div></div>`;
  }}).join("");
}}
function renderTimeline() {{
  $("timeline").innerHTML = events.map((e,i)=>`<div class="event ${{i===step?"current":""}}" data-index="${{i}}"><div class="event-name">${{i}} · ${{escapeText(e.event_type)}}</div><div class="event-detail">request=${{escapeText(e.request_id)}}${{e.chunk_index == null ? "":" · chunk="+e.chunk_index}}${{e.result == null ? "":" · "+escapeText(e.result)}}</div></div>`).join("");
  document.querySelectorAll(".event").forEach(n=>n.onclick=()=>{{ step=Number(n.dataset.index); render(); }});
}}
function renderTree() {{
  const nodes = new Map();
  events.forEach(e=>{{ if(e.key) {{
    const previous = nodes.get(e.key);
    nodes.set(e.key, {{key:e.key,parent:e.parent_key ?? (previous && previous.parent),active:(previous && previous.active) || e===currentEvent()}});
  }} }});
  const list = [...nodes.values()]; if(!list.length) {{ $("tree").innerHTML='<span class="muted">No key events</span>'; return; }}
  const depth = n => n.parent && nodes.has(n.parent) ? depth(nodes.get(n.parent))+1 : 0;
  list.forEach(n=>n.depth=depth(n)); const max=Math.max(...list.map(n=>n.depth)); const width=Math.max(500,(max+1)*180), byDepth={{}};
  list.forEach(n=>(byDepth[n.depth] ||= []).push(n)); list.forEach(n=>n.x=(byDepth[n.depth].indexOf(n)+1)*width/(byDepth[n.depth].length+1));
  const edges=list.filter(n=>n.parent&&nodes.has(n.parent)).map(n=>`<line class="tree-edge" x1="${{nodes.get(n.parent).x}}" y1="${{70+nodes.get(n.parent).depth*70}}" x2="${{n.x}}" y2="${{70+n.depth*70}}"/>`).join("");
  const labels=list.map(n=>`<g><circle class="tree-node ${{n.active?"current":""}}" cx="${{n.x}}" cy="${{70+n.depth*70}}" r="20"/><text class="tree-label" x="${{n.x}}" y="${{74+n.depth*70}}">${{escapeText(n.key.slice(0,6))}}</text></g>`).join("");
  $("tree").innerHTML=`<svg viewBox="0 0 ${{width}} ${{90+max*70}}">${{edges}}${{labels}}</svg>`;
}}
function render() {{
  $("step-label").textContent = events.length ? `Step ${{Math.max(0,step+1)}} / ${{events.length}}` : "No events";
  const e=currentEvent(); $("detail").textContent=e?JSON.stringify(e,null,2):"Select Next or click an event.";
  renderRequests(); renderTimeline(); renderTree();
}}
$("reset").onclick=()=>{{step=-1;render();}}; $("prev").onclick=()=>{{step=Math.max(-1,step-1);render();}}; $("next").onclick=()=>{{step=Math.min(events.length-1,step+1);render();}};
$("play").onclick=()=>{{ if(timer) return; timer=setInterval(()=>{{ if(step>=events.length-1){{clearInterval(timer);timer=null;return;}} step++;render(); }},700); }};
$("pause").onclick=()=>{{clearInterval(timer);timer=null;}}; renderSummary(); render();
</script>
</body></html>"""


def write_html(
    path: str | Path,
    report: ReplayReport | Mapping[str, Any],
    *,
    title: str = "nano-lmcache Trace",
) -> Path:
    """Write an offline visualizer and return its path."""
    output = Path(path)
    output.write_text(render_html(report, title=title), encoding="utf-8")
    return output
