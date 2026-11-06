"""Generate an offline HTML trace viewer using only the Python standard library.

The page itself lives in ``viewer.html`` as plain HTML/CSS/JavaScript, so it
can be read and edited without Python string escaping. This module only
embeds the report JSON and the title into that template.
"""

import html
import json
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

from nano_lmcache.replay.runner import ReplayReport

_TEMPLATE = files(__package__).joinpath("viewer.html")


def render_html(
    report: ReplayReport | Mapping[str, Any], *, title: str = "nano-lmcache Trace"
) -> str:
    """Return a standalone HTML visualizer for a replay report."""
    payload = report.to_dict() if isinstance(report, ReplayReport) else dict(report)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Prevent a JSON value from closing the script element when requests come
    # from an untrusted dataset.
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e")
    # Split at the data slot first so neither value can be mistaken for the
    # other's placeholder.
    head, tail = _TEMPLATE.read_text(encoding="utf-8").split("__DATA__")
    safe_title = html.escape(title)
    return (
        head.replace("__TITLE__", safe_title)
        + encoded
        + tail.replace("__TITLE__", safe_title)
    )


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
