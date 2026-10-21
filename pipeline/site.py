"""Write the JSON files that the static page reads."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from . import config
from .util import iso, utcnow


def _clean(o):
    """JSON has no NaN/Infinity; turn them into null."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):           # numpy scalars
        return _clean(o.item())
    return o


def write(name: str, payload: dict, out_dir: Path | None = None) -> Path:
    d = out_dir or config.SITE_DATA
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(_clean(payload), separators=(",", ":"), default=str))
    return p


def write_meta(status: dict, out_dir: Path | None = None) -> Path:
    # "owner/repo" lets the page's "My pick" button open GitHub's editor.
    return write("meta", {"updated_at": iso(utcnow()), "model_version": config.MODEL_VERSION,
                          "status": status, "edge_min": config.PAPER_EDGE_MIN,
                          "fee_rate": config.KALSHI_FEE_RATE,
                          "repo": os.environ.get("GITHUB_REPOSITORY")}, out_dir)
