"""Polite, cached HTTP.

Every response is stored under cache/http/. Historical data (a finished F1
season, an old page of LoL results) never changes, so it is cached forever;
"live" data gets a short time-to-live. Re-running the pipeline therefore
costs almost no requests, which keeps us inside the free APIs' rate limits.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests

from . import config

_session = requests.Session()
_session.headers["User-Agent"] = config.USER_AGENT
_last_call: dict[str, float] = {}

# Minimum seconds between calls to the same host. Jolpica allows ~4 req/s
# burst and 500 req/hour sustained.
MIN_INTERVAL = {"api.jolpi.ca": 0.35, "esports-api.lolesports.com": 0.2,
                "api.elections.kalshi.com": 0.15}


class FetchError(RuntimeError):
    pass


def _cache_path(url: str, params: dict | None) -> Path:
    key = url + ("?" + urlencode(sorted((params or {}).items())) if params else "")
    h = hashlib.sha1(key.encode()).hexdigest()[:20]
    host = urlparse(url).netloc.replace(":", "_")
    return config.CACHE / "http" / host / f"{h}.json"


def _throttle(host: str) -> None:
    gap = MIN_INTERVAL.get(host, 0.1)
    wait = _last_call.get(host, 0) + gap - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.monotonic()


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             ttl: float | None = 3600, retries: int = 5) -> dict:
    """GET a JSON document, using the disk cache when it is fresh enough.

    ttl=None means "cache forever" (use for immutable history).
    ttl=0 means "always refetch".
    """
    path = _cache_path(url, params)
    if path.exists() and ttl != 0:
        age = time.time() - path.stat().st_mtime
        if ttl is None or age < ttl:
            return json.loads(path.read_text())

    host = urlparse(url).netloc
    delay = 2.0
    for attempt in range(retries):
        _throttle(host)
        try:
            r = _session.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            err = e
        else:
            if r.status_code == 200:
                data = r.json()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
                return data
            if r.status_code in (429, 500, 502, 503, 504):
                err = FetchError(f"{r.status_code} from {url}")
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            else:
                raise FetchError(f"{r.status_code} from {r.url}: {r.text[:200]}")
        time.sleep(delay)
        delay = min(delay * 2, 60)
    # Network down: serve a stale copy rather than failing the whole run.
    if path.exists():
        print(f"  ! using stale cache for {url} ({err})")
        return json.loads(path.read_text())
    raise FetchError(f"giving up on {url}: {err}")
