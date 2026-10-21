"""Read-only Kalshi market data (public endpoints, no API key needed).

We only *read* prices so the model can be compared with the market. Nothing
here places orders.
"""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from . import config
from .http import get_json
from .util import parse_ts

ET = ZoneInfo("America/New_York")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _dollars(m: dict, key: str) -> float | None:
    """Kalshi now reports prices as dollar strings ('0.8100'); older payloads
    used integer cents. Accept either."""
    v = m.get(f"{key}_dollars")
    if v not in (None, ""):
        return float(v)
    v = m.get(key)
    return None if v is None else float(v) / 100


def _fp(m: dict, key: str) -> float:
    v = m.get(f"{key}_fp", m.get(key))
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def market_quote(m: dict) -> dict:
    bid, ask, last = _dollars(m, "yes_bid"), _dollars(m, "yes_ask"), _dollars(m, "last_price")
    # A one-sided or empty book makes the mid meaningless; fall back to last.
    if bid and ask and 0 < bid < ask <= 1 and ask - bid <= 0.25:
        mid = (bid + ask) / 2
    else:
        mid = last
    return {"ticker": m.get("ticker"), "yes_bid": bid, "yes_ask": ask, "last": last,
            "mid": mid, "volume": _fp(m, "volume"), "open_interest": _fp(m, "open_interest"),
            "status": m.get("status"), "result": m.get("result") or None,
            "yes_name": m.get("yes_sub_title") or m.get("title"),
            "close_time": m.get("close_time")}


def get_events(series: str, status: str | None = "open", ttl: float = 300) -> list[dict]:
    """All events of a series with their markets nested inside."""
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "limit": 200, "with_nested_markets": "true"}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        page = get_json(f"{config.KALSHI_API}/events", params=params, ttl=ttl)
        out.extend(page.get("events") or [])
        cursor = page.get("cursor")
        if not cursor or not page.get("events"):
            return out


def ticker_start_time(event_ticker: str) -> datetime | None:
    """KXLOLGAME-26OCT161900C9FUR -> 2026-10-16 19:00 America/New_York."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})", event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    yy, mon, dd, hh, mi = m.groups()
    try:
        return datetime(2000 + int(yy), _MONTHS[mon], int(dd), int(hh), int(mi), tzinfo=ET)
    except ValueError:
        return None


def lol_match_events(status: str | None = "open") -> list[dict]:
    """Normalized KXLOLGAME events: one per match, with a quote per team."""
    rows = []
    for ev in get_events(config.KALSHI_LOL_SERIES, status=status):
        quotes = [market_quote(m) for m in ev.get("markets") or []]
        if len(quotes) < 2:
            continue
        rows.append({"event_ticker": ev["event_ticker"], "title": ev.get("title"),
                     "start": ticker_start_time(ev["event_ticker"]),
                     "quotes": quotes})
    return rows


def f1_race_events(status: str | None = "open") -> list[dict]:
    """Normalized KXF1RACE events. Market tickers end in the driver's
    three-letter code (KXF1RACE-SIN26-VER), which matches Jolpica's `code`."""
    rows = []
    for ev in get_events(config.KALSHI_F1_SERIES, status=status):
        quotes = []
        for m in ev.get("markets") or []:
            q = market_quote(m)
            q["driver_code"] = (m.get("ticker") or "").rsplit("-", 1)[-1]
            quotes.append(q)
        when = None
        for m in ev.get("markets") or []:
            when = parse_ts(m.get("occurrence_datetime") or m.get("expected_expiration_time"))
            if when:
                break
        rows.append({"event_ticker": ev["event_ticker"], "title": ev.get("title"),
                     "start": when, "quotes": quotes})
    return rows
