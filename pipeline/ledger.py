"""The forecast ledger: the only honest scoreboard.

Backtests can be fooled (leakage, over-tuning); a forecast written down
*before* an event and scored afterwards cannot. Rules:

  * Rows are only ever appended. Nothing is edited or deleted.
  * A forecast is accepted only if it is logged before the event starts.
  * The forecast that counts is the last one before the start ("closing"),
    scored next to the Kalshi price at that same moment.

Paper trading uses those closing rows: if the model's probability beats the
ask by PAPER_EDGE_MIN after fees, we pretend to buy one $1 contract.
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from . import config
from .util import calibration_bins, iso, kalshi_fee, summarize, utcnow

PRED_COLS = ["logged_at", "sport", "event_id", "event_name", "event_start", "stage",
             "selection", "selection_id", "p_model", "model", "kalshi_ticker",
             "bid", "ask", "mid", "volume"]
RES_COLS = ["sport", "event_id", "selection_id", "outcome", "recorded_at"]


def _path(name):
    return config.LEDGER_DIR / name


def _read(name, cols) -> pd.DataFrame:
    p = _path(name)
    if p.exists() and p.stat().st_size:
        return pd.read_csv(p, dtype={"event_id": str, "selection_id": str})
    return pd.DataFrame(columns=cols)


def _append(name, cols, rows: list[dict]) -> None:
    if not rows:
        return
    p = _path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def _same(a, b, tol=0.005) -> bool:
    if (a is None or pd.isna(a)) and (b is None or pd.isna(b)):
        return True
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < tol


def log_predictions(rows: list[dict]) -> int:
    """Append forecasts for events that have not started. Skips a row when
    neither the model nor the market moved since the last logged snapshot."""
    now = utcnow()
    old = _read("predictions.csv", PRED_COLS)
    last = {}
    if len(old):
        for r in old.itertuples(index=False):
            last[(r.sport, str(r.event_id), r.stage, str(r.selection_id))] = r
    keep = []
    for r in rows:
        start = pd.Timestamp(r["event_start"])
        if start <= now:
            continue
        k = (r["sport"], str(r["event_id"]), r["stage"], str(r["selection_id"]))
        prev = last.get(k)
        if prev is not None and _same(prev.p_model, r["p_model"]) and _same(prev.mid, r.get("mid")):
            continue
        r = {**r, "logged_at": iso(now), "event_start": iso(start.to_pydatetime()),
             "p_model": round(float(r["p_model"]), 4)}
        keep.append(r)
    _append("predictions.csv", PRED_COLS, keep)
    return len(keep)


def record_results(rows: list[dict]) -> int:
    old = _read("results.csv", RES_COLS)
    seen = set(zip(old["sport"], old["event_id"].astype(str), old["selection_id"].astype(str)))
    new = [{**r, "recorded_at": iso(utcnow())} for r in rows
           if (r["sport"], str(r["event_id"]), str(r["selection_id"])) not in seen]
    _append("results.csv", RES_COLS, new)
    return len(new)


def closing(sport: str | None = None) -> pd.DataFrame:
    """Last pre-start forecast per (event, stage, selection), with outcome."""
    p = _read("predictions.csv", PRED_COLS)
    if sport:
        p = p[p["sport"] == sport]
    if p.empty:
        return p
    p = p[pd.to_datetime(p["logged_at"], utc=True) < pd.to_datetime(p["event_start"], utc=True)]
    p = p.sort_values("logged_at").groupby(["sport", "event_id", "stage", "selection_id"],
                                           as_index=False).tail(1)
    res = _read("results.csv", RES_COLS)[["sport", "event_id", "selection_id", "outcome"]]
    return p.merge(res, on=["sport", "event_id", "selection_id"], how="left")


def paper_trades(c: pd.DataFrame, allow_no: bool) -> pd.DataFrame:
    trades = []
    for r in c.itertuples(index=False):
        if pd.isna(r.outcome):
            continue
        p = float(r.p_model)
        if pd.notna(r.ask) and 0 < r.ask < 1:
            fee = kalshi_fee(r.ask, config.KALSHI_FEE_RATE)
            if p - r.ask - fee >= config.PAPER_EDGE_MIN:
                trades.append({"event_start": r.event_start, "event_name": r.event_name,
                               "side": "YES", "selection": r.selection, "price": r.ask,
                               "p_model": p, "pnl": float(r.outcome) - r.ask - fee})
                continue
        if allow_no and pd.notna(r.bid) and 0 < r.bid < 1:
            no_price = 1 - r.bid
            fee = kalshi_fee(no_price, config.KALSHI_FEE_RATE)
            if (1 - p) - no_price - fee >= config.PAPER_EDGE_MIN:
                trades.append({"event_start": r.event_start, "event_name": r.event_name,
                               "side": "NO", "selection": r.selection, "price": no_price,
                               "p_model": p, "pnl": (1 - float(r.outcome)) - no_price - fee})
    t = pd.DataFrame(trades)
    if len(t):
        t = t.sort_values("event_start")
        t["cum_pnl"] = t["pnl"].cumsum()
    return t


def _drop_ignored(c: pd.DataFrame, ignore) -> pd.DataFrame:
    """Leave out events that will never settle (e.g. stale LoL matches).
    Scoring-only: the ledger files themselves are never touched."""
    if not ignore or c.empty:
        return c
    return c[~c["event_id"].astype(str).isin({str(i) for i in ignore})]


def scorecard(sport: str, ignore=None) -> dict:
    c = _drop_ignored(closing(sport), ignore)
    if c.empty:
        return {"settled": 0, "pending": 0}
    settled = c[c["outcome"].notna()]
    out = {"settled_events": int(settled["event_id"].nunique()),
           "pending_events": int(c.loc[c["outcome"].isna(), "event_id"].nunique()),
           "model": summarize(settled["p_model"], settled["outcome"])}
    both = settled[settled["mid"].notna()]
    if len(both):
        out["head_to_head"] = {"n": int(len(both)),
                               "model": summarize(both["p_model"], both["outcome"]),
                               "market": summarize(both["mid"], both["outcome"])}
        out["calibration_model"] = calibration_bins(both["p_model"], both["outcome"])
        out["calibration_market"] = calibration_bins(both["mid"], both["outcome"])
    t = paper_trades(settled, allow_no=(sport == "f1"))
    out["paper"] = {"trades": int(len(t)),
                    "pnl": round(float(t["pnl"].sum()), 2) if len(t) else 0.0,
                    "roi": round(float(t["pnl"].sum() / t["price"].sum()), 4) if len(t) else None,
                    "hit_rate": round(float((t["pnl"] > 0).mean()), 4) if len(t) else None,
                    "curve": [{"t": r.event_start, "cum": round(r.cum_pnl, 2)}
                              for r in t.itertuples(index=False)] if len(t) else [],
                    "recent": t.tail(15).iloc[::-1].round(4).to_dict("records") if len(t) else []}
    out["edge_min"] = config.PAPER_EDGE_MIN
    return out


def recent_rows(sport: str, n: int = 40, ignore=None) -> list[dict]:
    c = _drop_ignored(closing(sport), ignore)
    if c.empty:
        return []
    c = c.sort_values("event_start", ascending=False).head(n)
    c = c.replace({np.nan: None})
    return c[["event_start", "event_name", "stage", "selection", "p_model", "mid", "outcome"]].to_dict("records")
