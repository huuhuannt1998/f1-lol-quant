"""F1 end to end: fetch -> features -> backtest -> next-race forecast -> page."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, kalshi, ledger, picks
from ..util import has_lightgbm, iso, nonfinite_counts, utcnow
from . import jolpica
from .model import STAGES, evaluate, make_model, plackett_luce, walk_forward

KINDS = ("baseline", "clogit", "gbm") if has_lightgbm() else ("baseline", "clogit")


def _f(x, nd=4):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def next_race(races: pd.DataFrame, results: pd.DataFrame, now: pd.Timestamp):
    done = set(zip(results["season"], results["round"]))
    r = races.copy()
    r["race_dt"] = pd.to_datetime(r["race_time"], utc=True)
    r = r[[(s, n) not in done for s, n in zip(r["season"], r["round"])]]
    r = r[r["race_dt"] > now - pd.Timedelta(hours=6)].sort_values("race_dt")
    return None if r.empty else r.iloc[0]


def run(fetch: bool = True) -> dict:
    print("F1")
    data = jolpica.update() if fetch else jolpica.load()
    races, results, quali = data["races"], data["results"], data["qualifying"]
    now = pd.Timestamp.now(tz="UTC")
    season = int(races["season"].max())
    nxt = next_race(races, results, now)
    up_key = (int(nxt["season"]), int(nxt["round"])) if nxt is not None else None

    from .model import build_entries
    ent = build_entries(races, results, quali, upcoming=up_key)
    bad = nonfinite_counts(ent, STAGES["post_quali"])
    print(f"  features: {'all finite' if not bad else f'NON-FINITE {bad}'} "
          f"({len(ent)} driver-races x {len(STAGES['post_quali'])} features)")

    first_test = max(int(results["season"].min()) + 3, season - 4)
    bt = walk_forward(ent, first_test, kinds=KINDS)
    metrics = evaluate(bt, KINDS)
    chosen = {}
    for stage, m in metrics.items():
        recent = {k: np.mean([v for s, v in m[k]["by_season"].items() if s >= season - 2] or [9])
                  for k in KINDS}
        chosen[stage] = min(recent, key=recent.get)
    print(f"  backtest races={bt['race_key'].nunique() if len(bt) else 0}; chosen {chosen}")

    out = {"generated_at": iso(utcnow()), "season": season, "next_race": None,
           "backtest": {"from_season": first_test, "chosen": chosen,
                        "metrics": {st: {k: {x: y for x, y in v.items() if x != "calibration" or k == chosen.get(st)}
                                         for k, v in m.items()} for st, m in metrics.items()}}}

    # ---- next race forecast
    done = ent[ent["win"].notna()]
    if nxt is not None:
        field = ent[(ent["season"] == up_key[0]) & (ent["round"] == up_key[1])].copy()
        stage = "post_quali" if field["quali_pos"].notna().sum() >= len(field) / 2 else "pre_quali"
        kind = chosen.get(stage, "clogit")
        m = make_model(kind, stage).fit(done)
        field["p_win"] = m.predict(field)
        sims = plackett_luce(field["p_win"].to_numpy())
        for k, v in sims.items():
            field[k] = v
        coefs = m.coefficients() if hasattr(m, "coefficients") else None

        try:
            evs = kalshi.f1_race_events(status="open")
        except Exception as e:  # noqa: BLE001
            print(f"  ! kalshi unavailable: {e}")
            evs = []
        ev = None
        for e in evs:
            if e["start"] is not None and abs(pd.Timestamp(e["start"]) - pd.Timestamp(nxt["race_dt"])) < pd.Timedelta(days=3):
                ev = e
        quotes = {q["driver_code"]: q for q in (ev["quotes"] if ev else [])}

        drivers, pred_rows = [], []
        for r in field.sort_values("p_win", ascending=False).itertuples(index=False):
            q = quotes.get(r.driver_code, {})
            drivers.append({"code": r.driver_code, "name": r.driver, "team": r.constructor, "id": r.driver_id,
                            "quali": None if pd.isna(r.quali_pos) else int(r.quali_pos),
                            "p_win": _f(r.p_win), "p_podium": _f(r.p_podium),
                            "p_points": _f(r.p_points), "exp_pos": _f(r.exp_pos, 1),
                            "kalshi": {x: _f(q.get(x)) for x in ("yes_bid", "yes_ask", "mid", "volume")} if q else None})
            pred_rows.append({"sport": "f1", "event_id": f"{up_key[0]}-{up_key[1]:02d}",
                              "event_name": f'{up_key[0]} {nxt["race_name"]}',
                              "event_start": nxt["race_time"], "stage": stage,
                              "selection": r.driver, "selection_id": r.driver_id,
                              "p_model": r.p_win, "model": f"{kind}-{config.MODEL_VERSION}",
                              "kalshi_ticker": q.get("ticker"), "bid": q.get("yes_bid"),
                              "ask": q.get("yes_ask"), "mid": q.get("mid"), "volume": q.get("volume")})
        n_logged = ledger.log_predictions(pred_rows)
        out["next_race"] = {"season": up_key[0], "round": up_key[1], "name": nxt["race_name"],
                            "circuit": nxt["circuit"], "country": nxt["country"],
                            "race_time": nxt["race_time"], "quali_time": nxt["quali_time"],
                            "sprint_time": nxt["sprint_time"] if pd.notna(nxt["sprint_time"]) else None,
                            "stage": stage, "model": kind, "coefficients": coefs,
                            "kalshi_event": ev["event_ticker"] if ev else None, "drivers": drivers}
        print(f"  next: {nxt['race_name']} ({stage}, {kind}); kalshi {'yes' if ev else 'no'}; +{n_logged} forecasts")

    # ---- outcomes into the ledger (current season)
    cur = results[results["season"] == season]
    ledger.record_results([{"sport": "f1", "event_id": f"{s}-{n:02d}", "selection_id": d,
                            "outcome": int(p == 1)}
                           for s, n, d, p in cur[["season", "round", "driver_id", "position"]].itertuples(index=False)])

    # ---- last race: what happened vs what the model said beforehand
    if len(cur):
        lr = int(cur["round"].max())
        race = races[(races["season"] == season) & (races["round"] == lr)].iloc[0]
        rr = cur[cur["round"] == lr].sort_values("position")
        pre = bt[(bt["race_key"] == season * 100 + lr) & (bt["stage"] == "post_quali")]
        pmap = dict(zip(pre["driver_id"], pre[f"p_{chosen.get('post_quali', 'clogit')}"])) if len(pre) else {}
        out["last_race"] = {"name": race["race_name"], "round": lr, "race_time": race["race_time"],
                            "results": [{"pos": int(x.position), "code": x.driver_code, "name": x.driver,
                                         "team": x.constructor, "grid": int(x.grid), "status": x.status,
                                         "points": x.points, "p_win_pre": _f(pmap.get(x.driver_id))}
                                        for x in rr.head(20).itertuples(index=False)]}

    # ---- calendar + standings
    winners = results[results["position"] == 1].set_index(["season", "round"])["driver"]
    cal = races[races["season"] == season].sort_values("round")
    out["calendar"] = [{"round": int(r.round), "name": r.race_name, "country": r.country,
                        "race_time": r.race_time, "sprint": isinstance(r.sprint_time, str),
                        "winner": winners.get((season, r.round))} for r in cal.itertuples(index=False)]
    try:
        out["standings"] = jolpica.standings(season) if fetch else {}
    except Exception as e:  # noqa: BLE001
        print(f"  ! standings unavailable: {e}")
        out["standings"] = {}
    if not out["standings"] and len(cur):
        # Offline fallback from race results (misses sprint points).
        d = (cur.groupby(["driver_code", "driver"])
                .agg(points=("points", "sum"), wins=("position", lambda s: int((s == 1).sum())),
                     team=("constructor", "last")).reset_index().sort_values("points", ascending=False))
        c = (cur.groupby("constructor").agg(points=("points", "sum"),
                                            wins=("position", lambda s: int((s == 1).sum())))
                .reset_index().sort_values("points", ascending=False))
        out["standings"] = {
            "drivers": [{"pos": i, "points": r.points, "wins": r.wins, "code": r.driver_code,
                         "name": r.driver, "team": r.team} for i, r in enumerate(d.itertuples(), 1)],
            "constructors": [{"pos": i, "points": r.points, "wins": r.wins, "name": r.constructor}
                             for i, r in enumerate(c.itertuples(), 1)]}
    out["ledger"] = ledger.scorecard("f1")
    out["ledger_recent"] = ledger.recent_rows("f1")
    out["my_picks"] = picks.summary("f1")
    return out
