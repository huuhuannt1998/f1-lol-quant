"""LoL end to end: fetch -> ratings -> backtest -> forecast -> market -> page."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, kalshi, ledger
from ..util import has_lightgbm, iso, name_similarity, nonfinite_counts, utcnow
from . import lolesports, oracles_elixir as oe
from .elo import team_key
from .model import (BASE_FEATURES, MatchModel, add_oe_diffs, evaluate,
                    historical_features, tbd_matches, tune_elo, upcoming_features, walk_forward)


def _f(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def match_kalshi(up: pd.DataFrame, events: list[dict]) -> dict[int, dict]:
    """Pair Kalshi match events with schedule rows by start time and names."""
    out = {}
    if up.empty:
        return out
    starts = pd.to_datetime(up["start_time"], utc=True)
    for ev in events:
        names = [s.strip() for s in (ev["title"] or "").split(" vs. ")]
        if len(names) != 2:
            continue
        cand = up.index if ev["start"] is None else up.index[
            (starts - pd.Timestamp(ev["start"])).abs() <= pd.Timedelta(hours=6)]
        best, best_s = None, 1.3
        for i in cand:
            a, b = up.at[i, "team1"], up.at[i, "team2"]
            s = max(name_similarity(a, names[0]) + name_similarity(b, names[1]),
                    name_similarity(a, names[1]) + name_similarity(b, names[0]))
            if s > best_s:
                best, best_s = i, s
        if best is None:
            continue
        q1 = max(ev["quotes"], key=lambda q: name_similarity(q["yes_name"], up.at[best, "team1"]))
        q2 = max(ev["quotes"], key=lambda q: name_similarity(q["yes_name"], up.at[best, "team2"]))
        if q1 is not q2:
            out[best] = {"event_ticker": ev["event_ticker"], "t1": q1, "t2": q2}
    return out


def eval_window_start(matches: pd.DataFrame, now: pd.Timestamp) -> pd.Timestamp:
    """Backtest the last year, but leave at least 8 months of history first."""
    t = pd.to_datetime(matches["start_time"], utc=True)
    tmin = t[matches["state"] == "completed"].min()
    return max(now - pd.Timedelta(days=365), tmin + pd.Timedelta(days=240))


def power_rankings(book, now: pd.Timestamp, exclude_tier2: bool = True) -> tuple[list, int]:
    """Active teams by Elo. Tier-2 teams (by their most recent domestic league)
    are left out: their ratings come from a closed pool and are not on the
    same scale as the majors'. Returns (rankings, number of tier-2 teams left out)."""
    cutoff = now - pd.Timedelta(days=150)
    month_ago = now - pd.Timedelta(days=30)
    ranks, n_tier2 = [], 0
    for t, rating in book.rating.items():
        hist = book.history.get(t) or []
        if book.games[t] < 10 or not hist or pd.Timestamp(hist[-1][0]) < cutoff:
            continue
        league = book.last_league.get(t) or book.home_league(t)
        if exclude_tier2 and lolesports.is_tier2(league, now.year):
            n_tier2 += 1
            continue
        before = [h for h in hist if pd.Timestamp(h[0]) < month_ago]
        ranks.append({"team": book.display.get(t, t), "league": league,
                      "elo": round(rating), "games": int(book.games[t]),
                      "change_30d": round(rating - before[-1][1]) if before else None})
    ranks.sort(key=lambda x: -x["elo"])
    return ranks, n_tier2


def run(fetch: bool = True, use_oe: bool = True) -> dict:
    print("LoL")
    now = pd.Timestamp.now(tz="UTC")
    matches = lolesports.prepare(lolesports.update() if fetch else lolesports.load_store(), now)
    stale = matches["state"] == "stale"
    print(f"  stale (not completed {config.LOL_STALE_HOURS}h after start, ignored): {int(stale.sum())}")
    stale_ids = set(matches.loc[stale, "match_id"].astype(str))
    # Outcomes come from every completed match, taken before any tier-2 drop,
    # so forecasts already in the ledger for dropped leagues still settle.
    settled_all = matches[(matches["state"] == "completed") & matches["winner"].notna()]
    if config.LOL_DROP_TIER2:
        t2 = lolesports.tier2_mask(matches)
        print(f"  LOL_DROP_TIER2: dropping {int(t2.sum())} tier-2 matches")
        matches = matches[~t2].reset_index(drop=True)

    form, oe_info = None, {"files": 0}
    if use_oe:
        if fetch:
            oe.download()
        games = oe.load_team_games()
        if not games.empty:
            form = oe.rolling_team_form(games)
            oe_info = {"files": len(list(oe.oe_dir().glob('*.csv'))), "team_games": int(len(games)),
                       "latest_game": iso(games["date"].max().to_pydatetime()),
                       "unmatched_sample": oe.unmatched_teams(matches, games)}
            print(f"  OE: {oe_info['team_games']} team-games; lolesports names not found in OE: "
                  f"{oe_info['unmatched_sample'] or 'none'}")

    t_all = pd.to_datetime(matches["start_time"], utc=True)
    done_mask = matches["state"] == "completed"
    tmin = t_all[done_mask].min()
    eval_start = eval_window_start(matches, now)

    tuned = tune_elo(matches, until=eval_start)
    print(f"  elo tuned: {tuned}")
    feats, book, ft = historical_features(matches, k=tuned["k"], regress=tuned["regress"])
    bad = nonfinite_counts(feats, BASE_FEATURES)
    print(f"  features: {'all finite' if not bad else f'NON-FINITE {bad}'} "
          f"({len(feats)} matches x {len(BASE_FEATURES)} base features)")
    feats = oe.attach(feats, form)
    oe_cols = add_oe_diffs(feats)
    if oe_cols:
        k1 = feats["team1_oe_games"].notna().mean() if "team1_oe_games" in feats else 0
        oe_info["match_coverage"] = round(float(k1), 3)
        print(f"  OE: match coverage {oe_info['match_coverage']:.1%} (team1 has OE history)")
    kinds = {"elo": [], "logreg": BASE_FEATURES, "gbm": BASE_FEATURES + oe_cols}
    if not has_lightgbm():
        print("  ! lightgbm unavailable (macOS: brew install libomp); skipping gbm")
        kinds.pop("gbm")

    bt = walk_forward(feats, kinds, eval_start)
    metrics = evaluate(bt, list(kinds)) if len(bt) else {}
    chosen = "elo"
    if metrics:
        ok = {k: v["all"]["log_loss"] for k, v in metrics.items() if v["all"].get("n", 0) >= 150}
        chosen = min(ok, key=ok.get) if ok else "elo"
    print(f"  backtest n={len(bt)}; chosen model: {chosen}")
    model = MatchModel(chosen, kinds[chosen]).fit(feats)

    # ---- forecasts for upcoming matches
    up = upcoming_features(matches, book, ft)
    if len(up):
        up = oe.attach(up, form)
        add_oe_diffs(up)
        for c in kinds[chosen]:
            if c not in up:
                up[c] = np.nan
        up["p1"] = model.predict(up)
    try:
        events = kalshi.lol_match_events(status="open")
    except Exception as e:  # noqa: BLE001 - market data is optional
        print(f"  ! kalshi unavailable: {e}")
        events = []
    km = match_kalshi(up, events)
    print(f"  upcoming {len(up)}, kalshi events {len(events)}, matched {len(km)}")

    pred_rows, upcoming = [], []
    for i, r in up.iterrows():
        k = km.get(i, {})
        q1, q2 = k.get("t1") or {}, k.get("t2") or {}
        item = {"id": r.match_id, "start": r.start_time, "state": r.state,
                "league": r.league_name, "intl": bool(r.intl), "block": r.block,
                "best_of": int(r.best_of), "team1": r.team1, "team2": r.team2,
                "code1": r.team1_code, "code2": r.team2_code,
                "elo1": _f(r.r1_pre, 0), "elo2": _f(r.r2_pre, 0),
                "p1": _f(r.p1), "p1_elo": _f(r.p_series_elo),
                "kalshi": None}
        if k:
            item["kalshi"] = {"event": k["event_ticker"],
                              "t1": {x: _f(q1.get(x)) for x in ("yes_bid", "yes_ask", "mid", "volume")},
                              "t2": {x: _f(q2.get(x)) for x in ("yes_bid", "yes_ask", "mid", "volume")}}
        upcoming.append(item)
        if r.state == "unstarted":
            for side, team, p, q in ((1, r.team1, r.p1, q1), (2, r.team2, 1 - r.p1, q2)):
                pred_rows.append({"sport": "lol", "event_id": r.match_id,
                                  "event_name": f"{r.team1} vs {r.team2} ({r.league_name})",
                                  "event_start": r.start_time, "stage": "pre_match",
                                  "selection": team, "selection_id": team_key(team),
                                  "p_model": p, "model": f"{chosen}-{config.MODEL_VERSION}",
                                  "kalshi_ticker": q.get("ticker"), "bid": q.get("yes_bid"),
                                  "ask": q.get("yes_ask"), "mid": q.get("mid"),
                                  "volume": q.get("volume")})
    n_logged = ledger.log_predictions(pred_rows)

    # Slots whose teams are not decided yet (later Worlds Swiss rounds,
    # bracket matches): listed on the page, never forecast or logged.
    tbd = tbd_matches(matches)
    for r in tbd.itertuples(index=False):
        upcoming.append({"id": r.match_id, "start": r.start_time, "state": r.state,
                         "league": r.league_name, "intl": r.league in lolesports.INTERNATIONAL,
                         "block": r.block, "best_of": int(r.best_of), "tbd": True,
                         "team1": r.team1 if isinstance(r.team1, str) and r.team1 else "TBD",
                         "team2": r.team2 if isinstance(r.team2, str) and r.team2 else "TBD",
                         "code1": None, "code2": None, "elo1": None, "elo2": None,
                         "p1": None, "p1_elo": None, "kalshi": None})
    upcoming.sort(key=lambda m: str(m["start"]))
    print(f"  scheduled with teams still TBD (shown, not forecast): {len(tbd)}")

    # ---- outcomes of finished matches feed the ledger
    recent_done = settled_all[pd.to_datetime(settled_all["start_time"], utc=True) > now - pd.Timedelta(days=45)]
    res_rows = []
    for r in recent_done.itertuples(index=False):
        res_rows += [{"sport": "lol", "event_id": r.match_id, "selection_id": team_key(r.team1),
                      "outcome": int(r.winner == 1)},
                     {"sport": "lol", "event_id": r.match_id, "selection_id": team_key(r.team2),
                      "outcome": int(r.winner == 2)}]
    n_res = ledger.record_results(res_rows)
    print(f"  ledger: +{n_logged} forecasts, +{n_res} outcomes")

    # ---- recent results with the pre-match Elo view (replayed, point-in-time)
    rec = feats[pd.to_datetime(feats["start_time"], utc=True) > now - pd.Timedelta(days=10)]
    results = [{"id": r.match_id, "start": r.start_time, "league": r.league_name,
                "best_of": int(r.best_of), "team1": r.team1, "team2": r.team2,
                "score1": None if pd.isna(r.t1_wins) else int(r.t1_wins),
                "score2": None if pd.isna(r.t2_wins) else int(r.t2_wins),
                "winner": int(r.winner), "p1_pre": _f(r.p_series_elo)}
               for r in rec.sort_values("start_time", ascending=False).itertuples(index=False)]

    # ---- power rankings (majors only; see power_rankings)
    ranks, n_tier2 = power_rankings(book, now)

    league_counts = matches[done_mask].groupby("league_name").size().sort_values(ascending=False)
    return {
        "generated_at": iso(utcnow()),
        "model": {"chosen": chosen, "elo": tuned, "features": kinds[chosen]},
        "upcoming": upcoming, "results": results, "rankings": ranks[:60],
        "backtest": {"from": iso(eval_start.to_pydatetime()), "n": int(len(bt)),
                     "metrics": {k: {s: v for s, v in m.items()
                                     if not s.startswith("calibration") or k == chosen}
                                 for k, m in metrics.items()}},
        "coverage": {"matches": int(done_mask.sum()),
                     "first": iso(tmin.to_pydatetime()) if pd.notna(tmin) else None,
                     "leagues": {k: int(v) for k, v in league_counts.head(25).items()},
                     "stale": len(stale_ids), "tier2_unranked": n_tier2,
                     "tier2_dropped": bool(config.LOL_DROP_TIER2),
                     "oracles_elixir": oe_info},
        "ledger": ledger.scorecard("lol", ignore=stale_ids),
        "ledger_recent": ledger.recent_rows("lol", ignore=stale_ids),
    }
