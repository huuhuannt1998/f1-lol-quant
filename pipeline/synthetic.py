"""Synthetic stand-ins for every data source.

Used by `python -m pipeline demo` and the tests, so the whole pipeline and
page can be exercised with no network. All teams, drivers and prices here
are invented. Because we know the true strengths that generated the data,
the demo also shows what a well-specified model *should* achieve.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from . import config, kalshi
from .ledger import PRED_COLS, RES_COLS
from .util import iso, utcnow

RNG = np.random.default_rng(7)
CITIES = ["Seoul", "Busan", "Shanghai", "Chengdu", "Berlin", "Madrid", "Paris", "Austin",
          "Toronto", "Sao Paulo", "Hanoi", "Taipei", "Tokyo", "Lima", "Oslo", "Lagos",
          "Cairo", "Sydney", "Denver", "Milan", "Kyiv", "Dublin", "Manila", "Osaka"]
BEASTS = ["Falcons", "Tigers", "Wolves", "Dragons", "Comets", "Sharks", "Ravens", "Titans",
          "Foxes", "Owls", "Vipers", "Bulls"]
LEAGUES = {"lck": ("LCK", 10, 1650, 3), "lpl": ("LPL", 12, 1640, 3), "lec": ("LEC", 10, 1540, 1),
           "lcs": ("LCS", 8, 1480, 1), "cblol-brazil": ("CBLOL", 8, 1400, 1),
           # not "vcs": that slug is tier-2 from 2025 and would vanish from the demo's rankings
           "lla": ("LLA", 8, 1420, 3), "lcp": ("LCP", 8, 1450, 3)}


def _p(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def _series(p, bo):
    need, a, b = bo // 2 + 1, 0, 0
    while a < need and b < need:
        if RNG.random() < p:
            a += 1
        else:
            b += 1
    return a, b


# ------------------------------------------------------------------ LoL

def make_lol(now):
    names = iter(RNG.permutation([f"{c} {b}" for c in CITIES for b in BEASTS]))
    teams = []
    for slug, (lname, n, mean, _) in LEAGUES.items():
        for _ in range(n):
            teams.append({"name": next(names), "league": slug, "league_name": lname,
                          "r": RNG.normal(mean, 80)})
    for t in teams:
        t["code"] = "".join(w[0] for w in t["name"].split())[:3].upper() + str(RNG.integers(1, 9))
    rows, oe_rows, mid = [], [], 0
    start = pd.Timestamp(config.LOL_HISTORY_START, tz="UTC") + pd.Timedelta(days=10)

    def play(a, b, when, slug, lname, bo, block, state="completed"):
        nonlocal mid
        mid += 1
        p = _p(a["r"], b["r"])
        row = {"match_id": f"m{mid}", "start_time": iso(when.to_pydatetime()), "state": state,
               "league": slug, "league_name": lname, "block": block, "best_of": bo,
               "team1": a["name"], "team1_code": a["code"], "team2": b["name"], "team2_code": b["code"],
               "t1_wins": None, "t2_wins": None, "winner": None, "true_p": p}
        if state == "completed":
            w1, w2 = _series(p, bo)
            row.update(t1_wins=w1, t2_wins=w2, winner=1 if w1 > w2 else 2)
            for g in range(w1 + w2):
                gid = f"{row['match_id']}g{g}"
                won1 = RNG.random() < p
                gd = RNG.normal((a["r"] - b["r"]) * 4, 1800)
                gt = when + pd.Timedelta(minutes=45 * g)
                for team, res, sgn in ((a, won1, 1), (b, not won1, -1)):
                    oe_rows.append({"gameid": gid, "league": lname, "date": gt.strftime("%Y-%m-%d %H:%M:%S"),
                                    "position": "team", "teamname": team["name"], "result": int(res),
                                    "golddiffat15": sgn * gd, "xpdiffat15": sgn * gd * 0.7,
                                    "csdiffat15": sgn * gd / 40, "firstdragon": int(RNG.random() < 0.5 + sgn * (p - 0.5)),
                                    "firsttower": int(RNG.random() < 0.5 + sgn * (p - 0.5)),
                                    "gamelength": RNG.normal(1900, 250), "kills": RNG.poisson(14), "deaths": RNG.poisson(14)})
        rows.append(row)

    week = start
    while week < now - pd.Timedelta(days=2):
        if week.month in (6, 7, 10, 11) and week.day < 8:
            for t in teams:
                t["r"] += RNG.normal(0, 25)          # roster/meta drift
        for slug, (lname, n, _, bo) in LEAGUES.items():
            lt = [t for t in teams if t["league"] == slug]
            order = RNG.permutation(len(lt))
            for i in range(0, len(lt) - 1, 2):
                when = week + pd.Timedelta(days=int(RNG.integers(0, 5)), hours=int(RNG.integers(8, 20)))
                if when < now:
                    play(lt[order[i]], lt[order[i + 1]], when, slug, lname, bo, "Regular Season")
        week += pd.Timedelta(days=7)

    # Worlds: top 3 from each league, a few days in and continuing next week.
    field = []
    for slug in LEAGUES:
        field += sorted([t for t in teams if t["league"] == slug], key=lambda t: -t["r"])[:3]
    w0 = now - pd.Timedelta(days=2)
    for d in range(0, 9):
        day = (w0 + pd.Timedelta(days=d)).normalize() + pd.Timedelta(hours=8)
        perm = RNG.permutation(len(field))
        for j in range(0, 6, 2):
            when = day + pd.Timedelta(hours=3 * j // 2)
            state = "completed" if when < now - pd.Timedelta(hours=2) else "unstarted"
            play(field[perm[j]], field[perm[j + 1]], when, "worlds", "Worlds", 3 if d > 4 else 1,
                 "Swiss Stage", state)
    return pd.DataFrame(rows), pd.DataFrame(oe_rows)


# ------------------------------------------------------------------- F1

TEAMS = ["Aurora", "Kestrel", "Vanguard", "Helix", "Nimbus", "Quasar", "Tempest", "Orion", "Zenith", "Corsa"]
POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def make_f1(now):
    drivers = [{"id": f"drv{i}", "code": "".join(RNG.choice(list("ABCDEFGHKLMNPRSTVZ"), 3)),
                "name": f"Driver {i:02d}", "skill": RNG.normal(0, 0.25)} for i in range(22)]
    pace = {t: RNG.normal(0, 0.6) for t in TEAMS}
    races, results, quali = [], [], []
    for season in range(2019, now.year + 1):
        if season in config.F1_REG_RESET_SEASONS:
            pace = {t: RNG.normal(0, 0.7) for t in TEAMS}
        lineup = RNG.permutation(len(drivers))[:20]
        seat = {drivers[j]["id"]: TEAMS[k // 2] for k, j in enumerate(lineup)}
        n_rounds = 22
        first = pd.Timestamp(f"{season}-03-08T14:00:00Z")
        if season == now.year:
            first = now - pd.Timedelta(days=7 * 19 + 5)
        for rnd in range(1, n_rounds + 1):
            when = first + pd.Timedelta(days=(rnd - 1) * (14 if season < now.year else 7))
            circ = f"circuit_{rnd}"
            races.append({"season": season, "round": rnd, "race_name": f"Grand Prix {rnd}",
                          "circuit_id": circ, "circuit": f"Circuit {rnd}", "country": "Nowhere",
                          "race_time": iso(when.to_pydatetime()),
                          "quali_time": iso((when - pd.Timedelta(days=1)).to_pydatetime()),
                          "sprint_time": None})
            for t in pace:
                pace[t] += RNG.normal(0, 0.05)
            if when > now:
                continue
            perf = {d["id"]: pace[seat[d["id"]]] + d["skill"] for d in drivers if d["id"] in seat}
            q = sorted(perf, key=lambda k: -(perf[k] + RNG.normal(0, 0.25)))
            for pos, did in enumerate(q, 1):
                quali.append({"season": season, "round": rnd, "driver_id": did, "constructor_id": seat[did].lower(),
                              "quali_pos": pos, "best_lap_s": 80 + pos * 0.12 + RNG.normal(0, 0.03)})
            score = {k: perf[k] - 0.04 * q.index(k) + RNG.normal(0, 0.35) for k in perf}
            dnf = {k: RNG.random() < 0.07 for k in perf}
            order = sorted(perf, key=lambda k: (dnf[k], -score[k]))
            for pos, did in enumerate(order, 1):
                d = next(x for x in drivers if x["id"] == did)
                results.append({"season": season, "round": rnd, "driver_id": did, "driver_code": d["code"],
                                "driver": d["name"], "constructor_id": seat[did].lower(), "constructor": seat[did],
                                "grid": q.index(did) + 1, "position": pos, "position_text": "R" if dnf[did] else str(pos),
                                "points": 0.0 if dnf[did] else float(POINTS[pos - 1] if pos <= 10 else 0),
                                "status": "Retired" if dnf[did] else "Finished", "laps": 50, "finished": not dnf[did]})
    return pd.DataFrame(races), pd.DataFrame(results), pd.DataFrame(quali)


# ---------------------------------------------------------------- wiring

def install(root=None) -> None:
    """Point the pipeline at a synthetic cache + ledger and fake Kalshi."""
    now = pd.Timestamp(utcnow())
    root = root or config.ROOT / "cache" / "demo"
    config.CACHE = root / "cache"
    config.LEDGER_DIR = root / "ledger"
    (config.CACHE / "lol").mkdir(parents=True, exist_ok=True)
    (config.CACHE / "oe").mkdir(parents=True, exist_ok=True)
    (config.CACHE / "f1").mkdir(parents=True, exist_ok=True)
    config.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    for f in config.LEDGER_DIR.glob("*.csv"):
        f.unlink()

    lol, oe = make_lol(now)
    lol.drop(columns="true_p").to_csv(config.CACHE / "lol" / "matches.csv", index=False)
    oe.to_csv(config.CACHE / "oe" / f"{now.year}_LoL_esports_match_data_from_OraclesElixir.csv", index=False)
    races, results, quali = make_f1(now)
    races.to_csv(config.CACHE / "f1" / "races.csv", index=False)
    results.to_csv(config.CACHE / "f1" / "results.csv", index=False)
    quali.to_csv(config.CACHE / "f1" / "qualifying.csv", index=False)

    # Fake market: true probability plus noise, 2-cent spread.
    up = lol[lol["state"] == "unstarted"]

    def fake_lol(status="open"):
        evs = []
        for r in up.itertuples(index=False):
            m = float(np.clip(r.true_p + RNG.normal(0, 0.06), 0.03, 0.97))
            st = pd.Timestamp(r.start_time).tz_convert(kalshi.ET)
            evs.append({"event_ticker": f"KXLOLGAME-{st:%y%b%d%H%M}".upper() + r.team1_code + r.team2_code,
                        "title": f"{r.team1} vs. {r.team2}", "start": st.to_pydatetime(),
                        "quotes": [{"ticker": f"X-{r.team1_code}", "yes_name": r.team1, "yes_bid": round(m - .01, 2),
                                    "yes_ask": round(m + .01, 2), "mid": round(m, 2), "volume": float(RNG.integers(200, 9000))},
                                   {"ticker": f"X-{r.team2_code}", "yes_name": r.team2, "yes_bid": round(1 - m - .01, 2),
                                    "yes_ask": round(1 - m + .01, 2), "mid": round(1 - m, 2), "volume": float(RNG.integers(200, 9000))}]})
        return evs

    nxt = races[pd.to_datetime(races["race_time"], utc=True) > now].iloc[0]
    last_lineup = results[(results["season"] == results["season"].max())]
    last_lineup = last_lineup[last_lineup["round"] == last_lineup["round"].max()]

    def fake_f1(status="open"):
        w = np.exp(-0.25 * np.arange(len(last_lineup)))
        w = w / w.sum()
        return [{"event_ticker": f"KXF1RACE-R{int(nxt['round'])}", "title": nxt["race_name"],
                 "start": pd.Timestamp(nxt["race_time"]).to_pydatetime(),
                 "quotes": [{"ticker": f"KXF1RACE-R-{c}", "driver_code": c, "yes_name": n,
                             "yes_bid": max(round(p - .01, 2), 0.0), "yes_ask": round(p + .01, 2) or 0.01,
                             "mid": round(p, 2), "volume": float(RNG.integers(50, 50000))}
                            for c, n, p in zip(last_lineup["driver_code"], last_lineup["driver"], w)]}]

    kalshi.lol_match_events = fake_lol
    kalshi.f1_race_events = fake_f1
    _fake_ledger_history(lol, now)


def _fake_ledger_history(lol: pd.DataFrame, now) -> None:
    """Six weeks of pretend logged forecasts so the track-record tab has
    something to show. The 'model' is the truth plus more noise than the
    'market', so the demo market should (correctly) win."""
    past = lol[(lol["state"] == "completed") &
               (pd.to_datetime(lol["start_time"], utc=True) > now - pd.Timedelta(days=42))]
    preds, res = [], []
    for r in past.itertuples(index=False):
        mkt = float(np.clip(r.true_p + RNG.normal(0, 0.04), 0.03, 0.97))
        mod = float(np.clip(r.true_p + RNG.normal(0, 0.09), 0.02, 0.98))
        start = pd.Timestamp(r.start_time)
        for team, p, m, out in ((r.team1, mod, mkt, int(r.winner == 1)), (r.team2, 1 - mod, 1 - mkt, int(r.winner == 2))):
            sid = team.lower()
            preds.append({"logged_at": iso((start - timedelta(hours=1)).to_pydatetime()), "sport": "lol",
                          "event_id": r.match_id, "event_name": f"{r.team1} vs {r.team2} ({r.league_name})",
                          "event_start": r.start_time, "stage": "pre_match", "selection": team, "selection_id": sid,
                          "p_model": round(p, 4), "model": "demo", "kalshi_ticker": "", "bid": round(m - .01, 2),
                          "ask": round(m + .01, 2), "mid": round(m, 2), "volume": 1000})
            res.append({"sport": "lol", "event_id": r.match_id, "selection_id": sid, "outcome": out,
                        "recorded_at": iso(now.to_pydatetime())})
    pd.DataFrame(preds, columns=PRED_COLS).to_csv(config.LEDGER_DIR / "predictions.csv", index=False)
    pd.DataFrame(res, columns=RES_COLS).to_csv(config.LEDGER_DIR / "results.csv", index=False)
