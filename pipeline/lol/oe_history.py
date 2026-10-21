"""Older LoL history rebuilt from Oracle's Elixir game files.

lolesports.com's schedule only goes back to Worlds 2023. Oracle's Elixir has
every pro game since 2014, one row per team per game. Grouping games into
series (same league, same two teams, consecutive game numbers, a few hours
apart) gives match results in the same shape as the lolesports store, so the
Elo replay can start years earlier. Most of the gain is international:
roughly 80 MSI + Worlds series a year, the only evidence of how regions
compare.

The series table is built once (`python -m pipeline oe-history`) and
committed as history/lol_oe_series.csv. Past seasons never change, so the
hourly job does not need the ~600 MB of old CSVs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from .elo import team_key
from .lolesports import COLUMNS, typed

# Oracle's Elixir league code -> our lolesports slug (only leagues we track).
OE_LEAGUES = {
    "WLDs": "worlds", "MSI": "msi",
    "LCK": "lck", "LPL": "lpl", "LEC": "lec", "EU LCS": "lec", "LCS": "lcs", "NA LCS": "lcs",
    "CBLOL": "cblol-brazil", "LMS": "pcs", "PCS": "pcs", "VCS": "vcs", "LJL": "ljl-japan", "LLA": "lla",
    "LCKC": "lck_challengers_league", "CK": "lck_challengers_league", "LDL": "ldl", "LFL": "lfl",
    "PRM": "prime_league", "TCL": "tcl", "LVP SL": "superliga", "EUM": "emea_masters", "NACL": "nacl",
}
GAP_HOURS = 10          # a later game of the same pair this long after the last one is a new series


def _team(name) -> str:
    """Normalized key, mapped to the lolesports spelling where OE differs."""
    k = team_key(name)
    return config.OE_TEAM_NAMES.get(k, k)


def games_from_team_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per game: league slug, time, game number, the two teams, winner."""
    rows = rows[rows["position"].astype(str).str.lower() == "team"]
    rows = rows[rows["league"].isin(OE_LEAGUES) & rows["teamname"].notna()].copy()
    rows["team"] = rows["teamname"].astype(str).map(_team)
    rows["result"] = pd.to_numeric(rows["result"], errors="coerce")
    out = []
    for gid, g in rows.groupby("gameid", sort=False):
        if len(g) != 2 or g["result"].sum() != 1 or g["team"].nunique() != 2:
            continue                                  # incomplete or inconsistent game
        a, b = g.iloc[0], g.iloc[1]
        out.append({"gameid": gid, "league": OE_LEAGUES[a["league"]], "oe_league": a["league"],
                    "date": a["date"], "game": a["game"], "team_a": a["team"], "team_b": b["team"],
                    "a_won": int(a["result"] == 1)})
    games = pd.DataFrame(out)
    if games.empty:
        return games
    games["t"] = pd.to_datetime(games["date"], utc=True, errors="coerce")
    return games.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)


def series_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive games of the same pair in the same league into series."""
    open_series: dict = {}          # (league, pair) -> current series dict
    done = []
    for r in games.itertuples(index=False):
        key = (r.league, frozenset((r.team_a, r.team_b)))
        s = open_series.get(key)
        new = (s is None or r.game == 1 or pd.isna(r.game)
               or (r.t - s["last"]) > pd.Timedelta(hours=GAP_HOURS))
        if new:
            if s is not None:
                done.append(s)
            s = {"first_game": r.gameid, "start": r.t, "last": r.t, "league": r.league,
                 "oe_league": r.oe_league, "team1": r.team_a, "team2": r.team_b, "w1": 0, "w2": 0}
            open_series[key] = s
        winner = r.team_a if r.a_won else r.team_b
        s["w1" if winner == s["team1"] else "w2"] += 1
        s["last"] = r.t
    done.extend(open_series.values())
    rows = []
    for s in done:
        w1, w2 = s["w1"], s["w2"]
        if w1 == w2:                                   # e.g. a 1-1 Bo2: no series winner
            continue
        best_of = 2 * max(w1, w2) - 1                  # 1 win -> Bo1, 2 -> Bo3, 3 -> Bo5
        rows.append({"match_id": f"oe:{s['first_game']}", "start_time": s["start"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "state": "completed", "league": s["league"], "league_name": s["oe_league"], "block": None,
                     "best_of": best_of, "team1": s["team1"], "team1_code": None,
                     "team2": s["team2"], "team2_code": None, "t1_wins": w1, "t2_wins": w2,
                     "winner": 1 if w1 > w2 else 2})
    return typed(pd.DataFrame(rows, columns=COLUMNS)).sort_values("start_time").reset_index(drop=True)


def download(years, dest: Path) -> None:
    """Fetch old OE yearly CSVs (past seasons: once) with gdown into dest."""
    import os
    import re
    import gdown
    dest.mkdir(parents=True, exist_ok=True)
    listing = gdown.download_folder(id=config.OE_DRIVE_FOLDER, skip_download=True, quiet=True, use_cookies=False)
    for f in listing or []:
        name = f.path.rsplit("/", 1)[-1]
        m = re.match(r"(\d{4})_LoL_esports_match_data", name)
        if not m or int(m.group(1)) not in years or (dest / name).exists():
            continue
        gdown.download(id=f.id, output=str(dest / name), quiet=True)
        os.utime(dest / name, None)
        print(f"  downloaded {name}")


def build(oe_dir: Path, out: Path) -> pd.DataFrame:
    """Read every OE yearly CSV in oe_dir (one at a time), write the series table."""
    cols = {"gameid", "league", "date", "game", "position", "teamname", "result"}
    games = []
    for p in sorted(oe_dir.glob("*_LoL_esports_match_data_from_OraclesElixir.csv")):
        rows = pd.read_csv(p, usecols=lambda c: c in cols, low_memory=False)
        games.append(games_from_team_rows(rows))
        print(f"  {p.name[:4]}: {len(games[-1])} games")
    series = series_from_games(pd.concat(games, ignore_index=True).sort_values("t"))
    out.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(out, index=False)
    print(f"  {len(series)} series -> {out}")
    return series


def add_history(store: pd.DataFrame, from_year: int | None = None) -> pd.DataFrame:
    """The lolesports store plus OE series from `from_year` up to the store's
    first match (so no match is counted twice). No file or from_year=None:
    the store unchanged."""
    from_year = config.LOL_OE_HISTORY_FROM if from_year is None else from_year
    path = config.LOL_OE_HISTORY_FILE
    if not from_year or not path.exists() or store.empty:
        return store
    hist = typed(pd.read_csv(path, dtype={"match_id": str}))
    t = pd.to_datetime(hist["start_time"], utc=True)
    first = pd.to_datetime(store["start_time"], utc=True).min()
    hist = hist[(t.dt.year >= from_year) & (t < first)]
    if hist.empty:
        return store
    return typed(pd.concat([hist, typed(store)], ignore_index=True)).sort_values("start_time").reset_index(drop=True)
