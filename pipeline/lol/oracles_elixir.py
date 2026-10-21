"""Optional enrichment from Oracle's Elixir game-level CSVs.

lolesports.com only tells us who won. Oracle's Elixir (OE) adds what happened
inside each game: gold difference at 15 minutes, first dragon, and so on. We
turn those into *rolling* team averages over past games and test whether they
add anything on top of Elo (see `python -m pipeline backtest`).

Leakage rule: a stat from game G may only describe a team in matches that
start after G. We enforce it with an as-of join on time (strictly before).

OE files are shared via a public Google Drive folder. If downloading fails,
put e.g. 2026_LoL_esports_match_data_from_OraclesElixir.csv into cache/oe/
by hand; the rest of the pipeline works with or without these files.
"""
from __future__ import annotations

import re
import time

import pandas as pd

from .. import config
from .elo import team_key

STATS = ["golddiffat15", "xpdiffat15", "csdiffat15", "firstdragon", "firsttower",
         "firstherald", "firstbaron", "gamelength", "kills", "deaths", "result"]
HALFLIFE_GAMES = 8


QUOTA_RETRY_H = 24     # Drive's per-file quota resets within about a day


def oe_dir():
    return config.CACHE / "oe"


def _quota_marker():
    return oe_dir() / ".drive_quota_hit"


def download(years=None, max_age_h: float = 20) -> list:
    """Fetch the yearly CSVs with gdown. Past years are downloaded once;
    the current year is refreshed when older than `max_age_h`.

    The OE files are popular, so Google Drive often refuses them with "Too
    many users have viewed or downloaded this file recently" (a per-file
    quota, not a gdown bug). After that we stop asking for QUOTA_RETRY_H
    hours instead of hammering Drive every run."""
    years = years or config.OE_YEARS
    d = oe_dir()
    d.mkdir(parents=True, exist_ok=True)
    if not config.OE_AUTO_DOWNLOAD:
        print("  OE: auto-download off (F1LOL_OE_DOWNLOAD=0); using local files only")
        return sorted(d.glob("*.csv"))
    mark = _quota_marker()
    if mark.exists() and time.time() - mark.stat().st_mtime < QUOTA_RETRY_H * 3600:
        print(f"  OE: Drive quota was hit in the last {QUOTA_RETRY_H}h; using local files only")
        return sorted(d.glob("*.csv"))
    try:
        import gdown
    except ImportError:
        print("  OE: gdown not installed; skipping (pip install gdown)")
        return sorted(d.glob("*.csv"))
    try:
        listing = gdown.download_folder(id=config.OE_DRIVE_FOLDER, skip_download=True,
                                        quiet=True, use_cookies=False)
    except Exception as e:  # noqa: BLE001 - Drive can fail many ways
        print(f"  OE: could not list Drive folder ({e}); using local files only")
        return sorted(d.glob("*.csv"))
    this_year = pd.Timestamp.now(tz="UTC").year
    for f in listing or []:
        name = f.path.rsplit("/", 1)[-1]
        m = re.match(r"(\d{4})_LoL_esports_match_data", name)
        if not m or int(m.group(1)) not in years:
            continue
        dest = d / name
        stale = (not dest.exists() or
                 (int(m.group(1)) == this_year and time.time() - dest.stat().st_mtime > max_age_h * 3600))
        if stale:
            try:
                gdown.download(id=f.id, output=str(dest), quiet=True)
                print(f"  OE: downloaded {name}")
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "Too many users" in msg:
                    mark.touch()
                    print(f"  OE: Google Drive download quota hit ({name}); not retrying for "
                          f"{QUOTA_RETRY_H}h. Manual path: README > Oracle's Elixir.")
                    break
                first = msg.strip().splitlines()[0] if msg.strip() else type(e).__name__
                print(f"  OE: failed {name}: {first}")
    return sorted(d.glob("*.csv"))


def load_team_games() -> pd.DataFrame:
    """One row per team per game (OE's position == 'team' rows)."""
    frames = []
    for p in sorted(oe_dir().glob("*.csv")):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:  # noqa: BLE001 - a half-downloaded file must not stop the LoL run
            print(f"  OE: skipping unreadable {p.name}: {type(e).__name__}")
            continue
        if not {"position", "gameid", "league", "date", "teamname"} <= set(df.columns):
            print(f"  OE: skipping {p.name}: not an Oracle's Elixir match-data file")
            continue
        df = df[df["position"].astype(str).str.lower() == "team"]
        keep = ["gameid", "league", "date", "teamname"] + [c for c in STATS if c in df.columns]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out = out.dropna(subset=["date", "teamname"]).drop_duplicates(["gameid", "teamname"])
    out["team"] = out["teamname"].map(team_key)
    return out.sort_values("date").reset_index(drop=True)


def unmatched_teams(matches: pd.DataFrame, games: pd.DataFrame, top: int = 12) -> list[str]:
    """lolesports team names (most frequent first) that never appear in the
    OE files over the period OE covers. These are what keeps match coverage
    down; the usual cause is naming ('Gen.G Esports' vs 'Gen.G'), fixed with
    config.LOL_TEAM_ALIASES or a looser join key."""
    if games.empty:
        return []
    t = pd.to_datetime(matches["start_time"], utc=True)
    recent = matches[(matches["state"] == "completed") & (t >= games["date"].min())
                     & (t <= games["date"].max())]
    names = pd.concat([recent["team1"], recent["team2"]], ignore_index=True)
    miss = names[~names.map(team_key).isin(set(games["team"]))]
    return [str(n) for n in miss.value_counts().index[:top]]


def rolling_team_form(games: pd.DataFrame) -> pd.DataFrame:
    """Exponentially weighted averages of each stat *after* each game.

    The value at row G includes game G; the as-of join in `attach` only
    uses rows strictly before a match, so no match sees its own games."""
    if games.empty:
        return games
    stats = [c for c in STATS if c in games.columns]
    games = games.copy()
    games[stats] = games[stats].apply(pd.to_numeric, errors="coerce")
    form = (games.groupby("team")[stats]
                 .transform(lambda s: s.ewm(halflife=HALFLIFE_GAMES, ignore_na=True).mean()))
    form.columns = [f"oe_{c}" for c in stats]
    form["oe_games"] = games.groupby("team").cumcount() + 1
    return pd.concat([games[["team", "date"]], form], axis=1)


def attach(matches: pd.DataFrame, form: pd.DataFrame) -> pd.DataFrame:
    """Add team1_/team2_ OE form columns using only games before each match."""
    if form is None or form.empty:
        return matches
    m = matches.copy()
    m["_t"] = pd.to_datetime(m["start_time"], utc=True)
    m["_order"] = range(len(m))
    form = form.sort_values("date")
    for side in ("team1", "team2"):
        m[f"_{side}_key"] = m[side].map(team_key)
        left = m[["_order", "_t", f"_{side}_key"]].sort_values("_t")
        j = pd.merge_asof(left, form, left_on="_t", right_on="date",
                          left_by=f"_{side}_key", right_by="team",
                          allow_exact_matches=False, direction="backward")
        cols = [c for c in form.columns if c.startswith("oe_")]
        j = j.set_index("_order")[cols].add_prefix(f"{side}_")
        m = m.join(j, on="_order")
    return m.drop(columns=["_t", "_order", "_team1_key", "_team2_key"])
