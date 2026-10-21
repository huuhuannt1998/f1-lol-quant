"""Pro LoL schedule and results from the (unofficial) lolesports.com API.

This is the same JSON the lolesports.com website loads. It has every league
(including the LPL), series scores, and upcoming matches, so it is our source
of truth for both results and ratings. It is undocumented and can change.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..http import get_json
from ..util import parse_ts, utcnow

INTERNATIONAL = {"worlds", "msi", "first_stand"}
# Cross-league events inside one region (not a team's own league).
CROSS_LEAGUE = {"americas_cup", "lta_cross"}
STORE = lambda: config.CACHE / "lol" / "matches.csv"  # noqa: E731

COLUMNS = ["match_id", "start_time", "state", "league", "league_name", "block",
           "best_of", "team1", "team1_code", "team2", "team2_code",
           "t1_wins", "t2_wins", "winner"]
NUMERIC = ["best_of", "t1_wins", "t2_wins", "winner"]


def typed(df: pd.DataFrame) -> pd.DataFrame:
    """Give every match frame the same columns and dtypes.

    Without this, a frame whose score columns are all None (a page of
    unstarted matches, or the empty store on the first run) has dtype
    object while the stored CSV has float. pandas then has to guess the
    result dtype on concat, warns that the guess will change, and leaves
    best_of as object."""
    df = df.reindex(columns=COLUMNS)
    for c in COLUMNS:
        if c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        else:
            df[c] = df[c].astype(object)
    df["match_id"] = df["match_id"].astype(str)
    return df


def _get(endpoint: str, ttl: float | None, **params) -> dict:
    params = {"hl": "en-US", **params}
    return get_json(f"{config.LOLESPORTS_API}/{endpoint}", params=params,
                    headers={"x-api-key": config.LOLESPORTS_KEY}, ttl=ttl)


def leagues() -> list[dict]:
    return _get("getLeagues", ttl=7 * 86400)["data"]["leagues"]


def league_ids() -> dict[str, str]:
    by_slug = {lg["slug"]: lg["id"] for lg in leagues()}
    missing = [s for s in config.LOL_LEAGUE_SLUGS if s not in by_slug]
    if missing:
        print(f"  (lolesports: no league for slugs {missing}; skipped)")
    return {s: by_slug[s] for s in config.LOL_LEAGUE_SLUGS if s in by_slug}


def parse_event(ev: dict) -> dict | None:
    if ev.get("type") != "match" or not ev.get("match"):
        return None
    m = ev["match"]
    teams = m.get("teams") or []
    if len(teams) != 2:
        return None
    t1, t2 = teams
    r1, r2 = t1.get("result") or {}, t2.get("result") or {}
    winner = 1 if r1.get("outcome") == "win" else 2 if r2.get("outcome") == "win" else None
    strategy = m.get("strategy") or {}
    return {
        "match_id": str(m.get("id")),
        "start_time": ev.get("startTime"),
        "state": ev.get("state"),
        "league": (ev.get("league") or {}).get("slug"),
        "league_name": (ev.get("league") or {}).get("name"),
        "block": ev.get("blockName"),
        "best_of": int(strategy.get("count") or 1),
        "team1": t1.get("name"), "team1_code": t1.get("code"),
        "team2": t2.get("name"), "team2_code": t2.get("code"),
        "t1_wins": r1.get("gameWins"), "t2_wins": r2.get("gameWins"),
        "winner": winner,
    }


def load_store() -> pd.DataFrame:
    p = STORE()
    if p.exists():
        return typed(pd.read_csv(p, dtype={"match_id": str}))
    return typed(pd.DataFrame(columns=COLUMNS))


def is_tier2(league: str | None, year: int) -> bool:
    if league not in config.LOL_TIER2_LEAGUES:
        return False
    since = config.LOL_TIER2_LEAGUES[league]
    return since is None or year >= since


def tier2_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows played in a tier-2 league (in a year it was tier-2)."""
    year = pd.to_datetime(df["start_time"], utc=True).dt.year
    out = pd.Series(False, index=df.index)
    for slug, since in config.LOL_TIER2_LEAGUES.items():
        out |= (df["league"] == slug) & (year >= (since or 0))
    return out


def prepare(matches: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Model-ready copy of the store: integer best_of, and matches that are
    still not completed LOL_STALE_HOURS after their start marked 'stale'.

    Stale rows are kept (lolesports may still settle them later; the store
    keeps whatever the API last said) but every consumer skips them: Elo uses
    only 'completed', forecasts skip 'stale', the ledger ignores their ids."""
    m = matches.copy()
    m["best_of"] = pd.to_numeric(m["best_of"], errors="coerce").fillna(1).astype(int)
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    t = pd.to_datetime(m["start_time"], utc=True)
    old = (m["state"] != "completed") & (t < now - pd.Timedelta(hours=config.LOL_STALE_HOURS))
    m.loc[old, "state"] = "stale"
    return m


def merge_store(store: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Newest copy of each match wins; matches before LOL_HISTORY_START go."""
    frames = [typed(f) for f in (store, fresh) if len(f)]
    if not frames:
        return typed(pd.DataFrame(columns=COLUMNS))
    merged = pd.concat(frames, ignore_index=True).drop_duplicates("match_id", keep="last")
    merged = merged[pd.to_datetime(merged["start_time"], utc=True)
                    >= pd.Timestamp(config.LOL_HISTORY_START, tz="UTC")]
    return merged.sort_values("start_time").reset_index(drop=True)


def update(max_pages: int = 400) -> pd.DataFrame:
    """Walk the schedule from newest to oldest, merging into the local store.

    Stops once a page contains only matches we already hold as completed and
    that are more than a few days old, so routine runs cost a handful of
    requests while the first run back-fills to LOL_HISTORY_START."""
    store = load_store()
    known_done = set(store.loc[store["state"] == "completed", "match_id"])
    ids = ",".join(league_ids().values())
    history_start = parse_ts(config.LOL_HISTORY_START + "T00:00:00Z")
    settled_before = utcnow() - pd.Timedelta(days=4)
    # Walk all the way back while the store does not reach LOL_HISTORY_START
    # yet (first run, or the start date was moved earlier). 45 days of slack:
    # seasons start in mid-January, so a finished backfill never re-triggers.
    earliest = pd.to_datetime(store["start_time"], utc=True).min()
    backfill = pd.isna(earliest) or earliest > pd.Timestamp(history_start) + pd.Timedelta(days=45)
    if backfill:
        print(f"  lolesports: back-filling history to {config.LOL_HISTORY_START}")

    rows, token = [], None
    # Also take one step "newer" to see upcoming matches beyond the first page.
    first = _get("getSchedule", ttl=600, leagueId=ids)["data"]["schedule"]
    pages = [first]
    newer = (first.get("pages") or {}).get("newer")
    if newer:
        pages.append(_get("getSchedule", ttl=600, leagueId=ids, pageToken=newer)["data"]["schedule"])
    token = (first.get("pages") or {}).get("older")
    for page in pages:
        rows += [r for r in map(parse_event, page.get("events") or []) if r]

    for _ in range(max_pages):
        if not token:
            break
        # 1 h, not longer: a match that finished a day or two ago sits on these
        # pages, and the hourly job should pick up its result on the next run.
        # The walk stops at the first fully-known old page, so this is 1-3 requests.
        page = _get("getSchedule", ttl=3600, leagueId=ids, pageToken=token)["data"]["schedule"]
        parsed = [r for r in map(parse_event, page.get("events") or []) if r]
        rows += parsed
        times = [parse_ts(r["start_time"]) for r in parsed if r["start_time"]]
        if not times or min(times) < history_start:
            break
        all_known = all(r["match_id"] in known_done for r in parsed)
        if all_known and max(times) < settled_before and not backfill:
            break
        token = (page.get("pages") or {}).get("older")

    fresh = pd.DataFrame(rows, columns=COLUMNS)
    merged = merge_store(store, fresh)
    STORE().parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(STORE(), index=False)
    print(f"  lolesports: {len(fresh)} rows fetched, {len(merged)} matches stored")
    return merged
