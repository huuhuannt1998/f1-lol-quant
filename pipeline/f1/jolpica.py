"""F1 schedule, results and qualifying from Jolpica (the Ergast successor).

Licensed CC BY-NC-SA 4.0: free for non-commercial use with attribution,
which the web page shows in its footer.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..http import get_json
from ..util import utcnow


def _paged(path: str, table: str, item: str, ttl) -> list[dict]:
    """Jolpica caps `limit` at 100, so walk the offsets."""
    out, offset = [], 0
    while True:
        d = get_json(f"{config.JOLPICA_API}/{path}.json",
                     params={"limit": 100, "offset": offset}, ttl=ttl)["MRData"]
        races = d[table]["Races"]
        out.extend(races)
        total, got = int(d["total"]), sum(len(r.get(item, [])) for r in races)
        offset += 100
        if offset >= total or got == 0:
            break
    # Pages split races across boundaries; merge them back by round.
    merged: dict[str, dict] = {}
    for r in out:
        key = r["round"]
        if key in merged:
            merged[key][item].extend(r.get(item, []))
        else:
            merged[key] = {**r, item: list(r.get(item, []))}
    return list(merged.values())


def _time_s(t: str | None) -> float | None:
    """'1:23.456' -> 83.456"""
    if not t:
        return None
    try:
        parts = t.split(":")
        return float(parts[-1]) + (60 * float(parts[-2]) if len(parts) > 1 else 0)
    except ValueError:
        return None


def _ts(date: str | None, time: str | None) -> str | None:
    if not date:
        return None
    return f"{date}T{(time or '12:00:00Z').replace('Z', '')}Z"


def fetch_season(season: int, current: bool) -> dict[str, pd.DataFrame]:
    ttl = 1800 if current else None   # finished seasons never change
    sched = get_json(f"{config.JOLPICA_API}/{season}.json", params={"limit": 100},
                     ttl=6 * 3600 if current else None)["MRData"]["RaceTable"]["Races"]
    races = pd.DataFrame([{
        "season": season, "round": int(r["round"]), "race_name": r["raceName"],
        "circuit_id": r["Circuit"]["circuitId"], "circuit": r["Circuit"]["circuitName"],
        "country": r["Circuit"]["Location"].get("country"),
        "race_time": _ts(r.get("date"), r.get("time")),
        "quali_time": _ts((r.get("Qualifying") or {}).get("date"), (r.get("Qualifying") or {}).get("time")),
        "sprint_time": _ts((r.get("Sprint") or {}).get("date"), (r.get("Sprint") or {}).get("time")),
    } for r in sched])

    res_rows, q_rows = [], []
    for r in _paged(f"{season}/results", "RaceTable", "Results", ttl):
        for x in r["Results"]:
            d, c = x["Driver"], x["Constructor"]
            status = x.get("status", "")
            res_rows.append({
                "season": season, "round": int(r["round"]), "driver_id": d["driverId"],
                "driver_code": d.get("code") or d["driverId"][:3].upper(),
                "driver": f'{d["givenName"]} {d["familyName"]}',
                "constructor_id": c["constructorId"], "constructor": c["name"],
                "grid": int(x.get("grid") or 0), "position": int(x["position"]),
                "position_text": x.get("positionText"), "points": float(x.get("points") or 0),
                "status": status, "laps": int(x.get("laps") or 0),
                "finished": status == "Finished" or status.startswith("+") or "Lap" in status,
            })
    for r in _paged(f"{season}/qualifying", "RaceTable", "QualifyingResults", ttl):
        for x in r["QualifyingResults"]:
            times = [t for t in (_time_s(x.get(k)) for k in ("Q3", "Q2", "Q1")) if t]
            q_rows.append({"season": season, "round": int(r["round"]),
                           "driver_id": x["Driver"]["driverId"],
                           "constructor_id": x["Constructor"]["constructorId"],
                           "quali_pos": int(x["position"]),
                           "best_lap_s": min(times) if times else None})
    return {"races": races, "results": pd.DataFrame(res_rows), "qualifying": pd.DataFrame(q_rows)}


def update(first: int = config.F1_FIRST_SEASON) -> dict[str, pd.DataFrame]:
    now = utcnow()
    parts = {"races": [], "results": [], "qualifying": []}
    for season in range(first, now.year + 1):
        got = fetch_season(season, current=(season == now.year))
        for k in parts:
            parts[k].append(got[k])
    out = {k: pd.concat([d for d in v if len(d)], ignore_index=True) for k, v in parts.items()}
    d = config.CACHE / "f1"
    d.mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        df.to_csv(d / f"{k}.csv", index=False)
    print(f"  jolpica: {len(out['races'])} races, {len(out['results'])} results, "
          f"{len(out['qualifying'])} qualifying rows")
    return out


def standings(season: int) -> dict[str, list[dict]]:
    """Official championship tables (these include sprint points, which the
    race results endpoint does not)."""
    out = {}
    for kind, key in (("driverstandings", "DriverStandings"), ("constructorstandings", "ConstructorStandings")):
        lists = get_json(f"{config.JOLPICA_API}/{season}/{kind}.json", params={"limit": 100},
                         ttl=1800)["MRData"]["StandingsTable"]["StandingsLists"]
        rows = lists[0][key] if lists else []
        if key == "DriverStandings":
            out["drivers"] = [{"pos": int(r.get("position") or 0), "points": float(r["points"]),
                               "wins": int(r["wins"]), "code": r["Driver"].get("code"),
                               "name": f'{r["Driver"]["givenName"]} {r["Driver"]["familyName"]}',
                               "team": (r.get("Constructors") or [{}])[-1].get("name")} for r in rows]
        else:
            out["constructors"] = [{"pos": int(r.get("position") or 0), "points": float(r["points"]),
                                    "wins": int(r["wins"]), "name": r["Constructor"]["name"]} for r in rows]
    return out


def load() -> dict[str, pd.DataFrame]:
    d = config.CACHE / "f1"
    return {k: pd.read_csv(d / f"{k}.csv") for k in ("races", "results", "qualifying")}
