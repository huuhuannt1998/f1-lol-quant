"""Your own picks, scored next to the model's.

Each pick is one small CSV file in ledger/my_picks/, made by the page's
"My pick" button (it opens GitHub's new-file editor) or by hand. A pick
counts only if its file was committed *before* the event started: the
commit time is the lock time, like a sealed envelope. Files are never
edited; a later file for the same event replaces an earlier one, as long
as it was also committed before the start.

Why bother: your judgement may hold information the model never sees
(roster news, the patch, draft habits). Scoring both on the same events,
with the same metrics, shows whether it does. With enough settled picks a
blend of the two can be tested walk-forward like any other model
(docs/LEARNING.md §11).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from . import config, ledger
from .util import summarize

COLS = ["picked_at", "sport", "event_id", "event_name", "event_start", "selection", "selection_id", "p_pick"]
BLEND_MIN = 100     # settled head-to-head picks before a blend is worth testing


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce")


def parse_git_log(text: str) -> dict[str, str]:
    """`git log --diff-filter=A --format=@%cI --name-only` -> {file name: commit time}."""
    times, cur = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("@"):
            cur = line[1:]
        elif line and cur:
            times.setdefault(Path(line).name, cur)
    return times


def commit_times(d: Path) -> dict[str, str]:
    """When each pick file was first committed. Empty if git is unavailable
    (then the time written in the file is used, e.g. for a local preview)."""
    try:
        out = subprocess.run(["git", "log", "--diff-filter=A", "--format=@%cI", "--name-only", "--", str(d)],
                             cwd=config.ROOT, capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:  # noqa: BLE001 - no git, not a repository, ...
        return {}
    return parse_git_log(out)


def load(times: dict[str, str] | None = None) -> pd.DataFrame:
    d = config.PICKS_DIR
    files = sorted(d.glob("*.csv")) if d.exists() else []
    if not files:
        return pd.DataFrame(columns=COLS + ["locked_at", "file"])
    times = commit_times(d) if times is None else times
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype={"event_id": str, "selection_id": str})
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
            print(f"  picks: skipping unreadable {f.name}: {type(e).__name__}")
            continue
        if not set(COLS) <= set(df.columns):
            print(f"  picks: skipping {f.name}: missing columns")
            continue
        for r in df.to_dict("records"):
            rows.append({**{c: r[c] for c in COLS}, "file": f.name,
                         "locked_at": times.get(f.name) or r["picked_at"]})
    out = pd.DataFrame(rows, columns=COLS + ["locked_at", "file"])
    out["p_pick"] = pd.to_numeric(out["p_pick"], errors="coerce")
    out.loc[out["p_pick"] > 1, "p_pick"] /= 100          # "65" written by hand means 65%
    for c in ("sport", "event_id", "selection_id"):
        out[c] = out[c].astype(str)
    return out


def accepted(picks: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Picks locked before their event started, the latest one per event;
    plus how many were rejected for being too late."""
    if picks.empty:
        return picks, 0
    lock, start = _ts(picks["locked_at"]), _ts(picks["event_start"])
    late = int((lock.notna() & start.notna() & (lock >= start)).sum())
    ok = lock.notna() & start.notna() & (lock < start) & picks["p_pick"].between(0.01, 0.99)
    keep = picks[ok].assign(_lock=lock[ok]).sort_values("_lock")
    keep = keep.groupby(["sport", "event_id"], as_index=False).tail(1)
    return keep.drop(columns="_lock"), late


def summary(sport: str, times: dict[str, str] | None = None) -> dict:
    """Scoreboard for the page: you vs the model on the same selections."""
    picks = load(times)
    picks, late = accepted(picks[picks["sport"] == sport])
    if picks.empty:
        return {"settled": 0, "pending": 0, "late_rejected": late, "open": {}}
    res = ledger._read("results.csv", ledger.RES_COLS)[["sport", "event_id", "selection_id", "outcome"]]
    res = res.astype({"event_id": str, "selection_id": str})
    m = picks.merge(res, on=["sport", "event_id", "selection_id"], how="left")
    model = ledger.closing(sport)
    if len(model):
        model = model.astype({"event_id": str, "selection_id": str})
        m = m.merge(model[["event_id", "selection_id", "p_model"]], on=["event_id", "selection_id"], how="left")
    else:
        m["p_model"] = float("nan")
    settled = m[m["outcome"].notna()]
    both = settled[settled["p_model"].notna()]
    upcoming = m[_ts(m["event_start"]) > pd.Timestamp.now(tz="UTC")]
    return {
        "settled": int(len(settled)), "pending": int(m["outcome"].isna().sum()), "late_rejected": late,
        "you": summarize(settled["p_pick"], settled["outcome"]),
        "head_to_head": ({"n": int(len(both)), "you": summarize(both["p_pick"], both["outcome"]),
                          "model": summarize(both["p_model"], both["outcome"])} if len(both) else None),
        "blend_ready": bool(len(both) >= BLEND_MIN), "blend_min": BLEND_MIN,
        "open": {r.event_id: {"selection": r.selection, "selection_id": r.selection_id,
                              "p": round(float(r.p_pick), 3), "locked_at": str(r.locked_at)}
                 for r in upcoming.itertuples(index=False)},
        "recent": [{"event_start": r.event_start, "event_name": r.event_name, "selection": r.selection,
                    "p_pick": round(float(r.p_pick), 3),
                    "p_model": None if pd.isna(r.p_model) else round(float(r.p_model), 3),
                    "outcome": int(r.outcome)}
                   for r in settled.sort_values("event_start", ascending=False).head(20).itertuples(index=False)],
    }
