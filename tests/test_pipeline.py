"""Correctness + leakage tests. Run: python -m pytest -q

The leakage tests are the important ones: they change a future outcome and
assert that no earlier forecast moves. If one fails, a feature is peeking.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from pipeline import config, kalshi, ledger
from pipeline.util import iso, utcnow


@pytest.fixture()
def tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "ledger")
    return tmp_path


# ------------------------------------------------------------------ LoL

def test_series_probability():
    from pipeline.lol.elo import p_series
    assert p_series(0.6, 1) == pytest.approx(0.6)
    assert p_series(0.6, 3) == pytest.approx(0.6 ** 2 + 2 * 0.6 ** 2 * 0.4)
    assert p_series(0.5, 5) == pytest.approx(0.5)
    assert p_series(0.7, 5) > p_series(0.7, 3) > 0.7


def _toy_matches(n=200, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"Team {c}" for c in "ABCDEFGH"]
    t0 = pd.Timestamp("2024-02-01", tz="UTC")
    rows = []
    for i in range(n):
        a, b = rng.choice(teams, 2, replace=False)
        w = int(rng.integers(1, 3))
        rows.append({"match_id": str(i), "start_time": iso((t0 + pd.Timedelta(days=i)).to_pydatetime()),
                     "state": "completed", "league": "lck", "league_name": "LCK", "block": "",
                     "best_of": 3, "team1": a, "team1_code": a[-1], "team2": b, "team2_code": b[-1],
                     "t1_wins": 2 if w == 1 else int(rng.integers(0, 2)),
                     "t2_wins": 2 if w == 2 else int(rng.integers(0, 2)), "winner": w})
    return pd.DataFrame(rows)


def test_lol_features_do_not_see_own_or_future_results():
    from pipeline.lol.model import historical_features
    m = _toy_matches()
    base, _, _ = historical_features(m)
    k = 120
    flipped = m.copy()
    flipped.loc[k:, "winner"] = 3 - flipped.loc[k:, "winner"]
    flipped.loc[k:, ["t1_wins", "t2_wins"]] = flipped.loc[k:, ["t2_wins", "t1_wins"]].values
    alt, _, _ = historical_features(flipped)
    cols = ["r1_pre", "r2_pre", "p_series_elo", "form_diff", "rest_diff"]
    # Matches up to and including k must be identical: k's own result
    # cannot leak into its own features.
    pd.testing.assert_frame_equal(base.loc[:k, cols], alt.loc[:k, cols])
    assert not np.allclose(base.loc[k + 5:, "r1_pre"], alt.loc[k + 5:, "r1_pre"])


def test_match_model_is_symmetric():
    from pipeline.lol.model import BASE_FEATURES, MatchModel, historical_features
    feats, _, _ = historical_features(_toy_matches(300))
    model = MatchModel("logreg", BASE_FEATURES).fit(feats)
    swapped = feats.copy()
    for c in ["elo_diff", "elo_logit", "form_diff", "rest_diff"]:
        swapped[c] = -swapped[c]
    np.testing.assert_allclose(model.predict(feats), 1 - model.predict(swapped), atol=1e-9)


def test_oe_attach_uses_only_earlier_games():
    from pipeline.lol import oracles_elixir as oe
    games = pd.DataFrame({"team": ["team a"] * 3,
                          "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
                          "oe_golddiffat15": [100.0, 200.0, 300.0]})
    matches = pd.DataFrame({"team1": ["Team A"], "team2": ["Team B"],
                            "start_time": ["2026-01-02T00:00:00Z"]})
    out = oe.attach(matches, games)
    assert out.loc[0, "team1_oe_golddiffat15"] == 100.0   # not the game at the same instant


def test_kalshi_ticker_time():
    t = kalshi.ticker_start_time("KXLOLGAME-26OCT161900C9FUR")
    assert (t.year, t.month, t.day, t.hour, t.minute) == (2026, 10, 16, 19, 0)
    assert t.utcoffset() == timedelta(hours=-4)


def test_kalshi_quote_parsing():
    q = kalshi.market_quote({"ticker": "X", "yes_bid_dollars": "0.8000", "yes_ask_dollars": "0.8100",
                             "last_price_dollars": "0.81", "volume_fp": "1116.53", "yes_sub_title": "Cloud9"})
    assert q["mid"] == pytest.approx(0.805) and q["volume"] == pytest.approx(1116.53)
    q = kalshi.market_quote({"yes_bid": 0, "yes_ask": 100, "last_price": 42})
    assert q["mid"] == pytest.approx(0.42)       # empty book -> last trade


def test_match_kalshi_pairs_by_names_and_time():
    from pipeline.lol.run import match_kalshi
    up = pd.DataFrame({"start_time": ["2026-10-16T23:00:00Z"], "team1": ["Cloud9"], "team2": ["FURIA"]})
    ev = {"event_ticker": "KXLOLGAME-26OCT161900C9FUR", "title": "Cloud9 vs. FURIA Esports",
          "start": kalshi.ticker_start_time("KXLOLGAME-26OCT161900C9FUR"),
          "quotes": [{"yes_name": "FURIA Esports", "mid": 0.2}, {"yes_name": "Cloud9", "mid": 0.8}]}
    got = match_kalshi(up, [ev])
    assert got[0]["t1"]["mid"] == 0.8 and got[0]["t2"]["mid"] == 0.2


# ------------------------------------------------------------------- F1

def _toy_f1(seasons=(2022, 2023), rounds=8, seed=1):
    rng = np.random.default_rng(seed)
    races, results, quali = [], [], []
    drivers = [(f"d{i}", f"team{i // 2}") for i in range(10)]
    for s in seasons:
        for r in range(1, rounds + 1):
            races.append({"season": s, "round": r, "race_name": f"GP{r}", "circuit_id": f"c{r}",
                          "circuit": "", "country": "", "race_time": f"{s}-0{min(r, 9)}-01T12:00:00Z",
                          "quali_time": None, "sprint_time": None})
            order = rng.permutation(len(drivers))
            for pos, j in enumerate(order, 1):
                d, c = drivers[j]
                results.append({"season": s, "round": r, "driver_id": d, "driver_code": d.upper(), "driver": d,
                                "constructor_id": c, "constructor": c, "grid": pos, "position": pos,
                                "position_text": str(pos), "points": max(0, 11 - pos), "status": "Finished",
                                "laps": 50, "finished": True})
                quali.append({"season": s, "round": r, "driver_id": d, "constructor_id": c,
                              "quali_pos": int(rng.integers(1, 11)), "best_lap_s": 80 + rng.random()})
    return pd.DataFrame(races), pd.DataFrame(results), pd.DataFrame(quali)


def test_f1_features_do_not_see_own_race():
    from pipeline.f1.model import FORM, build_entries
    races, results, quali = _toy_f1()
    base = build_entries(races, results, quali)
    res2 = results.copy()
    mask = (res2["season"] == 2023) & (res2["round"] >= 4)
    res2.loc[mask, "position"] = 11 - res2.loc[mask, "position"]
    res2.loc[mask, "points"] = 0.0
    alt = build_entries(races, res2, quali)
    key = ["season", "round", "driver_id"]
    a = base.set_index(key).sort_index()
    b = alt.set_index(key).sort_index()
    upto = a.index.get_level_values("season") * 100 + a.index.get_level_values("round") <= 202304
    pd.testing.assert_frame_equal(a.loc[upto, FORM], b.loc[upto, FORM])


def test_conditional_logit_probabilities_sum_to_one():
    from pipeline.f1.model import build_entries, make_model
    races, results, quali = _toy_f1()
    ent = build_entries(races, results, quali)
    for stage in ("pre_quali", "post_quali"):
        p = make_model("clogit", stage).fit(ent).predict(ent)
        sums = pd.Series(p).groupby(ent["race_key"].values).sum()
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-9)


def test_plackett_luce_consistency():
    from pipeline.f1.model import plackett_luce
    p = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    sims = plackett_luce(p, n_sims=40000)
    assert sims["p_podium"].sum() == pytest.approx(3.0, abs=1e-6)
    assert np.all(np.diff(sims["p_podium"]) <= 0.01)


# --------------------------------------------------------------- ledger

def test_ledger_rejects_started_events_and_dedupes(tmp_ledger):
    now = utcnow()
    row = {"sport": "lol", "event_id": "1", "event_name": "A vs B", "stage": "pre_match",
           "selection": "A", "selection_id": "a", "p_model": 0.6, "model": "t", "mid": 0.55,
           "event_start": iso(now + timedelta(hours=2))}
    assert ledger.log_predictions([row]) == 1
    assert ledger.log_predictions([row]) == 0                         # unchanged -> skipped
    assert ledger.log_predictions([{**row, "p_model": 0.65}]) == 1
    past = {**row, "event_id": "2", "event_start": iso(now - timedelta(minutes=1))}
    assert ledger.log_predictions([past]) == 0                        # already started


def test_paper_trade_math():
    c = pd.DataFrame([{"event_start": "x", "event_name": "e", "selection": "A", "p_model": 0.70,
                       "ask": 0.55, "bid": 0.53, "outcome": 1.0},
                      {"event_start": "y", "event_name": "e", "selection": "B", "p_model": 0.10,
                       "ask": 0.30, "bid": 0.28, "outcome": 0.0}])
    t = ledger.paper_trades(c, allow_no=True)
    assert list(t["side"]) == ["YES", "NO"]
    fee_yes = 0.02   # ceil(0.07*0.55*0.45*100)/100
    assert t.iloc[0]["pnl"] == pytest.approx(1 - 0.55 - fee_yes)


def test_my_picks_lock_rule_and_scoring(tmp_path, monkeypatch):
    from pipeline import picks
    monkeypatch.setattr(config, "PICKS_DIR", tmp_path / "my_picks")
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "ledger")
    config.PICKS_DIR.mkdir()
    start = "2026-10-23T17:00:00Z"

    def write(name, picked_at, team, p):
        (config.PICKS_DIR / name).write_text(",".join(picks.COLS) + "\n" +
                                            f"{picked_at},lol,m1,A vs B,{start},{team},{team.lower()},{p}\n")

    write("a.csv", "2026-10-23T10:00:00Z", "A", 0.7)
    write("b.csv", "2026-10-23T18:00:00Z", "B", 0.6)          # written after the start: never counts
    ledger.record_results([{"sport": "lol", "event_id": "m1", "selection_id": "a", "outcome": 1},
                           {"sport": "lol", "event_id": "m1", "selection_id": "b", "outcome": 0}])
    s = picks.summary("lol", times={})                         # no git: the time in the file is used
    assert s["settled"] == 1 and s["late_rejected"] == 1
    assert s["you"]["log_loss"] == pytest.approx(-np.log(0.7), abs=1e-4)
    # The commit time is the lock: committed after the start, the pick no longer counts.
    s = picks.summary("lol", times={"a.csv": "2026-10-23T17:30:00+00:00"})
    assert s["settled"] == 0 and s["late_rejected"] == 2


def test_parse_git_log_gives_first_commit_time_per_file():
    from pipeline.picks import parse_git_log
    out = "@2026-10-22T10:00:00+00:00\n\nledger/my_picks/b.csv\n@2026-10-21T09:00:00-04:00\n\nledger/my_picks/a.csv\n"
    assert parse_git_log(out) == {"b.csv": "2026-10-22T10:00:00+00:00", "a.csv": "2026-10-21T09:00:00-04:00"}


def test_oe_download_stamps_our_time_not_drives(tmp_path, monkeypatch):
    import os
    import sys
    import time
    from types import SimpleNamespace
    from pipeline.lol import oracles_elixir as oe
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(config, "OE_AUTO_DOWNLOAD", True)
    year = pd.Timestamp.now(tz="UTC").year
    name = f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"

    def fake_download(id, output, quiet):          # like gdown: file dated by Drive, 3 days ago
        open(output, "w").write("x\n")
        old = time.time() - 3 * 86400
        os.utime(output, (old, old))

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(
        download_folder=lambda **kw: [SimpleNamespace(path=f"OE/{name}", id="f")], download=fake_download))
    oe.download(years=[year])
    assert time.time() - (tmp_path / "oe" / name).stat().st_mtime < 60   # not re-downloaded next hour


def test_scorecard_can_ignore_never_settling_events(tmp_ledger):
    start = iso(utcnow() + timedelta(hours=2))
    rows = [{"sport": "lol", "event_id": e, "event_name": f"m{e}", "stage": "pre_match", "selection": "A",
             "selection_id": "a", "p_model": 0.6, "model": "t", "mid": None, "event_start": start}
            for e in ("1", "2")]
    assert ledger.log_predictions(rows) == 2
    ledger.record_results([{"sport": "lol", "event_id": "1", "selection_id": "a", "outcome": 1}])
    assert ledger.scorecard("lol")["pending_events"] == 1            # event 2 never settles
    sc = ledger.scorecard("lol", ignore={"2"})
    assert sc["settled_events"] == 1 and sc["pending_events"] == 0
    assert {r["event_name"] for r in ledger.recent_rows("lol", ignore={"2"})} == {"m1"}


# ------------------------------------------- data hygiene (PROMPT tasks 2-6)

def _lol_row(i, start, state, **kw):
    row = {"match_id": str(i), "start_time": start, "state": state, "league": "lck", "league_name": "LCK",
           "block": "", "best_of": 3, "team1": "A", "team1_code": "A", "team2": "B", "team2_code": "B",
           "t1_wins": None, "t2_wins": None, "winner": None}
    return {**row, **kw}


def test_lolesports_merge_keeps_dtypes_without_warnings():
    import warnings
    from pipeline.lol import lolesports
    first_page = pd.DataFrame([_lol_row(1, "2026-10-10T10:00:00Z", "completed", t1_wins=3, t2_wins=1, winner=1),
                               _lol_row(2, "2026-10-20T10:00:00Z", "unstarted")], columns=lolesports.COLUMNS)
    only_unstarted = pd.DataFrame([_lol_row(2, "2026-10-20T11:00:00Z", "unstarted")], columns=lolesports.COLUMNS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)    # the two warnings the first real run printed
        store = lolesports.merge_store(pd.DataFrame(columns=lolesports.COLUMNS), first_page)
        store = lolesports.merge_store(store, only_unstarted)
        out = lolesports.prepare(store, pd.Timestamp("2026-10-15T00:00:00Z"))
    assert list(store["match_id"]) == ["1", "2"]
    assert store["t1_wins"].dtype == float and store.loc[1, "start_time"] == "2026-10-20T11:00:00Z"
    assert out["best_of"].dtype.kind == "i"


def test_model_features_are_finite():
    from pipeline.f1.model import STAGES, build_entries
    from pipeline.lol.model import BASE_FEATURES, historical_features
    from pipeline.util import nonfinite_counts
    feats, _, _ = historical_features(_toy_matches())
    assert nonfinite_counts(feats, BASE_FEATURES) == {}
    assert nonfinite_counts(build_entries(*_toy_f1()), STAGES["post_quali"]) == {}
    bad = pd.DataFrame({"a": [1.0, np.nan, np.inf], "b": [0.0, 1.0, 2.0]})
    assert nonfinite_counts(bad, ["a", "b"]) == {"a": 2}


def _pool(league, prefix, t0, n, seed):
    """One closed league: its teams only ever play each other."""
    rng = np.random.default_rng(seed)
    teams = [f"{prefix} Team {c}" for c in "ABCD"]
    rows = []
    for i in range(n):
        a, b = rng.choice(teams, 2, replace=False)
        w = int(rng.integers(1, 3))
        rows.append(_lol_row(f"{league}{i}", iso((t0 + pd.Timedelta(days=i)).to_pydatetime()), "completed",
                             league=league, league_name=league.upper(), team1=a, team2=b,
                             t1_wins=2 if w == 1 else 1, t2_wins=2 if w == 2 else 1, winner=w))
    return pd.DataFrame(rows)


def test_power_rankings_leave_out_tier2_teams():
    from pipeline.lol import lolesports
    from pipeline.lol.model import historical_features
    from pipeline.lol.run import power_rankings
    assert lolesports.is_tier2("lfl", 2024) and lolesports.is_tier2("pcs", 2026)
    assert not lolesports.is_tier2("pcs", 2024) and not lolesports.is_tier2("lck", 2026)
    assert not lolesports.is_tier2(None, 2026)
    t0 = pd.Timestamp("2026-03-01", tz="UTC")
    m = pd.concat([_pool("lck", "K", t0, 120, 1), _pool("lfl", "F", t0, 120, 2)], ignore_index=True)
    m = m.sort_values("start_time", kind="stable").reset_index(drop=True)
    assert int(lolesports.tier2_mask(m).sum()) == 120
    _, book, _ = historical_features(m)
    now = pd.Timestamp("2026-09-01", tz="UTC")
    shown, left_out = power_rankings(book, now)
    raw, _ = power_rankings(book, now, exclude_tier2=False)
    assert {r["league"] for r in shown} == {"lck"} and left_out == 4
    assert {r["league"] for r in raw} == {"lck", "lfl"}


def test_calibration_slope_detects_under_confidence():
    from pipeline.util import calibration_slope
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20000)
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-2 * x))).astype(float)
    assert calibration_slope(1 / (1 + np.exp(-x)), y) == pytest.approx(2.0, abs=0.15)      # too timid
    assert calibration_slope(1 / (1 + np.exp(-2 * x)), y) == pytest.approx(1.0, abs=0.08)  # calibrated
    assert calibration_slope([0.6] * 5, [1] * 5) is None                                  # too few to say
    assert calibration_slope([0.5] * 30, [1, 0] * 15) is None                             # no signal at all
    assert calibration_slope([0.7] * 30, [1] * 30) is None                                # separable: no finite b


def test_upcoming_skips_stale_rows():
    from pipeline.lol import lolesports
    from pipeline.lol.model import historical_features, upcoming_features
    _, book, ft = historical_features(_toy_matches())
    soon = iso((pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).to_pydatetime())
    from pipeline.lol.model import tbd_matches
    m = lolesports.typed(pd.DataFrame([_lol_row(1, soon, "unstarted", team1="Team A", team2="Team B"),
                                       _lol_row(2, soon, "stale", team1="Team C", team2="Team D"),
                                       _lol_row(3, soon, "unstarted", team1="TBD", team2="TBD")]))
    m["best_of"] = m["best_of"].astype(int)
    assert list(upcoming_features(m, book, ft)["match_id"]) == ["1"]
    assert list(tbd_matches(m)["match_id"]) == ["3"]      # shown as a slot, never forecast


def test_oe_history_rebuilds_series():
    from pipeline.lol import oe_history

    def g(gid, date, game, a, b, a_won, league="LCK"):
        return [{"gameid": gid, "league": league, "date": date, "game": game, "position": "team",
                 "teamname": a, "result": int(a_won)},
                {"gameid": gid, "league": league, "date": date, "game": game, "position": "team",
                 "teamname": b, "result": int(not a_won)}]

    rows = pd.DataFrame(
        g("g1", "2022-06-01 08:00:00", 1, "T1", "Gen.G", True) +     # Bo3: T1 2-1
        g("g2", "2022-06-01 09:00:00", 2, "Gen.G", "T1", True) +
        g("g3", "2022-06-01 10:00:00", 3, "T1", "Gen.G", True) +
        g("g4", "2022-06-02 08:00:00", 1, "T1", "Gen.G", False) +    # next day: a new Bo1, Gen.G wins
        g("g5", "2022-06-01 08:30:00", 1, "X", "Y", True, league="Unknown League") +   # untracked league
        [{"gameid": "g6", "league": "LCK", "date": "2022-06-03 08:00:00", "game": 1, "position": "team",
          "teamname": None, "result": 1}])
    s = oe_history.series_from_games(oe_history.games_from_team_rows(rows))
    assert len(s) == 2
    first, second = s.iloc[0], s.iloc[1]
    assert first["best_of"] == 3 and {first["team1"], first["team2"]} == {"t1", "gen g esports"}
    assert first["winner"] == (1 if first["team1"] == "t1" else 2) and first["t1_wins"] + first["t2_wins"] == 3
    assert second["best_of"] == 1 and second["league"] == "lck"


def test_oe_history_stops_where_lolesports_starts(tmp_path, monkeypatch):
    from pipeline.lol import lolesports, oe_history
    path = tmp_path / "hist.csv"
    lolesports.typed(pd.DataFrame([_lol_row("oe:1", "2019-05-01T08:00:00Z", "completed", winner=1),
                                   _lol_row("oe:2", "2023-11-01T08:00:00Z", "completed", winner=1)])).to_csv(path, index=False)
    monkeypatch.setattr(config, "LOL_OE_HISTORY_FILE", path)
    store = lolesports.typed(pd.DataFrame([_lol_row("100", "2023-10-19T08:00:00Z", "completed", winner=2)]))
    assert list(oe_history.add_history(store, from_year=2018)["match_id"]) == ["oe:1", "100"]   # no overlap
    assert list(oe_history.add_history(store, from_year=2020)["match_id"]) == ["100"]
    monkeypatch.setattr(config, "LOL_OE_HISTORY_FROM", None)
    assert list(oe_history.add_history(store)["match_id"]) == ["100"]                          # switched off


def test_worlds_seed_lookup():
    from pipeline.lol.run import seed
    assert seed("worlds", "2026-10-23T23:00:00Z", "T1") == "LCK #3"
    assert seed("worlds", "2026-10-23T17:00:00Z", "Gen.G Esports") == "LCK #1"
    assert seed("worlds", "2025-10-23T17:00:00Z", "T1") is None      # other year: not listed
    assert seed("lck", "2026-06-01T08:00:00Z", "T1") is None         # seeds only for listed events
    assert seed("worlds", None, "T1") is None


def test_cross_league_events_do_not_change_a_teams_league():
    from pipeline.lol.model import historical_features
    t0 = pd.Timestamp("2026-03-01", tz="UTC")
    cup = _lol_row("ac1", iso((t0 + pd.Timedelta(days=40)).to_pydatetime()), "completed",
                   league="americas_cup", league_name="Americas Cup", team1="N Team A", team2="N Team B",
                   t1_wins=2, t2_wins=0, winner=1)
    m = pd.concat([_pool("lcs", "N", t0, 30, 4), pd.DataFrame([cup])], ignore_index=True)
    _, book, _ = historical_features(m)
    assert book.last_league["n team a"] == "lcs"


def test_prepare_marks_stale_matches():
    from pipeline.lol import lolesports
    m = lolesports.typed(pd.DataFrame([
        _lol_row(1, "2025-05-30T08:00:00Z", "unstarted", best_of=None),   # never played
        _lol_row(2, "2026-10-16T09:00:00Z", "inProgress"),                # live now
        _lol_row(3, "2026-10-20T09:00:00Z", "unstarted"),                 # upcoming
        _lol_row(4, "2026-10-10T09:00:00Z", "completed", winner=1)]))
    out = lolesports.prepare(m, pd.Timestamp("2026-10-16T12:00:00Z"))
    assert list(out["state"]) == ["stale", "inProgress", "unstarted", "completed"]
    assert out["best_of"].tolist() == [1, 3, 3, 3]


def test_oe_skips_bad_files_and_backs_off_after_quota(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from pipeline.lol import oracles_elixir as oe
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(config, "OE_AUTO_DOWNLOAD", True)
    (tmp_path / "oe").mkdir()
    (tmp_path / "oe" / "2026_LoL_esports_match_data_from_OraclesElixir.csv").write_text("<html>quota</html>\n")
    assert oe.load_team_games().empty                   # an HTML error page must not crash the LoL run

    good = tmp_path / "oe" / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
    good.write_text("gameid,league,date,position,teamname,result\n"
                    "g1,LCK,2025-06-01 10:00:00,team,Gen.G,1\ng1,LCK,2025-06-01 10:00:00,team,T1,0\n")
    assert sorted(oe.load_team_games()["team"]) == ["gen g esports", "t1"]   # OE's "Gen.G" joins lolesports' name
    good.unlink()

    # A fake gdown: the folder lists fine, every file download hits Drive's quota.
    calls = []

    def refuse(id, output, quiet):
        calls.append(id)
        raise RuntimeError("Failed to retrieve file url:\n\n\tToo many users have viewed or downloaded this file")

    listing = [SimpleNamespace(path="OE/2025_LoL_esports_match_data_from_OraclesElixir.csv", id="f25"),
               SimpleNamespace(path="OE/2024_LoL_esports_match_data_from_OraclesElixir.csv", id="f24")]
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download_folder=lambda **kw: listing,
                                                              download=refuse))
    assert len(oe.download(years=[2024, 2025])) == 1    # only the local file
    assert calls == ["f25"] and oe._quota_marker().exists()   # stopped at the first refusal

    def contacted(**kw):
        raise AssertionError("Drive contacted while backing off")

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download_folder=contacted, download=contacted))
    assert len(oe.download(years=[2024, 2025])) == 1    # within 24h: local files only
