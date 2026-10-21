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
    m = lolesports.typed(pd.DataFrame([_lol_row(1, soon, "unstarted", team1="Team A", team2="Team B"),
                                       _lol_row(2, soon, "stale", team1="Team C", team2="Team D")]))
    m["best_of"] = m["best_of"].astype(int)
    assert list(upcoming_features(m, book, ft)["match_id"]) == ["1"]


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
