"""Offline LoL experiments on cached data. No network; the only file written
is cache/backtest_lol.json (so results can be read back later).

    python -m pipeline backtest lol

Answers two questions from PROMPT.md:

  Task 3  Keep tier-2 leagues (ERLs, academies) or drop them?
  Task 5  Is the model under-confident internationally, and does a bigger
          ELO_INTL_K_MULT help?

Every variant is scored on the same walk-forward window, on slices that all
variants share, so the numbers are comparable:

  intl    international matches (Worlds, MSI, First Stand)
  majors  domestic matches outside the tier-2 leagues

Both choices (keep/drop tier-2, the international K multiplier) are *made*
on Elo log loss *before* the window (`pre_ll`), and the window only
*reports* how the chosen setting does. Choosing by the window's own numbers
would grade the choice on its own test data, the same mistake as a random
train/test split. (The gbm here uses BASE_FEATURES only, not OE features.)
"""
from __future__ import annotations

import json

import pandas as pd

from .. import config
from ..util import calibration_slope, has_lightgbm, summarize
from . import lolesports, oe_history
from .elo import run_elo
from .model import BASE_FEATURES, historical_features, tune_elo, walk_forward
from .run import eval_window_start, power_rankings

INTL_MULTS = (1.0, 1.5, 2.0, 3.0)


def _pre_window_ll(matches: pd.DataFrame, until: pd.Timestamp, k: float, regress: float,
                   intl_mult: float, burn_in_days: int = 180) -> dict:
    """Elo log loss *before* the backtest window on the two shared slices.
    The first months are skipped (ratings have not settled yet), as in tune_elo."""
    df, _ = run_elo(matches, k=k, regress=regress, intl_mult=intl_mult)
    t = pd.to_datetime(df["start_time"], utc=True)
    pre = (t < until) & (t > t.min() + pd.Timedelta(days=burn_in_days))
    intl = df["league"].isin(lolesports.INTERNATIONAL)
    out = {}
    for name, sel in (("intl", pre & intl), ("majors", pre & ~intl & ~lolesports.tier2_mask(df))):
        y = (df.loc[sel, "winner"] == 1).astype(int)
        out[name] = summarize(df.loc[sel, "p_series_elo"], y)["log_loss"] if sel.sum() >= 20 else None
    return out


def variant(matches: pd.DataFrame, eval_start: pd.Timestamp, now: pd.Timestamp,
            drop_tier2: bool, intl_mult: float) -> dict:
    m = matches[~lolesports.tier2_mask(matches)].reset_index(drop=True) if drop_tier2 else matches
    tuned = tune_elo(m, until=eval_start, intl_mult=intl_mult)
    feats, book, _ = historical_features(m, k=tuned["k"], regress=tuned["regress"], intl_mult=intl_mult)
    raw, _ = power_rankings(book, now, exclude_tier2=False)
    shown, _ = power_rankings(book, now, exclude_tier2=True)
    out = {"drop_tier2": drop_tier2, "intl_mult": intl_mult,
           "k": tuned["k"], "regress": tuned["regress"],
           "pre_ll": _pre_window_ll(m, eval_start, tuned["k"], tuned["regress"], intl_mult),
           "tier2_in_raw_top15": sum(lolesports.is_tier2(r["league"], now.year) for r in raw[:15]),
           "top15": [f"{r['team']} ({r['league']}) {r['elo']}" for r in shown[:15]],
           "models": {}}

    kinds = {"elo": [], "logreg": BASE_FEATURES}
    if has_lightgbm():
        kinds["gbm"] = BASE_FEATURES
    bt = walk_forward(feats, kinds, eval_start)
    if bt.empty:
        return out
    intl = bt["intl"] == 1
    majors = ~intl & ~lolesports.tier2_mask(bt)
    for name in kinds:
        col = f"p_{name}"
        out["models"][name] = {
            "intl": {**summarize(bt.loc[intl, col], bt.loc[intl, "y"]),
                     "slope": calibration_slope(bt.loc[intl, col], bt.loc[intl, "y"])},
            "majors": summarize(bt.loc[majors, col], bt.loc[majors, "y"]),
            "all": summarize(bt[col], bt["y"])}
    return out


def _fmt(x, nd=4):
    return "   –  " if x is None else f"{x:.{nd}f}"


def _print(r: dict) -> None:
    print(f"\n[{'drop' if r['drop_tier2'] else 'keep'} tier-2, intl K x{r['intl_mult']}]  "
          f"K={r['k']} shrink={r['regress']}  pre-window Elo ll: intl {_fmt(r['pre_ll']['intl'])} "
          f"majors {_fmt(r['pre_ll']['majors'])}  tier-2 teams in raw top 15: {r['tier2_in_raw_top15']}")
    print(f"  {'model':<7} {'intl n':>6} {'ll':>7} {'brier':>7} {'slope':>6}   "
          f"{'majors n':>8} {'ll':>7}   {'all n':>6} {'ll':>7}")
    for name, m in r["models"].items():
        i, mj, a = m["intl"], m["majors"], m["all"]
        print(f"  {name:<7} {i.get('n', 0):>6} {_fmt(i.get('log_loss')):>7} {_fmt(i.get('brier')):>7} "
              f"{_fmt(i.get('slope'), 2):>6}   {mj.get('n', 0):>8} {_fmt(mj.get('log_loss')):>7}   "
              f"{a.get('n', 0):>6} {_fmt(a.get('log_loss')):>7}")


def compare(intl_mults=INTL_MULTS, drops=(False, True), out_name: str = "backtest_lol.json") -> list[dict]:
    now = pd.Timestamp.now(tz="UTC")
    matches = lolesports.prepare(oe_history.add_history(lolesports.load_store()), now)
    if not (matches["state"] == "completed").any():
        raise SystemExit("No cached LoL matches. Run `python -m pipeline update --sport lol` first.")
    eval_start = eval_window_start(matches, now)
    print(f"LoL experiments: window from {eval_start:%Y-%m-%d}, "
          f"{int((matches['state'] == 'completed').sum())} completed matches (cached, no network)")
    print("slope = calibration slope on international matches: >1 under-confident, <1 over-confident")
    rows = []
    for drop in drops:
        for mult in intl_mults:
            r = variant(matches, eval_start, now, drop, mult)
            _print(r)
            rows.append(r)
    def pick(cands, slice_):
        ok = [r for r in cands if r["pre_ll"].get(slice_) is not None]
        return min(ok, key=lambda r: r["pre_ll"][slice_]) if ok else None

    t2 = pick(rows, "majors")                                         # keep vs drop
    ik = pick([r for r in rows if t2 and r["drop_tier2"] == t2["drop_tier2"]], "intl")
    if t2 and ik:
        print(f"\nChosen on data before {eval_start:%Y-%m-%d} (Elo log loss): "
              f"{'drop' if t2['drop_tier2'] else 'keep'} tier-2 (majors {t2['pre_ll']['majors']:.4f}), "
              f"intl K x{ik['intl_mult']} (intl {ik['pre_ll']['intl']:.4f}).\n"
              f"Its window rows above are the honest check; set config.LOL_DROP_TIER2 / "
              f"ELO_INTL_K_MULT from this line, not from the window.")
    print("\nTop 15 as the page would show it (keep tier-2, current intl K):")
    cur = next((r for r in rows if not r["drop_tier2"] and r["intl_mult"] == config.ELO_INTL_K_MULT), rows[0])
    for i, line in enumerate(cur["top15"], 1):
        print(f"  {i:>2}. {line}")
    path = config.CACHE / out_name
    path.write_text(json.dumps({"generated_at": now.isoformat(), "window_from": eval_start.isoformat(),
                                "variants": rows}, indent=1))
    print(f"\nsaved {path}")
    return rows
