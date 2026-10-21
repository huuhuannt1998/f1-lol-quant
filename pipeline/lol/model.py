"""Turn ratings + form into match probabilities, and test them honestly.

Three models, simplest first:
  elo     Elo game probability -> best-of-N series probability. No fitting.
  logreg  Logistic regression on a handful of point-in-time features.
  gbm     LightGBM on the same features (+ Oracle's Elixir form if present).

All are scored by walk-forward backtest: for each month, fit only on matches
that finished before that month, then predict it. A random train/test split
would leak the future (same teams, same patch) and flatter every model.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .. import config
from ..util import calibration_bins, calibration_slope, summarize
from .elo import EloBook, p_game, p_series, run_elo, team_key
from .lolesports import INTERNATIONAL

BASE_FEATURES = ["elo_diff", "elo_logit", "form_diff", "exp_min", "rest_diff", "intl", "best_of"]
OE_DIFFS = ["golddiffat15", "xpdiffat15", "firstdragon", "firsttower", "result"]


# ------------------------------------------------------------- features

class FormTracker:
    """Recent game win-rate and rest days, replayed in time order."""

    def __init__(self, window: int = 10):
        self.recent = defaultdict(lambda: deque(maxlen=window))
        self.last = {}

    def state(self, team: str, when: pd.Timestamp) -> tuple[float, float]:
        r = self.recent.get(team)
        wr = float(np.mean(r)) if r else 0.5
        last = self.last.get(team)
        rest = (when - last).total_seconds() / 86400 if last is not None else 30.0
        return wr, min(rest, 30.0)

    def update(self, row) -> None:
        when = pd.Timestamp(row.start_time)
        a, b = team_key(row.team1), team_key(row.team2)
        w1, w2 = row.t1_wins, row.t2_wins
        if pd.isna(w1) or pd.isna(w2) or w1 + w2 == 0:
            w1, w2 = (1, 0) if row.winner == 1 else (0, 1)
        for _ in range(int(w1)):
            self.recent[a].append(1); self.recent[b].append(0)
        for _ in range(int(w2)):
            self.recent[a].append(0); self.recent[b].append(1)
        self.last[a] = self.last[b] = when


def _feature_row(r1, r2, n1, n2, best_of, league, f1, f2) -> dict:
    pg = p_game(r1, r2)
    ps = min(max(p_series(pg, int(best_of)), 1e-4), 1 - 1e-4)
    return {"elo_diff": r1 - r2, "elo_logit": float(np.log(ps / (1 - ps))),
            "p_series_elo": ps, "form_diff": f1[0] - f2[0],
            "exp_min": float(np.log1p(min(n1, n2))), "rest_diff": f1[1] - f2[1],
            "intl": int(league in INTERNATIONAL), "best_of": int(best_of)}


def historical_features(matches: pd.DataFrame, k: float = config.ELO_K,
                        regress: float = config.ELO_SEASON_REGRESS,
                        intl_mult: float = config.ELO_INTL_K_MULT):
    """Features for every completed match, using only earlier matches."""
    done, book = run_elo(matches, k=k, regress=regress, intl_mult=intl_mult)
    ft = FormTracker()
    rows = []
    for r in done.itertuples(index=False):
        when = pd.Timestamp(r.start_time)
        a, b = team_key(r.team1), team_key(r.team2)
        rows.append(_feature_row(r.r1_pre, r.r2_pre, r.n1_pre, r.n2_pre, r.best_of,
                                 r.league, ft.state(a, when), ft.state(b, when)))
        ft.update(r)
    extra = pd.DataFrame(rows, index=done.index)
    feats = pd.concat([done, extra.drop(columns=[c for c in extra if c in done])], axis=1)
    feats["y"] = (feats["winner"] == 1).astype(int)
    return feats, book, ft


def upcoming_features(matches: pd.DataFrame, book: EloBook, ft: FormTracker,
                      horizon_days: int = 14) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    t = pd.to_datetime(matches["start_time"], utc=True)
    up = matches[~matches["state"].isin(["completed", "stale"])
                 & (t <= now + pd.Timedelta(days=horizon_days))
                 & (t >= now - pd.Timedelta(hours=12))].copy()
    up = up[~up["team1"].str.upper().isin(["TBD", "TBA"]) & ~up["team2"].str.upper().isin(["TBD", "TBA"])]
    rows = []
    for r in up.itertuples(index=False):
        when = pd.Timestamp(r.start_time)
        a, b = team_key(r.team1), team_key(r.team2)
        r1, r2 = book.get(r.team1, r.league), book.get(r.team2, r.league)
        rows.append({"r1_pre": r1, "r2_pre": r2, **_feature_row(
            r1, r2, book.games[a], book.games[b], r.best_of, r.league,
            ft.state(a, when), ft.state(b, when))})
    if not rows:
        return up.iloc[0:0]
    extra = pd.DataFrame(rows)
    return pd.concat([up.reset_index(drop=True), extra.drop(columns=[c for c in extra if c in up])], axis=1)


def add_oe_diffs(df: pd.DataFrame) -> list[str]:
    cols = []
    for s in OE_DIFFS:
        a, b = f"team1_oe_{s}", f"team2_oe_{s}"
        if a in df.columns and b in df.columns:
            df[f"oe_{s}_diff"] = df[a] - df[b]
            cols.append(f"oe_{s}_diff")
    return cols


# --------------------------------------------------------------- models

_ANTISYM = ["elo_diff", "elo_logit", "form_diff", "rest_diff"]


def _swap(X: pd.DataFrame) -> pd.DataFrame:
    """Same match seen from the other team's side. Training on both views
    stops the model learning that 'team1' (an arbitrary label) wins more."""
    Xs = X.copy()
    for c in Xs.columns:
        if c in _ANTISYM or c.startswith("oe_"):
            Xs[c] = -Xs[c]
    return Xs


class MatchModel:
    def __init__(self, kind: str, features: list[str]):
        self.kind, self.features = kind, features
        self.est = None

    def fit(self, df: pd.DataFrame) -> "MatchModel":
        if self.kind == "elo":
            return self
        X = df[self.features]
        X2 = pd.concat([X, _swap(X)], ignore_index=True)
        y2 = np.concatenate([df["y"].values, 1 - df["y"].values])
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.est = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
            self.est.fit(X2.fillna(0), y2)
        elif self.kind == "gbm":
            import lightgbm as lgb
            self.est = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=15,
                                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
            self.est.fit(X2, y2)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "elo":
            return df["p_series_elo"].values
        X = df[self.features]
        Xf = X.fillna(0) if self.kind == "logreg" else X
        Xs = _swap(X)
        Xsf = Xs.fillna(0) if self.kind == "logreg" else Xs
        p = self.est.predict_proba(Xf)[:, 1]
        q = self.est.predict_proba(Xsf)[:, 1]
        return (p + (1 - q)) / 2       # average both views -> exactly symmetric


# ------------------------------------------------------------- backtest

def walk_forward(feats: pd.DataFrame, kinds: dict[str, list[str]], start: pd.Timestamp,
                 min_train: int = 300) -> pd.DataFrame:
    t = pd.to_datetime(feats["start_time"], utc=True)
    months = pd.date_range(start.normalize().replace(day=1), t.max(), freq="MS", tz="UTC")
    out = []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        train, test = feats[t < m0], feats[(t >= m0) & (t < m1)]
        if len(test) == 0 or len(train) < min_train:
            continue
        block = test[["match_id", "start_time", "league", "best_of", "intl", "y"]].copy()
        for name, cols in kinds.items():
            block[f"p_{name}"] = MatchModel(name, cols).fit(train).predict(test)
        out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def tune_elo(matches: pd.DataFrame, until: pd.Timestamp, burn_in_days: int = 180,
             intl_mult: float = config.ELO_INTL_K_MULT) -> dict:
    """Pick K and off-season shrinkage on data *before* the evaluation window,
    so the backtest that follows is not graded on its own tuning data."""
    best = None
    for k in (16, 24, 32, 40):
        for regress in (0.15, 0.3, 0.5):
            df, _ = run_elo(matches, k=k, regress=regress, intl_mult=intl_mult)
            t = pd.to_datetime(df["start_time"], utc=True)
            sel = (t < until) & (t > t.min() + pd.Timedelta(days=burn_in_days))
            if sel.sum() < 100:
                continue
            ll = summarize(df.loc[sel, "p_series_elo"], (df.loc[sel, "winner"] == 1).astype(int))["log_loss"]
            if best is None or ll < best["log_loss"]:
                best = {"k": k, "regress": regress, "log_loss": ll, "n": int(sel.sum())}
    return best or {"k": config.ELO_K, "regress": config.ELO_SEASON_REGRESS}


def evaluate(bt: pd.DataFrame, names: list[str]) -> dict:
    res = {}
    intl = bt.intl == 1
    for n in names:
        col = f"p_{n}"
        res[n] = {"all": {**summarize(bt[col], bt["y"]),
                          "calibration_slope": calibration_slope(bt[col], bt["y"])},
                  "international": {**summarize(bt.loc[intl, col], bt.loc[intl, "y"]),
                                    "calibration_slope": calibration_slope(bt.loc[intl, col], bt.loc[intl, "y"])},
                  "bo1": summarize(bt.loc[bt.best_of == 1, col], bt.loc[bt.best_of == 1, "y"]),
                  "bo3+": summarize(bt.loc[bt.best_of >= 3, col], bt.loc[bt.best_of >= 3, "y"]),
                  "calibration": calibration_bins(bt[col], bt["y"]),
                  "calibration_intl": calibration_bins(bt.loc[intl, col], bt.loc[intl, "y"], n_bins=5)}
    return res
