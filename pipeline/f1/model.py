"""F1 race-winner model.

A race is a multi-way contest, so instead of 20 independent yes/no models we
use a *conditional logit*: every driver gets a score s = w . x, and

    P(driver i wins) = exp(s_i) / sum_j exp(s_j)        (softmax within a race)

That guarantees the probabilities in a race sum to 1, which matters when you
compare them with Kalshi's per-driver markets. Training maximises the
log-probability of the actual winner. Podium and points chances come from
simulating full finishing orders with the Plackett-Luce model (repeatedly
draw a winner, remove them, draw the next from the rest).

Two forecast stages:
  pre_quali   before qualifying: form features only
  post_quali  after qualifying: + qualifying position and gap to pole
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .. import config
from ..util import calibration_bins

FORM = ["drv_ewm_finish", "drv_ewm_points", "con_ewm_points", "drv_ewm_quali",
        "con_ewm_quali", "drv_dnf_rate", "drv_circuit_finish", "tm_quali_delta",
        "drv_starts_log", "early_season_reset"]
QUALI = ["quali_logpos", "quali_gap_pct", "front_row"]
STAGES = {"pre_quali": FORM, "post_quali": FORM + QUALI}
BASELINE = {"pre_quali": ["con_ewm_points"], "post_quali": ["quali_logpos"]}


# ------------------------------------------------------------- features

class _EWM:
    """Exponentially weighted mean that can be shrunk between seasons."""

    def __init__(self, halflife: float, prior: float):
        self.a = 1 - 0.5 ** (1 / halflife)
        self.v: dict = {}
        self.prior = prior

    def get(self, k):
        return self.v.get(k, self.prior)

    def add(self, k, x):
        self.v[k] = self.get(k) + self.a * (x - self.get(k)) if k in self.v else x

    def shrink(self, frac: float):
        for k in self.v:
            self.v[k] += frac * (self.prior - self.v[k])


def build_entries(races: pd.DataFrame, results: pd.DataFrame, quali: pd.DataFrame,
                  upcoming: tuple[int, int] | None = None) -> pd.DataFrame:
    """One row per driver per race with features known *before* that race.

    Replays races in order: compute features for race R from the state left
    by races < R, then fold R's results into the state."""
    hl = config.F1_EWM_HALFLIFE
    drv_fin, drv_pts = _EWM(hl, 12.0), _EWM(hl, 3.0)
    con_pts, drv_q, con_q = _EWM(hl, 6.0), _EWM(hl, 10.0), _EWM(hl, 10.0)
    dnf = _EWM(10, 0.15)
    tm_delta = _EWM(hl, 0.0)
    starts = defaultdict(int)
    circuit_hist = defaultdict(lambda: deque(maxlen=5))

    results = results.sort_values(["season", "round", "position"])
    order = races.sort_values(["season", "round"])[["season", "round", "circuit_id"]]
    res_by = {k: g for k, g in results.groupby(["season", "round"])}
    q_by = {k: g for k, g in quali.groupby(["season", "round"])}
    rows, prev_season = [], None

    for season, rnd, circuit in order.itertuples(index=False):
        key = (season, rnd)
        is_up = upcoming is not None and key == tuple(upcoming)
        if key not in res_by and not is_up:
            continue
        if season != prev_season and prev_season is not None:
            reset = season in config.F1_REG_RESET_SEASONS
            for e in (con_pts, con_q):
                e.shrink(0.6 if reset else 0.25)
            for e in (drv_fin, drv_pts, drv_q):
                e.shrink(0.3 if reset else 0.1)
        prev_season = season

        # Who is racing: actual results, else the qualifying list, else the
        # most recent race's line-up.
        if key in res_by:
            field = res_by[key][["driver_id", "driver_code", "driver", "constructor_id", "constructor"]]
        elif key in q_by:
            last = results.drop_duplicates("driver_id", keep="last").set_index("driver_id")
            field = q_by[key][["driver_id", "constructor_id"]].copy()
            for c in ("driver_code", "driver", "constructor"):
                field[c] = field["driver_id"].map(last[c]) if len(last) else None
            field["driver_code"] = field["driver_code"].fillna(field["driver_id"].str[:3].str.upper())
            field["driver"] = field["driver"].fillna(field["driver_id"])
            field["constructor"] = field["constructor"].fillna(field["constructor_id"])
        else:
            last_key = max(k for k in res_by if k < key)
            field = res_by[last_key][["driver_id", "driver_code", "driver", "constructor_id", "constructor"]]

        q = q_by.get(key)
        qmap = q.set_index("driver_id") if q is not None else None
        pole = q["best_lap_s"].min() if q is not None else np.nan
        for f in field.itertuples(index=False):
            d, c = f.driver_id, f.constructor_id
            hist = circuit_hist[(d, circuit)]
            row = {"season": season, "round": rnd, "circuit_id": circuit,
                   "driver_id": d, "driver_code": f.driver_code, "driver": f.driver,
                   "constructor_id": c, "constructor": f.constructor,
                   "drv_ewm_finish": drv_fin.get(d), "drv_ewm_points": drv_pts.get(d),
                   "con_ewm_points": con_pts.get(c), "drv_ewm_quali": drv_q.get(d),
                   "con_ewm_quali": con_q.get(c), "drv_dnf_rate": dnf.get(d),
                   # shrink a short circuit history toward overall form
                   "drv_circuit_finish": (sum(hist) + 2 * drv_fin.get(d)) / (len(hist) + 2),
                   "tm_quali_delta": tm_delta.get(d),
                   "drv_starts_log": np.log1p(starts[d]),
                   "early_season_reset": float(season in config.F1_REG_RESET_SEASONS and rnd <= 6)}
            if qmap is not None and d in qmap.index:
                qp = int(qmap.at[d, "quali_pos"])
                lap = qmap.at[d, "best_lap_s"]
                gap = (lap - pole) / pole * 100 if pd.notna(lap) and pole else 3.0
                row.update(quali_pos=qp, quali_logpos=np.log(qp), front_row=float(qp <= 2),
                           quali_gap_pct=min(gap, 5.0))
            else:
                row.update(quali_pos=np.nan, quali_logpos=np.log(15), front_row=0.0, quali_gap_pct=3.0)
            rows.append(row)

        if key not in res_by:
            continue
        # ---- fold this race's outcome into the state
        r = res_by[key]
        field_size = len(r)
        for x in r.itertuples(index=False):
            fin = x.position if x.finished else field_size
            drv_fin.add(x.driver_id, fin)
            drv_pts.add(x.driver_id, x.points)
            dnf.add(x.driver_id, 0.0 if x.finished else 1.0)
            circuit_hist[(x.driver_id, circuit)].append(fin)
            starts[x.driver_id] += 1
        for c, g in r.groupby("constructor_id"):
            con_pts.add(c, g["points"].mean())
        if q is not None:
            for x in q.itertuples(index=False):
                drv_q.add(x.driver_id, x.quali_pos)
            for c, g in q.groupby("constructor_id"):
                con_q.add(c, g["quali_pos"].mean())
                if len(g) == 2:
                    a, b = g.itertuples(index=False)
                    tm_delta.add(a.driver_id, a.quali_pos - b.quali_pos)
                    tm_delta.add(b.driver_id, b.quali_pos - a.quali_pos)

    ent = pd.DataFrame(rows)
    lab = results[["season", "round", "driver_id", "position", "finished", "points"]]
    ent = ent.merge(lab, on=["season", "round", "driver_id"], how="left")
    ent["win"] = (ent["position"] == 1).astype(float)
    ent["podium"] = (ent["position"] <= 3).astype(float)
    ent.loc[ent["position"].isna(), ["win", "podium"]] = np.nan
    ent["race_key"] = ent["season"] * 100 + ent["round"]
    return ent


# --------------------------------------------------------------- models

def _softmax_by(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.empty_like(scores, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        s = scores[m] - scores[m].max()
        e = np.exp(s)
        out[m] = e / e.sum()
    return out


class ConditionalLogit:
    """Softmax-within-race linear model with an L2 penalty, fitted by L-BFGS."""

    def __init__(self, features: list[str], l2: float = 1.0):
        self.features, self.l2 = features, l2

    def fit(self, df: pd.DataFrame) -> "ConditionalLogit":
        df = df[df["win"].notna()]
        X = df[self.features].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        X = (X - self.mu) / self.sd
        y = df["win"].to_numpy(float)
        groups = df["race_key"].to_numpy()
        # Precompute race boundaries for a fast vectorised softmax.
        order = np.argsort(groups, kind="stable")
        X, y, groups = X[order], y[order], groups[order]
        starts = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
        n_races = len(starts)

        def loss(w):
            s = X @ w
            mx = np.maximum.reduceat(s, starts)
            e = np.exp(s - np.repeat(mx, np.diff(np.r_[starts, len(s)])))
            z = np.add.reduceat(e, starts)
            logz = np.log(z) + mx
            p = e / np.repeat(z, np.diff(np.r_[starts, len(s)]))
            nll = -(y @ s) + logz.sum()
            grad = -(X.T @ y) + X.T @ p
            return (nll + 0.5 * self.l2 * w @ w) / n_races, (grad + self.l2 * w) / n_races

        res = minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        self.w = res.x
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = (df[self.features].to_numpy(float) - self.mu) / self.sd
        return _softmax_by(X @ self.w, df["race_key"].to_numpy())

    def coefficients(self) -> dict:
        return {f: round(float(w), 3) for f, w in zip(self.features, self.w)}


class RankGBM:
    """LightGBM LambdaRank on finishing order; scores -> softmax with a fitted
    temperature so they become probabilities."""

    def __init__(self, features: list[str]):
        self.features = features

    def _fit_ranker(self, df):
        import lightgbm as lgb
        rel = np.select([df["position"] == 1, df["position"] <= 3, df["position"] <= 10], [3, 2, 1], 0)
        est = lgb.LGBMRanker(n_estimators=200, learning_rate=0.03, num_leaves=7,
                             min_child_samples=40, subsample=0.8, subsample_freq=1, verbose=-1)
        est.fit(df[self.features], rel, group=df.groupby("race_key", sort=True).size().values)
        return est

    def fit(self, df: pd.DataFrame) -> "RankGBM":
        df = df[df["position"].notna()].sort_values("race_key")
        # Ranking scores are not probabilities. Learn the softmax temperature
        # on races the ranker did NOT see (the last 20%), otherwise it is
        # tuned on over-fitted in-sample scores and ends up overconfident.
        keys = df["race_key"].unique()
        cut = keys[int(len(keys) * 0.8)]
        fit_part, cal = df[df["race_key"] < cut], df[df["race_key"] >= cut]
        raw = self._fit_ranker(fit_part).predict(cal[self.features])
        g, y = cal["race_key"].to_numpy(), cal["win"].to_numpy()
        best = (np.inf, 1.0)
        for t in np.geomspace(0.05, 20, 60):
            p = _softmax_by(raw / t, g)
            best = min(best, (-np.mean(np.log(np.clip(p[y == 1], 1e-9, 1))), t))
        self.temp = best[1]
        self.est = self._fit_ranker(df)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _softmax_by(self.est.predict(df[self.features]) / self.temp, df["race_key"].to_numpy())


def make_model(kind: str, stage: str):
    if kind == "baseline":
        return ConditionalLogit(BASELINE[stage], l2=0.1)
    if kind == "clogit":
        return ConditionalLogit(STAGES[stage])
    if kind == "gbm":
        return RankGBM(STAGES[stage])
    raise ValueError(kind)


def plackett_luce(p_win: np.ndarray, n_sims: int = 20000, seed: int = 0) -> dict[str, np.ndarray]:
    """Simulate finishing orders. The Gumbel trick: adding Gumbel noise to
    log-strengths and sorting gives exactly a Plackett-Luce sample."""
    rng = np.random.default_rng(seed)
    logw = np.log(np.clip(p_win, 1e-9, 1))
    keys = logw[None, :] + rng.gumbel(size=(n_sims, len(p_win)))
    ranks = np.argsort(np.argsort(-keys, axis=1), axis=1) + 1
    return {"p_podium": (ranks <= 3).mean(0), "p_points": (ranks <= 10).mean(0),
            "exp_pos": ranks.mean(0)}


# ------------------------------------------------------------- backtest

def walk_forward(ent: pd.DataFrame, first_test_season: int, kinds=("baseline", "clogit", "gbm"),
                 refit_every: int = 4) -> pd.DataFrame:
    """For each test race, fit on races strictly before it (refitting every
    few rounds to save time) and predict. Returns per-driver predictions."""
    done = ent[ent["win"].notna()]
    out = []
    for stage in STAGES:
        models, last_fit = {}, None
        for rk in sorted(done.loc[done["season"] >= first_test_season, "race_key"].unique()):
            rnd = rk % 100
            if last_fit is None or rnd == 1 or rnd - last_fit >= refit_every or rnd < last_fit:
                train = done[done["race_key"] < rk]
                models = {k: make_model(k, stage).fit(train) for k in kinds}
                last_fit = rnd
            test = done[done["race_key"] == rk].copy()
            for k, m in models.items():
                test[f"p_{k}"] = m.predict(test)
            test["stage"] = stage
            out.append(test)
    return pd.concat(out, ignore_index=True)


def evaluate(bt: pd.DataFrame, kinds=("baseline", "clogit", "gbm")) -> dict:
    res = {}
    for stage, g in bt.groupby("stage"):
        res[stage] = {}
        winners = g[g["win"] == 1]
        for k in kinds:
            col = f"p_{k}"
            hit = g.loc[g.groupby("race_key")[col].idxmax()]
            res[stage][k] = {
                "races": int(g["race_key"].nunique()),
                "winner_log_loss": round(float(-np.log(np.clip(winners[col], 1e-9, 1)).mean()), 4),
                "win_brier": round(float(((g[col] - g["win"]) ** 2).mean()), 5),
                "favourite_won": round(float((hit["win"] == 1).mean()), 4),
                "calibration": calibration_bins(g[col], g["win"], n_bins=10),
                "by_season": {int(s): round(float(-np.log(np.clip(w[col], 1e-9, 1)).mean()), 4)
                              for s, w in winners.groupby("season")},
            }
    return res
