"""Elo ratings for pro LoL teams, computed strictly in time order.

Why Elo first: it is the strongest single feature in most public LoL models,
it is easy to reason about, and it gives every other model a baseline to beat.

Key ideas (each one a knob in config.py):
  * One update per *game*, so a 3-0 says more than a 3-2.
  * New teams start at their league's average rating, not a global 1500.
    Otherwise a fresh LCK team and a fresh minor-region team look equal.
  * Ratings shrink part-way toward the league mean each new year (rosters
    change over the off-season).
  * International games move ratings more: they are the only evidence of how
    regions compare.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import comb

import numpy as np
import pandas as pd

from .. import config
from ..util import norm_name
from .lolesports import CROSS_LEAGUE, INTERNATIONAL


def team_key(name: str) -> str:
    k = norm_name(name)
    seen = set()
    while k in config.LOL_TEAM_ALIASES and k not in seen:
        seen.add(k)
        k = config.LOL_TEAM_ALIASES[k]
    return k


def p_game(r1: float, r2: float, scale: float = config.ELO_SCALE) -> float:
    return 1.0 / (1.0 + 10 ** ((r2 - r1) / scale))


def p_series(p: float, best_of: int) -> float:
    """Chance to win a best-of-N, assuming independent games with win prob p.
    Need w = ceil(N/2) wins before the opponent does: sum over the number of
    games k the opponent takes first."""
    w = best_of // 2 + 1
    return float(sum(comb(w - 1 + k, k) * p ** w * (1 - p) ** k for k in range(w)))


class EloBook:
    def __init__(self, k: float = config.ELO_K, regress: float = config.ELO_SEASON_REGRESS,
                 intl_mult: float = config.ELO_INTL_K_MULT):
        self.k, self.regress, self.intl_mult = k, regress, intl_mult
        self.rating: dict[str, float] = {}
        self.games: Counter = Counter()
        self.year_seen: dict[str, int] = {}
        self.home: dict[str, Counter] = defaultdict(Counter)
        self.last_league: dict[str, str] = {}   # most recent domestic league
        self.display: dict[str, str] = {}
        self.history: dict[str, list] = defaultdict(list)   # (time, rating)

    # ---- league bookkeeping
    def home_league(self, t: str) -> str | None:
        c = self.home.get(t)
        return c.most_common(1)[0][0] if c else None

    def league_mean(self, league: str | None) -> float:
        if not league:
            return config.ELO_INIT
        vals = [r for t, r in self.rating.items() if self.home_league(t) == league]
        return float(np.mean(vals)) if len(vals) >= 3 else config.ELO_INIT

    def _ensure(self, t: str, league: str, when: pd.Timestamp) -> None:
        if t not in self.rating:
            prior_league = None if league in INTERNATIONAL else league
            self.rating[t] = self.league_mean(prior_league)
            self.year_seen[t] = when.year
        elif self.year_seen.get(t) != when.year:
            m = self.league_mean(self.home_league(t))
            self.rating[t] += self.regress * (m - self.rating[t])
            self.year_seen[t] = when.year
        if league not in INTERNATIONAL:
            self.home[t][league] += 1
            if league not in CROSS_LEAGUE:
                self.last_league[t] = league

    def get(self, name: str, league: str | None = None) -> float:
        t = team_key(name)
        if t in self.rating:
            return self.rating[t]
        return self.league_mean(league if league not in INTERNATIONAL else None)

    # ---- the update
    def update(self, row) -> dict:
        when = pd.Timestamp(row.start_time)
        a, b = team_key(row.team1), team_key(row.team2)
        self.display[a], self.display[b] = row.team1, row.team2
        self._ensure(a, row.league, when)
        self._ensure(b, row.league, when)
        ra, rb = self.rating[a], self.rating[b]
        pg = p_game(ra, rb)
        pre = {"r1_pre": ra, "r2_pre": rb, "p_game": pg,
               "p_series_elo": p_series(pg, int(row.best_of)),
               "n1_pre": self.games[a], "n2_pre": self.games[b]}

        w1, w2 = row.t1_wins, row.t2_wins
        if pd.isna(w1) or pd.isna(w2) or (w1 + w2) == 0:
            w1, w2 = (1, 0) if row.winner == 1 else (0, 1)
        n = w1 + w2
        k = self.k * (self.intl_mult if row.league in INTERNATIONAL else 1.0)
        delta = k * (w1 - n * pg)
        self.rating[a] = ra + delta
        self.rating[b] = rb - delta
        self.games[a] += n
        self.games[b] += n
        self.history[a].append((row.start_time, self.rating[a]))
        self.history[b].append((row.start_time, self.rating[b]))
        return pre


def run_elo(matches: pd.DataFrame, **kw) -> tuple[pd.DataFrame, EloBook]:
    """Replay every completed match in time order. Returns the matches with
    the ratings each team had *before* the match (safe to use as features)."""
    done = matches[(matches["state"] == "completed") & matches["winner"].notna()].copy()
    done = done.sort_values("start_time").reset_index(drop=True)
    book = EloBook(**kw)
    pre = [book.update(r) for r in done.itertuples(index=False)]
    return pd.concat([done, pd.DataFrame(pre)], axis=1), book
