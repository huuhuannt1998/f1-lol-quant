"""Paths and tunable settings in one place.

Everything a learner might want to tweak (Elo K-factor, EWMA half-lives,
paper-trading thresholds) lives here so experiments don't require hunting
through the code.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Raw downloads + HTTP cache. Large and never committed. Point it at the SSD
# (or anywhere) with F1LOL_CACHE=/path/to/cache.
CACHE = Path(os.environ.get("F1LOL_CACHE", ROOT / "cache"))
# JSON files the web page reads. Committed by the GitHub Action.
SITE_DATA = ROOT / "data"
# Append-only forecast record. Committed, never rewritten. The hourly GitHub
# Action is its only writer: it sets F1LOL_LEDGER_DIR to the repo's ledger/.
# Runs on your Mac default to cache/ledger_local/, so a local run never makes
# the committed CSVs diverge from the Action's (which would conflict on pull).
LEDGER_DIR = Path(os.environ.get("F1LOL_LEDGER_DIR") or CACHE / "ledger_local")
# Your own picks: one CSV per pick, committed to the repo from the page
# (pipeline/picks.py). Always the repo's folder, wherever the pipeline runs.
PICKS_DIR = ROOT / "ledger" / "my_picks"

USER_AGENT = "f1-lol-quant/0.1 (personal research project; github pages)"

# --------------------------------------------------------------------- LoL
LOLESPORTS_API = "https://esports-api.lolesports.com/persisted/gw"
# Public key used by lolesports.com's own front end (not a secret).
LOLESPORTS_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
# Match history used to build ratings. Older seasons add little once ratings
# have converged, and every extra page is another request.
LOL_HISTORY_START = "2024-01-01"
# Leagues to pull. Unknown slugs are skipped with a warning. Small regional
# leagues are worth including: they are where Worlds' minor-region teams
# build up their ratings.
LOL_LEAGUE_SLUGS = [
    "worlds", "msi", "first_stand", "americas_cup",
    "lck", "lpl", "lec", "lcs", "cblol-brazil", "lcp",
    "lta_n", "lta_s", "lta_cross",            # 2025-only LTA era
    "pcs", "vcs", "ljl-japan", "lla",
    "lck_challengers_league", "ldl", "emea_masters", "nacl",
    "lfl", "superliga", "prime_league", "tcl",
]
# Known renames so a team keeps its rating across sponsor changes.
# normalized-old-name -> normalized-new-name
LOL_TEAM_ALIASES = {
    "fredit brion": "oksavingsbank brion",
    "brion": "oksavingsbank brion",
    "kwangdong freecs": "dn freecs",
    "dn freecs": "dn soopers",
    "fearx": "bnk fearx",
}

# Second-tier leagues: national/regional leagues (ERLs) and academy leagues,
# plus PCS/VCS/LJL from 2025 on, when they became feeders of the LCP. Value =
# first year the league counts as tier-2 (None = always). Their teams almost
# never meet major-region teams, so their Elo floats in a closed pool and is
# not comparable with LCK/LPL ratings (docs/LEARNING.md §2).
LOL_TIER2_LEAGUES = {
    "lfl": None, "emea_masters": None, "superliga": None, "prime_league": None,
    "tcl": None, "lck_challengers_league": None, "ldl": None, "nacl": None,
    "pcs": 2025, "vcs": 2025, "ljl-japan": 2025,
}
# False: keep tier-2 matches for ratings, training and forecasts, and only
# leave tier-2 teams out of the power rankings (a cross-pool comparison).
# True: drop tier-2 matches entirely. Decide with `python -m pipeline backtest lol`.
LOL_DROP_TIER2 = False
# A match still not completed this long after its scheduled start is "stale"
# (postponed, cancelled, or never updated by lolesports) and ignored everywhere.
LOL_STALE_HOURS = 48

ELO_INIT = 1500.0
ELO_K = 24.0                 # tuned by `python -m pipeline backtest`
ELO_SCALE = 400.0
ELO_SEASON_REGRESS = 0.25    # shrink toward league mean each new year
# International games carry extra info: they are the only link between regions.
# 3.0 from `python -m pipeline backtest lol` (2026-10-21): Elo log loss on
# international matches *before* the window improves 1.5 -> 3 (0.668 -> 0.653)
# and is flat beyond (x4 0.664, x6 0.652), while domestic Elo log loss worsens
# (0.558 -> 0.564 -> 0.582). The window agreed for the live gbm (intl 0.622 ->
# 0.604, majors 0.607 -> 0.604), on only 76 international matches.
ELO_INTL_K_MULT = 3.0

# Oracle's Elixir (optional enrichment): public Google Drive folder that the
# OE downloads page links to. If gdown cannot fetch it, drop the yearly CSVs
# into cache/oe/ by hand and everything else keeps working.
OE_DRIVE_FOLDER = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
OE_YEARS = [2024, 2025, 2026]
# OE spells some teams differently from lolesports. normalized OE name ->
# normalized lolesports name; used only to join OE stats (Elo is unaffected).
# The run log lists lolesports names still missing from OE.
OE_TEAM_NAMES = {
    "gen g": "gen g esports",
    "jd gaming": "beijing jdg esports",
    "weibo gaming": "weibogaming",
    "team we": "xi an team we",
    "ninjas in pyjamas": "shenzhen ninjas in pyjamas",
    "nongshim redforce": "nongshim red force",
    "kt rolster challengers": "kt challengers",
    "dn soopers challengers": "dns challengers",
    "kiwoom drx challengers": "krx challengers",
    "dplus kia challengers": "dk challengers",
    "hanwha life esports challengers": "hle challengers",
    "hanjin brion challengers": "bro challengers",
    "team liquid": "team liquid alienware",
    "cloud9": "cloud9 kia",
    "lng esports": "suzhou lng esports",
    "red canids": "red kalunga",
    "thundertalk gaming": "thunder talk gaming",
    "deep cross gaming": "relove deep cross gaming",
    "ici japon corp esport": "ici japon corp",
    "team heretics academy": "heretics academy",
}
# F1LOL_OE_DOWNLOAD=0 turns the Drive download off (the GitHub Action does
# this: Drive's per-file quota makes hourly downloads unreliable). Files
# already in cache/oe/ are still used.
OE_AUTO_DOWNLOAD = os.environ.get("F1LOL_OE_DOWNLOAD", "1") != "0"

# ---------------------------------------------------------------------- F1
JOLPICA_API = "https://api.jolpi.ca/ergast/f1"
F1_FIRST_SEASON = 2014          # start of the hybrid era
F1_EWM_HALFLIFE = 5.0           # races
F1_REG_RESET_SEASONS = {2014, 2022, 2026}   # big regulation changes

# ------------------------------------------------------------------ Kalshi
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_LOL_SERIES = "KXLOLGAME"      # one event per match, one market per team
KALSHI_LOL_OUTRIGHT = "KXLOL"        # tournament champion
KALSHI_F1_SERIES = "KXF1RACE"        # race winner, one market per driver

# Paper trading: a hypothetical $1 contract is "bought" when the model's
# probability beats the price by at least this much after fees.
PAPER_EDGE_MIN = 0.05
# Kalshi's general trading fee is roughly 0.07 * P * (1 - P) per contract.
# Check the current fee schedule before trusting P&L numbers.
KALSHI_FEE_RATE = 0.07

MODEL_VERSION = "v0.1"
