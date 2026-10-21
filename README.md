# F1 & LoL Quant

A static GitHub Pages site plus a Python pipeline. It forecasts **Formula 1 race winners** and **pro League of Legends match winners**, puts each forecast next to the **Kalshi** price, and writes every forecast to an append-only ledger *before* the event starts. Afterwards it scores the model against the market.

This is a personal learning project: free data, read-only market data, no order placement. It is not financial advice.

```
GitHub Action (hourly) ──► python -m pipeline update ──► data/*.json + ledger/*.csv ──► commit ──► GitHub Pages
                               │
                               ├─ LoL: lolesports.com schedule/results ─► Elo ─► logreg / LightGBM ─► series win %
                               │        (+ Oracle's Elixir rolling game stats, optional)
                               ├─ F1:  Jolpica results & qualifying ─► conditional logit ─► Plackett–Luce sims
                               └─ Kalshi public prices (KXLOLGAME, KXF1RACE) ─► edge, paper trades
```

## Pages

| Tab | What it shows |
|---|---|
| League of Legends | Upcoming matches (14 days) with model win %, Kalshi price and edge; recent results; Elo power rankings |
| Formula 1 | Next race: win / podium / points % per driver vs Kalshi; last race vs the pre-race forecast; standings; calendar |
| Track record | Ledger only: settled forecasts, model vs market log loss and Brier, calibration chart, paper-trading P&L |
| Method | Walk-forward backtest tables for every model, the live F1 model weights, data credits |

## How the models are trained

Nothing here comes from another site's predictions. The APIs supply only schedules, results and game statistics. Every forecast comes from models trained in this repo, from scratch, on every run. The ideas are explained step by step, with exercises, in [docs/LEARNING.md](docs/LEARNING.md). This section is the map.

### The loop, every hour

The GitHub Action runs `python -m pipeline update` at :17 past each hour. Each run:

1. **Fetches** new schedules and results (lolesports, Jolpica, optionally Oracle's Elixir).
2. **Rebuilds every feature from scratch** by replaying history in time order. Each match or race gets features computed only from what had happened *before* it (no leakage, see below).
3. **Tunes** the few hyperparameters (Elo's K and off-season shrink) on data from *before* the evaluation window.
4. **Backtests** every candidate model walk-forward on the most recent period and scores it.
5. **Picks** the model with the best backtest log loss, refits it on *all* finished events, and forecasts the upcoming ones.
6. **Logs** each forecast to `ledger/predictions.csv` before the event starts, and records results as they come in (`ledger/results.csv`).

So you don't retrain anything by hand. As games and races finish, the next hourly run includes them: Elo moves, form averages update, the models are refit, and the backtest and ledger numbers on the Track record and Method tabs change with them.

### League of Legends: Elo, then a classifier on top

**Target.** `y = 1` if team 1 wins the *series*. Kalshi and the page both work at the series level.

**Training data.** About 28,800 completed pro series since 2014: from October 2023 the lolesports schedule (21 leagues), and before that series rebuilt from Oracle's Elixir game files (`pipeline/lol/oe_history.py`, committed as `history/lol_oe_series.csv`; 1,164 MSI and Worlds series). The older history cut international log loss from 0.622 to 0.599. Leagues include (Worlds, MSI, First Stand, LCK, LPL, LEC, LCS, CBLOL, LCP and second-tier leagues). The source is lolesports.com's schedule.

**Step 1: Elo ratings** (`pipeline/lol/elo.py`). This is online learning: one pass through history, with no separate fitting step.
- P(A beats B in one game) = 1 / (1 + 10^((R_B − R_A)/400)).
- After every match, each game counts: R_A += K × (games A won − games played × p). A 3–0 therefore moves ratings more than a 3–2.
- K = 24 and off-season shrink = 0.5 are chosen by grid search (K ∈ {16, 24, 32, 40}, shrink ∈ {0.15, 0.3, 0.5}). The search scores log loss on matches *before* the backtest window.
- International games use K × 3.0 (`ELO_INTL_K_MULT`). They are the only evidence of how regions compare. The multiplier was chosen with `python -m pipeline backtest lol`, again on pre-window data.
- A new team starts at its league's average rating. Each new year, ratings shrink part-way toward the league mean (roster changes).
- A game probability becomes a series probability with the negative-binomial formula (`p_series`): a 60% game edge is 64.8% in a Bo3 and 68.3% in a Bo5.

**Step 2: point-in-time features** (`pipeline/lol/model.py::historical_features`). Each is computed from the state *before* the match:

| Feature | Meaning |
|---|---|
| `elo_diff` | team 1's rating minus team 2's |
| `elo_logit` | log-odds of Elo's series probability |
| `form_diff` | difference in win rate over each team's last 10 games |
| `exp_min` | log(1 + games played) of the less experienced team (new teams are less certain) |
| `rest_diff` | days since each team's last match, capped at 30 |
| `intl`, `best_of` | international event flag; Bo1/Bo3/Bo5 |
| `oe_*_diff` (optional) | Oracle's Elixir in-game stats, as exponentially weighted averages over past games (half-life 8 games): gold and XP difference at 15 min, first dragon, first tower, game result. Joined *as of* each match's start, never including that match. |

**Step 3: three candidate models** (`MatchModel`):
- **elo**: the Elo series probability as is. No fitting; it's the baseline to beat.
- **logreg**: standardized features → L2-regularized logistic regression (C = 0.5). It's linear, so it can only re-weight features.
- **gbm**: LightGBM gradient-boosted trees (250 trees, learning rate 0.03, 15 leaves, at least 40 samples per leaf, 80% row and column subsampling, L2 = 1). It can learn interactions, for example that form matters more in a Bo1. The settings are deliberately conservative because a few thousand rows overfit easily.
- **Symmetry trick:** "team 1" is an arbitrary label. Every match is trained from both sides (antisymmetric features negated, label flipped), and the two views are averaged at prediction time. P(A beats B) + P(B beats A) is then exactly 1.

**Step 4: walk-forward validation.** For each month of the last year, fit on every match *before* that month and predict that month (about 2,000 predictions). A random train/test split would let the model see the future (same teams, same patch) and flatter every number.

### Formula 1: a race is a multi-way contest

**Target.** Who wins the race. The model needs probabilities for all 20+ drivers that sum to 1.

**Training data.** Every race and qualifying session since 2014 (275 races) from Jolpica, the Ergast successor.

**Two forecast moments.** `pre_quali` (form only) and `post_quali` (adds the grid). They are separate models because they may use different information.

**Features** (`pipeline/f1/model.py::build_entries`). Exponentially weighted averages with a half-life of 5 races, shrunk toward a prior between seasons, and shrunk harder in regulation-reset years (2014, 2022, 2026):
- the driver's recent finishing position, points and qualifying position;
- the constructor's recent points and qualifying;
- DNF rate (half-life 10);
- finishing record at this circuit (last 5 visits, shrunk toward overall form);
- qualifying gap to the teammate;
- log(starts), as experience;
- an early-season flag in reset years.

After qualifying it adds log(qualifying position), the gap to pole in %, and a front-row flag.

**Models:**
- **clogit**, a conditional logit. Each driver gets a score s = w·x and P(win) = exp(s_i) / Σ_j exp(s_j) *within the race* (a softmax). Training maximizes the log-probability of the actual winner, with an L2 penalty, using L-BFGS on standardized features.
- **baseline**: the same model with one feature (constructor form before qualifying, qualifying position after).
- **gbm**: LightGBM LambdaRank, a learning-to-rank model, trained on finishing order (relevance 3 for the win, 2 for the podium, 1 for the points). A softmax temperature turns its scores into probabilities; the temperature is fitted on races the ranker did not train on.

**Simulation.** Podium and points chances come from 20,000 simulated finishing orders: the Plackett–Luce model, sampled with the Gumbel trick.

**Validation.** Walk-forward over every race since 2022 (109 races), refitting every 4 rounds. The main metric is *winner log loss*: the average −log(probability given to the actual winner). Guessing uniformly over 20 drivers scores 3.0.

### How a model is judged (and chosen)

- **Log loss** punishes confident mistakes hard. Lower is better; a coin flip scores 0.693.
- **Brier score** is the mean squared error of the probability.
- **Calibration**: when the model says 70%, does that side win about 70% of the time? The *calibration slope* puts this in one number: 1 means calibrated, above 1 means under-confident.
- **Accuracy** comes last. It ignores *how* confident the model was.

The live model is whichever has the best backtest log loss: LoL on the whole window, F1 averaged over the last three seasons. That is a mild selection bias, and it is reported, not hidden.

Live numbers on 2026-10-21 (they change every hour; see the Method tab):

| | Model | Log loss | Brier | Accuracy / favourite won |
|---|---|---|---|---|
| LoL (2,029 matches) | Elo | 0.621 | 0.214 | 65.7% |
| | logreg | 0.622 | 0.214 | 65.6% |
| | **gbm (live)** | **0.613** | **0.212** | **66.5%** |
| F1 after quali (109 races) | **baseline (live)** | 1.334 | – | 59.6% |
| | clogit | 1.111 | – | 61.5% |
| F1 before quali | **clogit (live)** | **1.527** | – | 44.0% |
| | baseline | 1.790 | – | 64.2% |

Two lessons are hiding in that table:
- **Accuracy and log loss can disagree.** Before qualifying, the one-feature baseline picks the winner more often (64%), but its probabilities are worse (log loss 1.79 vs 1.53). Probabilities, not picks, are what a forecast is judged on.
- **The selection rule matters.** After qualifying, clogit is better over all 109 races, but the live choice uses only the last three seasons, where the baseline did slightly better.

With Oracle's Elixir in-game stats the LoL gbm improved to log loss 0.609 (international 0.602) in a local test; the Action uses them too since 2026-10-21.

### Your own picks

Each upcoming match card and the next F1 race have a **My pick** control: choose who wins and how sure you are, then press *Save on GitHub*. GitHub's new-file editor opens with a one-line pick file for `ledger/my_picks/`, and pressing *Commit changes* locks it in. The commit time is the lock: a pick committed after the start never counts. The Track record tab then scores you exactly like the model, and after about 100 settled picks a "model + you" blend can be tested (LEARNING.md §11).

### Why the numbers can be trusted, and where they can't

- **No leakage.** Tests in `tests/test_pipeline.py` change a future result and check that no earlier feature moves.
- **Walk-forward only.** No random splits; hyperparameters are tuned on data before the evaluation window.
- **The ledger** is the honest scoreboard. A forecast is written down *before* the event and scored after it, so it can't be fooled by leakage or over-tuning. Compare it with the backtest once a few weeks of results are in.
- **Weak spots.** Few international matches (cross-region strength rests on about 75 matches a year at MSI, Worlds and First Stand); roster changes; F1's 2026 regulation reset (little data); grid penalties not modelled.

## Run it on your Mac

Shortest path, from the VS Code terminal: `bash run.sh` (set up + fetch + forecast), then `bash run.sh serve`. The manual steps are:

```bash
cd /Volumes/ResearchSSD/F1-LoLQuant
python3 -m venv .venv && source .venv/bin/activate
brew install libomp                 # LightGBM needs OpenMP on macOS
pip install -r requirements.txt

python -m pipeline demo             # synthetic data, no network: check everything works
python -m pipeline update           # real data: first run back-fills (~5–10 min), later runs ~1 min
python -m pytest -q                 # correctness + leakage tests

python3 -m http.server 8000         # then open http://localhost:8000
```

Raw downloads go to `cache/`, which is ignored by git and lives on the SSD. Set `F1LOL_CACHE=/somewhere/else` to move it. `python -m pipeline update --offline` re-runs the models on cached data with no network. `--sport lol` or `--sport f1` runs one sport only.

The demo writes to `cache/demo_site/`, so it never touches the real `data/` or `ledger/`.

Once the site is published, the hourly GitHub Action owns the committed `data/` and `ledger/`. Local runs write their forecasts to `cache/ledger_local/` instead (set `F1LOL_LEDGER_DIR` to change that), so the committed ledger never diverges from the Action's. They still rewrite `data/*.json` for a local preview, so run `git restore data` before `git pull`.

## Publish on GitHub Pages

1. Create an empty public repo (for example `f1-lol-quant`) and push this folder to `main`.
2. Go to **Settings → Pages → Build and deployment**, set **Source: Deploy from a branch**, then pick **main** and **/ (root)**.
3. Go to **Actions → Update forecasts → Run workflow** once. After that it runs hourly at :17.
4. The site appears at `https://<user>.github.io/<repo>/`.

The workflow caches raw data between runs and commits only `data/` and `ledger/`. Public repos get unlimited Actions minutes.

## Layout

```
index.html, static/          the page (plain HTML/CSS/JS, no build step)
data/                        JSON the page reads (written by the pipeline)
ledger/predictions.csv       every forecast, appended, never edited
ledger/results.csv           outcomes
pipeline/
  config.py                  every tunable knob (Elo K, half-lives, paper-trade edge, fees)
  http.py                    cached, rate-limited GET with retries
  kalshi.py                  read-only market data
  ledger.py                  logging, closing-line scoring, paper trades
  lol/  lolesports.py  oracles_elixir.py  elo.py  model.py  run.py
  f1/   jolpica.py  model.py  run.py
  synthetic.py               fake data for `demo` and tests
tests/                       leakage + correctness tests
docs/LEARNING.md             the ML ideas in this repo, with exercises
docs/plans/                  the original planning documents (git-ignored, local only)
```

## Data sources & terms

- **Jolpica-F1** (Ergast successor). CC BY-NC-SA 4.0: non-commercial, attribution, share-alike. Rate limit is roughly 4 req/s burst and 500 req/hour, and `http.py` throttles to stay under it.
- **lolesports.com** schedule API. Unofficial and undocumented, the same JSON the website uses. It can change without notice.
- **Oracle's Elixir**. Free for fan and analyst use, with credit to Tim Sevenhuysen. The files are shared via Google Drive, and Drive often refuses them with "Too many users have viewed or downloaded this file recently". That is a per-file quota, not a bug here. After a refusal the pipeline waits 24 hours before asking again, and the GitHub Action never downloads OE (`F1LOL_OE_DOWNLOAD=0`). Everything works without OE. To add it by hand:
  1. Open the downloads page at oracleselixir.com and follow the link to the match-data folder on Google Drive. A browser signed in to Google usually gets through when `gdown` doesn't.
  2. Download `2024_LoL_esports_match_data_from_OraclesElixir.csv`, and the same for 2025 and 2026 (about 100 MB each).
  3. Put them in `cache/oe/` and keep the file names unchanged. Then `python -m pipeline update --offline --sport lol` picks them up, and the log prints the match coverage plus the team names that didn't join.
  4. Refresh the current-year file now and then; past years don't change.
- **Kalshi** public market-data endpoints. Read-only, no key needed.

Keep the project non-commercial while it runs on these sources. Selling forecasts would need licences (see `docs/plans/RaceIQ…`). Not affiliated with Formula 1, the FIA, Riot Games or Kalshi.

## Known limitations

- **LoL cross-region strength** comes only from MSI, Worlds and First Stand games, so Worlds forecasts between regions are the least reliable numbers on the site.
- **Roster changes**: a renamed or rebuilt team either keeps its old rating (via `LOL_TEAM_ALIASES`) or starts at its league average. Neither is ideal.
- **F1 2026 is a regulation reset**. The model shrinks constructor form at season start, but early-season forecasts lean on little data.
- **F1 grid penalties** aren't applied. Qualifying position stands in for grid.
- **No tournament simulation** yet. Kalshi's Worlds-champion market (`KXLOL`) isn't modelled; see the exercises in `docs/LEARNING.md`.
- **Paper P&L** assumes you could fill one contract at the displayed ask. Real liquidity on these markets is often only a few hundred dollars.
