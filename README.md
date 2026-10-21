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
