# Learning notes: the ML in this repo

Each section names one idea, where it lives in the code, and an exercise to try. Run `python -m pipeline update --offline` after a change to re-score everything on cached data. Use `python -m pipeline demo` if you have no real data yet.

## 1. Define the prediction moment first

A forecast is only meaningful relative to *when* it is made. LoL forecasts are "before the match starts". F1 has two moments: before qualifying (`pre_quali`) and after qualifying (`post_quali`). They are different models because they are allowed different inputs.

**Leakage** means using information that would not exist at prediction time. It is the most common way ML projects fool themselves. In sports it usually sneaks in through:
- rolling averages that include the current game,
- random train/test splits (the test set's teams and patch are already "seen"),
- tuning hyperparameters on the same period you report results for.

*Where:* `lol/model.py::historical_features` replays matches in time order and records each team's state **before** folding the result in. `f1/model.py::build_entries` does the same per race. `lol/oracles_elixir.py::attach` uses an as-of join with `allow_exact_matches=False`.

*Exercise:* break it on purpose. In `FormTracker.state`, call `update` before reading the state, then re-run the backtest. Accuracy jumps, and `tests/test_pipeline.py::test_lol_features_do_not_see_own_or_future_results` fails. That jump is what leakage looks like.

## 2. Ratings: Elo

`p(A beats B in one game) = 1 / (1 + 10^((R_B − R_A)/400))`. After each series, `R_A += K × (games A won − games played × p)`.

- **K** sets how fast ratings react. If it's too small, ratings lag roster changes. If it's too big, they chase noise.
- **League priors**: a new team starts at its league's average rating, not a global 1500.
- **Off-season regression** pulls ratings toward the league mean every January.

*Where:* `lol/elo.py`. `tune_elo` grid-searches K and the shrink amount on data **before** the evaluation window.

### Closed pools: why Galions once ranked above T1

Elo only measures strength *relative to the teams you played*. Points move between the two teams in a match, so a league that never plays anyone else is a closed pool: its average stays wherever it started, and its best team ends up as far above that average as its results allow. French LFL teams start at 1500 like everyone else and play almost only each other (and other national leagues at EMEA Masters). The LFL champion therefore reaches a rating similar to an LCK champion's, even though the two never met. That is how Galions and Solary ranked #2 and #6, above T1 and BLG.

The same goes for academy leagues (LCK Challengers, NACL) and, since 2025, PCS, VCS and LJL, which now feed the LCP. The list is `config.LOL_TIER2_LEAGUES`.

International events are the only bridges between pools: Worlds, MSI and First Stand set the relative strength of the LCK, the LPL and the LEC. Tier-2 pools have almost no bridges, so their ratings are not on the major-region scale at all.

Three fixes, each with a cost:

| Option | What it does | Cost |
|---|---|---|
| (a) drop tier-2 leagues (`LOL_DROP_TIER2 = True`) | ratings come only from leagues connected by international play | fewer training rows for logreg/gbm; no forecasts for tier-2 matches; a promoted team starts from its new league's average |
| (b) per-league offsets learned from inter-league games | the principled fix (hierarchical Elo) | tier-2 pools have so few bridge games that the offsets would be mostly guesswork |
| (c) keep tier-2 for ratings, training and forecasts, but leave tier-2 teams out of the power rankings | forecasts *inside* a pool are fine (both teams are on the same scale); only cross-pool comparisons were wrong | a promoted team (e.g. NACL → LCS) brings its tier-2 rating with it, which may be inflated |

The current setting is (c). It changes nothing the models see, so the backtest numbers cannot get worse, and the rankings no longer mix scales. Whether (a) is better is an empirical question about those promoted teams. `python -m pipeline backtest lol` answers it on the slices both options share: international matches and major-league matches. It makes the choice on Elo log loss from *before* the backtest window. The window numbers only check the choice. Picking whichever option scores best on the window would quietly tune on the test set (§1).

*Exercise:* run `python -m pipeline backtest lol`. Read the "Chosen on data before …" line, then check that the chosen row also holds up on the window's `intl` and `majors` columns. Look at `tier-2 teams in raw top 15` to see how inflated the closed pools are.

*Exercise:* add a K that decays with games played (high for new teams, low for veterans). Does log loss improve?

## 3. From game to series probability

If one game is won with probability *p* and games are independent, a best-of-N is a negative-binomial sum (`lol/elo.py::p_series`). A 60% game edge becomes 64.8% in a Bo3 and 68.3% in a Bo5. Kalshi's match markets settle on the **series**, so this conversion matters.

*Exercise:* games aren't really independent (momentum, side selection). Fit a "series factor" that scales p before conversion and see if the backtest likes it.

## 4. Models on top of Elo

`MatchModel` in `lol/model.py`:
- **logreg**: logistic regression. It's linear, so it can only re-weight features. Train on both orientations of every match (`_swap`) so "team1" carries no meaning, then average both views at prediction time.
- **gbm**: LightGBM can learn interactions (form matters more in Bo1), but it overfits easily on a few thousand rows. That's why it uses `min_child_samples=40`, a small `num_leaves` and subsampling.

The live model is whichever scored the best walk-forward log loss. That's a mild selection bias: report it, don't hide it.

*Exercise:* add a feature such as "side-selection advantage" or "patch age" in `_feature_row` and compare the backtest tables.

## 5. Multi-way races: the conditional logit

Twenty drivers, one winner. Each driver gets a linear score *s = w · x*, and `P(i wins) = softmax(s)_i` within the race. Training maximises the log-probability of the actual winner (`f1/model.py::ConditionalLogit`, with a hand-written gradient so you can read it). The probabilities always sum to 1, which lets you compare them directly with Kalshi's per-driver prices.

`RankGBM` uses LightGBM's LambdaRank to learn an *ordering*, then turns ranking scores into probabilities with a softmax **temperature**. The temperature is fitted on held-out races, because fitting it in-sample makes the model wildly overconfident. This is a calibration lesson in one line.

*Exercise:* add a "rain" feature (OpenF1 has weather) and see whether the post-qualifying model improves at wet races.

## 6. Simulation: Plackett–Luce

To get podium and points chances, `plackett_luce` samples full finishing orders. Adding Gumbel noise to log-strengths and sorting gives an exact Plackett–Luce sample (the "Gumbel-max trick"). One known weakness: it assumes a driver's strength is the same for P1 as for P10. In reality a DNF risk hits everyone equally.

*Exercise:* add an explicit DNF probability per driver (from `drv_dnf_rate`) before sampling the order. Compare podium calibration.

## 7. Scoring probabilities

- **Log loss** `−mean(y log p + (1−y) log(1−p))`. A coin flip scores 0.693. It punishes confident misses hard.
- **Brier score** is the mean squared error of the probability. A coin flip scores 0.25.
- **Calibration** asks whether your 70% forecasts win 70% of the time (`util.calibration_bins`, and the chart on the Track record tab).
- **F1 winner log loss** is `−mean log p(actual winner)`. A uniform guess over 20 drivers scores log 20 ≈ 3.0.

Accuracy alone is misleading for betting: you can be "right" 70% of the time and still lose money buying 80¢ favourites.

## 8. Backtesting vs. the ledger

**Walk-forward backtest** (`walk_forward` in both model files): for each month or race, fit on everything strictly before it, then predict it. It's honest about time, but it still uses today's code and today's hyperparameters.

The **ledger** (`pipeline/ledger.py`) is the real test. Forecasts are appended before the start and never edited, and scoring uses the last one before the start ("closing" forecast) next to the Kalshi price at that moment. If the model can't beat the market's log loss on the ledger, it has no edge, whatever the backtest says.

## 9. Edge, fees, and why the market is hard to beat

Buying YES at the ask *a* with true probability *p* has expected profit `p − a − fee` per $1 contract. The fee is roughly `0.07 × a × (1 − a)`; check Kalshi's current schedule. A model needs to be better than the *market*, not just better than Elo. Markets aggregate many people's models. Liquidity is thin too: around $1K traded on a typical Worlds match, so even a real edge only scales so far.

*Exercise:* change `PAPER_EDGE_MIN` in `config.py` and watch how the paper P&L and trade count trade off. Too low and fees eat you; too high and you never trade.

## 10. Ideas for next steps

1. **Worlds champion market (`KXLOL`)**: simulate the remaining bracket thousands of times using the match model, and compare with Kalshi's outright prices.
2. **OpenF1 practice pace**: long-run pace from FP2 is the classic pre-qualifying signal.
3. **Glicko-2** instead of Elo, so each rating carries its own uncertainty, which is useful right after roster changes.
4. **Draft-aware LoL model**: after champion select, before game 1. Needs pick/ban data from Oracle's Elixir.
5. **Bayesian shrinkage** for circuit history and head-to-head records.

## 11. You vs the model: is your judgement a feature?

The page lets you lock in your own pick (who wins, and how sure you are) before each match or race. It is saved as a small file in `ledger/my_picks/` and counts only if it was committed before the start (`pipeline/picks.py`). The Track record tab scores you exactly like the model: log loss, Brier, accuracy, on the same events.

Why give a **probability** and not just a team: a pick of "T1" can't tell a coin-flip hunch from a sure thing, so it can't be scored with log loss or combined with anything.

Can your picks improve the model? Only if they carry information the model doesn't have (roster news, the patch, draft habits), and that has to be *measured*, not assumed. The standard tool is a **blend**, also called stacking:

`P(win) = σ(a · logit(p_model) + b · logit(p_you))`

Fit `a` and `b` walk-forward, like everything else: for each month, use only picks that settled before it. If the blend's log loss beats the model's, your judgement adds information and the blend can go live. If `b` comes out near 0, the model already knows what you know. A 50/50 average of the two is not the same thing: if your picks are noisy, it makes forecasts worse.

You need roughly 100+ settled picks on events the model also forecast before `a` and `b` mean anything; the page counts them for you.

*Exercise:* once you have 100 picks, fit the blend with `sklearn.linear_model.LogisticRegression(fit_intercept=False)` on the two logits, walk-forward by month, and compare its log loss with the model's on the same events.
