# NBA Match Predictor — Roadmap

## Status (2026-09-27): v3 shipped on branch `v2`
v3 is built and evaluated. Results are in `analysis/V3_RESULTS.md`, and the design is in
`docs/MODEL_V3_DESIGN.md`:
- player ratings from about 423k play-by-play stints
- availability from official injury reports (LightGBM)
- minutes redistribution
- the ridge combiner
- the Streamlit app, with injury-aware predictions and a what-if explorer

Live accuracy is 66.9% over the 2021-25 walk-forward, and 67.3% on the 2025-26 holdout (log-loss 0.594).

### Next up
- [ ] **Deploy:** connect Streamlit Community Cloud to the repo (main file `app/streamlit_app.py`).
- [ ] **Schedule `scripts/daily_update.py`:** e.g. Windows Task Scheduler at 11:00 and 17:30 ET, then commit and
      push `data/app/`. The season starts 2026-10-20.
- [ ] **Betting-line benchmark:** historical closing lines, to report accuracy against Vegas.
- [ ] **Playoff matchups:** re-test matchup & style on playoffs once 2026-27 adds about 85 more playoff games.
      It currently shows a non-significant hint (1.5 SE).
- [ ] **Dated rosters:** season-by-season transaction logs would remove the small "newcomer" backtest bias.
- [ ] **Late scratches:** poll the injury report closer to tip-off on game days.

---

## Original plan

Stack: **nba_api** (data) · **pandas / scikit-learn / LightGBM** (model) · **Streamlit** (app).
Working mode (updated): Claude built phases 0–5; Albert reviews.

## Where v1 stands (reproduce with `analysis/audit_v1.py`)

Test window: 2024-02-01 → end of 2023-24 regular season (607 games).

| Model | Win acc | Score RMSE |
|---|---|---|
| Always pick home team | 51.7% | – |
| v1 notebook (RF classifier, per-row) | 61.7%* | 13.75 |
| v1 features + `RandomForestRegressor` | – | 12.78 |
| Plain MOV-adjusted Elo (no ML) | **63.3%** | – |
| Vegas closing line (typical) | ~68–70% | – |

\*Not a real per-game number: in 42% of games the model predicts both teams win or both lose.

Known v1 issues:
1. Every game appears twice, and the two predictions per game can contradict each other.
2. `RandomForestClassifier` is used for score prediction.
3. There are no opponent-form features; `team_code`, `opponent_code` and `arena_code` are ordinal ints.
4. Hyperparameters were tuned on the test set, and the notebook was run out of order.
5. There is only one season of data.

**Realistic target:** 66–68% accuracy, log-loss < 0.62, and calibrated probabilities. Treat anything above 72% as a leakage bug until proven otherwise.

---

## Phase 0: Repo hygiene
- [ ] Target layout:
  ```
  src/nbapred/        # importable package
    data/             # fetch + clean (nba_api)
    features/         # elo.py, form.py, rest.py, players.py
    models/           # train.py, predict.py, evaluate.py
  app/                # Streamlit (Home.py + pages/)
  data/raw/ data/processed/   # parquet; raw gitignored, processed small enough to commit
  analysis/           # notebooks + one-off scripts
  tests/
  ```
- [ ] Add `pyproject.toml` (or `requirements.txt`) with pinned versions.
- [ ] Move the v1 notebooks and CSVs into `legacy/` rather than deleting them. They're useful for the README's "v1 → v2" story.
- [ ] Decide what happens to the old `../mlMatchPredictor` folder; this clone is a superset of it.

**Done when:** `pip install -e .` works and `python -c "import nbapred"` runs.

## Phase 1: Data pipeline (nba_api)
- [ ] `fetch_games(season)`: use `leaguegamefinder.LeagueGameFinder(season_nullable="2023-24", league_id_nullable="00", season_type_nullable="Regular Season")`.
  - It returns one row per team per game.
  - Home and away come from `MATCHUP`: `"LAL vs. DEN"` means home, `"LAL @ DEN"` means away.
- [ ] `fetch_team_advanced(season)`: try `teamgamelogs.TeamGameLogs(season_nullable=..., measure_type_player_game_logs_nullable="Advanced")`. That gets a whole season of per-game advanced stats in one call, instead of 1,230 box-score requests. Check the parameter name against the nba_api version you install; it's oddly named.
- [ ] `fetch_player_logs(season)`: use `playergamelogs.PlayerGameLogs` (minutes and +/- per player per game). You'll need this in Phase 3.
- [ ] Cache every response to `data/raw/{endpoint}_{season}.parquet`, and never re-fetch a completed season. Add `time.sleep(0.6)` between calls and `timeout=60`.
- [ ] Build **`games.parquet`: one row per game**. Columns: `game_id, date, season, home_team, away_team, home_pts, away_pts`, plus the home and away advanced stats.
- [ ] Backfill 2015-16 → 2024-25, and include playoffs with an `is_playoff` flag.
- [ ] Write the tests:
  - every `game_id` appears exactly once
  - there are no ties
  - each season has ~1,230 regular-season games (2019-20 and 2020-21 are shorter, which is expected)

**Gotcha:** stats.nba.com often blocks or hangs requests from cloud IPs (GitHub Actions, Streamlit Cloud). Design for this now: fetching runs locally (or on a self-hosted runner) and the app only reads committed parquet or SQLite files.

## Phase 2: Evaluation harness (build before any new features)
- [ ] `evaluate.py` takes a `predict_fn(train_df, test_df) -> p_home_win, pred_margin, pred_total`.
- [ ] **Walk-forward by season:** train on seasons < S and test on S, for S in the last 4 seasons.
  - Tune on those folds only.
  - Keep **2024-25 as a final holdout** that you touch once.
- [ ] Metrics:
  - accuracy
  - **log-loss and Brier score** (these are primary; accuracy is noisy at n≈1,230)
  - margin MAE/RMSE and total RMSE
  - a calibration table (10 probability bins)
- [ ] Built-in baselines: home team wins, and Elo (next item). Every future model is reported against them.
- [ ] Optional but strong for the README: join historical closing spreads or moneylines (Kaggle has several NBA odds datasets) and report accuracy vs. Vegas.

**Done when:** one command prints a table of baseline vs. model per season.

## Phase 3: Features (every one must use only info available before tip-off)
Leakage rule: every rolling feature needs `shift(1)` inside the team group. Write a test that shuffles future rows and asserts the past features don't change.

- [ ] **Elo:** K≈20, home advantage ≈ 70–100 Elo points, MOV multiplier, and regress 25% toward the mean between seasons. Use it both as a standalone baseline and as a feature (`elo_diff`, `elo_win_prob`).
- [ ] **Form:** keep your exponential decay, but vectorise it (`groupby("team")[cols].transform(lambda s: s.shift(1).ewm(alpha=...).mean())`, see `audit_v1.py`).
  - Add a season-to-date version and a slow/fast pair (e.g. half-lives of 5 and 20 games).
  - Decide how the first ~10 games of a season borrow from last season.
- [ ] **Matchup framing:** features are **home minus away** differentials (plus a few raw levels, e.g. pace, for totals). Drop `team_code`, `opponent_code` and `arena_code`.
- [ ] **Schedule:** rest days per team, back-to-back flags, 3-in-4 nights, games in the last 7 days, and travel distance since the last game (a lookup table of arena lat/lon).
- [ ] **Player availability (biggest gain):**
  - For each team-game, compute a minutes-weighted rating (e.g. rolling +/- per 36 or on/off) of the players who actually played, vs. the team's usual rotation.
  - Historically, "who played" is known from player logs. For live predictions you need the official injury report. nba_api doesn't have it; look at the NBA's injury report PDFs or the `nbainjuries` package.
  - Do this last in the phase; it's the hardest.
- [ ] Log each feature group's effect on walk-forward log-loss (an ablation table for the README).

## Phase 4: Models
- [ ] **Predict margin and total, not two scores and a separate win label.**
  - `margin = home − away` (regression) and `total` (regression).
  - Scores are `(total ± margin) / 2`.
  - `p_home_win = Φ(pred_margin / σ)`, where σ is the residual std from validation (≈12).
  - This guarantees the scoreline and the winner agree.
  - Also try a direct classifier for `p_home_win` and compare log-loss.
- [ ] Models to compare:
  - logistic/ridge regression (strong on differentials; likely hard to beat)
  - LightGBM with early stopping on the validation fold
  - an ensemble of the two
- [ ] Calibrate with `CalibratedClassifierCV` (isotonic or sigmoid) if the reliability table is off.
- [ ] Use SHAP (or coefficients for the linear model) for per-prediction explanations. The app needs this.
- [ ] `train.py` saves the model, feature list, σ, training date and validation metrics to `models/model_v2.joblib` plus `metrics.json`.

**Done when:** the 2024-25 holdout is evaluated once and the number goes in the README.

## Phase 5: Streamlit app
Architecture: `scripts/daily_update.py` fetches yesterday's results and today's schedule, updates Elo and features, and writes `data/processed/predictions.parquet`. Run it locally or on a schedule, then commit and push. The app **only reads files** (`@st.cache_data`), because of the cloud-IP gotcha above.

- [ ] Today's schedule: `nba_api.live.nba.endpoints.scoreboard.ScoreBoard()` for today. For future dates, check whether your nba_api version has `ScheduleLeagueV2`.
- [ ] **Pages** (`app/pages/` or `st.navigation`):
  1. **Today:** a card per game with the win-probability bar, predicted score, and the top 3 factors from SHAP.
  2. **Matchup explorer:** pick any two teams and a home side, and see the prediction and the factor breakdown.
  3. **Teams:** Elo history chart across seasons and current form stats.
  4. **Track record:** the running accuracy, log-loss and calibration of logged predictions. Only log predictions made *before* the games, and add Vegas if you have lines. This page is what makes the project credible.
  5. **About / Methodology:** data, features, validation, the v1 → v2 comparison, and limitations.
- [ ] Deploy to Streamlit Community Cloud from the GitHub repo. You'll need to make the repo public, or use a private-app slot.
- [ ] Update the README: a screenshot, the live link, the results table and an architecture sketch.

---

## Review checkpoints
Ping Claude for a review at the end of each phase, especially:
- **after Phase 1:** data correctness (duplicates, home/away mapping, season edges)
- **after Phase 3:** leakage audit
- **before the holdout run in Phase 4:** make sure nothing was tuned on it
