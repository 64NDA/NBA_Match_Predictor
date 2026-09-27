# NBA Match Predictor

Predicts NBA game winners and scorelines. It uses Elo, player availability, recent form and schedule
features, with a Streamlit app for upcoming games, what-if injury scenarios, team ratings and a public
track record.

## Results

All numbers are out of sample: each game was predicted by a model trained only on earlier seasons.

| | Accuracy | Log-loss |
|---|---|---|
| Always pick the home team | 55.5% | 0.687 |
| Elo only | 68.2% | 0.604 |
| **This model, 2025-26 holdout (evaluated once)** | **68.6%** | **0.590** |
| This model, walk-forward 2021-22 → 2024-25 (5,281 games) | 66.6% | 0.611 |
| v1 (single season, random forest) | ~62%* | – |

\*v1 was scored per team-row. In 42% of games it predicted both teams to win, or both to lose.

The feature ablation (walk-forward accuracy) shows where the gains come from:
- Elo alone: 64.8%
- adding player availability: **66.5%**
- adding form and schedule: 66.6%

## How it works

- **Data:** every game since 2015-16 from the official NBA stats API (`nba_api`): team box scores,
  advanced stats and player box scores. Responses are cached, and a full backfill takes about 130 requests.
- **One row per game:** features are home-minus-away differences, so predictions can't contradict each other.
- **Player availability:** each player has a rolling Game Score rating. Team strength is the sum of the top
  8 available players, compared with its full-strength roster.
- **Elo:** margin-of-victory Elo, with between-season regression and tuned K and home advantage.
- **Model:** ridge regression on the margin and on the total. `P(home win) = Φ(margin/σ)`, so the scoreline
  and the winner always agree. Every prediction splits exactly into per-factor contributions (the app's "Why?" panel).
- **Leakage guards:** two tests protect against leakage.
  - `test_no_lookahead_leakage` rebuilds all features on truncated data and asserts nothing changes.
  - `test_serving_features_match_training_matrix` proves the app computes features exactly as training did.

## Usage

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"

python scripts/build_data.py          # download + build tables (cached)
python scripts/evaluate.py            # walk-forward comparison of baselines / feature sets / models
python scripts/train.py               # fit production model -> models/model.joblib
python scripts/daily_update.py        # refresh season, predict next 7 days, export data/app/
streamlit run app/streamlit_app.py
pytest
```

`scripts/train.py --holdout` evaluates the held-out season. It is meant to be run once.

## Layout

```
src/nbapred/
  data/       fetch.py (cached nba_api calls), build.py (games / team_games / player_games tables)
  features/   elo.py, players.py, form.py, schedule.py, build.py (the per-game matrix)
  models/     predictors.py (models + explanations), evaluate.py (walk-forward harness)
  predict.py  TeamState: serving-time features for any matchup
  live.py     schedule, rosters, prediction log, app exports
app/          Streamlit (streamlit_app.py + views/)
data/app/     the small files the app reads (committed); data/raw + data/processed are rebuildable caches
legacy/       v1 notebooks and scraped CSVs
```

## Deploying

stats.nba.com often blocks cloud IPs, so the app never calls it. Run `scripts/daily_update.py` locally,
commit `data/app/` and push. Streamlit Community Cloud serves the app from `app/streamlit_app.py` using
`requirements.txt`.

## Limitations

- The model doesn't read injury reports. Live predictions assume full rosters; use the Matchup explorer
  to mark players out. The backtest used actual lineups, so live accuracy will be somewhat lower.
- Early in a season, Elo and form still describe last season's team. Roster strength updates straight away.
- It hasn't been compared against betting lines yet (see ROADMAP.md).
