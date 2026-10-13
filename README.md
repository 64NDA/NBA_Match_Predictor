# NBA Match Predictor

Predicts NBA winners and scorelines from **who is actually playing tonight**.

- **Player ratings** measure each player's offence and defence from every 10-man stretch of play, over 11 seasons.
- **Availability** comes from the official NBA injury report, turned into a chance of playing by a LightGBM model.
- **Team strength** is those ratings weighted by expected minutes, combined with Elo, home court, and rest and
  travel.

A Streamlit app shows upcoming games with injury-aware probabilities, a what-if explorer where you set each
player's status, player and team ratings, and a public track record.

- **Plain-English explanation:** [docs/HOW_IT_WORKS_SIMPLE.md](docs/HOW_IT_WORKS_SIMPLE.md)
- **Technical design:** [docs/MODEL_V3_DESIGN.md](docs/MODEL_V3_DESIGN.md)
- **Full results:** [analysis/V3_RESULTS.md](analysis/V3_RESULTS.md)

## Results

**Live** means predicting with the injury report as it stood an hour before tip-off: what the app does. Log-loss
scores the probabilities (lower is better).

| | Walk-forward 2021-22 → 2024-25 | 2025-26 holdout* |
|---|---|---|
| **v3, live** | **67.1% · 0.606** | **69.0% · 0.591** |
| v3, no injury report | 66.3% · 0.614 | 68.1% · 0.600 |
| v2, live (no injury information) | 65.1% · 0.627 | 68.4% · 0.606 |
| Elo only | 64.8% · 0.627 | 68.2% · 0.604 |
| Always pick the home team | 55.5% · 0.687 | 55.5% · 0.687 |
| v1 (one season, random forest) | ~62%, per team-row, and self-contradicting | – |

\*A second look, after post-holdout bug fixes. The first look (67.3% · 0.594) and the reasons are in
[analysis/V3_RESULTS.md](analysis/V3_RESULTS.md).

**v3 against v2, both live:**
- log-loss is significantly better on both the walk-forward (8 SE) and the holdout (2.6 SE)
- accuracy is +2.0pp over four seasons, and +0.6pp (within noise) on the unusually predictable 2025-26 season

**Where the gain comes from:** almost all of it is **player ratings × minutes** (−0.019 log-loss).

**Player and team strength, tested:**
- On-court impact (from play-by-play) starting from a box-score estimate beats both box-score-only and impact
  with no starting point.
- Adding players up by expected minutes beats equal weights (the v2 way) and star-weighting.
- Rating and minutes memories (half-life 1 season, and 8 games) sit on a flat optimum.
- A light head-to-head "matchup history" term didn't help overall, so it's shown in the app as context instead.

**Time-correct rosters:** every prediction, historical or live, uses the roster as it stood that day. The NBA's
dated trades, signings and waivers are applied in the backtest, and live predictions use the official rosters.

**What didn't help, reported not hidden:**
- style matchups: slightly worse overall, a non-significant hint in the playoffs
- recent-form stats
- tree-based matchup models
- a LightGBM or neural-net second opinion

Details are in [analysis/V3_RESULTS.md](analysis/V3_RESULTS.md).

## How a prediction is made
1. **Player impact:** a ridge regression (RAPM) over about 423k on-court stints.
   - Lineups are rebuilt from play-by-play substitutions.
   - Low-minute players are shrunk toward a box-score prior.
   - It's refit weekly, using only earlier games, and gives offence and defence in points per 100 possessions.
2. **Availability:** a LightGBM model gives P(plays) from the official injury report (at T−60 min) and the
   player's recent absence history.
3. **Minutes:** usual minutes, with absent players' minutes redistributed to 240. Stars' minutes rise in the
   playoffs, learned from past playoffs.
4. **Team projection:** Σ rating × expected minutes gives the projected margin, the total, and the injury cost.
5. **Combiner:** a ridge regression over the player projection, Elo, home court, rest and travel, and the playoff
   flag. It predicts the margin and the total. P(win) = Φ(margin/σ), with σ ≈ 11.4 fitted out-of-sample.
   Scores come from the margin and the total, so they always agree with the pick.

## Testing
- **Walk-forward evaluation** with paired standard errors; a model change is kept only beyond 2 SE.
- **A one-time holdout.**
- **34 automated tests**, including:
  - **no look-ahead:** rebuilding everything from truncated data reproduces every earlier feature, including the
    weekly player ratings
  - **train/serve parity:** the app's features equal the training features exactly
  - checks that rebuilt lineups match the NBA's official rotations
  - injury-report parsing, and stint sanity across old and new data feeds

## Run the app locally

The app only reads the small files committed in `data/app/`, so it runs straight from a fresh clone with no data
downloads.

**Windows (PowerShell):**
```powershell
git clone https://github.com/64NDA/NBA_Match_Predictor.git
cd NBA_Match_Predictor
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\streamlit run app/streamlit_app.py
```

**macOS / Linux:**
```bash
git clone https://github.com/64NDA/NBA_Match_Predictor.git
cd NBA_Match_Predictor
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. Use Python 3.11 or newer. Stop the app with Ctrl+C.

- **Already set up?** Just run `.venv\Scripts\streamlit run app/streamlit_app.py` (Windows) or
  `.venv/bin/streamlit run app/streamlit_app.py` (macOS/Linux) from the project folder.
- **Different port:** add `--server.port 8502`.
- **Fresh predictions:** `python scripts/daily_update.py` refreshes `data/app/`. The first time, this needs the
  full data pipeline below; after that, each run only fetches new games (a few minutes).

## Usage

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"

python scripts/build_data.py              # game and box-score tables (cached nba_api calls)
python scripts/backfill_v3.py stats       # one-off: play-by-play for every game (~3 h, resumable)
python scripts/backfill_v3.py injuries    # one-off: official injury reports since Dec 2018
python scripts/build_v3.py                # stints -> player ratings -> availability -> matrix (~15 min)
python scripts/evaluate_v3.py             # the experiment suite
python scripts/train.py --version v3      # production model -> models/v3/
python scripts/daily_update.py            # refresh, predict the next 7 days, export data/app/
streamlit run app/streamlit_app.py
pytest
```

`python scripts/train.py --version v3 --holdout` evaluates the held-out season (meant to be run once).
The v2 model is kept runnable: `scripts/train.py --version v2`, `scripts/evaluate.py`.

## Layout
```
src/nbapred/
  data/        fetch.py (cached nba_api), build.py (tables), injuries.py (official report + ESPN fallback)
  features/    stints.py, impact.py (RAPM + prior), availability.py, projection.py, style.py, elo.py,
               matrix_v3.py (shared assemble), matrix_v2.py + gamescore.py (v2), form.py, schedule.py
  models/      v3.py, availability.py (P(plays)), v2.py, baselines.py, evaluate.py
  serving_v3.py, live.py
app/           Streamlit (streamlit_app.py + views/)
analysis/      experiments and results write-ups
docs/          design docs
legacy/        v1 notebooks and CSVs
```

## Deploying
stats.nba.com blocks most cloud IPs, so the app never calls it.
1. Run `scripts/daily_update.py` locally.
2. Commit `data/app/`.
3. Push.

Streamlit Community Cloud serves `app/streamlit_app.py`, installing the dependencies from `requirements.txt`.

## Limitations
- Late scratches after the last pre-game injury report aren't known.
- Contract expiries aren't in the roster-moves feed, so unsigned free agents aren't known until someone signs
  them. A learned "new season" availability feature covers this.
- No bookmaker line data yet: closing lines (about 68–70%) are the real bar.
