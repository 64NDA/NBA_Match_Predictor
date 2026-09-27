import streamlit as st

from common import meta

st.title("How it works")
m = meta()
wf, ho = m["model_meta"].get("walk_forward", {}), m["holdout_metrics"]

st.markdown(f"""
**Data.** Every NBA game since 2015-16 (regular season, play-in and playoffs) from the official NBA stats API
via `nba_api`: team box scores, advanced stats, and every player's box score.

**One row per game.** Each game is described from the home team's point of view as *home minus away*
differences, so the model can't predict both teams to win (a bug in v1 of this project).

**Features.** In order of importance:
1. **Player availability.** Every player has a rolling rating (recent Game Score per game). A team's strength
   for a game is the sum of its top 8 *available* players. Missing players lower it. In the backtest,
   "available" means "actually played", so this is an upper bound on live accuracy. The app assumes full
   rosters unless you mark players out.
2. **Elo.** A margin-of-victory Elo rating. Ratings regress 25% toward average between seasons.
3. **Recent form.** Exponentially weighted averages (half-lives of 5 and 20 games) of net rating, the four
   factors, pace and points, both for and against.
4. **Rest & travel.** Days of rest, back-to-backs, games in the last 7 days, and distance travelled.

**Model.** A ridge regression predicts the home **margin**, and a second one predicts the **total**. The two
give the scoreline. The win probability is `Φ(margin / σ)`, with σ ≈ {m['model_meta'].get('sigma', 14):.1f}
points estimated out of sample, so the score and the winner always agree. LightGBM was tested and was
no better.

**Validation.** Walk-forward by season: train on every season before it, test on it. Tuning used 2021-22 to
2024-25 only. 2025-26 was held out and evaluated once.

| | Accuracy | Log-loss |
|---|---|---|
| Always pick the home team | {ho.get('home_team', {}).get('acc', float('nan')):.1%} | {ho.get('home_team', {}).get('log_loss', float('nan')):.3f} |
| Elo only | {ho.get('elo', {}).get('acc', float('nan')):.1%} | {ho.get('elo', {}).get('log_loss', float('nan')):.3f} |
| **This model, 2025-26 holdout** | **{ho.get('model', {}).get('acc', float('nan')):.1%}** | **{ho.get('model', {}).get('log_loss', float('nan')):.3f}** |
| This model, walk-forward 2021-25 | {wf.get('acc', float('nan')):.1%} | {wf.get('log_loss', float('nan')):.3f} |
| v1 of this project (one season, random forest) | ~62%* | – |

\\*The v1 figure was per team-row. In 42% of games v1 predicted both teams to win, or both to lose.

**Limitations.**
- The model doesn't read injury reports.
- Early in a season, Elo and form describe last season's roster. Only roster strength adapts straight away.
- Neutral-site games are too rare for the margin model to learn.
- NBA games are noisy: bookmakers' closing lines hit roughly 68-70%.
""")
