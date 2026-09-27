import streamlit as st

from common import app_json

st.title("How it works")
ho = app_json("holdout_metrics_v3")
meta = app_json("model_meta_v3")
wf = meta.get("walk_forward", {})
f = lambda k, m, fmt: format(ho[k][m], fmt) if k in ho and m in ho[k] else "–"

st.markdown(f"""
A smart fan asks four questions before a game. The model answers each one **in points** and adds them up.

1. **Who's been winning, and by how much?** An *Elo* rating. Every team starts at 1505 and gains or loses points
   with each result; big wins and wins over good teams count more.
2. **Who's playing tonight, and how good are they?** Every player has an offence and a defence rating. They come
   from watching the score during every stretch of play he's on court, over hundreds of thousands of stretches,
   after accounting for who he played with and against. That's how defence gets counted. Each player's chance of
   playing comes from the official NBA injury report and his recent history. His minutes come from his usual
   role, and a missing player's minutes go to his teammates. Team strength = Σ rating × minutes, so stars count
   the most, and an injury costs exactly the gap between the star and his replacement.
3. **Where, and how tired?** Home court (none at neutral venues, a little more in the playoffs), each team's own
   home edge, rest, back-to-backs and travel.
4. **Do the styles clash?** *Tested, and left out.* Four-factor matchups, shot-profile clashes (a rim-attacking
   team against a shot-blocker, a three-heavy team against a defence that takes threes away), size, and
   playoff-series history. Over the regular season these made predictions slightly *worse*. In the playoffs
   they showed a small hint of helping, but 361 playoff games is too few to trust. The code is kept, to re-test
   as more playoffs are played. Recent-form stats were dropped the same way: once player ratings are in, they add
   nothing.

A ridge regression learns from 10 seasons how much to trust each clue. It predicts the **margin** and the **total
points**, which give the scoreline, and the margin is turned into a **win probability**. The score and the winner
therefore never disagree.

**Machine learning inside:**
- LightGBM predicts whether each player will play; it beat logistic regression on 2021-25.
- Ridge regression produces the player ratings and the final combiner.
- Tested and not used: random forest and LightGBM as style-matchup models, and LightGBM and a small neural
  network as a second opinion blended with ridge. None beat ridge by more than noise; the LightGBM matchup model
  was worse.

**Testing.** Each season is predicted by a model that has only seen earlier seasons, using the injury report as it
stood **an hour before tip-off**. Model choices were made on 2021-22 to 2024-25. **2025-26 was held out**:
evaluated once, then re-evaluated after four bug fixes found afterwards (the table below is that second look). Automated tests check that no feature uses information from after tip-off, and that this app
computes exactly the same features as training.

| 2025-26 holdout (second look, after bug fixes: see the results write-up) | Accuracy | Log-loss |
|---|---|---|
| **v3, injury report 1 h before tip-off** | **{f('v3 (live)', 'acc', '.1%')}** | **{f('v3 (live)', 'log_loss', '.3f')}** |
| v3, no injury report | {f('v3 (blind)', 'acc', '.1%')} | {f('v3 (blind)', 'log_loss', '.3f')} |
| v3, if we knew exactly who'd play (ceiling) | {f('v3 (known)', 'acc', '.1%')} | {f('v3 (known)', 'log_loss', '.3f')} |
| v2 (previous version), live: no injury information | {f('v2 (live)', 'acc', '.1%')} | {f('v2 (live)', 'log_loss', '.3f')} |
| v2, if it knew who'd play | {f('v2 (known)', 'acc', '.1%')} | {f('v2 (known)', 'log_loss', '.3f')} |
| Elo only | {f('elo', 'acc', '.1%')} | {f('elo', 'log_loss', '.3f')} |
| Always pick the home team | {f('home_team', 'acc', '.1%')} | {f('home_team', 'log_loss', '.3f')} |

Walk-forward 2021-22 to 2024-25: **{format(wf.get('acc', float('nan')), '.1%')}** accuracy, log-loss
**{format(wf.get('log_loss', float('nan')), '.3f')}**. Log-loss scores the percentages, not just the picks; lower is
better, and confident misses are punished hardest.

**Rosters over time.** Every prediction uses the roster as it stood on that day. The backtest replays the NBA's
dated trades, signings and waivers since 2015. Live predictions use the official rosters.

**Limits.** Basketball is noisy: bookmakers' closing lines get roughly 3 games in 10 wrong. Late scratches after
the last injury report aren't known. The roster-moves feed has no contract expiries, so an unsigned free agent is
only known once someone signs him.
""")
