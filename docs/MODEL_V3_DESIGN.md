# Model v3 — design

Status: **proposed, awaiting sign-off** (2026-09-26). v2 is committed on branch `v2` (c669421).

## 1. Philosophy: an aggregate base, plus refinements

A prediction is built in layers, and each layer is measured in **points of home margin**, so it can be
explained:

```
                ┌──────────────────────────────────────────────────────────────┐
  BASE          │ Elo (team results)        + Player projection (who plays)     │  "how good are they"
                ├──────────────────────────────────────────────────────────────┤
  REFINEMENTS   │ Matchup / style           + Court (home, altitude, neutral)   │  "how do they fit together
                │ Context (rest, travel, playoffs)                             │   tonight"
                └──────────────────────────────────────────────────────────────┘
                       ridge combiner  →  margin, total  →  P(win), scoreline
```

**Selection rule.** A refinement group ships only if it improves walk-forward log-loss by more than 2
paired standard errors (the same rule as in `analysis/EXPERIMENTS.md`). The 2025-26 holdout is evaluated
once, at the end.

---

## 2. Data

| Table | Source (nba_api / other) | Grain | Seasons | Status |
|---|---|---|---|---|
| games, team_games | LeagueGameFinder, TeamGameLogs (Advanced, Four Factors) | game / team-game | 2015-16 → | have |
| player_games | PlayerGameLogs | player-game box score | 2015-16 → | have |
| **rotations** | GameRotation | player on/off intervals per game | 2015-16 → | new, 1 call/game |
| **pbp** | PlayByPlayV3 | events with score, clock, shot distance | 2015-16 → | new, 1 call/game |
| **stints** | derived from rotations + pbp | segments with the same 10 players on court: duration, points each side, possessions | 2015-16 → | derived |
| **player_bio** | PlayerIndex (1 call/season) | height, weight, position | 2015-16 → | new |
| **injury_reports** | official NBA PDFs `ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<hh>PM.pdf` | player status snapshot | 2018-19 → (verified back to Jan 2019) | new, parsed with pdfplumber |
| **injuries_live** | ESPN `site.api.espn.com/.../nba/injuries` | current statuses | live | fallback only |
| rosters | CommonTeamRoster | current roster | live | have |

**Backfill cost:** about 14.5k games × 2 calls plus about 1.3k PDFs, **roughly 6–7 hours** one-off at 0.8 s per
request. It is resumable (every response is cached) and backs off on errors. The daily increment is about 15
games × 2 calls.

---

## 3. Base layer A: Elo (kept, one change)

**Unchanged from v2:**
- margin-of-victory Elo using FiveThirtyEight's multiplier `(|MOV|+3)^0.8 / (7.5 + 0.006·winner_elo_diff)`
- K = 15, home advantage = 50 Elo points (≈1.8 pts), with **none at neutral sites**
- ratings are read before each game and updated after it

**Change: a smarter start to each season.** v2 regresses 25% toward the league mean (1505). v3 regresses
toward **the team's roster-based projection on opening night** instead:

`elo_start = 0.75 · elo_end + 0.25 · (1505 + c · projected_net_rating)`

This fixes "Elo describes last season's team" after big offseason moves. `c` is fitted on the walk-forward
seasons, and v2's version (c = 0) stays as the baseline.

Feature: `diff_elo` (points-equivalent = elo_diff / 28).

---

## 4. Base layer B: player projection (replaces v2's Game Score "top-8 sum")

### 4.1 Player impact: hybrid RAPM with a box-score prior, split into offence and defence

**Stints.** Rotations give each player's on/off times. Splitting the game at every substitution gives
segments where the same 10 players are on court. The pbp score timeline gives the points each side scored
in each segment. Possessions per segment are estimated as `duration × game possessions / 48 min`.

**Regression.** Each stint gives two rows, one per team on offence:

```
y = 100 · points / possessions − league ORtg
  = Σ O_i (5 offensive players) − Σ D_j (5 defensive players) + home_offence·h + ε
```

- The regression is weighted by `possessions × time-decay`, with a 1-season half-life (0.5 and 2 are also tested).
- The ridge penalty shrinks each player **toward his box-score prior**, not toward 0. It fits the deviation
  from the prior, and λ is tuned by walk-forward.
- **O** (offensive impact) and **D** (defensive impact) are in points per 100 possessions relative to
  average; positive is good for both.
- It is solved as a sparse ridge (about 300k rows × about 2k players, a few seconds per fit).

**Box-score prior ("BPM-lite"), two linear models (O and D):**
- **Inputs:** a player's rolling per-100 rates: points, true-shooting relative to league, 3PA, FTA, AST,
  OREB, DREB, STL, BLK, TOV, PF, usage, plus minutes share.
- **Target:** long-window RAPM with no prior, fitted on 2015-16 → 2019-20 **only**. The weights are then
  frozen, so they can't leak into the 2021-22+ test seasons.
- **Rookies:** a player with no history gets a replacement-level prior of about −2.0 per 100 (tuned).

**Leak-free cadence.** RAPM is refit **every 7 days**, using stints strictly before that date. A game on date
d uses the latest fit before d. At serving time it's refit daily.

### 4.2 Minutes model
- **Base expected minutes when playing:** EWM (half-life 8 games) of minutes in games played for the current
  team, with the starter flag taken from rotations.
- **Redistribution.** Absent players' minutes (`Σ (1 − p_i) · m_i`) go to available teammates in proportion
  to their usual minutes, capped at 40 per player. Team total is 240.
- **Playoff rotation tightening** (the playoff effect you mentioned). A learned multiplier by rotation rank,
  estimated from 2015–2020 playoffs versus the same players' regular season. The top players' minutes go up
  and the deep bench is cut, so **stars count for more in the playoffs automatically**.
- Validated by minutes MAE on games with absences.

### 4.3 Availability: P(player plays)
Three sources, in order of priority:
1. **Manual override** (Matchup explorer).
2. **Injury report status → probability**, calibrated on 2018-19 → 2020-21 reports joined to box scores
   (roughly Out 0.00, Doubtful ~0.05, Questionable ~0.5, Probable ~0.95; learned, not hard-coded). The
   report used is **the latest one published at least 60 minutes before tip-off**, so the backtest is leak-free.
3. **History model when a player isn't listed:** logistic regression on
   - played the last team game
   - consecutive team games missed (0 / 1 / 2–3 / 4–9 / 10+)
   - share of the last 10 team games played
   - days since last appearance

   This also handles long-term injuries when no report exists, which is the "blind" case.

### 4.4 Team projection for a game

Here `e_i` is expected minutes after availability and redistribution:

```
ORtg_T = lgORtg + Σ O_i · e_i / 48          DRtg_T = lgORtg − Σ D_i · e_i / 48
possessions = pace_home · pace_away / lg_pace     (pace = EWM of team pace, half-life 20 games)
proj_pts_home = possessions · (ORtg_home + DRtg_away − lgORtg) / 100      (same for away)
```

**Features:**
- `proj_margin` = proj_pts_home − proj_pts_away
- `proj_total`
- `injury_cost_h` and `injury_cost_a`: the full-strength net rating minus the expected net rating

Star power comes out of `O_i · e_i` without a separate feature, and the value of each absence is in points.

---

## 5. Refinement layer C: matchup and style

Each item below is a *candidate* and ships only if it passes the selection rule. Expect small effects in
the regular season; playoffs are where style matters most.

1. **Four-factor matchup.**
   - For each factor f (eFG%, TOV%, OREB%, FT rate): `exp_f_home = off_f_home + def_f_allowed_away − league_f`, and the same for the away team.
   - `four_factor_margin` = Σ w_f · (exp_f_home − exp_f_away), with the weights w_f fitted by regression.
   - This captures battles the net ratings average out, such as an elite offensive-rebounding team facing a poor defensive-rebounding one.
2. **Shot-profile interactions.**
   - From pbp shot distance and value, build each team's offensive profile: rim (<4 ft), mid-range and 3PA rates. Also build its defensive allowed profile and rim FG% allowed.
   - Features: `(rim_rate_off_home − lg) × (rim_fg_allowed_away − lg)`, the same for 3PA, and both directions.
   - This is the "their style counters ours" effect: a rim-attacking team against elite rim protection, or a three-heavy team against a defence that runs teams off the line.
3. **Size matchup.** Minutes-weighted height of each expected lineup (player_bio × e_i). Features: height
   difference, plus its interaction with the rebounding factors.
4. **Pace clash (for totals).** `|pace_home − pace_away|`, and whose tempo prevails (the historical share).
5. **Playoff series adjustment.** Within a series, the shrunk mean residual (actual minus predicted margin)
   of earlier games. This captures coaching adjustments and matchup problems revealed in the series.
6. **Head-to-head residual (regular season).** A heavily shrunk mean residual of this pairing over the last
   2 seasons. It's expected to be noise and is included only to test that.

---

## 6. Refinement layer D: court and context
- **League home court:** the combiner's intercept, multiplied by `(1 − neutral)` so neutral sites get none.
- **Team-specific home edge:** an EWM of the team's home residual minus its away residual, heavily shrunk,
  plus an altitude flag (DEN, UTA). It was borderline in v2 testing and gets re-tested with the better base.
- **Playoff × home court:** the home edge is larger in the playoffs (v2 found +1.6 pts).
- **Rest and travel (as v2):** rest days, back-to-backs, 3-in-4 nights, travel km, plus a **time-zone change** flag.

---

## 6b. Tree-based machine learning (random forest / gradient boosting)

v2 tested LightGBM as the *main* model, and it was significantly worse than ridge (+0.0041 ± 0.0012
log-loss). With about 13k games and a mostly linear signal, trees fit noise. v3 therefore uses trees where
their strength, **finding interactions and non-linear patterns automatically**, actually matters. Each use is
a candidate that must pass the selection rule:

| Where | Model | Why trees |
|---|---|---|
| **Matchup residual model** | LightGBM **and** random forest, trained on the *residual* (actual margin − base-model margin) using style profiles, four factors, shot profiles, size, pace and playoff flag | Learns "style X struggles against style Y" without hand-written interaction formulas. Kept shallow (depth 2–3), strongly regularised, and early-stopped on inner seasons. Its output is one feature, `ml_matchup_adj`, in points. |
| **Availability model** | gradient boosting vs logistic regression | P(play) is non-linear: a player who missed exactly 1 game behaves very differently from one who missed 10. Scored by the log-loss of P(play). |
| **Minutes model** | gradient boosting vs the proportional rule | Who absorbs minutes depends on position and role; trees can learn that. Scored by minutes MAE. |
| **Second opinion** | LightGBM on the full feature set, stacked with ridge (weights fitted on out-of-fold predictions) | An independent model. Kept only if the stack beats ridge alone; otherwise its disagreement is shown in the app as a "model uncertainty" hint. |

Random forest is included explicitly as a nod to v1, and compared head-to-head with LightGBM in each role.

## 7. Combiner and outputs
- **Margin model:** ridge over grouped point-scale features (about 15–25 after selection).
- **Total model:** ridge on `proj_total`, pace-clash features and league scoring level.
- **Win probability:** `Φ(margin / σ)`, with σ chosen by **log-loss over the pooled out-of-sample
  predictions of the last 3 inner seasons**. This fixes v2's underconfidence (slope 1.11).
- **Scoreline:** `(total ± margin) / 2`.
- **Explanation:** exact per-group contributions, relative to two evenly matched teams at a neutral venue in
  the regular season:
  - **Home court**
  - **Team strength (Elo)**
  - **Player ratings** (at full strength)
  - **Availability / injuries**
  - **Matchup & style**
  - **Rest & travel**
  - **Playoff**

## 8. How a prediction happens (serving)
1. `daily_update.py` refreshes today's games, rotations, pbp and box scores, refits RAPM, and updates Elo,
   minutes and form.
2. It fetches the latest official injury report (falling back to ESPN), then the official rosters.
3. For each upcoming game:
   1. p_i for every rostered player
   2. expected minutes (with playoff mode if relevant)
   3. team O and D ratings and pace
   4. projected points
   5. the matchup and context features
   6. ridge margin and total
   7. P(win), scoreline and explanation
4. Predictions are logged **before tip-off**, then scored when results arrive.
5. The app's explorer re-runs step 3 with your status overrides per player (Available / Probable /
   Questionable / Doubtful / Out).

## 9. Evaluation (reported for every model version)
- **Walk-forward, 2021-22 → 2024-25:**
  - train on all prior seasons (from 2016-17, since 2015-16 is burn-in)
  - tune only here
  - report accuracy, log-loss, Brier score, margin MAE, total RMSE and calibration slope
- **Three availability conditions:**
  - **(a) known lineups**, the ceiling
  - **(b) live-realistic:** injury report at T−60 min plus the history model. **This is the headline number.**
  - **(c) blind:** the history model only
- **Holdout 2025-26:** evaluated once, at the end, for v3 against v2 and Elo.
- **Group ablation table:** base → +players → +availability → +matchup → +court/context.
- **Tests:** the existing leakage and parity tests, extended as follows:
  - RAPM fit dates are strictly before each game
  - injury reports are published at least 60 min before tip-off
  - serving features equal training features for v3

## 10. Codebase: keeping v1 and v2 alongside v3

| Kept as | What |
|---|---|
| `legacy/` | the v1 notebooks, scraper and CSVs (unchanged) |
| `src/nbapred/models/v2/` + `features/gamescore.py` (renamed from `players.py`) | the v2 model: ridge on the Elo + Game Score + form + schedule matrix |
| `src/nbapred/models/v3/` | the new model |
| `scripts/train.py --version v2\|v3`, `scripts/evaluate.py --version …` | shared scripts |
| `models/v2/`, `models/v3/` | saved models and metrics |

The app serves v3. The Track record page compares v3, v2 and Elo side by side.

## 11. Build order
0. Start the rotations, pbp and injury-PDF backfill in the background (about 6–7 h).
1. Legacy renames and the v2/v3 model registry; the v2 tests keep passing.
2. Stints, the box prior and weekly RAPM.
3. The minutes model and the availability model.
4. The injury report parser, historical calibration and the ESPN live fallback.
5. The team projection and the Elo season-start change.
6. The matchup/style and court/context features, including the tree-based matchup residual model (§6b).
7. The combiner, the selection experiments and the 3-condition evaluation; then the holdout, once.
8. Serving (`TeamState` v3), parity tests and the daily update.
9. App: injury badges, a player-value table (O/D, minutes, points added), status controls in the explorer,
   a style-matchup panel, the v3 vs v2 vs Elo track record, and corrected docs.

## 12. Expected outcome (honest)
- **Live-realistic accuracy:** about 66–67%, up from the current real ~65%. Most of the gain comes from
  availability and better player values.
- **Known-lineup ceiling:** about 67–68%.
- **Matchup and style:** likely 0–0.3pp in the regular season, possibly more in the playoffs. They're kept
  only if they pass the selection rule.
- **For reference:** bookmakers' closing lines are about 68–70%.
