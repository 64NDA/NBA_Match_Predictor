# v3 results (2026-09-27)

Raw outputs: `v3_experiments_output.txt`, `v3_selection_output.txt`, `v3_holdout_output.txt` (first look),
`v3_holdout_output_second_look.txt`. Reproduce with
`scripts/evaluate_v3.py`, `analysis/v3_selection.py` and `scripts/train.py --version v3 --holdout`.

**Method.**
- **Walk-forward:** 2021-22 → 2024-25 (5,281 games). Each season is predicted by a model trained on the seasons
  before it.
- **Paired comparisons:** d = difference in log-loss on the same games. **BETTER / worse** means beyond 2 standard
  errors.
- **Holdout:** 2025-26 was evaluated once, after the configuration was frozen.
- **"Live"** means using the official injury report as it stood at least 60 minutes before tip-off.

## Headline

All numbers below use **time-correct rosters**: roster moves are applied as of each game date (see
"Bugs found").

| | Walk-forward 2021-25 | 2025-26 holdout, 1st look | 2025-26 holdout, 2nd look |
|---|---|---|---|
| **v3, live (injury report)** | **67.1% · 0.6060** | 67.3% · 0.5936 | **68.8% · 0.5908** |
| v3, no injury report | 66.4% · 0.6140 | 66.6% · 0.6043 | 68.1% · 0.6003 |
| v3, knowing who played (ceiling) | 67.3% · 0.6042 | – | 68.3% · 0.5871 |
| v2, live (no availability information) | 65.1% · 0.6272 | 68.4% · 0.6058 | 68.4% · 0.6058 |
| v2, knowing who played | 66.6% · 0.6106 | 68.6% · 0.5896 | 68.6% · 0.5896 |
| Elo only | 64.8% · 0.6268 | 68.2% · 0.6038 | 68.2% · 0.6038 |
| Home team | 55.5% · 0.6874 | 55.5% · 0.6873 | 55.5% · 0.6873 |

**About the two holdout looks:**
- The **1st look** was run once with the frozen configuration.
- After it, checking the production output found four bugs:
  - minutes squeezed stars when rosters were large
  - the offseason was read as an injury
  - rosters ignored trades and signings
  - end-of-season absences carried into the next season
- Fixing them changed features, not the configuration, and the holdout was re-evaluated as a **2nd look**.
- The 2nd look is no longer a clean first look. The walk-forward remains the main evidence.

**v3 live vs v2 live** (the fair comparison, since v2 has no injury information):

| | log-loss difference | accuracy difference |
|---|---|---|
| Walk-forward | **−0.021 ± 0.003 (8 SE)** | **+2.0pp** |
| Holdout, 1st look | −0.012 ± 0.006 (2.1 SE) | −1.1pp (within noise) |
| Holdout, 2nd look | −0.015 ± 0.006 (2.6 SE) | +0.5pp (within noise) |

On the 2nd look, **v3 live also beats Elo by 2.2 SE**. 2025-26 was an unusually predictable season: even plain
Elo got 68%.

**Knowing who played:** v3 beats v2 over the walk-forward (−0.006 ± 0.002, 3 SE).

The honest summary:
- v3's **probabilities** are consistently and significantly better than v2's and Elo's in live use.
- Its **pick accuracy** is clearly better over four seasons, and slightly better (within noise) on the fifth.
- σ fell from 14.3 to 11.4, which fixes v2's underconfidence.

## What each part is worth (walk-forward, live)

| Step (cumulative, from home court + Elo) | d log-loss | Verdict |
|---|---|---|
| + **Player ratings** (RAPM with box-score prior, × expected minutes) | **−0.019** | BETTER |
| + Injuries (separate injury-cost term) | −0.0003 | ~same (already inside the player projection) |
| + Recent form | −0.0001 | ~same |
| + **Rest & travel** | **−0.002** | BETTER |
| + Playoff flag | 0 | ~same |
| + Matchup & style | +0.001 | slightly worse |
| + Series / head-to-head | 0 | ~same |

Leave-one-out from all groups gives the same picture:
- **Needed:** player ratings (+0.006 without them), home court (+0.002), rest & travel (+0.002).
- **Better removed:** matchup & style (−0.001, BETTER when removed).

## Matchups and the playoffs

On the **361 playoff games**, adding matchup & style gave −0.0022 ± 0.0015 (1.5 SE). That's a hint, not
significant. Series history gave nothing. I didn't hunt for more playoff-specific variants on so few games, since
that is how overfitting happens. The code stays in to re-test as playoff seasons accumulate.

## Machine learning

| Where | Result |
|---|---|
| **Availability (will he play?)** | **LightGBM beat logistic regression: log-loss 0.248 vs 0.271 with the injury report, 0.382 vs 0.400 without.** Used. |
| Matchup residual, LightGBM | worse (+0.0033 ± 0.0016) |
| Matchup residual, random forest | ~same (+0.0008 ± 0.0011) |
| Ridge + LightGBM stack | ~same (+0.0003 ± 0.0002) |
| Ridge + neural network (MLP 32-16) stack | ~same (0.0000 ± 0.0001) |
| Ridge alpha 100 vs 10 | ~same |
| Elo season start toward roster projection (c = 0.5, 1) | ~same |

Trees and neural nets help where the problem is **non-linear with plenty of data**: 475k player-availability
rows, where missing 1 game and missing 10 behave very differently. They don't help at game level, with about 13k
games and a mostly additive signal.

## Final configuration (`models/v3/config.json`)
- **Groups:** home court, Elo, player ratings, injuries, rest & travel, playoff flag.
- **Model:** ridge (no intercept), trained on the live-availability view.
- This ties the best alternatives within noise and keeps the Elo and injury terms visible in the app's
  "Why?" chart.

## Bugs found by the testing on the way
- **Two look-ahead leaks:** whole-season league averages, and same-day playoff/series updates. Caught by the
  truncation test and fixed.
- **Missed free throws in the 2015-17 feeds** carry a "0-0" score, which corrupted stint points. That made those
  seasons' player ratings nonsense (Curry at offence +59, defence −38). Fixed with a running maximum, and a
  regression test now guards it.
- **A collinear box-score prior, and an unpinned offence/defence level.** Fixed.
- **"Full strength" counted rarely-used players as playing**, which diluted stars' minutes. Redefined as "nobody
  injured, normal rotation role".
- **Too many bodies squeezed the stars:** minutes were scaled down proportionally. They're now trimmed from the
  end of the bench.
- **The offseason looked like an injury:** "days since last game" is now counted only while the team was playing.
- **Rosters ignored time** (spotted by Albert). Departed players stayed in their old team's pool, and new
  signings only appeared once they'd played. Fixed with the NBA's dated player-movement feed (9,892 signings,
  waivers and trades since July 2015), applied before each game day in training and serving alike. Live, the
  official rosters are authoritative.
- **End-of-season absences carried into the next season:** a team resting players in April made them look
  unlikely to play on opening night. Absence features are now counted within the current season only, and are
  "unknown" before a team's first game.

## Known limitations
- The moves feed has no contract expiries, so unsigned free agents and retirements aren't known until a new team
  signs them. The `new_season` availability feature learns how often such players return.
- Late scratches after the last pre-game report aren't known.
- Bookmakers' closing lines (about 68–70%) are still the bar to beat. There's no line data in the repo yet.
