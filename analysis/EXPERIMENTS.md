# Accuracy experiments (2026-09-26)

Reproduce with `python analysis/experiments.py`; the full output is in `experiments_output.txt`.

**Method.**
- Every variant uses the walk-forward setup over 2021-22 → 2024-25 (5,281 games) and is compared with the
  production model *on the same games*.
- `d_logloss` < 0 is better. **BETTER / worse** means more than 2 paired standard errors.
- For scale, a 0.3pp accuracy difference is about 16 games, which is noise.
- The 2025-26 holdout was **not** used.

## 1. Home-court advantage: weighted correctly, leave it alone

| | Home edge, evenly matched teams |
|---|---|
| Model (intercept + average travel/rest terms) | **+2.0 pts → 55.6%** |
| Actual, 2021-22 → 2025-26 | +1.6 to +2.6 pts → 54–58% |
| Walk-forward average predicted vs. actual home margin | 2.24 vs 2.17 (bias 0.07 pts) |

None of these variants moved the result outside noise:
- dropping the 2020 bubble games
- dropping 2020-21 (few fans)
- re-centring home advantage on the last 2 or 3 seasons
- removing away-travel distance
- recency weighting

A **team-specific home edge** (e.g. Denver's altitude) came closest: −0.0003 ± 0.0002, +0.3pp. That's not
significant.

Why the calibration chart *looked* like home over-weighting: the calibration slope is **1.11**, meaning the
model is slightly **underconfident**. Its probabilities are pulled towards 50% on both sides, so the
away-favoured bins show the home team winning less than predicted. The intercept is −0.04 (≈ 1pp), which is
negligible.

## 2. Margin → probability
σ comes from a single season's residual standard deviation, 14.3. Blowouts inflate that, so the
probabilities come out too flat.
- Even the best σ chosen *with hindsight* (12.5) only improves log-loss by about 0.001.
- Estimating σ by log-loss, pooled over 3 inner seasons, gives −0.0002 ± 0.0004. That is principled but tiny.
- A Platt/logistic mapping is no better.

## 3. Regularisation, training window, model family
- Ridge alpha from 1 to 300: all ~same.
- Training from 2018 instead of 2016: ~same, slightly worse.
- Direct logistic regression: ~same. Ridge + logistic ensemble: ~same.
- LightGBM for the margin: **worse** (+0.0041 ± 0.0012).

## 4. Player availability
| Variant | d_logloss | d_acc | |
|---|---|---|---|
| **+ star power (top-3 available players)** | −0.0008 ± 0.0004 | +0.11pp | **BETTER** |
| Top 6 instead of top 8 | −0.0007 ± 0.0005 | +0.19pp | ~same |
| Top 10 | +0.0024 ± 0.0005 | −0.61pp | worse |
| Rating half-life 10 games | +0.0010 ± 0.0005 | −0.32pp | worse |
| Half-life 30 / 40 | ~0 | ~0 | ~same |
| **Candidate: star top-3 + pooled σ** | −0.0011 ± 0.0006 | +0.11pp | ~same (1.8 SE) |

## 5. Live conditions: the finding that matters
The backtest knows who actually played each game. Live, without an injury report, the app assumes the whole
roster is available.

| Setting | Accuracy | Log-loss |
|---|---|---|
| Backtest (real lineups) | 66.6% | 0.611 |
| **App today:** trained on real lineups, predicting with "everyone plays" | **65.1%** | **0.627** |
| Trained and predicting with "everyone plays" | 65.4% | 0.624 |
| No player features at all | 65.0% | 0.624 |

- **Honest live accuracy is about 65%.**
- The player-availability gain exists **only** when the lineup is known.
- The current live setup is the worst of the three options.

## Recommendations, in order of value
1. **Ingest the official NBA injury report**, which is published before every game. It recovers about +1.2–1.5pp,
   larger than every tuning result here combined.
2. **Until then, use the right model for the information available.** Default predictions should use a model
   trained with "everyone plays". The Matchup explorer and injury-aware predictions should use the
   lineup-trained model.
3. **Correct the stated accuracy** in the README and the app's "How it works" page (live ≈ 65% without injury data).
4. **Small, safe gains:** add the star-power feature and estimate σ pooled over 3 seasons (about −0.001 log-loss).
5. **Don't bother** with home-advantage re-weighting, LightGBM, alpha tuning, or a shorter training window.

## Code-review notes (prediction side)
- **Fixed:** `players._top(values, n=TOP_N)` bound `TOP_N` when the function was defined, so changing `TOP_N`
  silently did nothing. Production was unaffected because the default is unchanged, but experiments were not.
  It now reads the value at call time.
- **Neutral-site games:** the margin model still applies home court. There are about 10 per season, which is
  too few to learn from; zeroing the intercept for them is a reasonable manual rule.
- **Game Score flatters high-usage players on bad teams:** UTA's roster strength is 105 but its Elo is 1308.
  A better player rating is the next modelling lever after injuries.
