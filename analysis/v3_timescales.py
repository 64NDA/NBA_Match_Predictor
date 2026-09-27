"""Which time windows should the displayed and modelled statistics use?

Tests, on the 2021-22 -> 2024-25 walk-forward (holdout untouched), with the final v3 config:
  - player-rating memory: RAPM recency half-life (0.5 / 1 / 2 seasons) x ridge strength lambda (1k / 3k / 10k)
  - box-score prior memory: half-life of the per-100 rates (0.5 / 1 / 2 seasons)
  - "usual minutes" memory: EWM half-life in games played (4 / 8 / 16), and a current-season-only variant
Each variant rebuilds only what it affects and is compared with the current settings, paired per game.

    python analysis/v3_timescales.py > analysis/v3_timescales_output.txt
"""
import json
import sys

import joblib
import pandas as pd

import nbapred.features.availability as AV
from nbapred.config import MODELS, PROCESSED
from nbapred.data import fetch, transactions
from nbapred.data.build import load, tipoffs
from nbapred.features import impact as IM, matrix_v3 as MX
from nbapred.features.projection import fit_playoff_curve
from nbapred.features.stints import load_stints
from nbapred.features.style import heights
from nbapred.models import v3 as V3
from nbapred.models.availability import AvailabilityModel, predict_full
from nbapred.models.evaluate import paired, walk_forward

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
pd.set_option("display.width", 200)
cfg = json.loads((MODELS / "v3" / "config.json").read_text())
games, tg, pg = load("games"), load("team_games"), load("player_games")
stints, shots = load_stints(), pd.read_parquet(PROCESSED / "team_shots.parquet")
prior = joblib.load(PROCESSED / "box_prior.joblib")
rookie_min = joblib.load(PROCESSED / "rookie_minutes.joblib")
hts = heights(fetch.player_bio())
lg = IM.league_ortg(stints)
box = IM.box_table(pg, tg)
dates = IM.fit_dates(games.date.min(), games.date.max())
base_cands = AV.load_candidates()
base_ratings = pd.read_parquet(PROCESSED / "player_ratings.parquet")


def evaluate(ratings, cands):
    curve = fit_playoff_curve(cands[cands.season <= 2020])
    m = MX.build_matrix(games, tg, cands, ratings, IM.ratings_for_games(ratings, games).to_dict(), prior.rookie, curve,
                        hts, shots, rookie_min=rookie_min, save_projection=False)
    return walk_forward(m, V3.fit_predict(cfg["groups"]))


def rebuild_candidates(min_half_life=None, season_only=False):
    """Candidate table with a different 'usual minutes' memory; availability model refitted (fit seasons only)."""
    if min_half_life:
        AV._A = 1 - 0.5 ** (1 / min_half_life)
    orig = AV.History.update
    if season_only:
        def update(self, team, box, date, season):  # forget minutes at the start of each player's new season
            for p in box["player_id"]:
                if self.last_season.get(p, season) < season:
                    self.min_ewm.pop(p, None)
            orig(self, team, box, date, season)
        AV.History.update = update
    try:
        c, _ = AV.candidate_table(games, pg, tipoffs(games), AV.name_keys(fetch.player_bio(), pg), transactions.load())
    finally:
        AV.History.update, AV._A = orig, 1 - 0.5 ** (1 / AV.MIN_HALF_LIFE)
    model = AvailabilityModel("lgbm").fit(c)
    c["p_live"], c["p_blind"], c["p_full"] = model.predict(c), model.predict(c, blind=True), predict_full(model, c)
    return c


results = {"current (RAPM half-life 1 season, lambda 3000, box 1 season, minutes 8 games)": evaluate(base_ratings, base_cands)}
BASE = next(iter(results))

print("== Player-rating memory and shrinkage ==", flush=True)
for hl in (182, 365, 730):
    for lam in (1000, 3000, 10000):
        if (hl, lam) == (365, 3000):
            continue
        c = IM.ImpactConfig(lam=lam, half_life_days=hl, window_days=max(3 * 365, 3 * hl))
        results[f"RAPM half-life {hl}d, lambda {lam}"] = evaluate(IM.weekly_ratings(stints, box, prior, c, lg, dates), base_cands)
        print(f"  done {hl} {lam}", flush=True)
print(paired(results, BASE).round(4).to_string(), flush=True)

print("\n== Box-score prior memory ==", flush=True)
box_res = {BASE: results[BASE]}
for bhl in (182, 730):
    c = IM.ImpactConfig(box_half_life_days=bhl)
    box_res[f"box-rate half-life {bhl}d"] = evaluate(IM.weekly_ratings(stints, box, prior, c, lg, dates), base_cands)
print(paired(box_res, BASE).round(4).to_string(), flush=True)

print("\n== Usual-minutes memory ==", flush=True)
min_res = {BASE: results[BASE]}
for mhl in (4, 16):
    min_res[f"minutes half-life {mhl} games"] = evaluate(base_ratings, rebuild_candidates(min_half_life=mhl))
min_res["minutes: current season only (8 games)"] = evaluate(base_ratings, rebuild_candidates(season_only=True))
print(paired(min_res, BASE).round(4).to_string(), flush=True)
