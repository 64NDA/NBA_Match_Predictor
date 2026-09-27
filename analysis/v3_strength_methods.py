"""How should a player's strength be measured, and how should players add up to a team?

Walk-forward 2021-22 -> 2024-25 (holdout untouched), final v3 config, paired vs the current method.
  Player strength: (a) RAPM with box-score prior [current]  (b) box-score estimate only
                   (c) RAPM with no prior (shrunk toward average instead)
  Team aggregation: (1) expected minutes [current]  (2) top 8 by minutes, equal weight (v2-style)
                    (3) star-weighted: minutes^1.5, rescaled to 240 (starters count more per minute)

    python analysis/v3_strength_methods.py > analysis/v3_strength_methods_output.txt
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

import nbapred.features.projection as PJ
from nbapred.config import MODELS, PROCESSED
from nbapred.data import fetch
from nbapred.data.build import load
from nbapred.features import availability as AV, impact as IM, matrix_v3 as MX
from nbapred.features.stints import load_stints
from nbapred.features.style import heights
from nbapred.models import v3 as V3
from nbapred.models.evaluate import paired, walk_forward

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
pd.set_option("display.width", 200)
cfg = json.loads((MODELS / "v3" / "config.json").read_text())
games, tg, pg = load("games"), load("team_games"), load("player_games")
shots = pd.read_parquet(PROCESSED / "team_shots.parquet")
prior = joblib.load(PROCESSED / "box_prior.joblib")
rookie_min = joblib.load(PROCESSED / "rookie_minutes.joblib")
curve = joblib.load(PROCESSED / "playoff_curve.joblib")
hts = heights(fetch.player_bio())
cands = AV.load_candidates()
if "p_healthy" not in cands:  # candidates built before the rename: same prediction, old column name
    cands["p_healthy"] = cands["p_full"]
ratings =pd.read_parquet(PROCESSED / "player_ratings.parquet")


def evaluate(r):
    m = MX.build_matrix(games, tg, cands, r, IM.ratings_for_games(r, games).to_dict(), prior.rookie, curve, hts, shots,
                        rookie_min=rookie_min, save_projection=False)
    return walk_forward(m, V3.fit_predict(cfg["groups"]))


class ZeroPrior(IM.BoxPrior):
    """No box-score information: every player starts at average (RAPM shrinks toward 0)."""
    def predict(self, rates):
        return pd.DataFrame({"O_prior": 0.0, "D_prior": 0.0}, index=rates.index)


res = {"(a)(1) current: RAPM + prior, minutes-weighted": evaluate(ratings)}
BASE = next(iter(res))

res["(b)(1) box-score estimate only"] = evaluate(ratings.assign(O=ratings["O_prior"], D=ratings["D_prior"]))
stints = load_stints()
zero = ZeroPrior(prior.coef_o * 0, prior.coef_d * 0, prior.repl, (0.0, 0.0), prior.mean, prior.scale)
res["(c)(1) RAPM, no box-score prior"] = evaluate(
    IM.weekly_ratings(stints, IM.box_table(pg, tg), zero, IM.ImpactConfig(), IM.league_ortg(stints),
                      sorted(pd.to_datetime(ratings["fit_date"].unique()))))
print("== Player strength ==")
print(paired(res, BASE).round(4).to_string(), flush=True)

orig = PJ.expected_minutes


def top8_equal(p, usual, mult=None):
    e = orig(p, usual, mult)
    out = np.zeros_like(e)
    top = np.argsort(-e)[:8]
    out[top] = np.where(e[top] > 0, 30.0, 0.0)   # 240 / 8
    return out


def star_weighted(p, usual, mult=None):
    e = orig(p, usual, mult)
    w = e ** 1.5
    return w * (e.sum() / w.sum()) if w.sum() > 0 else e


agg = {BASE: res[BASE]}
for name, fn in (("(a)(2) top 8, equal weight (v2-style)", top8_equal), ("(a)(3) star-weighted (minutes^1.5)", star_weighted)):
    PJ.expected_minutes = fn
    MX.expected_minutes = fn
    try:
        agg[name] = evaluate(ratings)
    finally:
        PJ.expected_minutes = MX.expected_minutes = orig
print("\n== Team aggregation ==")
print(paired(agg, BASE).round(4).to_string(), flush=True)
