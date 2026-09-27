"""Can a *light* matchup-history term help? (e.g. Denver vs Minnesota, where the same cores keep meeting)

For every game, look at earlier meetings of the same two teams and ask how much each beat or missed the full
model's own pre-game prediction (the "surprise", from the home side of today's game). Combine them as a heavily
shrunk weighted average:
    feature = sum(w_i * surprise_i) / (sum(w_i) + k)
    w_i     = recency (half-life in days)  x  roster continuity (share of that game's minutes played by players
              in today's rotation, geometric mean over both teams)
Variants: plain (no continuity), continuity-weighted, x closeness (only matters when teams are evenly matched),
playoffs-only. Predictions for past games come from models trained only on *earlier seasons* (season S predicted by
a model fit on seasons < S), so the feature is leak-free.

Walk-forward 2021-22 -> 2024-25 (holdout untouched); paired vs the model without it; playoff games shown separately.

    python analysis/v3_matchup_history.py > analysis/v3_matchup_history_output.txt
"""
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from nbapred.config import HOLDOUT_SEASON, MODELS
from nbapred.data.build import load
from nbapred.features import availability as AV
from nbapred.features.matrix_v3 import load_matrix
from nbapred.models import v3 as V3
from nbapred.models.evaluate import FIRST_TRAIN, paired, walk_forward

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
pd.set_option("display.width", 200)
cfg = json.loads((MODELS / "v3" / "config.json").read_text())
m = load_matrix().sort_values(["date", "game_id"]).reset_index(drop=True)
pg = load("player_games")
cands = AV.load_candidates()

# 1. out-of-fold predicted margins for every game from 2017-18 on (season S <- model fit on seasons < S)
oof = []
# (the 2015-16 burn-in season is allowed as training here: it only generates past surprises, never scores)
for s in range(FIRST_TRAIN + 1, HOLDOUT_SEASON):   # holdout season never predicted or used here
    tr, te = m[m.season < s], m[m.season == s]
    oof.append(pd.Series(V3.V3Model(cfg["groups"]).fit(tr).predict(te, "live")["pred_margin"].to_numpy(), index=te.index))
m["oof_margin"] = pd.concat(oof)
m["surprise"] = m["margin"] - m["oof_margin"]

# 2. who played in each game, and today's rotation (players the model expects to play 15+ minutes)
box = {k: dict(zip(v.player_id, v["min"])) for k, v in pg.groupby(["game_id", "team_id"])}
rot = cands[(cands.min_ewm >= 15)].groupby(["game_id", "team_id"])["player_id"].apply(set).to_dict()


def continuity(past_gid, team, today_gid):
    mins, current = box.get((past_gid, team), {}), rot.get((today_gid, team), set())
    tot = sum(mins.values())
    return sum(v for p, v in mins.items() if p in current) / tot if tot else 0.0


def build(half_life_days=365, k=3.0, use_continuity=True, max_meetings=12):
    past = defaultdict(list)          # frozenset(pair) -> [(date, game_id, home_id, surprise)]
    out = np.zeros(len(m))
    for date, day in m.groupby("date", sort=True):
        for i, r in day.iterrows():
            num = den = 0.0
            for d0, gid0, home0, surprise0 in past[frozenset((r.home_id, r.away_id))][-max_meetings:]:
                sign = 1.0 if home0 == r.home_id else -1.0
                w = 0.5 ** ((date - d0).days / half_life_days)
                if use_continuity:
                    w *= np.sqrt(continuity(gid0, r.home_id, r.game_id) * continuity(gid0, r.away_id, r.game_id))
                num += w * sign * surprise0
                den += w
            out[i] = num / (den + k)
        for i, r in day.iterrows():
            if not np.isnan(r.surprise):
                past[frozenset((r.home_id, r.away_id))].append((date, r.game_id, r.home_id, r.surprise))
    return out


closeness = 1 - 2 * np.abs(m["elo_prob"] - 0.5)       # 1 = evenly matched, 0 = mismatch
variants = {
    "plain (recency only, k=3)": build(use_continuity=False),
    "continuity-weighted (k=3)": build(),
    "continuity-weighted, stronger shrink (k=8)": build(k=8.0),
    "continuity-weighted, 2-season memory": build(half_life_days=730),
}
cw = variants["continuity-weighted (k=3)"]
variants["continuity-weighted x closeness"] = cw * closeness.to_numpy()
variants["continuity-weighted, playoffs only"] = cw * m["is_playoff"].to_numpy()

base = walk_forward(m, V3.fit_predict(cfg["groups"]))
res, res_po = {"without matchup history": base}, {"without matchup history": base[base.is_playoff == 1]}
for name, feat in variants.items():
    col = "mh_" + str(abs(hash(name)) % 10**6)
    m[col] = feat
    V3.GROUPS["Matchup history"] = [col]
    wf = walk_forward(m, V3.fit_predict(cfg["groups"] + ["Matchup history"]))
    res[name], res_po[name] = wf, wf[wf.is_playoff == 1]
    print(f"  done: {name}  (feature sd {np.std(feat):.2f})", flush=True)

print("\n== All games ==")
print(paired(res, "without matchup history").round(4).to_string())
print("\n== Playoff games only ==")
print(paired(res_po, "without matchup history").round(4).to_string())

# how big is the effect for a famous matchup? (latest DEN-MIN meeting in the walk-forward seasons)
teams = load("games").drop_duplicates("home_id").set_index("home_team")["home_id"]
dm = m[(m.home_id.isin([teams["DEN"], teams["MIN"]])) & (m.away_id.isin([teams["DEN"], teams["MIN"]]))]
dm = dm[dm.season < HOLDOUT_SEASON].tail(6)
print("\nDEN-MIN, last meetings in the test window (continuity-weighted feature, points for the home side):")
print(pd.DataFrame({"date": dm.date.dt.date, "home": dm.home_team, "margin": dm.margin, "model": dm.oof_margin.round(1),
                    "matchup_history": cw[dm.index].round(2)}).to_string(index=False))
