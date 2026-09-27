"""v3 experiments: walk-forward 2021-22 -> 2024-25 (holdout 2025-26 untouched), paired against a baseline.

    python scripts/evaluate_v3.py > analysis/v3_experiments_output.txt

Every row reports d_logloss vs the baseline with a paired standard error; BETTER / worse = beyond 2 SE.
"""
import sys

import pandas as pd

from nbapred.data.build import load
from nbapred.features import matrix_v2 as M2
from nbapred.features.matrix_v3 import load_matrix, with_elo
from nbapred.config import PROCESSED
from nbapred.models import baselines as B, v2 as P2
from nbapred.models import v3 as V3
from nbapred.models.evaluate import paired, walk_forward

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220, "display.max_columns", 20)
m = load_matrix()
games = load("games")
wf = lambda fp, mx=m: walk_forward(mx, fp)
show = lambda title, res, base: print(f"\n== {title} ==\n" + paired(res, base).round(4).to_string())

BASE = ["Home court", "Team strength (Elo)"]
ALL = list(V3.GROUPS)

# 1. Baselines and v2, on the same games
v2 = walk_forward(M2.load_matrix(), P2.margin_total(P2.FEATURE_SETS["all"]))
res = {"home team": wf(B.home_baseline), "v2 (production)": v2, "v3 base: home + Elo": wf(V3.fit_predict(BASE))}
res["v3 all groups"] = wf(V3.fit_predict(ALL))
show("Baselines (train+test on live availability)", res, "v2 (production)")

# 2. Group ablation: add one group at a time to the base
steps, cur = {}, list(BASE)
for g in ["Player ratings", "Injuries", "Recent form", "Rest & travel", "Playoff game", "Matchup & style", "Series / head-to-head"]:
    cur = cur + [g]
    steps[f"+ {g}"] = wf(V3.fit_predict(list(cur)))
show("Ablation (cumulative)", {"base": res["v3 base: home + Elo"], **steps}, "base")

# 3. Leave-one-group-out from all groups
loo = {"all": res["v3 all groups"]}
for g in ALL:
    loo[f"- {g}"] = wf(V3.fit_predict([x for x in ALL if x != g]))
show("Leave one group out", loo, "all")

# 4. Availability conditions
cond = {"train live / test live (headline)": res["v3 all groups"],
        "train known / test known (ceiling)": wf(V3.fit_predict(ALL, "known")),
        "train known / test live": wf(V3.fit_predict(ALL, "known", "live")),
        "train blind / test blind": wf(V3.fit_predict(ALL, "blind")),
        "train live / test blind": wf(V3.fit_predict(ALL, "live", "blind"))}
show("Availability conditions", cond, "train live / test live (headline)")

# 5. Machine-learning refinements
ml = {"ridge": res["v3 all groups"],
      "+ LightGBM matchup residual": wf(V3.fit_predict(ALL, trees="lgbm")),
      "+ random forest matchup residual": wf(V3.fit_predict(ALL, trees="rf")),
      "ridge + LightGBM stack": wf(V3.fit_predict(ALL, stack="lgbm")),
      "ridge + neural net (MLP) stack": wf(V3.fit_predict(ALL, stack="mlp")),
      "ridge alpha 100": wf(V3.fit_predict(ALL, alpha=100.0))}
show("Machine-learning refinements", ml, "ridge")

# 6. Roster-aware Elo season starts
proj = pd.read_parquet(PROCESSED / "team_projection.parquet")
elo = {f"Elo c={c}": wf(V3.fit_predict(ALL), with_elo(m, games, proj, c)) for c in (0.0, 0.5, 1.0)}
show("Elo season start (c = weight on roster projection)", elo, "Elo c=0.0")
