"""Final v3 selection: candidate configs head to head, and fair v2 comparisons (same availability information).

    python analysis/v3_selection.py > analysis/v3_selection_output.txt

v2 only ever saw "who played" in its backtest. Live, without injury data, it has to assume everyone plays, so
the fair live comparison is v2 predicting with no availability information vs v3 with the injury report.
"""
import sys

import pandas as pd

from nbapred.features import matrix_v2 as M2
from nbapred.features.matrix_v3 import load_matrix
from nbapred.models import v2 as P2, v3 as V3
from nbapred.models.evaluate import paired, walk_forward

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
sys.path.insert(0, "analysis")
from experiments import live_like  # v2 predicting with "everyone available" (no injury information)

m, m2 = load_matrix(), M2.load_matrix()
wf = lambda fp, mx=m: walk_forward(mx, fp)
ALL = list(V3.GROUPS)
CANDIDATES = {
    "A all groups": ALL,
    "B all - Matchup & style": [g for g in ALL if g != "Matchup & style"],
    "C minimal (home, players, rest)": ["Home court", "Player ratings", "Rest & travel"],
    "D C + Elo + Injuries + Playoff": ["Home court", "Team strength (Elo)", "Player ratings", "Injuries",
                                      "Rest & travel", "Playoff game"],
}
res = {name: wf(V3.fit_predict(groups)) for name, groups in CANDIDATES.items()}
print("== Candidate configs (train + test with the injury report) ==")
print(paired(res, "A all groups").round(4).to_string())

sig = {name: round(V3.V3Model(groups).fit(m[m.season.between(2016, 2024)]).sigma, 2) for name, groups in CANDIDATES.items()}
print("\nsigma (fit on 2016-2024):", sig)

print("\n== Fair comparisons with v2 ==")
fair = {
    "v3 D, injury report (live)": res["D C + Elo + Injuries + Playoff"],
    "v2, live (no availability info)": walk_forward(m2, live_like(P2.FEATURE_SETS["all"])),
    "v2, knowing who played": walk_forward(m2, P2.margin_total(P2.FEATURE_SETS["all"])),
    "v3 D, knowing who played": wf(V3.fit_predict(CANDIDATES["D C + Elo + Injuries + Playoff"], "known")),
    "v3 D, no injury report": wf(V3.fit_predict(CANDIDATES["D C + Elo + Injuries + Playoff"], "blind")),
}
print(paired(fair, "v2, live (no availability info)").round(4).to_string())
print("\nknown-lineup ceiling, v3 vs v2:")
print(paired({k: fair[k] for k in ("v2, knowing who played", "v3 D, knowing who played")}, "v2, knowing who played").round(4).to_string())
