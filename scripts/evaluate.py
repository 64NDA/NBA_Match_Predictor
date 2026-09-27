"""Walk-forward comparison of baselines, feature-set ablation, and model types.

    python scripts/evaluate.py            # rebuilds the feature matrix first
    python scripts/evaluate.py --cached   # reuse data/processed/matrix.parquet

Never touches the holdout season (see config.HOLDOUT_SEASON).
"""
import argparse

import pandas as pd

from nbapred.features.matrix_v2 import build_and_save, load_matrix
from nbapred.models import baselines as B, v2 as P
from nbapred.models.evaluate import calibration, compare, report, walk_forward

pd.set_option("display.width", 160, "display.precision", 3)

parser = argparse.ArgumentParser()
parser.add_argument("--cached", action="store_true")
args = parser.parse_args()
m = load_matrix() if args.cached else build_and_save()

fs = P.FEATURE_SETS
print("\n== Baselines & feature-set ablation (ridge margin model) ==")
print(compare(m, {
    "home team": B.home_baseline,
    "elo": B.elo_baseline,
    **{f"ridge[{k}]": P.margin_total(v) for k, v in fs.items()},
}))

print("\n== Model types on all features ==")
best = P.margin_total(fs["all"])
print(compare(m, {
    "ridge": best,
    "lgbm": P.margin_total(fs["all"], "lgbm", "lgbm"),
    "logistic": B.logistic(fs["all"]),
    "ridge+lgbm": B.ensemble(best, P.margin_total(fs["all"], "lgbm", "lgbm")),
}))

preds = walk_forward(m, best)
print("\n== Ridge[all] per season ==")
print(report(preds))
print("\n== Ridge[all] regular season vs playoffs ==")
print(report(preds, by="is_playoff"))
print("\n== Calibration ==")
print(calibration(preds))
