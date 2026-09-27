"""Final model: one-off holdout evaluation, then fit on everything and save.

    python scripts/train.py --holdout   # evaluate once on HOLDOUT_SEASON (train on seasons before it)
    python scripts/train.py             # fit production model on all seasons -> models/model.joblib
"""
import argparse
import json
from datetime import date

import joblib

from nbapred.config import HOLDOUT_SEASON, MODELS
from nbapred.features.build import load_matrix
from nbapred.models import predictors as P
from nbapred.models.evaluate import FIRST_TRAIN, calibration, metrics, report, walk_forward

FEATURES = P.FEATURE_SETS["all"]

parser = argparse.ArgumentParser()
parser.add_argument("--holdout", action="store_true")
args = parser.parse_args()
m = load_matrix()
MODELS.mkdir(exist_ok=True)

if args.holdout:
    # walk_forward() refuses the holdout on purpose, so split explicitly here
    train, test = m[(m.season >= FIRST_TRAIN) & (m.season < HOLDOUT_SEASON)], m[m.season == HOLDOUT_SEASON]
    model = P.MarginTotalModel(FEATURES).fit(train)
    preds = test[["season", "is_playoff", "home_win", "margin", "total"]].join(model.predict(test))
    elo = test[["home_win", "margin", "total"]].join(P.elo_baseline(train, test))
    home = test[["home_win", "margin", "total"]].join(P.home_baseline(train, test))
    result = {"model": metrics(preds), "elo": metrics(elo), "home_team": metrics(home)}
    print(f"HOLDOUT {HOLDOUT_SEASON}:")
    for k, v in result.items():
        print(f"  {k:10s}", {a: round(float(b), 4) for a, b in v.items()})
    print(report(preds, by="is_playoff"))
    print(calibration(preds))
    (MODELS / "holdout_metrics.json").write_text(json.dumps(
        {k: {a: float(b) for a, b in v.items()} for k, v in result.items()}, indent=2))
else:
    train = m[m.season >= FIRST_TRAIN]
    model = P.MarginTotalModel(FEATURES).fit(train)
    cv = metrics(walk_forward(m, P.margin_total(FEATURES)))
    meta = {"trained": date.today().isoformat(), "seasons": [int(train.season.min()), int(train.season.max())],
            "games": len(train), "sigma": model.sigma, "features": FEATURES,
            "walk_forward": {k: float(v) for k, v in cv.items()}}
    joblib.dump(model, MODELS / "model.joblib")
    (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta | {"features": len(FEATURES)}, indent=2))
