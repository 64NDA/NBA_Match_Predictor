"""Final models: one-off holdout evaluation, then fit on everything and save.

    python scripts/train.py --version v3 --holdout   # evaluate once on HOLDOUT_SEASON (train on seasons before it)
    python scripts/train.py --version v3             # fit production model on all seasons -> models/v3/
    python scripts/train.py --version v2 [--holdout] # legacy v2 model -> models/v2/

v3 settings come from models/v3/config.json (written from the experiment results in analysis/).
"""
import argparse
import json
from datetime import date

import joblib
import pandas as pd

from nbapred.config import HOLDOUT_SEASON, MODELS, PROCESSED
from nbapred.data.build import load
from nbapred.models import baselines as B
from nbapred.models.evaluate import FIRST_TRAIN, calibration, metrics, report, walk_forward

parser = argparse.ArgumentParser()
parser.add_argument("--version", choices=["v2", "v3"], default="v3")
parser.add_argument("--holdout", action="store_true")
args = parser.parse_args()
OUT = MODELS / args.version
OUT.mkdir(parents=True, exist_ok=True)
jsonable = lambda d: {k: {a: float(b) for a, b in v.items()} for k, v in d.items()}


def v2_matrix_and_model():
    from nbapred.features.matrix_v2 import load_matrix
    from nbapred.models import v2 as P
    return load_matrix(), P.FEATURE_SETS["all"], lambda: P.MarginTotalModel(P.FEATURE_SETS["all"])


if args.version == "v2":
    from nbapred.models import v2 as P
    m, features, make = v2_matrix_and_model()
    if args.holdout:
        # walk_forward() refuses the holdout on purpose, so split explicitly here
        train, test = m[(m.season >= FIRST_TRAIN) & (m.season < HOLDOUT_SEASON)], m[m.season == HOLDOUT_SEASON]
        preds = test[["season", "is_playoff", "home_win", "margin", "total"]].join(make().fit(train).predict(test))
        result = {"model": metrics(preds),
                  "elo": metrics(test[["home_win", "margin", "total"]].join(B.elo_baseline(train, test))),
                  "home_team": metrics(test[["home_win", "margin", "total"]].join(B.home_baseline(train, test)))}
        print(f"HOLDOUT {HOLDOUT_SEASON}:", json.dumps(jsonable(result), indent=1))
        (OUT / "holdout_metrics.json").write_text(json.dumps(jsonable(result), indent=2))
    else:
        train = m[m.season >= FIRST_TRAIN]
        model = make().fit(train)
        cv = metrics(walk_forward(m, P.margin_total(features)))
        meta = {"trained": date.today().isoformat(), "seasons": [int(train.season.min()), int(train.season.max())],
                "games": len(train), "sigma": model.sigma, "features": features,
                "walk_forward": {k: float(v) for k, v in cv.items()}}
        joblib.dump(model, OUT / "model.joblib")
        (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta | {"features": len(features)}, indent=2))

else:
    from nbapred.features.matrix_v3 import load_matrix, with_elo
    from nbapred.models import v2 as P2, v3 as V3
    cfg = json.loads((OUT / "config.json").read_text())
    m = load_matrix()
    if cfg.get("elo_c", 0):
        m = with_elo(m, load("games"), pd.read_parquet(PROCESSED / "team_projection.parquet"), cfg["elo_c"])
    kw = dict(alpha=cfg.get("alpha", 10.0), trees=cfg.get("trees"), stack=cfg.get("stack"))
    cond = cfg.get("train_condition", "live")
    make = lambda: V3.V3Model(cfg["groups"], cond, **kw)
    keep = ["game_id", "season", "is_playoff", "home_win", "margin", "total"]
    if args.holdout:
        train, test = m[(m.season >= FIRST_TRAIN) & (m.season < HOLDOUT_SEASON)], m[m.season == HOLDOUT_SEASON]
        model = make().fit(train)
        result = {f"v3 ({c})": metrics(test[keep].join(model.predict(test, c))) for c in ("live", "known", "blind")}
        # v2, Elo and home team on the same games
        m2, _, make2 = v2_matrix_and_model()
        tr2, te2 = m2[(m2.season >= FIRST_TRAIN) & (m2.season < HOLDOUT_SEASON)], m2[m2.season == HOLDOUT_SEASON]
        v2_model = make2().fit(tr2)
        result["v2 (live)"] = metrics(te2[keep].join(v2_model.predict(P2.no_availability(te2))))
        result["v2 (known)"] = metrics(te2[keep].join(v2_model.predict(te2)))
        result["elo"] = metrics(te2[keep].join(B.elo_baseline(tr2, te2)))
        result["home_team"] = metrics(te2[keep].join(B.home_baseline(tr2, te2)))
        preds = test[keep].join(model.predict(test, "live"))
        print(f"HOLDOUT {HOLDOUT_SEASON}:")
        print(pd.DataFrame(jsonable(result)).T.round(4).to_string())
        print(report(preds, by="is_playoff").round(4).to_string())
        print(calibration(preds).round(3).to_string())
        (OUT / "holdout_metrics.json").write_text(json.dumps(jsonable(result), indent=2))
    else:
        train = m[m.season >= FIRST_TRAIN]
        model = make().fit(train)
        cv = metrics(walk_forward(m, V3.fit_predict(cfg["groups"], cond, **kw)))
        meta = {"trained": date.today().isoformat(), "seasons": [int(train.season.min()), int(train.season.max())],
                "games": len(train), "sigma": model.sigma, "config": cfg,
                "walk_forward": {k: float(v) for k, v in cv.items()}}
        joblib.dump(model, OUT / "model.joblib")
        (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
