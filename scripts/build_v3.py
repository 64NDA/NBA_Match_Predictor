"""Build every v3 input and the v3 modelling table.

    python scripts/build_v3.py                 # all steps
    python scripts/build_v3.py --from ratings  # resume from a step (earlier outputs are read from disk)
    python scripts/build_v3.py --dev           # quick run on whatever data is downloaded (prior fit on the
                                               # latest season; for shaking out bugs, not for results)

Steps: stints -> ratings (box prior + weekly RAPM) -> candidates (availability history + injury reports)
       -> availability (P(plays) model) -> shots -> matrix
"""
import argparse
import sys
import time

import joblib
import numpy as np
import pandas as pd

from nbapred.config import PROCESSED
from nbapred.data import fetch, transactions as TX
from nbapred.data.build import load, tipoffs
from nbapred.features import availability as AV, impact as IM, matrix_v3 as MX, stints as ST, style as SY
from nbapred.features.projection import fit_playoff_curve, rookie_minutes
from nbapred.models.availability import AvailabilityModel, compare, predict_healthy

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
STEPS = ["stints", "ratings", "candidates", "availability", "shots", "matrix"]
parser = argparse.ArgumentParser()
parser.add_argument("--from", dest="start", default="stints", choices=STEPS)
parser.add_argument("--dev", action="store_true")
parser.add_argument("--incremental", action="store_true", help="reuse the box prior and past rating fits (daily)")
parser.add_argument("--elo-c", type=float, default=0.0, help="roster-aware Elo season start (0 = v2 behaviour)")
args = parser.parse_args()
run = lambda step: STEPS.index(step) >= STEPS.index(args.start)
t0 = time.time()
log = lambda msg: print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

games, tg, pg = load("games"), load("team_games"), load("player_games")
cfg = IM.ImpactConfig()

if run("stints"):
    if args.incremental and (PROCESSED / "stints.parquet").exists():
        old = ST.load_stints()
        new = ST.build_stints(games[~games.game_id.isin(old.game_id)], tg, pg)
        stints = pd.concat([old, new], ignore_index=True) if len(new) else old
        stints.to_parquet(PROCESSED / "stints.parquet", index=False)
    else:
        stints = ST.build_and_save()
    log(f"stints: {len(stints):,} from {stints.game_id.nunique():,} games")
else:
    stints = ST.load_stints()

lg_by_season = IM.league_ortg(stints)

if run("ratings"):
    box = IM.box_table(pg, tg)
    if args.dev and not args.incremental:
        IM.PRIOR_SEASONS = (int(stints.season.max()), int(stints.season.max()))
    lo, hi = IM.PRIOR_SEASONS
    prior = (joblib.load(PROCESSED / "box_prior.joblib") if args.incremental
             else IM.fit_box_prior(stints, box, cfg, float(lg_by_season.loc[lo:hi].mean())))
    log(f"box prior fitted on {IM.PRIOR_SEASONS}; rookie prior O={prior.rookie[0]:+.2f} D={prior.rookie[1]:+.2f}")
    start = stints.date.min() if args.dev else games.date.min()
    dates = IM.fit_dates(start, games.date.max())
    old = None
    if args.incremental and (PROCESSED / "player_ratings.parquet").exists():
        # daily use: keep past fits (they only use stints before their date, so they can't change) and
        # add the new ones. The last existing fit is redone in case its week's data arrived late.
        old = pd.read_parquet(PROCESSED / "player_ratings.parquet")
        keep_until = old["fit_date"].max()
        old = old[old["fit_date"] < keep_until]
        dates = [d for d in dates if d >= keep_until]
    new = IM.weekly_ratings(stints, box, prior, cfg, lg_by_season, dates)
    ratings = pd.concat([old, new], ignore_index=True) if old is not None else new
    IM.save(ratings, prior)
    log(f"ratings: {len(dates)} weekly fits")
else:
    ratings = pd.read_parquet(PROCESSED / "player_ratings.parquet")
    prior = joblib.load(PROCESSED / "box_prior.joblib")

if run("candidates"):
    keys = AV.name_keys(fetch.player_bio(), pg)
    cands, _ = AV.candidate_table(games, pg, tipoffs(games), keys, TX.load())
    AV.save(cands)
    log(f"candidates: {len(cands):,} rows, injury report coverage {(cands.status != 'no_report').mean():.1%}")
else:
    cands = AV.load_candidates()

if run("availability"):
    cmp = compare(cands)
    print(cmp.round(4).to_string())
    kind = min(("logistic", "lgbm"), key=lambda k: cmp.loc[k, "log_loss"])   # judged on 2021-24, fitted on 2016-20
    model = AvailabilityModel(kind).fit(cands)
    cands["p_live"] = model.predict(cands)
    cands["p_blind"] = model.predict(cands, blind=True)
    cands["p_healthy"] = predict_healthy(model, cands)
    AV.save(cands)
    joblib.dump(model, PROCESSED / "availability_model.joblib")
    log(f"availability model: {kind}")

if run("shots"):
    shots = SY.build_shots(games)
    log(f"shots: {len(shots):,} team-games")
else:
    shots = pd.read_parquet(PROCESSED / "team_shots.parquet")

if run("matrix"):
    curve = fit_playoff_curve(cands[cands.season <= 2020] if not args.dev else cands)
    heights = SY.heights(fetch.player_bio())
    fit_for_game = IM.ratings_for_games(ratings, games)
    bio = fetch.player_bio()
    rookie_min = rookie_minutes(bio.set_index("PERSON_ID")["DRAFT_NUMBER"])
    m = MX.build_matrix(games, tg, cands, ratings, fit_for_game.to_dict(), prior.rookie, curve, heights, shots,
                        rookie_min=rookie_min, elo_c=args.elo_c)
    joblib.dump(rookie_min, PROCESSED / "rookie_minutes.joblib")
    joblib.dump(curve, PROCESSED / "playoff_curve.joblib")
    MX.save(m)
    log(f"matrix: {m.shape}")
