"""No look-ahead in v3: rebuilding everything from data truncated at a cutoff must reproduce every feature of
every earlier game, including the weekly player-rating fits and the availability table."""
import joblib
import numpy as np
import pandas as pd
import pytest

from nbapred.config import PROCESSED

NEEDED = ["matrix_v3.parquet", "player_ratings.parquet", "stints.parquet", "team_shots.parquet",
          "availability_model.joblib", "box_prior.joblib", "playoff_curve.joblib"]
pytestmark = pytest.mark.skipif(not all((PROCESSED / f).exists() for f in NEEDED), reason="run scripts/build_v3.py first")


def test_truncated_rebuild_matches():
    from nbapred.data import fetch
    from nbapred.data.build import load, tipoffs
    from nbapred.features import impact as IM, matrix_v3 as MX
    from nbapred.features.availability import candidate_table, name_keys
    from nbapred.features.stints import load_stints
    from nbapred.features.style import heights
    from nbapred.models.availability import predict_full

    games, tg, pg = load("games"), load("team_games"), load("player_games")
    stints, ratings = load_stints(), pd.read_parquet(PROCESSED / "player_ratings.parquet")
    prior, model = joblib.load(PROCESSED / "box_prior.joblib"), joblib.load(PROCESSED / "availability_model.joblib")
    curve, shots = joblib.load(PROCESSED / "playoff_curve.joblib"), pd.read_parquet(PROCESSED / "team_shots.parquet")
    full = pd.read_parquet(PROCESSED / "matrix_v3.parquet").set_index("game_id")

    fit_dates = np.sort(ratings["fit_date"].unique())
    cutoff = pd.Timestamp(fit_dates[len(fit_dates) // 2])
    cut = lambda d: d[d["date"] <= cutoff]
    g, t, p, s = cut(games), cut(tg), cut(pg), cut(stints)

    # player ratings: refit every fit date up to the cutoff from truncated stints/box scores
    lg = IM.league_ortg(s)
    dates = [pd.Timestamp(d) for d in fit_dates if d <= cutoff][-3:]
    re = IM.weekly_ratings(s, IM.box_table(p, t), prior, IM.ImpactConfig(), lg, dates)
    orig = ratings[ratings["fit_date"].isin(dates)]
    a = re.set_index(["fit_date", "player_id"]).sort_index()[["O", "D"]]
    b = orig.set_index(["fit_date", "player_id"]).sort_index()[["O", "D"]]
    assert a.index.equals(b.index)
    # a leak shifts ratings by ~1e-2 (seen when league averages used whole seasons); a different summation
    # order in the sparse solve moves them by ~1e-6
    assert (a - b).abs().to_numpy().max() < 1e-4, f"max rating change {(a - b).abs().to_numpy().max():.2e}"

    # matrix from truncated inputs
    keys = name_keys(fetch.player_bio(), pg)
    from nbapred.data import transactions
    cands, _ = candidate_table(g, p, tipoffs(games), keys, transactions.load())
    cands["p_live"], cands["p_blind"] = model.predict(cands), model.predict(cands, blind=True)
    cands["p_full"] = predict_full(model, cands)
    rt = ratings[ratings["fit_date"] <= cutoff]
    m = MX.build_matrix(g, t, cands, rt, IM.ratings_for_games(rt, g).to_dict(), prior.rookie, curve,
                        heights(fetch.player_bio()), shots[shots["game_id"].isin(g["game_id"])],
                        rookie_min=joblib.load(PROCESSED / "rookie_minutes.joblib")).set_index("game_id")
    cols = [c for c in m.columns if pd.api.types.is_numeric_dtype(m[c])]
    pd.testing.assert_frame_equal(m[cols], full.loc[m.index, cols], rtol=1e-7, check_dtype=False)
