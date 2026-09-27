"""The v3 serving snapshot must reproduce the v3 training matrix exactly (train/serve parity)."""
import joblib
import numpy as np
import pandas as pd
import pytest

from nbapred.config import PROCESSED

NEEDED = ["matrix_v3.parquet", "player_ratings.parquet", "candidates.parquet", "team_shots.parquet",
          "team_projection.parquet", "availability_model.joblib", "box_prior.joblib", "playoff_curve.joblib",
          "rookie_minutes.joblib"]
pytestmark = pytest.mark.skipif(not all((PROCESSED / f).exists() for f in NEEDED), reason="run scripts/build_v3.py first")


@pytest.fixture(scope="module")
def data():
    from nbapred.data import fetch
    from nbapred.data.build import load, tipoffs
    from nbapred.features.availability import name_keys
    from nbapred.features.style import heights
    g, tg, pg = load("games"), load("team_games"), load("player_games")
    return dict(games=g, team_games=tg, player_games=pg, shots=pd.read_parquet(PROCESSED / "team_shots.parquet"),
                ratings=pd.read_parquet(PROCESSED / "player_ratings.parquet"),
                prior=joblib.load(PROCESSED / "box_prior.joblib"), curve=joblib.load(PROCESSED / "playoff_curve.joblib"),
                model=joblib.load(PROCESSED / "availability_model.joblib"),
                keys=name_keys(fetch.player_bio(), pg), tips=tipoffs(g), heights=heights(fetch.player_bio()),
                proj=pd.read_parquet(PROCESSED / "team_projection.parquet"),
                rookie_min=joblib.load(PROCESSED / "rookie_minutes.joblib"),
                matrix=pd.read_parquet(PROCESSED / "matrix_v3.parquet"))


def model_columns():
    from nbapred.models.v3 import GROUPS, PER_CONDITION, TOTAL_FEATURES, TREE_FEATURES
    from nbapred.features.matrix_v3 import CONDITIONS
    cols = set(TOTAL_FEATURES + TREE_FEATURES + ["elo_prob"]) | {f for fs in GROUPS.values() for f in fs}
    expanded = set()
    for c in cols:
        expanded |= {f"{c}_{k}" for k in CONDITIONS} if c in PER_CONDITION else {c}
    return sorted(expanded)


def test_serving_matches_training(data):
    from nbapred.data import injuries
    from nbapred.features.availability import resolve_report
    from nbapred.data import transactions
    from nbapred.serving_v3 import TeamState

    d, m = data, data["matrix"]
    rated = m[m["date"] > d["ratings"]["fit_date"].min() + pd.Timedelta(days=30)]
    day = rated["date"].drop_duplicates().sort_values().iloc[len(rated["date"].unique()) // 2]
    state = TeamState.build(d["games"], d["team_games"], d["player_games"], d["shots"], d["ratings"], d["prior"].rookie,
                            d["curve"], d["heights"], d["rookie_min"], d["model"], d["keys"], d["tips"], d["proj"],
                            transactions.load(), as_of=day)
    cols = model_columns()
    pg = d["player_games"]
    todays = m[m["date"] == day]
    assert len(todays) > 0
    for _, r in todays.iterrows():
        rep = injuries.report_before(d["tips"][r.game_id].to_pydatetime(), offline=True)
        status = resolve_report(rep, state.history.team_of, d["keys"]) if rep is not None else None
        box = pg[pg["game_id"] == r.game_id]
        played = {t: set(b["player_id"]) for t, b in box.groupby("team_id")}
        live = state.features(r.home_id, r.away_id, day, r.season, r.is_playoff, r.neutral, status,
                              known=played).iloc[0]
        np.testing.assert_allclose(live[cols].astype(float).to_numpy(), r[cols].astype(float).to_numpy(),
                                   rtol=1e-7, atol=1e-9, err_msg=f"{r.game_id}: " + ", ".join(
                                       c for c in cols if not np.isclose(live[c], r[c], rtol=1e-7, atol=1e-9)))
