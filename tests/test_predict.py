import pandas as pd
import pytest

from nbapred.config import PROCESSED

pytestmark = pytest.mark.skipif(not (PROCESSED / "games.parquet").exists(), reason="run scripts/build_data.py first")


@pytest.fixture(scope="module")
def data():
    from nbapred.data.build import load
    from nbapred.features.build import build_matrix
    g, tg, pg = load("games"), load("team_games"), load("player_games")
    return g, tg, pg, build_matrix(g, tg, pg)


@pytest.mark.parametrize("day", ["2024-02-10", "2024-10-22"])  # mid-season, and a season opener
def test_serving_features_match_training_matrix(data, day):
    """TeamState (used by the app) must reproduce the training features exactly."""
    from nbapred.models.predictors import FEATURE_SETS, TOTAL_FEATURES
    from nbapred.predict import TeamState

    g, tg, pg, matrix = data
    day = pd.Timestamp(day)
    state = TeamState.build(g, tg, pg, as_of=day)
    todays = matrix[matrix["date"] == day]
    assert len(todays) > 0
    cols = sorted(set(FEATURE_SETS["all"] + TOTAL_FEATURES + ["elo_prob"]))
    for _, r in todays.iterrows():
        lineup = pg[(pg["game_id"] == r.game_id)]
        live = state.features(r.home_id, r.away_id, day, r.season, r.is_playoff, r.neutral,
                              lineup[lineup.team_id == r.home_id]["player_id"].tolist(),
                              lineup[lineup.team_id == r.away_id]["player_id"].tolist()).iloc[0]
        pd.testing.assert_series_equal(live[cols].astype(float), r[cols].astype(float),
                                       check_names=False, rtol=1e-9)
