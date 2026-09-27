import pytest

from nbapred.config import PROCESSED

pytestmark = pytest.mark.skipif(not (PROCESSED / "games.parquet").exists(), reason="run scripts/build_data.py first")


@pytest.fixture(scope="module")
def games():
    from nbapred.data.build import load
    return load("games")


@pytest.fixture(scope="module")
def team_games():
    from nbapred.data.build import load
    return load("team_games")


def test_one_row_per_game(games):
    assert games["game_id"].is_unique


def test_two_team_rows_per_game_one_home(team_games):
    per_game = team_games.groupby("game_id")["home"].agg(["size", "sum"])
    assert (per_game["size"] == 2).all()
    assert (per_game["sum"] == 1).all()


def test_no_ties_and_distinct_teams(games):
    assert (games["margin"] != 0).all()
    assert (games["home_id"] != games["away_id"]).all()


def test_regular_season_sizes(games):
    rs = games[games["season_type"] == "Regular Season"].groupby("season").size()
    shortened = {2019: 1059, 2020: 1080}  # 2019-20: 971 + 88 bubble games; 2020-21: 72 games
    for season, n in rs.items():
        assert n == shortened.get(season, 1230), f"{season}: {n} games"


def test_scores_agree_between_tables(games, team_games):
    home = team_games[team_games["home"] == 1].set_index("game_id")["pts"]
    assert (games.set_index("game_id")["home_pts"] == home.loc[games["game_id"]].values).all()


def test_advanced_stats_present(team_games):
    assert team_games["off_rating"].notna().mean() > 0.99
