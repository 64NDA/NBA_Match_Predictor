import numpy as np
import pandas as pd
import pytest

from nbapred.config import PROCESSED
from nbapred.features.elo import Elo
from nbapred.features.gamescore import PlayerTracker

has_data = pytest.mark.skipif(not (PROCESSED / "games.parquet").exists(), reason="run scripts/build_data.py first")


@has_data
def test_no_lookahead_leakage():
    """Features for games before a cutoff must be identical whether or not later games exist."""
    from nbapred.data.build import load
    from nbapred.features.matrix_v2 import build_matrix

    g, tg, pg = load("games"), load("team_games"), load("player_games")
    g, tg, pg = (d[d["season"] >= 2023] for d in (g, tg, pg))
    cutoff = pd.Timestamp("2024-01-15")

    full = build_matrix(g, tg, pg).set_index("game_id")
    cut = build_matrix(*(d[d["date"] <= cutoff] for d in (g, tg, pg))).set_index("game_id")

    targets = {"margin", "total", "home_win", "home_pts", "away_pts"}
    cols = [c for c in cut.columns if c not in targets and pd.api.types.is_numeric_dtype(cut[c])]
    pd.testing.assert_frame_equal(cut[cols], full.loc[cut.index, cols], check_exact=False, rtol=1e-9)


def test_elo_home_advantage_and_symmetry():
    elo = Elo()
    assert 0.55 < elo.win_prob(1500, 1500) < 0.65
    assert elo.win_prob(1500, 1500, neutral=True) == pytest.approx(0.5)
    rh, ra, _ = elo.update(1, 2, margin=10, season=2020)
    assert elo.ratings[1] - rh == pytest.approx(ra - elo.ratings[2])  # zero-sum


def test_elo_regresses_between_seasons():
    elo = Elo(ratings={1: 1700.0}, season=2020)
    elo._new_season(2021)
    assert elo.ratings[1] == pytest.approx(1505 + 0.75 * 195)


def test_missing_star_lowers_available_strength():
    t = PlayerTracker()
    box = pd.DataFrame({"player_id": range(9), "player": list("abcdefghi"),
                        "game_score": [30, 12, 10, 9, 8, 7, 6, 5, 4]})
    for _ in range(30):
        t.process_game(1, box)
    full, avail_all = t.strength(1, list(range(9)))
    _, avail_no_star = t.strength(1, list(range(1, 9)))
    assert full == pytest.approx(avail_all)
    assert full - avail_no_star > 15
