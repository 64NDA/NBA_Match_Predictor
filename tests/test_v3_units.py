"""Unit tests for v3 building blocks (no processed data needed except where noted)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nbapred.config import RAW
from nbapred.data import injuries
from nbapred.features.projection import MAX_MINUTES, TEAM_MINUTES, expected_minutes
from nbapred.features.stints import _name_parts


@pytest.mark.parametrize("raw,key", [
    ("McCullarJr.,Kevin", "kevinmccullar"), ("Kevin McCullar Jr.", "kevinmccullar"),
    ("Dončić,Luka", "lukadoncic"), ("Luka Doncic", "lukadoncic"), ("Towns,Karl-Anthony", "karlanthonytowns")])
def test_injury_name_keys_match_across_formats(raw, key):
    assert injuries.name_key(raw) == key


def test_injury_parser_strips_glued_team_words():
    assert injuries._clean_name("PHI Minnesota Covington, Robert") == "Covington, Robert"
    assert injuries._clean_name("Orlando Carter Jr., Wendell") == "Carter Jr., Wendell"
    assert injuries._clean_name("Brooklyn Van Vleet, Fred") == "Van Vleet, Fred"


def test_report_slots_use_the_right_url_format():
    old = injuries._slots_before(datetime(2023, 1, 15, 18, 30))
    new = injuries._slots_before(datetime(2026, 1, 15, 18, 30))
    assert old[0][0] == "06PM" and all("_" not in s for s, _ in old)
    assert new[0][0] == "06_30PM"
    assert all(t <= datetime(2026, 1, 15, 18, 30) for _, t in new)


@pytest.mark.parametrize("text,parts", [("Jay. Williams", ("jay", "williams", "")), ("Carter Jr.", ("", "carter", "jr")),
                                        ("Dennis Schröder", ("dennis", "schroder", "")), ("L. James", ("l", "james", ""))])
def test_substitution_name_parsing(text, parts):
    assert _name_parts(text) == parts


def test_expected_minutes_fill_240_and_respect_caps():
    usual = np.array([36, 34, 32, 30, 28, 20, 18, 15, 12, 8.0])
    full = expected_minutes(np.ones(10), usual)
    assert full.sum() == pytest.approx(TEAM_MINUTES, abs=0.01)
    # star out: his minutes go to teammates, nobody past the cap
    p = np.ones(10); p[0] = 0
    e = expected_minutes(p, usual)
    assert e[0] == 0 and e.sum() == pytest.approx(TEAM_MINUTES, abs=0.01)
    assert (e <= np.minimum(MAX_MINUTES, usual * 1.35) + 1e-9).all()
    assert (e[1:] >= full[1:] - 1e-9).all()


def test_expected_minutes_trim_the_bench_not_the_stars():
    usual = np.array([36, 34, 30, 28, 26, 22, 20, 18, 15, 12, 10, 8, 6, 4, 4.0])  # 15 bodies, 273 usual minutes
    e = expected_minutes(np.ones(15), usual)
    assert e.sum() == pytest.approx(TEAM_MINUTES)
    assert (e[:9] == usual[:9]).all()        # the top of the rotation keeps its minutes
    assert e[-3:].sum() == 0                 # the end of the bench is cut


def test_pbp_lineups_agree_with_official_rotations():
    """Where official GameRotation data exists, lineups rebuilt from play-by-play must match it."""
    from nbapred.data.build import load
    from nbapred.features.stints import game_stints, pbp_stints
    games = [p.stem for p in (RAW / "rotations").glob("*.parquet") if (RAW / "pbp" / p.name).exists()]
    games = [g for g in games if not pd.read_parquet(RAW / "rotations" / f"{g}.parquet").empty]
    if not games:
        pytest.skip("no games with both rotation and play-by-play cached")
    meta, pg = load("games").set_index("game_id"), load("player_games")
    cols = [f"h{i}" for i in range(1, 6)] + [f"a{i}" for i in range(1, 6)]
    for g in games:
        pbp = pd.read_parquet(RAW / "pbp" / f"{g}.parquet")
        ref = game_stints(pd.read_parquet(RAW / "rotations" / f"{g}.parquet"), pbp)
        ours = pbp_stints(pbp, pg[pg.game_id == g][["player_id", "player", "team_id", "min"]],
                          meta.at[g, "home_id"], meta.at[g, "away_id"])
        lineups = set(map(tuple, ours[cols].to_numpy()))
        agree = sum(r.end - r.start for r in ref.itertuples() if tuple(getattr(r, c) for c in cols) in lineups)
        assert agree / (ref.end - ref.start).sum() > 0.97, g


@pytest.mark.parametrize("season_prefix", ["00215", "00216", "00224"])  # 2015-16/2016-17 feeds put 0-0 on missed FTs
def test_stint_points_are_sane_in_old_and_new_feeds(season_prefix):
    """Stint points must never be negative, must track stint length, and can't exceed the game's final score."""
    from nbapred.data.build import load
    from nbapred.features.stints import pbp_stints
    files = sorted((RAW / "pbp").glob(f"{season_prefix}*.parquet"))[:30]
    if not files:
        pytest.skip("play-by-play not cached")
    meta, pg = load("games").set_index("game_id"), load("player_games")
    totals, lengths = [], []
    for f in files:
        g = f.stem
        st = pbp_stints(pd.read_parquet(f), pg[pg.game_id == g][["player_id", "player", "team_id", "min"]],
                        meta.at[g, "home_id"], meta.at[g, "away_id"])
        assert (st[["home_pts", "away_pts"]] >= 0).all().all(), g
        assert st.home_pts.sum() <= meta.at[g, "home_pts"] and st.away_pts.sum() <= meta.at[g, "away_pts"], g
        totals += list(st.home_pts + st.away_pts)
        lengths += list(st.end - st.start)
    assert np.corrcoef(totals, lengths)[0, 1] > 0.8


def test_roster_moves_respect_dates():
    """Traded/waived players leave the pool on the day after the move; signings join before their first game."""
    from nbapred.features.availability import History
    h = History()
    box = lambda ids: pd.DataFrame({"player_id": ids, "min": [30.0] * len(ids)})
    h.update(1, box([10, 11, 12]), pd.Timestamp("2024-01-01"), 2023)      # team 1 plays with 10, 11, 12
    moves = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-06"]),
                          "action": ["join", "leave", "join"], "team_id": [2, 1, 1], "player_id": [10, 11, 99]})
    h.apply_moves(moves, pd.Timestamp("2024-01-05"))                        # nothing dated before the 5th
    assert h.pool(1) == {10, 11, 12}
    h.apply_moves(moves, pd.Timestamp("2024-01-06"))                        # the 5th's moves: 10 traded, 11 waived
    assert h.pool(1) == {12} and 10 in h.pool(2) and h.team_of[11] is None
    h.apply_moves(moves, pd.Timestamp("2024-01-07"))                        # 99 signed: a candidate before playing
    assert h.pool(1) == {12, 99}
    f = h.features(1, 99, pd.Timestamp("2024-01-07"), 2023)
    assert f["new_to_team"] == 1 and f["new_season"] == 0
    g = h.features(1, 12, pd.Timestamp("2024-10-20"), 2024)
    assert g["new_season"] == 1


def test_absences_do_not_carry_over_seasons():
    """Sitting out the end of a season (rest, tanking) must not look like an absence on opening night."""
    from nbapred.features.availability import History
    h = History()
    box = lambda ids: pd.DataFrame({"player_id": ids, "min": [30.0] * len(ids)})
    h.update(1, box([10, 11]), pd.Timestamp("2024-03-01"), 2023)
    for d in range(2, 12):                                         # 11 sits the last 10 games
        h.update(1, box([10]), pd.Timestamp(f"2024-03-{d:02d}"), 2023)
    assert h.features(1, 11, pd.Timestamp("2024-03-12"), 2023)["missed_streak"] == 10
    opener = h.features(1, 11, pd.Timestamp("2024-10-22"), 2024)
    assert np.isnan(opener["missed_streak"]) and np.isnan(opener["share_last10"]) and opener["season_games"] == 0
    h.update(1, box([10]), pd.Timestamp("2024-10-22"), 2024)         # misses the opener too
    assert h.features(1, 11, pd.Timestamp("2024-10-24"), 2024)["missed_streak"] == 1


def test_full_strength_restores_only_players_with_an_injury_signal():
    from nbapred.features.projection import full_strength_p, injury_signal
    c = pd.DataFrame({"status": ["Out", "none", "none", "Questionable"], "missed_streak": [3, 0, 2, np.nan]})
    signal = injury_signal(c)
    assert list(signal) == [True, False, True, True]         # listed, healthy, missed games, listed
    p_play = np.array([0.02, 0.60, 0.30, 0.50])               # the healthy player at 60% is a coach's-decision case
    p_healthy = np.array([0.97, 0.95, 0.96, 0.97])
    full = full_strength_p(p_play, p_healthy, signal)
    assert full[1] == 0.60                                     # untouched: not an injury
    assert list(full[[0, 2, 3]]) == [0.97, 0.96, 0.97]
