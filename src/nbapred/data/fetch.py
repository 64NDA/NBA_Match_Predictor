"""Cached, rate-limited downloads from stats.nba.com via nba_api.

Every response is stored as data/raw/<kind>_<season>_<type>.parquet. Completed seasons are
never re-downloaded; pass refresh=True for the in-progress season.
"""
import json
import time

import pandas as pd
from nba_api.stats.endpoints import (gamerotation, leaguegamefinder, playbyplayv3, playergamelogs,
                                     playerindex, scheduleleaguev2, teamgamelogs)

from nbapred.config import RAW, season_str

REQUEST_GAP_S = 0.6
TIMEOUT_S = 60
RETRIES = 3

_last_call = 0.0


def _throttled_all(fn):
    """Call an nba_api endpoint with spacing between requests and retries; return every data frame."""
    global _last_call
    for attempt in range(RETRIES):
        wait = REQUEST_GAP_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            _last_call = time.time()
            return fn().get_data_frames()
        except json.JSONDecodeError:
            raise  # empty body = the NBA has no data for this request; retrying won't help
        except Exception as e:  # nba_api surfaces timeouts/HTTP errors as plain exceptions
            if attempt == RETRIES - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}")
            time.sleep(5 * (attempt + 1))


def _throttled(fn):
    return _throttled_all(fn)[0]


def _cached(kind: str, season: int, season_type: str, fn, refresh: bool) -> pd.DataFrame:
    path = RAW / f"{kind}_{season}_{season_type.replace(' ', '')}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    print(f"fetching {kind} {season_str(season)} {season_type}")
    df = _throttled(fn)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def team_games(season: int, season_type: str, refresh: bool = False) -> pd.DataFrame:
    """Basic box score, one row per team per game."""
    return _cached("team_basic", season, season_type, lambda: leaguegamefinder.LeagueGameFinder(
        season_nullable=season_str(season), league_id_nullable="00",
        season_type_nullable=season_type, timeout=TIMEOUT_S), refresh)


def team_logs(season: int, season_type: str, measure: str, refresh: bool = False) -> pd.DataFrame:
    """Per-game team logs for a measure type ('Advanced', 'Four Factors', ...)."""
    kind = "team_" + measure.lower().replace(" ", "")
    return _cached(kind, season, season_type, lambda: teamgamelogs.TeamGameLogs(
        season_nullable=season_str(season), season_type_nullable=season_type,
        measure_type_player_game_logs_nullable=measure, timeout=TIMEOUT_S), refresh)


def player_logs(season: int, season_type: str, refresh: bool = False) -> pd.DataFrame:
    """Per-game player box scores."""
    return _cached("player_basic", season, season_type, lambda: playergamelogs.PlayerGameLogs(
        season_nullable=season_str(season), season_type_nullable=season_type,
        timeout=TIMEOUT_S), refresh)


# ---- v3: per-game and auxiliary data -------------------------------------------------------------
# Per-game responses are cached one file per game under data/raw/<kind>/<game_id>.parquet so a backfill
# can be interrupted and resumed at any point.


PBP_COLUMNS = ["gameId", "actionNumber", "clock", "period", "teamId", "personId", "scoreHome", "scoreAway",
               "actionType", "subType", "isFieldGoal", "shotResult", "shotValue", "shotDistance", "location",
               "description", "playerName", "playerNameI", "xLegacy", "yLegacy"]


def _per_game(kind: str, game_id: str, fn, refresh: bool = False) -> pd.DataFrame:
    path = RAW / kind / f"{game_id}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def rotation(game_id: str, refresh: bool = False) -> pd.DataFrame:
    """Player on-court intervals (tenths of a second from tip-off), both teams, with a home flag.

    Some older games (mostly 2015-16) have no rotation data; they're cached as an empty frame and
    lineups for them are rebuilt from play-by-play substitutions instead.
    """
    def fn():
        try:
            away, home = _throttled_all(lambda: gamerotation.GameRotation(game_id=game_id, timeout=TIMEOUT_S))
        except json.JSONDecodeError:
            return pd.DataFrame({"missing": pd.Series(dtype=bool)})
        return pd.concat([away.assign(home=0), home.assign(home=1)], ignore_index=True)
    return _per_game("rotations", game_id, fn, refresh)


def pbp(game_id: str, refresh: bool = False) -> pd.DataFrame:
    def fn():
        df = _throttled(lambda: playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=TIMEOUT_S))
        return df[[c for c in PBP_COLUMNS if c in df.columns]]
    return _per_game("pbp", game_id, fn, refresh)


def schedule_history(season: int, refresh: bool = False) -> pd.DataFrame:
    """Full season schedule incl. tip-off times (used to time-stamp injury reports)."""
    return _cached("schedule", season, "All", lambda: scheduleleaguev2.ScheduleLeagueV2(
        season=season_str(season), timeout=TIMEOUT_S), refresh)


def player_bio(refresh: bool = False) -> pd.DataFrame:
    """Every player in history: height, weight, position, draft slot (one call)."""
    path = RAW / "player_bio.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = _throttled(lambda: playerindex.PlayerIndex(historical_nullable="1", timeout=TIMEOUT_S))
    df.to_parquet(path, index=False)
    return df

