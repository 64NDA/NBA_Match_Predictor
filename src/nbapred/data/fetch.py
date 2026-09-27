"""Cached, rate-limited downloads from stats.nba.com via nba_api.

Every response is stored as data/raw/<kind>_<season>_<type>.parquet. Completed seasons are
never re-downloaded; pass refresh=True for the in-progress season.
"""
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs, teamgamelogs

from nbapred.config import RAW, season_str

REQUEST_GAP_S = 0.8
TIMEOUT_S = 60
RETRIES = 3

_last_call = 0.0


def _throttled(fn):
    global _last_call
    for attempt in range(RETRIES):
        wait = REQUEST_GAP_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            _last_call = time.time()
            return fn().get_data_frames()[0]
        except Exception as e:  # nba_api surfaces timeouts/HTTP errors as plain exceptions
            if attempt == RETRIES - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}")
            time.sleep(5 * (attempt + 1))


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
