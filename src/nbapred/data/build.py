"""Turn raw nba_api responses into three tidy tables in data/processed/:

team_games.parquet   one row per team per game (both perspectives), box + advanced stats
games.parquet        one row per game, home/away orientation — the modelling unit
player_games.parquet one row per player per game played
"""
import numpy as np
import pandas as pd

from nbapred.config import PROCESSED, SEASON_TYPES, all_seasons
from nbapred.data import fetch

BASIC = {"PTS": "pts", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a", "FTM": "ftm",
         "FTA": "fta", "OREB": "oreb", "DREB": "dreb", "AST": "ast", "STL": "stl", "BLK": "blk",
         "TOV": "tov", "PF": "pf"}
ADVANCED = {"OFF_RATING": "off_rating", "DEF_RATING": "def_rating", "NET_RATING": "net_rating",
            "PACE": "pace", "POSS": "poss", "EFG_PCT": "efg_pct", "TS_PCT": "ts_pct",
            "TM_TOV_PCT": "tov_pct", "OREB_PCT": "oreb_pct", "DREB_PCT": "dreb_pct", "AST_PCT": "ast_pct"}
FOUR_FACTORS = {"FTA_RATE": "fta_rate"}
PLAYER = {"PLAYER_ID": "player_id", "PLAYER_NAME": "player", "TEAM_ID": "team_id", "GAME_ID": "game_id",
          "MIN": "min", "PLUS_MINUS": "plus_minus", **BASIC}


def _team_games_one(season: int, season_type: str, refresh: bool) -> pd.DataFrame:
    basic = fetch.team_games(season, season_type, refresh)
    if basic.empty:
        return basic
    adv = fetch.team_logs(season, season_type, "Advanced", refresh)
    ff = fetch.team_logs(season, season_type, "Four Factors", refresh)

    df = basic[["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP", *BASIC]].rename(
        columns={"GAME_ID": "game_id", "GAME_DATE": "date", "TEAM_ID": "team_id",
                 "TEAM_ABBREVIATION": "team", **BASIC})
    df = df.merge(adv[["GAME_ID", "TEAM_ID", *ADVANCED]].rename(
        columns={"GAME_ID": "game_id", "TEAM_ID": "team_id", **ADVANCED}), on=["game_id", "team_id"], how="left")
    df = df.merge(ff[["GAME_ID", "TEAM_ID", *FOUR_FACTORS]].rename(
        columns={"GAME_ID": "game_id", "TEAM_ID": "team_id", **FOUR_FACTORS}), on=["game_id", "team_id"], how="left")

    df["date"] = pd.to_datetime(df["date"])
    df["season"] = season
    df["season_type"] = season_type
    df["is_playoff"] = int(season_type != "Regular Season")
    # "LAL vs. DEN" = LAL at home, "LAL @ DEN" = LAL away. Neutral-site games (Paris, Mexico City,
    # NBA Cup knockouts) are "A @ B" on *both* rows; B is the designated home team.
    matchup = df.pop("MATCHUP")
    has_vs = matchup.str.contains(" vs. ")
    df["neutral"] = (~has_vs.groupby(df["game_id"]).transform("any")).astype(int)
    designated_home = matchup.str.split(" @ ").str[-1]
    df["home"] = np.where(df["neutral"] == 1, df["team"] == designated_home, has_vs).astype(int)
    return df


def build_team_games(refresh_season: int | None = None) -> pd.DataFrame:
    parts = [_team_games_one(s, t, refresh=(s == refresh_season))
             for s in all_seasons() for t in SEASON_TYPES]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    # Attach the opponent's row from the same game
    opp = df[["game_id", "team_id", "team", "pts"]].rename(
        columns={"team_id": "opp_id", "team": "opp", "pts": "opp_pts"})
    df = df.merge(opp, on="game_id")
    df = df[df["team_id"] != df["opp_id"]]
    df["margin"] = df["pts"] - df["opp_pts"]
    df["win"] = (df["margin"] > 0).astype(int)
    return df.sort_values(["date", "game_id", "home"]).reset_index(drop=True)


def build_games(team_games: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_id", "date", "season", "season_type", "is_playoff", "neutral"]
    h = team_games[team_games["home"] == 1]
    a = team_games[team_games["home"] == 0]
    g = h[keys + ["team_id", "team", "pts"]].merge(
        a[["game_id", "team_id", "team", "pts"]], on="game_id", suffixes=("_home", "_away"))
    g = g.rename(columns={"team_id_home": "home_id", "team_home": "home_team", "pts_home": "home_pts",
                          "team_id_away": "away_id", "team_away": "away_team", "pts_away": "away_pts"})
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["total"] = g["home_pts"] + g["away_pts"]
    g["home_win"] = (g["margin"] > 0).astype(int)
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def build_player_games(team_games: pd.DataFrame, refresh_season: int | None = None) -> pd.DataFrame:
    parts = []
    for s in all_seasons():
        for t in SEASON_TYPES:
            p = fetch.player_logs(s, t, refresh=(s == refresh_season))
            if not p.empty:
                parts.append(p[list(PLAYER)].rename(columns=PLAYER))
    p = pd.concat(parts, ignore_index=True)
    p = p.merge(team_games[["game_id", "team_id", "date", "season", "is_playoff"]], on=["game_id", "team_id"])
    # Hollinger Game Score: a cheap single-number box-score rating per player-game
    p["game_score"] = (p.pts + 0.4 * p.fgm - 0.7 * p.fga - 0.4 * (p.fta - p.ftm) + 0.7 * p.oreb
                       + 0.3 * p.dreb + p.stl + 0.7 * p.ast + 0.7 * p.blk - 0.4 * p.pf - p.tov)
    return p.sort_values(["date", "game_id", "team_id"]).reset_index(drop=True)


def build_all(refresh_season: int | None = None) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tg = build_team_games(refresh_season)
    tg.to_parquet(PROCESSED / "team_games.parquet", index=False)
    build_games(tg).to_parquet(PROCESSED / "games.parquet", index=False)
    build_player_games(tg, refresh_season).to_parquet(PROCESSED / "player_games.parquet", index=False)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")
