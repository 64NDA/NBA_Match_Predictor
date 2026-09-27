"""Live data for the app: schedule, rosters, upcoming predictions, prediction log, app exports."""
import shutil
from datetime import date

import joblib
import pandas as pd
from nba_api.stats.endpoints import commonteamroster, scheduleleaguev2
from nba_api.stats.static import teams as static_teams

from nbapred.config import APP_DATA, HOLDOUT_SEASON, MODELS, RAW, season_str
from nbapred.data.fetch import _throttled
from nbapred.models import predictors as P
from nbapred.models.evaluate import FIRST_TRAIN, walk_forward
from nbapred.predict import TeamState

# gameId prefix -> season type. 001 preseason, 003 all-star, 006 NBA Cup final (exhibition) are skipped
GAME_TYPES = {"002": 0, "004": 1, "005": 1}


def team_names() -> pd.DataFrame:
    return pd.DataFrame(static_teams.get_teams()).rename(
        columns={"id": "team_id", "abbreviation": "team", "full_name": "name"})[["team_id", "team", "name"]]


def schedule(season: int) -> pd.DataFrame:
    s = _throttled(lambda: scheduleleaguev2.ScheduleLeagueV2(season=season_str(season), timeout=60))
    s = s[s["gameId"].str[:3].isin(GAME_TYPES)]
    return pd.DataFrame({
        "game_id": s["gameId"], "date": pd.to_datetime(s["gameDateEst"]).dt.tz_localize(None).dt.normalize(),
        "tipoff_utc": s["gameDateTimeUTC"], "status": s["gameStatus"], "is_playoff": s["gameId"].str[:3].map(GAME_TYPES),
        "neutral": s["isNeutral"].astype(bool).astype(int), "home_id": s["homeTeam_teamId"], "away_id": s["awayTeam_teamId"],
        "label": s["gameLabel"].fillna(""), "arena": s["arenaName"]})


def rosters(season: int) -> tuple[dict, dict]:
    """Official rosters ({team_id: [player_id]}, {player_id: name}), cached per day."""
    path = RAW / f"rosters_{date.today().isoformat()}.parquet"
    if path.exists():
        r = pd.read_parquet(path)
    else:
        parts = []
        for t in team_names()["team_id"]:
            df = _throttled(lambda: commonteamroster.CommonTeamRoster(team_id=t, season=season_str(season), timeout=60))
            parts.append(df[["TeamID", "PLAYER_ID", "PLAYER"]])
        r = pd.concat(parts)
        r.to_parquet(path, index=False)
    return (r.groupby("TeamID")["PLAYER_ID"].apply(list).to_dict(),
            dict(zip(r["PLAYER_ID"], r["PLAYER"])))


def predict_games(model, state: TeamState, games: pd.DataFrame, season: int) -> pd.DataFrame:
    """games: schedule rows. Processes in date order so rest/travel account for earlier scheduled games."""
    names = team_names().set_index("team_id")
    rows = []
    for g in games.sort_values(["date", "tipoff_utc"]).itertuples():
        x = state.features(g.home_id, g.away_id, g.date, season, g.is_playoff, g.neutral)
        pred = model.predict(x).iloc[0]
        why = P.grouped_explanation(model, x.iloc[0])
        rows.append({"game_id": g.game_id, "date": g.date, "tipoff_utc": g.tipoff_utc, "label": g.label,
                     "home_id": g.home_id, "away_id": g.away_id,
                     "home_team": names.loc[g.home_id, "team"], "away_team": names.loc[g.away_id, "team"],
                     **pred.to_dict(), "elo_prob": x["elo_prob"].iloc[0],
                     "home_missing": x["h_missing_strength"].iloc[0], "away_missing": x["a_missing_strength"].iloc[0],
                     **{f"why_{k}": v for k, v in why.items()}})
        host = names.loc[g.home_id, "team"]
        state.add_scheduled(g.home_id, g.date, host)
        state.add_scheduled(g.away_id, g.date, host)
    return pd.DataFrame(rows)


def update_log(upcoming: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest *pre-game* prediction per game and attach results once games are played."""
    path = APP_DATA / "prediction_log.parquet"
    log = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    stamp = pd.Timestamp.now().floor("s")
    if not upcoming.empty:
        new = upcoming.assign(predicted_at=stamp)
        log = pd.concat([log[~log["game_id"].isin(new["game_id"])] if len(log) else log, new], ignore_index=True)
    if len(log):
        res = games[["game_id", "home_pts", "away_pts", "margin", "total", "home_win"]]
        log = log.drop(columns=[c for c in res.columns if c != "game_id" and c in log], errors="ignore").merge(
            res, on="game_id", how="left")
        log.to_parquet(path, index=False)
    return log


def export_backtest(matrix: pd.DataFrame) -> None:
    """Out-of-sample predictions: walk-forward seasons + the holdout (model trained on seasons before it)."""
    wf = walk_forward(matrix, P.margin_total(P.FEATURE_SETS["all"]))
    train = matrix[(matrix.season >= FIRST_TRAIN) & (matrix.season < HOLDOUT_SEASON)]
    test = matrix[matrix.season == HOLDOUT_SEASON]
    ho = test[wf.columns.intersection(test.columns)].join(P.MarginTotalModel(P.FEATURE_SETS["all"]).fit(train).predict(test))
    keep = ["game_id", "date", "season", "is_playoff", "home_team", "away_team", "home_win", "margin", "total",
            "p_home", "pred_margin", "pred_total"]
    bt = pd.concat([wf, ho])[keep].merge(matrix[["game_id", "elo_prob"]], on="game_id")
    bt.to_parquet(APP_DATA / "backtest.parquet", index=False)


def export_elo_history(matrix: pd.DataFrame) -> None:
    h = matrix[["date", "season", "home_team", "home_elo"]].set_axis(["date", "season", "team", "elo"], axis=1)
    a = matrix[["date", "season", "away_team", "away_elo"]].set_axis(["date", "season", "team", "elo"], axis=1)
    pd.concat([h, a]).sort_values("date").to_parquet(APP_DATA / "elo_history.parquet", index=False)


def save_state(state: TeamState) -> None:
    joblib.dump(state, APP_DATA / "state.joblib")
    shutil.copy(MODELS / "model.joblib", APP_DATA / "model.joblib")
