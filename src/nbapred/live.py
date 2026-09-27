"""Live data for the app: schedule, rosters, upcoming predictions, prediction log, app exports."""
import shutil
from datetime import date

import joblib
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import commonteamroster, scheduleleaguev2
from nba_api.stats.static import teams as static_teams

from nbapred.config import APP_DATA, HOLDOUT_SEASON, MODELS, RAW, season_str
from nbapred.data.fetch import _throttled
from nbapred.models import baselines as B, v2 as P
from nbapred.models.evaluate import FIRST_TRAIN, walk_forward
from nbapred.serving_v2 import TeamState

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


# ---- v3 ------------------------------------------------------------------------------------------

def injury_detail(state, team: int, date, season: int, status: dict | None, roster: list | None,
                  is_playoff: int = 0) -> list[dict]:
    """Players with an injury signal (on the injury report, or missed the team's recent games this season) who have a
    real role (8+ usual minutes). For each: status, chance of playing, and `effect`: the change in team strength
    (points per 100 possessions, about points per game) from his likely absence, after his minutes go to teammates.
    Everyone not listed is expected to play his normal minutes."""
    lu = state.lineup(team, pd.Timestamp(date), season, status, roster, is_playoff)
    if lu.empty:
        return []
    effects = state.absence_effects(lu).set_index("player")["effect"]
    signal = lu["status"].isin(["Out", "Doubtful", "Questionable", "Probable"]) | (lu["missed_streak"].fillna(0) >= 1)
    out = []
    for r in lu[signal & (lu["usual_min"] >= 8)].itertuples():
        out.append({"player_id": int(r.player_id), "player": r.player,
                    "status": r.status if r.status in ("Out", "Doubtful", "Questionable", "Probable") else "not listed",
                    "missed": int(r.missed_streak) if r.missed_streak == r.missed_streak else 0,
                    "p_play": float(r.p_play), "usual_min": float(r.usual_min),
                    "effect": float(effects.get(r.player, 0.0))})
    return sorted(out, key=lambda d: d["effect"])


def absence_total(state, team: int, date, season: int, status: dict | None, roster: list | None,
                  is_playoff: int = 0) -> float:
    """Team strength tonight minus at full strength (points per 100 possessions): the combined effect of absences."""
    lu = state.lineup(team, pd.Timestamp(date), season, status, roster, is_playoff)
    return float(lu["value_tonight"].sum() - lu["value_full"].sum()) if len(lu) else 0.0


def team_summary(games: pd.DataFrame, state, season: int) -> pd.DataFrame:
    """Per team: this season's record (or last season's if it hasn't started), last-10 record and average margin,
    and current Elo. For the explorer's comparison table."""
    rows = []
    elo = state._elo(season)
    for t, abbr in state.teams.items():
        mine = games[(games.home_id == t) | (games.away_id == t)].sort_values("date")
        margin = pd.Series(mine.margin.to_numpy() * (mine.home_id == t).map({True: 1, False: -1}).to_numpy(), index=mine.index)
        won = margin > 0
        cur = mine.season == season
        rec_season = season if cur.any() else season - 1
        rs = (mine.season == rec_season) & (mine.is_playoff == 0)
        last10 = mine.index[-10:]
        rows.append({"team_id": t, "team": abbr, "record_season": rec_season,
                     "wins": int(won[rs].sum()), "losses": int((~won[rs]).sum()),
                     "last10_w": int(won.loc[last10].sum()), "last10_l": int((~won.loc[last10]).sum()),
                     "last10_margin": float(margin.loc[last10].mean()), "elo": float(elo.get(t, 1505))})
    out = pd.DataFrame(rows)
    out["elo_rank"] = out["elo"].rank(ascending=False, method="min").astype(int)
    return out


def predict_games_v3(model, state, games: pd.DataFrame, season: int, status: dict | None, rosters: dict) -> pd.DataFrame:
    """Upcoming games -> predictions with explanation groups and injury detail. Processes in date order so
    rest/travel account for earlier scheduled games."""
    names = team_names().set_index("team_id")
    rows = []
    for g in games.sort_values(["date", "tipoff_utc"]).itertuples():
        x = state.features(g.home_id, g.away_id, g.date, season, g.is_playoff, g.neutral, status, rosters=rosters)
        pred = model.predict(x, "live").iloc[0]
        why = model.explain(x.iloc[0], "live")
        rows.append({"game_id": g.game_id, "date": g.date, "tipoff_utc": g.tipoff_utc, "label": g.label,
                     "home_id": g.home_id, "away_id": g.away_id,
                     "home_team": names.loc[g.home_id, "team"], "away_team": names.loc[g.away_id, "team"],
                     **pred.to_dict(), "elo_prob": float(x["elo_prob"].iloc[0]),
                     "injury_cost": float(x["injury_cost_live"].iloc[0]),
                     "home_injuries": injury_detail(state, g.home_id, g.date, season, status, rosters.get(g.home_id), g.is_playoff),
                     "away_injuries": injury_detail(state, g.away_id, g.date, season, status, rosters.get(g.away_id), g.is_playoff),
                     "home_absences": absence_total(state, g.home_id, g.date, season, status, rosters.get(g.home_id), g.is_playoff),
                     "away_absences": absence_total(state, g.away_id, g.date, season, status, rosters.get(g.away_id), g.is_playoff),
                     **{f"why_{k}": v for k, v in why.items()}})
        host = names.loc[g.home_id, "team"]
        state.add_scheduled(g.home_id, g.date, host)
        state.add_scheduled(g.away_id, g.date, host)
    return pd.DataFrame(rows)


def export_v3(state, model, cfg: dict, season: int, status: dict | None = None, rosters: dict | None = None) -> None:
    """Everything the v3 app pages read: snapshot, model, metrics, player values, out-of-sample backtest."""
    from nbapred.config import PROCESSED
    from nbapred.data.build import load
    from nbapred.features.matrix_v3 import load_matrix, with_elo
    from nbapred.models import v3 as V3

    joblib.dump(state, APP_DATA / "state_v3.joblib")
    joblib.dump(model, APP_DATA / "model_v3.joblib")
    for name in ("model_meta", "holdout_metrics"):
        src = MODELS / "v3" / f"{name}.json"
        if src.exists():
            shutil.copy(src, APP_DATA / f"{name}_v3.json")

    # Display centring. Predictions only use differences, so the absolute level is arbitrary (O is pinned to
    # average 0 in the fit, D floats). One offset for everything: an average team (league mean of full-strength
    # ratings) shows as 0, and each player is shifted by a fifth of that, so a team's strength is exactly the sum
    # of its players' displayed values (every team's minutes add up to 5 x 48).
    import json
    fits = pd.read_parquet(PROCESSED / "player_ratings.parquet")
    latest = fits[fits["fit_date"] == fits["fit_date"].max()]
    names = team_names()
    today = max(pd.Timestamp.today().normalize(), state.as_of)
    full = [state.team_inputs(t, today, a, 0, state.candidates(t, today, season, status, (rosters or {}).get(t)))
            for t, a in zip(names.team_id, names.team)]
    team_offset = float(np.mean([f["O_full_live"] + f["D_full_live"] for f in full]))
    team_o_offset = float(np.mean([f["O_full_live"] for f in full]))
    player_offset = team_offset / 5
    d_offset = player_offset          # O is already centred at 0, so the whole shift lands on D
    (APP_DATA / "display.json").write_text(json.dumps({"player_offset": player_offset, "team_net_offset": team_offset,
                                                        "team_o_offset": team_o_offset,
                                                        "ratings_fit_date": str(fits["fit_date"].max().date())}))

    # player values: rating x usual minutes, on each player's current team. Also the box-score-only estimate
    # (production) next to the impact rating, and how much recency-weighted court time the rating rests on.
    r = state.ratings.copy()
    r["D"] = r["D"] - d_offset
    extra = latest.set_index("player_id").reindex(r.index)
    r["O_box"] = extra["O_prior"]
    r["D_box"] = extra["D_prior"] - d_offset
    r["net_box"] = r["O_box"] + r["D_box"]
    r["sample_poss"] = extra["poss"].fillna(0.0)
    r["confidence"] = pd.cut(r["sample_poss"], [-1, 1500, 5000, float("inf")], labels=["low", "medium", "high"]).astype(str)
    r["player"] = state.names.reindex(r.index).values
    # current team: the official roster when we have it, otherwise the latest move or appearance
    on_roster = {p: t for t, ps in (rosters or {}).items() for p in ps}
    r["team_id"] = [on_roster.get(p, None if rosters else state.history.team_of.get(p)) for p in r.index]
    r["usual_min"] = [state.history.min_ewm.get(p, 0.0) for p in r.index]
    r["net"] = r["O"] + r["D"]
    r["pts_added"] = r["net"] * r["usual_min"] / 48   # per game, vs an average player in those minutes
    teams = team_names().set_index("team_id")["team"]
    r["team"] = r["team_id"].map(teams)
    r.reset_index().to_parquet(APP_DATA / "player_values.parquet", index=False)

    # out-of-sample backtest: walk-forward seasons + the holdout (model trained on seasons before it)
    m = load_matrix()
    if cfg.get("elo_c", 0):
        m = with_elo(m, load("games"), pd.read_parquet(PROCESSED / "team_projection.parquet"), cfg["elo_c"])
    kw = dict(alpha=cfg.get("alpha", 10.0), trees=cfg.get("trees"), stack=cfg.get("stack"))
    cond = cfg.get("train_condition", "live")
    wf = walk_forward(m, V3.fit_predict(cfg["groups"], cond, **kw))
    train = m[(m.season >= FIRST_TRAIN) & (m.season < HOLDOUT_SEASON)]
    test = m[m.season == HOLDOUT_SEASON]
    ho = test[wf.columns.intersection(test.columns)].join(
        V3.V3Model(cfg["groups"], cond, **kw).fit(train).predict(test, cond))
    keep = ["game_id", "date", "season", "is_playoff", "home_team", "away_team", "home_win", "margin", "total",
            "p_home", "pred_margin", "pred_total"]
    bt = pd.concat([wf, ho])[keep].merge(m[["game_id", "elo_prob"]], on="game_id")
    bt.to_parquet(APP_DATA / "backtest_v3.parquet", index=False)
