"""Refresh data and regenerate everything the Streamlit app reads (data/app/).

    python scripts/daily_update.py            # refresh the season, predict the next 7 days of games
    python scripts/daily_update.py --days 3

Serves v3; v2 predictions are made alongside so the Track record page can compare versions live.
Run it locally (stats.nba.com tends to block cloud IPs), then commit data/app/ and push: Streamlit
Cloud redeploys from the repo.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from nbapred import live
from nbapred.config import APP_DATA, MODELS, PROCESSED, ROOT, current_season
from nbapred.data import fetch, injuries, transactions
from nbapred.data.build import build_all, load, tipoffs
from nbapred.features import matrix_v2 as M2
from nbapred.features.availability import name_keys, resolve_report
from nbapred.features.style import heights
from nbapred.serving_v2 import TeamState as TeamStateV2
from nbapred.serving_v3 import TeamState

sys.stdout.reconfigure(encoding="utf-8")  # player names on the Windows console
parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=7)
args = parser.parse_args()
season = current_season()
APP_DATA.mkdir(parents=True, exist_ok=True)

print(f"1/6 refreshing {season} data")
build_all(refresh_season=season)
transactions.fetch(refresh=True)  # roster moves: who is on which team today
games, tg, pg = load("games"), load("team_games"), load("player_games")
for g in games.loc[games.season == season, "game_id"]:
    fetch.pbp(g)  # cached: only new games are downloaded

print("2/6 rebuilding v3 inputs (incremental)")
subprocess.run([sys.executable, str(ROOT / "scripts" / "build_v3.py"), "--incremental"], check=True)
matrix_v2 = M2.build_and_save()

print("3/6 team state, rosters, injury report")
cfg = json.loads((MODELS / "v3" / "config.json").read_text())
keys = name_keys(fetch.player_bio(), pg)
state = TeamState.build(games, tg, pg, pd.read_parquet(PROCESSED / "team_shots.parquet"),
                        pd.read_parquet(PROCESSED / "player_ratings.parquet"),
                        joblib.load(PROCESSED / "box_prior.joblib").rookie, joblib.load(PROCESSED / "playoff_curve.joblib"),
                        heights(fetch.player_bio()), joblib.load(PROCESSED / "rookie_minutes.joblib"),
                        joblib.load(PROCESSED / "availability_model.joblib"), keys,
                        tipoffs(games), pd.read_parquet(PROCESSED / "team_projection.parquet"), transactions.load(),
                        elo_c=cfg.get("elo_c", 0.0), moves_until=pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
roster_ids, roster_names = live.rosters(season)
state.names = pd.concat([state.names, pd.Series(roster_names)]).groupby(level=0).last()
now_et = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
report = injuries.latest_report(now_et)
source = "official NBA report"
if report is None:
    report, source = injuries.espn_report(), "ESPN"
status = resolve_report(report, {**state.history.team_of, **{p: t for t, ps in roster_ids.items() for p in ps}}, keys)
print(f"   injury source: {source}, {len(status)} players listed")
(APP_DATA / "injury_status.json").write_text(json.dumps(
    {"source": source, "as_of": now_et.isoformat(timespec="minutes"), "status": {str(k): v for k, v in status.items()},
     "rosters": {str(t): [int(p) for p in ps] for t, ps in roster_ids.items()}}))

print("4/6 predicting upcoming games (v3, and v2 for comparison)")
sched = live.schedule(season)
todo = sched[(sched["status"] == 1) & (sched["date"] >= pd.Timestamp.today().normalize())]
if len(todo):
    start = todo["date"].min()  # in the offseason this jumps ahead to opening night
    todo = todo[todo["date"] < start + pd.Timedelta(days=args.days)]
model = joblib.load(MODELS / "v3" / "model.joblib")
upcoming = live.predict_games_v3(model, state, todo, season, status, roster_ids) if len(todo) else pd.DataFrame()
if len(upcoming):
    s2 = TeamStateV2.build(games, tg, pg)
    s2.tracker.set_rosters(roster_ids, roster_names)
    v2 = live.predict_games(joblib.load(MODELS / "v2" / "model.joblib"), s2, todo, season)
    upcoming = upcoming.merge(v2[["game_id", "p_home"]].rename(columns={"p_home": "p_home_v2"}), on="game_id", how="left")
    upcoming["injury_source"] = source
upcoming.to_parquet(APP_DATA / "upcoming.parquet", index=False)
log = live.update_log(upcoming.drop(columns=["home_injuries", "away_injuries"], errors="ignore"), games)

print("5/6 exporting app data")
live.export_v3(state, model, cfg, season, status, roster_ids)
live.export_backtest(matrix_v2)
live.export_elo_history(matrix_v2)
live.team_names().to_parquet(APP_DATA / "teams.parquet", index=False)
live.team_summary(games, state, season).to_parquet(APP_DATA / "team_summary.parquet", index=False)
# every result since 2015-16, for the explorer's head-to-head history
games[["game_id", "date", "season", "is_playoff", "home_team", "away_team", "home_pts", "away_pts"]].to_parquet(
    APP_DATA / "results.parquet", index=False)

print(f"6/6 done: {len(upcoming)} upcoming games predicted, {log['home_win'].notna().sum() if len(log) else 0} logged results")
if len(upcoming):
    print(upcoming[["date", "away_team", "home_team", "p_home", "p_home_v2", "pred_away_pts", "pred_home_pts"]]
          .round(2).head(15).to_string(index=False))
