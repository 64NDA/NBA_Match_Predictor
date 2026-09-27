"""Refresh data and regenerate everything the Streamlit app reads (data/app/).

    python scripts/daily_update.py            # refresh current season, predict the next 7 days of games
    python scripts/daily_update.py --days 3

Run it locally (stats.nba.com tends to block cloud IPs), then commit data/app/ and push —
Streamlit Cloud redeploys from the repo.
"""
import argparse
import sys

import joblib
import pandas as pd

from nbapred import live
from nbapred.config import APP_DATA, MODELS, current_season
from nbapred.data.build import build_all, load
from nbapred.features.build import build_and_save
from nbapred.predict import TeamState

sys.stdout.reconfigure(encoding="utf-8")  # player names on the Windows console

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=7)
args = parser.parse_args()
season = current_season()
APP_DATA.mkdir(parents=True, exist_ok=True)

print(f"1/5 refreshing {season} data")
build_all(refresh_season=season)
games = load("games")
print("2/5 rebuilding feature matrix")
matrix = build_and_save()

print("3/5 team state + official rosters")
state = TeamState.build(games, load("team_games"), load("player_games"))
state.tracker.set_rosters(*live.rosters(season))

print("4/5 predicting upcoming games")
model = joblib.load(MODELS / "model.joblib")
sched = live.schedule(season)
todo = sched[(sched["status"] == 1) & (sched["date"] >= pd.Timestamp.today().normalize())]
if len(todo):
    start = todo["date"].min()  # in the offseason this jumps ahead to opening night
    todo = todo[todo["date"] < start + pd.Timedelta(days=args.days)]
upcoming = live.predict_games(model, state, todo, season) if len(todo) else pd.DataFrame()
upcoming.to_parquet(APP_DATA / "upcoming.parquet", index=False)
log = live.update_log(upcoming, games)

print("5/5 exporting app data")
# state was mutated by predict_games (scheduled games added); rebuild a clean one for the app
state = TeamState.build(games, load("team_games"), load("player_games"))
state.tracker.set_rosters(*live.rosters(season))
live.save_state(state)
live.export_backtest(matrix)
live.export_elo_history(matrix)
live.team_names().to_parquet(APP_DATA / "teams.parquet", index=False)

print(f"done: {len(upcoming)} upcoming games predicted, {log['home_win'].notna().sum() if len(log) else 0} logged results")
if len(upcoming):
    print(upcoming[["date", "away_team", "home_team", "p_home", "pred_home_pts", "pred_away_pts"]]
          .round(2).head(15).to_string(index=False))
