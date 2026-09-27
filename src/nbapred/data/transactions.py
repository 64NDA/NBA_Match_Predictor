"""Dated roster moves from the NBA's player-movement feed (2015-07 onwards): signings, waivers, trades,
waiver claims. Used to know which team each player belongs to on any date, using only moves announced
before that date.

    joins TEAM_ID : Signing, AwardOnWaivers, Trade (TEAM_ID is the receiving team)
    leaves TEAM_ID: Waive
    no change     : ContractConverted (two-way -> standard, same team)
A move dated d takes effect for games on dates after d.
"""
import json

import pandas as pd
import requests

from nbapred.config import RAW

URL = "https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json"
PATH = RAW / "player_movement.json"
JOIN, LEAVE = {"Signing", "AwardOnWaivers", "Trade"}, {"Waive"}


def fetch(refresh: bool = False) -> None:
    if PATH.exists() and not refresh:
        return
    r = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"})
    r.raise_for_status()
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(r.text, encoding="utf-8")


def load(refresh: bool = False) -> pd.DataFrame:
    """Moves that change a roster: date, action ('join' / 'leave'), team_id, player_id. Sorted by date."""
    fetch(refresh)
    rows = json.loads(PATH.read_text(encoding="utf-8"))["NBA_Player_Movement"]["rows"]
    df = pd.DataFrame(rows)
    df = df[(df["PLAYER_ID"] > 0) & df["Transaction_Type"].isin(JOIN | LEAVE)]
    out = pd.DataFrame({"date": pd.to_datetime(df["TRANSACTION_DATE"].str[:10]),
                        "action": df["Transaction_Type"].map(lambda t: "join" if t in JOIN else "leave"),
                        "team_id": df["TEAM_ID"].astype(int), "player_id": df["PLAYER_ID"].astype(int),
                        "order": df["GroupSort"].str.extract(r"(\d+)$")[0].astype(float).fillna(0)})
    return out.sort_values(["date", "order"], kind="stable").reset_index(drop=True)
