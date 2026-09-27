"""Team style profiles, for matchup ("does their style beat ours?") features.

Per team-game from play-by-play shots:
    rim_rate   share of field-goal attempts within 4 ft      rim_pct  FG% on those
    mid_rate   share of 2-pt attempts beyond 4 ft
    three_rate share of attempts that are 3s                 three_pct
and the same numbers *allowed* (what opponents did against this team). Threes are classified by
shotValue (distance is sometimes 0 on heaves); distance only splits 2s into rim vs mid-range.

Profiles are smoothed with the same shift(1)-EWM as recent form, so only earlier games count.
"""
import numpy as np
import pandas as pd

from nbapred.config import PROCESSED, RAW

SHOT_COLS = ["fga", "rim_fga", "rim_fgm", "mid_fga", "three_fga", "three_fgm"]
PROFILE = ["rim_rate", "rim_pct", "mid_rate", "three_rate", "three_pct"]
HALF_LIFE = 20


def game_shots(pbp: pd.DataFrame) -> pd.DataFrame:
    """Shot counts per team for one game."""
    fg = pbp[pbp["isFieldGoal"] == 1]
    three = fg["shotValue"].astype(int) == 3
    rim = ~three & (pd.to_numeric(fg["shotDistance"], errors="coerce").fillna(99) <= 4)
    made = fg["shotResult"] == "Made"
    d = pd.DataFrame({"team_id": fg["teamId"], "fga": 1, "rim_fga": rim.astype(int), "rim_fgm": (rim & made).astype(int),
                      "mid_fga": (~three & ~rim).astype(int), "three_fga": three.astype(int),
                      "three_fgm": (three & made).astype(int)})
    return d.groupby("team_id")[SHOT_COLS].sum().reset_index()


def build_shots(games: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for gid in games["game_id"]:
        f = RAW / "pbp" / f"{gid}.parquet"
        if f.exists():
            p = pd.read_parquet(f, columns=["teamId", "isFieldGoal", "shotValue", "shotDistance", "shotResult"])
            if len(p):
                parts.append(game_shots(p).assign(game_id=gid))
    shots = pd.concat(parts, ignore_index=True)
    shots.to_parquet(PROCESSED / "team_shots.parquet", index=False)
    return shots


def _rates(d: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    return pd.DataFrame({
        f"{prefix}rim_rate": d["rim_fga"] / d["fga"], f"{prefix}rim_pct": d["rim_fgm"] / d["rim_fga"].replace(0, np.nan),
        f"{prefix}mid_rate": d["mid_fga"] / d["fga"], f"{prefix}three_rate": d["three_fga"] / d["fga"],
        f"{prefix}three_pct": d["three_fgm"] / d["three_fga"].replace(0, np.nan)}, index=d.index)


def team_style(team_games: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team_id): pre-game EWM of the team's shot profile, for and allowed."""
    tg = team_games[["game_id", "team_id", "opp_id", "date"]].merge(shots, on=["game_id", "team_id"], how="left")
    opp = shots.rename(columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in SHOT_COLS}})
    tg = tg.merge(opp, on=["game_id", "opp_id"], how="left").sort_values(["date", "game_id"])
    # EWM the *counts* (not the per-game rates) so a 60-shot game counts more than a 40-shot one
    cols = SHOT_COLS + [f"opp_{c}" for c in SHOT_COLS]
    sm = tg.groupby("team_id")[cols].transform(lambda s: s.shift(1).ewm(halflife=HALF_LIFE, min_periods=1).mean())
    own = _rates(sm[SHOT_COLS])
    allowed = _rates(sm[[f"opp_{c}" for c in SHOT_COLS]].set_axis(SHOT_COLS, axis=1), prefix="allowed_")
    return pd.concat([tg[["game_id", "team_id"]], own, allowed], axis=1)


def heights(player_bio: pd.DataFrame) -> pd.Series:
    """player_id -> height in inches ('6-6' -> 78)."""
    h = player_bio.set_index("PERSON_ID")["HEIGHT"].astype(str).str.extract(r"(\d+)-(\d+)").astype(float)
    return (h[0] * 12 + h[1]).dropna()


def latest_style(team_games: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Profile after each team's most recent game (index team_id): what its next game would use."""
    tg = team_games[["game_id", "team_id", "opp_id", "date"]].merge(shots, on=["game_id", "team_id"], how="left")
    opp = shots.rename(columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in SHOT_COLS}})
    tg = tg.merge(opp, on=["game_id", "opp_id"], how="left").sort_values(["date", "game_id"])
    cols = SHOT_COLS + [f"opp_{c}" for c in SHOT_COLS]
    sm = tg.groupby("team_id")[cols].apply(lambda d: d.ewm(halflife=HALF_LIFE, min_periods=1).mean().iloc[-1])
    own = _rates(sm[SHOT_COLS])
    allowed = _rates(sm[[f"opp_{c}" for c in SHOT_COLS]].set_axis(SHOT_COLS, axis=1), prefix="allowed_")
    return pd.concat([own, allowed], axis=1)
