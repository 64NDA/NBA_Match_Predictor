"""Recent-form features: exponentially weighted averages of each team's *previous* games.

This is the v1 exponential-decay idea, vectorised, with a fast and a slow half-life, and with the
opponent's numbers in each game included so defence is captured (e.g. opp_efg_pct = eFG% allowed).
"""
import pandas as pd

# Offence/overall stats from the team's own row
OWN = ["margin", "net_rating", "off_rating", "def_rating", "pace", "efg_pct", "tov_pct", "oreb_pct", "fta_rate", "pts"]
# Defensive four factors: what the opponent did against this team
ALLOWED = ["efg_pct", "tov_pct", "oreb_pct", "fta_rate", "pts"]
HALF_LIVES = {"fast": 5, "slow": 20}
STATS = OWN + [f"opp_{c}" for c in ALLOWED]


def _with_allowed(team_games: pd.DataFrame) -> pd.DataFrame:
    """Attach the opponent's stats from the same game as opp_<stat>."""
    tg = team_games.drop(columns=[f"opp_{c}" for c in ALLOWED], errors="ignore").sort_values(["date", "game_id"])
    opp = tg[["game_id", "team_id", *ALLOWED]].rename(columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in ALLOWED}})
    return tg.merge(opp, on=["game_id", "opp_id"])


def team_form(team_games: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (game_id, team_id) with pre-game form columns named <stat>_<speed>."""
    tg = _with_allowed(team_games)

    out = tg[["game_id", "team_id"]].copy()
    grouped = tg.groupby("team_id")[STATS]
    for speed, hl in HALF_LIVES.items():
        # shift(1): only games strictly before this one contribute
        f = grouped.transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=1).mean())
        out[[f"{c}_{speed}" for c in STATS]] = f.values
    out["games_played"] = tg.groupby(["team_id", "season"]).cumcount().values
    return out


def latest_form(team_games: pd.DataFrame) -> pd.DataFrame:
    """Form as of *after* each team's most recent game — what you'd use to predict its next game."""
    tg = _with_allowed(team_games)
    rows = {}
    for speed, hl in HALF_LIVES.items():
        last = tg.groupby("team_id")[STATS].apply(lambda d: d.ewm(halflife=hl).mean().iloc[-1])
        last.columns = [f"{c}_{speed}" for c in STATS]
        rows[speed] = last
    return pd.concat(rows.values(), axis=1)
