"""Assemble the modelling table: one row per game, home-vs-away features, targets.

Columns:
  h_<f>, a_<f>   per-team pre-game features for home / away
  diff_<f>       h_<f> - a_<f>
  sum_<f>        h_<f> + a_<f>   (used by the totals model)
  margin, total, home_win        targets
"""
import pandas as pd

from nbapred.config import PROCESSED
from nbapred.data.build import load
from nbapred.features.elo import Elo
from nbapred.features.form import team_form
from nbapred.features.players import player_features
from nbapred.features.schedule import schedule_features

TEAM_FEATURES_SKIP = {"game_id", "team_id", "games_played"}


def _team_level(team_games: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    form = team_form(team_games)
    sched = schedule_features(team_games)
    players, _ = player_features(player_games)
    return form.merge(sched, on=["game_id", "team_id"]).merge(players, on=["game_id", "team_id"], how="left")


LEAGUE_HALF_LIFE_DAYS = 40  # fixed, not derived from the data, so it can't depend on future games


def _league_ewm(games: pd.DataFrame) -> pd.Series:
    """EWM of the average game total, indexed by game date, including that date's games."""
    return games.groupby("date")["total"].mean().ewm(halflife=LEAGUE_HALF_LIFE_DAYS).mean()


def league_scoring(games: pd.DataFrame) -> pd.Series:
    """League-wide scoring level before each game (totals drift ~20 pts over the decade).
    Shifted one date so same-day games are excluded."""
    return games["date"].map(_league_ewm(games).shift(1))


def league_level(games: pd.DataFrame) -> float:
    """Scoring level after the last date in `games` — the value for the next game day."""
    return float(_league_ewm(games).iloc[-1])


def build_matrix(games: pd.DataFrame, team_games: pd.DataFrame, player_games: pd.DataFrame,
                 elo: Elo | None = None) -> pd.DataFrame:
    team = _team_level(team_games, player_games)
    feats = [c for c in team.columns if c not in TEAM_FEATURES_SKIP]

    g = games.copy()
    for side, id_col in (("h", "home_id"), ("a", "away_id")):
        t = team.rename(columns={"team_id": id_col, "games_played": f"{side}_games_played",
                                 **{c: f"{side}_{c}" for c in feats}})
        g = g.merge(t, on=["game_id", id_col], how="left")
    derived = {f"diff_{c}": g[f"h_{c}"] - g[f"a_{c}"] for c in feats}
    derived |= {f"sum_{c}": g[f"h_{c}"] + g[f"a_{c}"] for c in feats}
    g = pd.concat([g, pd.DataFrame(derived)], axis=1)

    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    elo = elo or Elo()
    g = g.merge(elo.run(g), on="game_id")
    g["diff_elo"] = g["home_elo"] - g["away_elo"]
    g["league_total"] = league_scoring(g)
    return g


def build_and_save() -> pd.DataFrame:
    m = build_matrix(load("games"), load("team_games"), load("player_games"))
    m.to_parquet(PROCESSED / "matrix.parquet", index=False)
    return m


def load_matrix() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "matrix.parquet")
