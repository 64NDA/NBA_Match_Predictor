"""Serving-time features: a snapshot of every team's state after the last completed game.

TeamState reuses the training feature code (Elo, team_form's EWM, PlayerTracker, schedule rules),
and tests/test_predict.py checks it reproduces the training matrix exactly, so the app can't drift
from what the model was trained on.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nbapred.features.build import league_level
from nbapred.features.elo import Elo
from nbapred.features.form import latest_form
from nbapred.features.players import PlayerTracker, player_features
from nbapred.features.schedule import ARENAS, REST_CAP, _haversine_km


@dataclass
class TeamState:
    elo: Elo
    form: pd.DataFrame                 # index team_id, <stat>_<speed> columns
    tracker: PlayerTracker
    history: dict                      # team_id -> list of (date, host_abbr), played + scheduled
    league_total: float
    teams: pd.DataFrame                # team_id -> abbreviation
    as_of: pd.Timestamp

    @classmethod
    def build(cls, games: pd.DataFrame, team_games: pd.DataFrame, player_games: pd.DataFrame,
              as_of: pd.Timestamp | None = None) -> "TeamState":
        """Snapshot using only games strictly before `as_of` (default: everything)."""
        if as_of is not None:
            games, team_games, player_games = (d[d["date"] < as_of] for d in (games, team_games, player_games))
        games = games.sort_values(["date", "game_id"])
        elo = Elo()
        elo.run(games)
        _, tracker = player_features(player_games)
        tg = team_games.sort_values("date").assign(
            host=lambda d: np.where(d["home"] == 1, d["team"], d["opp"]))
        history = {t: list(zip(g["date"], g["host"])) for t, g in tg.groupby("team_id")}
        total = league_level(games)
        teams = tg.drop_duplicates("team_id", keep="last").set_index("team_id")["team"]
        return cls(elo, latest_form(team_games), tracker, history, total, teams.to_frame(),
                   as_of or games["date"].max() + pd.Timedelta(days=1))

    # ---- per-team pieces ------------------------------------------------------------------------

    def _elo_for(self, season: int) -> dict:
        """Ratings as they'd be at the start of `season` (applies between-season regression)."""
        if self.elo.season is not None and season != self.elo.season:
            return {t: 1505 + self.elo.carryover * (r - 1505) for t, r in self.elo.ratings.items()}
        return self.elo.ratings

    def _schedule(self, team_id: int, date: pd.Timestamp, host: str) -> dict:
        past = [(d, h) for d, h in self.history.get(team_id, []) if d < date]
        if not past:
            return {"rest_days": REST_CAP, "b2b": 0, "games_last7": 0, "travel_km": 0.0}
        last_date, last_host = past[-1]
        rest = (date - last_date).days
        travel = 0.0
        if last_host in ARENAS and host in ARENAS:
            travel = float(_haversine_km(np.array([ARENAS[last_host]]), np.array([ARENAS[host]]))[0])
        return {"rest_days": min(rest, REST_CAP), "b2b": int(rest == 1),
                "games_last7": sum(date - pd.Timedelta(days=7) <= d < date for d, _ in past),
                "travel_km": travel}

    def _team(self, team_id: int, date, host: str, available: list[int] | None) -> dict:
        full, avail = self.tracker.strength(team_id, available)
        return {**self.form.loc[team_id].to_dict(), **self._schedule(team_id, date, host),
                "full_strength": full, "available_strength": avail, "missing_strength": max(full - avail, 0.0)}

    def add_scheduled(self, team_id: int, date: pd.Timestamp, host: str) -> None:
        """Record a scheduled game so later games see correct rest/travel."""
        self.history.setdefault(team_id, []).append((date, host))
        self.history[team_id].sort()

    # ---- public ---------------------------------------------------------------------------------

    def features(self, home_id: int, away_id: int, date, season: int, is_playoff: int = 0, neutral: int = 0,
                 home_available: list[int] | None = None, away_available: list[int] | None = None) -> pd.DataFrame:
        """One matrix-shaped row for a hypothetical/upcoming game. `*_available` = expected lineups
        (player ids); None means the full current roster."""
        date = pd.Timestamp(date)
        host = self.teams.loc[home_id, "team"]
        h = self._team(home_id, date, host, home_available)
        a = self._team(away_id, date, host, away_available)
        row = {"home_id": home_id, "away_id": away_id, "date": date, "season": season,
               "is_playoff": is_playoff, "neutral": neutral, "league_total": self.league_total}
        for c in h:
            row[f"h_{c}"], row[f"a_{c}"] = h[c], a[c]
            row[f"diff_{c}"], row[f"sum_{c}"] = h[c] - a[c], h[c] + a[c]
        ratings = self._elo_for(season)
        rh, ra = ratings.get(home_id, 1505), ratings.get(away_id, 1505)
        row.update(home_elo=rh, away_elo=ra, diff_elo=rh - ra, elo_prob=self.elo.win_prob(rh, ra, bool(neutral)))
        return pd.DataFrame([row])

    def roster(self, team_id: int) -> pd.DataFrame:
        """Current pool with ratings, best first — for the app's lineup picker."""
        pool = self.tracker.pool(team_id)
        return (pd.DataFrame({"player_id": list(pool), "rating": list(pool.values())})
                .assign(player=lambda d: d["player_id"].map(self.tracker.name).fillna("(new player)"))
                .sort_values("rating", ascending=False).reset_index(drop=True))
