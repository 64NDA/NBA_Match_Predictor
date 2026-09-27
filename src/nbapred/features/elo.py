"""Margin-of-victory Elo (FiveThirtyEight-style).

Ratings are read *before* each game and updated after it, so the pre-game columns are leak-free.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MEAN = 1505


@dataclass
class Elo:
    k: float = 15.0
    home_adv: float = 50.0  # tuned on walk-forward seasons; home edge has shrunk since ~2019
    carryover: float = 0.75  # fraction of rating kept between seasons (rest regresses to MEAN)
    ratings: dict = field(default_factory=dict)
    season: int | None = None
    # v3: {(season, team_id): Elo target} -- between seasons, regress toward the roster-based projection
    # instead of the league mean. Empty = v2 behaviour.
    season_targets: dict = field(default_factory=dict)

    def win_prob(self, home_rating: float, away_rating: float, neutral: bool = False) -> float:
        diff = home_rating - away_rating + (0 if neutral else self.home_adv)
        return 1 / (1 + 10 ** (-diff / 400))

    def _new_season(self, season: int):
        if self.season is not None and season != self.season:
            self.ratings = {t: self.season_targets.get((season, t), MEAN) + self.carryover
                            * (r - self.season_targets.get((season, t), MEAN)) for t, r in self.ratings.items()}
        self.season = season

    def update(self, home: int, away: int, margin: int, season: int, neutral: bool = False) -> tuple[float, float, float]:
        """Process one game; return (pre_home, pre_away, pre-game home win prob)."""
        self._new_season(season)
        rh, ra = self.ratings.get(home, MEAN), self.ratings.get(away, MEAN)
        p = self.win_prob(rh, ra, neutral)
        won = margin > 0
        winner_diff = (rh + (0 if neutral else self.home_adv) - ra) * (1 if won else -1)
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * winner_diff)
        delta = self.k * mult * (int(won) - p)
        self.ratings[home], self.ratings[away] = rh + delta, ra - delta
        return rh, ra, p

    def run(self, games: pd.DataFrame) -> pd.DataFrame:
        """games: one row per game with game_id, season, home_id, away_id, margin, neutral (sorted by date)."""
        out = np.array([self.update(r.home_id, r.away_id, r.margin, r.season, bool(r.neutral))
                        for r in games[["home_id", "away_id", "margin", "season", "neutral"]].itertuples()])
        return pd.DataFrame({"game_id": games["game_id"].values, "home_elo": out[:, 0],
                             "away_elo": out[:, 1], "elo_prob": out[:, 2]})
