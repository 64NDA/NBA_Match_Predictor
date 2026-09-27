"""Player availability.

Each player carries an exponentially weighted rating of Game Score per game (which folds in how many
minutes he usually plays). For every team-game we compare:

  full_strength      top-N ratings among the team's current rotation pool
  available_strength top-N ratings among the players who actually suited up

The gap is "missing_strength" — how much talent is out injured/resting. Historically "who played" is
known from the box score; for live predictions the app lets the caller pass the expected lineup.

Everything is computed in one chronological pass: ratings are read before a game and updated after,
so the features are leak-free by construction.
"""
from collections import defaultdict

import pandas as pd

HALF_LIFE = 20          # player games
TOP_N = 8               # roughly a playoff rotation
POOL_WINDOW = 20        # a player stays in a team's pool for this many team games after his last appearance
ROOKIE_PRIOR = 4.0      # Game Score prior for a player's first appearance
_ALPHA = 1 - 0.5 ** (1 / HALF_LIFE)


def _top(values, n: int | None = None) -> float:
    return float(sum(sorted(values, reverse=True)[:n or TOP_N]))


class PlayerTracker:
    def __init__(self):
        self.rating: dict[int, float] = {}
        self.name: dict[int, str] = {}
        self.team_of: dict[int, int] = {}                             # most recent team per player
        self.last_seen = defaultdict(dict)                            # team -> {player: team game number}
        self.team_games = defaultdict(int)                            # team -> games played so far
        self.rosters: dict[int, set[int]] = {}                        # official rosters, for live use

    def set_rosters(self, rosters: dict[int, list[int]], names: dict[int, str] | None = None) -> None:
        """Override the inferred pools with official rosters (captures trades/signings before a player
        has appeared for his new team, and drops released or retired players)."""
        self.rosters = {t: set(ps) for t, ps in rosters.items()}
        self.team_of.update({p: t for t, ps in rosters.items() for p in ps})
        self.name.update(names or {})

    def pool(self, team_id: int) -> dict[int, float]:
        """Current rotation pool for a team: {player_id: rating}."""
        if team_id in self.rosters:
            return {p: self.rating.get(p, ROOKIE_PRIOR) for p in self.rosters[team_id]}
        n = self.team_games[team_id]
        return {p: self.rating[p] for p, seen in self.last_seen[team_id].items()
                if self.team_of.get(p) == team_id and n - seen <= POOL_WINDOW}

    def strength(self, team_id: int, available: list[int] | None = None) -> tuple[float, float]:
        """(full_strength, available_strength). `available` = player ids expected to play."""
        pool = self.pool(team_id)
        if available is None:
            full = _top(pool.values())
            return full, full
        avail = {p: self.rating.get(p, ROOKIE_PRIOR) for p in available}
        # new arrivals (trades/signings) count towards full strength too
        return _top({**pool, **avail}.values()), _top(avail.values())

    def process_game(self, team_id: int, lines: pd.DataFrame) -> None:
        """Update ratings and pool with one team-game's box score."""
        self.team_games[team_id] += 1
        n = self.team_games[team_id]
        for p, name, gs in zip(lines["player_id"], lines["player"], lines["game_score"]):
            r = self.rating.get(p, ROOKIE_PRIOR)
            self.rating[p] = r + _ALPHA * (gs - r)
            self.name[p] = name
            self.team_of[p] = team_id
            self.last_seen[team_id][p] = n


def player_features(player_games: pd.DataFrame) -> tuple[pd.DataFrame, PlayerTracker]:
    """One row per (game_id, team_id): full_strength, available_strength, missing_strength.

    Also returns the tracker in its end state, for predicting upcoming games.
    """
    pg = player_games.sort_values(["date", "game_id", "team_id"])
    tracker = PlayerTracker()
    rows = []
    # Process a whole date at a time: read features for every game that day before any update,
    # so a same-day game can never see another same-day result.
    for _, day in pg.groupby("date", sort=True):
        feats = [(gid, tid, *tracker.strength(tid, lines["player_id"].tolist()))
                 for (gid, tid), lines in day.groupby(["game_id", "team_id"])]
        for (gid, tid), lines in day.groupby(["game_id", "team_id"]):
            tracker.process_game(tid, lines)
        rows.extend(feats)
    out = pd.DataFrame(rows, columns=["game_id", "team_id", "full_strength", "available_strength"])
    out["missing_strength"] = (out["full_strength"] - out["available_strength"]).clip(lower=0)
    return out, tracker
