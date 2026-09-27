"""Who might play, and for how long, respecting who is on which team *on that date*.

`candidate_table` makes one chronological pass over every team-game. Before each game day it applies the roster
moves announced before that day (signings, trades, waivers: data/transactions.py), then emits a row per player
who *could* play for the team:
  - its current members: players whose latest move or appearance puts them on the team, and who have played or
    joined within the last POOL_WINDOW team games (traded or waived players drop out on the day of the move;
    new signings are candidates before their first game)
  - anyone on that game's injury report for the team
  - anyone who then appears in the box score without a recorded move (rare; flagged `via_box_only`, since
    historically we only know about them *because* they played)
Every feature is known before tip-off; `played` and `minutes` are the outcomes. The same `History` state is reused
at serving time, where official rosters are authoritative.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nbapred.config import PROCESSED
from nbapred.data import injuries

POOL_WINDOW = 20
MIN_HALF_LIFE = 8          # games, for "usual minutes when he plays"
STATUSES = ["none", "Available", "Probable", "Questionable", "Doubtful", "Out"]  # "none" = not listed
_A = 1 - 0.5 ** (1 / MIN_HALF_LIFE)


@dataclass
class History:
    """Per-player state, updated after each game and by roster moves. Everything read from it is pre-game."""
    team_of: dict = field(default_factory=dict)               # player -> current team (None after a waiver)
    team_game_no: dict = field(default_factory=lambda: defaultdict(int))
    last_game_no: dict = field(default_factory=lambda: defaultdict(dict))  # team -> {player: team game no. last played}
    joined_no: dict = field(default_factory=lambda: defaultdict(dict))     # team -> {player: team game no. when joined}
    played_flags: dict = field(default_factory=lambda: defaultdict(list))  # (team, player) -> [(team game no., 0/1)]
    last_date: dict = field(default_factory=dict)
    last_season: dict = field(default_factory=dict)
    team_last_date: dict = field(default_factory=dict)       # team -> date of its most recent game
    min_ewm: dict = field(default_factory=dict)
    starts_ewm: dict = field(default_factory=dict)            # share of recent games he was top-5 in minutes
    team_season: dict = field(default_factory=dict)           # team -> season of its most recent game
    season_start_no: dict = field(default_factory=dict)       # team -> team game number before its first game this season
    tx_cursor: int = 0                                        # roster moves applied so far

    def apply_moves(self, moves: pd.DataFrame, before: pd.Timestamp) -> None:
        """Apply roster moves dated strictly before `before` that haven't been applied yet (moves sorted by date)."""
        dates, acts = moves["date"].to_numpy(), moves["action"].to_numpy()
        teams, players = moves["team_id"].to_numpy(), moves["player_id"].to_numpy()
        before = np.datetime64(pd.Timestamp(before))
        while self.tx_cursor < len(moves) and dates[self.tx_cursor] < before:
            i, self.tx_cursor = self.tx_cursor, self.tx_cursor + 1
            p, t = int(players[i]), int(teams[i])
            if acts[i] == "join":
                self.team_of[p] = t
                self.joined_no[t][p] = self.team_game_no[t]
            elif self.team_of.get(p) == t:
                self.team_of[p] = None   # waived: a free agent until someone signs him

    def pool(self, team: int) -> set:
        """Current members who have played for, or joined, the team within the last POOL_WINDOW team games."""
        n = self.team_game_no[team]
        recent = {}
        for d in (self.last_game_no[team], self.joined_no[team]):
            for p, k in d.items():
                recent[p] = max(recent.get(p, k), k)
        return {p for p, k in recent.items() if self.team_of.get(p) == team and n - k < POOL_WINDOW}

    def features(self, team: int, player: int, date: pd.Timestamp, season: int) -> dict:
        """Absence features are counted within the current season only: end-of-season rest, tanking or
        shutdowns say nothing about opening night (long injuries that span the summer are on the injury report).
        Before a team's first game of a season they are unknown (NaN)."""
        n = self.team_game_no[team]
        last = self.last_game_no[team].get(player)
        new_team_season = self.team_season.get(team) != season
        start = n if new_team_season else self.season_start_no.get(team, 0)
        season_n = n - start
        joined = self.joined_no[team].get(player)
        since = max(v for v in (last, joined, start) if v is not None)   # games he could have played this stint
        flags = [v for g, v in self.played_flags.get((team, player), []) if g > start][-10:]
        return {
            "season_games": min(season_n, 20),
            "played_last": (int(last == n) if last is not None else 0) if season_n else np.nan,
            "missed_streak": min(n - since, 82) if season_n else np.nan,
            "share_last10": float(np.mean(flags)) if flags else np.nan,
            # days the player has been out *while his team kept playing*: 0 at every season start (the offseason
            # isn't an absence), large for a long-term injury
            "days_since": (min(max((self.team_last_date[team] - self.last_date[player]).days, 0), 365)
                           if player in self.last_date and team in self.team_last_date else 365),
            "min_ewm": self.min_ewm.get(player, 0.0),
            "starts_ewm": self.starts_ewm.get(player, 0.0),
            "new_to_team": int(last is None),
            # last appeared in an earlier season with no roster move since: some such players (unsigned free
            # agents, retirements) never return, and the moves feed can't say which
            "new_season": int(self.last_season.get(player, season) < season),
        }

    def update(self, team: int, box: pd.DataFrame, date: pd.Timestamp, season: int) -> None:
        """box: this team's player lines for the game (player_id, min)."""
        pool_before = self.pool(team) | set(box["player_id"])
        if self.team_season.get(team) != season:
            self.team_season[team] = season
            self.season_start_no[team] = self.team_game_no[team]
        self.team_last_date[team] = date
        self.team_game_no[team] += 1
        n = self.team_game_no[team]
        top5 = set(box.nlargest(5, "min")["player_id"])
        played = set(box["player_id"])
        for p in pool_before:
            if self.team_of.get(p, team) == team or p in played:
                self.played_flags[(team, p)].append((n, int(p in played)))
        for p, m in zip(box["player_id"], box["min"]):
            self.team_of[p] = team
            self.last_game_no[team][p] = n
            self.last_date[p] = date
            self.last_season[p] = season
            self.min_ewm[p] = self.min_ewm.get(p, m) + _A * (m - self.min_ewm.get(p, m))
            s = float(p in top5)
            self.starts_ewm[p] = self.starts_ewm.get(p, s) + _A * (s - self.starts_ewm.get(p, s))


def resolve_report(report: pd.DataFrame, team_of: dict, keys: pd.DataFrame) -> dict:
    """Injury report rows -> {player_id: status}. keys: name_key -> player_id (may repeat); ties are broken
    by the player's current team, since two active players rarely share a name *and* a team."""
    out = {}
    for name_key, status, team in zip(report["name_key"], report["status"], report["team_id"]):
        ids = keys.get(name_key, [])
        if len(ids) > 1:
            ids = [p for p in ids if team_of.get(p) == team] or ids
        if len(ids) == 1:
            out[ids[0]] = status
    return out


def name_keys(player_bio: pd.DataFrame, player_games: pd.DataFrame) -> dict:
    """name_key -> [player_id] from the player index and box-score names."""
    k = defaultdict(set)
    for p, f, l in zip(player_bio["PERSON_ID"], player_bio["PLAYER_FIRST_NAME"], player_bio["PLAYER_LAST_NAME"]):
        k[injuries.name_key(f"{l},{f}")].add(p)
    for p, n in player_games.drop_duplicates("player_id")[["player_id", "player"]].itertuples(index=False):
        k[injuries.name_key(n)].add(p)
    return {a: sorted(b) for a, b in k.items()}


def candidate_rows(hist: History, team: int, date, season: int, status: dict | None,
                   box_players: set | None = None, roster: list | None = None) -> list[dict]:
    """Candidates for one team on one date (shared by training and serving).
    roster: an official roster, authoritative when given (live). box_players: players who then appeared
    (training / backtests), added if missing and flagged `via_box_only`."""
    if roster is not None:
        members = set(roster)
        listed = {p for p in (status or {}) if p in members}
    else:
        members = hist.pool(team)
        listed = {p for p in (status or {}) if hist.team_of.get(p) == team or p in (box_players or set())}
    base = members | listed
    extra = (box_players or set()) - base
    rows = [{"player_id": p, **hist.features(team, p, date, season),
             "status": (status or {}).get(p, "none") if status is not None else "no_report",
             "via_box_only": int(p in extra)} for p in base | extra]
    if roster is not None:
        # an official roster confirms the player is with the team this season, which is exactly the uncertainty
        # `new_season` stands for historically (unsigned free agents, retirements)
        for r in rows:
            r["new_season"] = 0
    return rows


def candidate_table(games: pd.DataFrame, player_games: pd.DataFrame, tipoffs: pd.Series, keys: dict,
                    moves: pd.DataFrame) -> tuple[pd.DataFrame, History]:
    """One row per (game, team, candidate player). Injury reports are read from the local cache only."""
    hist = History()
    box = {k: v for k, v in player_games[["game_id", "team_id", "player_id", "min"]].groupby(["game_id", "team_id"])}
    rows = []
    for day, day_games in games.sort_values(["date", "game_id"]).groupby("date", sort=True):
        hist.apply_moves(moves, day)
        pending = []
        for g in day_games.itertuples():
            tip = tipoffs.get(g.game_id)
            rep = injuries.report_before(tip.to_pydatetime(), offline=True) if tip is not None and tip.year >= 2018 else None
            status = resolve_report(rep, hist.team_of, keys) if rep is not None else None
            for team in (g.home_id, g.away_id):
                b = box.get((g.game_id, team))
                if b is None:
                    continue
                mins = dict(zip(b["player_id"], b["min"]))
                for r in candidate_rows(hist, team, day, g.season, status, set(mins)):
                    rows.append({"game_id": g.game_id, "date": day, "season": g.season, "is_playoff": g.is_playoff,
                                 "team_id": team, **r, "played": int(r["player_id"] in mins),
                                 "minutes": mins.get(r["player_id"], 0.0)})
                pending.append((team, b, g.season))
        for team, b, season in pending:  # update only after the whole day: same-day games can't see each other
            hist.update(team, b, day, season)
    return pd.DataFrame(rows), hist


def save(table: pd.DataFrame) -> None:
    table.to_parquet(PROCESSED / "candidates.parquet", index=False)


def load_candidates() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "candidates.parquet")
