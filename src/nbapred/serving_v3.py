"""v3 serving: a snapshot of everything needed to build a feature row for any upcoming game.

Every piece reuses the training code (same EWMs, same trackers, same `matrix_v3.assemble`), and
tests/test_serving_v3.py checks the snapshot reproduces the training matrix exactly.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nbapred.features import matrix_v3 as MX
from nbapred.features.availability import History, candidate_rows, candidate_table
from nbapred.features.elo import MEAN, Elo
from nbapred.features.form import latest_form
from nbapred.features.projection import expected_minutes, team_projection, usual_minutes
from nbapred.models.availability import predict_full
from nbapred.features.schedule import ARENAS, REST_CAP, _haversine_km
from nbapred.features.style import latest_style


@dataclass
class TeamState:
    ratings: pd.DataFrame          # index player_id: O, D (the latest weekly fit before as_of)
    rookie: tuple
    playoff_curve: pd.Series
    heights: pd.Series
    rookie_min: pd.Series          # usual minutes for players with no history, by draft slot
    history: History               # availability history after the last completed game
    availability_model: object
    form: pd.DataFrame
    style: pd.DataFrame
    team_hca: pd.Series
    league: dict
    elo: Elo
    residuals: MX.ResidualTracker
    schedule: dict                 # team_id -> [(date, host abbreviation)]
    teams: pd.Series               # team_id -> abbreviation
    names: pd.Series               # player_id -> name
    as_of: pd.Timestamp

    @classmethod
    def build(cls, games, team_games, player_games, shots, ratings_table, prior_rookie, playoff_curve, heights,
              rookie_min, availability_model, keys, tipoffs, projections, moves, elo_c=0.0, as_of=None,
              moves_until=None) -> "TeamState":
        """Snapshot from games strictly before `as_of` (default: everything). Roster moves are applied up to
        `moves_until` (default `as_of`): for live use, pass today so offseason moves count."""
        if as_of is not None:
            games, team_games, player_games = (d[d["date"] < as_of] for d in (games, team_games, player_games))
        as_of = as_of if as_of is not None else games["date"].max() + pd.Timedelta(days=1)
        fits = ratings_table[ratings_table["fit_date"] < as_of]
        latest = fits[fits["fit_date"] == fits["fit_date"].max()].set_index("player_id")[["O", "D"]]
        _, hist = candidate_table(games, player_games, tipoffs, keys, moves)
        hist.apply_moves(moves, moves_until if moves_until is not None else as_of)
        elo_df, elo = MX.elo_v3(games, projections, elo_c)
        _, resid = MX.residual_features(games, elo_df, elo.home_adv)
        tg = team_games.sort_values("date").assign(host=lambda d: np.where(d["home"] == 1, d["team"], d["opp"]))
        sched = {t: list(zip(g["date"], g["host"])) for t, g in tg.groupby("team_id")}
        return cls(latest, prior_rookie, playoff_curve, heights, rookie_min, hist, availability_model,
                   latest_form(team_games), latest_style(team_games, shots), MX.team_home_edge(team_games, latest=True),
                   MX.latest_league(team_games, MX.team_style(team_games, shots), games), elo, resid, sched,
                   tg.drop_duplicates("team_id", keep="last").set_index("team_id")["team"],
                   player_games.drop_duplicates("player_id", keep="last").set_index("player_id")["player"], as_of)

    # ---- pieces -----------------------------------------------------------------------------------
    def candidates(self, team: int, date, season: int, status: dict | None, roster: list | None = None,
                   box_players: set | None = None) -> pd.DataFrame:
        """Same rows as availability.candidate_table. roster = official roster (authoritative, live);
        box_players = who actually played (tests/backtests only)."""
        return pd.DataFrame(candidate_rows(self.history, team, pd.Timestamp(date), season, status, box_players, roster))

    def _schedule(self, team: int, date: pd.Timestamp, host: str) -> dict:
        past = [(d, h) for d, h in self.schedule.get(team, []) if d < date]
        if not past:
            return {"rest_days": REST_CAP, "b2b": 0, "games_last7": 0, "travel_km": 0.0}
        last_date, last_host = past[-1]
        rest = (date - last_date).days
        travel = 0.0
        if last_host in ARENAS and host in ARENAS:
            travel = float(_haversine_km(np.array([ARENAS[last_host]]), np.array([ARENAS[host]]))[0])
        return {"rest_days": min(rest, REST_CAP), "b2b": int(rest == 1),
                "games_last7": sum(date - pd.Timedelta(days=7) <= d < date for d, _ in past), "travel_km": travel}

    def _elo(self, season: int, projections_now: dict | None = None) -> dict:
        """Ratings at `season`, applying between-season regression if it hasn't started yet."""
        if self.elo.season is not None and season != self.elo.season:
            tgt = projections_now or {}
            return {t: tgt.get(t, MEAN) + self.elo.carryover * (r - tgt.get(t, MEAN)) for t, r in self.elo.ratings.items()}
        return self.elo.ratings

    def team_inputs(self, team: int, date, host: str, is_playoff: int, cands: pd.DataFrame,
                    known: set | None = None, overrides: dict | None = None) -> dict:
        """Everything `assemble` needs for one side. overrides: {player_id: P(plays)} from the app."""
        c = cands.assign(is_playoff=is_playoff)
        p_live = self.availability_model.predict(c)
        p_blind = self.availability_model.predict(c, blind=True)
        c = c.assign(p_full=predict_full(self.availability_model, c))
        if overrides:
            ov = c["player_id"].map(overrides)
            p_live = np.where(ov.notna(), ov, p_live)
        known_p = c["player_id"].isin(known).astype(float).to_numpy() if known is not None else p_live
        out = {}
        for cond, p in (("known", known_p), ("live", p_live), ("blind", p_blind)):
            pr = team_projection(c.assign(p_play=p), self.ratings, self.rookie, self.playoff_curve, bool(is_playoff),
                                 self.rookie_min)
            out.update({f"{k}_{cond}": v for k, v in pr.items()})
        e = expected_minutes(np.asarray(p_live, dtype=float), usual_minutes(c, self.rookie_min))
        hgt = self.heights.reindex(c["player_id"]).fillna(78.0).to_numpy()
        out["height"] = float(hgt @ e / max(e.sum(), 1e-9))
        out.update(self.form.loc[team].to_dict())
        out.update(self.style.loc[team].to_dict())
        out.update(self._schedule(team, pd.Timestamp(date), host))
        out["team_hca"] = float(self.team_hca.get(team, 0.0))
        return out

    def features(self, home_id, away_id, date, season, is_playoff=0, neutral=0, status=None,
                 rosters: dict | None = None, known: dict | None = None, overrides: dict | None = None,
                 season_targets: dict | None = None) -> pd.DataFrame:
        """One matrix_v3-shaped row. status: {player_id: injury status} (None = no report).
        rosters: {team_id: [player_id]} official rosters, authoritative (live). known: {team_id: set(played)}
        for the 'known lineups' condition (tests/backtests). overrides: {player_id: P(plays)}."""
        date = pd.Timestamp(date)
        host = self.teams.loc[home_id]
        sides = {}
        for team in (home_id, away_id):
            cands = self.candidates(team, date, season, status, (rosters or {}).get(team),
                                    (known or {}).get(team) if known else None)
            sides[team] = self.team_inputs(team, date, host, is_playoff, cands,
                                           (known or {}).get(team) if known else None, overrides)
        ratings = self._elo(season, season_targets)
        rh, ra = ratings.get(home_id, MEAN), ratings.get(away_id, MEAN)
        s_res, h_res = self.residuals.read(season, home_id, away_id, is_playoff)
        ctx = {"league": self.league, "neutral": neutral, "is_playoff": is_playoff, "home_team": host,
               "home_elo": rh, "away_elo": ra, "elo_prob": self.elo.win_prob(rh, ra, bool(neutral)),
               "series_resid": s_res, "h2h_resid": h_res}
        return pd.DataFrame([{"home_id": home_id, "away_id": away_id, "date": date, "season": season,
                              "neutral": neutral, **MX.assemble(sides[home_id], sides[away_id], ctx)}])

    def add_scheduled(self, team: int, date: pd.Timestamp, host: str) -> None:
        self.schedule.setdefault(team, []).append((pd.Timestamp(date), host))
        self.schedule[team].sort()
