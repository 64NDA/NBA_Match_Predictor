"""v3 modelling table: one row per game, built from per-team inputs by ONE shared `assemble` function.

Per-team inputs (all pre-game):
    projection   team offence/defence (O, D) for the expected lineup, and at full strength, under three
                 availability conditions: known (who actually played), live (injury report at T-60 min +
                 history model), blind (history model only)
    form         EWM team stats (v2 form module): pace, four factors for and allowed, net rating
    style        shot profile for and allowed (rim / mid / three)
    schedule     rest, back-to-back, games in 7 days, travel
    height       minutes-weighted height of the expected lineup
    team_hca     the team's own home-vs-road edge (EWM)
Per-game context: Elo (roster-aware season starts), league levels, playoff / neutral flags, playoff-series
and head-to-head residuals vs Elo.

`assemble` turns those into the feature row. The serving path (serving_v3.TeamState) calls the same
function, and tests check both paths agree.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from nbapred.config import PROCESSED
from nbapred.features.elo import MEAN, Elo
from nbapred.features.form import team_form
from nbapred.features.projection import expected_minutes, team_projection, usual_minutes
from nbapred.features.schedule import schedule_features
from nbapred.features.style import PROFILE, team_style

CONDITIONS = ("known", "live", "blind")
FORM_KEEP = ["pace_slow", "net_rating_fast", "net_rating_slow", "margin_slow",
             "efg_pct_slow", "tov_pct_slow", "oreb_pct_slow", "fta_rate_slow",
             "opp_efg_pct_slow", "opp_tov_pct_slow", "opp_oreb_pct_slow", "opp_fta_rate_slow"]
LEAGUE_STATS = ["off_rating", "pace", "efg_pct", "tov_pct", "oreb_pct", "fta_rate"]
ALTITUDE = {"DEN", "UTA"}
ELO_PTS = 28.0          # Elo points per point of margin
SERIES_SHRINK, H2H_SHRINK = 2.0, 4.0
TEAM_HCA_HALF_LIFE = 100


# ---- the shared assembly -----------------------------------------------------------------------------

def assemble(h: dict, a: dict, ctx: dict) -> dict:
    """Feature row for one game from home inputs `h`, away inputs `a` and game context `ctx`."""
    lg = ctx["league"]
    home = 1 - ctx["neutral"]
    row = {"home": home, "is_playoff": ctx["is_playoff"], "playoff_home": ctx["is_playoff"] * home,
           "altitude": int(ctx["home_team"] in ALTITUDE) * home,
           "team_hca": h["team_hca"] * home,
           "diff_elo": (ctx["home_elo"] - ctx["away_elo"]) / ELO_PTS,
           "elo_prob": ctx["elo_prob"], "league_total": lg["total"],
           "series_resid": ctx["series_resid"], "h2h_resid": ctx["h2h_resid"]}

    # players: projected strength and injury cost, per availability condition
    poss = h["pace_slow"] * a["pace_slow"] / lg["pace"]
    for c in CONDITIONS:
        hn, an = h[f"O_{c}"] + h[f"D_{c}"], a[f"O_{c}"] + a[f"D_{c}"]
        row[f"proj_net_{c}"] = hn - an
        row[f"injury_cost_{c}"] = (h[f"O_full_{c}"] + h[f"D_full_{c}"] - hn) - (a[f"O_full_{c}"] + a[f"D_full_{c}"] - an)
        pts_h = poss / 100 * (lg["off_rating"] + h[f"O_{c}"] - a[f"D_{c}"])
        pts_a = poss / 100 * (lg["off_rating"] + a[f"O_{c}"] - h[f"D_{c}"])
        row[f"proj_margin_{c}"], row[f"proj_total_{c}"] = pts_h - pts_a, pts_h + pts_a
    row["height_diff"] = h["height"] - a["height"]

    # form (kept compact: Elo + players carry most of it)
    for s in ("net_rating_fast", "net_rating_slow", "margin_slow"):
        row[f"diff_{s}"] = h[s] - a[s]
    row["sum_pace"] = h["pace_slow"] + a["pace_slow"]
    row["pace_gap"] = abs(h["pace_slow"] - a["pace_slow"])

    # four-factor matchups: my offence vs your defence, minus yours vs mine (league-centred)
    for f in ("efg_pct", "tov_pct", "oreb_pct", "fta_rate"):
        L = lg[f]
        mine = (h[f"{f}_slow"] - L) + (a[f"opp_{f}_slow"] - L)
        yours = (a[f"{f}_slow"] - L) + (h[f"opp_{f}_slow"] - L)
        row[f"ff_{f}"] = mine - yours
    # style clashes: products, so they're large only when a strength meets the matching weakness
    clash = lambda x, y, Lx, Ly: (h[x] - Lx) * (a[y] - Ly) - (a[x] - Lx) * (h[y] - Ly)
    row["clash_three"] = clash("three_rate", "allowed_three_rate", lg["three_rate"], lg["three_rate"])
    row["clash_rim"] = clash("rim_rate", "allowed_rim_pct", lg["rim_rate"], lg["rim_pct"])
    row["clash_reb"] = clash("oreb_pct_slow", "opp_oreb_pct_slow", lg["oreb_pct"], lg["oreb_pct"])
    row["clash_tov"] = clash("tov_pct_slow", "opp_tov_pct_slow", lg["tov_pct"], lg["tov_pct"])
    row["clash_size_reb"] = row["height_diff"] * row["ff_oreb_pct"]

    # rest & travel
    for s in ("rest_days", "b2b", "games_last7", "travel_km"):
        row[f"diff_{s}"] = h[s] - a[s]
    row["h_b2b"], row["a_b2b"], row["a_travel_km"] = h["b2b"], a["b2b"], a["travel_km"]

    # style vectors themselves, for the tree-based matchup model
    for side, d in (("h", h), ("a", a)):
        for s in PROFILE + [f"allowed_{p}" for p in PROFILE]:
            row[f"{side}_{s}"] = d[s]
    return row


# ---- historical inputs ---------------------------------------------------------------------------

LEAGUE_HALF_LIFE_DAYS = 40


def _league_daily(team_games: pd.DataFrame, style: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """EWM of league averages over game dates, *including* each date's games."""
    tg = team_games.merge(style, on=["game_id", "team_id"], how="left")
    daily = tg.groupby("date")[LEAGUE_STATS + ["three_rate", "rim_rate", "rim_pct"]].mean()
    daily["total"] = games.groupby("date")["total"].mean()
    return daily.ewm(halflife=LEAGUE_HALF_LIFE_DAYS).mean()


def league_levels(team_games: pd.DataFrame, style: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """League averages before each game date (shifted one date so same-day games are excluded)."""
    return _league_daily(team_games, style, games).shift(1).bfill()


def latest_league(team_games: pd.DataFrame, style: pd.DataFrame, games: pd.DataFrame) -> dict:
    return _league_daily(team_games, style, games).iloc[-1].to_dict()


def team_home_edge(team_games: pd.DataFrame, latest: bool = False) -> pd.DataFrame:
    """Each team's own home edge: (EWM margin at home - EWM margin on the road) / 2, before each game.
    latest=True returns the value after each team's most recent game (index team_id), for serving."""
    tg = team_games.sort_values(["team_id", "date"])
    shift = 0 if latest else 1
    ew = lambda s: s.shift(shift).ewm(halflife=TEAM_HCA_HALF_LIFE, ignore_na=True, min_periods=5).mean()
    home = tg["margin"].where(tg["home"] == 1).groupby(tg["team_id"]).transform(ew)
    away = tg["margin"].where(tg["home"] == 0).groupby(tg["team_id"]).transform(ew)
    edge = (home.groupby(tg["team_id"]).ffill() - away.groupby(tg["team_id"]).ffill()) / 2
    out = pd.DataFrame({"game_id": tg["game_id"], "team_id": tg["team_id"], "team_hca": edge.fillna(0).values})
    return out.groupby("team_id")["team_hca"].last() if latest else out


def projections(games, cands, ratings, fit_for_game, prior_rookie, playoff_curve, heights, rookie_min) -> pd.DataFrame:
    """Per (game, team): O/D expected and at full strength for each availability condition, plus height."""
    by_fit = {d: r.set_index("player_id")[["O", "D"]] for d, r in ratings.groupby("fit_date")}
    rows = []
    for (gid, team), c in cands.groupby(["game_id", "team_id"], sort=False):
        fit = fit_for_game.get(gid)
        rat = by_fit.get(fit, pd.DataFrame(columns=["O", "D"]))
        is_po = bool(c["is_playoff"].iloc[0])
        row = {"game_id": gid, "team_id": team}
        for cond, p in (("known", c["played"].astype(float)), ("live", c["p_live"]), ("blind", c["p_blind"])):
            pr = team_projection(c.assign(p_play=p.to_numpy()), rat, prior_rookie, playoff_curve, is_po, rookie_min)
            row.update({f"{k}_{cond}": v for k, v in pr.items()})
        # height of the expected (live) lineup
        e = expected_minutes(c["p_live"].to_numpy(), usual_minutes(c, rookie_min))
        hgt = heights.reindex(c["player_id"]).fillna(78.0).to_numpy()
        row["height"] = float(hgt @ e / max(e.sum(), 1e-9))
        rows.append(row)
    return pd.DataFrame(rows)


def elo_v3(games: pd.DataFrame, proj: pd.DataFrame, c: float, **elo_kw) -> tuple[pd.DataFrame, Elo]:
    """Elo whose between-season regression targets the roster projection (live, full strength) at each team's
    first game of the season. c = 0 reproduces v2."""
    targets = {}
    if c:
        first = (games[["game_id", "season", "date", "home_id", "away_id"]]
                 .melt(id_vars=["game_id", "season", "date"], value_name="team_id")
                 .sort_values("date").drop_duplicates(["season", "team_id"]))
        p = first.merge(proj, on=["game_id", "team_id"])
        targets = {(s, t): MEAN + c * ELO_PTS * (o + d) for s, t, o, d in
                   zip(p["season"], p["team_id"], p["O_full_live"], p["D_full_live"])}
    elo = Elo(season_targets=targets, **elo_kw)
    return elo.run(games), elo


class ResidualTracker:
    """Playoff-series and head-to-head residuals (actual margin - Elo expected margin), from earlier games.
    Stored from the lower team id's perspective and flipped to the asking game's home team."""

    def __init__(self):
        self.series, self.h2h = defaultdict(list), defaultdict(list)

    def read(self, season, home_id, away_id, is_playoff) -> tuple[float, float]:
        pair, sign = frozenset((home_id, away_id)), (1 if home_id < away_id else -1)
        s_hist = self.series[(season, pair)] if is_playoff else []
        h_hist = [x for (sn, x) in self.h2h[pair] if sn >= season - 1]
        return (sign * sum(s_hist) / (len(s_hist) + SERIES_SHRINK),
                sign * sum(h_hist) / (len(h_hist) + H2H_SHRINK) if not is_playoff else 0.0)

    def update(self, season, home_id, away_id, is_playoff, resid) -> None:
        pair, sign = frozenset((home_id, away_id)), (1 if home_id < away_id else -1)
        if is_playoff:
            self.series[(season, pair)].append(sign * resid)
        else:
            self.h2h[pair].append((season, sign * resid))


def expected_elo_margin(home_elo, away_elo, neutral, home_adv):
    return (home_elo - away_elo + home_adv * (1 - neutral)) / ELO_PTS


def residual_features(games: pd.DataFrame, elo: pd.DataFrame, home_adv: float) -> tuple[pd.DataFrame, ResidualTracker]:
    """Pre-game series / head-to-head residual for every game (same-day games can't see each other)."""
    g = games.merge(elo, on="game_id").sort_values(["date", "game_id"])
    tr, out = ResidualTracker(), []
    for _, day in g.groupby("date", sort=True):
        for r in day.itertuples():
            s, h = tr.read(r.season, r.home_id, r.away_id, r.is_playoff)
            out.append({"game_id": r.game_id, "series_resid": s, "h2h_resid": h})
        for r in day.itertuples():
            tr.update(r.season, r.home_id, r.away_id, r.is_playoff,
                      r.margin - expected_elo_margin(r.home_elo, r.away_elo, r.neutral, home_adv))
    return pd.DataFrame(out), tr


def build_matrix(games, team_games, cands, ratings, fit_for_game, prior_rookie, playoff_curve,
                 heights, shots, rookie_min=None, elo_c: float = 0.0) -> pd.DataFrame:
    proj = projections(games, cands, ratings, fit_for_game, prior_rookie, playoff_curve, heights, rookie_min)
    proj.to_parquet(PROCESSED / "team_projection.parquet", index=False)  # lets experiments re-run Elo variants
    form = team_form(team_games)
    style = team_style(team_games, shots)
    sched = schedule_features(team_games)
    hca = team_home_edge(team_games)
    team = (proj.merge(form, on=["game_id", "team_id"], how="left").merge(style, on=["game_id", "team_id"], how="left")
            .merge(sched, on=["game_id", "team_id"], how="left").merge(hca, on=["game_id", "team_id"], how="left"))
    team = team.set_index(["game_id", "team_id"])
    league = league_levels(team_games, style, games)
    elo_df, elo = elo_v3(games, proj, elo_c)
    resid, _ = residual_features(games, elo_df, elo.home_adv)
    g = games.merge(elo_df, on="game_id").merge(resid, on="game_id")

    rows = []
    for r in g.itertuples():
        if (r.game_id, r.home_id) not in team.index or (r.game_id, r.away_id) not in team.index:
            continue
        h, a = team.loc[(r.game_id, r.home_id)].to_dict(), team.loc[(r.game_id, r.away_id)].to_dict()
        ctx = {"league": league.loc[r.date].to_dict(), "neutral": r.neutral, "is_playoff": r.is_playoff,
               "home_team": r.home_team, "home_elo": r.home_elo, "away_elo": r.away_elo, "elo_prob": r.elo_prob,
               "series_resid": r.series_resid, "h2h_resid": r.h2h_resid}
        rows.append({"game_id": r.game_id, "date": r.date, "season": r.season, "home_team": r.home_team,
                     "away_team": r.away_team, "home_id": r.home_id, "away_id": r.away_id, "neutral": r.neutral,
                     "margin": r.margin, "total": r.total, "home_win": r.home_win, **assemble(h, a, ctx)})
    out = pd.DataFrame(rows)
    ints = ["season", "neutral", "home_win", "is_playoff", "home", "playoff_home", "altitude"]
    out[ints] = out[ints].astype(int)
    return out


def save(m: pd.DataFrame) -> None:
    m.to_parquet(PROCESSED / "matrix_v3.parquet", index=False)


def load_matrix() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "matrix_v3.parquet")


def with_elo(m: pd.DataFrame, games: pd.DataFrame, proj: pd.DataFrame, c: float) -> pd.DataFrame:
    """Swap in Elo (and the residual features built on it) for a different season-start setting c."""
    elo_df, elo = elo_v3(games, proj, c)
    resid, _ = residual_features(games, elo_df, elo.home_adv)
    x = m.drop(columns=["diff_elo", "elo_prob", "series_resid", "h2h_resid"]).merge(elo_df, on="game_id").merge(resid, on="game_id")
    x["diff_elo"] = (x["home_elo"] - x["away_elo"]) / ELO_PTS
    return x.drop(columns=["home_elo", "away_elo"])
