"""Player impact ratings: offence (O) and defence (D), in points per 100 possessions vs average.

Two ingredients:

1. Box-score prior ("BPM-lite"). A player's recent per-100-possession box rates (scoring efficiency,
   shot volume, rebounds, assists, steals, blocks, turnovers, fouls) -> O_prior, D_prior via linear
   weights. The weights are fitted ONCE, on PRIOR_SEASONS only, by regressing multi-season RAPM on the
   box rates, then frozen, so they can't leak into the test seasons.

2. RAPM (regularised adjusted plus-minus) on stints, refit weekly. Each stint gives two rows, one per
   team on offence:
       100 * points / possessions - league_ortg  =  sum O (5 offence)  -  sum D (5 defence)  +  home term
   weighted by possessions x time decay, ridge-penalised towards the prior (we solve for the deviation
   from the prior). Only stints strictly before the fit date are used.

A game on date d uses the latest weekly fit made before d (see `ratings_for_games`).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from nbapred.config import PROCESSED

BOX_STATS = ["pts", "fga", "fta", "fg3a", "ast", "oreb", "dreb", "stl", "blk", "tov", "pf"]
PRIOR_SEASONS = (2015, 2019)       # box-prior weights are fitted on these seasons only
FIT_EVERY_DAYS = 7
PRIOR_RIDGE = 20.0


@dataclass
class ImpactConfig:
    lam: float = 3000.0             # ridge strength, in possession units
    half_life_days: float = 365.0   # stint recency weighting
    window_days: float = 3 * 365.0  # stints older than this are ignored (weight < 1/8)
    box_half_life_days: float = 365.0
    box_pseudo_poss: float = 400.0  # shrink box rates of low-minute players toward replacement level


# ---- player box rates ------------------------------------------------------------------------------

def player_possessions(player_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.Series:
    """Possessions each player was on court for: his share of team minutes x team possessions x 5."""
    team_min = player_games.groupby(["game_id", "team_id"])["min"].transform("sum")
    poss = player_games.merge(team_games[["game_id", "team_id", "poss"]], on=["game_id", "team_id"], how="left")["poss"]
    return (player_games["min"] / team_min * 5 * poss.values).fillna(0)


def box_table(player_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Per player-game box stats plus possessions and 'points above league-average efficiency'."""
    pg = player_games[["player_id", "game_id", "team_id", "date", "season", "min", *BOX_STATS]].copy()
    pg["poss"] = player_possessions(player_games, team_games).values
    shots = pg["fga"] + 0.44 * pg["fta"]
    # league true shooting from the *previous* season: a whole-season average would peek at future games
    ts = pg.groupby("season")["pts"].sum() / (2 * shots.groupby(pg["season"]).sum())
    lg_ts = pg["season"].map(prior_season(ts))
    pg["pts_above_avg"] = pg["pts"] - 2 * lg_ts * shots   # scoring efficiency, volume-weighted
    return pg


RATE_COLS = [*BOX_STATS, "pts_above_avg"]
# Prior inputs: raw points are dropped because pts_above_avg = pts - 2 * league_TS * (fga + 0.44 fta) already
# carries them, and keeping both makes the regression collinear (huge cancelling weights).
PRIOR_COLS = [c for c in RATE_COLS if c != "pts"]


def prior_season(by_season: pd.Series) -> pd.Series:
    """Each season -> the previous season's value (the first season keeps its own). Leak-free league levels."""
    return by_season.shift(1).fillna(by_season.iloc[0])


def league_ortg(stints: pd.DataFrame) -> pd.Series:
    """Season -> league points per 100 possessions, taken from the previous season (leak-free)."""
    ortg = stints.groupby("season").apply(lambda d: 100 * (d.home_pts.sum() + d.away_pts.sum()) / (2 * d.poss.sum()))
    return prior_season(ortg)


def box_rates(box: pd.DataFrame, as_of: pd.Timestamp, cfg: ImpactConfig, repl: pd.Series) -> pd.DataFrame:
    """Recency-weighted per-100 rates for every player, using games strictly before `as_of`.
    Shrunk toward replacement-level rates `repl` with `box_pseudo_poss` pseudo-possessions."""
    past = box[box["date"] < as_of]
    w = 0.5 ** ((as_of - past["date"]).dt.days / cfg.box_half_life_days)
    sums = past[RATE_COLS].mul(w, axis=0).groupby(past["player_id"]).sum()
    poss = (past["poss"] * w).groupby(past["player_id"]).sum()
    k = cfg.box_pseudo_poss
    rates = (sums + k * repl[RATE_COLS] / 100).div(poss + k, axis=0) * 100
    rates["w_poss"] = poss
    return rates


# ---- RAPM ----------------------------------------------------------------------------------------

def _design(stints: pd.DataFrame, players: pd.Index):
    """Home and away 5-man lineups as (n, 5) arrays of column positions into `players`."""
    pos = lambda cols: players.get_indexer(stints[cols].to_numpy().ravel()).reshape(-1, 5)
    return pos([f"h{i}" for i in range(1, 6)]), pos([f"a{i}" for i in range(1, 6)]), len(players)


def fit_rapm(stints: pd.DataFrame, as_of: pd.Timestamp, prior: pd.DataFrame | None, cfg: ImpactConfig,
             lg_ortg: float) -> pd.DataFrame:
    """Ridge on stints before `as_of`, shrunk toward `prior` (index player_id, cols O_prior, D_prior).
    Returns index player_id, columns O, D, poss (weighted possessions played)."""
    st = stints[(stints["date"] < as_of) & (stints["date"] >= as_of - pd.Timedelta(days=cfg.window_days))]
    st = st[st["poss"] > 0]
    players = pd.Index(pd.unique(st[[f"h{i}" for i in range(1, 6)] + [f"a{i}" for i in range(1, 6)]].to_numpy().ravel()))
    H, A, p = _design(st, players)
    w = st["poss"].to_numpy() * 0.5 ** ((as_of - st["date"]).dt.days.to_numpy() / cfg.half_life_days)
    y_h = 100 * st["home_pts"].to_numpy() / st["poss"].to_numpy() - lg_ortg
    y_a = 100 * st["away_pts"].to_numpy() / st["poss"].to_numpy() - lg_ortg

    pri = (prior.reindex(players).fillna(0) if prior is not None
           else pd.DataFrame(0.0, index=players, columns=["O_prior", "D_prior"]))
    o0, d0 = pri["O_prior"].to_numpy(), pri["D_prior"].to_numpy()
    # residual targets after the prior: y - (sum O_prior(off) - sum D_prior(def))
    r_h = y_h - (o0[H].sum(1) - d0[A].sum(1))
    r_a = y_a - (o0[A].sum(1) - d0[H].sum(1))

    # Sparse design for beta = [dO (p), dD (p), home_off, away_off]: +1 on 5 offence O cols, -1 on 5 defence D cols
    k, n = 2 * p + 2, len(st)
    def rows(off, dfn, extra_col):
        cols = np.concatenate([off, dfn + p, np.full((n, 1), extra_col)], axis=1)
        vals = np.concatenate([np.ones((n, 5)), -np.ones((n, 5)), np.ones((n, 1))], axis=1)
        return sparse.csr_matrix((vals.ravel(), cols.ravel(), np.arange(0, 11 * n + 1, 11)), shape=(n, k))
    X = sparse.vstack([rows(H, A, 2 * p), rows(A, H, 2 * p + 1)]).tocsr()
    wr = np.concatenate([w, w])
    XtW = X.T.multiply(wr).tocsr()
    XtWX = (XtW @ X).toarray()
    XtWy = XtW @ np.concatenate([r_h, r_a])
    pen = np.full(k, cfg.lam)
    pen[-2:] = 1.0  # home/away offence terms barely penalised
    beta = np.linalg.solve(XtWX + np.diag(pen), XtWy)
    poss = np.bincount(np.concatenate([H.ravel(), A.ravel()]), weights=np.tile(np.repeat(w, 5), 2), minlength=p)
    O, D = o0 + beta[:p], d0 + beta[p:2 * p]
    # Adding the same constant to every O and D leaves every prediction unchanged (offence minus defence), so
    # fix that free level: the possession-weighted average offence is 0. D is shifted by the same amount.
    shift = np.average(O, weights=np.maximum(poss, 1e-9))
    return pd.DataFrame({"O": O - shift, "D": D - shift, "poss": poss}, index=pd.Index(players, name="player_id"))


# ---- box prior weights ---------------------------------------------------------------------------

@dataclass
class BoxPrior:
    coef_o: pd.Series
    coef_d: pd.Series
    repl: pd.Series           # replacement-level rates (low-minute players)
    rookie: tuple             # (O, D) for players with no history at all
    mean: pd.Series           # input standardisation
    scale: pd.Series

    def predict(self, rates: pd.DataFrame) -> pd.DataFrame:
        X = (rates[PRIOR_COLS] - self.mean) / self.scale
        return pd.DataFrame({"O_prior": X @ self.coef_o[PRIOR_COLS] + self.coef_o["const"],
                             "D_prior": X @ self.coef_d[PRIOR_COLS] + self.coef_d["const"]}, index=rates.index)


def fit_box_prior(stints: pd.DataFrame, box: pd.DataFrame, cfg: ImpactConfig, lg_ortg: float) -> BoxPrior:
    """Regress long-window RAPM (no prior) on box rates over PRIOR_SEASONS; possession-weighted."""
    lo, hi = PRIOR_SEASONS
    st = stints[stints["season"].between(lo, hi)]
    bx = box[box["season"].between(lo, hi)]
    end = st["date"].max() + pd.Timedelta(days=1)
    flat = ImpactConfig(lam=cfg.lam, half_life_days=1e9, window_days=50 * 365)  # no decay, no window
    rapm = fit_rapm(st, end, None, flat, lg_ortg)

    # plain (unweighted by time, unshrunk) career rates over the same window
    sums = bx[RATE_COLS].groupby(bx["player_id"]).sum()
    poss = bx.groupby("player_id")["poss"].sum()
    rates = sums.div(poss, axis=0) * 100
    d = rates.join(rapm, how="inner")
    d = d[poss.reindex(d.index) > 1000]   # enough court time for both sides to be meaningful
    mean, scale = d[PRIOR_COLS].mean(), d[PRIOR_COLS].std()
    X = np.column_stack([((d[PRIOR_COLS] - mean) / scale).to_numpy(), np.ones(len(d))])
    W = np.sqrt(d["poss"].to_numpy())[:, None]
    # standardised inputs, ridge on the slopes only (not the constant); strength in "players' worth" of weight
    ridge = np.diag([PRIOR_RIDGE * W.mean() ** 2] * len(PRIOR_COLS) + [0.0])
    coef = {}
    for side in ("O", "D"):
        coef[side] = pd.Series(np.linalg.solve((X * W).T @ (X * W) + ridge, (X * W).T @ (d[side].to_numpy() * W[:, 0])),
                               index=[*PRIOR_COLS, "const"])
    low = rates.join(rapm, how="inner")[(poss.reindex(rates.join(rapm, how="inner").index) < 500)]
    repl = (sums.loc[low.index].sum() / poss.loc[low.index].sum() * 100)
    rookie = (float(np.average(low["O"], weights=low["poss"])), float(np.average(low["D"], weights=low["poss"])))
    return BoxPrior(coef["O"], coef["D"], repl, rookie, mean, scale)


# ---- weekly ratings over history -----------------------------------------------------------------

def fit_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.normalize(), end.normalize() + pd.Timedelta(days=FIT_EVERY_DAYS), freq=f"{FIT_EVERY_DAYS}D"))


def weekly_ratings(stints: pd.DataFrame, box: pd.DataFrame, prior: BoxPrior, cfg: ImpactConfig,
                   lg_ortg: pd.Series, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Ratings at each fit date. lg_ortg: league offensive rating by season (from games before each date
    is overkill; the season mean moves slowly and only centres y)."""
    out = []
    season_of = lambda d: d.year if d.month >= 8 else d.year - 1
    for d in dates:
        rates = box_rates(box, d, cfg, prior.repl)
        pri = prior.predict(rates)
        lg = lg_ortg.get(season_of(d), lg_ortg.iloc[-1])
        if (stints["date"] < d).any():
            r = fit_rapm(stints, d, pri, cfg, lg)
            r = r.join(pri, how="outer")
            r["O"] = r["O"].fillna(r["O_prior"])
            r["D"] = r["D"].fillna(r["D_prior"])
        else:
            r = pri.assign(O=pri["O_prior"], D=pri["D_prior"], poss=0.0)
        out.append(r.assign(fit_date=d).reset_index())
    return pd.concat(out, ignore_index=True)


def ratings_for_games(ratings: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """game_id -> fit_date to use: the latest fit strictly before the game's date."""
    fits = np.sort(ratings["fit_date"].unique())
    pos = np.searchsorted(fits, games["date"].to_numpy(), side="left") - 1
    return pd.Series(np.where(pos >= 0, fits[np.clip(pos, 0, None)], np.datetime64("NaT")), index=games["game_id"])


def save(ratings: pd.DataFrame, prior: BoxPrior):
    import joblib
    ratings.to_parquet(PROCESSED / "player_ratings.parquet", index=False)
    joblib.dump(prior, PROCESSED / "box_prior.joblib")
