"""Team projection for one game: who plays, for how long, and how good that makes the team.

    expected minutes e_i  = P(plays)_i x usual minutes_i, then scaled so the team adds up to 240,
                            with nobody pushed past min(40, 1.35 x his usual minutes)
    team offence  O_team  = sum O_i x e_i / 48         (points per 100 possessions vs average)
    team defence  D_team  = sum D_i x e_i / 48

Because a star has both a big rating and big minutes, he dominates the sum; when he sits, his minutes go
to worse players and the team loses the difference. That's "star power" without a separate feature.
"""
import numpy as np
import pandas as pd

TEAM_MINUTES = 240.0
MAX_MINUTES = 40.0
MAX_STRETCH = 1.35
NEW_PLAYER_MINUTES = 10.0   # usual minutes for someone with no history and no draft information


def rookie_minutes(draft_number: pd.Series) -> pd.Series:
    """Usual minutes for players with no NBA history, by draft slot: lottery picks play, second-rounders and
    undrafted players mostly don't. (Rough league norms for first-season rotation minutes.)"""
    d = pd.to_numeric(draft_number, errors="coerce")
    return pd.Series(np.select([d <= 5, d <= 14, d <= 30, d <= 60], [26.0, 20.0, 12.0, 6.0], 4.0), index=draft_number.index)


def usual_minutes(cands: pd.DataFrame, rookie_min: pd.Series | None) -> np.ndarray:
    """min_ewm, with players who have never played filled from the draft-slot table."""
    usual = cands["min_ewm"].to_numpy(dtype=float)
    if rookie_min is not None:
        fill = rookie_min.reindex(cands["player_id"]).fillna(NEW_PLAYER_MINUTES).to_numpy()
    else:
        fill = np.full(len(usual), NEW_PLAYER_MINUTES)
    return np.where(usual > 0, usual, fill)


def expected_minutes(p_play: np.ndarray, usual: np.ndarray, playoff_mult: np.ndarray | None = None) -> np.ndarray:
    """Distribute 240 minutes over candidates given their chance of playing and usual minutes.
    Too few expected minutes: scale everyone up (capped). Too many: trim from the bottom of the rotation."""
    base = np.where(usual > 0, usual, NEW_PLAYER_MINUTES)
    if playoff_mult is not None:
        base = base * playoff_mult
    want = p_play * base
    if want.sum() <= 0:
        return want
    if want.sum() >= TEAM_MINUTES:
        # more bodies than minutes: coaches trim the end of the bench, not the stars. Fill in rotation order
        # (usual minutes, highest first) until 240 are used.
        order = np.argsort(-base, kind="stable")
        before = np.concatenate([[0.0], np.cumsum(want[order])[:-1]])
        out = np.empty_like(want)
        out[order] = np.clip(TEAM_MINUTES - before, 0, want[order])
        return out
    cap = p_play * np.minimum(MAX_MINUTES, base * MAX_STRETCH)
    # scale factor s so that sum(min(s * want, cap)) == 240 (bisection; monotone in s)
    lo, hi = 0.0, 10.0
    if np.minimum(hi * want, cap).sum() < TEAM_MINUTES:
        return np.minimum(hi * want, cap)  # not enough bodies to fill 240; give everyone their cap
    for _ in range(50):
        mid = (lo + hi) / 2
        if np.minimum(mid * want, cap).sum() < TEAM_MINUTES:
            lo = mid
        else:
            hi = mid
    return np.minimum(hi * want, cap)


def playoff_multiplier(rank: np.ndarray, curve: pd.Series | None) -> np.ndarray | None:
    """Rotation tightening in the playoffs: minutes multiplier by rank in usual minutes (1 = most)."""
    if curve is None:
        return None
    return curve.reindex(np.clip(rank, 1, curve.index.max())).to_numpy()


INJURY_STATUSES = {"Out", "Doubtful", "Questionable", "Probable"}


def injury_signal(cands: pd.DataFrame) -> np.ndarray:
    """Players with a reason to think they might be hurt: on the injury report, or missed the team's recent game(s)
    this season. Only these are restored when computing full strength."""
    listed = cands["status"].isin(INJURY_STATUSES).to_numpy()
    missing = pd.to_numeric(cands["missed_streak"], errors="coerce").fillna(0).to_numpy() >= 1
    return listed | missing


def full_strength_p(p_play: np.ndarray, p_healthy: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Full strength = tonight's lineup with only the possibly-injured players restored to their healthy chance of
    playing. Everyone else keeps tonight's chance, so the difference is exactly the effect of absences."""
    return np.where(signal, np.maximum(p_play, p_healthy), p_play)


def team_projection(cands: pd.DataFrame, ratings: pd.DataFrame, rookie: tuple,
                    playoff_curve: pd.Series | None = None, is_playoff: bool = False,
                    rookie_min: pd.Series | None = None) -> dict:
    """cands: one team's candidates with columns player_id, p_play, p_healthy, status, missed_streak, min_ewm.
    p_healthy = his chance of playing if he weren't hurt (see full_strength_p).
    ratings: index player_id, columns O, D. Returns team O/D for the expected lineup and at full strength."""
    r = ratings.reindex(cands["player_id"])
    O = r["O"].fillna(rookie[0]).to_numpy()
    D = r["D"].fillna(rookie[1]).to_numpy()
    usual = usual_minutes(cands, rookie_min)
    rank = (-usual).argsort().argsort() + 1
    mult = playoff_multiplier(rank, playoff_curve) if is_playoff else None
    e = expected_minutes(cands["p_play"].to_numpy(), usual, mult)
    p_full = full_strength_p(cands["p_play"].to_numpy(), cands["p_healthy"].to_numpy(), injury_signal(cands))
    full = expected_minutes(p_full, usual, mult)
    return {"O": float(O @ e / 48), "D": float(D @ e / 48),
            "O_full": float(O @ full / 48), "D_full": float(D @ full / 48)}


def fit_playoff_curve(cands: pd.DataFrame, max_rank: int = 12) -> pd.Series:
    """Playoff minutes / usual minutes by rotation rank, from past playoffs (players who played)."""
    c = cands[(cands["is_playoff"] == 1) & (cands["played"] == 1) & (cands["min_ewm"] > 0)].copy()
    c["rank"] = c.groupby(["game_id", "team_id"])["min_ewm"].rank(ascending=False, method="first").clip(upper=max_rank)
    ratio = c.groupby("rank").apply(lambda d: d["minutes"].sum() / d["min_ewm"].sum())
    return ratio.reindex(range(1, max_rank + 1)).ffill()
