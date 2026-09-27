"""Stints: stretches of a game where the same 10 players are on court, with the points each side scored.

Built from GameRotation (who is on court, in tenths of a second from tip-off) and PlayByPlayV3 (score
timeline). Possessions per stint are estimated from time: duration x the game's possessions per second.
These are the observations for the player-impact (RAPM) regression.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from nbapred.config import PROCESSED, RAW

_CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")
REG_PERIOD, OT_PERIOD = 7200, 3000  # tenths of a second


def _elapsed(period: pd.Series, clock: pd.Series) -> np.ndarray:
    """PBP (period, 'PT10M50.00S' remaining) -> tenths of a second since tip-off."""
    mm = clock.str.extract(_CLOCK).astype(float)
    remaining = (mm[0] * 60 + mm[1]) * 10
    p = period.astype(int)
    start = np.where(p <= 4, (p - 1) * REG_PERIOD, 4 * REG_PERIOD + (p - 5) * OT_PERIOD)
    length = np.where(p <= 4, REG_PERIOD, OT_PERIOD)
    return (start + length - remaining).to_numpy()


def game_stints(rot: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per 5-on-5 stint: start, end (tenths), home_pts, away_pts, h1..h5, a1..a5 (player ids)."""
    s = pbp[pbp["scoreHome"].astype(str) != ""]
    t = _elapsed(s["period"], s["clock"])
    order = np.argsort(t, kind="stable")
    t, sh, sa = t[order], s["scoreHome"].astype(int).to_numpy()[order], s["scoreAway"].astype(int).to_numpy()[order]

    def score_at(x):  # score after every event at or before time x
        i = np.searchsorted(t, x, side="right") - 1
        return (sh[i], sa[i]) if i >= 0 else (0, 0)

    bounds = np.unique(np.concatenate([rot["IN_TIME_REAL"], rot["OUT_TIME_REAL"]]))
    rin, rout, rhome, rpid = (rot[c].to_numpy() for c in ("IN_TIME_REAL", "OUT_TIME_REAL", "home", "PERSON_ID"))
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        on = (rin <= a) & (rout >= b)
        home, away = np.sort(rpid[on & (rhome == 1)]), np.sort(rpid[on & (rhome == 0)])
        if len(home) != 5 or len(away) != 5:
            continue  # data glitch; skip rather than guess
        h0, a0 = score_at(a)
        h1, a1 = score_at(b)
        rows.append((a, b, h1 - h0, a1 - a0, *home, *away))
    cols = ["start", "end", "home_pts", "away_pts"] + [f"h{i}" for i in range(1, 6)] + [f"a{i}" for i in range(1, 6)]
    return pd.DataFrame(rows, columns=cols)


_SUB = re.compile(r"SUB:\s*(?P<inn>.+?)\s+FOR\s+(?P<out>.+)$")


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _name_parts(s: str) -> tuple[str, str, str]:
    """-> (first-name prefix, last name, suffix), accent-folded and lower-case.
    'Dennis Schröder' -> ('dennis', 'schroder', ''); 'Jay. Williams' -> ('jay', 'williams', '');
    'Carter Jr.' -> ('', 'carter', 'jr'); 'L. James' -> ('l', 'james', '')."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    words = [re.sub(r"[^a-z]", "", w) for w in s.split()]
    words = [w for w in words if w]
    suffix = words.pop() if len(words) > 1 and words[-1] in _SUFFIXES else ""
    last = words.pop() if words else ""
    return (words[0] if words else ""), last, suffix


def _last_name(s: str) -> str:
    return _name_parts(s)[1]


def _period_bounds(period: int) -> tuple[int, int]:
    start = (period - 1) * REG_PERIOD if period <= 4 else 4 * REG_PERIOD + (period - 5) * OT_PERIOD
    return start, start + (REG_PERIOD if period <= 4 else OT_PERIOD)


def pbp_stints(pbp: pd.DataFrame, box: pd.DataFrame, home_id: int, away_id: int) -> pd.DataFrame:
    """Rebuild 5-on-5 stints from play-by-play alone (GameRotation is heavily throttled).

    box: that game's player box scores (player_id, player, team_id, min).

    Within a period, substitutions are logged (the outgoing player by id, the incoming one by surname,
    matched against the box score and disambiguated by who is on the bench). Changes *between* periods
    are not logged, so each period's starters are inferred:
      pass 1: anyone who appears in an event before being subbed in, or is subbed out first;
      pass 2: if a team has 4, the 5th is the teammate who is silent all period (no events, no subs)
              and has at least a period's worth of box-score minutes not explained by other periods.
    Periods that still can't be pinned to 5-on-5 are skipped rather than guessed.
    """
    team_of = dict(zip(box["player_id"], box["team_id"]))
    box_min = dict(zip(box["player_id"], box["min"]))
    by_last = {}
    for pid, name, tid in zip(box["player_id"], box["player"], box["team_id"]):
        by_last.setdefault((tid, _last_name(name)), []).append((pid, name))

    # Scorekeepers log some plays late (a 5:23 miss logged after a 2:30 timeout), so order by game clock,
    # with log order breaking ties (keeps FT - sub - FT sequences at the same second intact).
    ev = pbp.assign(t=_elapsed(pbp["period"], pbp["clock"]))
    ev = ev.sort_values(["period", "t", "actionNumber"], kind="stable").reset_index(drop=True)
    ev["is_sub"] = ev["actionType"] == "Substitution"
    # Running score after each event. cummax because scores never go down, and older feeds (2015-17) put
    # "0 - 0" on missed free throws instead of leaving the score blank.
    after_h = pd.to_numeric(ev["scoreHome"].replace("", np.nan)).ffill().fillna(0).cummax().astype(int).to_numpy()
    after_a = pd.to_numeric(ev["scoreAway"].replace("", np.nan)).ffill().fillna(0).cummax().astype(int).to_numpy()
    before_h, before_a = np.concatenate([[0], after_h]), np.concatenate([[0], after_a])  # score before event i

    parts = {pid: _name_parts(name) for pid, name in zip(box["player_id"], box["player"])}

    def incoming(r, on_court):
        """Resolve 'SUB: <in> FOR <out>' to the incoming player's id, or None if it can't be pinned down.
        The NBA disambiguates shared surnames with a first-name prefix ('Jay. Williams') or a suffix
        ('Carter' vs 'Carter Jr.'); the incoming player must also be off the court."""
        m = _SUB.search(str(r.description))
        if not m:
            return None
        prefix, last, suffix = _name_parts(m["inn"])
        cands = [pid for pid, _ in by_last.get((r.teamId, last), []) if pid not in on_court]
        if len(cands) > 1:
            cands = [c for c in cands if parts[c][2] == suffix] or cands
        if len(cands) > 1 and prefix:
            cands = [c for c in cands if parts[c][0].startswith(prefix)] or cands
        return cands[0] if len(cands) == 1 else None

    periods = {p: pe for p, pe in ev.groupby("period", sort=True)}

    # pass 1: starters from events, and who is "silent" (no events, no subs) in each period
    starters, silent = {}, {}
    for p, pe in periods.items():
        subbed_in, st = set(), {home_id: set(), away_id: set()}
        touched = set()
        for r in pe.itertuples():
            if r.is_sub:
                touched.add(r.personId)
                inn = incoming(r, set())
                if r.personId in team_of and r.personId not in subbed_in:
                    st[team_of[r.personId]].add(r.personId)
                if inn is not None:
                    subbed_in.add(inn)
                    touched.add(inn)
            elif r.personId in team_of:
                touched.add(r.personId)
                if r.personId not in subbed_in:
                    st[team_of[r.personId]].add(r.personId)
        starters[p] = st
        silent[p] = {pid for pid in team_of if pid not in touched}

    def run_period(p, st):
        """Stints for one period from its starters; returns (rows, seconds on court per player) or None."""
        pe, (p_start, p_end) = periods[p], _period_bounds(p)
        on = {k: set(v) for k, v in st.items()}
        rows, secs = [], {}
        seg_i, seg_t = pe.index[0], p_start

        def close(until_i, until_t):
            if until_t > seg_t:
                rows.append((seg_t, until_t, before_h[until_i] - before_h[seg_i], before_a[until_i] - before_a[seg_i],
                             *sorted(on[home_id]), *sorted(on[away_id])))
                for pid in on[home_id] | on[away_id]:
                    secs[pid] = secs.get(pid, 0) + (until_t - seg_t) / 10

        for r in pe[pe["is_sub"]].itertuples():
            if r.teamId not in on:
                continue
            close(r.Index, r.t)
            seg_i, seg_t = r.Index, r.t
            inn = incoming(r, on[r.teamId])
            if inn is None or r.personId not in on[r.teamId]:
                return None  # lost track of this lineup
            on[r.teamId].discard(r.personId)
            on[r.teamId].add(inn)
        # close at the first event of the next period, not this period's last row: late corrections are
        # sometimes logged at the end of the game with an early period number
        later = ev.index[ev["period"] > p]
        close(later.min() if len(later) else len(ev), p_end)
        return rows, secs

    ok = lambda st: len(st[home_id]) == 5 and len(st[away_id]) == 5
    results, used = {}, {}
    for p, st in starters.items():
        if ok(st) and (res := run_period(p, st)) is not None:
            results[p] = res
            for pid, sec in res[1].items():
                used[pid] = used.get(pid, 0) + sec

    # pass 2: fill a missing 5th starter with a silent teammate who has enough unexplained minutes
    for p, st in starters.items():
        if p in results:
            continue
        p_len = (_period_bounds(p)[1] - _period_bounds(p)[0]) / 10
        fixed = {k: set(v) for k, v in st.items()}
        for tid in (home_id, away_id):
            if len(fixed[tid]) == 4:
                spare = {pid: box_min[pid] * 60 - used.get(pid, 0) for pid in silent[p]
                         if team_of[pid] == tid and pid not in fixed[tid]}
                best = [pid for pid, sec in sorted(spare.items(), key=lambda kv: -kv[1]) if sec >= 0.9 * p_len]
                if len(best) >= 1:
                    fixed[tid].add(best[0])
        if ok(fixed) and (res := run_period(p, fixed)) is not None:
            results[p] = res
            for pid, sec in res[1].items():
                used[pid] = used.get(pid, 0) + sec

    rows = [row for p in sorted(results) for row in results[p][0]]
    cols = ["start", "end", "home_pts", "away_pts"] + [f"h{i}" for i in range(1, 6)] + [f"a{i}" for i in range(1, 6)]
    return pd.DataFrame(rows, columns=cols)


def build_stints(games: pd.DataFrame, team_games: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """All stints for games whose play-by-play is cached. Adds possessions and game metadata."""
    poss = team_games.groupby("game_id")["poss"].mean()
    boxes = {gid: b for gid, b in player_games[["game_id", "player_id", "player", "team_id", "min"]].groupby("game_id")}
    parts = []
    for g in games.itertuples():
        pp = RAW / "pbp" / f"{g.game_id}.parquet"
        if not pp.exists() or g.game_id not in boxes:
            continue
        pbp = pd.read_parquet(pp)
        if pbp.empty:
            continue
        st = pbp_stints(pbp, boxes[g.game_id], g.home_id, g.away_id)
        if st.empty:
            continue
        game_len = _period_bounds(int(pbp["period"].max()))[1]
        st["poss"] = (st["end"] - st["start"]) / game_len * poss.get(g.game_id, np.nan)
        st["game_id"], st["date"], st["season"], st["is_playoff"] = g.game_id, g.date, g.season, g.is_playoff
        st["home_id"], st["away_id"] = g.home_id, g.away_id
        parts.append(st)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_and_save() -> pd.DataFrame:
    from nbapred.data.build import load
    st = build_stints(load("games"), load("team_games"), load("player_games"))
    st.to_parquet(PROCESSED / "stints.parquet", index=False)
    return st


def load_stints() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "stints.parquet")
