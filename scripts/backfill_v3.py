"""One-off (resumable) backfill of the v3 data.

    python scripts/backfill_v3.py stats      # schedules, player bios, play-by-play (stats.nba.com)
    python scripts/backfill_v3.py rotations  # official rotations (optional; the endpoint throttles heavily)
    python scripts/backfill_v3.py injuries   # official injury report PDFs (NBA CDN; safe to run in parallel)

Everything is cached per game / per report, so re-running skips what's done. Progress goes to stdout.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import pandas as pd

from nbapred.config import RAW
from nbapred.data import fetch, injuries
from nbapred.data.build import load, tipoffs


def per_game(games: pd.DataFrame, part: str) -> None:
    # stats.nba.com throttles bursts, so: steady pace, short timeout, no inner retries, and exponential
    # back-off in this loop until calls succeed again. GameRotation throttles far harder than PlayByPlayV3,
    # so lineups are rebuilt from play-by-play substitutions and rotations are only an optional extra.
    fetch.REQUEST_GAP_S, fetch.TIMEOUT_S, fetch.RETRIES = (0.7 if part == "stats" else 2.0), 15, 1
    fetch.player_bio()
    kind, get = ("pbp", fetch.pbp) if part == "stats" else ("rotations", fetch.rotation)
    todo = [g for g in games["game_id"][::-1] if not (RAW / kind / f"{g}.parquet").exists()]  # newest first
    print(f"{len(todo)} of {len(games)} games to fetch", flush=True)
    t0, failed, streak = time.time(), [], 0
    for i, g in enumerate(todo, 1):
        try:
            get(g)
            streak = 0
        except Exception as e:  # keep going; failures are retried on the next run
            failed.append(g)
            streak += 1
            pause = min(60 * 2 ** (streak - 1), 900)
            print(f"  FAILED {g}: {type(e).__name__}; backing off {pause}s", flush=True)
            time.sleep(pause)
        if i % 100 == 0:
            rate = (time.time() - t0) / i
            print(f"{i}/{len(todo)}  {rate:.2f}s/game  eta {(len(todo) - i) * rate / 3600:.1f}h  failed {len(failed)}", flush=True)
    print(f"done. failed: {len(failed)}")


def reports(games: pd.DataFrame) -> None:
    # PDFs come from the NBA's CDN (not stats.nba.com) and parsing is CPU-bound, so use a few processes
    tips = tipoffs(games)
    tips = tips[tips >= datetime(2018, 12, 1)].sort_values()   # official reports start around Dec 2018
    todo = [t.to_pydatetime() for t in tips.drop_duplicates()]
    print(f"{len(todo)} distinct tip-off times from {len(tips)} games", flush=True)
    found = 0
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, df in enumerate(pool.map(injuries.report_before, todo, chunksize=20), 1):
            found += df is not None
            if i % 200 == 0:
                print(f"{i}/{len(todo)}  found {found}", flush=True)
    print(f"done. reports found for {found}/{len(todo)} tip-off times")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    part = sys.argv[1]
    games = load("games").sort_values(["date", "game_id"])
    if part in ("stats", "rotations"):
        per_game(games, part)
    elif part == "injuries":
        reports(games)
