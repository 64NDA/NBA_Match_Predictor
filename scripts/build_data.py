"""Download (cached) and rebuild all processed tables.

    python scripts/build_data.py                 # use cache for everything
    python scripts/build_data.py --refresh 2026  # re-download the in-progress season
"""
import argparse

from nbapred.data.build import build_all, load

parser = argparse.ArgumentParser()
parser.add_argument("--refresh", type=int, default=None, help="season start year to re-download")
args = parser.parse_args()

build_all(refresh_season=args.refresh)
g = load("games")
print(g.groupby(["season", "season_type"]).size().unstack(fill_value=0))
