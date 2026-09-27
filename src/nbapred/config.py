from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
APP_DATA = DATA / "app"   # small artifacts the Streamlit app reads (committed to git)

FIRST_SEASON = 2015          # 2015-16
LAST_COMPLETE_SEASON = 2025  # 2025-26
HOLDOUT_SEASON = 2025        # never used for training or tuning until the final evaluation

SEASON_TYPES = ["Regular Season", "PlayIn", "Playoffs"]


def season_str(start_year: int) -> str:
    """2023 -> '2023-24'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def current_season(today: date | None = None) -> int:
    """Start year of the season in progress (or about to start): Aug-Dec -> this year."""
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def all_seasons() -> list[int]:
    """Every season with data, including the in-progress one (empty responses are skipped)."""
    return list(range(FIRST_SEASON, max(LAST_COMPLETE_SEASON, current_season()) + 1))
