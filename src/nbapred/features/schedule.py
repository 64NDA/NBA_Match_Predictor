"""Rest, back-to-backs, schedule density and travel.

Lower priority than Elo/players, but cheap. Game location is approximated by the home team's city
(neutral-site games like Paris/Mexico City are rare enough to ignore).
"""
import numpy as np
import pandas as pd

# Approximate arena coordinates by team abbreviation (current and historical abbreviations)
ARENAS = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.975), "CHA": (35.225, -80.839),
    "CHI": (41.881, -87.674), "CLE": (41.497, -81.688), "DAL": (32.790, -96.810), "DEN": (39.749, -105.008),
    "DET": (42.341, -83.055), "GSW": (37.768, -122.388), "HOU": (29.751, -95.362), "IND": (39.764, -86.156),
    "LAC": (33.945, -118.341), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051), "MIA": (25.781, -80.188),
    "MIL": (43.045, -87.917), "MIN": (44.980, -93.276), "NOP": (29.949, -90.082), "NYK": (40.751, -73.994),
    "OKC": (35.463, -97.515), "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.438), "TOR": (43.643, -79.379),
    "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}
REST_CAP = 7


def _haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def schedule_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team_id): rest_days, b2b, games_last7, travel_km."""
    tg = team_games.sort_values(["team_id", "date"]).copy()
    host = np.where(tg["home"] == 1, tg["team"], tg["opp"])
    tg["loc"] = pd.Series(host, index=tg.index).map(ARENAS)

    prev_date = tg.groupby("team_id")["date"].shift(1)
    rest = (tg["date"] - prev_date).dt.days
    tg["rest_days"] = rest.clip(upper=REST_CAP).fillna(REST_CAP)
    tg["b2b"] = (rest == 1).astype(int)

    # games in the 7 days before this one (not counting this one)
    counts = tg.set_index("date").groupby("team_id")["game_id"].transform(
        lambda s: s.rolling("7D", closed="left").count())
    tg["games_last7"] = counts.fillna(0).values

    prev_loc = tg.groupby("team_id")["loc"].shift(1)
    both = tg["loc"].notna() & prev_loc.notna()
    tg["travel_km"] = 0.0
    if both.any():
        tg.loc[both, "travel_km"] = _haversine_km(np.array(prev_loc[both].tolist()), np.array(tg.loc[both, "loc"].tolist()))
    return tg[["game_id", "team_id", "rest_days", "b2b", "games_last7", "travel_km"]]
