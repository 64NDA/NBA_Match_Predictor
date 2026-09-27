import streamlit as st

from common import app_json, prediction_card, table
from definitions import explain

st.title("Upcoming games")
up = table("upcoming")
if up.empty:
    st.info("No scheduled games in the next week. Run `scripts/daily_update.py` once the schedule is out.")
    st.stop()

dates = sorted(up["date"].dt.date.unique())
day = st.segmented_control("Date", dates, default=dates[0], format_func=lambda d: d.strftime("%a %d %b"))
games = up[up["date"].dt.date == (day or dates[0])]
inj = app_json("injury_status")
st.caption(f"Rosters: official NBA rosters. Injuries: {inj.get('source', 'unknown source')}, as of "
           f"{inj.get('as_of', '?')} ET. Under each game, every player with an injury concern is listed: he's on "
           "the injury report, or has missed his team's recent games this season. Each shows his chance of playing "
           "and how much his likely absence changes the team's strength, in points, after teammates absorb his "
           "minutes. Everyone else is expected to play his normal minutes.")

cols = st.columns(2)
for i, (_, r) in enumerate(games.iterrows()):
    with cols[i % 2]:
        prediction_card(r)

explain("p_play", "absences", "full_strength", "min_tonight", "win_prob")
