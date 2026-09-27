import streamlit as st

from common import app_json, prediction_card, table

st.title("Upcoming games")
up = table("upcoming")
if up.empty:
    st.info("No scheduled games in the next week. Run `scripts/daily_update.py` once the schedule is out.")
    st.stop()

dates = sorted(up["date"].dt.date.unique())
day = st.segmented_control("Date", dates, default=dates[0], format_func=lambda d: d.strftime("%a %d %b"))
games = up[up["date"].dt.date == (day or dates[0])]
inj = app_json("injury_status")
st.caption(f"Injuries: {inj.get('source', 'unknown source')}, as of {inj.get('as_of', '?')} ET. "
           "Each player's chance of playing comes from his status and recent history. Use the **Matchup explorer** "
           "to change anyone's status and see how the prediction moves.")

cols = st.columns(2)
for i, (_, r) in enumerate(games.iterrows()):
    with cols[i % 2]:
        prediction_card(r)
