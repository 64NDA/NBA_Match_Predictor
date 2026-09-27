import pandas as pd
import streamlit as st

from common import prediction_card, table

st.title("Upcoming games")
up = table("upcoming")
if up.empty:
    st.info("No scheduled games in the next week. Run `scripts/daily_update.py` once the schedule is out.")
    st.stop()

dates = sorted(up["date"].dt.date.unique())
day = st.segmented_control("Date", dates, default=dates[0], format_func=lambda d: d.strftime("%a %d %b"))
games = up[up["date"].dt.date == (day or dates[0])]
st.caption("Win probabilities assume every rostered player is available — the model doesn't read injury "
           "reports. Use the **Matchup explorer** to mark players out and see how the line moves.")

cols = st.columns(2)
for i, (_, r) in enumerate(games.iterrows()):
    with cols[i % 2]:
        prediction_card(r, key=r.game_id)
