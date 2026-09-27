import streamlit as st

from common import table, teams

st.title("Players")
st.caption("Each player's impact on the scoreboard while he's on court, in points per 100 possessions compared with "
           "an average player. It's learned from every 10-man stretch of play, so it counts defence and spacing, "
           "not just box-score stats. **Points added** multiplies that by his usual minutes.")

pv = table("player_values")
if pv.empty:
    st.info("Run `scripts/daily_update.py` to generate player values.")
    st.stop()

t = teams().sort_values("name")
c1, c2 = st.columns([2, 1])
team = c1.selectbox("Team", ["All teams"] + list(t["team"]),
                    format_func=lambda a: a if a == "All teams" else t.set_index("team").loc[a, "name"])
min_minutes = c2.slider("Minimum usual minutes", 0, 36, 15)

d = pv[pv["usual_min"] >= min_minutes]
if team != "All teams":
    d = d[d["team"] == team]
d = d.sort_values("pts_added", ascending=False)[["player", "team", "O", "D", "net", "usual_min", "pts_added"]]
d.columns = ["Player", "Team", "Offence", "Defence", "Net", "Usual min", "Points added / game"]
st.dataframe(d, hide_index=True, use_container_width=True, height=620, column_config={
    "Offence": st.column_config.NumberColumn(format="%+.1f"), "Defence": st.column_config.NumberColumn(format="%+.1f"),
    "Net": st.column_config.NumberColumn(format="%+.1f"), "Usual min": st.column_config.NumberColumn(format="%.1f"),
    "Points added / game": st.column_config.NumberColumn(format="%+.2f")})
