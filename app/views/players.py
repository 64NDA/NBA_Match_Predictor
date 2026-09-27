import streamlit as st

from common import app_json, page_header, table, teams
from definitions import explain, helptext

page_header("Ratings & value", "Players")
disp = app_json("display")
st.caption(f"Current official rosters. Ratings are from the fit on {disp.get('ratings_fit_date', '?')}. "
           "Sorted by **value**: how many points per game a player adds in the minutes he plays.")

pv = table("player_values")
if pv.empty:
    st.info("Run `scripts/daily_update.py` to generate player values.")
    st.stop()

t = teams().sort_values("name")
c1, c2, c3 = st.columns([2, 1, 1])
team = c1.selectbox("Team", ["All teams"] + list(t["team"]),
                    format_func=lambda a: a if a == "All teams" else t.set_index("team").loc[a, "name"])
min_minutes = c2.slider("Minimum usual minutes", 0, 36, 15)
view = c3.segmented_control("Show", ["Value", "Impact vs box score"], default="Value")

d = pv[pv["team"].notna() & (pv["usual_min"] >= min_minutes)]
if team != "All teams":
    d = d[d["team"] == team]
d = d.sort_values("pts_added", ascending=False)

num = lambda fmt, key=None, **kw: st.column_config.NumberColumn(format=fmt, help=helptext(key) if key else None, **kw)
if view == "Impact vs box score":
    cols = {"player": "Player", "team": "Team", "net": "Impact (net)", "net_box": "Box-score estimate",
            "O": "Impact: offence", "O_box": "Box: offence", "D": "Impact: defence", "D_box": "Box: defence",
            "sample_poss": "Sample", "confidence": "Confidence"}
    cfg = {"Impact (net)": num("%+.1f", "impact"), "Box-score estimate": num("%+.1f", "box"),
           "Impact: offence": num("%+.1f"), "Box: offence": num("%+.1f"), "Impact: defence": num("%+.1f"),
           "Box: defence": num("%+.1f"), "Sample": num("%,.0f", "sample")}
else:
    cols = {"player": "Player", "team": "Team", "pts_added": "Value (pts/game)", "usual_min": "Usual min",
            "net": "Impact per 100", "O": "Offence", "D": "Defence", "confidence": "Confidence"}
    cfg = {"Value (pts/game)": num("%+.2f", "value"), "Usual min": num("%.1f", "usual_min"),
           "Impact per 100": num("%+.1f", "impact"), "Offence": num("%+.1f"), "Defence": num("%+.1f"),
           "Confidence": st.column_config.TextColumn(help=helptext("sample"))}
st.dataframe(d[list(cols)].rename(columns=cols), hide_index=True, use_container_width=True, height=560,
             column_config=cfg)

explain("impact", "per100", "value", "box", "sample", "usual_min")
st.caption("Example: Neemias Queta and Tyrese Maxey have similar *impact per 100 possessions*. Queta's is almost all "
           "defence and Maxey's almost all offence. But Maxey plays about 38 minutes to Queta's 24, so Maxey's "
           "*value per game* is clearly higher. Use **Value** to compare how much players matter.")
