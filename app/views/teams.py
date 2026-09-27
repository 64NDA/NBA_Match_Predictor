import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AWAY, HOME, app_json, state, table, teams
from nbapred.config import current_season

st.title("Teams")
s, t = state(), teams()
inj = app_json("injury_status")
report = {int(k): v for k, v in inj.get("status", {}).items()}
rosters = {int(k): v for k, v in inj.get("rosters", {}).items()}
today = max(pd.Timestamp.today().normalize(), s.as_of)
ratings = s._elo(current_season())
disp = app_json("display")
net_off, o_off = disp.get("team_net_offset", 0.0), disp.get("team_o_offset", 0.0)


@st.cache_data
def power_table(_state_as_of):
    rows = []
    for r in t.itertuples():
        inp = s.team_inputs(r.team_id, today, r.team, 0,
                            s.candidates(r.team_id, today, current_season(), report, rosters.get(r.team_id)))
        full, tonight = inp["O_full_live"] + inp["D_full_live"], inp["O_live"] + inp["D_live"]
        rows.append({"Team": r.name, "Elo": ratings.get(r.team_id, 1505), "Full strength": full - net_off,
                     "Absences": tonight - full, "Offence": inp["O_full_live"] - o_off,
                     "Defence": inp["D_full_live"] - (net_off - o_off)})
    out = pd.DataFrame(rows).sort_values("Full strength", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


st.subheader("Power ratings")
st.caption("**Full strength** = projected net rating in points per 100 possessions, where 0 is an average team, with "
           "the whole roster healthy. It's built from player ratings and minutes on the current official roster. "
           "**Absences** = how the current injury report changes that. It's usually negative; it's positive when the "
           "players out are below the teammates who take their minutes. Elo is results-based and carries over from "
           "last season.")
pw = power_table(str(s.as_of))
lim = float(pw["Full strength"].abs().max())
st.dataframe(pw, use_container_width=True, height=420, column_config={
    "Elo": st.column_config.NumberColumn(format="%.0f"),
    "Full strength": st.column_config.ProgressColumn(format="%+.1f", min_value=-lim, max_value=lim),
    "Absences": st.column_config.NumberColumn(format="%+.1f"),
    "Offence": st.column_config.NumberColumn(format="%+.1f"), "Defence": st.column_config.NumberColumn(format="%+.1f")})

# ---- Elo history ------------------------------------------------------------------------------
st.subheader("Elo history")
hist = table("elo_history")
names = dict(zip(t["team"], t["name"]))
c1, c2 = st.columns(2)
first = c1.selectbox("Team", sorted(names), index=sorted(names).index("BOS"), format_func=names.get)
second = c2.selectbox("Compare with (optional)", [None] + sorted(names), format_func=lambda a: names.get(a, "—"))
seasons = sorted(hist["season"].unique())
lo, hi = st.select_slider("Seasons", options=seasons, value=(seasons[0], seasons[-1]),
                          format_func=lambda y: f"{y}-{str(y + 1)[-2:]}")

fig = go.Figure()
for team, color in ((first, HOME), (second, AWAY)):
    if not team:
        continue
    h = hist[(hist.team == team) & hist.season.between(lo, hi)]
    # weekly mean keeps the line readable over a decade; gaps between seasons stay gaps
    h = h.set_index("date").groupby("season")["elo"].resample("W").mean().dropna().reset_index()
    for i, (_, seg) in enumerate(h.groupby("season")):
        fig.add_scatter(x=seg.date, y=seg.elo, mode="lines", line=dict(color=color, width=2), name=team,
                        legendgroup=team, showlegend=i == 0, hovertemplate=f"{team} %{{x|%d %b %Y}}: %{{y:.0f}}<extra></extra>")
fig.add_hline(y=1505, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"),
              annotation_text="league average", annotation_position="bottom right")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                  legend=dict(orientation="h", y=1.08), yaxis_title="Elo",
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  xaxis=dict(gridcolor="rgba(128,128,128,0.15)"), yaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---- roster -----------------------------------------------------------------------------------
st.subheader(f"{names[first]} roster")
pv = table("player_values")
r = pv[pv["team"] == first].sort_values("pts_added", ascending=False)[["player", "O", "D", "net", "usual_min", "pts_added"]]
r.columns = ["Player", "Offence", "Defence", "Net", "Usual min", "Points added / game"]
st.dataframe(r, hide_index=True, use_container_width=True, column_config={
    c: st.column_config.NumberColumn(format="%+.1f") for c in ["Offence", "Defence", "Net"]} | {
    "Usual min": st.column_config.NumberColumn(format="%.1f"),
    "Points added / game": st.column_config.NumberColumn(format="%+.2f")})
