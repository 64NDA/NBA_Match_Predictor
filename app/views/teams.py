import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AWAY, HOME, state, table, teams
from nbapred.config import current_season

st.title("Teams")
s, t = state(), teams()
ratings = s._elo_for(current_season())

# ---- power table ------------------------------------------------------------------------------
rows = []
for r in t.itertuples():
    full, _ = s.tracker.strength(r.team_id)
    f = s.form.loc[r.team_id] if r.team_id in s.form.index else None
    rows.append({"Team": r.name, "team": r.team, "Elo": ratings.get(r.team_id, 1505), "Roster strength": full,
                 "Net rating (last ~20)": f["net_rating_slow"] if f is not None else None})
power = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
power.index += 1
st.subheader("Power ratings")
st.caption("Elo carries over from last season (regressed 25% to the mean). Roster strength = sum of the top 8 "
           "player ratings on the current official roster, so it already reflects offseason moves.")
st.dataframe(power.drop(columns="team"), use_container_width=True, height=420,
             column_config={"Elo": st.column_config.NumberColumn(format="%.0f"),
                            "Roster strength": st.column_config.ProgressColumn(
                                format="%.1f", min_value=0, max_value=float(power["Roster strength"].max())),
                            "Net rating (last ~20)": st.column_config.NumberColumn(format="%+.1f")})

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
for abbr, color in ((first, HOME), (second, AWAY)):
    if not abbr:
        continue
    h = hist[(hist.team == abbr) & hist.season.between(lo, hi)]
    # weekly mean keeps the line readable over a decade; gaps between seasons stay gaps
    h = h.set_index("date").groupby("season")["elo"].resample("W").mean().dropna().reset_index()
    for i, (_, seg) in enumerate(h.groupby("season")):
        fig.add_scatter(x=seg.date, y=seg.elo, mode="lines", line=dict(color=color, width=2), name=abbr,
                        legendgroup=abbr, showlegend=i == 0, hovertemplate=f"{abbr} %{{x|%d %b %Y}}: %{{y:.0f}}<extra></extra>")
fig.add_hline(y=1505, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"),
              annotation_text="league average", annotation_position="bottom right")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                  legend=dict(orientation="h", y=1.08), yaxis_title="Elo",
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  xaxis=dict(gridcolor="rgba(128,128,128,0.15)"), yaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---- roster -----------------------------------------------------------------------------------
st.subheader(f"{names[first]} roster")
tid = t.loc[t.team == first, "team_id"].iloc[0]
r = s.roster(tid)
r.index += 1
st.dataframe(r[["player", "rating"]].rename(columns={"player": "Player", "rating": "Rating (Game Score/game)"}),
             use_container_width=True, column_config={"Rating (Game Score/game)": st.column_config.NumberColumn(format="%.1f")})
