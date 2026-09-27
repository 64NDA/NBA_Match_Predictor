import pandas as pd
import streamlit as st

from common import STATUS_OPTIONS, app_json, model, prob_bar, state, teams, why_chart
from nbapred.config import current_season

st.title("Matchup explorer")
st.caption("Pick any two teams and set each player's status. The prediction updates as you go.")

s, m, t = state(), model(), teams().sort_values("name")
names = dict(zip(t["team_id"], t["name"]))
abbr = dict(zip(t["team_id"], t["team"]))
ids = list(t["team_id"])
inj = app_json("injury_status")
report = {int(k): v for k, v in inj.get("status", {}).items()}
rosters = {int(k): v for k, v in inj.get("rosters", {}).items()}
disp = app_json("display")

c1, c2 = st.columns(2)
away_id = c1.selectbox("Away team", ids, index=ids.index(t.loc[t.team == "BOS", "team_id"].iloc[0]), format_func=names.get)
home_id = c2.selectbox("Home team", ids, index=ids.index(t.loc[t.team == "LAL", "team_id"].iloc[0]), format_func=names.get)
if home_id == away_id:
    st.warning("Pick two different teams.")
    st.stop()

o1, o2, o3 = st.columns(3)
# Rosters, ratings and injuries are as of today, so only today or later makes sense
today = max(pd.Timestamp.today().normalize(), s.as_of)
game_date = pd.Timestamp(o1.date_input("Game date", value=today.date(), min_value=today.date(),
                                       help="Uses today's rosters, player ratings and injury report"))
playoff = int(o2.toggle("Playoff game"))
neutral = int(o3.toggle("Neutral venue"))


def roster_editor(col, team_id):
    """Editable roster: status per player, defaulting to today's injury report."""
    c = s.candidates(team_id, game_date, season, report, rosters.get(team_id))
    r = s.ratings.reindex(c["player_id"])
    df = pd.DataFrame({"player_id": c["player_id"], "Player": c["player_id"].map(s.names).fillna("(new player)"),
                       "Usual min": c["min_ewm"].round(1).to_numpy(),
                       "Rating": (r["O"].fillna(s.rookie[0]) + r["D"].fillna(s.rookie[1])
                                  - disp.get("player_d_offset", 0.0)).round(1).to_numpy(),
                       "Status": [report.get(p, "Available") if report.get(p) in STATUS_OPTIONS else "Available"
                                  for p in c["player_id"]]})
    df = df.sort_values("Usual min", ascending=False).reset_index(drop=True)
    col.markdown(f"**{names[team_id]}**")
    edited = col.data_editor(
        df, hide_index=True, disabled=["Player", "Usual min", "Rating"], key=f"roster_{team_id}_{game_date.date()}",
        column_order=["Player", "Status", "Usual min", "Rating"], height=min(38 + 35 * len(df), 460),
        column_config={"Status": st.column_config.SelectboxColumn(options=STATUS_OPTIONS, required=True),
                       "Rating": st.column_config.NumberColumn(help="Offence + defence impact, points per 100 possessions vs average"),
                       "Usual min": st.column_config.NumberColumn(help="Recent minutes per game when he plays")})
    return dict(zip(edited["player_id"], edited["Status"]))


season = current_season(game_date.date())
l1, l2 = st.columns(2)
status = {**roster_editor(l1, away_id), **roster_editor(l2, home_id)}
x = s.features(home_id, away_id, game_date, season, playoff, neutral, status, rosters=rosters)
p = m.predict(x, "live").iloc[0]

st.divider()
a, b = st.columns([3, 2])
with a:
    st.subheader(f"{names[away_id]} @ {names[home_id]}")
    prob_bar(abbr[away_id], abbr[home_id], p.p_home)
    st.markdown(f"#### {abbr[away_id]} {p.pred_away_pts:.0f} – {p.pred_home_pts:.0f} {abbr[home_id]}")
    st.caption(f"Predicted margin {p.pred_margin:+.1f} for {abbr[home_id]} · predicted total {p.pred_total:.0f} · "
               f"Elo alone: {abbr[home_id]} {x['elo_prob'].iloc[0]:.0%}")
with b:
    for team in (home_id, away_id):
        inp = s.team_inputs(team, game_date, abbr[home_id], playoff,
                            s.candidates(team, game_date, season, status, rosters.get(team)))
        tonight, full = inp["O_live"] + inp["D_live"], inp["O_full_live"] + inp["D_full_live"]
        st.metric(f"{abbr[team]} strength tonight", f"{tonight - disp.get('team_net_offset', 0.0):+.1f}",
                  delta=f"{tonight - full:+.1f} from absences" if abs(tonight - full) >= 0.05 else None,
                  help="Projected net rating of the expected lineup, in points per 100 possessions; 0 = an average team")

st.markdown("##### What's driving it")
why_chart(m.explain(x.iloc[0], "live"), abbr[home_id], abbr[away_id], height=320)
