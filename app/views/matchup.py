import pandas as pd
import streamlit as st

from common import model, prob_bar, state, teams, why_chart
from nbapred.models.predictors import grouped_explanation

st.title("Matchup explorer")
st.caption("Pick any two teams, mark players as out, and see how the prediction changes.")

s, m, t = state(), model(), teams().sort_values("name")
names = dict(zip(t["team_id"], t["name"]))
abbr = dict(zip(t["team_id"], t["team"]))
ids = list(t["team_id"])

c1, c2 = st.columns(2)
away_id = c1.selectbox("Away team", ids, index=ids.index(t.loc[t.team == "BOS", "team_id"].iloc[0]),
                       format_func=names.get)
home_id = c2.selectbox("Home team", ids, index=ids.index(t.loc[t.team == "LAL", "team_id"].iloc[0]),
                       format_func=names.get)
if home_id == away_id:
    st.warning("Pick two different teams.")
    st.stop()

o1, o2, o3 = st.columns(3)
game_date = o1.date_input("Game date", value=max(pd.Timestamp.today(), s.as_of).date())
playoff = o2.toggle("Playoff game")
neutral = o3.toggle("Neutral venue")


def lineup_picker(col, team_id):
    roster = s.roster(team_id)
    labels = {r.player_id: f"{r.player} ({r.rating:.1f})" for r in roster.itertuples()}
    out = col.multiselect(f"{abbr[team_id]} players out", list(labels), format_func=labels.get,
                          help="Rating = recent Game Score per game. The top 8 available players count.")
    return [p for p in roster["player_id"] if p not in out]


l1, l2 = st.columns(2)
away_avail = lineup_picker(l1, away_id)
home_avail = lineup_picker(l2, home_id)

season = pd.Timestamp(game_date).year if pd.Timestamp(game_date).month >= 8 else pd.Timestamp(game_date).year - 1
x = s.features(home_id, away_id, game_date, season, int(playoff), int(neutral), home_avail, away_avail)
p = m.predict(x).iloc[0]

st.divider()
a, b = st.columns([3, 2])
with a:
    st.subheader(f"{names[away_id]} @ {names[home_id]}")
    prob_bar(abbr[away_id], abbr[home_id], p.p_home)
    st.markdown(f"#### {abbr[away_id]} {p.pred_away_pts:.0f} – {p.pred_home_pts:.0f} {abbr[home_id]}")
    st.caption(f"Predicted margin {p.pred_margin:+.1f} for {abbr[home_id]} · predicted total {p.pred_total:.0f} · "
               f"Elo alone: {abbr[home_id]} {x['elo_prob'].iloc[0]:.0%}")
    if neutral:
        st.caption("Neutral venue: Elo drops home advantage, but the margin model has too few neutral games "
                   "to learn it, so its 'Home court' term still applies.")
with b:
    miss_h, miss_a = x["h_missing_strength"].iloc[0], x["a_missing_strength"].iloc[0]
    st.metric(f"{abbr[home_id]} strength available", f"{x['h_available_strength'].iloc[0]:.0f}",
              delta=f"-{miss_h:.1f} missing" if miss_h > 0 else None, delta_color="normal")
    st.metric(f"{abbr[away_id]} strength available", f"{x['a_available_strength'].iloc[0]:.0f}",
              delta=f"-{miss_a:.1f} missing" if miss_a > 0 else None, delta_color="normal")

st.markdown("##### What's driving it")
why_chart(grouped_explanation(m, x.iloc[0]), abbr[home_id], abbr[away_id], height=300)
