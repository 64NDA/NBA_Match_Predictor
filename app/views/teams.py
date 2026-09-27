import pandas as pd
import streamlit as st

from common import AWAY, HOME, app_json, elo_chart, page_header, state, table, teams
from definitions import explain, helptext
from nbapred.config import current_season

page_header("League", "Teams", "Power ratings from current rosters, Elo history and each team's players.")
s, t = state(), teams()
inj = app_json("injury_status")
report = {int(k): v for k, v in inj.get("status", {}).items()}
rosters = {int(k): v for k, v in inj.get("rosters", {}).items()}
today = max(pd.Timestamp.today().normalize(), s.as_of)
season = current_season()
elo = s._elo(season)
disp = app_json("display")
net_off, o_off = disp.get("team_net_offset", 0.0), disp.get("team_o_offset", 0.0)
summary = table("team_summary").set_index("team_id") if not table("team_summary").empty else None


@st.cache_data
def power_table(_state_as_of):
    rows = []
    for r in t.itertuples():
        inp = s.team_inputs(r.team_id, today, r.team, 0,
                            s.candidates(r.team_id, today, season, report, rosters.get(r.team_id)))
        full, tonight = inp["O_full_live"] + inp["D_full_live"], inp["O_live"] + inp["D_live"]
        row = {"Team": r.name, "Full strength": full - net_off, "Absences": tonight - full,
               "Offence": inp["O_full_live"] - o_off, "Defence": inp["D_full_live"] - (net_off - o_off),
               "Elo": elo.get(r.team_id, 1505)}
        if summary is not None and r.team_id in summary.index:
            srow = summary.loc[r.team_id]
            row["Record"] = f"{int(srow.wins)}–{int(srow.losses)} ({int(srow.record_season)}-{str(int(srow.record_season) + 1)[-2:]})"
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("Full strength", ascending=False).reset_index(drop=True)
    out.index += 1
    return out


st.subheader("Power ratings")
st.caption(f"From the current official rosters and player ratings (fit of {disp.get('ratings_fit_date', '?')}). "
           "0 = an average team; the units are points per 100 possessions, about points per game.")
pw = power_table(str(s.as_of))
lim = float(pw["Full strength"].abs().max())
st.dataframe(pw, use_container_width=True, height=420, column_config={
    "Full strength": st.column_config.ProgressColumn(format="%+.1f", min_value=-lim, max_value=lim, help=helptext("full_strength")),
    "Absences": st.column_config.NumberColumn(format="%+.1f", help=helptext("absences")),
    "Offence": st.column_config.NumberColumn(format="%+.1f", help="Points per 100 possessions scored above an average team, at full strength"),
    "Defence": st.column_config.NumberColumn(format="%+.1f", help="Points per 100 possessions kept off the board compared with an average team"),
    "Elo": st.column_config.NumberColumn(format="%.0f", help=helptext("elo")),
    "Record": st.column_config.TextColumn(help="Regular-season wins–losses (last season's until this one starts)")})
explain("team_strength", "full_strength", "absences", "elo", "pool")

# ---- Elo history ------------------------------------------------------------------------------
st.subheader("Elo history")
names = dict(zip(t["team"], t["name"]))
c1, c2 = st.columns(2)
first = c1.selectbox("Team", sorted(names), index=sorted(names).index("BOS"), format_func=names.get)
second = c2.selectbox("Compare with (optional)", [None] + sorted(names), format_func=lambda a: names.get(a, "—"))
seasons = sorted(table("elo_history")["season"].unique())
lo, hi = st.select_slider("Seasons", options=seasons, value=(seasons[0], seasons[-1]),
                          format_func=lambda y: f"{y}-{str(y + 1)[-2:]}")
elo_chart([(first, HOME), (second, AWAY)], seasons=(lo, hi), height=380)

# ---- roster -----------------------------------------------------------------------------------
st.subheader(f"{names[first]}: current roster")
pv = table("player_values")
r = pv[pv["team"] == first].sort_values("pts_added", ascending=False)[
    ["player", "pts_added", "usual_min", "net", "O", "D", "confidence"]]
r.columns = ["Player", "Value (pts/game)", "Usual min", "Impact per 100", "Offence", "Defence", "Confidence"]
st.dataframe(r, hide_index=True, use_container_width=True, column_config={
    "Value (pts/game)": st.column_config.NumberColumn(format="%+.2f", help=helptext("value")),
    "Usual min": st.column_config.NumberColumn(format="%.1f", help=helptext("usual_min")),
    "Impact per 100": st.column_config.NumberColumn(format="%+.1f", help=helptext("impact")),
    "Offence": st.column_config.NumberColumn(format="%+.1f"), "Defence": st.column_config.NumberColumn(format="%+.1f"),
    "Confidence": st.column_config.TextColumn(help=helptext("sample"))})
explain("value", "impact", "usual_min", "sample", title="How the player numbers are calculated")
