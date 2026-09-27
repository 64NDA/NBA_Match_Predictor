import pandas as pd
import streamlit as st

from common import (AWAY, HOME, STATUS_OPTIONS, app_json, elo_chart, model, page_header, prob_bar, stat_comparison,
                    state, table, teams, why_chart)
from definitions import explain, helptext
from nbapred.config import current_season

page_header("What if", "Matchup explorer", "Any two teams, today or later, with today's rosters, player ratings and "
            "injury report. Change a player's status and everything below updates.")

s, m, t = state(), model(), teams().sort_values("name")
names, abbr = dict(zip(t["team_id"], t["name"])), dict(zip(t["team_id"], t["team"]))
ids = list(t["team_id"])
inj = app_json("injury_status")
report = {int(k): v for k, v in inj.get("status", {}).items()}
rosters = {int(k): v for k, v in inj.get("rosters", {}).items()}
disp = app_json("display")
summary = table("team_summary").set_index("team_id") if not table("team_summary").empty else None

c1, c2 = st.columns(2)
away_id = c1.selectbox("Away team", ids, index=ids.index(t.loc[t.team == "BOS", "team_id"].iloc[0]), format_func=names.get)
home_id = c2.selectbox("Home team", ids, index=ids.index(t.loc[t.team == "LAL", "team_id"].iloc[0]), format_func=names.get)
if home_id == away_id:
    st.warning("Pick two different teams.")
    st.stop()

o1, o2, o3 = st.columns(3)
today = max(pd.Timestamp.today().normalize(), s.as_of)
game_date = pd.Timestamp(o1.date_input("Game date", value=today.date(), min_value=today.date(),
                                       help="Rosters, ratings and injuries are as of today, so only today or later."))
playoff = int(o2.toggle("Playoff game"))
neutral = int(o3.toggle("Neutral venue"))
season = current_season(game_date.date())


# ---- statuses -------------------------------------------------------------------------------------
def status_editor(col, team_id):
    lu = s.lineup(team_id, game_date, season, report, rosters.get(team_id), playoff)
    df = pd.DataFrame({"player_id": lu["player_id"], "Player": lu["player"],
                       "Status": [x if x in STATUS_OPTIONS else "Available" for x in lu["status"]],
                       "Usual min": lu["usual_min"].round(1)})
    col.markdown(f"**{names[team_id]}**")
    ed = col.data_editor(df, hide_index=True, disabled=["Player", "Usual min"], key=f"st_{team_id}_{game_date.date()}",
                         column_order=["Player", "Status", "Usual min"], height=min(38 + 35 * len(df), 400),
                         column_config={"Status": st.column_config.SelectboxColumn(options=STATUS_OPTIONS, required=True,
                                                                                    help=helptext("p_play")),
                                        "Usual min": st.column_config.NumberColumn(help=helptext("usual_min"))})
    return dict(zip(ed["player_id"], ed["Status"]))


with st.expander("Set player statuses (defaults to today's injury report)", expanded=False):
    e1, e2 = st.columns(2)
    status = {**status_editor(e1, away_id), **status_editor(e2, home_id)}

x = s.features(home_id, away_id, game_date, season, playoff, neutral, status, rosters=rosters)
p = m.predict(x, "live").iloc[0]
lineups = {tid: s.lineup(tid, game_date, season, status, rosters.get(tid), playoff) for tid in (away_id, home_id)}

# ---- prediction ---------------------------------------------------------------------------------
st.divider()
st.subheader(f"{names[away_id]} @ {names[home_id]}")
prob_bar(abbr[away_id], abbr[home_id], p.p_home)
st.markdown(f"#### {abbr[away_id]} {p.pred_away_pts:.0f} – {p.pred_home_pts:.0f} {abbr[home_id]}")
st.caption(f"Predicted margin {p.pred_margin:+.1f} for {abbr[home_id]} · predicted total {p.pred_total:.0f}")

# ---- side-by-side comparison -------------------------------------------------------------------
net_off = disp.get("team_net_offset", 0.0)
p_off = disp.get("player_offset", net_off / 5)   # per player; 5 players on court -> sums to net_off
elo_now = s._elo(season)
for lu in lineups.values():   # display centring: a team's strength = the sum of its players' values
    lu["net"] = lu["net"] - p_off
    lu["value_tonight"] = lu["net"] * lu["min_tonight"] / 48
    lu["value_full"] = lu["net"] * lu["min_full"] / 48


def side(tid):
    lu = lineups[tid]
    tonight, full = lu["value_tonight"].sum(), lu["value_full"].sum()
    row = {"Elo": f"{elo_now.get(tid, 1505):.0f}", "Strength tonight": f"{tonight:+.1f}",
           "Full strength": f"{full:+.1f}", "Absences": f"{tonight - full:+.1f}"}
    if summary is not None and tid in summary.index:
        r = summary.loc[tid]
        row["Elo rank"] = f"{int(r.elo_rank)} of 30"
        row[f"Record ({int(r.record_season)}-{str(int(r.record_season) + 1)[-2:]})"] = f"{int(r.wins)}–{int(r.losses)}"
        row["Last 10 (avg margin)"] = f"{int(r.last10_w)}–{int(r.last10_l)} ({r.last10_margin:+.1f})"
    sched = s._schedule(tid, game_date, abbr[home_id])
    row["Days of rest"] = f"{sched['rest_days']:.0f}" + (" (back-to-back)" if sched["b2b"] else "")
    return row


comp = pd.DataFrame({abbr[away_id]: side(away_id), abbr[home_id]: side(home_id)})
elo_p = float(x["elo_prob"].iloc[0])
comp.loc["Win chance: Elo alone"] = [f"{1 - elo_p:.0%}", f"{elo_p:.0%}"]
comp.loc["Win chance: full model"] = [f"{1 - p.p_home:.0%}", f"{p.p_home:.0%}"]
a, b = st.columns([2, 3])
with a:
    st.markdown("##### Side by side")
    stat_comparison(comp, abbr[away_id], abbr[home_id])
    st.caption("Strength = projected net rating (points per 100 possessions; 0 = an average team). "
               "Elo is results-only; strength is who's playing. They can disagree, e.g. after a big trade.")
with b:
    st.markdown("##### Elo over the last two seasons")
    elo_chart([(abbr[away_id], AWAY), (abbr[home_id], HOME)], seasons=(season - 2, season), height=300)

# ---- head-to-head history -----------------------------------------------------------------------
res = table("results")
if not res.empty:
    A, H = abbr[away_id], abbr[home_id]
    h2h = res[((res.home_team == A) & (res.away_team == H)) | ((res.home_team == H) & (res.away_team == A))].copy()
    st.markdown(f"##### Head to head: {A} vs {H}")
    if h2h.empty:
        st.caption("No meetings since 2015-16.")
    else:
        h2h["winner"] = h2h.home_team.where(h2h.home_pts > h2h.away_pts, h2h.away_team)
        h2h["h_margin"] = (h2h.home_pts - h2h.away_pts).where(h2h.home_team == H, h2h.away_pts - h2h.home_pts)
        rec = lambda d: f"{H} {int((d.winner == H).sum())}–{int((d.winner == A).sum())}"
        blocks = []
        for sn in sorted(h2h.season.unique())[-3:][::-1]:
            d = h2h[h2h.season == sn]
            blocks.append(f"**{sn}-{str(sn + 1)[-2:]}:** {rec(d)}" + (" (incl. playoffs)" if d.is_playoff.any() else ""))
        blocks.append(f"**Since 2015-16:** {rec(h2h)}, average margin {h2h.h_margin.mean():+.1f} for {H}")
        po = h2h[h2h.is_playoff == 1]
        if len(po):
            series = po.groupby("season").apply(lambda d: d.winner.value_counts().idxmax())
            blocks.append("**Playoff series:** " + ", ".join(f"{sn}-{str(sn + 1)[-2:]} {w}" for sn, w in series.items()))
        st.markdown(" · ".join(blocks))
        last = h2h.sort_values("date", ascending=False).head(10)
        view = pd.DataFrame({"Date": last.date.dt.date, "Game": last.away_team + " @ " + last.home_team,
                             "Score": last.away_pts.astype(str) + "–" + last.home_pts.astype(str),
                             "Winner": last.winner, "Type": last.is_playoff.map({0: "", 1: "playoffs"})})
        st.dataframe(view, hide_index=True, use_container_width=True, height=min(38 + 35 * len(view), 390))
        # results vs the model's own out-of-sample expectation, from the home side of *this* game
        bt = table("backtest_v3")
        if not bt.empty:
            vm = bt[bt.game_id.isin(h2h.game_id)].merge(h2h[["game_id", "h_margin"]], on="game_id").sort_values("date")
            if len(vm):
                sign = (vm.home_team == H).map({True: 1.0, False: -1.0})
                vm["beat"] = vm.h_margin - sign * vm.pred_margin
                recent = vm.tail(8)
                st.markdown(f"**Against the model's expectations:** over the last {len(recent)} meetings since "
                            f"2021-22, {H} beat the model's predicted margin by **{recent.beat.mean():+.1f}** points "
                            f"per game (individual games: " + ", ".join(f"{b:+.0f}" for b in recent.beat) + ").")
        st.caption("Shown for context only. We tested adding matchup history to the model (past meetings compared "
                   "with the model's own predictions, weighted by how many of today's players took part, and stronger "
                   "when the teams are evenly matched). For a few long-running pairings it pointed the right way, but "
                   "across all teams it didn't improve predictions, so the model doesn't use it.")

# ---- lineups ------------------------------------------------------------------------------------
st.markdown("##### Lineups tonight")
l1, l2 = st.columns(2)
for col, tid in ((l1, away_id), (l2, home_id)):
    lu = lineups[tid]
    show = lu[lu["min_full"] > 0.5].copy()
    show["Chance"] = show["p_play"]
    view = show[["player", "status", "Chance", "min_tonight", "net", "value_tonight"]].rename(columns={
        "player": "Player", "status": "Status", "min_tonight": "Min tonight", "net": "Impact /100",
        "value_tonight": "Value tonight"})
    view["Status"] = view["Status"].replace({"none": "", "no_report": ""})
    col.markdown(f"**{names[tid]}**: strength tonight {lu['value_tonight'].sum():+.1f} "
                 f"(the sum of the Value column)")
    col.dataframe(view, hide_index=True, use_container_width=True, column_config={
        "Chance": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1, help=helptext("p_play")),
        "Min tonight": st.column_config.NumberColumn(format="%.1f", help=helptext("min_tonight")),
        "Impact /100": st.column_config.NumberColumn(format="%+.1f", help=helptext("impact")),
        "Value tonight": st.column_config.NumberColumn(format="%+.2f", help=helptext("value"))})
    eff = s.absence_effects(lu)
    if len(eff):
        col.caption("Absences: " + " · ".join(f"{r.player} ({r.p_play:.0%} to play) {r.effect:+.1f}"
                                              for r in eff.itertuples()))

st.markdown("##### What's driving the prediction")
why_chart(m.explain(x.iloc[0], "live"), abbr[home_id], abbr[away_id], height=300)
explain("team_strength", "full_strength", "absences", "p_play", "min_tonight", "value", "elo", "win_prob")
