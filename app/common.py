"""Shared loaders and UI pieces for the Streamlit pages."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so pickled nbapred objects load without `pip install -e .`
APP_DATA = ROOT / "data" / "app"

# Validated categorical slots 1-2 (see dataviz palette): home = blue, away = orange, everywhere.
HOME, AWAY = "#2a78d6", "#eb6834"
GROUP_ORDER = ["Baseline", "Home court", "Team strength (Elo)", "Player ratings", "Injuries", "Recent form",
               "Matchup & style", "Matchup (trees)", "Series / head-to-head", "Rest & travel", "Playoff game",
               "Blend (gradient boosting)", "Blend (neural net)"]
STATUS_OPTIONS = ["Available", "Probable", "Questionable", "Doubtful", "Out"]


@st.cache_resource
def model():
    return joblib.load(APP_DATA / "model_v3.joblib")


@st.cache_resource
def state():
    return joblib.load(APP_DATA / "state_v3.joblib")


@st.cache_data
def table(name: str) -> pd.DataFrame:
    path = APP_DATA / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def app_json(name: str) -> dict:
    p = APP_DATA / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def teams() -> pd.DataFrame:
    return table("teams")


def team_label(abbr: str) -> str:
    t = teams().set_index("team")
    return t.loc[abbr, "name"] if abbr in t.index else abbr


def prob_bar(away: str, home: str, p_home: float):
    """Horizontal split bar: away share on the left (orange), home share on the right (blue)."""
    pa = 1 - p_home
    st.markdown(
        f"""<div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:2px">
          <span><b>{away}</b> {pa:.0%}</span><span>{p_home:.0%} <b>{home}</b></span></div>
        <div style="display:flex;height:10px;gap:2px" role="img" aria-label="{away} {pa:.0%}, {home} {p_home:.0%}">
          <div style="width:{pa*100:.1f}%;background:{AWAY};border-radius:4px 0 0 4px"></div>
          <div style="width:{p_home*100:.1f}%;background:{HOME};border-radius:0 4px 4px 0"></div></div>""",
        unsafe_allow_html=True)


def why_chart(why: pd.Series, home: str, away: str, height: int = 280):
    """Diverging bars: each factor's contribution to the predicted home margin, in points."""
    why = why.reindex([g for g in GROUP_ORDER if g in why.index])
    why = why[why.abs() >= 0.05]
    colors = [HOME if v > 0 else AWAY for v in why.values]
    # values live in the category labels, so negative bars can't collide with them
    labels = [f"{g}  <b>{v:+.1f}</b>" for g, v in why.items()]
    fig = go.Figure(go.Bar(
        x=why.values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
        customdata=why.index, hovertemplate="%{customdata}: %{x:+.1f} pts<extra></extra>"))
    fig.update_layout(
        height=height, margin=dict(l=0, r=10, t=10, b=0), showlegend=False, bargap=0.35,
        xaxis=dict(title=f"← favours {away}   ·   points   ·   favours {home} →", zeroline=True,
                   zerolinewidth=1, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _surname(name: str) -> str:
    words = [w for w in name.split() if w.rstrip(".") not in ("Jr", "Sr", "II", "III", "IV")]
    return words[-1] if words else name


def availability_line(team: str, injuries, total) -> str:
    """'BOS −1.2 from absences: Tatum *Out*, 5% to play (−2.1) · Hauser *Questionable*, 55% (−0.3)'.
    Lists every player with an injury signal (see definitions: absences)."""
    items = list(injuries) if injuries is not None else []
    if not items:
        return f"**{team}**: no injury concerns"
    parts = []
    for i in items:
        why = i["status"] if i["status"] != "not listed" else f"missed last {i['missed']}"
        parts.append(f"{_surname(i['player'])} *{why}*, {i['p_play']:.0%} to play ({i['effect']:+.1f})")
    head = f"**{team}** {total:+.1f} from absences" if total is not None and total == total else f"**{team}**"
    return head + ": " + " · ".join(parts)


def prediction_card(r: pd.Series):
    with st.container(border=True):
        top = st.columns([3, 2])
        top[0].markdown(f"**{team_label(r.away_team)}** @ **{team_label(r.home_team)}**"
                        + (f"  \n<span style='font-size:0.8rem;opacity:0.7'>{r.label}</span>" if r.get("label") else ""),
                        unsafe_allow_html=True)
        top[1].markdown(f"<div style='text-align:right;font-size:1.3rem;font-weight:600'>"
                        f"{r.pred_away_pts:.0f} – {r.pred_home_pts:.0f}</div>"
                        f"<div style='text-align:right;font-size:0.75rem;opacity:0.7'>predicted score (away – home)</div>",
                        unsafe_allow_html=True)
        prob_bar(r.away_team, r.home_team, r.p_home)
        st.caption(availability_line(r.away_team, r.get("away_injuries"), r.get("away_absences")) + "  \n"
                   + availability_line(r.home_team, r.get("home_injuries"), r.get("home_absences")))
        with st.expander("Why?"):
            why = pd.Series({c[4:]: r[c] for c in r.index if c.startswith("why_") and pd.notna(r[c])})
            why_chart(why, r.home_team, r.away_team)
            extra = f" · v2 model: {r.home_team} {r.p_home_v2:.0%}" if pd.notna(r.get("p_home_v2")) else ""
            st.caption(f"Elo alone: {r.home_team} {r.elo_prob:.0%}{extra}. Bars are points of predicted home margin "
                       f"and add up to the prediction ({r.pred_margin:+.1f}).")


def elo_chart(teams_colors: list[tuple[str, str]], seasons: tuple[int, int] | None = None, height: int = 340):
    """Elo over time for up to two teams (weekly means; gaps between seasons stay gaps)."""
    hist = table("elo_history")
    if seasons is None:
        s_all = sorted(hist["season"].unique())
        seasons = (s_all[0], s_all[-1])
    fig = go.Figure()
    for team, color in teams_colors:
        if not team:
            continue
        h = hist[(hist.team == team) & hist.season.between(*seasons)]
        h = h.set_index("date").groupby("season")["elo"].resample("W").mean().dropna().reset_index()
        for i, (_, seg) in enumerate(h.groupby("season")):
            fig.add_scatter(x=seg.date, y=seg.elo, mode="lines", line=dict(color=color, width=2), name=team,
                            legendgroup=team, showlegend=i == 0,
                            hovertemplate=f"{team} %{{x|%d %b %Y}}: %{{y:.0f}}<extra></extra>")
    fig.add_hline(y=1505, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"),
                  annotation_text="league average", annotation_position="bottom right")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                      legend=dict(orientation="h", y=1.1), yaxis_title="Elo",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(gridcolor="rgba(128,128,128,0.15)"), yaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
