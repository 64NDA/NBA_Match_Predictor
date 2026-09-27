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


def injury_badges(team: str, injuries) -> str:
    """Compact text list: 'LAL: Dončić 5% (Out) · Reaves 55% (Questionable)'."""
    items = list(injuries) if injuries is not None else []
    if not items:
        return ""
    surname = lambda n: [w for w in n.split() if w.rstrip(".") not in ("Jr", "Sr", "II", "III", "IV")][-1]
    parts = [f"{surname(i['player'])} {i['p_play']:.0%}"
             + (f" ({i['status']})" if i["status"] not in ("none", "no_report") else "") for i in items[:5]]
    return f"**{team}**: " + " · ".join(parts)


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
        inj = [b for b in (injury_badges(r.away_team, r.get("away_injuries")),
                           injury_badges(r.home_team, r.get("home_injuries"))) if b]
        if inj:
            st.caption("Chance of playing — " + "  \n".join(inj))
        with st.expander("Why?"):
            why = pd.Series({c[4:]: r[c] for c in r.index if c.startswith("why_") and pd.notna(r[c])})
            why_chart(why, r.home_team, r.away_team)
            extra = f" · v2 model: {r.home_team} {r.p_home_v2:.0%}" if pd.notna(r.get("p_home_v2")) else ""
            st.caption(f"Elo alone: {r.home_team} {r.elo_prob:.0%}{extra}. Bars are points of predicted home margin "
                       f"and add up to the prediction ({r.pred_margin:+.1f}).")
