"""Shared loaders, styling and UI pieces for the Streamlit pages."""
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

# NBA-app inspired palette (also in .streamlit/config.toml). Home = blue, away = red everywhere: the pair passes the
# colour-blind separation, lightness and contrast checks on the card surface. Gold is the accent (controls, "our
# model"), so red and blue only ever mean the two teams.
HOME, AWAY, GOLD = "#3B82F6", "#E03A3E", "#F2B632"
TEXT, MUTED, SURFACE, BORDER, GRID = "#EDEEF0", "#9CA3AF", "#17181D", "#2A2D35", "rgba(255,255,255,0.07)"
HEADING_FONT = "'Barlow Condensed', sans-serif"
GROUP_ORDER = ["Baseline", "Home court", "Team strength (Elo)", "Player ratings", "Injuries", "Recent form",
               "Matchup & style", "Matchup (trees)", "Series / head-to-head", "Rest & travel", "Playoff game",
               "Blend (gradient boosting)", "Blend (neural net)"]
STATUS_OPTIONS = ["Available", "Probable", "Questionable", "Doubtful", "Out"]

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
/* page header */
.nbap-kicker {{ font-family: {HEADING_FONT}; color: {GOLD}; font-weight: 700; font-size: 0.85rem;
               letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.1rem; }}
.nbap-title {{ font-family: {HEADING_FONT}; font-weight: 800; font-size: 2.7rem; line-height: 1;
              text-transform: uppercase; letter-spacing: 0.01em; margin: 0; color: {TEXT}; }}
.nbap-rule {{ width: 72px; height: 4px; border-radius: 2px; margin: 0.6rem 0 0.6rem;
             background: linear-gradient(90deg, {HOME} 0 50%, {AWAY} 50% 100%); }}
.nbap-sub {{ color: {MUTED}; font-size: 0.95rem; margin-bottom: 0.6rem; }}
h2, h3, h4, h5 {{ text-transform: uppercase; letter-spacing: 0.02em; }}
/* bordered containers become cards */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .gc) {{ background: {SURFACE}; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ transition: border-color 0.15s ease; }}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{ border-color: rgba(242, 182, 50, 0.45); }}
/* game card */
.gc-meta {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 0.78rem;
           text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }}
.gc-meta b {{ color: {GOLD}; font-weight: 600; }}
.gc-row {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 0.5rem; }}
.gc-team.home {{ text-align: right; }}
.gc-abbr {{ font-family: {HEADING_FONT}; font-weight: 800; font-size: 2.3rem; line-height: 1; color: {TEXT}; }}
.gc-name {{ color: {MUTED}; font-size: 0.8rem; margin-top: 0.1rem; }}
.gc-side {{ display: inline-block; font-size: 0.65rem; letter-spacing: 0.12em; font-weight: 700; padding: 0.05rem 0.4rem;
           border-radius: 0.3rem; margin-bottom: 0.3rem; }}
.gc-score {{ text-align: center; }}
.gc-pts {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 2rem; line-height: 1; color: {TEXT}; }}
.gc-pts span {{ color: {MUTED}; margin: 0 0.35rem; }}
.gc-lbl {{ color: {MUTED}; font-size: 0.65rem; letter-spacing: 0.14em; margin-top: 0.2rem; }}
.gc-pct {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 1.15rem; margin-top: 0.25rem; }}
.gc-bar {{ display: flex; height: 8px; gap: 3px; margin: 0.7rem 0 0.3rem; }}
.gc-bar div {{ border-radius: 4px; }}
/* broadcast-style stat comparison */
.sc {{ width: 100%; }}
.sc-head, .sc-row {{ display: grid; grid-template-columns: 1fr 1.3fr 1fr; align-items: center; }}
.sc-head {{ font-family: {HEADING_FONT}; font-weight: 800; font-size: 1.3rem; padding-bottom: 0.3rem;
           border-bottom: 1px solid {BORDER}; }}
.sc-row {{ padding: 0.42rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
.sc-row .l {{ text-align: left; font-weight: 600; }}
.sc-row .r {{ text-align: right; font-weight: 600; }}
.sc-row .m {{ text-align: center; color: {MUTED}; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }}
/* metrics and misc */
[data-testid="stMetricValue"] {{ font-family: {HEADING_FONT}; font-weight: 700; }}
[data-testid="stMetricLabel"] {{ text-transform: uppercase; letter-spacing: 0.05em; }}
[data-testid="stExpander"] summary p {{ font-weight: 600; }}
[data-testid="stSidebarNav"] a span {{ font-weight: 500; }}
.nbap-brand {{ font-family: {HEADING_FONT}; font-weight: 800; font-size: 1.35rem; text-transform: uppercase;
              letter-spacing: 0.04em; line-height: 1.05; color: {TEXT}; }}
.nbap-brand span {{ color: {GOLD}; }}
.nbap-brand-sub {{ color: {MUTED}; font-size: 0.75rem; margin-top: 0.2rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="nbap-kicker">{kicker}</div><h1 class="nbap-title">{title}</h1>'
                f'<div class="nbap-rule"></div>' + (f'<div class="nbap-sub">{subtitle}</div>' if subtitle else ""),
                unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    """Dark, quiet chart chrome that matches the theme: transparent background, muted axes, faint grid."""
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=MUTED, size=12),
                      hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT, family="Inter")),
                      legend=dict(font=dict(color=TEXT)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.25)", linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.25)", linecolor=BORDER)
    return fig


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
    """Split bar: away share on the left (red), home share on the right (blue)."""
    pa = 1 - p_home
    st.markdown(
        f"""<div style="display:flex;justify-content:space-between;font-family:{HEADING_FONT};font-weight:700;
             font-size:1.1rem"><span style="color:{AWAY}">{away} {pa:.0%}</span>
             <span style="color:{HOME}">{p_home:.0%} {home}</span></div>
        <div class="gc-bar" role="img" aria-label="{away} {pa:.0%}, {home} {p_home:.0%}">
          <div style="width:{pa*100:.1f}%;background:{AWAY}"></div>
          <div style="width:{p_home*100:.1f}%;background:{HOME}"></div></div>""",
        unsafe_allow_html=True)


def stat_comparison(comp: pd.DataFrame, away: str, home: str) -> None:
    """Away value | stat name | home value, like a broadcast graphic. comp: index = stat, columns = [away, home]."""
    rows = "".join(f'<div class="sc-row"><div class="l">{r[away]}</div><div class="m">{name}</div>'
                   f'<div class="r">{r[home]}</div></div>' for name, r in comp.iterrows())
    st.markdown(f'<div class="sc"><div class="sc-head"><div style="color:{AWAY}">{away}</div><div></div>'
                f'<div style="color:{HOME};text-align:right">{home}</div></div>{rows}</div>', unsafe_allow_html=True)


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
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), showlegend=False, bargap=0.35,
                      xaxis=dict(title=f"← favours {away}   ·   points   ·   favours {home} →", zeroline=True,
                                 zerolinewidth=1), yaxis=dict(autorange="reversed", tickfont=dict(color=TEXT)))
    st.plotly_chart(style_fig(fig, height), use_container_width=True, config={"displayModeBar": False})


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


def _tipoff_et(utc) -> str:
    try:
        t = pd.Timestamp(utc)
        t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert("America/New_York")
        return t.strftime("%a %d %b · %I:%M %p ET").replace(" 0", " ")
    except (ValueError, TypeError):
        return ""


def prediction_card(r: pd.Series):
    pa = 1 - r.p_home
    fav_away = pa > r.p_home
    with st.container(border=True):
        st.markdown(f"""
<div class="gc">
  <div class="gc-meta"><span>{_tipoff_et(r.get("tipoff_utc"))}</span><span><b>{r.get("label") or ""}</b></span></div>
  <div class="gc-row">
    <div class="gc-team away">
      <div class="gc-side" style="background:rgba(224,58,62,0.15);color:{AWAY}">AWAY</div>
      <div class="gc-abbr">{r.away_team}</div><div class="gc-name">{team_label(r.away_team)}</div>
      <div class="gc-pct" style="color:{AWAY};{'' if fav_away else 'opacity:0.75'}">{pa:.0%}</div>
    </div>
    <div class="gc-score">
      <div class="gc-pts">{r.pred_away_pts:.0f}<span>–</span>{r.pred_home_pts:.0f}</div>
      <div class="gc-lbl">PREDICTED SCORE</div>
    </div>
    <div class="gc-team home">
      <div class="gc-side" style="background:rgba(59,130,246,0.15);color:{HOME}">HOME</div>
      <div class="gc-abbr">{r.home_team}</div><div class="gc-name">{team_label(r.home_team)}</div>
      <div class="gc-pct" style="color:{HOME};{'opacity:0.75' if fav_away else ''}">{r.p_home:.0%}</div>
    </div>
  </div>
  <div class="gc-bar" role="img" aria-label="{r.away_team} {pa:.0%}, {r.home_team} {r.p_home:.0%}">
    <div style="width:{pa*100:.1f}%;background:{AWAY}"></div><div style="width:{r.p_home*100:.1f}%;background:{HOME}"></div>
  </div>
</div>""", unsafe_allow_html=True)
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
            fig.add_scatter(x=seg.date, y=seg.elo, mode="lines", line=dict(color=color, width=2.5), name=team,
                            legendgroup=team, showlegend=i == 0,
                            hovertemplate=f"{team} %{{x|%d %b %Y}}: %{{y:.0f}}<extra></extra>")
    fig.add_hline(y=1505, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
                  annotation_text="league average", annotation_position="bottom right",
                  annotation_font=dict(color=MUTED))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                      legend=dict(orientation="h", y=1.1), yaxis_title="Elo")
    st.plotly_chart(style_fig(fig, height), use_container_width=True, config={"displayModeBar": False})
