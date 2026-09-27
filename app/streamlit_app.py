from pathlib import Path

import streamlit as st

st.set_page_config(page_title="NBA Match Predictor", page_icon="🏀", layout="wide")

from common import inject_css  # noqa: E402  (after set_page_config)

inject_css()
st.logo(str(Path(__file__).parent / "assets" / "ball.svg"), size="large")
st.sidebar.markdown('<div class="nbap-brand">NBA Match<br><span>Predictor</span></div>'
                    '<div class="nbap-brand-sub">Win chances and scores from who\'s playing tonight</div>',
                    unsafe_allow_html=True)

pages = st.navigation([
    st.Page("views/upcoming.py", title="Upcoming games", icon=":material/calendar_month:", default=True),
    st.Page("views/matchup.py", title="Matchup explorer", icon=":material/sports_basketball:"),
    st.Page("views/teams.py", title="Teams", icon=":material/groups:"),
    st.Page("views/players.py", title="Players", icon=":material/person:"),
    st.Page("views/track_record.py", title="Track record", icon=":material/insights:"),
    st.Page("views/about.py", title="How it works", icon=":material/info:"),
])
pages.run()
