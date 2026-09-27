import streamlit as st

st.set_page_config(page_title="NBA Match Predictor", page_icon="🏀", layout="wide")

pages = st.navigation([
    st.Page("views/upcoming.py", title="Upcoming games", icon=":material/calendar_month:", default=True),
    st.Page("views/matchup.py", title="Matchup explorer", icon=":material/sports_basketball:"),
    st.Page("views/teams.py", title="Teams", icon=":material/groups:"),
    st.Page("views/track_record.py", title="Track record", icon=":material/insights:"),
    st.Page("views/about.py", title="How it works", icon=":material/info:"),
])
pages.run()
