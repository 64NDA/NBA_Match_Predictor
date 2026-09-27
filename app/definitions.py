"""One plain-English definition per displayed number, including the time window it's based on.

Windows are read from the code's own settings, so the explanations can't drift from what's computed.
"""
import streamlit as st

from nbapred.features.availability import MIN_HALF_LIFE, POOL_WINDOW
from nbapred.features.elo import Elo
from nbapred.features.impact import FIT_EVERY_DAYS, ImpactConfig
from nbapred.data.injuries import LEAD

_cfg = ImpactConfig()
_elo = Elo()
_hl_seasons = _cfg.half_life_days / 365


def _seasons(x: float) -> str:
    return "one season" if abs(x - 1) < 0.05 else f"{x:.1f} seasons"


DEFS = {
    "impact": (
        "**Impact rating (offence, defence, net).** Points per 100 possessions compared with an average player, "
        "while he's on court. It's learned from every stretch of play with the same 10 players on the floor since "
        "2015-16, after accounting for **all four teammates and all five opponents**. So it measures the player, "
        "not his team: a good player on a bad team still rates well if the score improves when he's on court. "
        f"**Time window:** recent play counts more. A stretch's weight halves every {_seasons(_hl_seasons)}, so a "
        f"game a season ago counts half as much as one today, and anything older than "
        f"{_cfg.window_days / 365:.0f} seasons is ignored. Players with little court time are pulled toward their "
        f"box-score estimate. Refit every {FIT_EVERY_DAYS} days."),
    "per100": (
        "**Per 100 possessions vs per game.** A rating says how good someone is *while he plays*. A bench big man "
        "can rate as highly as a star guard per possession, but the star plays far more minutes. For "
        "*how much a player matters*, look at **Value** (per game)."),
    "value": (
        "**Value (points per game).** Net impact × his minutes ÷ 48: how many points per game he adds compared "
        "with an average player in the same minutes. This is the number that goes into team strength."),
    "box": (
        "**Box-score estimate.** What his stat line alone suggests (efficiency above league average, shot volume, "
        "rebounds, assists, steals, blocks, turnovers, fouls), using the same time weighting. It measures "
        "*production*. The impact rating measures *what happened on the scoreboard with him on court*. When they "
        "disagree, the impact rating is seeing things box scores miss, like defence, spacing and screening, or "
        "noise when the sample is small."),
    "sample": (
        "**Rating certainty.** How sure we are of the player's *rating*, based on how much recent court time it rests "
        "on (recency-weighted possessions). Low = under 1,500: the rating leans mostly on his box-score estimate and "
        "could move a lot. Medium = 1,500 to 5,000. High = over 5,000. This is **not** about whether he'll play "
        "tonight; that's the *chance of playing* on the Upcoming and Matchup pages."),
    "usual_min": (
        "**Usual minutes.** His average minutes *in games he played*, weighted toward recent games: the weight "
        f"halves every {MIN_HALF_LIFE} games. It carries over from last season until he plays this season. Rookies "
        "with no NBA minutes get a default by draft slot."),
    "p_play": (
        "**Chance of playing.** From the official NBA injury report as it stood at least "
        f"{int(LEAD.total_seconds() // 60)} minutes before tip-off, plus his absences *this season* (last "
        "season's rest days don't carry over). A LightGBM model learned what each status really means from 2016-17 "
        "to 2020-21 games. Healthy, unlisted rotation players come out at about 95–99%."),
    "min_tonight": (
        "**Minutes tonight.** Chance of playing × usual minutes, then re-shared so the team adds up to 240 minutes: "
        "absent players' minutes go to teammates in proportion to their usual minutes. If there are more bodies than "
        "minutes, the end of the bench is cut, not the starters. Stars play a bit more in the playoffs."),
    "team_strength": (
        "**Team strength.** The sum of every player's value tonight: projected net rating in points per 100 "
        "possessions (about points per game), where 0 is an average team. It uses the **current official roster** "
        "and today's player ratings."),
    "full_strength": (
        "**Full strength.** The same lineup with only the players who might be hurt (on the injury report, or who "
        "missed recent games this season) restored to their normal chance of playing. Everyone else is unchanged."),
    "absences": (
        "**Absences.** Tonight minus full strength: how much the injury situation changes the team, after the missing "
        "minutes go to teammates. Each player's own effect is worked out the same way, one at a time. It's usually "
        "negative, and positive only if the players likely out are worse than the teammates who replace them."),
    "elo": (
        "**Elo.** A results-only rating: wins and margins, nothing about who played. Every team starts at 1505. After "
        f"each game the winner takes points from the loser (K = {_elo.k:.0f}); big wins and upsets move it more. Home "
        f"court is worth {_elo.home_adv:.0f} Elo points (about {_elo.home_adv / 28:.1f} points of margin). "
        f"**Time window:** every game since 2015-16 counts, but recent games dominate, and each offseason every "
        f"team is pulled {1 - _elo.carryover:.0%} of the way back to average. 28 Elo points ≈ 1 point of margin."),
    "win_prob": (
        "**Win probability and score.** The model predicts the margin and the total points from team strength, Elo, "
        "home court, rest and travel. The win probability is how often a margin like that holds up, given the "
        "typical 11–12 point spread of real results around predictions. The score is total ± margin, split in two."),
    "pool": (
        f"**Who counts for a team.** The official current roster. In the backtest, the roster as it stood that day, "
        f"rebuilt from the NBA's dated trades, signings and waivers, keeping players who have played or joined in the "
        f"team's last {POOL_WINDOW} games."),
}


def explain(*keys: str, title: str = "How these numbers are calculated") -> None:
    with st.expander(title):
        for k in keys:
            st.markdown(DEFS[k])


def helptext(key: str) -> str:
    """Column-tooltip version (markdown bold stripped)."""
    return DEFS[key].replace("**", "")
