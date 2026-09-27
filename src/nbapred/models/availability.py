"""P(player plays tonight), from recent history and the injury report.

Fitted ONCE on FIT_SEASONS (all before the first test season) and frozen, like the box-score prior, so it
can't leak into walk-forward test seasons or the holdout. That window is also where the meaning of each
injury status is learned (e.g. how often "Questionable" players actually play).

Rows flagged `via_box_only` (a player who appeared without any recorded roster move, so historically we only know
about him *because* he played) are excluded from fitting and treated as certain to play.
"""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

FIT_SEASONS = (2016, 2020)
STATUS_LEVELS = ["no_report", "none", "Available", "Probable", "Questionable", "Doubtful", "Out"]


def design(c: pd.DataFrame) -> pd.DataFrame:
    """Model inputs. Streaks and gaps are bucketed so the logistic model can bend like the trees do.
    Absence features are unknown (NaN) before a team's first game of the season: an explicit indicator carries
    that, with neutral values underneath so the logistic model can use the rows too."""
    x = pd.DataFrame(index=c.index)
    x["season_start"] = c["missed_streak"].isna().astype(int)
    x["season_games"] = c["season_games"] / 20
    x["played_last"] = c["played_last"].fillna(1)
    for lo, hi in [(1, 1), (2, 3), (4, 9), (10, 82)]:
        x[f"missed_{lo}_{hi}"] = c["missed_streak"].between(lo, hi).astype(int)
    x["share_last10"] = c["share_last10"].fillna(1.0)  # a constant, never a batch statistic (train/serve parity)
    x["days_since_long"] = (c["days_since"] > 30).astype(int)
    x["min_ewm"] = c["min_ewm"] / 48
    x["starts_ewm"] = c["starts_ewm"]
    x["new_to_team"] = c["new_to_team"]
    x["new_season"] = c["new_season"]
    x["is_playoff"] = c["is_playoff"]
    for s in STATUS_LEVELS[1:]:  # no_report is the reference level
        x[f"status_{s}"] = (c["status"] == s).astype(int)
    return x


def trainable(c: pd.DataFrame) -> pd.Series:
    return c["via_box_only"] == 0


@dataclass
class AvailabilityModel:
    kind: str = "logistic"
    model: object = None

    def fit(self, c: pd.DataFrame) -> "AvailabilityModel":
        d = c[trainable(c) & c["season"].between(*FIT_SEASONS)]
        X, y = design(d), d["played"]
        if self.kind == "logistic":
            self.model = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        else:
            self.model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                            min_child_samples=200, verbose=-1).fit(X, y)
        return self

    def predict(self, c: pd.DataFrame, blind: bool = False) -> np.ndarray:
        """P(plays). blind=True ignores the injury report (the 'no report' condition)."""
        c = c.assign(status="no_report") if blind else c
        p = self.model.predict_proba(design(c))[:, 1]
        return np.where(c["via_box_only"].to_numpy() == 1, 1.0, p)


def compare(c: pd.DataFrame, test_seasons=(2021, 2024)) -> pd.DataFrame:
    """Log-loss of P(plays) on later seasons for logistic vs LightGBM, with and without the report."""
    t = c[trainable(c) & c["season"].between(*test_seasons)]
    rows = {}
    for kind in ("logistic", "lgbm"):
        m = AvailabilityModel(kind).fit(c)
        for blind in (False, True):
            p = np.clip(m.predict(t, blind=blind), 1e-4, 1 - 1e-4)
            rows[f"{kind}{' (no report)' if blind else ''}"] = {"log_loss": log_loss(t["played"], p),
                                                                "acc": ((p > .5) == t["played"]).mean()}
    base = t["played"].mean()
    rows["always base rate"] = {"log_loss": log_loss(t["played"], np.full(len(t), base)), "acc": max(base, 1 - base)}
    return pd.DataFrame(rows).T


def healthy(c: pd.DataFrame) -> pd.DataFrame:
    """The same candidates as if nobody were injured: no current absence streak, no injury status, but their
    normal rotation role kept. Used for 'full strength'. Established rotation players (15+ usual minutes)
    also get their recent-availability share reset, since for them a low share means injury, not the bench."""
    regular = c["min_ewm"] >= 15
    return c.assign(played_last=1, missed_streak=0, days_since=np.minimum(c["days_since"], 3), status="no_report",
                    share_last10=np.where(regular, 1.0, c["share_last10"]), new_season=0)


def predict_healthy(model: "AvailabilityModel", c: pd.DataFrame) -> np.ndarray:
    """Chance of playing if he weren't hurt (used only for players with an injury signal)."""
    return model.predict(healthy(c), blind=True)
