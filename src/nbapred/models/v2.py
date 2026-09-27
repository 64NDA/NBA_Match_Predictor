"""v2 model (legacy): ridge margin/total on the Elo + Game Score + form + schedule matrix.

Kept runnable for comparison with v3. Baselines shared by all versions live in models/baselines.py.

Every model predicts the home margin and the game total. Win probability is derived from the margin,
p_home = Phi(margin / sigma), so the scoreline and the winner can never disagree (the v1 bug).
sigma is the residual std of the margin model measured on the last training season *out of sample*.
"""
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nbapred.features.form import ALLOWED, OWN

FORM = [f"{s}_{sp}" for s in OWN + [f"opp_{c}" for c in ALLOWED] for sp in ("fast", "slow")]
PLAYERS = ["available_strength", "missing_strength"]
SCHEDULE = ["rest_days", "b2b", "games_last7", "travel_km"]

FEATURE_SETS = {
    "elo": ["diff_elo"],
    "elo+players": ["diff_elo"] + [f"diff_{c}" for c in PLAYERS],
    "elo+players+form": ["diff_elo"] + [f"diff_{c}" for c in PLAYERS + FORM],
    "all": ["diff_elo", "is_playoff"] + [f"diff_{c}" for c in PLAYERS + FORM + SCHEDULE]
           + ["h_b2b", "a_b2b", "a_travel_km"],
}
TOTAL_FEATURES = ["league_total", "is_playoff"] + [f"sum_{c}" for c in (
    "pace_fast", "pace_slow", "pts_fast", "pts_slow", "opp_pts_fast", "opp_pts_slow",
    "off_rating_slow", "def_rating_slow", "available_strength", "missing_strength", "b2b")]


def _ridge(alpha=10.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _lgbm():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)


@dataclass
class MarginTotalModel:
    features: list[str]
    margin_factory: callable = _ridge
    total_factory: callable = _ridge
    sigma: float = 12.0
    margin_model: object = field(default=None, repr=False)
    total_model: object = field(default=None, repr=False)

    def _fit_models(self, df):
        d = df.dropna(subset=self.features + TOTAL_FEATURES)
        m = self.margin_factory().fit(d[self.features], d["margin"])
        t = self.total_factory().fit(d[TOTAL_FEATURES], d["total"])
        return m, t

    def fit(self, train: pd.DataFrame) -> "MarginTotalModel":
        # sigma from an honest inner split: fit on all but the last season, measure on the last
        last = train["season"].max()
        inner, val = train[train["season"] < last], train[train["season"] == last].dropna(subset=self.features)
        m, _ = self._fit_models(inner)
        self.sigma = float(np.std(val["margin"] - m.predict(val[self.features])))
        self.margin_model, self.total_model = self._fit_models(train)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].fillna(0)  # missing only for a team's very first game
        margin = self.margin_model.predict(X)
        total = self.total_model.predict(df[TOTAL_FEATURES].fillna(df[TOTAL_FEATURES].mean()))
        return pd.DataFrame({"p_home": norm.cdf(margin / self.sigma), "pred_margin": margin,
                             "pred_total": total, "pred_home_pts": (total + margin) / 2,
                             "pred_away_pts": (total - margin) / 2}, index=df.index)

    def explain(self, row: pd.Series) -> pd.Series:
        """Per-feature contribution (points of home margin) for one game, ridge models only.

        The reference is two evenly matched teams in a regular-season game with no rest/travel edge:
        every feature = 0. `baseline` is the margin predicted there (i.e. pure home-court advantage),
        and baseline + contributions = the predicted margin exactly.
        """
        scaler, ridge = self.margin_model.named_steps["standardscaler"], self.margin_model.named_steps["ridge"]
        w = ridge.coef_ / scaler.scale_                       # coefficients on the raw feature scale
        x = row[self.features].astype(float).fillna(0).values
        baseline = ridge.intercept_ - float(w @ scaler.mean_)
        return pd.concat([pd.Series({"baseline": baseline}), pd.Series(w * x, index=self.features)])


def margin_total(features, margin="ridge", total="ridge"):
    factories = {"ridge": _ridge, "lgbm": _lgbm}

    def fit_predict(train, test):
        return MarginTotalModel(features, factories[margin], factories[total]).fit(train).predict(test)
    return fit_predict


# ---- explanations ----------------------------------------------------------------------------

FEATURE_GROUPS = {
    "Team strength (Elo)": ["diff_elo"],
    "Player availability": [f"diff_{c}" for c in PLAYERS],
    "Recent form": [f"diff_{c}" for c in FORM],
    "Rest & travel": [f"diff_{c}" for c in SCHEDULE] + ["h_b2b", "a_b2b", "a_travel_km"],
    "Playoff game": ["is_playoff"],
}


def grouped_explanation(model: MarginTotalModel, row: pd.Series) -> pd.Series:
    """Contributions to the home margin (points) by feature group, plus 'Home court' baseline."""
    c = model.explain(row)
    out = {"Home court": c["baseline"]}
    for group, cols in FEATURE_GROUPS.items():
        out[group] = c[[f for f in cols if f in c.index]].sum()
    return pd.Series(out)


def no_availability(df: pd.DataFrame) -> pd.DataFrame:
    """v2's features as it would see them live without injury data: everyone on the roster counts as available.
    (v2's backtest used who actually played, which isn't known before tip-off.)"""
    df = df.copy()
    for side in ("h", "a"):
        df[f"{side}_available_strength"] = df[f"{side}_full_strength"]
        df[f"{side}_missing_strength"] = 0.0
    df["diff_available_strength"] = df.h_available_strength - df.a_available_strength
    df["diff_missing_strength"] = 0.0
    return df
