"""Baselines and model-combining helpers shared by every model version.

Each is a fit_predict(train, test) -> DataFrame(p_home, pred_margin, pred_total).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nbapred.features.elo import Elo


def home_baseline(train, test):
    return pd.DataFrame({"p_home": train["home_win"].mean(), "pred_margin": train["margin"].mean(),
                         "pred_total": np.nan}, index=test.index)


def elo_baseline(train, test):
    # FiveThirtyEight rule of thumb: 28 Elo points ~ 1 point of spread (home_adv included)
    home_adv = np.where(test["neutral"] == 1, 0, Elo().home_adv)
    return pd.DataFrame({"p_home": test["elo_prob"], "pred_margin": (test["diff_elo"] + home_adv) / 28,
                         "pred_total": np.nan}, index=test.index)


def logistic(features, C=0.05):
    def fit_predict(train, test):
        d = train.dropna(subset=features)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(d[features], d["home_win"])
        return pd.DataFrame({"p_home": clf.predict_proba(test[features].fillna(0))[:, 1],
                             "pred_margin": np.nan, "pred_total": np.nan}, index=test.index)
    return fit_predict


def ensemble(*fps):
    """Average p_home / margins / totals of several fit_predicts."""
    def fit_predict(train, test):
        parts = [fp(train, test) for fp in fps]
        return sum(parts) / len(parts)
    return fit_predict


