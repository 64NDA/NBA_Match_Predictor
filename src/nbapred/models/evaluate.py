"""Walk-forward evaluation.

For each test season S: fit on seasons [first_train, S), predict S. The holdout season is excluded
from everything here — evaluate it once with `holdout_report` when the model is final.
"""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from nbapred.config import HOLDOUT_SEASON

FIRST_TRAIN = 2016          # 2015-16 is burn-in for Elo / form / player ratings
TEST_SEASONS = [2021, 2022, 2023, 2024]

# fit_predict(train, test) -> DataFrame indexed like test with p_home, pred_margin, pred_total
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def metrics(df: pd.DataFrame) -> dict:
    p = df["p_home"].clip(1e-4, 1 - 1e-4)
    out = {"games": len(df),
           "acc": ((p > 0.5) == df["home_win"]).mean(),
           "log_loss": log_loss(df["home_win"], p, labels=[0, 1]),
           "brier": brier_score_loss(df["home_win"], p)}
    if "pred_margin" in df and df["pred_margin"].notna().all():
        err = df["pred_margin"] - df["margin"]
        out["margin_mae"] = err.abs().mean()
        out["margin_rmse"] = np.sqrt((err ** 2).mean())
    if "pred_total" in df and df["pred_total"].notna().all():
        out["total_rmse"] = np.sqrt(((df["pred_total"] - df["total"]) ** 2).mean())
    return out


def walk_forward(matrix: pd.DataFrame, fit_predict: FitPredict, test_seasons=TEST_SEASONS) -> pd.DataFrame:
    """Return all out-of-sample predictions (with targets) across the test seasons."""
    assert HOLDOUT_SEASON not in test_seasons
    preds = []
    for s in test_seasons:
        train = matrix[(matrix["season"] >= FIRST_TRAIN) & (matrix["season"] < s)]
        test = matrix[matrix["season"] == s]
        p = fit_predict(train, test)
        preds.append(test[["game_id", "date", "season", "is_playoff", "home_team", "away_team",
                           "home_win", "margin", "total"]].join(p))
    return pd.concat(preds)


def report(preds: pd.DataFrame, by: str = "season") -> pd.DataFrame:
    rows = {k: metrics(g) for k, g in preds.groupby(by)}
    rows["ALL"] = metrics(preds)
    return pd.DataFrame(rows).T


def calibration(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    b = pd.cut(preds["p_home"], np.linspace(0, 1, bins + 1))
    return preds.groupby(b, observed=True).agg(games=("home_win", "size"), predicted=("p_home", "mean"),
                                               actual=("home_win", "mean"))


def compare(matrix: pd.DataFrame, models: dict[str, FitPredict]) -> pd.DataFrame:
    """Walk-forward summary (all test seasons pooled) for several models side by side."""
    return pd.DataFrame({name: metrics(walk_forward(matrix, fp)) for name, fp in models.items()}).T


def per_game_scores(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-game log-loss and hit (for paired comparisons)."""
    p = preds["p_home"].clip(1e-4, 1 - 1e-4)
    y = preds["home_win"]
    return pd.DataFrame({"ll": -(y * np.log(p) + (1 - y) * np.log(1 - p)), "hit": ((p > 0.5) == y).astype(float),
                         "game_id": preds["game_id"]}).set_index("game_id")


def paired(results: dict[str, pd.DataFrame], baseline: str) -> pd.DataFrame:
    """Compare walk-forward predictions on the games they share, against `baseline`.
    d_logloss < 0 is better; verdict needs |d| > 2 paired standard errors."""
    base = per_game_scores(results[baseline])
    rows = {}
    for name, preds in results.items():
        s = per_game_scores(preds)
        common = s.index.intersection(base.index)
        d = s.loc[common] - base.loc[common]
        m = metrics(preds)
        rows[name] = {"games": len(common), "acc": s.loc[common, "hit"].mean(), "log_loss": s.loc[common, "ll"].mean(),
                      "margin_mae": m.get("margin_mae", np.nan), "d_logloss": d["ll"].mean(),
                      "se": d["ll"].std() / np.sqrt(len(d)) if len(d) > 1 else 0.0, "d_acc_pp": 100 * d["hit"].mean()}
    df = pd.DataFrame(rows).T
    df["verdict"] = np.where(df["d_logloss"] < -2 * df["se"], "BETTER",
                             np.where(df["d_logloss"] > 2 * df["se"], "worse", "~same"))
    df.loc[baseline, "verdict"] = "(baseline)"
    return df
