"""v3 model: ridge combiner over point-scale features, plus optional tree-based refinements.

    margin model   ridge on the selected feature groups
    total model    ridge on projected total, pace and league scoring level
    P(home win)    Phi(margin / sigma), sigma chosen by log-loss on pooled out-of-sample predictions of the
                   last 3 inner seasons (fixes v2's underconfidence)
    matchup trees  optional: a shallow LightGBM or random forest fitted to the ridge's out-of-fold residuals
                   from style/matchup inputs; its prediction is added to the margin
    stack          optional: LightGBM margin model blended with ridge, weight fitted out-of-fold

Features for players come in three availability conditions (known / live / blind). A model is trained on
one condition and can be tested on any, e.g. train 'known', test 'live' = what the app does with an
injury report.
"""
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nbapred.features.style import PROFILE

GROUPS = {
    "Home court": ["home", "playoff_home", "altitude", "team_hca"],
    "Team strength (Elo)": ["diff_elo"],
    "Player ratings": ["proj_net"],
    "Injuries": ["injury_cost"],
    "Recent form": ["diff_net_rating_fast", "diff_net_rating_slow", "diff_margin_slow"],
    "Matchup & style": ["ff_efg_pct", "ff_tov_pct", "ff_oreb_pct", "ff_fta_rate", "clash_three", "clash_rim",
                        "clash_reb", "clash_tov", "height_diff", "clash_size_reb"],
    "Series / head-to-head": ["series_resid", "h2h_resid"],
    "Rest & travel": ["diff_rest_days", "diff_b2b", "diff_games_last7", "diff_travel_km", "h_b2b", "a_b2b", "a_travel_km"],
    "Playoff game": ["is_playoff"],
}
PER_CONDITION = {"proj_net", "injury_cost", "proj_margin", "proj_total"}
TOTAL_FEATURES = ["proj_total", "sum_pace", "pace_gap", "league_total", "is_playoff"]
TREE_FEATURES = ([f"{s}_{p}" for s in ("h", "a") for p in PROFILE + [f"allowed_{x}" for x in PROFILE]]
                 + ["ff_efg_pct", "ff_tov_pct", "ff_oreb_pct", "ff_fta_rate", "height_diff", "sum_pace", "pace_gap",
                    "is_playoff", "proj_net"])


def view(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Expose one availability condition's player columns under generic names (proj_net, injury_cost, ...)."""
    return df.rename(columns={f"{c}_{condition}": c for c in PER_CONDITION})


def features_for(groups: list[str]) -> list[str]:
    return [f for g in groups for f in GROUPS[g]]


def _ridge(alpha: float, intercept: bool = True):
    """intercept=False for the margin: every margin feature is a home-minus-away difference or a flag that is 0
    at a neutral venue, so two identical teams on neutral ground must come out at 0 and home court lives entirely
    in the 'home' term. Scaling keeps zeros at zero (no centring)."""
    if intercept:
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=alpha, fit_intercept=False))


@dataclass
class V3Model:
    groups: list[str]
    train_condition: str = "live"
    alpha: float = 10.0
    trees: str | None = None        # None | "lgbm" | "rf": matchup residual model
    stack: str | None = None        # None | "lgbm" | "mlp": blend ridge with a second margin model
    sigma: float = 13.0
    margin_model: object = field(default=None, repr=False)
    total_model: object = field(default=None, repr=False)
    tree_model: object = field(default=None, repr=False)
    stack_model: object = field(default=None, repr=False)
    stack_w: float = 0.0

    @property
    def features(self):
        return features_for(self.groups)

    # -- pieces -------------------------------------------------------------------------------------
    def _fit_margin(self, d):
        return _ridge(self.alpha, intercept=False).fit(d[self.features], d["margin"])

    def _tree(self):
        if self.trees == "rf":
            return RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=100, max_features=0.5,
                                         n_jobs=-1, random_state=0)
        return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.02, max_depth=3, num_leaves=7,
                                 min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                 reg_lambda=10.0, verbose=-1)

    def _stacker(self):
        if self.stack == "mlp":
            # small neural network: 2 hidden layers, strong L2, early stopping on a time-ordered 10% split
            return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0,
                                                                early_stopping=True, validation_fraction=0.1,
                                                                max_iter=500, random_state=0))
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=100,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)

    def _oof(self, d, k_seasons=3):
        """Out-of-fold ridge margins for the last k seasons of d (each predicted by earlier seasons only)."""
        last = int(d["season"].max())
        parts = []
        for s in range(last - k_seasons + 1, last + 1):
            tr, va = d[d["season"] < s], d[d["season"] == s]
            if len(tr) and len(va):
                parts.append(va.assign(oof=self._fit_margin(tr).predict(va[self.features])))
        return pd.concat(parts)

    # -- fit / predict ------------------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "V3Model":
        d = view(train, self.train_condition).dropna(subset=["margin", "total"])
        # every feature is a centred difference or a 0/1 flag, so a missing value (e.g. no shot history yet)
        # means "no edge": 0, the same as at prediction time
        d = d.assign(**{c: d[c].fillna(0) for c in set(self.features + TOTAL_FEATURES) - {"league_total"}})
        oof = self._oof(d)
        margin_oof = oof["oof"].to_numpy()

        if self.trees:
            # trees learn what ridge's residual has left, from style/matchup inputs; out-of-fold targets
            self.tree_model = self._tree().fit(oof[TREE_FEATURES], oof["margin"] - oof["oof"])
            margin_oof = margin_oof + self.tree_model.predict(oof[TREE_FEATURES])
        if self.stack:
            parts = []
            last = int(d["season"].max())
            for s in range(last - 2, last + 1):
                tr, va = d[d["season"] < s], d[d["season"] == s]
                parts.append(self._stacker().fit(tr[self.features], tr["margin"]).predict(va[self.features]))
            g_oof = np.concatenate(parts)
            y = oof["margin"].to_numpy()
            self.stack_w = float(minimize_scalar(lambda w: np.mean((y - ((1 - w) * margin_oof + w * g_oof)) ** 2),
                                                 bounds=(0, 1), method="bounded").x)
            margin_oof = (1 - self.stack_w) * margin_oof + self.stack_w * g_oof
            self.stack_model = self._stacker().fit(d[self.features], d["margin"])

        y = oof["home_win"].to_numpy()
        ll = lambda s: -np.mean(y * norm.logcdf(margin_oof / s) + (1 - y) * norm.logcdf(-margin_oof / s))
        self.sigma = float(minimize_scalar(ll, bounds=(6, 25), method="bounded").x)

        self.margin_model = self._fit_margin(d)
        self.total_model = _ridge(self.alpha).fit(d[TOTAL_FEATURES], d["total"])
        return self

    def predict(self, df: pd.DataFrame, condition: str | None = None) -> pd.DataFrame:
        x = view(df, condition or self.train_condition)
        margin = self.margin_model.predict(x[self.features].fillna(0))
        if self.tree_model is not None:
            margin = margin + self.tree_model.predict(x[TREE_FEATURES].fillna(x[TREE_FEATURES].mean()))
        if self.stack_model is not None:
            margin = (1 - self.stack_w) * margin + self.stack_w * self.stack_model.predict(x[self.features].fillna(0))
        total = self.total_model.predict(x[TOTAL_FEATURES].fillna(x[TOTAL_FEATURES].mean()))
        return pd.DataFrame({"p_home": norm.cdf(margin / self.sigma), "pred_margin": margin, "pred_total": total,
                             "pred_home_pts": (total + margin) / 2, "pred_away_pts": (total - margin) / 2},
                            index=df.index)

    def explain(self, row: pd.Series, condition: str | None = None) -> pd.Series:
        """Points of home margin per group; they sum (with 'Matchup (trees)' / blending if used) to the prediction.
        Reference = two evenly matched teams, neutral venue, regular season, no rest/travel edge (all features 0)."""
        x = view(row.to_frame().T, condition or self.train_condition).iloc[0]
        sc, rg = self.margin_model.named_steps["standardscaler"], self.margin_model.named_steps["ridge"]
        w = rg.coef_ / sc.scale_  # no centring and no intercept: contributions are exactly w * x
        vals = x[self.features].astype(float).fillna(0).to_numpy()
        contrib = pd.Series(w * vals, index=self.features)
        out = {}
        for g in self.groups:
            out[g] = float(contrib[GROUPS[g]].sum())
        if self.tree_model is not None:
            out["Matchup (trees)"] = float(self.tree_model.predict(x[TREE_FEATURES].to_frame().T.astype(float))[0])
        s = pd.Series(out)
        if self.stack_model is not None:
            full = float(self.predict(row.to_frame().T, condition)["pred_margin"].iloc[0])
            s = s * (1 - self.stack_w)
            s["Blend (gradient boosting)" if self.stack == "lgbm" else "Blend (neural net)"] = full - s.sum()
        return s


def fit_predict(groups, train_condition="live", test_condition=None, **kw):
    """Walk-forward adapter: fit on train, predict test under `test_condition` (default = train's)."""
    def fp(train, test):
        return V3Model(groups, train_condition, **kw).fit(train).predict(test, test_condition or train_condition)
    return fp
