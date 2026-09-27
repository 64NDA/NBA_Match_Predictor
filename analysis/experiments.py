"""Accuracy experiments (walk-forward 2021-22 -> 2024-25 only; the 2025-26 holdout is not touched).

Every variant is compared with the current production config on the same games. Differences are
reported with a paired standard error, so noise isn't mistaken for improvement:
    d_logloss < 0 is better; |d_logloss| < ~2 SE is noise.

    python analysis/experiments.py
"""
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nbapred.features.players as PL
from nbapred.data.build import load
from nbapred.features.build import load_matrix
from nbapred.models import predictors as P
from nbapred.models.evaluate import walk_forward

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
ALL = P.FEATURE_SETS["all"]


@dataclass
class Variant:
    """A margin model with switchable training/calibration choices."""
    features: list
    alpha: float = 10.0
    prob: str = "sigma"             # sigma: Phi(m / residual sd) | sigma_mle: sd chosen by log-loss | platt
    drop: callable = None           # train rows to exclude
    recent_hca: int = 0             # re-centre intercept on the last k training seasons' mean residual
    weights: callable = None

    def _fit(self, d):
        est = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        kw = {"ridge__sample_weight": self.weights(d)} if self.weights else {}
        return est.fit(d[self.features], d["margin"], **kw)

    def fit_predict(self, train, test):
        d = train.dropna(subset=self.features)
        if self.drop is not None:
            d = d[~self.drop(d)]
        last = d.season.max()
        inner, val = d[d.season < last], d[d.season == last]
        vm = self._fit(inner).predict(val[self.features])
        if self.prob == "sigma":
            to_p = (lambda s: lambda x: norm.cdf(x / s))(np.std(val.margin - vm))
        elif self.prob == "sigma_mle":
            ll = lambda s: -np.mean(val.home_win * norm.logcdf(vm / s) + (1 - val.home_win) * norm.logcdf(-vm / s))
            s = minimize_scalar(ll, bounds=(6, 25), method="bounded").x
            to_p = lambda x, s=s: norm.cdf(x / s)
        else:  # platt: logistic(a + b * margin), also absorbs any home bias
            lr = LogisticRegression(C=1e6).fit(vm.reshape(-1, 1), val.home_win)
            to_p = lambda x, lr=lr: lr.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]
        model = self._fit(d)
        shift = 0.0
        if self.recent_hca:
            r = d[d.season > last - self.recent_hca]
            shift = float((r.margin - model.predict(r[self.features])).mean())
        margin = model.predict(test[self.features].fillna(0)) + shift
        return pd.DataFrame({"p_home": to_p(margin), "pred_margin": margin, "pred_total": np.nan}, index=test.index)


def score(m, fp, base=None):
    wf = walk_forward(m, fp)
    p = wf.p_home.clip(1e-4, 1 - 1e-4)
    wf["ll"] = -(wf.home_win * np.log(p) + (1 - wf.home_win) * np.log(1 - p))
    wf["hit"] = ((p > 0.5) == wf.home_win).astype(float)
    out = {"log_loss": wf.ll.mean(), "acc": wf.hit.mean(), "margin_mae": (wf.pred_margin - wf.margin).abs().mean()}
    if base is not None:
        d = wf.set_index("game_id")[["ll", "hit"]] - base.set_index("game_id")[["ll", "hit"]]
        out |= {"d_logloss": d.ll.mean(), "se": d.ll.std() / np.sqrt(len(d)), "d_acc_pp": 100 * d.hit.mean()}
    return out, wf


def run(m, variants, base_wf):
    rows = {name: score(m, v.fit_predict if isinstance(v, Variant) else v, base_wf)[0] for name, v in variants.items()}
    df = pd.DataFrame(rows).T
    df["verdict"] = np.where(df.d_logloss < -2 * df.se, "BETTER", np.where(df.d_logloss > 2 * df.se, "worse", "~same"))
    return df


class PooledSigma(Variant):
    """sigma chosen by log-loss on out-of-sample predictions pooled over the last k inner seasons
    (one season alone is too noisy to estimate it)."""
    k = 3

    def fit_predict(self, train, test):
        d = train.dropna(subset=self.features)
        last = d.season.max()
        vm, y = [], []
        for s in range(last - self.k + 1, last + 1):
            v = d[d.season == s]
            vm.append(self._fit(d[d.season < s]).predict(v[self.features]))
            y.append(v.home_win.values)
        vm, y = np.concatenate(vm), np.concatenate(y)
        s = minimize_scalar(lambda s: -np.mean(y * norm.logcdf(vm / s) + (1 - y) * norm.logcdf(-vm / s)),
                            bounds=(6, 25), method="bounded").x
        margin = self._fit(d).predict(test[self.features].fillna(0))
        return pd.DataFrame({"p_home": norm.cdf(margin / s), "pred_margin": margin, "pred_total": np.nan}, index=test.index)


def no_injury_info(df):
    """What the model sees live without an injury report: everyone on the roster counts as available."""
    df = df.copy()
    for side in ("h", "a"):
        df[f"{side}_available_strength"] = df[f"{side}_full_strength"]
        df[f"{side}_missing_strength"] = 0.0
    df["diff_available_strength"] = df.h_available_strength - df.a_available_strength
    df["diff_missing_strength"] = 0.0
    return df


def live_like(features, train_blind=False):
    def fp(train, test):
        tr = no_injury_info(train) if train_blind else train
        return P.MarginTotalModel(features).fit(tr).predict(no_injury_info(test))
    return fp


# ---- extra features ----------------------------------------------------------------------------

def add_team_hca(m, team_games, half_life=60):
    """Each team's own home edge: EWM of its margin at home minus EWM of its margin away (prior games)."""
    tg = team_games.sort_values(["team_id", "date"])
    g = tg.groupby("team_id")["margin"]
    home = tg["margin"].where(tg.home == 1).groupby(tg.team_id).transform(
        lambda s: s.shift(1).ewm(halflife=half_life, ignore_na=True).mean())
    away = tg["margin"].where(tg.home == 0).groupby(tg.team_id).transform(
        lambda s: s.shift(1).ewm(halflife=half_life, ignore_na=True).mean())
    # shift(1) on the masked series still lets a game see the previous *same-venue* game only: leak-free
    tg = tg.assign(team_hca=(home.ffill() - away.ffill()))
    tg["team_hca"] = tg.groupby("team_id")["team_hca"].transform(lambda s: s.shift(1).ffill())
    x = m.merge(tg[["game_id", "team_id", "team_hca"]].rename(columns={"team_id": "home_id", "team_hca": "h_team_hca"}),
                on=["game_id", "home_id"], how="left")
    x = x.merge(tg[["game_id", "team_id", "team_hca"]].rename(columns={"team_id": "away_id", "team_hca": "a_team_hca"}),
                on=["game_id", "away_id"], how="left")
    return x.fillna({"h_team_hca": 0, "a_team_hca": 0})


def with_player_params(m, pg, top_n=8, half_life=20, star_n=3):
    """Recompute player features with different settings; also add a top-`star_n` 'star power' diff."""
    PL.TOP_N, PL.HALF_LIFE, PL._ALPHA = top_n, half_life, 1 - 0.5 ** (1 / half_life)
    feats, tracker = PL.player_features(pg)
    PL.TOP_N, PL.HALF_LIFE, PL._ALPHA = 8, 20, 1 - 0.5 ** (1 / 20)

    # star power: top-k available ratings, recomputed from the same pre-game ratings is expensive, so use a
    # second pass with TOP_N = star_n
    PL.TOP_N, PL.HALF_LIFE, PL._ALPHA = star_n, half_life, 1 - 0.5 ** (1 / half_life)
    stars, _ = PL.player_features(pg)
    PL.TOP_N, PL.HALF_LIFE, PL._ALPHA = 8, 20, 1 - 0.5 ** (1 / 20)
    feats["star_available"] = stars["available_strength"].values

    x = m.drop(columns=[c for c in m.columns if any(k in c for k in ("full_strength", "available_strength", "missing_strength"))])
    for side, idc in (("h", "home_id"), ("a", "away_id")):
        f = feats.rename(columns={"team_id": idc, **{c: f"{side}_{c}" for c in feats.columns if c not in ("game_id", "team_id")}})
        x = x.merge(f, on=["game_id", idc], how="left")
    for c in ("full_strength", "available_strength", "missing_strength", "star_available"):
        x[f"diff_{c}"] = x[f"h_{c}"] - x[f"a_{c}"]
        x[f"sum_{c}"] = x[f"h_{c}"] + x[f"a_{c}"]
    return x


if __name__ == "__main__":
    m = load_matrix()
    tg, pg = load("team_games"), load("player_games")
    base_scores, base_wf = score(m, P.margin_total(ALL))
    p_ll = base_wf.assign(p=base_wf.p_home.clip(1e-4, 1 - 1e-4))
    base_wf["ll"] = -(p_ll.home_win * np.log(p_ll.p) + (1 - p_ll.home_win) * np.log(1 - p_ll.p))
    base_wf["hit"] = ((base_wf.p_home > 0.5) == base_wf.home_win).astype(float)
    print("BASELINE (production):", {k: round(v, 4) for k, v in base_scores.items()}, "\n")

    bubble = lambda d: (d.season == 2019) & (d.date >= "2020-07-30")
    print("== 1. Home-court advantage ==")
    print(run(m, {
        "same config via harness": Variant(ALL),
        "drop bubble games": Variant(ALL, drop=bubble),
        "drop bubble + 2020-21 (few fans)": Variant(ALL, drop=lambda d: bubble(d) | (d.season == 2020)),
        "intercept = last 2 seasons": Variant(ALL, recent_hca=2),
        "intercept = last 3 seasons": Variant(ALL, recent_hca=3),
        "no absolute away travel": Variant([f for f in ALL if f != "a_travel_km"]),
    }, base_wf).round(4).to_string())
    mh = add_team_hca(m, tg)
    print(run(mh, {"+ team-specific home edge (hl 60)": Variant(ALL + ["h_team_hca", "a_team_hca"])}, base_wf).round(4).to_string())

    print("\n== 2. Margin -> probability ==")
    print(run(m, {
        "sigma by log-loss": Variant(ALL, prob="sigma_mle"),
        "platt (logistic on margin)": Variant(ALL, prob="platt"),
    }, base_wf).round(4).to_string())

    print("\n== 3. Regularisation / training window (with sigma by log-loss) ==")
    print(run(m, {
        **{f"alpha {a}": Variant(ALL, alpha=a, prob="sigma_mle") for a in (1, 30, 100, 300)},
        "train from 2018": Variant(ALL, prob="sigma_mle", drop=lambda d: d.season < 2018),
        "recency weight 0.85/season": Variant(ALL, prob="sigma_mle", weights=lambda d: 0.85 ** (d.season.max() - d.season)),
    }, base_wf).round(4).to_string())

    print("\n== 4. Player-availability settings ==")
    star = ALL + ["diff_star_available"]
    rows = []
    for top_n, hl in [(8, 20), (6, 20), (10, 20), (5, 20), (8, 10), (8, 30), (8, 40)]:
        mp = with_player_params(m, pg, top_n, hl)
        rows.append(run(mp, {f"top{top_n} hl{hl}": Variant(ALL), f"top{top_n} hl{hl} + star top3": Variant(star)}, base_wf))
    print(pd.concat(rows).round(4).to_string())
    ms = with_player_params(m, pg, 8, 20)
    print(run(ms, {"sigma pooled 3 seasons": PooledSigma(ALL),
                   "CANDIDATE: star top3 + sigma pooled": PooledSigma(star)}, base_wf).round(4).to_string())

    print("\n== 5. Model family (with sigma by log-loss where applicable) ==")
    print(run(m, {
        "logistic (direct)": P.logistic(ALL),
        "ridge + logistic avg": P.ensemble(Variant(ALL, prob="sigma_mle").fit_predict, P.logistic(ALL)),
        "lgbm margin": P.margin_total(ALL, "lgbm", "ridge"),
    }, base_wf).round(4).to_string())

    print("\n== 6. Live conditions: no injury report at prediction time ==")
    print(run(m, {"train real lineups, predict blind (= app today)": live_like(ALL),
                  "train blind, predict blind": live_like(ALL, train_blind=True),
                  "no player features": P.margin_total([f for f in ALL if "strength" not in f])}, base_wf).round(4).to_string())
