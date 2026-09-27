"""Reproduce the notebook pipeline and test suspected issues."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss

raw = pd.read_csv("legacy/2024nbamatches.csv")
raw["Date"] = pd.to_datetime(raw["Date"])
raw["game_id"] = range(len(raw))

STATS = {"TS PCT": "true_shooting", "EFG PCT": "effective_field_goal", "3P AR": "3pt_attempted",
         "FT AR": "free_throws_attempted", "TRB PCT": "rebounds", "AST PCT": "assists", "STL PCT": "steals",
         "BLK PCT": "blocks", "TO PCT": "turnovers", "OFF RT": "offensive_rating", "DEF RT": "defensive_rating"}

def side(home):
    me, op = ("2", "1") if home else ("1", "2")
    d = pd.DataFrame({"game_id": raw.game_id, "date": raw.Date, "arena": raw.Arena,
                      "team": raw["Home/Neutral" if home else "Visitor/Neutral"],
                      "opponent": raw["Visitor/Neutral" if home else "Home/Neutral"], "home": int(home),
                      "points": raw["H_PTS" if home else "V_PTS"], "opponent_points": raw["V_PTS" if home else "H_PTS"]})
    for k, v in STATS.items():
        d[v] = raw[f"Team {me} {k}"]; d[f"opponent_{v}"] = raw[f"Team {op} {k}"]
    return d

m = pd.concat([side(True), side(False)], ignore_index=True)
m["win"] = (m.points > m.opponent_points).astype(int)
for c in ["team", "opponent", "arena"]:
    m[f"{c}_code"] = m[c].astype("category").cat.codes
cols = ["points", "opponent_points"] + [x for v in STATS.values() for x in (v, f"opponent_{v}")]

# Exponentially weighted mean of PRIOR games, per team (same as notebook's decay=0.1, but vectorised)
m = m.sort_values(["team", "date"])
alpha = 1 - np.exp(-0.1)
w = m.groupby("team")[cols].transform(lambda s: s.shift(1).ewm(alpha=alpha, adjust=True).mean())
wcols = [f"weighted_{c}" for c in cols]
m[wcols] = w.values
m = m.sort_values(["date", "game_id", "home"]).reset_index(drop=True)

split = "2024-02-01"
tr, te = m[m.date < split], m[m.date >= split]
base = ["team_code", "opponent_code", "arena_code", "home"]
P = base + wcols
print(f"rows train/test: {len(tr)}/{len(te)}  games in test: {te.game_id.nunique()}")

# 0. Baselines
te_home = te[te.home == 1]
print("\n[baseline] always pick home team:", round(te_home.win.mean(), 3))

# 1. Notebook model as written (classifier on scores, per-row eval)
rf = RandomForestClassifier(n_estimators=300, min_samples_split=5, max_depth=10, random_state=1)
rf.fit(tr[P].fillna(-1), tr[["points", "opponent_points"]])
sp = rf.predict(te[P].fillna(-1))
print("\n[notebook] score RMSE w/ RandomForestClassifier:", round(np.sqrt(mean_squared_error(te[["points", "opponent_points"]], sp)), 2))
rf.fit(tr[P].fillna(-1), tr.win)
wp = rf.predict(te[P].fillna(-1))
print("[notebook] win acc (per row):", round(accuracy_score(te.win, wp), 3))
te2 = te.assign(pred=wp)
g = te2.groupby("game_id").pred.sum()
print(f"[notebook] games where model predicts BOTH teams win: {(g==2).mean():.1%}, BOTH lose: {(g==0).mean():.1%}")

# 2. Regressor instead of classifier for scores
rr = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=1, n_jobs=-1)
rr.fit(tr[P].fillna(-1), tr[["points", "opponent_points"]])
print("\n[fix] score RMSE w/ RandomForestRegressor:", round(np.sqrt(mean_squared_error(te[["points", "opponent_points"]], rr.predict(te[P].fillna(-1)))), 2))

# 3. One row per game, home-minus-away differential of each team's form
h = m[m.home == 1].set_index("game_id"); a = m[m.home == 0].set_index("game_id")
G = pd.DataFrame(index=h.index)
G["date"] = h.date; G["home_win"] = h.win
for c in wcols:
    G[f"diff_{c}"] = h[c] - a.loc[h.index, c]
G = G.dropna()
dcols = [c for c in G.columns if c.startswith("diff_")]
gtr, gte = G[G.date < split], G[G.date >= split]
lr = LogisticRegression(C=0.1, max_iter=2000).fit(gtr[dcols], gtr.home_win)
p = lr.predict_proba(gte[dcols])[:, 1]
print("\n[fix] per-game logistic reg on form differentials: acc", round(accuracy_score(gte.home_win, p > .5), 3),
      " log-loss", round(log_loss(gte.home_win, p), 3), f"(games={len(gte)})")

# 4. Simple Elo
elo = {t: 1500 for t in m.team.unique()}; K, HCA = 20, 70; correct = n = 0
for _, r in raw.sort_values("Date").iterrows():
    H, V = r["Home/Neutral"], r["Visitor/Neutral"]
    ph = 1 / (1 + 10 ** ((elo[V] - elo[H] - HCA) / 400)); hw = int(r.H_PTS > r.V_PTS)
    if r.Date >= pd.Timestamp(split): correct += int((ph > .5) == hw); n += 1
    mov = abs(r.H_PTS - r.V_PTS); mult = np.log(mov + 1) * 2.2 / ((elo[H] - elo[V]) * (1 if hw else -1) * .001 + 2.2)
    d = K * mult * (hw - ph); elo[H] += d; elo[V] -= d
print("[baseline] plain Elo (MOV-adjusted) acc:", round(correct / n, 3))
