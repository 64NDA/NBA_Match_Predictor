import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import log_loss

from common import HOME, table

st.title("Track record")
GRAY = "rgba(128,128,128,0.7)"
ll = lambda y, p: log_loss(y, np.clip(p, 1e-4, 1 - 1e-4), labels=[0, 1])
acc = lambda y, p: ((p > 0.5) == y.astype(bool)).mean()

# ---- live log ---------------------------------------------------------------------------------
log = table("prediction_log")
done = log[log["home_win"].notna()] if len(log) else log
st.subheader("This season (live)")
if len(done):
    y = done["home_win"]
    a, b, c, d = st.columns(4)
    a.metric("Games", len(done))
    b.metric("v3 accuracy", f"{acc(y, done.p_home):.1%}", delta=f"{(acc(y, done.p_home) - acc(y, done.elo_prob)) * 100:+.1f} pts vs Elo")
    if "p_home_v2" in done and done["p_home_v2"].notna().any():
        c.metric("v2 accuracy", f"{acc(y, done.p_home_v2):.1%}")
    d.metric("v3 log-loss", f"{ll(y, done.p_home):.3f}")
    st.dataframe(done.sort_values("date", ascending=False)[
        ["date", "away_team", "home_team", "p_home", "p_home_v2", "elo_prob", "pred_away_pts", "pred_home_pts",
         "away_pts", "home_pts"]], use_container_width=True, hide_index=True)
else:
    st.info("No completed games logged yet. Predictions are logged *before* tip-off by the daily update (using "
            "the injury report available then) and scored here once the results are in.")

# ---- backtest ---------------------------------------------------------------------------------
v3, v2 = table("backtest_v3"), table("backtest")
bt = v3.merge(v2[["game_id", "p_home"]].rename(columns={"p_home": "p_v2"}), on="game_id", how="left")
st.subheader("Backtest (out-of-sample)")
st.caption("Every game was predicted by a model trained only on earlier seasons, using the injury report as it stood "
           "an hour before tip-off. 2021-25 were used to choose the model; **2025-26 was held out and evaluated once**.")


def summarise(d):
    y = d["home_win"]
    return pd.Series({"Games": len(d), "v3 acc": acc(y, d.p_home), "v2 acc": acc(y, d.p_v2), "Elo acc": acc(y, d.elo_prob),
                      "Home-team acc": y.mean(), "v3 log-loss": ll(y, d.p_home), "v2 log-loss": ll(y, d.p_v2),
                      "v3 margin MAE": (d.pred_margin - d.margin).abs().mean()})


per = bt.dropna(subset=["p_v2"]).groupby("season").apply(summarise)
per.index = [f"{s}-{str(s + 1)[-2:]}" + (" (holdout)" if s == bt.season.max() else "") for s in per.index]
st.dataframe(per, use_container_width=True, column_config={
    "Games": st.column_config.NumberColumn(format="%d"),
    **{c: st.column_config.NumberColumn(format="percent") for c in ["v3 acc", "v2 acc", "Elo acc", "Home-team acc"]},
    **{c: st.column_config.NumberColumn(format="%.3f") for c in ["v3 log-loss", "v2 log-loss"]},
    "v3 margin MAE": st.column_config.NumberColumn(format="%.1f pts")})

left, right = st.columns(2)
with left:
    st.markdown("##### Calibration")
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot", width=1), name="perfect", hoverinfo="skip")
    bins = np.linspace(0, 1, 11)
    for col, name, color, width in (("p_v2", "v2", GRAY, 1.5), ("p_home", "v3", HOME, 2.5)):
        cal = bt.groupby(pd.cut(bt[col], bins), observed=True).agg(pred=(col, "mean"), actual=("home_win", "mean"), n=("home_win", "size"))
        fig.add_scatter(x=cal.pred, y=cal.actual, mode="lines+markers", line=dict(color=color, width=width), name=name,
                        marker=dict(size=8 if name == "v3" else 6, line=dict(width=2, color="white")), customdata=cal.n,
                        hovertemplate=name + ": predicted %{x:.0%} → won %{y:.0%} (%{customdata} games)<extra></extra>")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1),
                      xaxis=dict(title="predicted home win probability", tickformat=".0%", range=[0, 1], gridcolor="rgba(128,128,128,0.15)"),
                      yaxis=dict(title="actual home win rate", tickformat=".0%", range=[0, 1], gridcolor="rgba(128,128,128,0.15)"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("On the dotted line, '70%' really wins 70% of the time.")

with right:
    st.markdown("##### Accuracy by confidence (v3)")
    conf = np.maximum(bt.p_home, 1 - bt.p_home)
    buckets = pd.cut(conf, [0.5, 0.6, 0.7, 0.8, 1.0], labels=["50–60%", "60–70%", "70–80%", "80%+"], include_lowest=True)
    correct = (bt.p_home > .5) == bt.home_win
    by = pd.DataFrame({"bucket": buckets, "correct": correct}).groupby("bucket", observed=True)["correct"].agg(["mean", "size"])
    fig = go.Figure(go.Bar(x=by.index.astype(str), y=by["mean"], marker=dict(color=HOME, cornerradius=4),
                           text=[f"{v:.0%}" for v in by["mean"]], textposition="outside", customdata=by["size"],
                           hovertemplate="%{x} confident: %{y:.1%} correct (%{customdata} games)<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), bargap=0.4,
                      yaxis=dict(title="picks correct", tickformat=".0%", range=[0, 1.05], gridcolor="rgba(128,128,128,0.15)"),
                      xaxis=dict(title="model confidence in its pick"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"{by['size'].sum():,} games; bucket sizes in the tooltip.")
