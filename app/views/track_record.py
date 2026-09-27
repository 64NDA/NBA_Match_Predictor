import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import log_loss

from common import HOME, table

st.title("Track record")

# ---- live log ---------------------------------------------------------------------------------
log = table("prediction_log")
done = log[log["home_win"].notna()] if len(log) else log
st.subheader("This season (live)")
if len(done):
    acc = ((done.p_home > 0.5) == done.home_win.astype(bool)).mean()
    elo_acc = ((done.elo_prob > 0.5) == done.home_win.astype(bool)).mean()
    a, b, c = st.columns(3)
    a.metric("Games", len(done))
    b.metric("Model accuracy", f"{acc:.1%}", delta=f"{(acc - elo_acc) * 100:+.1f} pts vs Elo")
    c.metric("Log-loss", f"{log_loss(done.home_win, done.p_home.clip(1e-4, 1 - 1e-4), labels=[0, 1]):.3f}")
    st.dataframe(done.sort_values("date", ascending=False)[
        ["date", "away_team", "home_team", "p_home", "pred_away_pts", "pred_home_pts", "away_pts", "home_pts"]],
        use_container_width=True, hide_index=True)
else:
    st.info("No completed games logged yet. Predictions are logged *before* tip-off by the daily update, "
            "then scored here once results come in.")

# ---- backtest ---------------------------------------------------------------------------------
bt = table("backtest")
st.subheader("Backtest (out-of-sample)")
st.caption("Every game below was predicted by a model trained only on earlier seasons. 2021-25 were used for "
           "model selection; **2025-26 was held out and evaluated once**.")


def summarise(d):
    y = d.home_win
    return pd.Series({
        "Games": len(d),
        "Model acc": ((d.p_home > .5) == y).mean(), "Elo acc": ((d.elo_prob > .5) == y).mean(),
        "Home-team acc": y.mean(),
        "Model log-loss": log_loss(y, d.p_home.clip(1e-4, 1 - 1e-4), labels=[0, 1]),
        "Elo log-loss": log_loss(y, d.elo_prob.clip(1e-4, 1 - 1e-4), labels=[0, 1]),
        "Margin MAE": (d.pred_margin - d.margin).abs().mean(),
    })


per = bt.groupby("season").apply(summarise)
per.index = [f"{s}-{str(s + 1)[-2:]}" + (" (holdout)" if s == bt.season.max() else "") for s in per.index]
st.dataframe(per, use_container_width=True, column_config={
    "Games": st.column_config.NumberColumn(format="%d"),
    **{c: st.column_config.NumberColumn(format="percent") for c in ["Model acc", "Elo acc", "Home-team acc"]},
    **{c: st.column_config.NumberColumn(format="%.3f") for c in ["Model log-loss", "Elo log-loss"]},
    "Margin MAE": st.column_config.NumberColumn(format="%.1f pts")})

left, right = st.columns(2)

# calibration: predicted vs actual home win rate, 10 bins
with left:
    st.markdown("##### Calibration")
    bins = pd.cut(bt.p_home, np.linspace(0, 1, 11))
    cal = bt.groupby(bins, observed=True).agg(pred=("p_home", "mean"), actual=("home_win", "mean"), n=("home_win", "size"))
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="rgba(128,128,128,0.5)", dash="dot", width=1),
                    name="perfect", hoverinfo="skip")
    fig.add_scatter(x=cal.pred, y=cal.actual, mode="lines+markers", line=dict(color=HOME, width=2),
                    marker=dict(size=9, line=dict(width=2, color="white")), name="model",
                    customdata=cal.n, hovertemplate="predicted %{x:.0%} → won %{y:.0%} (%{customdata} games)<extra></extra>")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      xaxis=dict(title="predicted home win probability", tickformat=".0%", range=[0, 1],
                                 gridcolor="rgba(128,128,128,0.15)"),
                      yaxis=dict(title="actual home win rate", tickformat=".0%", range=[0, 1],
                                 gridcolor="rgba(128,128,128,0.15)"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Points on the dotted line mean '70%' really wins 70% of the time.")

# accuracy by confidence
with right:
    st.markdown("##### Accuracy by confidence")
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
