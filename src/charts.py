"""Altair charts for the Streamlit demo. Colours come from a fixed
categorical order so each entity keeps its colour on every chart."""
import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
from sklearn.metrics import roc_curve

from src.build_dataset import CONGESTION_THRESHOLD_H

BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9893"
PORT_ORDER = ["Houston", "LA_LongBeach", "NY_NJ", "Savannah"]
PORT_COLORS = alt.Scale(domain=PORT_ORDER, range=[BLUE, ORANGE, AQUA, YELLOW])
MODEL_ORDER = ["LightGBM ensemble (ours)", "Logistic regression",
               "Baseline: port historical rate", "Baseline: rule on same-type queue per berth"]
MODEL_COLORS = alt.Scale(domain=MODEL_ORDER, range=[BLUE, ORANGE, AQUA, YELLOW])
TEST_START = pd.Timestamp("2023-03-01")


def ship_map(sample: pd.DataFrame, port: str) -> pdk.Deck:
    d = sample[sample["port"] == port].copy()
    d["color"] = d["state"].map({"At anchor (waiting)": [235, 104, 52, 170],
                                 "At berth (moored)": [42, 120, 214, 170]})
    d["time"] = d["BaseDateTime"].dt.strftime("%d %b %Y %H:%M")
    d["VesselName"] = d["VesselName"].fillna("unknown")
    layer = pdk.Layer("ScatterplotLayer", d, get_position=["LON", "LAT"], get_fill_color="color",
                      get_radius=120, radius_min_pixels=2, radius_max_pixels=6, pickable=True)
    view = pdk.ViewState(latitude=d["LAT"].median(), longitude=d["LON"].median(), zoom=9.2)
    return pdk.Deck(layers=[layer], initial_view_state=view, map_style=None,
                    tooltip={"text": "{VesselName}\n{state}\n{time} UTC"})


def _test_band():
    band = pd.DataFrame({"start": [TEST_START], "end": [pd.Timestamp("2023-03-31")],
                         "label": ["March = test period (unseen)"]})
    rect = alt.Chart(band).mark_rect(color=GREY, opacity=0.12).encode(x="start:T", x2="end:T")
    text = alt.Chart(band).mark_text(align="left", baseline="top", dx=4, dy=4, color="#52514e") \
        .encode(x="start:T", y=alt.value(0), text="label")
    return rect + text


def queue_over_time(state: pd.DataFrame) -> alt.Chart:
    daily = state.assign(day=state["hour"].dt.floor("D")) \
        .groupby(["port", "day"], as_index=False)["queue_anchored"].mean()
    hover = alt.selection_point(fields=["day"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(daily).encode(x=alt.X("day:T", title=None))
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("queue_anchored:Q", title="Ships waiting at anchor (daily avg)"),
        color=alt.Color("port:N", scale=PORT_COLORS, title="Port"))
    points = base.mark_point(size=60, filled=True).encode(
        y="queue_anchored:Q", color=alt.Color("port:N", scale=PORT_COLORS),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    rule = base.mark_rule(color=GREY).encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("day:T", title="Day")] +
                [alt.Tooltip(f"{p}:Q", title=p, format=".1f") for p in PORT_ORDER]
    ).transform_pivot("port", value="queue_anchored", groupby=["day"]).add_params(hover)
    return (_test_band() + lines + points + rule).properties(height=320)


WAIT_BUCKETS = ["0 h (no wait)", "0-12 h", "12-24 h", "24-48 h", "48-96 h", "96 h+"]


def wait_distribution(calls: pd.DataFrame) -> alt.Chart:
    bins = [-0.01, 0.01, 12, CONGESTION_THRESHOLD_H, 48, 96, np.inf]
    d = calls.assign(bucket=pd.cut(calls["anchor_wait_h"], bins, labels=WAIT_BUCKETS, right=False))
    d = d.groupby(["port", "bucket"], observed=False).size().rename("n").reset_index()
    d["share"] = d["n"] / d.groupby("port")["n"].transform("sum")
    d["label"] = np.where(d["bucket"].isin(WAIT_BUCKETS[3:]), "Congested (24 h+)", "Not congested")
    return alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("bucket:N", sort=WAIT_BUCKETS, title="Anchor wait before berthing", axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="Share of port calls"),
        color=alt.Color("label:N", scale=alt.Scale(domain=["Not congested", "Congested (24 h+)"],
                                                   range=[GREY, ORANGE]), title=None),
        tooltip=["port", alt.Tooltip("bucket:N", title="Wait"), alt.Tooltip("n:Q", title="Calls"),
                 alt.Tooltip("share:Q", format=".0%", title="Share")],
    ).properties(width=170, height=200).facet(
        column=alt.Column("port:N", sort=PORT_ORDER, title=None)
    ).configure_legend(orient="top")


def congestion_by_type(calls: pd.DataFrame) -> alt.Chart:
    d = calls.groupby(["port", "vessel_group"]).agg(rate=("congested", "mean"), n=("congested", "size")).reset_index()
    return alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("port:N", sort=PORT_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("vessel_group:N", sort=["cargo", "tanker"], scale=alt.Scale(paddingInner=0.04)),
        y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title="Share congested (24 h+ wait)"),
        color=alt.Color("vessel_group:N", scale=alt.Scale(domain=["cargo", "tanker"], range=[BLUE, AQUA]),
                        title="Vessel type", legend=alt.Legend(orient="top")),
        tooltip=["port", "vessel_group", alt.Tooltip("rate:Q", format=".0%", title="Congested"),
                 alt.Tooltip("n:Q", title="Calls")],
    ).properties(height=280)


def roc_chart(test: pd.DataFrame, metrics: dict) -> alt.Chart:
    rows = []
    for name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(test["congested"], test[f"score: {name}"])
        auc = metrics["models"][name]["roc_auc"]
        rows.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "model": name, "auc": auc}))
    d = pd.concat(rows)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=GREY) \
        .encode(x="x:Q", y="y:Q")
    lines = alt.Chart(d).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("fpr:Q", title="False alarm rate (non-congested ships flagged)",
                axis=alt.Axis(format="%", tickCount=5)),
        y=alt.Y("tpr:Q", title="Catch rate (congested ships flagged)", axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color("model:N", scale=MODEL_COLORS, title=None,
                        legend=alt.Legend(orient="bottom", columns=2, labelLimit=320)),
        tooltip=["model", alt.Tooltip("auc:Q", title="ROC-AUC", format=".3f"),
                 alt.Tooltip("fpr:Q", format=".2f", title="False alarms"),
                 alt.Tooltip("tpr:Q", format=".2f", title="Caught")])
    return (diag + lines).properties(height=360)


def confusion_chart(test: pd.DataFrame, threshold: float) -> alt.Chart:
    pred = np.where(test["probability"] >= threshold, "Flagged", "Not flagged")
    actual = np.where(test["congested"] == 1, "Congested", "Not congested")
    d = pd.DataFrame({"Predicted": pred, "Actual": actual}).value_counts().rename("n").reset_index()
    d["outcome"] = np.select(
        [(d["Predicted"] == "Flagged") & (d["Actual"] == "Congested"),
         (d["Predicted"] == "Not flagged") & (d["Actual"] == "Not congested"),
         d["Predicted"] == "Flagged"],
        ["Caught delay", "Correctly clear", "False alarm"], "Missed delay")
    base = alt.Chart(d).encode(
        x=alt.X("Predicted:N", sort=["Flagged", "Not flagged"], title="Model said",
                axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("Actual:N", sort=["Congested", "Not congested"], title="What happened"))
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("n:Q", scale=alt.Scale(range=["#e8f0fb", BLUE]), legend=None),
        tooltip=["outcome", alt.Tooltip("n:Q", title="Ships")])
    text = base.mark_text(fontSize=15, fontWeight="bold").encode(
        text=alt.Text("label:N"),
        color=alt.condition(alt.datum.n > d["n"].max() * 0.55, alt.value("white"), alt.value("#0b0b0b"))
    ).transform_calculate(label="datum.outcome + ': ' + datum.n")
    return (cells + text).properties(height=260)


def confidence_chart(metrics: dict) -> alt.Chart:
    d = pd.DataFrame(metrics["confidence_bands"]).T.reset_index(names="band")
    bars = alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=BLUE).encode(
        x=alt.X("band:N", sort=["High", "Medium", "Low"], title="Confidence band", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Accuracy on March"),
        tooltip=["band", alt.Tooltip("accuracy:Q", format=".1%"), alt.Tooltip("n:Q", title="Ships")])
    labels = bars.mark_text(dy=-8, color="#0b0b0b").encode(text=alt.Text("accuracy:Q", format=".0%"))
    return (bars + labels).properties(height=260)


def timeline_chart(test: pd.DataFrame) -> alt.Chart:
    d = test.assign(day=pd.to_datetime(test["arrival_time"]).dt.floor("D"))
    d = d.groupby(["port", "day"]).agg(actual=("congested", "mean"), predicted=("probability", "mean"),
                                       n=("congested", "size")).reset_index()
    d = d.melt(id_vars=["port", "day", "n"], value_vars=["actual", "predicted"], var_name="series", value_name="share")
    d["series"] = d["series"].map({"actual": "Actual share congested", "predicted": "Avg predicted risk"})
    return alt.Chart(d).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30)).encode(
        x=alt.X("day:T", title=None, axis=alt.Axis(format="%d %b")),
        y=alt.Y("share:Q", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Actual share congested", "Avg predicted risk"],
                                                    range=[ORANGE, BLUE]), title=None,
                        legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["Actual share congested", "Avg predicted risk"],
                                                              range=[[1, 0], [5, 3]]), legend=None),
        tooltip=["port", alt.Tooltip("day:T", format="%d %b"), "series",
                 alt.Tooltip("share:Q", format=".0%"), alt.Tooltip("n:Q", title="Arrivals that day")],
    ).properties(width=360, height=170).facet(
        facet=alt.Facet("port:N", sort=PORT_ORDER, title=None), columns=2)


def importance_chart(metrics: dict, feature_text: dict) -> alt.Chart:
    s = pd.Series(metrics["shap_importance"]).sort_values(ascending=False).head(12)
    d = pd.DataFrame({"feature": [feature_text.get(f, (f,))[0] for f in s.index], "impact": s.values})
    return alt.Chart(d).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=BLUE).encode(
        x=alt.X("impact:Q", title="Average influence on predictions (mean |SHAP|)"),
        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=["feature", alt.Tooltip("impact:Q", format=".3f")]).properties(height=360)


def calibration_chart(test: pd.DataFrame) -> alt.Chart:
    parts = []
    for col, name in [("probability_raw", "Before recalibration"), ("probability", "After Platt recalibration")]:
        q = pd.qcut(test[col], 8, duplicates="drop")
        g = test.groupby(q, observed=True).agg(predicted=(col, "mean"), actual=("congested", "mean"),
                                               n=("congested", "size")).reset_index(drop=True)
        parts.append(g.assign(series=name))
    d = pd.concat(parts)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=GREY)         .encode(x="x:Q", y="y:Q")
    domain = ["Before recalibration", "After Platt recalibration"]
    line = alt.Chart(d).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60)).encode(
        x=alt.X("predicted:Q", title="Predicted risk", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("actual:Q", title="Actual share congested", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=[ORANGE, BLUE]), title=None,
                        legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=[[5, 3], [1, 0]]), legend=None),
        tooltip=["series", alt.Tooltip("predicted:Q", format=".0%"), alt.Tooltip("actual:Q", format=".0%"),
                 alt.Tooltip("n:Q", title="Ships")])
    return (diag + line).properties(height=360)


def interaction_chart(grid: pd.DataFrame) -> alt.Chart:
    """Heat map of average predicted risk over queue x upcoming gusts."""
    d = grid.assign(queue=grid["queue_per_berth_same_type"].map("{:.2f}".format),
                    gust=grid["max_gust_next_24h"].astype(int).astype(str) + " km/h")
    gust_order = [f"{g} km/h" for g in sorted(grid["max_gust_next_24h"].astype(int).unique())]
    queue_order = sorted(d["queue"].unique(), key=float, reverse=True)
    base = alt.Chart(d).encode(
        x=alt.X("gust:N", sort=gust_order, title="Strongest gust in next 24 h", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("queue:N", sort=queue_order, title="Same-type queue per berth"))
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("risk:Q", scale=alt.Scale(range=["#e8f0fb", BLUE]), legend=alt.Legend(format="%", title="Avg risk")),
        tooltip=["queue", "gust", alt.Tooltip("risk:Q", format=".1%")])
    text = base.mark_text(fontSize=12).encode(
        text=alt.Text("risk:Q", format=".0%"),
        color=alt.condition(alt.datum.risk > grid["risk"].quantile(0.6), alt.value("white"), alt.value("#0b0b0b")))
    return (cells + text).properties(height=300)


def conformal_chart(test: pd.DataFrame) -> alt.Chart:
    """How often each conformal prediction set occurs and how often it was right."""
    order = ["delay", "no delay", "either"]
    covered = np.where(test["prediction_set"] == "either", True,
                       np.where(test["prediction_set"] == "delay", test["congested"] == 1, test["congested"] == 0))
    d = test.assign(covered=covered).groupby("prediction_set").agg(
        n=("covered", "size"), right=("covered", "mean")).reindex(order).dropna().reset_index()
    d["label"] = d.apply(lambda r: f"{int(r.n)} ships, {r.right:.0%} right", axis=1)
    bars = alt.Chart(d).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=BLUE).encode(
        x=alt.X("n:Q", title="Ships in March"),
        y=alt.Y("prediction_set:N", sort=order, title="Prediction set"),
        tooltip=["prediction_set", "n", alt.Tooltip("right:Q", format=".0%", title="Contains true outcome")])
    text = bars.mark_text(align="left", dx=4, color="#0b0b0b").encode(text="label:N")
    return (bars + text).properties(height=180)
