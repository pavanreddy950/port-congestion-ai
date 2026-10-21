"""Streamlit demo: port congestion risk with confidence and explanations.

Run:  streamlit run app.py
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import charts
from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.model import FEATURE_TEXT, explain_row, recommend, summarise

st.set_page_config(page_title="Port Congestion Risk", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(MODELS_DIR / "congestion_model.joblib")
    test = pd.read_csv(REPORTS_DIR / "test_predictions.csv", parse_dates=["arrival_time", "berth_time"])
    with open(REPORTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    calls = pd.read_csv(PROCESSED_DIR / "port_calls.csv", parse_dates=["arrival_time", "berth_time"])
    state = pd.read_parquet(PROCESSED_DIR / "port_hourly_state.parquet")
    ship_sample = pd.read_parquet(PROCESSED_DIR / "map_sample.parquet")
    interaction = pd.read_csv(REPORTS_DIR / "interaction_queue_weather.csv")
    return model, test, metrics, calls, state, ship_sample, interaction


model, test, metrics, calls, state, ship_sample, interaction = load()
SET_TEXT = {"delay": "Delay (no-delay ruled out)", "no delay": "No delay (delay ruled out)",
            "either": "Either outcome possible"}


def show_prediction(row: pd.DataFrame, actual: pd.Series | None = None):
    pred = model.predict(row).iloc[0]
    shap_row = model.shap_values(row).iloc[0]
    drivers = explain_row(row.iloc[0], shap_row)

    c1, c2, c3, c4 = st.columns(4)
    flagged = "FLAGGED" if pred.probability >= model.threshold else "not flagged"
    c1.metric("Probability of 24 h+ anchor wait", f"{pred.probability:.0%}",
              f"{flagged} (alert threshold {model.threshold:.0%})", delta_color="off")
    c2.metric("Confidence", f"{int(pred.confidence * 100)}% ({pred.confidence_band})")
    c3.metric("90% prediction set", SET_TEXT.get(pred.prediction_set, pred.prediction_set),
              help="Conformal prediction: the outcomes the model cannot rule out, with a 90% coverage target. "
                   "It is stricter than the alert, which is tuned to catch most delays, so a ship just above "
                   "the alert line can still have 'no delay' as its only outcome.")
    if actual is not None:
        c4.metric("What actually happened", f"{actual.anchor_wait_h:.1f} h wait",
                  "congested" if actual.congested else "not congested",
                  delta_color="inverse" if actual.congested else "normal")

    st.success(summarise(pred.probability, drivers, model.threshold))
    st.subheader("Why")
    for d in drivers:
        arrow = "🔺" if d["direction"] == "raises" else "🔻"
        st.write(f"{arrow} {d['text']}  (impact {d['impact']:+.2f})")

    st.subheader("Suggested action")
    st.info(recommend(pred.probability, pred.confidence_band, drivers, model.threshold))

    fig, ax = plt.subplots(figsize=(6, 3))
    top = shap_row.reindex(shap_row.abs().sort_values().index[-8:])
    ax.barh([FEATURE_TEXT[f][0] for f in top.index], top.values,
            color=["#c0504d" if v > 0 else "#3b6ea5" for v in top.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("contribution to risk (log-odds)")
    st.pyplot(fig)


st.title("🚢 Port Congestion & Delay Risk")
st.caption("Real AIS vessel data (MarineCadastre, Jan-Mar 2023) for LA/Long Beach, Houston, NY/NJ and Savannah. "
           "Congested = ship waits 24 h or more at anchor before getting a berth.")

tab_overview, tab_replay, tab_whatif, tab_eval, tab_ai = st.tabs(
    ["Overview: the data", "Replay a real arrival", "What-if scenario", "Model evaluation", "🤖 AI Port Analyst"])

with tab_overview:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AIS positions analysed", "11.3 M")
    k2.metric("Cargo & tanker vessels", f"{calls['MMSI'].nunique():,}")
    k3.metric("Port calls (Jan-Mar 2023)", f"{len(calls):,}")
    k4.metric("Congested (24 h+ at anchor)", f"{calls['congested'].mean():.0%}")

    st.subheader("Where ships wait vs where they berth")
    st.caption("Each dot is a real AIS position. Orange = anchored and waiting offshore, blue = moored at a terminal. "
               "The time between the two is the anchor wait we predict. Hover a dot for the vessel.")
    map_port = st.radio("Port", charts.PORT_ORDER, horizontal=True, key="map_port")
    st.pydeck_chart(charts.ship_map(ship_sample, map_port), height=430)

    st.subheader("Anchorage queue over time")
    st.caption("Daily average number of ships waiting at anchor. The model learns from Jan-Feb and is tested on March.")
    st.altair_chart(charts.queue_over_time(state), width="stretch")

    st.subheader("How long do ships wait at each port?")
    st.caption("Houston queues for days, NY/NJ almost never. Bars in orange are what we label 'congested'.")
    st.altair_chart(charts.wait_distribution(calls))

    st.subheader("Congestion depends on vessel type, not just the port")
    st.caption("Tankers and container ships use different berths. That is why the model counts the queue "
               "of same-type ships.")
    st.altair_chart(charts.congestion_by_type(calls), width="stretch")

with tab_replay:
    st.write("Pick a real ship arrival from the March 2023 test period (the model never saw these).")
    port = st.selectbox("Port", sorted(test["port"].unique()))
    sub = test[test["port"] == port].sort_values("arrival_time")
    labels = sub.apply(lambda r: f"{r.arrival_time:%d %b %H:%M} | MMSI {r.MMSI} | {r.vessel_group} | "
                                 f"actual wait {r.anchor_wait_h:.0f} h", axis=1)
    choice = st.selectbox("Arrival", labels.index, format_func=lambda i: labels[i])
    show_prediction(sub.loc[[choice]], sub.loc[choice])

with tab_whatif:
    st.write("Start from a real arrival and change the conditions to see how the risk responds.")
    base = test.sample(1, random_state=1).copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        base["port"] = st.selectbox("Port ", sorted(test["port"].unique()))
        base["vessel_group"] = st.radio("Vessel type", ["cargo", "tanker"], horizontal=True)
        rows = test[(test["port"] == base["port"].iloc[0]) & (test["vessel_group"] == base["vessel_group"].iloc[0])]
        if rows.empty:
            rows = test[test["port"] == base["port"].iloc[0]]
        # start every other input from this port/type's typical (median) arrival
        for col in base.columns.intersection(rows.select_dtypes("number").columns):
            base[col] = rows[col].median()
        base["length"] = st.slider("Vessel length (m)", 80, 400, int(rows["length"].median()))
        base["draft"] = st.slider("Draft (m)", 4.0, 20.0, float(round(rows["draft"].median(), 1)))
    with c2:
        st.caption(f"Counts are for {base['vessel_group'].iloc[0]} ships, which compete for the same berths.")
        q = st.slider("Same-type ships waiting at anchor", 0, int(test["queue_same_type"].max()) + 5,
                      int(rows["queue_same_type"].median()))
        b = st.slider("Same-type ships at berth", 0, int(test["berths_same_type"].max()) + 5,
                      int(rows["berths_same_type"].median()))
        type_cap = max((rows["berths_same_type"] / rows["occupancy_same_type"]).median(), 1)
        base["queue_same_type"], base["berths_same_type"] = q, b
        base["queue_per_berth_same_type"] = q / type_cap
        base["occupancy_same_type"] = b / type_cap
        base["recent_congested_share_7d_same_type"] = st.slider(
            "Share of same-type ships delayed 24 h+ last week", 0.0, 1.0,
            float(rows["recent_congested_share_7d_same_type"].median()))
        base["recent_avg_wait_7d_same_type"] = st.slider(
            "Avg same-type anchor wait last week (h)", 0.0, 150.0,
            float(rows["recent_avg_wait_7d_same_type"].median()))
    with c3:
        base["wind_gusts_10m"] = st.slider("Wind gusts (km/h)", 0, 120, 30)
        base["max_gust_next_24h"] = st.slider("Max gust next 24 h (km/h)", 0, 120, 35)
        base["wave_height"] = st.slider("Wave height (m)", 0.0, 6.0, 1.0)
        base["max_wave_next_24h"] = st.slider("Max wave next 24 h (m)", 0.0, 6.0, 1.2)
        base["hour_of_day"] = st.slider("Arrival hour (UTC)", 0, 23, 12)
    st.divider()
    show_prediction(base)

with tab_eval:
    ours = metrics["models"]["LightGBM ensemble (ours)"]
    st.write(f"Time-based split: trained on Jan-Feb 2023 ({metrics['n_train']} calls), "
             f"tested on March 2023 ({metrics['n_test']} calls) that the model never saw.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC", f"{ours['roc_auc']:.3f}", "best of 4 models", delta_color="off")
    k2.metric("Congested ships caught (recall)", f"{ours['recall']:.0%}")
    k3.metric("Flags that were right (precision)", f"{ours['precision']:.0%}")
    k4.metric("Accuracy when confidence is High", f"{metrics['confidence_bands']['High']['accuracy']:.0%}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Our model vs baselines")
        st.caption("Higher and further left is better. The dashed line is random guessing.")
        st.altair_chart(charts.roc_chart(test, metrics), width="stretch")
    with c2:
        st.subheader("What drives the predictions")
        st.caption("Average SHAP influence. Queue of same-type ships and their recent delays dominate.")
        st.altair_chart(charts.importance_chart(metrics, FEATURE_TEXT), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Outcomes at the alert threshold ({model.threshold:.0%})")
        st.altair_chart(charts.confusion_chart(test, model.threshold), width="stretch")
    with c2:
        st.subheader("Is the confidence score useful?")
        st.caption("Yes: when the model says High confidence it is right far more often.")
        st.altair_chart(charts.confidence_chart(metrics), width="stretch")

    st.subheader("March, day by day: predicted risk vs what actually happened")
    st.altair_chart(charts.timeline_chart(test))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Calibration")
        cal = metrics["calibration"]
        st.caption(f"Platt recalibration on held-out weeks: Brier {cal['brier_raw']:.3f} -> {cal['brier_calibrated']:.3f}. "
                   "Over-confidence above 50% remains (a known limitation).")
        st.altair_chart(charts.calibration_chart(test), width="stretch")
    with c2:
        st.subheader("All metrics")
        st.dataframe(pd.DataFrame(metrics["models"]).T.round(3))
        st.write("Per port (ROC-AUC within each port):")
        st.dataframe(pd.DataFrame(metrics["per_port"]).T)

    c1, c2 = st.columns(2)
    with c1:
        conf = metrics["conformal"]
        st.subheader("Conformal prediction sets (90% target)")
        st.caption(f"{conf['share_single']:.0%} of ships get a single outcome, right {conf['single_accuracy']:.0%} of "
                   f"the time. Overall coverage {conf['coverage']:.1%}: below 90% because congestion dropped from "
                   "35% (calibration weeks) to 27% (March).")
        st.altair_chart(charts.conformal_chart(test), width="stretch")
    with c2:
        st.subheader("Weather x traffic interaction")
        st.caption("Average predicted risk for 300 March arrivals with the queue and upcoming gusts set to each value. "
                   "The queue dominates; wind adds more risk when the queue is long.")
        st.altair_chart(charts.interaction_chart(interaction), width="stretch")

with tab_ai:
    from src import analyst

    st.write("Ask about March 2023 arrivals in plain English. The AI (OpenAI function calling) answers by calling "
             "the congestion model as a tool, so every number comes from the model, not from the AI.")
    key = os.getenv("OPENAI_API_KEY", "")
    key = key if key.startswith("sk-") else None   # ignore the .env placeholder
    if not key:
        st.warning("No OpenAI API key found. Put OPENAI_API_KEY=sk-... in a .env file in the project folder "
                   "and restart the app.")
    examples = ["Which Houston tankers arriving on 9 March 2023 are most at risk, and why?",
                "Explain visit 5389 and what the planner should do.",
                "For visit 5389, what if the same-type queue per berth dropped to 0.2?",
                "How busy was Savannah on 15 March 2023 at 12:00, and how reliable is the model there?"]
    st.caption("Try: " + " | ".join(f"*{e}*" for e in examples))
    if "chat" not in st.session_state:
        st.session_state.chat, st.session_state.history = [], []
    for role, text, trace in st.session_state.chat:
        with st.chat_message(role):
            if trace:
                with st.expander(f"Model tools used ({len(trace)})"):
                    st.json(trace)
            st.markdown(text)
    question = st.chat_input("Ask the port analyst...")
    if question:
        if not key:
            st.warning("The analyst needs OPENAI_API_KEY in the .env file.")
        else:
            from openai import OpenAI
            st.session_state.chat.append(("user", question, None))
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"), st.spinner("Calling the model..."):
                try:
                    answer, trace, st.session_state.history = analyst.ask(
                        question, st.session_state.history, client=OpenAI(api_key=key))
                except Exception as e:
                    answer, trace = f"The AI request failed: {e}", []
                if trace:
                    with st.expander(f"Model tools used ({len(trace)})"):
                        st.json(trace)
                st.markdown(answer)
            st.session_state.chat.append(("assistant", answer, trace))
