"""Build the technical report PDF from the saved metrics and predictions.

Usage:
    python -m src.make_report
"""
import json

import joblib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.build_dataset import CONGESTION_THRESHOLD_H
from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.model import explain_row, recommend, summarise

AUTHOR = "Pavan Reddy K"
EXAMPLE_VISIT_ID = 5389   # a Houston tanker from the March test set
REPO_URL = "https://github.com/pavanreddy950/port-congestion-ai"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=15, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, alignment=1, textColor=colors.grey)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=7, spaceAfter=3)
BODY = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=11.8, spaceAfter=3)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=10, bulletIndent=2, spaceAfter=1.5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=9.5, spaceAfter=0)


def p(text):
    return Paragraph(text, BODY)


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")))
    t.setStyle(TableStyle(style))
    return t


def main():
    m = json.load(open(REPORTS_DIR / "metrics.json"))
    calls = pd.read_csv(PROCESSED_DIR / "port_calls.csv")
    test = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    model = joblib.load(MODELS_DIR / "congestion_model.joblib")

    port_stats = calls.groupby("port").agg(n=("visit_id", "size"), cong=("congested", "mean"),
                                           med=("anchor_wait_h", "median"),
                                           p90=("anchor_wait_h", lambda s: s.quantile(0.9)))
    ex = test[test["visit_id"] == EXAMPLE_VISIT_ID]
    ex_pred = model.predict(ex).iloc[0]
    ex_drivers = explain_row(ex.iloc[0], model.shap_values(ex).iloc[0], top_k=4)
    ours = m["models"]["LightGBM ensemble (ours)"]
    cal, conf = m["calibration"], m["conformal"]
    grid = pd.read_csv(REPORTS_DIR / "interaction_queue_weather.csv")
    gq = sorted(grid["queue_per_berth_same_type"].unique())
    gg = sorted(grid["max_gust_next_24h"].unique())
    risk = grid.set_index(["queue_per_berth_same_type", "max_gust_next_24h"])["risk"]

    story = [
        Paragraph("Predicting Port Congestion from Real AIS Vessel Data", H1),
        Paragraph(f"Aiqon AI Internship take-home &nbsp;|&nbsp; {AUTHOR} &nbsp;|&nbsp; "
                  f'<link href="{REPO_URL}" color="blue">{REPO_URL.removeprefix("https://")}</link>', SUB),
        Spacer(1, 6),

        Paragraph("1. Problem and definition of congestion", H2),
        p("The goal is to estimate, at the moment a ship arrives at a port area, how likely it is to be delayed by "
          "congestion, how confident that estimate is, and why. I define congestion from the ship's own behaviour: "
          f"<b>a port call is congested if the vessel waits {CONGESTION_THRESHOLD_H} hours or more at anchor before "
          "getting a berth</b>. Anchorage waiting is the most direct, observable symptom of berth congestion and "
          "maps onto a planning decision (adjust speed, warn the customer, re-book trucks). 24 h was chosen because "
          "it is an operationally meaningful delay (a missed day of cargo handling); it sits at roughly the 70th "
          "percentile of observed waits, giving a 31% positive rate."),

        Paragraph("2. Data", H2),
        p("<b>Source:</b> real AIS vessel positions from MarineCadastre.gov (NOAA/BOEM), 90 daily files for "
          "1 Jan - 31 Mar 2023 (~320 MB each, ~29 GB in total). Each file is streamed, filtered to cargo and tanker "
          "vessels (AIS types 70-89) inside four port areas, and the raw file is discarded. Result: "
          f"11.3 M positions from 2,985 vessels. Hourly weather (wind, gusts, rain) and offshore wave height for "
          "each port come from the free Open-Meteo historical archive."),
        p("<b>From pings to port calls:</b> pings of one vessel in one port are grouped into a visit (a gap of more than "
          "12 h starts a new visit). Anchored = nav status 'at anchor' and speed &lt; 1 kn; moored = status "
          "'moored' and speed &lt; 1 kn. Anchor wait = first moored ping minus first anchored ping. A location check "
          "confirmed that anchoring happens at the offshore anchorages (e.g. Houston around 29.25N off Galveston) and "
          "mooring happens at the terminals (around 29.64N up the ship channel)."),
        p("<b>Cleaning (5,901 visits to 3,644 port calls):</b> dropped visits already moored at the first ping (285, "
          "arrival not observed), visits that started before data began (173) or in the last 5 days (319) "
          "(censored waits), and visits that never berthed (1,480: bunkering, lightering, transit or anchorage-only "
          "calls)."),
        table([["Port", "Calls", "Congested", "Median wait", "90th pct wait"]] +
              [[k, int(r.n), f"{r.cong:.0%}", f"{r.med:.1f} h", f"{r.p90:.1f} h"] for k, r in port_stats.iterrows()],
              [3.6 * cm, 2 * cm, 2.3 * cm, 2.6 * cm, 2.8 * cm]),
        Spacer(1, 3),
        p("The ports behave very differently: Houston (mostly tankers) queues for days, LA/Long Beach and Savannah "
          "have occasional queues, and NY/NJ almost never queues. Variability across ports is a real feature of the "
          "problem, so the port is a model input, and features are normalised by each port's own capacity."),

        Paragraph("3. Features (all known at arrival time)", H2),
        p("Every feature uses only information available when the ship arrives, to avoid look-ahead leakage. "
          "Port state is taken from the last complete hour before arrival; recent history only uses ships that had "
          "already berthed."),
        table([
            ["Group", "Features", "Why"],
            ["Queue and capacity", "ships at anchor, ships at berth, occupancy vs capacity, queue per berth, "
             "24 h queue change; the same counts for <b>the same vessel type</b> only",
             "Tankers and container ships use different berths, so a tanker competes only with the tanker queue. "
             "Capacity = 95th percentile of simultaneously moored ships (train period)."],
            ["Recent delay history", "avg wait and share delayed 24 h+ over the last 7 days (all and same type), "
             "arrivals in last 24 h / 7 days", "Congestion persists; this captures the historical delay pattern and "
             "arrival density."],
            ["Vessel", "length, draft, cargo/tanker", "Deep-draft ships can be tide or berth restricted."],
            ["Time", "hour of day, day of week, weekend", "Pilot and terminal shift patterns."],
            ["Weather", "wind, gusts, rain, wave height, max gust/wave in next 24 h",
             "Pilotage and berthing get suspended in high wind or swell."],
        ], [3 * cm, 7 * cm, 7.3 * cm]),

        Paragraph("4. Model, confidence score and explanations", H2),
        *bullets([
            "<b>Prediction:</b> a bagged ensemble of 25 LightGBM classifiers, each trained on a bootstrap sample. "
            "Probability = mean of the members, recalibrated with <b>Platt scaling</b> fitted on held-out validation "
            "predictions (strictly increasing, so ranking and alerts are unchanged). Gradient-boosted trees handle "
            "missing values natively and pick up interactions (for example weather times queue) without hand-crafting.",
            "<b>Confidence score:</b> (a) agreement between ensemble members (1 - spread/0.25), multiplied by "
            "(b) a data-support factor that drops when inputs fall outside the 1st-99th percentile seen in training. "
            "Bands: High &ge; 0.75, Medium &ge; 0.5, otherwise Low. In addition, <b>split conformal prediction</b> "
            "(calibrated on the validation weeks) returns a prediction set targeting 90% coverage: "
            "{delay}, {no delay}, or 'either' when the data cannot rule out one outcome.",
            "<b>Explanations:</b> SHAP values averaged over members give each feature's push up or down on the risk. "
            "The top drivers are combined into one summary sentence and mapped to an action (see section 6). An "
            "<b>AI Port Analyst</b> (OpenAI function calling) lets a planner ask questions in plain English; the LLM "
            "calls the model as a tool, so every number it quotes comes from the model, not from the LLM.",
            f"<b>Alert threshold:</b> tuned on a validation window (15-28 Feb, never the test set) by maximising "
            f"Youden's J, which weighs missed delays and false alarms equally. Result: {ours['threshold']:.2f}.",
        ]),

        Paragraph("5. Evaluation", H2),
        p(f"<b>Time-based split</b>, which mirrors real use and avoids leaking the future: train on Jan-Feb "
          f"({m['n_train']} calls, 32% congested), test on March ({m['n_test']} calls, 27% congested). Each model "
          "is compared against simple baselines, with each one's threshold tuned on the same validation window."),
        table([["Model", "ROC-AUC", "PR-AUC", "Brier", "Precision", "Recall", "F1"]] +
              [[k, f"{v['roc_auc']:.3f}", f"{v['pr_auc']:.3f}", f"{v['brier']:.3f}", f"{v['precision']:.2f}",
                f"{v['recall']:.2f}", f"{v['f1']:.2f}"] for k, v in m["models"].items()],
              [6.2 * cm, 2.2 * cm, 1.7 * cm, 1.6 * cm, 1.8 * cm, 1.6 * cm, 1.3 * cm]),
        Spacer(1, 3),
        p(f"The ensemble has the best ranking (ROC-AUC {ours['roc_auc']:.3f}, PR-AUC {ours['pr_auc']:.3f}) and "
          f"catches {ours['recall']:.0%} of congested arrivals at {ours['precision']:.0%} precision. "
          "<b>The margin over the port-average baseline is modest</b>: much of the signal is simply 'which port and "
          "which vessel type'. Within a single port, ranking is good for LA/Long Beach "
          f"(AUC {m['per_port']['LA_LongBeach']['roc_auc']:.2f}) but weak for Houston "
          f"({m['per_port']['Houston']['roc_auc']:.2f}) and Savannah ({m['per_port']['Savannah']['roc_auc']:.2f}). "
          "<b>The confidence score is meaningful:</b> High-confidence predictions are "
          f"{m['confidence_bands']['High']['accuracy']:.0%} accurate (n={m['confidence_bands']['High']['n']}), "
          f"against {m['confidence_bands']['Medium']['accuracy']:.0%} for Medium and "
          f"{m['confidence_bands']['Low']['accuracy']:.0%} for Low."),
        Image(str(REPORTS_DIR / "importance_calibration.png"), width=17 * cm, height=6.5 * cm),
        p("Left: the dominant drivers are the same-type queue per berth and recent same-type delays, which is "
          f"operationally sensible. Right: Platt recalibration lowers the Brier score from {cal['brier_raw']:.3f} to "
          f"{cal['brier_calibrated']:.3f} (the best of all models), but the over-confidence above 0.5 remains "
          f"(expected calibration error {cal['ece_raw']:.3f} to {cal['ece_calibrated']:.3f})."),
        p(f"<b>Conformal prediction sets:</b> {conf['share_single']:.0%} of March arrivals get a single-outcome set, and "
          f"those are right {conf['single_accuracy']:.0%} of the time; the rest are marked 'either'. Overall coverage is "
          f"{conf['coverage']:.1%} against the 90% target. The shortfall is expected because the congested rate "
          "dropped from 35% in the calibration weeks to 27% in March, which breaks the exchangeability assumption."),
        p("<b>Weather times traffic:</b> a two-way partial dependence on 300 March arrivals shows that the queue "
          "dominates and weather adds a smaller risk that grows with the queue:"),
        table([["Same-type queue per berth"] + [f"gusts {g:.0f} km/h" for g in gg]] +
              [[f"{q:.2f}"] + [f"{risk[(q, g)]:.0%}" for g in gg] for q in (gq[0], gq[len(gq) // 2], gq[-1])],
              [4.2 * cm] + [2.2 * cm] * len(gg)),

        Paragraph("6. From prediction to decision: an example", H2),
        p(f"<b>Houston tanker arriving {pd.Timestamp(ex.iloc[0]['arrival_time']):%d %b %Y %H:%M} UTC</b> (test set). "
          f"Prediction: <b>{ex_pred.probability:.0%}</b> risk (alert threshold {model.threshold:.0%}), confidence "
          f"<b>{ex_pred.confidence:.0%} ({ex_pred.confidence_band})</b>. Actual outcome: "
          f"{ex.iloc[0]['anchor_wait_h']:.0f} h at anchor. Conformal prediction set: '{ex_pred.prediction_set}'."),
        p(f"<b>Summary:</b> {summarise(ex_pred.probability, ex_drivers, model.threshold)} Top reasons:"),
        *bullets([d["text"] for d in ex_drivers]),
        p(f"<b>Suggested action:</b> {recommend(ex_pred.probability, ex_pred.confidence_band, ex_drivers, model.threshold)}"),
        p("Operational uses: a carrier can slow-steam into a known queue (same berth time, less fuel and emissions); "
          "a shipper can warn customers and move truck/rail appointments; a port can see which ship types are "
          "backing up. The Streamlit demo (<i>app.py</i>) replays real March arrivals, supports what-if scenarios and "
          "includes the AI Port Analyst chat (for example: 'Which Houston tankers arriving on 9 March are most at risk, "
          "and why?')."),

        Paragraph("7. Assumptions and limitations", H2),
        *bullets([
            "<b>Label noise:</b> AIS nav status is entered manually and is sometimes stale. More importantly, not all "
            "anchorage time is congestion: tankers often wait for cargo readiness, orders or lightering. This "
            "commercial waiting is invisible in AIS and is the most likely reason Houston is hard to predict.",
            "<b>Approximate geography:</b> ports are rectangles, not berth or terminal polygons, and the destination "
            "terminal is unknown. Capacity is a proxy (95th percentile of moored ships), not the real berth count.",
            "<b>Weather:</b> observed weather is used as a stand-in for the forecast a planner would have, which is "
            "slightly optimistic.",
            "<b>Coverage:</b> 3 winter months and 4 US ports, so no seasonality (holiday peaks, hurricane season) "
            "is learned. The congested rate drifts from 32% to 27% between train and test.",
            "<b>Selection:</b> excluding censored and never-berthed visits removes some of the longest waits and all "
            "anchorage-only calls.",
            "<b>Calibration and drift:</b> recalibration is fitted on only two weeks (593 calls), and over-confidence "
            "above 0.5 remains; conformal coverage falls below target when the congestion rate shifts.",
            "<b>LLM analyst:</b> it needs an OpenAI API key and sends tool results (model outputs, no personal data) "
            "to OpenAI. It only phrases and combines tool results; the predictions themselves come from the model.",
        ]),
        Paragraph("8. Next steps", H2),
        p("Use berth and terminal polygons and AIS destination to model terminal-level queues; add a full year of "
          "data for seasonality; predict expected wait hours (regression or survival model) as well as the 24 h "
          "flag; retrain and recalibrate on a rolling window with drift monitoring (adaptive conformal prediction keeps "
          "coverage under drift); and serve the model behind an API for live AIS feeds."),
    ]

    out = REPORTS_DIR / "Port_Congestion_Report.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                      title="Predicting Port Congestion from Real AIS Data", author=AUTHOR).build(story)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
