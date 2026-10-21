# Predicting Port Congestion from Real AIS Vessel Data

Aiqon AI Internship take-home. This system estimates, for a ship arriving at a port, **how likely it is to wait 24 h or more at anchor** before getting a berth. It gives a **confidence score** (plus a conformal prediction set), **plain-English reasons**, and suggests an **operational action**. An **AI Port Analyst** lets planners ask questions in plain English.

- **Data:** real AIS ship positions from [MarineCadastre.gov](https://marinecadastre.gov/ais/), Jan–Mar 2023. That is 11.3 M positions from 2,985 cargo and tanker vessels, turned into **3,644 port calls** at LA/Long Beach, Houston, NY/NJ and Savannah, plus hourly weather and waves from [Open-Meteo](https://open-meteo.com/).
- **Model:** a bagged ensemble of 25 LightGBM classifiers, recalibrated with Platt scaling, with SHAP explanations and split conformal prediction sets.
- **AI Port Analyst:** an LLM agent (OpenAI function calling) that answers questions by calling the model as a tool, so every number it quotes comes from the model.
- **Evaluation:** trained on Jan–Feb and tested on March (time-based split), compared against 3 baselines.
- **Report:** the 2–3 page technical report is submitted separately as a PDF. Run `python -m src.make_report` to rebuild it into `reports/`.

## Results (March 2023 test set, 1,114 port calls)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline: port historical rate | 0.729 | 0.421 | 0.35 | 0.98 | 0.51 |
| Baseline: rule on same-type queue per berth | 0.663 | 0.381 | 0.45 | 0.60 | 0.52 |
| Logistic regression | 0.750 | 0.455 | 0.42 | 0.81 | 0.56 |
| **LightGBM ensemble (ours)** | **0.752** | **0.462** | 0.43 | 0.72 | 0.54 |

Our model also has the best Brier score (0.177 after recalibration; port baseline 0.178).

- High-confidence predictions are **89% accurate**, against about 51–55% for medium and low confidence, so the confidence score carries real information.
- **Conformal prediction:** 68% of arrivals get a single-outcome set (delay or no delay), which is right 79% of the time. Coverage is 85.5% against the 90% target, because congestion fell from 35% to 27% between the calibration weeks and March.
- The top drivers are the **queue of same-type ships per berth** and **recent delays of same-type ships**. Weather adds risk mainly when the queue is already long (a weather × traffic interaction).
- The gain over the per-port baseline is modest, and Houston is hard to predict within the port. The report explains why: much tanker anchorage time is commercial waiting, which AIS cannot see.

## Definition of congestion

A port call is **congested** if the ship waits **at least 24 hours at anchor** (AIS status "at anchor", speed under 1 kn) before its first "moored" position. Every feature uses only information available when the ship arrives.

## Project structure

```
src/
  config.py         ports (bounding boxes), paths, AIS filters
  download_ais.py   download daily AIS files, keep cargo/tanker pings in port areas
  fetch_weather.py  hourly wind, gusts, rain and wave height per port (Open-Meteo)
  build_dataset.py  pings -> visits -> port calls, anchor-wait labels, features
  model.py          ensemble model, calibration, confidence score, conformal sets, SHAP explanations, actions
  train.py          time-split evaluation vs baselines, saves model and metrics
  analyst.py        AI Port Analyst: LLM agent with the model as tools (OpenAI function calling)
  check_data.py     quick summary of the final dataset
  predict_one.py    predict and explain one arrival from the command line
  make_report.py    builds the PDF report from the saved results
  charts.py         interactive charts and the ship map used by the demo
app.py              Streamlit demo
data/processed/     port_calls.csv (final dataset), weather, hourly port state
reports/            metrics.json, test predictions, figures
```

## How to run

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Quick start.** The processed dataset and the trained model are included in the repo:

```bash
streamlit run app.py         # opens the demo in your browser
python -m src.train          # optional: retrains and re-evaluates (about 2 min), rewrites models/ and reports/
python -m src.make_report    # optional: rebuilds the PDF report
```

**Quick checks from the terminal:**

```bash
python -m src.check_data          # summary of the final dataset per port
python -m src.predict_one         # predict a random March arrival: risk, confidence, reasons, action
python -m src.predict_one 5389    # predict a specific arrival by visit_id
```

**AI Port Analyst (optional; needs an OpenAI API key):**

Copy `.env.example` to `.env` and put your key in it (`.env` is git-ignored, so the key never reaches GitHub):

```bash
OPENAI_API_KEY=sk-...
```

```bash
python -m src.analyst "Which Houston tankers arriving on 9 March 2023 are most at risk, and why?"
```

The app's 🤖 tab reads the same key. The default model is `gpt-4.1-mini`; add `OPENAI_MODEL=...` to `.env` to change it.

**Full pipeline from raw AIS data.** This downloads about 29 GB in total, but only a few MB is kept. It takes 2–5 h depending on your connection.

```bash
python -m src.download_ais --start 2023-01-01 --end 2023-03-31 --workers 2
python -m src.fetch_weather --start 2023-01-01 --end 2023-03-31
python -m src.build_dataset
python -m src.train
```

## Demo app

- **Overview: the data:** a map of real AIS positions (ships waiting offshore vs moored at terminals), anchorage queues over time with the train/test split marked, wait-time distributions per port, and congestion by vessel type.
- **Replay a real arrival:** choose any March 2023 arrival (unseen during training) to see the predicted risk, confidence, top reasons, suggested action, and what actually happened.
- **What-if scenario:** change the queue, berth occupancy, recent delays, weather and vessel size, and watch the risk respond.
- **Model evaluation:** ROC curves comparing our model with the 3 baselines, a confusion matrix at the alert threshold, accuracy by confidence band, a day-by-day March timeline of predicted vs actual congestion, feature importance, calibration before and after recalibration, conformal prediction sets, and a weather × queue interaction heat map.
- **🤖 AI Port Analyst:** a chat where you ask, for example, *"Which Houston tankers arriving on 9 March are most at risk, and why?"*. The LLM calls the model's tools (find arrivals, explain an arrival, what-if, port status, model performance), and the app shows which tools it used.

## Key assumptions and limitations

Full details are in the report.

- AIS navigation status is entered by the crew and is sometimes stale. Some anchor time is commercial waiting, not congestion.
- Ports are approximated by rectangles, and berth capacity is estimated from the data (95th percentile of moored ships).
- Observed weather stands in for the forecast a planner would have.
- The data covers 3 winter months at 4 US ports, so no seasonality is learned.
- Probabilities are well ordered. After recalibration some over-confidence above 50% remains, and conformal coverage drops when the congestion rate drifts.
- The AI Port Analyst sends model outputs (no personal data) to OpenAI. It only phrases tool results; it does not make predictions itself.
