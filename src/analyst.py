"""AI Port Analyst: an LLM agent that answers planner questions by calling the
congestion model as a tool (OpenAI function calling).

The LLM never invents numbers: every risk, confidence, reason and outcome it
quotes comes from a tool below, which runs the trained model on real data.

Needs an OpenAI API key in the OPENAI_API_KEY environment variable.
Optional: OPENAI_MODEL (default gpt-5-mini).

Run:  python -m src.analyst "Which Houston tankers arriving on 9 March are at risk, and why?"
"""
import json
import os
import sys
from functools import lru_cache

import joblib
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.model import FEATURE_TEXT, explain_row, recommend, summarise

DEFAULT_MODEL = "gpt-5-mini"
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are a port operations analyst helping shipping planners.
You have tools backed by a machine-learning model trained on real AIS vessel data
(Jan-Feb 2023) for four US ports: Houston, LA_LongBeach, NY_NJ, Savannah.
Arrivals you can look up are the March 2023 test period (1-31 March 2023), which the
model never saw during training, so you can also report what actually happened.

"Congested" means the ship waits 24 hours or more at anchor before getting a berth.
Rules:
- Always use the tools for numbers. Never guess a risk, count or outcome.
- Quote the risk %, confidence band and the main reasons, then a concrete action
  (slow-steaming, warning the consignee, moving truck/rail slots, adding buffer time).
- Mention the prediction set when it is 'either' (the model cannot rule out either outcome).
- Be concise: short paragraphs or bullet points, plain English for non-experts.
- If asked about dates or ports outside the data, say so plainly."""


@lru_cache(maxsize=1)
def _data():
    model = joblib.load(MODELS_DIR / "congestion_model.joblib")
    test = pd.read_csv(REPORTS_DIR / "test_predictions.csv", parse_dates=["arrival_time", "berth_time"])
    state = pd.read_parquet(PROCESSED_DIR / "port_hourly_state.parquet")
    with open(REPORTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return model, test, state, metrics


def _predict(row: pd.DataFrame) -> dict:
    model, *_ = _data()
    pred = model.predict(row).iloc[0]
    drivers = explain_row(row.iloc[0], model.shap_values(row).iloc[0])
    return {
        "risk_of_24h_wait": round(float(pred.probability), 3),
        "flagged": bool(pred.probability >= model.threshold),
        "alert_threshold": round(model.threshold, 3),
        "confidence": round(float(pred.confidence), 2),
        "confidence_band": pred.confidence_band,
        "prediction_set_90pct": pred.prediction_set,
        "summary": summarise(pred.probability, drivers, model.threshold),
        "top_reasons": [{"reason": d["text"], "impact_log_odds": round(d["impact"], 2)} for d in drivers],
        "suggested_action": recommend(pred.probability, pred.confidence_band, drivers, model.threshold),
    }


# ---------------------------------------------------------------- tools

def find_arrivals(port: str, date: str | None = None, vessel_group: str | None = None,
                  min_risk: float = 0.0, limit: int = 10) -> dict:
    _, test, *_ = _data()
    d = test[test["port"] == port]
    if date:
        d = d[d["arrival_time"].dt.date == pd.Timestamp(date).date()]
    if vessel_group:
        d = d[d["vessel_group"] == vessel_group]
    d = d[d["probability"] >= min_risk].sort_values("probability", ascending=False)
    rows = [{"visit_id": int(r.visit_id), "arrival_time_utc": f"{r.arrival_time:%Y-%m-%d %H:%M}",
             "vessel_group": r.vessel_group, "MMSI": int(r.MMSI), "risk": round(float(r.probability), 3),
             "confidence_band": r.confidence_band, "actual_wait_h": round(float(r.anchor_wait_h), 1)}
            for r in d.head(limit).itertuples()]
    return {"port": port, "date": date, "matching_arrivals": len(d), "arrivals": rows}


def explain_arrival(visit_id: int) -> dict:
    _, test, *_ = _data()
    row = test[test["visit_id"] == int(visit_id)]
    if row.empty:
        return {"error": f"visit_id {visit_id} not found in the March 2023 test set"}
    r = row.iloc[0]
    return {"visit_id": int(visit_id), "port": r.port, "vessel_group": r.vessel_group, "MMSI": int(r.MMSI),
            "arrival_time_utc": f"{r.arrival_time:%Y-%m-%d %H:%M}", **_predict(row),
            "what_actually_happened": {"anchor_wait_h": round(float(r.anchor_wait_h), 1),
                                       "congested": bool(r.congested)}}


WHAT_IF_FIELDS = ["queue_same_type", "queue_per_berth_same_type", "recent_congested_share_7d_same_type",
                  "recent_avg_wait_7d_same_type", "wind_gusts_10m", "max_gust_next_24h", "wave_height",
                  "max_wave_next_24h", "draft", "length", "hour_of_day"]


def what_if(visit_id: int, changes: dict) -> dict:
    _, test, *_ = _data()
    row = test[test["visit_id"] == int(visit_id)].copy()
    if row.empty:
        return {"error": f"visit_id {visit_id} not found"}
    unknown = set(changes) - set(WHAT_IF_FIELDS)
    if unknown:
        return {"error": f"cannot change {sorted(unknown)}; allowed fields: {WHAT_IF_FIELDS}"}
    before = _predict(row)
    for k, v in changes.items():
        row[k] = float(v)
    after = _predict(row)
    return {"visit_id": int(visit_id), "changes": changes,
            "risk_before": before["risk_of_24h_wait"], "risk_after": after["risk_of_24h_wait"], "after": after}


def port_status(port: str, datetime_utc: str) -> dict:
    _, test, state, _ = _data()
    s = state[(state["port"] == port) & (state["hour"] <= pd.Timestamp(datetime_utc))].tail(1)
    if s.empty:
        return {"error": f"no data for {port} at {datetime_utc} (data covers Jan-Mar 2023)"}
    s = s.iloc[0]
    recent = test[(test["port"] == port) & (test["berth_time"] <= pd.Timestamp(datetime_utc))
                  & (test["berth_time"] > pd.Timestamp(datetime_utc) - pd.Timedelta(days=7))]
    return {"port": port, "hour_utc": f"{s.hour:%Y-%m-%d %H:00}",
            "ships_waiting_at_anchor": {"total": int(s.queue_anchored), "cargo": int(s.queue_cargo),
                                        "tanker": int(s.queue_tanker)},
            "ships_at_berth": {"total": int(s.berths_occupied), "cargo": int(s.berths_cargo),
                               "tanker": int(s.berths_tanker)},
            "queue_change_last_24h": None if pd.isna(s.queue_change_24h) else int(s.queue_change_24h),
            "march_ships_berthed_last_7_days": len(recent),
            "share_of_them_delayed_24h_plus": round(float(recent["congested"].mean()), 3) if len(recent) else None}


def model_performance() -> dict:
    *_, metrics = _data()
    return {k: metrics[k] for k in ["models", "confidence_bands", "per_port", "calibration", "conformal",
                                    "n_train", "n_test"] if k in metrics} | {
        "top_drivers": [FEATURE_TEXT[f][0] for f in list(metrics["shap_importance"])[:5]]}


PORT_ENUM = {"type": "string", "enum": ["Houston", "LA_LongBeach", "NY_NJ", "Savannah"]}
TOOLS = [
    {"type": "function", "function": {
        "name": "find_arrivals",
        "description": "List March 2023 arrivals at a port, highest predicted risk first.",
        "parameters": {"type": "object", "properties": {
            "port": PORT_ENUM,
            "date": {"type": "string", "description": "YYYY-MM-DD, between 2023-03-01 and 2023-03-31"},
            "vessel_group": {"type": "string", "enum": ["cargo", "tanker"]},
            "min_risk": {"type": "number", "description": "only arrivals with risk >= this (0-1)"},
            "limit": {"type": "integer", "description": "max rows, default 10"}},
            "required": ["port"]}}},
    {"type": "function", "function": {
        "name": "explain_arrival",
        "description": "Full prediction for one arrival: risk, confidence, prediction set, reasons, action and actual outcome.",
        "parameters": {"type": "object", "properties": {"visit_id": {"type": "integer"}}, "required": ["visit_id"]}}},
    {"type": "function", "function": {
        "name": "what_if",
        "description": "Re-run the model for one arrival after changing some conditions (scenario analysis).",
        "parameters": {"type": "object", "properties": {
            "visit_id": {"type": "integer"},
            "changes": {"type": "object", "description": f"field -> new value. Allowed fields: {WHAT_IF_FIELDS}"}},
            "required": ["visit_id", "changes"]}}},
    {"type": "function", "function": {
        "name": "port_status",
        "description": "Anchorage queue, berth occupancy and recent delays at a port at a given time (Jan-Mar 2023).",
        "parameters": {"type": "object", "properties": {
            "port": PORT_ENUM, "datetime_utc": {"type": "string", "description": "YYYY-MM-DD HH:MM"}},
            "required": ["port", "datetime_utc"]}}},
    {"type": "function", "function": {
        "name": "model_performance",
        "description": "How accurate the model is on the March test set, versus baselines, per port and by confidence.",
        "parameters": {"type": "object", "properties": {}}}},
]
TOOL_FUNCTIONS = {"find_arrivals": find_arrivals, "explain_arrival": explain_arrival, "what_if": what_if,
                  "port_status": port_status, "model_performance": model_performance}


def run_tool(name: str, arguments: str) -> str:
    try:
        result = TOOL_FUNCTIONS[name](**json.loads(arguments or "{}"))
    except Exception as e:  # report tool errors back to the LLM instead of crashing
        result = {"error": f"{type(e).__name__}: {e}"}
    return json.dumps(result, default=str)


def ask(question: str, history: list[dict] | None = None, client=None, model: str | None = None):
    """Answer a question. Returns (answer, tool_calls_made, updated_history)."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []),
                {"role": "user", "content": question}]
    trace = []
    for _ in range(MAX_TOOL_ROUNDS):
        msg = client.chat.completions.create(model=model, messages=messages, tools=TOOLS).choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            new_history = [*(history or []), {"role": "user", "content": question},
                           {"role": "assistant", "content": answer}]
            return answer, trace, new_history
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in msg.tool_calls]})
        for c in msg.tool_calls:
            trace.append({"tool": c.function.name, "arguments": json.loads(c.function.arguments or "{}")})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": run_tool(c.function.name, c.function.arguments)})
    return "I could not finish within the tool-call limit; please narrow the question.", trace, history or []


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first, e.g. in PowerShell:  $env:OPENAI_API_KEY = \"sk-...\"")
    q = " ".join(sys.argv[1:]) or "Which Houston tankers arriving on 9 March 2023 are most at risk, and why?"
    answer, trace, _ = ask(q)
    for t in trace:
        print(f"[tool] {t['tool']}({t['arguments']})")
    print("\n" + answer)
